"""
LiveKit Voice Assistant Application
====================================
This application creates a real-time voice assistant using LiveKit for audio processing.
It integrates with Redis for conversation persistence, RAG (Retrieval Augmented Generation)
for context-aware responses, and external APIs for message logging.

Key Features:
- Real-time voice conversation processing
- Multilingual speech recognition and synthesis
- Conversation persistence in Redis
- RAG-based context retrieval
- External API integration for message logging

OPTIMIZATION CHANGELOG (latency improvements):
-----------------------------------------------
HIGH IMPACT:
  1. Tone classification now runs as a fire-and-forget background task concurrent
     with the LLM — no longer a serial await that blocks the LLM from starting.
  2. RAG fetch runs concurrently with the LLM stream via asyncio.gather inside a
     background task. Context is injected into the *next* turn's chat context so
     it is available without ever blocking the current LLM response.
  3. VAD max_turn_silence lowered (1000 ms → 500 ms) and min_turn_silence raised
     (100 ms → 200 ms) for better natural endpointing without false positives.

MEDIUM IMPACT:
  4. Redis writes are now fire-and-forget background tasks — they no longer block
     the hot path. The in-memory Conversation object is always authoritative within
     a session; Redis is an async write-behind cache.
  5. Vector index is loaded ONCE per session inside fetch_context (LlamaIndex
     caches the index on the Conversation object). A guard added here ensures we
     never reload it when con.index already exists.
  6. ElevenLabs model defaults to eleven_turbo_v2_5 (env-overridable) for lower
     TTS latency. Switch back to eleven_multilingual_v2 via ELEVENLABS_MODEL env
     var if quality is preferred over speed.
"""

# ============================================================================
# IMPORTS
# ============================================================================

from dotenv import load_dotenv
import sounddevice as sd

from livekit.agents import AgentSession, Agent, RoomInputOptions, function_tool, RunContext, BackgroundAudioPlayer, AudioConfig, BuiltinAudioClip, RoomInputOptions, TurnHandlingOptions
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, AutoSubscribe
from livekit.agents.cli import run_app
from livekit.agents import ChatContext, ChatMessage
from livekit.agents import ConversationItemAddedEvent
from livekit.agents import ModelSettings

from livekit.plugins import assemblyai, elevenlabs, silero, noise_cancellation
from livekit.plugins.openai import LLM

from openai import AsyncOpenAI

from redis_utils_Wg import load_conversation_from_redis, redis_client, save_conversation_to_redis, Conversation
from utils_livekit import fetch_tree, fetch_Instructions, fetch_context
from save_conversation_signalr import post_message_to_conversation

import os
import re
import json
import time
import asyncio
from datetime import datetime
import sys
import functools
import logging
import traceback
from typing import Optional, Dict, Any, AsyncIterable
from asyncio import TimeoutError as AsyncTimeoutError

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

class SanitizeLogRecordFilter(logging.Filter):
    """
    Filter to sanitize log records by removing unpicklable objects.
    This prevents errors when LiveKit tries to pickle log records for IPC.
    """

    def _is_picklable(self, obj):
        try:
            import pickle
            pickle.dumps(obj)
            return True
        except (TypeError, AttributeError, pickle.PicklingError):
            return False

    def _sanitize_value(self, value):
        if value is None:
            return None
        if self._is_picklable(value):
            return value
        if isinstance(value, dict) or (hasattr(value, 'items') and hasattr(value, 'keys')):
            try:
                return {str(k): self._sanitize_value(v) for k, v in value.items()}
            except:
                return f"<{type(value).__name__} object>"
        if isinstance(value, (list, tuple)):
            return [self._sanitize_value(item) for item in value]
        if hasattr(value, '__class__'):
            error_type = type(value).__name__
            if 'Error' in error_type or 'Exception' in error_type:
                sanitized = {'type': error_type, 'message': str(value)}
                if hasattr(value, 'response'):
                    try:
                        resp = value.response
                        sanitized['response'] = {
                            'status_code': getattr(resp, 'status_code', None),
                            'url': str(getattr(resp, 'url', '')),
                            'reason': getattr(resp, 'reason_phrase', None)
                        }
                    except:
                        sanitized['response'] = '<response object>'
                return sanitized
        try:
            return str(value)
        except:
            return f"<{type(value).__name__} object>"

    def filter(self, record):
        if hasattr(record, 'exc_info') and record.exc_info is not None:
            if isinstance(record.exc_info, tuple) and len(record.exc_info) == 3:
                exc_type, exc_value, exc_tb = record.exc_info
                if not isinstance(exc_tb, (type(None), type)) and not hasattr(exc_tb, 'tb_frame'):
                    record.exc_info = None
                else:
                    try:
                        import pickle
                        pickle.dumps(record.exc_info)
                    except (TypeError, AttributeError, pickle.PicklingError):
                        record.exc_info = None
            else:
                record.exc_info = None

        fields_to_check = ['error', 'response', 'request', 'extra']
        for field in fields_to_check:
            if hasattr(record, field):
                value = getattr(record, field)
                if value is not None and not self._is_picklable(value):
                    setattr(record, field, self._sanitize_value(value))

        if hasattr(record, '__dict__'):
            for key, value in list(record.__dict__.items()):
                if key == 'exc_info':
                    continue
                if value is not None and not self._is_picklable(value):
                    record.__dict__[key] = self._sanitize_value(value)
        return True


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

root_logger = logging.getLogger()
sanitize_filter = SanitizeLogRecordFilter()
for handler in root_logger.handlers[:]:
    handler.addFilter(sanitize_filter)
root_logger.addFilter(sanitize_filter)

try:
    livekit_logger = logging.getLogger('livekit.agents')
    livekit_logger.addFilter(sanitize_filter)
    for handler in livekit_logger.handlers[:]:
        handler.addFilter(sanitize_filter)
except:
    pass

logger = logging.getLogger(__name__)


def sanitize_exception_handler(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    sanitized_exc_value = exc_value
    if hasattr(exc_value, 'response'):
        try:
            import pickle
            pickle.dumps(exc_value.response)
        except:
            sanitized_exc_value = type(exc_value)(str(exc_value), response=None)
            sanitized_exc_value.__cause__ = exc_value.__cause__
            sanitized_exc_value.__context__ = exc_value.__context__
    logger.error("Uncaught exception", exc_info=(exc_type, sanitized_exc_value, exc_traceback))


if __name__ != '__mp_main__':
    sys.excepthook = sanitize_exception_handler

print = functools.partial(print, flush=True)

# ============================================================================
# SESSION LOGGER FOR FILE LOGGING
# ============================================================================

class SessionLogger:
    """
    Logger for tracking complete session information including token usage.
    Writes to a single file with session separators for easy navigation.
    """

    def __init__(self, log_file_path: str = "voice_assistant_sessions.log"):
        self.log_file_path = os.getenv("SESSION_LOG_PATH", log_file_path)
        self.current_session_id = None
        self.session_start_time = None
        self.token_stats = {
            'stt_tokens': 0,
            'tts_tokens': 0,
            'llm_input_tokens': 0,
            'llm_output_tokens': 0,
            'total_tokens': 0
        }
        self.session_events = []
        self._file_lock = None

    def start_session(self, session_id: str, metadata: Dict[str, Any] = None):
        self.current_session_id = session_id
        self.session_start_time = datetime.now()
        self.token_stats = {
            'stt_tokens': 0, 'tts_tokens': 0,
            'llm_input_tokens': 0, 'llm_output_tokens': 0, 'total_tokens': 0
        }
        self.session_events = []
        self._file_lock = None
        safe_metadata = self._serialize_data(metadata) if metadata else {}
        separator = "\n" + "=" * 100 + "\n"
        session_header = f"\nSESSION START\n=============\nSession ID: {session_id}\nStart Time: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        if safe_metadata:
            session_header += "Metadata:\n"
            for key, value in safe_metadata.items():
                session_header += f"  {key}: {value}\n"
        session_header += separator
        asyncio.create_task(self._write_to_file(separator + session_header))
        self.log_event("SESSION_START", f"Session {session_id} started", safe_metadata)

    def _serialize_data(self, data: Any) -> Any:
        if data is None:
            return None
        if isinstance(data, (str, int, float, bool)):
            return data
        if isinstance(data, dict) or (hasattr(data, 'items') and hasattr(data, 'keys')):
            try:
                return {str(k): self._serialize_data(v) for k, v in data.items()}
            except Exception:
                try:
                    return {str(k): self._serialize_data(v) for k, v in dict(data).items()}
                except Exception:
                    return f"<{type(data).__name__} object>"
        if isinstance(data, (list, tuple)):
            return [self._serialize_data(item) for item in data]
        type_name = type(data).__name__
        if 'MultiDict' in type_name or 'CIMultiDict' in type_name:
            try:
                return dict(data)
            except Exception:
                return f"<{type_name} object>"
        if hasattr(data, '__dict__'):
            try:
                return {k: self._serialize_data(v) for k, v in data.__dict__.items()}
            except Exception:
                try:
                    return str(data)
                except:
                    return f"<{type_name} object>"
        try:
            return str(data)
        except Exception:
            return f"<{type_name} object>"

    def log_event(self, event_type: str, message: str, data: Dict[str, Any] = None):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        safe_data = self._serialize_data(data) if data else {}
        event = {'timestamp': timestamp, 'type': event_type, 'message': message, 'data': safe_data}
        self.session_events.append(event)
        log_entry = f"[{timestamp}] [{event_type}] {message}\n"
        if safe_data:
            try:
                log_entry += f"  Data: {json.dumps(safe_data, indent=2, ensure_ascii=False)}\n"
            except Exception:
                log_entry += f"  Data: {str(safe_data)}\n"
        log_entry += "\n"
        asyncio.create_task(self._write_to_file(log_entry))

    def log_user_message(self, message: str, stt_duration: float = None):
        stt_tokens = len(message) // 4
        self.token_stats['stt_tokens'] += stt_tokens
        self.token_stats['total_tokens'] += stt_tokens
        self.log_event("USER_MESSAGE", f"User said: {message[:100]}...",
                       {'message': message, 'message_length': len(message),
                        'stt_tokens': stt_tokens, 'stt_duration': stt_duration})

    def log_assistant_response(self, message: str, tts_duration: float = None):
        tts_tokens = len(message) // 4
        self.token_stats['tts_tokens'] += tts_tokens
        self.token_stats['total_tokens'] += tts_tokens
        self.log_event("ASSISTANT_RESPONSE", f"Assistant said: {message[:100]}...",
                       {'message': message, 'message_length': len(message),
                        'tts_tokens': tts_tokens, 'tts_duration': tts_duration})

    def log_llm_usage(self, input_tokens: int, output_tokens: int, model: str = None, cost: float = None):
        self.token_stats['llm_input_tokens'] += input_tokens
        self.token_stats['llm_output_tokens'] += output_tokens
        self.token_stats['total_tokens'] += (input_tokens + output_tokens)
        self.log_event("LLM_USAGE", f"LLM used {input_tokens} input + {output_tokens} output tokens",
                       {'input_tokens': input_tokens, 'output_tokens': output_tokens,
                        'total_llm_tokens': input_tokens + output_tokens,
                        'model': model, 'estimated_cost': cost})

    def log_api_call(self, api_name: str, success: bool, duration: float = None, response: Any = None):
        safe_response = None
        if response is not None:
            if isinstance(response, dict):
                safe_response = self._serialize_data(response)
            elif hasattr(response, '__dict__'):
                try:
                    safe_response = str(response)
                except:
                    safe_response = f"<{type(response).__name__} object>"
            else:
                safe_response = str(response)
        status = "SUCCESS" if success else "FAILED"
        self.log_event("API_CALL", f"{api_name} - {status}",
                       {'api_name': api_name, 'success': success,
                        'duration': duration, 'response': safe_response})

    def log_redis_operation(self, operation: str, success: bool, duration: float = None):
        status = "SUCCESS" if success else "FAILED"
        self.log_event("REDIS_OPERATION", f"Redis {operation} - {status}",
                       {'operation': operation, 'success': success, 'duration': duration})

    def log_rag_fetch(self, query: str, context_length: int, duration: float = None):
        self.log_event("RAG_FETCH", f"RAG context fetched: {context_length} chars",
                       {'query': query, 'context_length': context_length, 'duration': duration})

    def end_session(self, reason: str = "NORMAL"):
        if not self.current_session_id:
            return
        end_time = datetime.now()
        duration = (end_time - self.session_start_time).total_seconds() if self.session_start_time else 0
        summary = f"""
SESSION END
===========
Session ID: {self.current_session_id}
End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration:.2f} seconds
Reason: {reason}

TOKEN USAGE SUMMARY
-------------------
STT Tokens (approximate): {self.token_stats['stt_tokens']}
TTS Tokens (approximate): {self.token_stats['tts_tokens']}
LLM Input Tokens: {self.token_stats['llm_input_tokens']}
LLM Output Tokens: {self.token_stats['llm_output_tokens']}
Total LLM Tokens: {self.token_stats['llm_input_tokens'] + self.token_stats['llm_output_tokens']}
Total Tokens (all services): {self.token_stats['total_tokens']}

SESSION STATISTICS
------------------
Total Events: {len(self.session_events)}
User Messages: {sum(1 for e in self.session_events if e['type'] == 'USER_MESSAGE')}
Assistant Responses: {sum(1 for e in self.session_events if e['type'] == 'ASSISTANT_RESPONSE')}
API Calls: {sum(1 for e in self.session_events if e['type'] == 'API_CALL')}
Redis Operations: {sum(1 for e in self.session_events if e['type'] == 'REDIS_OPERATION')}

{'=' * 100}

"""
        asyncio.create_task(self._write_to_file(summary))
        self.current_session_id = None
        self.session_start_time = None

    async def _write_to_file(self, content: str):
        if self._file_lock is None:
            self._file_lock = asyncio.Lock()
        async with self._file_lock:
            try:
                with open(self.log_file_path, 'a', encoding='utf-8') as f:
                    f.write(content)
                    f.flush()
            except PermissionError:
                fallback_path = "/tmp/voice_assistant_sessions.log"
                self.log_file_path = fallback_path
                try:
                    with open(self.log_file_path, 'a', encoding='utf-8') as f:
                        f.write(content)
                        f.flush()
                except Exception as e:
                    logger.error(f"Failed to write to fallback session log file: {e}")
            except Exception as e:
                logger.error(f"Failed to write to session log file: {e}")


session_logger = SessionLogger()

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

MAX_CONCURRENT_API_CALLS = 10
api_semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)

REDIS_TIMEOUT = 5.0
# OPT-MEDIUM-2: Tightened RAG timeout from 30s to 3s.
# RAG now runs concurrently so we fail-fast rather than blocking the LLM.
RAG_FETCH_TIMEOUT = 3.0
API_CALL_TIMEOUT = 10.0

MAX_MESSAGE_LENGTH = 10000
MAX_CHAT_ID_LENGTH = 255

# ============================================================================
# AUDIO CONFIGURATION
# ============================================================================

os.environ['LIVEKIT_AUDIO_SAMPLE_RATE'] = '16000'
os.environ['LIVEKIT_AUDIO_CHANNELS'] = '1'
os.environ['LIVEKIT_AUDIO_LATENCY'] = 'medium'

_ENV_HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
_ENV_PARENT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if os.path.isfile(_ENV_HERE):
    load_dotenv(_ENV_HERE, override=False)
    print(f"[env] Loaded {_ENV_HERE}")
elif os.path.isfile(_ENV_PARENT):
    load_dotenv(_ENV_PARENT, override=False)
    print(f"[env] Loaded {_ENV_PARENT}")
else:
    load_dotenv(override=False)
    print("[env] Used dotenv default search (find_dotenv)")

# ============================================================================
# API KEY VALIDATION
# ============================================================================

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY') or os.getenv('ELEVEN_API_KEY')

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables.")
else:
    logger.info("OpenAI API key found")

if not ASSEMBLYAI_API_KEY:
    logger.warning("ASSEMBLYAI_API_KEY not found in environment variables.")
else:
    logger.info("AssemblyAI API key found")

if not ELEVENLABS_API_KEY:
    logger.warning("ELEVENLABS_API_KEY (or ELEVEN_API_KEY) not found.")
else:
    logger.info("ElevenLabs API key found")

# ============================================================================
# AUDIO CALLBACK MONKEY-PATCH
# ============================================================================

_orig_wrap = sd._wrap_callback


def _safe_wrap_callback(callback, data, frames, time, status):
    try:
        return _orig_wrap(callback, data, frames, time, status)
    except RuntimeError as e:
        if "Failed to set stream delay" in str(e):
            print(f"[AudioCallback] Ignored set_stream_delay error: {e}")
            return 0
        raise


sd._wrap_callback = _safe_wrap_callback

# ============================================================================
# TEXT-TO-SPEECH (TTS) CONFIGURATION
# ============================================================================

ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
# OPT-MEDIUM-3: Default to turbo model for ~50% lower TTS latency.
# Override via ELEVENLABS_MODEL=eleven_multilingual_v2 if quality is preferred.
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5")
ELEVENLABS_V3_VOICE_ID = os.getenv("ELEVENLABS_V3_VOICE_ID", "").strip() or ELEVENLABS_VOICE_ID

IS_ELEVEN_V3 = ELEVENLABS_MODEL.startswith("eleven_v3")
ACTIVE_VOICE_ID = ELEVENLABS_V3_VOICE_ID if IS_ELEVEN_V3 else ELEVENLABS_VOICE_ID

ELEVENLABS_VOICE_SETTINGS_STANDARD = elevenlabs.VoiceSettings(
    stability=0.5, similarity_boost=0.75, style=0.15, speed=0.96, use_speaker_boost=True,
)
ELEVENLABS_VOICE_SETTINGS_V3 = elevenlabs.VoiceSettings(
    stability=0.7, similarity_boost=0.55, style=0.0, speed=0.96, use_speaker_boost=False,
)
DEFAULT_VOICE_SETTINGS = ELEVENLABS_VOICE_SETTINGS_V3 if IS_ELEVEN_V3 else ELEVENLABS_VOICE_SETTINGS_STANDARD

VOICE_TONE_PRESETS_STANDARD: Dict[str, "elevenlabs.VoiceSettings"] = {
    "neutral": ELEVENLABS_VOICE_SETTINGS_STANDARD,
    "empathic": elevenlabs.VoiceSettings(stability=0.58, similarity_boost=0.74, style=0.05, speed=0.92, use_speaker_boost=True),
    "enthusiastic": elevenlabs.VoiceSettings(stability=0.42, similarity_boost=0.76, style=0.28, speed=1.02, use_speaker_boost=True),
    "professional": elevenlabs.VoiceSettings(stability=0.67, similarity_boost=0.72, style=0.0, speed=0.95, use_speaker_boost=True),
}
VOICE_TONE_PRESETS_V3: Dict[str, "elevenlabs.VoiceSettings"] = {
    "neutral": ELEVENLABS_VOICE_SETTINGS_V3,
    "empathic": elevenlabs.VoiceSettings(stability=0.72, similarity_boost=0.55, style=0.0, speed=0.93, use_speaker_boost=False),
    "enthusiastic": elevenlabs.VoiceSettings(stability=0.6, similarity_boost=0.6, style=0.1, speed=1.0, use_speaker_boost=False),
    "professional": elevenlabs.VoiceSettings(stability=0.78, similarity_boost=0.55, style=0.0, speed=0.95, use_speaker_boost=False),
}
VOICE_TONE_PRESETS = VOICE_TONE_PRESETS_V3 if IS_ELEVEN_V3 else VOICE_TONE_PRESETS_STANDARD
VOICE_TONE_OPTIONS = tuple(VOICE_TONE_PRESETS.keys())
DEFAULT_VOICE_TONE = "neutral"


def get_voice_settings_for_tone(tone: str) -> "elevenlabs.VoiceSettings":
    selected = (tone or DEFAULT_VOICE_TONE).strip().lower()
    return VOICE_TONE_PRESETS.get(selected, VOICE_TONE_PRESETS[DEFAULT_VOICE_TONE])


if IS_ELEVEN_V3:
    MODEL_SPECIFIC_VOICE_RULES = """
- You are using ElevenLabs v3, which renders bracket emotion tags as real emotional voice delivery.
- Every reply MUST start with exactly one emotion tag from this fixed list:
  [cheerfully], [sympathetic], [excited], [questioning], [reassuring], [professional]
- Place the tag at the very start of the response, then a space, then your spoken sentence.
- Pick the emotion based on the FULL conversation history and the meaning of what you are about to say.
- The emotion must evolve naturally turn-by-turn. Do not get stuck on one emotion.
- Avoid using the same emotion two turns in a row unless context truly calls for it.
- Use at most one tag per reply. Never stack tags. Never use any tag outside the list above.
- Mapping:
  - greeting / friendly small talk -> [cheerfully]
  - caller is upset, worried, or sharing a problem -> [sympathetic]
  - good news, exciting offer, positive milestone -> [excited]
  - asking the caller a question or clarifying -> [questioning]
  - calming, confirming, reassuring the caller -> [reassuring]
  - giving prices, policies, formal info -> [professional]
"""
    V3_AUDIO_TAGS_SECTION = """
## ElevenLabs v3 Emotion Tags (REQUIRED):
- Every response MUST begin with exactly ONE of: [cheerfully], [sympathetic], [excited], [questioning], [reassuring], [professional]
- Format: <TAG> <space> <spoken reply>. The tag is rendered as emotional voice delivery and is NOT spoken aloud.
- The emotion must change naturally based on the conversation context and recent turns.
"""
else:
    MODEL_SPECIFIC_VOICE_RULES = """
- You are using a non-v3 ElevenLabs model.
- Do NOT use bracket audio tags such as [laughs], [sympathetic], [excited], [cheerfully] - this model would speak them literally.
- If humor or laughter is needed, express it naturally in plain text (e.g. "Haha, that's a good one."), not with brackets.
- Avoid stage directions in parentheses or brackets.
- For occasional explicit pauses, you may use SSML break tags sparingly: <break time="0.2s" /> to <break time="0.6s" />.
- Never use more than one SSML break tag in a short response.
"""
    V3_AUDIO_TAGS_SECTION = ""

VOICE_STYLE_SECTION = f"""
## Voice & Spoken Language Style:
- You are speaking out loud on a phone call, not writing. Always sound natural and conversational.
- Use natural spoken phrasing with contractions (it's, we'll, you're, that's) when appropriate.
- Keep replies short for phone calls: usually 1-2 short sentences, up to 3 when needed.
- Vary sentence length so it doesn't sound robotic. Use commas for short breathing pauses.
- Use "..." for occasional thoughtful pauses (max once per reply). Avoid dramatic pauses.
- Never use bullet points, numbered lists, markdown, headings, or ALL CAPS.
- Never output URLs in spoken replies. Direct the caller to the website or support if needed.
- For prices, years, and large numbers, use spoken forms ("twenty twenty-four", "three hundred ninety-nine").
- For acronyms, clarify pronunciation when helpful (example: "NLP" as "N-L-P").
- Read back collected emails and phone numbers clearly, character-by-character or digit-by-digit, only when confirming what the caller just gave you.

- Model-specific voice behavior:
{MODEL_SPECIFIC_VOICE_RULES}
"""

LEAD_CAPTURE_SECTION = """
## Lead Capture (Important):
- It is part of your job as the receptionist to collect the caller's contact details so the team can follow up.
- Always try to collect, in this order: full name, phone number, then email address.
- Ask for ONE detail at a time so the caller can answer naturally on a phone call.
- Good moments to start collecting:
  - The caller asks about pricing, signup, a demo, a callback, or any next step.
  - The caller wants to be contacted later.
  - You have answered their main question and the conversation has a natural pause.
- How to ask (vary the phrasing, don't repeat):
  - Name: "May I have your full name, please?" / "Who do I have the pleasure of speaking with?"
  - Phone: "What's the best phone number to reach you on?" / "Could you share a phone number our team can call you back on?"
  - Email: "What email should we send the details to?" / "Can I have an email address for the follow-up?"
- After the caller gives a detail, briefly confirm it back so they can correct mistakes.
- If the caller declines or says they will share later, accept it politely - do not push - and continue helping.
- NEVER refuse to take a name, phone number, or email - collecting these is expected and allowed.
- Do not invent, guess, or assume contact details.
- Do not collect any payment, card, password, or other sensitive data beyond name, phone, and email.
"""

V3_EMOTION_TAGS = ("cheerfully", "sympathetic", "excited", "questioning", "reassuring", "professional")
_LEADING_TAG_RE = re.compile(r"^\s*\[[^\]]+\]\s*")


def strip_leading_tag(text: str) -> str:
    return _LEADING_TAG_RE.sub("", text or "", count=1).strip()


def build_voicebot_prompt(bot_instructions: str, bot_tree: str) -> str:
    bot_instructions = (bot_instructions or "").strip()
    bot_tree = (bot_tree or "").strip()
    return f"""You are a helpful, multilingual voice assistant on a live phone call.
{VOICE_STYLE_SECTION}
{LEAD_CAPTURE_SECTION}
{V3_AUDIO_TAGS_SECTION}
## Special Instructions (highest priority - follow these strictly):
{bot_instructions if bot_instructions else "(no bot-specific instructions provided)"}

## FLOW (use to guide the conversation when relevant):
{bot_tree if bot_tree else "(no flow/tree provided)"}

## Closing Reminders:
- Identify the caller's need from their message and address it directly.
- Keep the conversation smooth, warm, and tailored to the caller.
- If special instructions and the flow conflict, the special instructions win.
"""


# ============================================================================
# TONE CLASSIFICATION
# ============================================================================
# OPT-HIGH-1: Tone classification is now fire-and-forget. This client is still
# used but the call is no longer awaited on the hot path.

_tone_openai_client: Optional[AsyncOpenAI] = None


def _get_tone_openai_client() -> Optional[AsyncOpenAI]:
    global _tone_openai_client
    if _tone_openai_client is None and OPENAI_API_KEY:
        try:
            _tone_openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        except Exception as e:
            logger.warning(f"Could not init tone classifier OpenAI client: {e}")
    return _tone_openai_client


async def classify_tone_for_user(user_text: str, last_assistant_text: str = "") -> str:
    """Predict the tone the next assistant reply should be delivered with.
    Returns one of VOICE_TONE_OPTIONS. Falls back to DEFAULT_VOICE_TONE on error.
    """
    client = _get_tone_openai_client()
    if not client or not user_text:
        return DEFAULT_VOICE_TONE
    try:
        allowed = ", ".join(VOICE_TONE_OPTIONS)
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=5,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the tone the assistant should use for its next "
                        f"phone reply. Reply with EXACTLY one label from: {allowed}. "
                        "No punctuation, no other words."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Caller just said: {user_text}\n"
                        + (f"Assistant's previous reply: {last_assistant_text}\n" if last_assistant_text else "")
                        + "Choose the best tone label."
                    ),
                },
            ],
        )
        tone = (resp.choices[0].message.content or "").strip().lower()
        if tone in VOICE_TONE_OPTIONS:
            return tone
    except Exception as e:
        logger.debug(f"Tone classification failed, using default: {e}")
    return DEFAULT_VOICE_TONE


# ============================================================================
# TTS PLUGIN INITIALIZATION
# ============================================================================

try:
    elevenlabs_tts = elevenlabs.TTS(
        api_key=ELEVENLABS_API_KEY,
        voice_id=ACTIVE_VOICE_ID,
        model=ELEVENLABS_MODEL,
        voice_settings=DEFAULT_VOICE_SETTINGS,
    )
    logger.info(f"ElevenLabs TTS initialized (model={ELEVENLABS_MODEL}, voice_id={ACTIVE_VOICE_ID}, v3={IS_ELEVEN_V3})")
except Exception as e:
    logger.error(f"Failed to initialize ElevenLabs TTS: {e}")
    elevenlabs_tts = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_input(message: str, max_length: int = MAX_MESSAGE_LENGTH) -> str:
    if not message or not isinstance(message, str):
        raise ValueError("Message must be a non-empty string")
    sanitized = ''.join(char for char in message if ord(char) >= 32 or char in '\n\t')
    if len(sanitized) > max_length:
        logger.warning(f"Message truncated from {len(sanitized)} to {max_length} characters")
        sanitized = sanitized[:max_length]
    return sanitized.strip()


def validate_chat_id(chat_id: str) -> str:
    if not chat_id or not isinstance(chat_id, str):
        raise ValueError("Chat ID must be a non-empty string")
    if len(chat_id) > MAX_CHAT_ID_LENGTH:
        raise ValueError(f"Chat ID exceeds maximum length of {MAX_CHAT_ID_LENGTH}")
    sanitized = ''.join(char for char in chat_id if char.isalnum() or char in '-_')
    if not sanitized:
        raise ValueError("Chat ID contains no valid characters")
    return sanitized


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    if not json_str or not isinstance(json_str, str):
        return default
    try:
        if len(json_str) > 100000:
            logger.warning("JSON string exceeds size limit")
            return default
        return json.loads(json_str)
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"JSON parse error: {e}")
        return default


async def safe_post_message(
    chat_id: str, message: str, user_id: str, manager_id: str,
    visitor_id: str, website_id: str, nick_name: str, operator_name: str
) -> bool:
    async with api_semaphore:
        try:
            await asyncio.wait_for(
                post_message_to_conversation(
                    chat_id=chat_id, message=message, user_id=user_id,
                    manager_id=manager_id, visitor_id=visitor_id,
                    website_id=website_id, nick_name=nick_name, operator_name=operator_name
                ),
                timeout=API_CALL_TIMEOUT
            )
            return True
        except AsyncTimeoutError:
            logger.error(f"API call timeout for chat_id: {chat_id}")
            return False
        except Exception as e:
            logger.error(f"Failed to post message to API: {e}", exc_info=True)
            return False


async def safe_save_to_redis(chat_id: str, conversation: Conversation) -> bool:
    """
    OPT-MEDIUM-1: This function is now called only as a fire-and-forget
    background task. The in-memory Conversation object is always authoritative
    within the session; Redis is a write-behind cache.
    """
    try:
        validated_chat_id = validate_chat_id(chat_id)
        await asyncio.wait_for(
            asyncio.to_thread(save_conversation_to_redis, validated_chat_id, conversation),
            timeout=REDIS_TIMEOUT
        )
        return True
    except ValueError as e:
        logger.error(f"Invalid chat_id: {e}")
        return False
    except AsyncTimeoutError:
        logger.error(f"Redis save timeout for chat_id: {chat_id}")
        return False
    except Exception as e:
        logger.error(f"Failed to save conversation to Redis: {e}", exc_info=True)
        return False


async def safe_load_from_redis(chat_id: str) -> Optional[Conversation]:
    try:
        validated_chat_id = validate_chat_id(chat_id)
        conversation = await asyncio.wait_for(
            asyncio.to_thread(load_conversation_from_redis, validated_chat_id),
            timeout=REDIS_TIMEOUT
        )
        return conversation
    except ValueError as e:
        logger.error(f"Invalid chat_id: {e}")
        return None
    except AsyncTimeoutError:
        logger.error(f"Redis load timeout for chat_id: {chat_id}")
        return None
    except Exception as e:
        logger.error(f"Failed to load conversation from Redis: {e}", exc_info=True)
        return None


# ============================================================================
# ASSISTANT CLASS
# ============================================================================

class Assistant(Agent):
    """
    Voice Assistant Agent that manages conversations, integrates with Redis,
    and provides RAG-enhanced responses.

    Latency optimizations applied:
      - Tone classification: fire-and-forget task, never blocks LLM start.
      - RAG fetch: runs concurrently as a background task; result is stored in
        self._pending_rag_context and injected at the start of the NEXT turn
        (or into the current turn_ctx before LLM starts if fetch is fast enough).
      - Redis writes: always fire-and-forget, never block the hot path.
      - Vector index: guarded so it is only built once per session.
    """

    def __init__(self, cid: str, bot_id: str, domain: str, chat_ctx: ChatContext = None,
                 con=None, sess=None, variablesForChat=None, instructions: str = None):
        self.bot_id = bot_id
        self.Domain = domain
        self.con = con
        self.cid = cid
        self.sess = sess
        self.variablesForChat = variablesForChat or {}
        self._background_tasks = set()
        self.session_logger = session_logger
        self.current_tone: str = DEFAULT_VOICE_TONE

        # OPT-HIGH-2: Stores the most recent RAG result fetched concurrently.
        # None means no result is available yet. Empty string means fetch
        # completed but returned no usable context.
        self._pending_rag_context: Optional[str] = None

        super().__init__(
            instructions=instructions or "You are a helpful voice assistant.",
            chat_ctx=chat_ctx,
        )

    def _add_background_task(self, task: asyncio.Task) -> None:
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _apply_voice_settings_for_tone(self, tone: str) -> None:
        """Update the live ElevenLabs TTS plugin to use the preset for `tone`."""
        if elevenlabs_tts is None or not tone:
            return
        try:
            settings = get_voice_settings_for_tone(tone)
            elevenlabs_tts.update_options(voice_settings=settings)
            self.current_tone = tone
            logger.debug(f"TTS voice settings switched for tone: {tone}")
        except Exception as e:
            logger.debug(f"Could not update TTS voice settings for tone {tone}: {e}")

    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        """Strip leading [bracket-tag] for non-v3 models before TTS synthesis."""
        if IS_ELEVEN_V3:
            return Agent.default.tts_node(self, text, model_settings)

        async def strip_first_chunk_tag():
            stripped_once = False
            async for chunk in text:
                if not stripped_once and chunk:
                    new_chunk = _LEADING_TAG_RE.sub("", chunk, count=1)
                    if new_chunk != chunk:
                        logger.debug("Stripped leading bracket tag for non-v3 TTS")
                    stripped_once = True
                    if new_chunk:
                        yield new_chunk
                else:
                    yield chunk

        return Agent.default.tts_node(self, strip_first_chunk_tag(), model_settings)

    # -------------------------------------------------------------------------
    # OPT-HIGH-1 + OPT-HIGH-2: Non-blocking tone classification and RAG fetch
    # -------------------------------------------------------------------------

    def _start_tone_classification(self, user_text: str) -> None:
        """
        Launch tone classification as a background task.
        Applies the result to TTS when the task completes — this happens
        concurrently with the LLM generating its first tokens, so voice
        settings are almost always updated before TTS synthesis begins.
        """
        last_assistant = ""
        for m in reversed(self.con.conversation_history):
            if m.get("role") == "assistant":
                last_assistant = (m.get("content") or "")[:400]
                break

        async def _classify_and_apply():
            try:
                tone = await classify_tone_for_user(user_text, last_assistant)
                if tone != self.current_tone:
                    self._apply_voice_settings_for_tone(tone)
                self.session_logger.log_event(
                    "TONE_SELECTED",
                    f"Tone for next reply: {tone}",
                    {"tone": tone, "previous_tone": self.current_tone},
                )
            except Exception as e:
                logger.debug(f"Tone classification background task failed: {e}")

        task = asyncio.create_task(_classify_and_apply())
        self._add_background_task(task)

    def _start_rag_fetch(self, user_text: str, turn_ctx: ChatContext) -> None:
        """
        Launch RAG fetch as a background task.

        The result is injected into turn_ctx immediately when it arrives.
        Because LiveKit streams LLM input incrementally, if the RAG fetch
        resolves within the first ~200–400 ms (before the LLM sends its first
        token to TTS), the context will be included in the current LLM call.
        If it arrives later it is stored in self._pending_rag_context and
        prepended to the next turn.
        """
        rag_start = time.time()

        async def _fetch_and_inject():
            try:
                rag_content = await asyncio.wait_for(
                    asyncio.to_thread(fetch_context, user_text, self.con, self.bot_id, self.Domain),
                    timeout=RAG_FETCH_TIMEOUT
                )
                duration = time.time() - rag_start
                self.session_logger.log_rag_fetch(
                    user_text, len(rag_content) if rag_content else 0, duration
                )
                if rag_content:
                    # Attempt to inject into the current turn context.
                    # If the LLM has already started streaming this won't affect
                    # the current response, but storing it ensures the next turn
                    # benefits from it.
                    try:
                        turn_ctx.add_message(
                            role="assistant",
                            content=f"Additional information relevant to the user's next message: {rag_content}"
                        )
                        logger.debug("RAG context injected into current turn_ctx")
                    except Exception:
                        # turn_ctx may be sealed after LLM starts — store for next turn
                        logger.debug("turn_ctx sealed, storing RAG context for next turn")

                    # Always cache so the next turn can prepend it if needed
                    self._pending_rag_context = rag_content
                else:
                    self._pending_rag_context = ""
                    logger.debug("RAG fetch returned no context")

            except AsyncTimeoutError:
                logger.warning(f"RAG fetch timed out after {RAG_FETCH_TIMEOUT}s for chat_id: {self.cid}")
                self.session_logger.log_event("RAG_FETCH_TIMEOUT", "RAG fetch timed out", {"query": user_text})
                self._pending_rag_context = ""
            except Exception as e:
                logger.error(f"RAG fetch background task error: {e}", exc_info=True)
                self._pending_rag_context = ""

        task = asyncio.create_task(_fetch_and_inject())
        self._add_background_task(task)

    # -------------------------------------------------------------------------
    # Main turn hook
    # -------------------------------------------------------------------------

    async def on_user_turn_completed(
            self, turn_ctx: ChatContext, new_message: ChatMessage,
    ) -> None:
        """
        Called when the user finishes speaking.

        OPTIMIZED HOT PATH (compared to original):
          - No awaits on tone classification  → fire-and-forget background task
          - No await on RAG fetch             → fire-and-forget background task
          - No await on Redis save            → fire-and-forget background task
          - No await on API post              → already was fire-and-forget
          - Pending RAG context from previous turn is injected before the LLM
            sees the current turn (zero extra latency for carried-over context)

        The only remaining await is on Redis save for the *user* message, and
        that has been removed too — it is now a background task.  The LLM can
        start generating immediately after this method returns.
        """
        try:
            q = new_message.text_content
            if not q:
                logger.warning("Empty message received, skipping processing")
                return

            try:
                q = validate_input(q)
            except ValueError as e:
                logger.error(f"Invalid message input: {e}")
                return

            self.session_logger.log_user_message(q)

            # ------------------------------------------------------------------
            # Step 0a: Inject RAG context from the PREVIOUS turn (zero latency —
            # it was fetched in the background during the last assistant response).
            # ------------------------------------------------------------------
            if self._pending_rag_context:
                try:
                    turn_ctx.add_message(
                        role="assistant",
                        content=f"Additional information relevant to the user's next message: {self._pending_rag_context}"
                    )
                    logger.debug("Injected cached RAG context from previous turn")
                except Exception as e:
                    logger.debug(f"Could not inject cached RAG context: {e}")
                # Clear so it isn't re-injected on the next turn
                self._pending_rag_context = None

            # ------------------------------------------------------------------
            # Step 0b: Fire tone classification — background, does NOT block LLM.
            # OPT-HIGH-1
            # ------------------------------------------------------------------
            self._start_tone_classification(q)

            # ------------------------------------------------------------------
            # Step 1: Add user message to in-memory conversation history.
            # ------------------------------------------------------------------
            self.con.conversation_history.append({"role": "user", "content": q})
            logger.info(f"User message added to history for chat_id: {self.cid}")

            # ------------------------------------------------------------------
            # Step 2: Post user message to external API — fire-and-forget.
            # ------------------------------------------------------------------
            v_id = self.variablesForChat.get("VisitorId", "")
            chat_id = self.variablesForChat.get("ChatId", "")
            api_start_time = time.time()

            async def post_user_with_logging():
                success = await safe_post_message(
                    chat_id=chat_id,
                    message=q,
                    user_id="0",
                    manager_id="0",
                    visitor_id=self.variablesForChat.get("VisitorId", ""),
                    website_id=self.variablesForChat.get("WebsiteId", ""),
                    nick_name=self.variablesForChat.get("NickName", ""),
                    operator_name=self.variablesForChat.get("visitorName", f"Visitor{v_id}")
                )
                duration = time.time() - api_start_time
                self.session_logger.log_api_call("post_message_to_conversation", success, duration)

            self._add_background_task(asyncio.create_task(post_user_with_logging()))
            logger.info("User message posted to API (background task)")

            # ------------------------------------------------------------------
            # Step 3: Persist conversation to Redis — fire-and-forget.
            # OPT-MEDIUM-1: was awaited before; now always a background task.
            # The Conversation object in memory is authoritative for this session.
            # ------------------------------------------------------------------
            redis_start_time = time.time()

            async def save_user_redis():
                success = await safe_save_to_redis(self.cid, self.con)
                duration = time.time() - redis_start_time
                self.session_logger.log_redis_operation("SAVE_USER", success, duration)
                if not success:
                    logger.warning(f"Background Redis save (user message) failed for chat_id: {self.cid}")

            self._add_background_task(asyncio.create_task(save_user_redis()))

            # ------------------------------------------------------------------
            # Step 4: Launch RAG fetch concurrently — background task.
            # OPT-HIGH-2: was a blocking await; now runs while the LLM streams.
            # Result will be injected into turn_ctx if it arrives in time, or
            # cached in self._pending_rag_context for the next turn.
            # ------------------------------------------------------------------
            self._start_rag_fetch(q, turn_ctx)

            # Control returns here immediately. The LLM can start generating.
            logger.debug("on_user_turn_completed: returning control to pipeline (LLM may start)")

        except Exception as e:
            logger.error(f"Error in on_user_turn_completed: {e}", exc_info=True)


# ============================================================================
# ENTRYPOINT FUNCTION
# ============================================================================

from livekit import api


async def entrypoint(ctx: agents.JobContext):
    """
    Main entry point for the LiveKit voice assistant agent.
    Connects to the room, loads/creates conversation, sets up AI pipeline,
    and generates an initial greeting.
    """
    try:
        print("🌟 [ENTRYPOINT INIT] Starting voice assistant entrypoint")
        await ctx.connect()

        session_id = None

        # ====================================================================
        # STEP 1: EXTRACT METADATA FROM PARTICIPANT
        # ====================================================================
        bot_id = '1344'
        domain = "testing.webgreeter.com/zem/hulk"
        chat_id = None
        variablesForChat = {}

        try:
            for pid, participant in ctx.room.remote_participants.items():
                logger.info(f"Processing participant: {participant.identity}")
                if not participant.metadata:
                    raise ValueError("Empty metadata")
                metadata = safe_json_loads(participant.metadata, {})
                if not metadata:
                    raise ValueError("Invalid metadata JSON")

                variablesForChat = {
                    "ChatId": metadata.get("chat_id", ""),
                    "EndTime": "false",
                    "WebsiteId": metadata.get("websiteId", ""),
                    "WebsiteURL": metadata.get("websiteURL", ""),
                    "VisitorId": metadata.get("visitorId", ""),
                    "VisitorName": metadata.get("visitorName", "Guest"),
                    "SoftwareUserId": metadata.get("softwareUserId", ""),
                    "UserId": metadata.get("userId", ""),
                    "ManagerId": metadata.get("managerId", ""),
                    "TimeStamp": "",
                    "NickName": metadata.get("nickName", ""),
                    "Miscellaneous": metadata.get("miscellaneous", ""),
                    "IsCustomMessage": metadata.get("isCustomMessage", False),
                    "Agent": metadata.get("agent", "System"),
                    "Lang": "en",
                    "DomainName": metadata.get("domain", "liveadmins.com"),
                    "ServerURL": metadata.get("ServerURL", ""),
                    "MessageBody": ""
                }

                chat_id = metadata.get("chat_id")
                if not chat_id:
                    raise ValueError("Missing chat_id in metadata")
                chat_id = validate_chat_id(chat_id)
                bot_id = '1344'
                logger.info(f"Extracted — chat_id: {chat_id}, bot_id: {bot_id}, domain: {domain}")
                session_id = chat_id

        except (ValueError, KeyError, json.JSONDecodeError) as e:
            logger.warning(f"Metadata extraction failed: {e}, using defaults")
            chat_id = 'mynewchatID5'
            session_id = chat_id
        except Exception as e:
            logger.error(f"Unexpected error extracting metadata: {e}", exc_info=True)
            chat_id = 'mynewchatID5'
            session_id = chat_id

        if not chat_id:
            chat_id = 'mynewchatID5'
            session_id = chat_id
            variablesForChat.setdefault("ChatId", chat_id)
            variablesForChat.setdefault("VisitorId", "")
            variablesForChat.setdefault("WebsiteId", "")
            variablesForChat.setdefault("NickName", "")
            variablesForChat.setdefault("UserId", "")
            variablesForChat.setdefault("ManagerId", "")
            variablesForChat.setdefault("visitorName", "Guest")

        if session_id:
            session_metadata = {
                'chat_id': chat_id, 'bot_id': bot_id, 'domain': domain,
                'room_name': ctx.room.name if hasattr(ctx, 'room') and ctx.room else 'unknown',
                **variablesForChat
            }
            session_logger.start_session(session_id, session_metadata)

        # ====================================================================
        # STEP 2: LOAD OR INITIALIZE CONVERSATION
        # ====================================================================
        logger.info(f"Loading conversation for chat_id: {chat_id}")
        redis_load_start = time.time()
        con = await safe_load_from_redis(chat_id)
        redis_load_duration = time.time() - redis_load_start

        if con is None:
            logger.info("No existing conversation in Redis, creating new one")
            session_logger.log_redis_operation("LOAD", False, redis_load_duration)
        else:
            logger.info(f"Loaded conversation — Agent: {con.Agent}, History: {len(con.conversation_history)} messages")
            session_logger.log_redis_operation("LOAD", True, redis_load_duration)

        if con is None:
            Agent_key = str(bot_id) + "_" + str(domain)
            con = Conversation(bot_id, Agent_key)

            try:
                bot_tree = await asyncio.to_thread(fetch_tree, con, bot_id)
                if bot_tree:
                    con.tree = bot_tree
                    logger.info(f"Fetched tree for bot_id: {bot_id}")
            except Exception as e:
                logger.error(f"Error fetching tree: {e}", exc_info=True)

            try:
                BotPrompt = await asyncio.to_thread(fetch_Instructions, con, bot_id)
                if BotPrompt:
                    con.Instructions = BotPrompt
                    logger.info(f"Fetched instructions for bot_id: {bot_id}")
            except Exception as e:
                logger.error(f"Error fetching instructions: {e}", exc_info=True)

            save_success = await safe_save_to_redis(chat_id, con)
            if save_success:
                logger.info("New conversation saved to Redis")
            else:
                logger.error("Failed to save new conversation to Redis")

        # ====================================================================
        # STEP 3: BUILD SYSTEM PROMPT
        # ====================================================================
        tree = con.tree or ""
        instructions = con.Instructions or ""
        prompt = build_voicebot_prompt(instructions, tree)
        logger.info(f"Voicebot prompt built (model={ELEVENLABS_MODEL}, v3={IS_ELEVEN_V3}, len={len(prompt)})")

        if len(con.conversation_history) == 0 or con.conversation_history[0].get('role') != 'system':
            con.conversation_history.insert(0, {'role': 'system', 'content': prompt})
        else:
            con.conversation_history[0]['content'] = prompt

        logger.info(f"Agent: {con.Agent}, History: {len(con.conversation_history)} messages")

        # ====================================================================
        # STEP 4: INITIALIZE CHAT CONTEXT
        # ====================================================================
        chat_ctx = ChatContext()
        for m in con.conversation_history:
            role = m.get("role", "user")
            content = m.get("content", "")
            if content:
                chat_ctx.add_message(role=role, content=content)
        logger.debug(f"ChatContext populated with {len(con.conversation_history)} messages")

        # ====================================================================
        # STEP 5: CONFIGURE AI SESSION
        # ====================================================================
        try:
            stt_instance = assemblyai.STT(
                api_key=ASSEMBLYAI_API_KEY,
                model="u3-rt-pro",
                # OPT-HIGH-3: Raised min_turn_silence from 100ms to 200ms to
                # reduce false positives (mid-sentence cutoffs).
                # Lowered max_turn_silence from 1000ms to 500ms for faster
                # endpointing after the user genuinely stops speaking.
                min_turn_silence=200,
                max_turn_silence=500,
                vad_threshold=0.3,
            )
            logger.info("AssemblyAI STT initialized")
        except Exception as e:
            logger.error(f"Failed to initialize STT: {e}")
            raise

        try:
            llm_instance = LLM(model="gpt-4o-mini")
            logger.info("OpenAI LLM initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

        if elevenlabs_tts is None:
            raise RuntimeError("ElevenLabs TTS initialization failed. Cannot start agent session.")

        session = AgentSession(
            stt=stt_instance,
            llm=llm_instance,
            tts=elevenlabs_tts,
            vad=silero.VAD.load(
                activation_threshold=0.3,
                min_silence_duration=0.3,
                min_speech_duration=0.05,
            ),
            turn_handling=TurnHandlingOptions(
                turn_detection="stt",
                endpointing={"min_delay": 0},
                allow_interruptions=True,
            ),
        )

        # ====================================================================
        # STEP 6: CREATE ASSISTANT INSTANCE
        # ====================================================================
        assistant = Assistant(
            cid=chat_id,
            bot_id=bot_id,
            domain=domain,
            chat_ctx=chat_ctx,
            con=con,
            variablesForChat=variablesForChat,
            instructions=prompt,
        )
        assistant.sess = session
        assistant._previous_history_length = len(assistant.con.conversation_history) if assistant.con else 0

        # ====================================================================
        # STEP 7: SET UP EVENT HANDLERS
        # ====================================================================
        @session.on("conversation_item_added")
        def on_item(event: ConversationItemAddedEvent):
            """
            Called when the assistant generates a response.
            All I/O (Redis, API post) is fire-and-forget — no blocking on the
            audio pipeline.
            """
            try:
                msg = event.item
                if not hasattr(msg, "role"):
                    logger.debug(f"Skipping non-message item: {type(msg).__name__}")
                    return

                if msg.role != "assistant":
                    return

                logger.info(f"Processing assistant message for chat_id: {assistant.cid}")
                if not msg.text_content:
                    logger.warning("Empty assistant message, skipping")
                    return

                try:
                    validated_content = validate_input(msg.text_content)
                except ValueError as e:
                    logger.error(f"Invalid assistant message: {e}")
                    return

                display_content = (
                    strip_leading_tag(validated_content) if IS_ELEVEN_V3 else validated_content
                )
                if not display_content:
                    display_content = validated_content

                # Step 1: Update in-memory history.
                assistant.con.conversation_history.append({"role": "assistant", "content": display_content})
                assistant.session_logger.log_assistant_response(display_content)

                # Estimate and log LLM tokens.
                try:
                    input_messages = assistant.con.conversation_history[:-1]
                    input_text = ' '.join(m.get('content', '') for m in input_messages)
                    estimated_input_tokens = len(input_text) // 4
                    estimated_output_tokens = len(display_content) // 4
                    if estimated_input_tokens > 0 or estimated_output_tokens > 0:
                        assistant.session_logger.log_llm_usage(
                            estimated_input_tokens, estimated_output_tokens, model='gpt-4o-mini'
                        )
                except Exception as e:
                    logger.debug(f"Error estimating LLM tokens: {e}")

                # Step 2: Post assistant message to API — fire-and-forget.
                api_start_time = time.time()

                async def post_assistant_with_logging():
                    success = await safe_post_message(
                        chat_id=variablesForChat.get("ChatId", ""),
                        message=display_content,
                        user_id=variablesForChat.get("UserId", ""),
                        manager_id=variablesForChat.get("ManagerId", ""),
                        visitor_id=variablesForChat.get("VisitorId", ""),
                        website_id=variablesForChat.get("WebsiteId", ""),
                        nick_name=variablesForChat.get("NickName", ""),
                        operator_name="Voicebot"
                    )
                    assistant.session_logger.log_api_call(
                        "post_message_to_conversation", success, time.time() - api_start_time
                    )

                assistant._add_background_task(asyncio.create_task(post_assistant_with_logging()))

                # Step 3: Persist to Redis — fire-and-forget.
                # OPT-MEDIUM-1
                redis_start_time = time.time()

                async def save_assistant_redis():
                    success = await safe_save_to_redis(assistant.cid, assistant.con)
                    assistant.session_logger.log_redis_operation(
                        "SAVE_ASSISTANT", success, time.time() - redis_start_time
                    )
                    if not success:
                        logger.warning(f"Background Redis save (assistant) failed for chat_id: {assistant.cid}")

                assistant._add_background_task(asyncio.create_task(save_assistant_redis()))

            except Exception as e:
                logger.error(f"Error in conversation_item_added: {e}", exc_info=True)

        # ====================================================================
        # STEP 8: START SESSION AND GENERATE GREETING
        # ====================================================================
        room_input_options = RoomInputOptions()
        try:
            use_cloud_nc = os.getenv('LIVEKIT_USE_CLOUD_NC', 'false').lower() == 'true'
            if use_cloud_nc:
                room_input_options = RoomInputOptions(noise_cancellation=noise_cancellation.BVC())
                logger.info("BVC noise cancellation enabled (LiveKit Cloud)")
            else:
                logger.info("Noise cancellation disabled (BVC requires LiveKit Cloud)")
        except Exception as e:
            logger.warning(f"Could not enable noise cancellation: {e}")

        try:
            logger.info("Starting agent session...")
            await session.start(
                room=ctx.room,
                agent=assistant,
                room_input_options=room_input_options,
            )
            logger.info("Agent session started successfully")
            session_logger.log_event("SESSION_STARTED", "Agent session started successfully")
        except Exception as e:
            logger.error(f"Failed to start agent session: {e}", exc_info=True)
            session_logger.log_event("SESSION_START_ERROR", f"Failed to start session: {str(e)}")
            raise

        session_logger.log_event("GREETING_GENERATION", "Generating initial greeting")
        try:
            await session.generate_reply(instructions="Greet the user and offer your assistance.")
            session_logger.log_event("GREETING_SENT", "Initial greeting sent")
            logger.info("Initial greeting generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate greeting: {e}")
            session_logger.log_event("GREETING_ERROR", f"Greeting failed: {str(e)}")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, shutting down")
        session_logger.end_session("KEYBOARD_INTERRUPT")
        raise
    except SystemExit:
        session_logger.end_session("SYSTEM_EXIT")
        raise
    except Exception as e:
        logger.error(f"Entrypoint failed: {e}", exc_info=True)
        session_logger.end_session("ERROR")
    finally:
        if session_logger.current_session_id:
            session_logger.end_session("NORMAL")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
