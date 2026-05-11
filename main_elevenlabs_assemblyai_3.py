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
  3. VAD max_turn_silence lowered (1000 ms -> 500 ms) and min_turn_silence raised
     (100 ms -> 200 ms) for better natural endpointing without false positives.

MEDIUM IMPACT:
  4. Redis writes are now fire-and-forget background tasks.
  5. Vector index is loaded ONCE per session (LlamaIndex caches on Conversation).
  6. ElevenLabs model defaults to eleven_turbo_v2_5 for lower TTS latency.

BUGFIXES:
---------
  BF-1. RAG_FETCH_TIMEOUT raised to 8s (env: RAG_FETCH_TIMEOUT_S).
  BF-2. Redis save/load retries once before giving up (env: REDIS_TIMEOUT_S).
  BF-3. API 401/403 logged as AUTH FAILURE warning, not retried.

FEATURE ADDITIONS:
------------------
  F-1. Auto-disconnect on goodbye phrases.
  F-2. Silence timeout (default 60s, env: SILENCE_TIMEOUT_S).
  F-3. Sales agent handoff with hold music:
         - Detects request for human/sales agent.
         - Collects name -> phone -> email one field at a time.
         - Plays hold music (BuiltinAudioClip.OFFICE_HOURS or custom URL via
           HOLD_MUSIC_URL env var) for HOLD_MUSIC_DURATION_S seconds.
         - Speaks farewell-style confirmation, then resumes normal conversation.
  F-4. Removed sounddevice import and monkey-patch (was suppressing a harmless
       "Failed to set stream delay" error from APM; the dependency is not needed).
"""

# ============================================================================
# IMPORTS
# ============================================================================

from dotenv import load_dotenv

# sounddevice import removed (see F-4 in changelog above)

from livekit.agents import (
    AgentSession, Agent, RoomInputOptions, function_tool, RunContext,
    BackgroundAudioPlayer, AudioConfig, BuiltinAudioClip,
    TurnHandlingOptions, AutoSubscribe,
)
from livekit import agents
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
    """Filter to sanitize log records by removing unpicklable objects."""

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
            except Exception:
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
                    except Exception:
                        sanitized['response'] = '<response object>'
                return sanitized
        try:
            return str(value)
        except Exception:
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
        for field in ['error', 'response', 'request', 'extra']:
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
except Exception:
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
        except Exception:
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
    """Logger for tracking complete session information including token usage."""

    def __init__(self, log_file_path: str = "voice_assistant_sessions.log"):
        self.log_file_path = os.getenv("SESSION_LOG_PATH", log_file_path)
        self.current_session_id = None
        self.session_start_time = None
        self.token_stats = {
            'stt_tokens': 0, 'tts_tokens': 0,
            'llm_input_tokens': 0, 'llm_output_tokens': 0, 'total_tokens': 0
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
        header = (
            f"\nSESSION START\n=============\n"
            f"Session ID: {session_id}\n"
            f"Start Time: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        if safe_metadata:
            header += "Metadata:\n"
            for key, value in safe_metadata.items():
                header += f"  {key}: {value}\n"
        header += separator
        asyncio.create_task(self._write_to_file(separator + header))
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
                except Exception:
                    return f"<{type_name} object>"
        try:
            return str(data)
        except Exception:
            return f"<{type_name} object>"

    def log_event(self, event_type: str, message: str, data: Dict[str, Any] = None):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        safe_data = self._serialize_data(data) if data else {}
        self.session_events.append({'timestamp': timestamp, 'type': event_type,
                                    'message': message, 'data': safe_data})
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
        safe_response = str(response) if response is not None else None
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
                self.log_file_path = "/tmp/voice_assistant_sessions.log"
                try:
                    with open(self.log_file_path, 'a', encoding='utf-8') as f:
                        f.write(content)
                        f.flush()
                except Exception as e:
                    logger.error(f"Failed to write to fallback log file: {e}")
            except Exception as e:
                logger.error(f"Failed to write to session log file: {e}")


session_logger = SessionLogger()

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

MAX_CONCURRENT_API_CALLS = 10
api_semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)

REDIS_TIMEOUT = float(os.getenv("REDIS_TIMEOUT_S", "8.0"))
RAG_FETCH_TIMEOUT = float(os.getenv("RAG_FETCH_TIMEOUT_S", "8.0"))
API_CALL_TIMEOUT = 10.0
MAX_MESSAGE_LENGTH = 10000
MAX_CHAT_ID_LENGTH = 255
REDIS_MAX_RETRIES = 1
REDIS_RETRY_DELAY = 1.0

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

logger.info("OpenAI API key found") if OPENAI_API_KEY else logger.warning("OPENAI_API_KEY not found.")
logger.info("AssemblyAI API key found") if ASSEMBLYAI_API_KEY else logger.warning("ASSEMBLYAI_API_KEY not found.")
logger.info("ElevenLabs API key found") if ELEVENLABS_API_KEY else logger.warning("ELEVENLABS_API_KEY not found.")

# ============================================================================
# TEXT-TO-SPEECH (TTS) CONFIGURATION
# ============================================================================

ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
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
- The emotion must evolve naturally turn-by-turn. Do not get stuck on one emotion.
- Use at most one tag per reply. Never stack tags. Never use any tag outside the list above.
"""
    V3_AUDIO_TAGS_SECTION = """
## ElevenLabs v3 Emotion Tags (REQUIRED):
- Every response MUST begin with exactly ONE of: [cheerfully], [sympathetic], [excited], [questioning], [reassuring], [professional]
- Format: <TAG> <space> <spoken reply>. The tag is rendered as emotional voice delivery and is NOT spoken aloud.
"""
else:
    MODEL_SPECIFIC_VOICE_RULES = """
- You are using a non-v3 ElevenLabs model.
- Do NOT use bracket audio tags - this model would speak them literally.
- For occasional explicit pauses, use SSML break tags sparingly: <break time="0.2s" /> to <break time="0.6s" />.
- Never use more than one SSML break tag in a short response.
"""
    V3_AUDIO_TAGS_SECTION = ""

VOICE_STYLE_SECTION = f"""
## Voice & Spoken Language Style:
- You are speaking out loud on a phone call, not writing. Always sound natural and conversational.
- Use contractions (it's, we'll, you're, that's) when appropriate.
- Keep replies short: usually 1-2 short sentences, up to 3 when needed.
- Never use bullet points, numbered lists, markdown, headings, or ALL CAPS.
- Never output URLs in spoken replies.
- For prices and large numbers, use spoken forms ("three hundred ninety-nine").
- Model-specific voice behavior:
{MODEL_SPECIFIC_VOICE_RULES}
"""

LEAD_CAPTURE_SECTION = """
## Lead Capture (Important):
- Collect caller contact details so the team can follow up: name -> phone -> email (one at a time).
- Good moments: caller asks about pricing, signup, a demo, a callback, or any next step.
- After the caller gives a detail, briefly confirm it back.
- If the caller declines, accept politely and continue helping.
- NEVER collect payment, card, password, or any data beyond name, phone, and email.
"""

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
## Special Instructions (highest priority):
{bot_instructions if bot_instructions else "(no bot-specific instructions provided)"}

## FLOW:
{bot_tree if bot_tree else "(no flow/tree provided)"}

## Closing Reminders:
- Identify the caller's need and address it directly.
- Keep the conversation smooth, warm, and tailored to the caller.
- If special instructions and the flow conflict, the special instructions win.
"""


# ============================================================================
# TONE CLASSIFICATION
# ============================================================================

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
    """Predict the tone the next assistant reply should be delivered with."""
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
                        f"Classify the tone for the next phone reply. "
                        f"Reply with EXACTLY one label from: {allowed}. No punctuation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Caller just said: {user_text}\n"
                        + (f"Previous assistant reply: {last_assistant_text}\n" if last_assistant_text else "")
                        + "Choose the best tone label."
                    ),
                },
            ],
        )
        tone = (resp.choices[0].message.content or "").strip().lower()
        if tone in VOICE_TONE_OPTIONS:
            return tone
    except Exception as e:
        logger.debug(f"Tone classification failed: {e}")
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
            return default
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"JSON parse error: {e}")
        return default


async def safe_post_message(
    chat_id: str, message: str, user_id: str, manager_id: str,
    visitor_id: str, website_id: str, nick_name: str, operator_name: str
) -> bool:
    """BF-3: Distinguishes 401/403 auth failures from transient errors."""
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
            err_str = str(e)
            if "401" in err_str or "403" in err_str or "Access denied" in err_str:
                logger.warning(
                    f"[API AUTH FAILURE] Post rejected for chat_id: {chat_id}. "
                    f"Likely cause: server IP not whitelisted. Error: {err_str}"
                )
            else:
                logger.error(f"Failed to post message to API: {e}", exc_info=True)
            return False


async def safe_save_to_redis(chat_id: str, conversation: Conversation) -> bool:
    """BF-2: Retries once before giving up."""
    try:
        validated_chat_id = validate_chat_id(chat_id)
    except ValueError as e:
        logger.error(f"Invalid chat_id for Redis save: {e}")
        return False
    last_exc = None
    for attempt in range(REDIS_MAX_RETRIES + 1):
        try:
            await asyncio.wait_for(
                asyncio.to_thread(save_conversation_to_redis, validated_chat_id, conversation),
                timeout=REDIS_TIMEOUT
            )
            return True
        except AsyncTimeoutError:
            last_exc = f"timeout after {REDIS_TIMEOUT}s"
            logger.warning(f"Redis save attempt {attempt + 1} timed out for chat_id: {chat_id}")
        except Exception as e:
            last_exc = str(e)
            logger.warning(f"Redis save attempt {attempt + 1} failed for chat_id: {chat_id}: {e}")
        if attempt < REDIS_MAX_RETRIES:
            await asyncio.sleep(REDIS_RETRY_DELAY)
    logger.error(f"Redis save failed after {REDIS_MAX_RETRIES + 1} attempts for chat_id: {chat_id}. Last: {last_exc}")
    return False


async def safe_load_from_redis(chat_id: str) -> Optional[Conversation]:
    """BF-2: Retries once before returning None."""
    try:
        validated_chat_id = validate_chat_id(chat_id)
    except ValueError as e:
        logger.error(f"Invalid chat_id for Redis load: {e}")
        return None
    last_exc = None
    for attempt in range(REDIS_MAX_RETRIES + 1):
        try:
            conversation = await asyncio.wait_for(
                asyncio.to_thread(load_conversation_from_redis, validated_chat_id),
                timeout=REDIS_TIMEOUT
            )
            return conversation
        except AsyncTimeoutError:
            last_exc = f"timeout after {REDIS_TIMEOUT}s"
            logger.warning(f"Redis load attempt {attempt + 1} timed out for chat_id: {chat_id}")
        except Exception as e:
            last_exc = str(e)
            logger.warning(f"Redis load attempt {attempt + 1} failed for chat_id: {chat_id}: {e}")
        if attempt < REDIS_MAX_RETRIES:
            await asyncio.sleep(REDIS_RETRY_DELAY)
    logger.error(f"Redis load failed after {REDIS_MAX_RETRIES + 1} attempts for chat_id: {chat_id}. Last: {last_exc}")
    return None


# ============================================================================
# AUTO-DISCONNECT CONFIGURATION
# ============================================================================

SILENCE_TIMEOUT_S = float(os.getenv("SILENCE_TIMEOUT_S", "60.0"))
GOODBYE_HANGUP_DELAY_S = float(os.getenv("GOODBYE_HANGUP_DELAY_S", "3.0"))

GOODBYE_PHRASES = (
    "goodbye", "good bye", "bye bye", "bye-bye",
    "see you", "see ya", "talk later", "talk to you later",
    "take care", "have a good", "have a great", "have a nice",
    "no thanks", "that's all", "that is all", "i'm done", "i am done",
    "no more questions", "nothing else", "all good thanks", "all good thank you",
    "thanks bye", "thank you bye", "cheers bye",
)


def _is_goodbye(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in GOODBYE_PHRASES)


# ============================================================================
# SALES AGENT HANDOFF CONFIGURATION
# ============================================================================

# State machine values
HANDOFF_STATE_NONE = "none"
HANDOFF_STATE_COLLECTING_NAME = "collecting_name"
HANDOFF_STATE_COLLECTING_PHONE = "collecting_phone"
HANDOFF_STATE_COLLECTING_EMAIL = "collecting_email"
HANDOFF_STATE_PLAYING_HOLD = "playing_hold"
HANDOFF_STATE_DONE = "done"

# How long to play hold music before the confirmation message.
HOLD_MUSIC_DURATION_S = float(os.getenv("HOLD_MUSIC_DURATION_S", "5.0"))

# Optional custom hold music URL (publicly accessible MP3/OGG).
# Leave empty to use LiveKit's built-in OFFICE_HOURS clip.
HOLD_MUSIC_URL = os.getenv("HOLD_MUSIC_URL", "").strip()

# Phrases that indicate the user wants a human/sales agent.
SALES_AGENT_PHRASES = (
    "talk to", "speak to", "speak with", "connect me", "connect to",
    "transfer me", "transfer to", "sales agent", "sales team", "sales person",
    "human agent", "live agent", "real agent", "real person", "human support",
    "customer support", "contact sales", "reach sales", "get sales",
    "i want to talk", "i need to talk", "i'd like to talk",
    "i want to speak", "i need to speak",
    "can i speak", "can i talk", "can you connect",
    "agent please", "representative", "sales rep",
)


def _wants_sales_agent(text: str) -> bool:
    """Return True if the user is requesting a human/sales agent."""
    lower = text.lower()
    return any(phrase in lower for phrase in SALES_AGENT_PHRASES)


# ============================================================================
# ASSISTANT CLASS
# ============================================================================

class Assistant(Agent):
    """
    Voice Assistant Agent with optimized latency pipeline, auto-disconnect,
    and sales agent handoff with hold music.

    Sales handoff state machine:
        NONE -> COLLECTING_NAME -> COLLECTING_PHONE -> COLLECTING_EMAIL
             -> PLAYING_HOLD -> DONE

    Hold music flow (_play_hold_and_confirm):
        1. Bot says "I'm sending your details, please hold..."
        2. BackgroundAudioPlayer plays OFFICE_HOURS (or custom URL).
        3. After HOLD_MUSIC_DURATION_S seconds, music stops.
        4. Bot says "I've forwarded your details. An agent will contact you soon."
        5. Bot asks if there is anything else it can help with.
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
        self._pending_rag_context: Optional[str] = None

        # Auto-disconnect
        self._silence_timer_task: Optional[asyncio.Task] = None
        self._session_ref: Optional[AgentSession] = None
        self._room_ref = None

        # Sales handoff state
        self._handoff_state: str = HANDOFF_STATE_NONE
        self._handoff_data: Dict[str, str] = {"name": "", "phone": "", "email": ""}

        super().__init__(
            instructions=instructions or "You are a helpful voice assistant.",
            chat_ctx=chat_ctx,
        )

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _add_background_task(self, task: asyncio.Task) -> None:
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _apply_voice_settings_for_tone(self, tone: str) -> None:
        if elevenlabs_tts is None or not tone:
            return
        try:
            elevenlabs_tts.update_options(voice_settings=get_voice_settings_for_tone(tone))
            self.current_tone = tone
        except Exception as e:
            logger.debug(f"Could not update TTS voice settings: {e}")

    # -------------------------------------------------------------------------
    # TTS node — strip leading bracket tag for non-v3 models
    # -------------------------------------------------------------------------

    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        if IS_ELEVEN_V3:
            return Agent.default.tts_node(self, text, model_settings)

        async def strip_first_chunk_tag():
            stripped_once = False
            async for chunk in text:
                if not stripped_once and chunk:
                    new_chunk = _LEADING_TAG_RE.sub("", chunk, count=1)
                    stripped_once = True
                    if new_chunk:
                        yield new_chunk
                else:
                    yield chunk

        return Agent.default.tts_node(self, strip_first_chunk_tag(), model_settings)

    # -------------------------------------------------------------------------
    # Auto-disconnect
    # -------------------------------------------------------------------------

    def _reset_silence_timer(self) -> None:
        """(Re)start the silence watchdog. Call after every user or assistant turn."""
        if self._silence_timer_task and not self._silence_timer_task.done():
            self._silence_timer_task.cancel()

        async def _watchdog():
            try:
                await asyncio.sleep(SILENCE_TIMEOUT_S)
                logger.info(f"Silence timeout {SILENCE_TIMEOUT_S}s reached for {self.cid}. Disconnecting.")
                self.session_logger.log_event("AUTO_DISCONNECT", f"Silence {SILENCE_TIMEOUT_S}s",
                                              {"reason": "silence_timeout", "chat_id": self.cid})
                await self._do_disconnect(farewell=None)
            except asyncio.CancelledError:
                pass

        task = asyncio.create_task(_watchdog())
        self._silence_timer_task = task
        self._add_background_task(task)

    async def _do_disconnect(self, farewell: Optional[str] = None) -> None:
        """Optionally speak a farewell, then disconnect. Safe to call multiple times."""
        if self._silence_timer_task and not self._silence_timer_task.done():
            self._silence_timer_task.cancel()
        try:
            if farewell and self._session_ref is not None:
                await self._session_ref.generate_reply(instructions=farewell)
                await asyncio.sleep(GOODBYE_HANGUP_DELAY_S)
        except Exception as e:
            logger.debug(f"Farewell generation error: {e}")
        try:
            if self._room_ref is not None:
                await self._room_ref.disconnect()
                logger.info(f"Room disconnected for chat_id: {self.cid}")
        except Exception as e:
            logger.error(f"Error disconnecting room: {e}")

    # -------------------------------------------------------------------------
    # OPT-HIGH-1: Non-blocking tone classification
    # -------------------------------------------------------------------------

    def _start_tone_classification(self, user_text: str) -> None:
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
                self.session_logger.log_event("TONE_SELECTED", f"Tone: {tone}",
                                              {"tone": tone, "previous": self.current_tone})
            except Exception as e:
                logger.debug(f"Tone classification failed: {e}")

        self._add_background_task(asyncio.create_task(_classify_and_apply()))

    # -------------------------------------------------------------------------
    # OPT-HIGH-2: Non-blocking RAG fetch
    # -------------------------------------------------------------------------

    def _start_rag_fetch(self, user_text: str, turn_ctx: ChatContext) -> None:
        rag_start = time.time()

        async def _fetch_and_inject():
            try:
                rag_content = await asyncio.wait_for(
                    asyncio.to_thread(fetch_context, user_text, self.con, self.bot_id, self.Domain),
                    timeout=RAG_FETCH_TIMEOUT
                )
                duration = time.time() - rag_start
                self.session_logger.log_rag_fetch(user_text, len(rag_content) if rag_content else 0, duration)
                if rag_content:
                    try:
                        turn_ctx.add_message(
                            role="assistant",
                            content=f"Additional information relevant to the user's next message: {rag_content}"
                        )
                    except Exception:
                        pass
                    self._pending_rag_context = rag_content
                else:
                    self._pending_rag_context = ""
            except AsyncTimeoutError:
                logger.warning(f"RAG fetch timed out after {RAG_FETCH_TIMEOUT}s for {self.cid}")
                self.session_logger.log_event("RAG_FETCH_TIMEOUT", "RAG timed out", {"query": user_text})
                self._pending_rag_context = ""
            except Exception as e:
                logger.error(f"RAG fetch error: {e}", exc_info=True)
                self._pending_rag_context = ""

        self._add_background_task(asyncio.create_task(_fetch_and_inject()))

    # -------------------------------------------------------------------------
    # Sales handoff — hold music + confirmation
    # -------------------------------------------------------------------------

    async def _play_hold_and_confirm(self) -> None:
        """
        Full hold music sequence:
          1. Bot speaks "Sending details, please hold..."
          2. Background hold music plays (OFFICE_HOURS or custom URL).
          3. After HOLD_MUSIC_DURATION_S seconds, music stops.
          4. Bot speaks confirmation that agent will follow up.
          5. State set to DONE.
        """
        if self._session_ref is None or self._room_ref is None:
            logger.warning("Cannot play hold music — session/room refs not wired.")
            return

        # Build a short human-readable summary of collected details.
        parts = []
        if self._handoff_data.get("name"):
            parts.append(f"name {self._handoff_data['name']}")
        if self._handoff_data.get("phone"):
            parts.append(f"phone {self._handoff_data['phone']}")
        if self._handoff_data.get("email"):
            parts.append(f"email {self._handoff_data['email']}")
        summary_str = ", ".join(parts) if parts else "your details"

        try:
            # Step 1 — announce hold.
            await self._session_ref.generate_reply(
                instructions=(
                    "Tell the caller warmly and briefly that you are now forwarding their details "
                    "to a sales agent and they should please hold for just a moment. "
                    "Keep it to one natural, conversational sentence."
                )
            )

            # Step 2 — start background hold music.
            audio_player = BackgroundAudioPlayer(room=self._room_ref)
            try:
                if HOLD_MUSIC_URL:
                    # Custom publicly-accessible MP3 or OGG.
                    await audio_player.start(
                        audio=AudioConfig(url=HOLD_MUSIC_URL, volume=0.4)
                    )
                    logger.info(f"Hold music started from URL: {HOLD_MUSIC_URL}")
                else:
                    # LiveKit built-in clip — no external dependency needed.
                    await audio_player.start(
                        audio=AudioConfig(builtin=BuiltinAudioClip.OFFICE_HOURS, volume=0.4)
                    )
                    logger.info("Hold music started (BuiltinAudioClip.OFFICE_HOURS)")

                # Step 3 — wait for hold duration.
                await asyncio.sleep(HOLD_MUSIC_DURATION_S)

            finally:
                try:
                    await audio_player.stop()
                    logger.info("Hold music stopped")
                except Exception as e:
                    logger.debug(f"Error stopping audio player: {e}")

            # Step 4 — speak confirmation.
            await self._session_ref.generate_reply(
                instructions=(
                    f"Tell the caller that you have successfully forwarded their details "
                    f"({summary_str}) to the sales agent and that the agent will be in touch "
                    f"with them very soon. Be warm, brief, and reassuring. "
                    f"Then ask if there is anything else you can help them with today."
                )
            )

            self._handoff_state = HANDOFF_STATE_DONE
            self.session_logger.log_event(
                "HANDOFF_COMPLETE", "Sales handoff sequence finished",
                {"handoff_data": self._handoff_data, "chat_id": self.cid}
            )
            logger.info(f"Sales handoff complete for {self.cid}. Data: {self._handoff_data}")

        except Exception as e:
            logger.error(f"Error in _play_hold_and_confirm: {e}", exc_info=True)
            self._handoff_state = HANDOFF_STATE_DONE  # Don't loop on error

    # -------------------------------------------------------------------------
    # DRY helper: fire API post + Redis save as background tasks
    # -------------------------------------------------------------------------

    def _fire_post_and_save(self, message: str) -> None:
        """Post message to external API and save conversation to Redis — both fire-and-forget."""
        v_id = self.variablesForChat.get("VisitorId", "")
        chat_id = self.variablesForChat.get("ChatId", "")
        api_t = time.time()

        async def _post():
            success = await safe_post_message(
                chat_id=chat_id, message=message,
                user_id="0", manager_id="0",
                visitor_id=self.variablesForChat.get("VisitorId", ""),
                website_id=self.variablesForChat.get("WebsiteId", ""),
                nick_name=self.variablesForChat.get("NickName", ""),
                operator_name=self.variablesForChat.get("visitorName", f"Visitor{v_id}")
            )
            self.session_logger.log_api_call("post_message_to_conversation", success, time.time() - api_t)

        redis_t = time.time()

        async def _save():
            success = await safe_save_to_redis(self.cid, self.con)
            self.session_logger.log_redis_operation("SAVE_USER", success, time.time() - redis_t)
            if not success:
                logger.warning(f"Background Redis save failed for chat_id: {self.cid}")

        self._add_background_task(asyncio.create_task(_post()))
        self._add_background_task(asyncio.create_task(_save()))

    # -------------------------------------------------------------------------
    # Main turn hook
    # -------------------------------------------------------------------------

    async def on_user_turn_completed(
            self, turn_ctx: ChatContext, new_message: ChatMessage,
    ) -> None:
        """
        Called when the user finishes speaking.

        Intercept order (highest to lowest priority):
          1. Goodbye detection  -> farewell + disconnect.
          2. Handoff state machine:
               NONE + sales intent  -> ask for name.
               COLLECTING_NAME      -> save name, ask for phone.
               COLLECTING_PHONE     -> save phone, ask for email.
               COLLECTING_EMAIL     -> save email, trigger hold music sequence.
          3. Normal pipeline: RAG + tone classification + LLM.

        All I/O (Redis, API, RAG, tone) is fire-and-forget — the LLM starts
        immediately after this method returns.
        """
        try:
            q = new_message.text_content
            if not q:
                logger.warning("Empty message received, skipping")
                return

            try:
                q = validate_input(q)
            except ValueError as e:
                logger.error(f"Invalid message input: {e}")
                return

            self.session_logger.log_user_message(q)
            self._reset_silence_timer()

            # ------------------------------------------------------------------
            # 1. Goodbye detection (works at any handoff stage)
            # ------------------------------------------------------------------
            if _is_goodbye(q):
                logger.info(f"Goodbye detected for {self.cid}.")
                self.session_logger.log_event("AUTO_DISCONNECT", "Goodbye phrase",
                                              {"reason": "goodbye_phrase", "msg": q})
                self.con.conversation_history.append({"role": "user", "content": q})
                self._add_background_task(
                    asyncio.create_task(self._do_disconnect(
                        farewell="Say a warm, brief goodbye and thank the caller for calling."
                    ))
                )
                return

            # ------------------------------------------------------------------
            # 2. Sales agent handoff state machine
            # ------------------------------------------------------------------

            # Initial trigger: user asks for a human/sales agent.
            if self._handoff_state == HANDOFF_STATE_NONE and _wants_sales_agent(q):
                logger.info(f"Sales agent request detected for {self.cid}. Starting handoff.")
                self.session_logger.log_event("HANDOFF_STARTED", "User requested sales agent",
                                              {"msg": q, "chat_id": self.cid})
                self._handoff_state = HANDOFF_STATE_COLLECTING_NAME
                self.con.conversation_history.append({"role": "user", "content": q})
                # Inject a targeted system instruction so the LLM asks only for the name.
                turn_ctx.add_message(
                    role="system",
                    content=(
                        "The user wants to be connected to a sales agent. "
                        "Your job now is to collect their contact details so the agent can reach them. "
                        "Ask ONLY for their full name in one warm, natural sentence."
                    )
                )
                self._fire_post_and_save(q)
                return

            # Collecting name.
            if self._handoff_state == HANDOFF_STATE_COLLECTING_NAME:
                self._handoff_data["name"] = q.strip()
                self._handoff_state = HANDOFF_STATE_COLLECTING_PHONE
                self.con.conversation_history.append({"role": "user", "content": q})
                logger.info(f"Handoff: name='{q.strip()}' collected for {self.cid}")
                turn_ctx.add_message(
                    role="system",
                    content=(
                        f"The caller's name is '{q.strip()}'. "
                        "Thank them briefly, then ask ONLY for their best phone number."
                    )
                )
                self._fire_post_and_save(q)
                return

            # Collecting phone.
            if self._handoff_state == HANDOFF_STATE_COLLECTING_PHONE:
                self._handoff_data["phone"] = q.strip()
                self._handoff_state = HANDOFF_STATE_COLLECTING_EMAIL
                self.con.conversation_history.append({"role": "user", "content": q})
                logger.info(f"Handoff: phone='{q.strip()}' collected for {self.cid}")
                turn_ctx.add_message(
                    role="system",
                    content=(
                        f"The caller's phone number is '{q.strip()}'. "
                        "Confirm it briefly, then ask ONLY for their email address."
                    )
                )
                self._fire_post_and_save(q)
                return

            # Collecting email — all details in hand, launch hold sequence.
            if self._handoff_state == HANDOFF_STATE_COLLECTING_EMAIL:
                self._handoff_data["email"] = q.strip()
                self._handoff_state = HANDOFF_STATE_PLAYING_HOLD
                self.con.conversation_history.append({"role": "user", "content": q})
                logger.info(f"Handoff: email='{q.strip()}' collected for {self.cid}. Triggering hold.")
                self.session_logger.log_event(
                    "HANDOFF_DATA_COLLECTED", "All contact details collected, starting hold",
                    {"handoff_data": self._handoff_data, "chat_id": self.cid}
                )
                self._fire_post_and_save(q)
                # Launch hold music + confirmation as a background task.
                # Return immediately — pipeline is not blocked.
                self._add_background_task(asyncio.create_task(self._play_hold_and_confirm()))
                return

            # ------------------------------------------------------------------
            # 3. Normal turn: inject previous RAG context, fire tone + RAG, LLM.
            # ------------------------------------------------------------------

            # Inject RAG context fetched during the previous turn (zero latency).
            if self._pending_rag_context:
                try:
                    turn_ctx.add_message(
                        role="assistant",
                        content=f"Additional information relevant to the user's next message: {self._pending_rag_context}"
                    )
                    logger.debug("Injected cached RAG context from previous turn")
                except Exception as e:
                    logger.debug(f"Could not inject cached RAG context: {e}")
                self._pending_rag_context = None

            # Fire tone classification in background (OPT-HIGH-1).
            self._start_tone_classification(q)

            # Update in-memory history.
            self.con.conversation_history.append({"role": "user", "content": q})
            logger.info(f"User message added to history for chat_id: {self.cid}")

            # Post + save (fire-and-forget).
            self._fire_post_and_save(q)

            # Fire RAG fetch in background (OPT-HIGH-2).
            self._start_rag_fetch(q, turn_ctx)

            logger.debug("on_user_turn_completed: returning control — LLM may start now")

        except Exception as e:
            logger.error(f"Error in on_user_turn_completed: {e}", exc_info=True)


# ============================================================================
# ENTRYPOINT FUNCTION
# ============================================================================

from livekit import api


async def entrypoint(ctx: agents.JobContext):
    """Main entry point for the LiveKit voice assistant agent."""
    try:
        print("🌟 [ENTRYPOINT INIT] Starting voice assistant entrypoint")
        await ctx.connect()

        session_id = None
        bot_id = '1344'
        domain = "testing.webgreeter.com/zem/hulk"
        chat_id = None
        variablesForChat = {}

        # ====================================================================
        # STEP 1: EXTRACT METADATA FROM PARTICIPANT
        # ====================================================================
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
                logger.info(f"Extracted — chat_id: {chat_id}, bot_id: {bot_id}")
                session_id = chat_id

        except (ValueError, KeyError, json.JSONDecodeError) as e:
            logger.warning(f"Metadata extraction failed: {e}, using defaults")
            chat_id = 'mynewchatID5'
            session_id = chat_id
        except Exception as e:
            logger.error(f"Unexpected metadata error: {e}", exc_info=True)
            chat_id = 'mynewchatID5'
            session_id = chat_id

        if not chat_id:
            chat_id = 'mynewchatID5'
            session_id = chat_id
            for k, v in [("ChatId", chat_id), ("VisitorId", ""), ("WebsiteId", ""),
                         ("NickName", ""), ("UserId", ""), ("ManagerId", ""), ("visitorName", "Guest")]:
                variablesForChat.setdefault(k, v)

        if session_id:
            session_logger.start_session(session_id, {
                'chat_id': chat_id, 'bot_id': bot_id, 'domain': domain,
                'room_name': ctx.room.name if hasattr(ctx, 'room') and ctx.room else 'unknown',
                **variablesForChat
            })

        # ====================================================================
        # STEP 2: LOAD OR INITIALIZE CONVERSATION
        # ====================================================================
        logger.info(f"Loading conversation for chat_id: {chat_id}")
        redis_t = time.time()
        con = await safe_load_from_redis(chat_id)
        session_logger.log_redis_operation("LOAD", con is not None, time.time() - redis_t)

        if con is not None:
            logger.info(f"Loaded conversation — {len(con.conversation_history)} messages")
        else:
            logger.info("No existing conversation in Redis, creating new one")
            Agent_key = f"{bot_id}_{domain}"
            con = Conversation(bot_id, Agent_key)
            try:
                bot_tree = await asyncio.to_thread(fetch_tree, con, bot_id)
                if bot_tree:
                    con.tree = bot_tree
            except Exception as e:
                logger.error(f"Error fetching tree: {e}", exc_info=True)
            try:
                BotPrompt = await asyncio.to_thread(fetch_Instructions, con, bot_id)
                if BotPrompt:
                    con.Instructions = BotPrompt
            except Exception as e:
                logger.error(f"Error fetching instructions: {e}", exc_info=True)
            await safe_save_to_redis(chat_id, con)

        # ====================================================================
        # STEP 3: BUILD SYSTEM PROMPT
        # ====================================================================
        prompt = build_voicebot_prompt(con.Instructions or "", con.tree or "")
        logger.info(f"Prompt built (len={len(prompt)}, model={ELEVENLABS_MODEL}, v3={IS_ELEVEN_V3})")

        if not con.conversation_history or con.conversation_history[0].get('role') != 'system':
            con.conversation_history.insert(0, {'role': 'system', 'content': prompt})
        else:
            con.conversation_history[0]['content'] = prompt

        # ====================================================================
        # STEP 4: INITIALIZE CHAT CONTEXT
        # ====================================================================
        chat_ctx = ChatContext()
        for m in con.conversation_history:
            if m.get("content"):
                chat_ctx.add_message(role=m.get("role", "user"), content=m["content"])

        # ====================================================================
        # STEP 5: CONFIGURE AI SESSION
        # ====================================================================
        try:
            stt_instance = assemblyai.STT(
                api_key=ASSEMBLYAI_API_KEY,
                model="u3-rt-pro",
                min_turn_silence=200,   # raised from 100ms — fewer false positives
                max_turn_silence=500,   # lowered from 1000ms — faster endpointing
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
            raise RuntimeError("ElevenLabs TTS initialization failed.")

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
        assistant._previous_history_length = len(con.conversation_history)

        # ====================================================================
        # STEP 7: SET UP EVENT HANDLERS
        # ====================================================================
        @session.on("conversation_item_added")
        def on_item(event: ConversationItemAddedEvent):
            """Called when the assistant generates a response. All I/O is fire-and-forget."""
            try:
                msg = event.item
                if not hasattr(msg, "role") or msg.role != "assistant":
                    return
                if not msg.text_content:
                    return

                try:
                    validated_content = validate_input(msg.text_content)
                except ValueError:
                    return

                display_content = (
                    strip_leading_tag(validated_content) if IS_ELEVEN_V3 else validated_content
                ) or validated_content

                # Update in-memory history.
                assistant.con.conversation_history.append({"role": "assistant", "content": display_content})
                assistant.session_logger.log_assistant_response(display_content)

                # Estimate LLM tokens.
                try:
                    input_text = ' '.join(m.get('content', '') for m in assistant.con.conversation_history[:-1])
                    assistant.session_logger.log_llm_usage(
                        len(input_text) // 4, len(display_content) // 4, model='gpt-4o-mini'
                    )
                except Exception:
                    pass

                # Post to API — fire-and-forget.
                api_t = time.time()

                async def post_assistant():
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
                        "post_message_to_conversation", success, time.time() - api_t
                    )

                assistant._add_background_task(asyncio.create_task(post_assistant()))

                # Save to Redis — fire-and-forget.
                redis_t = time.time()

                async def save_assistant():
                    success = await safe_save_to_redis(assistant.cid, assistant.con)
                    assistant.session_logger.log_redis_operation(
                        "SAVE_ASSISTANT", success, time.time() - redis_t
                    )

                assistant._add_background_task(asyncio.create_task(save_assistant()))

                # Reset silence watchdog.
                assistant._reset_silence_timer()

            except Exception as e:
                logger.error(f"Error in conversation_item_added: {e}", exc_info=True)

        # ====================================================================
        # STEP 8: START SESSION AND GENERATE GREETING
        # ====================================================================
        room_input_options = RoomInputOptions()
        try:
            if os.getenv('LIVEKIT_USE_CLOUD_NC', 'false').lower() == 'true':
                room_input_options = RoomInputOptions(noise_cancellation=noise_cancellation.BVC())
                logger.info("BVC noise cancellation enabled")
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
            # Wire session + room refs for auto-disconnect and hold music.
            assistant._session_ref = session
            assistant._room_ref = ctx.room
            logger.info("Agent session started successfully")
            session_logger.log_event("SESSION_STARTED", "Agent session started")
        except Exception as e:
            logger.error(f"Failed to start agent session: {e}", exc_info=True)
            raise

        try:
            await session.generate_reply(instructions="Greet the user and offer your assistance.")
            logger.info("Initial greeting generated")
            # Start silence watchdog after greeting.
            assistant._reset_silence_timer()
        except Exception as e:
            logger.error(f"Failed to generate greeting: {e}")

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