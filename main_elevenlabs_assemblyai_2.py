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
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Environment configuration
from dotenv import load_dotenv

# Audio processing library (used by LiveKit, needs monkey-patching)
import sounddevice as sd

# LiveKit core components
from livekit.agents import AgentSession, Agent, RoomInputOptions, function_tool, RunContext, BackgroundAudioPlayer, AudioConfig, BuiltinAudioClip, RoomInputOptions, TurnHandlingOptions
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, AutoSubscribe
from livekit.agents.cli import run_app
from livekit.agents import ChatContext, ChatMessage
from livekit.agents import ConversationItemAddedEvent
from livekit.agents import ModelSettings

# LiveKit plugins for STT, TTS, VAD, and noise cancellation
from livekit.plugins import assemblyai, elevenlabs, silero, noise_cancellation
# from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins.openai import LLM

# Direct OpenAI async client for lightweight tone classification calls.
# (The livekit plugin LLM is used for the main streaming pipeline; this client
# is only used out-of-band so we can switch voice settings per tone.)
from openai import AsyncOpenAI

# Redis utilities for conversation persistence
from redis_utils_Wg import load_conversation_from_redis, redis_client, save_conversation_to_redis, Conversation

# Utility functions for fetching bot configuration and RAG context
from utils_livekit import fetch_tree, fetch_Instructions, fetch_context

# External API integration for message posting
from save_conversation_signalr import post_message_to_conversation

# Standard library imports
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

# SignalR integration (currently commented out - alternative message delivery method)
# from signalr import hub_connection, build_message_packet
# from signalR import RawSignalRManager

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
# Custom filter to sanitize log records and prevent pickle errors
class SanitizeLogRecordFilter(logging.Filter):
    """
    Filter to sanitize log records by removing unpicklable objects.
    This prevents errors when LiveKit tries to pickle log records for IPC.
    """
    
    def _is_picklable(self, obj):
        """Check if an object can be pickled."""
        try:
            import pickle
            pickle.dumps(obj)
            return True
        except (TypeError, AttributeError, pickle.PicklingError):
            return False
    
    def _sanitize_value(self, value):
        """Recursively sanitize a value to make it picklable."""
        if value is None:
            return None
        
        # Check if already picklable
        if self._is_picklable(value):
            return value
        
        # Handle dict-like objects (including CIMultiDictProxy)
        if isinstance(value, dict) or (hasattr(value, 'items') and hasattr(value, 'keys')):
            try:
                return {str(k): self._sanitize_value(v) for k, v in value.items()}
            except:
                return f"<{type(value).__name__} object>"
        
        # Handle lists/tuples
        if isinstance(value, (list, tuple)):
            return [self._sanitize_value(item) for item in value]
        
        # Handle error objects (HTTPStatusError, etc.)
        if hasattr(value, '__class__'):
            error_type = type(value).__name__
            if 'Error' in error_type or 'Exception' in error_type:
                sanitized = {
                    'type': error_type,
                    'message': str(value)
                }
                # Extract response info if it's an HTTP error
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
        
        # Try to convert to string
        try:
            return str(value)
        except:
            return f"<{type(value).__name__} object>"
    
    def filter(self, record):
        """
        Sanitize log record by cleaning up fields that might contain unpicklable objects.
        Special handling for exc_info to preserve tuple structure.
        
        Args:
            record: Log record to filter
            
        Returns:
            True (always allow the record, but sanitize it)
        """
        # Special handling for exc_info - it must remain a tuple (type, value, traceback)
        # If it contains unpicklable objects, clear it to prevent pickle errors
        # The log message will still contain the error information
        if hasattr(record, 'exc_info') and record.exc_info is not None:
            # Validate that exc_info is a proper tuple
            if isinstance(record.exc_info, tuple) and len(record.exc_info) == 3:
                exc_type, exc_value, exc_tb = record.exc_info
                
                # Validate that all components are the correct types
                if not isinstance(exc_tb, (type(None), type)) and not hasattr(exc_tb, 'tb_frame'):
                    # Traceback is corrupted (not a traceback object)
                    record.exc_info = None
                else:
                    # Check if the exception value contains unpicklable objects
                    try:
                        import pickle
                        # Try to pickle the whole exc_info tuple
                        pickle.dumps(record.exc_info)
                        # If successful, leave it as-is
                    except (TypeError, AttributeError, pickle.PicklingError):
                        # If unpicklable, clear exc_info to prevent errors
                        # The error message is already in record.getMessage(), so we don't lose information
                        record.exc_info = None
            else:
                # If exc_info is not a proper tuple, clear it to prevent errors
                record.exc_info = None
        
        # Sanitize other fields that might contain unpicklable objects
        # Skip exc_info as we handled it above
        fields_to_check = ['error', 'response', 'request', 'extra']
        
        for field in fields_to_check:
            if hasattr(record, field):
                value = getattr(record, field)
                if value is not None:
                    if not self._is_picklable(value):
                        sanitized = self._sanitize_value(value)
                        setattr(record, field, sanitized)
        
        # Also sanitize the __dict__ if it contains unpicklable values
        # But skip exc_info as we already handled it
        if hasattr(record, '__dict__'):
            for key, value in list(record.__dict__.items()):
                if key == 'exc_info':
                    continue  # Already handled above
                if value is not None and not self._is_picklable(value):
                    sanitized = self._sanitize_value(value)
                    record.__dict__[key] = sanitized
        
        return True

# Configure logging framework for proper log levels and formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Add sanitization filter to root logger to catch all logs
# This must be done before LiveKit initializes its handlers
root_logger = logging.getLogger()
sanitize_filter = SanitizeLogRecordFilter()

# Remove any existing filters and add ours first
# This ensures our filter runs before LiveKit's handlers process records
for handler in root_logger.handlers[:]:
    handler.addFilter(sanitize_filter)

# Also add to root logger itself
root_logger.addFilter(sanitize_filter)

# Add filter to LiveKit's logger specifically if it exists
try:
    livekit_logger = logging.getLogger('livekit.agents')
    livekit_logger.addFilter(sanitize_filter)
    for handler in livekit_logger.handlers[:]:
        handler.addFilter(sanitize_filter)
except:
    pass

logger = logging.getLogger(__name__)

# Custom exception handler to sanitize exceptions before logging
def sanitize_exception_handler(exc_type, exc_value, exc_traceback):
    """
    Custom exception handler that sanitizes exceptions before logging.
    This prevents unpicklable objects from being included in log records.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    # Sanitize the exception value if it contains unpicklable objects
    sanitized_exc_value = exc_value
    if hasattr(exc_value, 'response'):
        try:
            # Check if response contains unpicklable objects
            import pickle
            pickle.dumps(exc_value.response)
        except:
            # Create a sanitized version
            sanitized_exc_value = type(exc_value)(
                str(exc_value),
                response=None  # Remove unpicklable response
            )
            sanitized_exc_value.__cause__ = exc_value.__cause__
            sanitized_exc_value.__context__ = exc_value.__context__
    
    # Use standard logging
    logger.error(
        "Uncaught exception",
        exc_info=(exc_type, sanitized_exc_value, exc_traceback)
    )

# Set custom exception handler (but only if not in a subprocess)
# LiveKit uses multiprocessing, so we need to be careful
if __name__ != '__mp_main__':
    sys.excepthook = sanitize_exception_handler

# Keep print for backward compatibility but prefer logger
print = functools.partial(print, flush=True)

# ============================================================================
# SESSION LOGGER FOR FILE LOGGING
# ============================================================================
# Session logger that writes all session data to a text file with token tracking

class SessionLogger:
    """
    Logger for tracking complete session information including token usage.
    Writes to a single file with session separators for easy navigation.
    
    Log file format:
    - Each session is separated by a clear separator line
    - Session start includes metadata
    - All events are logged with timestamps
    - Token usage is tracked for STT, TTS, and LLM
    - Session end includes summary statistics
    
    Log file location: voice_assistant_sessions.log (in the working directory)
    """
    
    def __init__(self, log_file_path: str = "voice_assistant_sessions.log"):
        """
        Initialize session logger.
        
        Args:
            log_file_path: Path to the log file (default: voice_assistant_sessions.log)
        """
        self.log_file_path = os.getenv("SESSION_LOG_PATH", log_file_path)
        self.current_session_id = None
        self.session_start_time = None
        self.token_stats = {
            'stt_tokens': 0,  # Approximate tokens for STT (characters / 4)
            'tts_tokens': 0,  # Approximate tokens for TTS (characters / 4)
            'llm_input_tokens': 0,  # LLM input tokens
            'llm_output_tokens': 0,  # LLM output tokens
            'total_tokens': 0  # Total tokens used
        }
        self.session_events = []
        self._file_lock = None  # Will be created when needed (asyncio.Lock is not picklable)
        
    def start_session(self, session_id: str, metadata: Dict[str, Any] = None):
        """
        Start a new session logging.
        
        Args:
            session_id: Unique session identifier (chat_id)
            metadata: Session metadata (bot_id, domain, visitor info, etc.)
        """
        self.current_session_id = session_id
        self.session_start_time = datetime.now()
        self.token_stats = {
            'stt_tokens': 0,
            'tts_tokens': 0,
            'llm_input_tokens': 0,
            'llm_output_tokens': 0,
            'total_tokens': 0
        }
        self.session_events = []  # Store only serializable data
        self._file_lock = None  # Reset lock (will be created when needed)
        
        # Serialize metadata to ensure it's safe
        safe_metadata = self._serialize_data(metadata) if metadata else {}
        
        # Write session separator
        separator = "\n" + "=" * 100 + "\n"
        session_header = f"""
SESSION START
=============
Session ID: {session_id}
Start Time: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        if safe_metadata:
            session_header += "Metadata:\n"
            for key, value in safe_metadata.items():
                session_header += f"  {key}: {value}\n"
        
        session_header += separator
        
        asyncio.create_task(self._write_to_file(separator + session_header))
        self.log_event("SESSION_START", f"Session {session_id} started", safe_metadata)
    
    def _serialize_data(self, data: Any) -> Any:
        """
        Safely serialize data, handling non-serializable objects.
        Specifically handles CIMultiDictProxy and other aiohttp objects.
        
        Args:
            data: Data to serialize
            
        Returns:
            Serializable version of the data
        """
        if data is None:
            return None
        
        if isinstance(data, (str, int, float, bool)):
            return data
        
        # Handle dict-like objects that might not be JSON serializable (e.g., CIMultiDictProxy)
        if isinstance(data, dict) or (hasattr(data, 'items') and hasattr(data, 'keys')):
            try:
                return {str(k): self._serialize_data(v) for k, v in data.items()}
            except Exception:
                # If items() fails, try to convert to dict first
                try:
                    return {str(k): self._serialize_data(v) for k, v in dict(data).items()}
                except Exception:
                    return f"<{type(data).__name__} object>"
        
        if isinstance(data, (list, tuple)):
            return [self._serialize_data(item) for item in data]
        
        # Handle common non-serializable types (aiohttp objects, etc.)
        type_name = type(data).__name__
        if 'MultiDict' in type_name or 'CIMultiDict' in type_name:
            # Handle aiohttp MultiDict objects
            try:
                return dict(data)
            except Exception:
                return f"<{type_name} object>"
        
        # Handle objects with __dict__
        if hasattr(data, '__dict__'):
            try:
                return {k: self._serialize_data(v) for k, v in data.__dict__.items()}
            except Exception:
                try:
                    return str(data)
                except:
                    return f"<{type_name} object>"
        
        # Try to convert to string as last resort
        try:
            return str(data)
        except Exception:
            return f"<{type_name} object>"
    
    def log_event(self, event_type: str, message: str, data: Dict[str, Any] = None):
        """
        Log an event during the session.
        
        Args:
            event_type: Type of event (e.g., "USER_MESSAGE", "ASSISTANT_RESPONSE", "API_CALL")
            message: Event message
            data: Additional event data
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Serialize data to ensure it's JSON-safe
        safe_data = self._serialize_data(data) if data else {}
        
        event = {
            'timestamp': timestamp,
            'type': event_type,
            'message': message,
            'data': safe_data
        }
        self.session_events.append(event)
        
        log_entry = f"[{timestamp}] [{event_type}] {message}\n"
        if safe_data:
            try:
                log_entry += f"  Data: {json.dumps(safe_data, indent=2, ensure_ascii=False)}\n"
            except Exception as e:
                # Fallback if JSON serialization still fails
                log_entry += f"  Data: {str(safe_data)}\n"
                logger.debug(f"Failed to serialize event data: {e}")
        log_entry += "\n"
        
        asyncio.create_task(self._write_to_file(log_entry))
    
    def log_user_message(self, message: str, stt_duration: float = None):
        """
        Log a user message and track STT tokens.
        
        Args:
            message: Transcribed user message
            stt_duration: Time taken for STT processing (seconds)
        """
        # Approximate STT tokens (characters / 4 is a rough estimate)
        stt_tokens = len(message) // 4
        self.token_stats['stt_tokens'] += stt_tokens
        self.token_stats['total_tokens'] += stt_tokens
        
        data = {
            'message': message,
            'message_length': len(message),
            'stt_tokens': stt_tokens,
            'stt_duration': stt_duration
        }
        self.log_event("USER_MESSAGE", f"User said: {message[:100]}...", data)
    
    def log_assistant_response(self, message: str, tts_duration: float = None):
        """
        Log an assistant response and track TTS tokens.
        
        Args:
            message: Assistant response text
            tts_duration: Time taken for TTS processing (seconds)
        """
        # Approximate TTS tokens (characters / 4 is a rough estimate)
        tts_tokens = len(message) // 4
        self.token_stats['tts_tokens'] += tts_tokens
        self.token_stats['total_tokens'] += tts_tokens
        
        data = {
            'message': message,
            'message_length': len(message),
            'tts_tokens': tts_tokens,
            'tts_duration': tts_duration
        }
        self.log_event("ASSISTANT_RESPONSE", f"Assistant said: {message[:100]}...", data)
    
    def log_llm_usage(self, input_tokens: int, output_tokens: int, model: str = None, cost: float = None):
        """
        Log LLM token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: LLM model name
            cost: Estimated cost (if available)
        """
        self.token_stats['llm_input_tokens'] += input_tokens
        self.token_stats['llm_output_tokens'] += output_tokens
        self.token_stats['total_tokens'] += (input_tokens + output_tokens)
        
        data = {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_llm_tokens': input_tokens + output_tokens,
            'model': model,
            'estimated_cost': cost
        }
        self.log_event("LLM_USAGE", f"LLM used {input_tokens} input + {output_tokens} output tokens", data)
    
    def log_api_call(self, api_name: str, success: bool, duration: float = None, response: Any = None):
        """
        Log an API call.
        
        Args:
            api_name: Name of the API called
            success: Whether the call was successful
            duration: Time taken for the API call (seconds)
            response: API response data (will be safely serialized)
        """
        # Safely serialize response
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
        
        data = {
            'api_name': api_name,
            'success': success,
            'duration': duration,
            'response': safe_response
        }
        status = "SUCCESS" if success else "FAILED"
        self.log_event("API_CALL", f"{api_name} - {status}", data)
    
    def log_redis_operation(self, operation: str, success: bool, duration: float = None):
        """
        Log a Redis operation.
        
        Args:
            operation: Type of operation (GET, SET, etc.)
            success: Whether the operation was successful
            duration: Time taken (seconds)
        """
        data = {
            'operation': operation,
            'success': success,
            'duration': duration
        }
        status = "SUCCESS" if success else "FAILED"
        self.log_event("REDIS_OPERATION", f"Redis {operation} - {status}", data)
    
    def log_rag_fetch(self, query: str, context_length: int, duration: float = None):
        """
        Log RAG context fetch.
        
        Args:
            query: User query
            context_length: Length of retrieved context
            duration: Time taken (seconds)
        """
        data = {
            'query': query,
            'context_length': context_length,
            'duration': duration
        }
        self.log_event("RAG_FETCH", f"RAG context fetched: {context_length} chars", data)
    
    def end_session(self, reason: str = "NORMAL"):
        """
        End the current session and write summary.
        
        Args:
            reason: Reason for session end (NORMAL, ERROR, TIMEOUT, etc.)
        """
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
        
        # Reset session
        self.current_session_id = None
        self.session_start_time = None
    
    async def _write_to_file(self, content: str):
        """
        Thread-safe file writing.
        
        Args:
            content: Content to write to file
        """
        # Create lock if it doesn't exist (asyncio.Lock is not picklable, so create on demand)
        if self._file_lock is None:
            self._file_lock = asyncio.Lock()
        
        async with self._file_lock:
            try:
                with open(self.log_file_path, 'a', encoding='utf-8') as f:
                    f.write(content)
                    f.flush()  # Ensure immediate write
            except PermissionError:
                # Fallback to /tmp when current working directory is not writable.
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


# Global session logger instance
session_logger = SessionLogger()

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================
# Rate limiting: Maximum concurrent API calls
MAX_CONCURRENT_API_CALLS = 10
# Semaphore for rate limiting API calls (can be created outside async context)
api_semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)

# Timeouts (in seconds)
REDIS_TIMEOUT = 5.0
RAG_FETCH_TIMEOUT = 30.0
API_CALL_TIMEOUT = 10.0

# Input validation limits
MAX_MESSAGE_LENGTH = 10000  # Maximum message length in characters
MAX_CHAT_ID_LENGTH = 255  # Maximum chat ID length

# ============================================================================
# AUDIO CONFIGURATION
# ============================================================================
# Configure LiveKit audio settings before initialization
# These settings affect audio quality, processing complexity, and latency

os.environ['LIVEKIT_AUDIO_SAMPLE_RATE'] = '16000'  # 16kHz - standard for voice, good compatibility
os.environ['LIVEKIT_AUDIO_CHANNELS'] = '1'  # Mono channel - simpler processing, sufficient for voice
os.environ['LIVEKIT_AUDIO_LATENCY'] = 'medium'  # Balance between latency and stability

# Load environment variables from .env file (API keys, Redis config, etc.).
# Search the script's folder first, then walk up to the project root so the
# same .env works regardless of which cwd the worker is launched from.
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
# Validate that required API keys are present
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY') or os.getenv('ELEVEN_API_KEY')

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables. OpenAI LLM may fail.")
else:
    logger.info("OpenAI API key found in environment")

if not ASSEMBLYAI_API_KEY:
    logger.warning("ASSEMBLYAI_API_KEY not found in environment variables. AssemblyAI STT may fail.")
else:
    logger.info("AssemblyAI API key found in environment")

if not ELEVENLABS_API_KEY:
    logger.warning("ELEVENLABS_API_KEY (or ELEVEN_API_KEY) not found in environment variables. ElevenLabs TTS may fail.")
else:
    logger.info("ElevenLabs API key found in environment")

# ============================================================================
# AUDIO CALLBACK MONKEY-PATCH
# ============================================================================
# Fix for sounddevice RuntimeError when using noise cancellation (APM)
# This prevents crashes when the audio stream delay cannot be set properly
# The error is non-critical and can be safely ignored

# Store the original callback wrapper function
_orig_wrap = sd._wrap_callback


def _safe_wrap_callback(callback, data, frames, time, status):
    """
    Wrapper function that safely handles audio callback errors.
    
    Args:
        callback: The original callback function
        data: Audio data buffer
        frames: Number of frames
        time: Timestamp
        status: Status flags
        
    Returns:
        Integer return value required by CFFI (C Foreign Function Interface)
    """
    try:
        return _orig_wrap(callback, data, frames, time, status)
    except RuntimeError as e:
        # Ignore non-critical stream delay errors (common with noise cancellation)
        if "Failed to set stream delay" in str(e):
            print(f"[AudioCallback] Ignored set_stream_delay error: {e}")
            return 0  # Return integer so CFFI doesn't complain
        raise  # Re-raise other RuntimeErrors


# Replace the original callback wrapper with our safe version
sd._wrap_callback = _safe_wrap_callback

# ============================================================================
# TEXT-TO-SPEECH (TTS) CONFIGURATION
# ============================================================================
# Model-aware ElevenLabs TTS configuration. Mirrors the per-model voice tuning,
# tone presets, lead-capture rules, and v3 emotion tag selection that we use in
# the (V2) WebSocket project, but adapted to the LiveKit Agents pipeline:
#   - Voice settings are switched per turn via `tts.update_options(...)`.
#   - Emotion tags are produced by the LLM (prompt-driven) and rendered by v3.
#   - For non-v3 models the leading [tag] is stripped inside `tts_node` so
#     it is never spoken literally.

ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
# Optional separate voice id for v3 (some voices behave better there).
ELEVENLABS_V3_VOICE_ID = os.getenv("ELEVENLABS_V3_VOICE_ID", "").strip() or ELEVENLABS_VOICE_ID

IS_ELEVEN_V3 = ELEVENLABS_MODEL.startswith("eleven_v3")
ACTIVE_VOICE_ID = ELEVENLABS_V3_VOICE_ID if IS_ELEVEN_V3 else ELEVENLABS_VOICE_ID

# --- Per-model default voice settings ---------------------------------------
# Standard models (multilingual_v2, flash) - balanced/clear for phone calls.
ELEVENLABS_VOICE_SETTINGS_STANDARD = elevenlabs.VoiceSettings(
    stability=0.5,
    similarity_boost=0.75,
    style=0.15,
    speed=0.96,
    use_speaker_boost=True,
)

# v3 - softer top-end, lower similarity_boost to reduce harshness, no speaker
# boost (v3 doesn't expose it well). Slightly slower for warmer feel.
ELEVENLABS_VOICE_SETTINGS_V3 = elevenlabs.VoiceSettings(
    stability=0.7,
    similarity_boost=0.55,
    style=0.0,
    speed=0.96,
    use_speaker_boost=False,
)

DEFAULT_VOICE_SETTINGS = (
    ELEVENLABS_VOICE_SETTINGS_V3 if IS_ELEVEN_V3 else ELEVENLABS_VOICE_SETTINGS_STANDARD
)

# --- Tone -> voice settings presets (per model) -----------------------------
VOICE_TONE_PRESETS_STANDARD: Dict[str, "elevenlabs.VoiceSettings"] = {
    "neutral": ELEVENLABS_VOICE_SETTINGS_STANDARD,
    "empathic": elevenlabs.VoiceSettings(
        stability=0.58, similarity_boost=0.74, style=0.05, speed=0.92, use_speaker_boost=True
    ),
    "enthusiastic": elevenlabs.VoiceSettings(
        stability=0.42, similarity_boost=0.76, style=0.28, speed=1.02, use_speaker_boost=True
    ),
    "professional": elevenlabs.VoiceSettings(
        stability=0.67, similarity_boost=0.72, style=0.0, speed=0.95, use_speaker_boost=True
    ),
}

VOICE_TONE_PRESETS_V3: Dict[str, "elevenlabs.VoiceSettings"] = {
    "neutral": ELEVENLABS_VOICE_SETTINGS_V3,
    "empathic": elevenlabs.VoiceSettings(
        stability=0.72, similarity_boost=0.55, style=0.0, speed=0.93, use_speaker_boost=False
    ),
    "enthusiastic": elevenlabs.VoiceSettings(
        stability=0.6, similarity_boost=0.6, style=0.1, speed=1.0, use_speaker_boost=False
    ),
    "professional": elevenlabs.VoiceSettings(
        stability=0.78, similarity_boost=0.55, style=0.0, speed=0.95, use_speaker_boost=False
    ),
}

VOICE_TONE_PRESETS = VOICE_TONE_PRESETS_V3 if IS_ELEVEN_V3 else VOICE_TONE_PRESETS_STANDARD
VOICE_TONE_OPTIONS = tuple(VOICE_TONE_PRESETS.keys())
DEFAULT_VOICE_TONE = "neutral"


def get_voice_settings_for_tone(tone: str) -> "elevenlabs.VoiceSettings":
    """Return the VoiceSettings preset for the given tone (model-aware)."""
    selected = (tone or DEFAULT_VOICE_TONE).strip().lower()
    return VOICE_TONE_PRESETS.get(selected, VOICE_TONE_PRESETS[DEFAULT_VOICE_TONE])


# --- v3 emotion tags --------------------------------------------------------
V3_EMOTION_TAGS = (
    "cheerfully",
    "sympathetic",
    "excited",
    "questioning",
    "reassuring",
    "professional",
)
# Regex used to detect/strip a leading bracket tag in any model's text.
_LEADING_TAG_RE = re.compile(r"^\s*\[[^\]]+\]\s*")


def strip_leading_tag(text: str) -> str:
    """Remove a leading [bracket-tag] from a text reply, if present."""
    return _LEADING_TAG_RE.sub("", text or "", count=1).strip()


# --- Voice rules sections (mirrors backend/config.py in the V2 project) -----
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

Few-shot examples (caller turn -> your reply):
- Caller: "Hi, what do you guys do?" -> "[cheerfully] Hi there! We're a customer support platform with live chat, chatbots, and live call solutions."
- Caller: "Honestly, your bot has been frustrating today." -> "[sympathetic] I'm really sorry about that. Let's see how I can make it right for you."
- Caller: "Can you tell me about pricing?" -> "[professional] Sure - our managed live chat starts at three hundred ninety-nine dollars per month."
- Caller: "Awesome, sign me up!" -> "[excited] That's wonderful! Let's get you set up right away."
- Caller: "I'm not sure what you mean." -> "[questioning] Let me clarify - did you mean the live chat service or the chatbot?"
- Caller: "Will this take long to set up?" -> "[reassuring] Don't worry, it's usually quick - we walk you through every step."
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
  - Phone: read it back in groups (example: "five-five-five, two-three-four, eight-nine-zero-one - is that right?").
  - Email: read it as "<local> at <domain>" (example: "john dot smith at gmail dot com - is that correct?").
  - Name: repeat it once for clarity ("Thanks, John!").
- If the caller declines or says they will share later, accept it politely - do not push - and continue helping.
- NEVER refuse to take a name, phone number, or email - collecting these is expected and allowed.
- Do not invent, guess, or assume contact details.
- Do not collect any payment, card, password, or other sensitive data beyond name, phone, and email.
"""


def build_voicebot_prompt(bot_instructions: str, bot_tree: str) -> str:
    """Build the full system prompt by layering our voice/lead/emotion rules
    on top of the bot-builder API's Special Instructions and Flow tree.
    
    The bot-specific Instructions still take precedence for business logic,
    while the voice/style/lead-capture rules govern how the agent speaks.
    """
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


# --- Tone classification (used to switch voice_settings per turn) ----------
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
    
    Returns one of VOICE_TONE_OPTIONS. Falls back to DEFAULT_VOICE_TONE on
    any failure - this is best-effort and must not block the conversation.
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


# --- Initialize the TTS plugin ---------------------------------------------
try:
    elevenlabs_tts = elevenlabs.TTS(
        api_key=ELEVENLABS_API_KEY,
        voice_id=ACTIVE_VOICE_ID,
        model=ELEVENLABS_MODEL,
        voice_settings=DEFAULT_VOICE_SETTINGS,
    )
    logger.info(
        f"ElevenLabs TTS initialized (model={ELEVENLABS_MODEL}, voice_id={ACTIVE_VOICE_ID}, v3={IS_ELEVEN_V3})"
    )
except Exception as e:
    logger.error(f"Failed to initialize ElevenLabs TTS: {e}")
    logger.error("Check ELEVENLABS_API_KEY/ELEVEN_API_KEY, ELEVENLABS_VOICE_ID, and ELEVENLABS_MODEL")
    elevenlabs_tts = None



# ============================================================================
# ASSISTANT CLASS
# ============================================================================
# Custom Agent class that extends LiveKit's Agent base class
# Handles conversation management, message persistence, and RAG context injection

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_input(message: str, max_length: int = MAX_MESSAGE_LENGTH) -> str:
    """
    Validate and sanitize user input.
    
    Args:
        message: Input message to validate
        max_length: Maximum allowed length
        
    Returns:
        Sanitized message
        
    Raises:
        ValueError: If message is invalid
    """
    if not message or not isinstance(message, str):
        raise ValueError("Message must be a non-empty string")
    
    # Remove null bytes and control characters (except newlines and tabs)
    sanitized = ''.join(char for char in message if ord(char) >= 32 or char in '\n\t')
    
    if len(sanitized) > max_length:
        logger.warning(f"Message truncated from {len(sanitized)} to {max_length} characters")
        sanitized = sanitized[:max_length]
    
    return sanitized.strip()


def validate_chat_id(chat_id: str) -> str:
    """
    Validate chat ID format and length.
    
    Args:
        chat_id: Chat ID to validate
        
    Returns:
        Validated chat ID
        
    Raises:
        ValueError: If chat ID is invalid
    """
    if not chat_id or not isinstance(chat_id, str):
        raise ValueError("Chat ID must be a non-empty string")
    
    if len(chat_id) > MAX_CHAT_ID_LENGTH:
        raise ValueError(f"Chat ID exceeds maximum length of {MAX_CHAT_ID_LENGTH}")
    
    # Remove potentially dangerous characters
    sanitized = ''.join(char for char in chat_id if char.isalnum() or char in '-_')
    
    if not sanitized:
        raise ValueError("Chat ID contains no valid characters")
    
    return sanitized


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely parse JSON string with validation.
    
    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON object or default value
    """
    if not json_str or not isinstance(json_str, str):
        return default
    
    try:
        # Limit JSON size to prevent DoS
        if len(json_str) > 100000:  # 100KB limit
            logger.warning("JSON string exceeds size limit")
            return default
        
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return default
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON: {e}")
        return default


async def safe_post_message(
    chat_id: str,
    message: str,
    user_id: str,
    manager_id: str,
    visitor_id: str,
    website_id: str,
    nick_name: str,
    operator_name: str
) -> bool:
    """
    Safely post message to external API with error handling and rate limiting.
    
    Args:
        chat_id: Chat ID
        message: Message content
        user_id: User ID
        manager_id: Manager ID
        visitor_id: Visitor ID
        website_id: Website ID
        nick_name: Nick name
        operator_name: Operator name
        
    Returns:
        True if successful, False otherwise
    """
    async with api_semaphore:  # Rate limiting
        try:
            await asyncio.wait_for(
                post_message_to_conversation(
                    chat_id=chat_id,
                    message=message,
                    user_id=user_id,
                    manager_id=manager_id,
                    visitor_id=visitor_id,
                    website_id=website_id,
                    nick_name=nick_name,
                    operator_name=operator_name
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
    Safely save conversation to Redis with error handling and timeout.
    
    Args:
        chat_id: Chat ID
        conversation: Conversation object to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Validate chat_id
        validated_chat_id = validate_chat_id(chat_id)
        
        # Use asyncio.to_thread for timeout control
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
    """
    Safely load conversation from Redis with error handling and timeout.
    
    Args:
        chat_id: Chat ID
        
    Returns:
        Conversation object or None if not found/error
    """
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
# Custom Agent class that extends LiveKit's Agent base class
# Handles conversation management, message persistence, and RAG context injection

class Assistant(Agent):
    """
    Voice Assistant Agent that manages conversations, integrates with Redis,
    and provides RAG-enhanced responses.
    
    Attributes:
        bot_id: Unique identifier for the bot configuration
        Domain: Domain/website identifier for context
        con: Conversation object (stores history, tree, instructions)
        cid: Chat ID (unique conversation identifier)
        sess: AgentSession reference
        variablesForChat: Dictionary of chat metadata (visitor info, website info, etc.)
        _background_tasks: Set of background tasks for tracking
    """
    
    def __init__(self, cid: str, bot_id: str, domain: str, chat_ctx: ChatContext = None, con=None, sess=None, variablesForChat=None, instructions: str = None):
        """
        Initialize the Assistant agent.
        
        Args:
            cid: Chat ID (unique conversation identifier)
            bot_id: Bot configuration ID
            domain: Domain/website identifier
            chat_ctx: Pre-populated chat context (conversation history)
            con: Conversation object (Redis-backed)
            sess: AgentSession reference
            variablesForChat: Dictionary containing chat metadata
            instructions: Pre-built voicebot system prompt (built via build_voicebot_prompt).
                If omitted, a generic fallback is used.
        """
        # Store instance variables for conversation management
        self.bot_id = bot_id
        self.Domain = domain
        self.con = con  # Conversation object (contains history, tree, instructions)
        self.cid = cid  # Chat ID for Redis key
        self.sess = sess  # Session reference
        self.variablesForChat = variablesForChat or {}  # Chat metadata dictionary
        self._background_tasks = set()  # Track background tasks to prevent memory leaks
        self.session_logger = session_logger  # Session logger for file logging
        
        # Tone tracking - drives dynamic VoiceSettings via tts.update_options()
        # before each TTS synthesis. Reset to default on each new session.
        self.current_tone: str = DEFAULT_VOICE_TONE
        
        # Initialize parent Agent class with the dynamically-built voicebot
        # prompt (model-aware: includes v3 emotion tag rules when applicable,
        # plus voice/style and lead-capture sections layered on top of the
        # bot-builder API's special instructions and flow tree).
        super().__init__(
            instructions=instructions or "You are a helpful voice assistant.",
            chat_ctx=chat_ctx,
        )

    def _add_background_task(self, task: asyncio.Task) -> None:
        """
        Add a background task and ensure it's cleaned up when done.
        
        Args:
            task: Background task to track
        """
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
    
    def _apply_voice_settings_for_tone(self, tone: str) -> None:
        """Update the live ElevenLabs TTS plugin to use the preset for `tone`.
        
        Safe to call multiple times - update_options applies to the next
        synthesis request, not the in-flight one.
        """
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
        """Pre-process the LLM text stream before it reaches the TTS engine.
        
        For v3 models the leading [emotion] tag is the actual emotional
        delivery instruction and MUST stay in the text passed to TTS.
        
        For non-v3 models the LLM is told not to use bracket tags, but if
        one slips through we strip it here so it isn't spoken literally.
        """
        if IS_ELEVEN_V3:
            # v3 needs the leading [tag] to render the emotion - pass through.
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
    
    async def on_user_turn_completed(
            self, turn_ctx: ChatContext, new_message: ChatMessage,
    ) -> None:
        """
        Called when the user finishes speaking (turn completed).
        This hook allows us to:
        1. Save the user message to conversation history
        2. Post the message to external API for logging
        3. Persist conversation to Redis
        4. Fetch RAG context and inject it into the conversation
        
        Args:
            turn_ctx: Turn context (used to add messages to conversation)
            new_message: The user's message that was just transcribed
        """
        try:
            # Extract and validate the text content from the transcribed message
            q = new_message.text_content
            if not q:
                logger.warning("Empty message received, skipping processing")
                return
            
            # Validate and sanitize input
            try:
                q = validate_input(q)
            except ValueError as e:
                logger.error(f"Invalid message input: {e}")
                return
            
            # Log user message to session file
            start_time = time.time()
            self.session_logger.log_user_message(q)
            
            # Step 0: Classify tone for the upcoming reply and switch the
            # ElevenLabs voice_settings preset BEFORE the LLM streams text to
            # TTS. This is fast (~200ms with gpt-4o-mini) and best-effort -
            # if it fails we keep the previous tone.
            try:
                last_assistant = ""
                for m in reversed(self.con.conversation_history):
                    if m.get("role") == "assistant":
                        last_assistant = (m.get("content") or "")[:400]
                        break
                next_tone = await classify_tone_for_user(q, last_assistant)
                if next_tone != self.current_tone:
                    self._apply_voice_settings_for_tone(next_tone)
                self.session_logger.log_event(
                    "TONE_SELECTED",
                    f"Tone for next reply: {next_tone}",
                    {"tone": next_tone, "previous_tone": self.current_tone},
                )
            except Exception as e:
                logger.debug(f"Tone selection skipped due to error: {e}")
            
            # Step 1: Add user message to conversation history
            self.con.conversation_history.append({
                "role": "user",
                "content": q,
            })
            
            logger.info(f"User message added to history for chat_id: {self.cid}")
            
            # Step 2: Post user message to external API (async, non-blocking with error handling)
            # This logs the message in the chat system for UI display and history
            v_id = self.variablesForChat.get("VisitorId", "")
            chat_id = self.variablesForChat.get("ChatId", "")
            
            # Log API call start
            api_start_time = time.time()
            
            # Create background task with proper error handling
            async def post_with_logging():
                success = await safe_post_message(
                    chat_id=chat_id,
                    message=q,
                    user_id="0",  # Hardcoded as per requirement
                    manager_id="0",  # Hardcoded as per requirement
                    visitor_id=self.variablesForChat.get("VisitorId", ""),
                    website_id=self.variablesForChat.get("WebsiteId", ""),
                    nick_name=self.variablesForChat.get("NickName", ""),
                    operator_name=self.variablesForChat.get("visitorName", f"Visitor{v_id}")
                )
                duration = time.time() - api_start_time
                self.session_logger.log_api_call("post_message_to_conversation", success, duration)
                return success
            
            task = asyncio.create_task(post_with_logging())
            self._add_background_task(task)
            
            logger.info("User message posted to API (background task)")

            # Step 3: Persist conversation to Redis with proper error handling
            # Save the updated conversation history (with new user message) to Redis
            logger.debug(f"Saving conversation to Redis for chat_id: {self.cid}")
            logger.debug(f"Conversation history length: {len(self.con.conversation_history)}")
            
            redis_start_time = time.time()
            save_success = await safe_save_to_redis(self.cid, self.con)
            redis_duration = time.time() - redis_start_time
            self.session_logger.log_redis_operation("SAVE", save_success, redis_duration)
            
            if not save_success:
                logger.error(f"Failed to save conversation to Redis for chat_id: {self.cid}")
                # Continue execution even if save fails - conversation is in memory

            # Step 4: Fetch RAG (Retrieval Augmented Generation) context with timeout
            # This retrieves relevant information from the knowledge base based on user query
            # Uses vector store (LlamaIndex) to find semantically similar content
            # Runs in a thread pool to avoid blocking the async event loop
            try:
                rag_start_time = time.time()
                rag_content = await asyncio.wait_for(
                    asyncio.to_thread(
                        fetch_context,  # Function that queries vector store
                        q,  # User query
                        self.con,  # Conversation object (contains cached vector index)
                        self.bot_id,  # Bot ID for knowledge base path
                        self.Domain  # Domain for knowledge base path
                    ),
                    timeout=RAG_FETCH_TIMEOUT
                )
                rag_duration = time.time() - rag_start_time
                
                # Log RAG fetch
                self.session_logger.log_rag_fetch(q, len(rag_content) if rag_content else 0, rag_duration)
                
                # Step 5: Inject RAG context into conversation
                # Add the retrieved context as an assistant message so the LLM can use it
                # This enables context-aware responses without modifying the user's message
                if rag_content:
                    turn_ctx.add_message(
                        role="assistant",
                        content=f"Additional information relevant to the user's next message: {rag_content}"
                    )
                    logger.debug("RAG context injected into conversation")
                else:
                    logger.debug("No RAG context retrieved")
                    
            except AsyncTimeoutError:
                logger.warning(f"RAG context fetch timeout for chat_id: {self.cid}")
                self.session_logger.log_event("RAG_FETCH", "RAG fetch timeout", {"query": q})
                # Continue without RAG context
            except Exception as e:
                logger.error(f"Error fetching RAG context: {e}", exc_info=True)
                self.session_logger.log_event("RAG_FETCH_ERROR", f"RAG fetch error: {str(e)}", {"query": q})
                # Continue without RAG context - don't break the conversation

        except Exception as e:
            logger.error(f"Error in on_user_turn_completed: {e}", exc_info=True)
            # Don't re-raise - allow conversation to continue even if this hook fails

        # Note: This method completes before the assistant generates its reply
        # The assistant's reply will be handled by the conversation_item_added event handler


# ============================================================================
# ENTRYPOINT FUNCTION
# ============================================================================
# Main entry point for the LiveKit agent worker
# This function is called when a new voice session starts

from livekit import api
# from livekit.rtc.room import DataPacket

async def entrypoint(ctx: agents.JobContext):
    """
    Main entry point for the LiveKit voice assistant agent.
    
    This function orchestrates the entire voice conversation setup:
    1. Connects to LiveKit room
    2. Extracts metadata from participant
    3. Loads or creates conversation from Redis
    4. Fetches bot configuration (flow tree and instructions)
    5. Sets up AI session (STT, LLM, TTS, VAD)
    6. Starts conversation and generates greeting
    
    Args:
        ctx: JobContext containing room information and participant data
    """

    # SignalR manager initialization (currently disabled)
    # signalr_manager = RawSignalRManager("wss://blue.thelivechatsoftware.com/signalrserver/signalr")

    # Room recording setup (currently disabled)
    # Uncomment to enable audio recording of conversations
    # req = api.RoomCompositeEgressRequest(
    #     room_name=ctx.room.name,
    #     audio_only=True,
    #     file_outputs=[api.EncodedFileOutput(
    #         file_type=api.EncodedFileType.OGG,
    #         filepath="my-room-test.ogg",
    #     )],
    # )
    # lkapi = api.LiveKitAPI()
    # res = await lkapi.egress.start_room_composite_egress(req)
    # await lkapi.aclose()
    
    try:
        print("🌟 [ENTRYPOINT INIT] Starting voice assistant entrypoint")
        
        # Connect to the LiveKit room (establishes WebRTC connection)
        await ctx.connect()
        
        # Initialize session logger (will be set with chat_id later)
        session_id = None
        
        # ========================================================================
        # STEP 1: EXTRACT METADATA FROM PARTICIPANT
        # ========================================================================
        # Default parameters (used as fallback if metadata extraction fails)
        bot_id = '1344'  # Default bot ID
        domain = "testing.webgreeter.com/zem/hulk"  # Default domain
        # domain = 'liveadmins.com'  # Alternative default domain
        chat_id = None
        print(f"Default Parameters BOT ID: {bot_id}")
        variablesForChat = {}  # Dictionary to store chat metadata
        # Extract metadata from participant (sent from frontend/UI)
        # The metadata contains chat_id, bot_id, domain, visitor info, etc.
        try:
            # Iterate through all remote participants (usually just one user)
            for pid, participant in ctx.room.remote_participants.items():
                logger.info(f"Processing participant: {participant.identity}")
                logger.debug(f"Metadata from UI: {participant.metadata}")
                
                # Safely parse JSON metadata string into dictionary with validation
                if not participant.metadata:
                    logger.warning("Participant metadata is empty, using defaults")
                    raise ValueError("Empty metadata")
                
                metadata = safe_json_loads(participant.metadata, {})
                if not metadata:
                    logger.warning("Failed to parse metadata JSON, using defaults")
                    raise ValueError("Invalid metadata JSON")
                
                logger.debug(f"Parsed metadata: {metadata}")
                # Build variablesForChat dictionary with safe defaults using .get()
                # This dictionary is used for API calls and message logging
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
                logger.debug(f"variablesForChat: {variablesForChat}")
                
                # Extract core parameters from metadata with safe access
                chat_id = metadata.get("chat_id")
                if not chat_id:
                    logger.warning("chat_id not found in metadata, using default")
                    raise ValueError("Missing chat_id in metadata")
                
                # Validate chat_id
                try:
                    chat_id = validate_chat_id(chat_id)
                except ValueError as e:
                    logger.error(f"Invalid chat_id: {e}")
                    raise
                
                bot_id = '1344'  # Currently hardcoded (as per requirement)
                # TODO: Uncomment below when migrating to botbuilder system
                # bot_id = metadata.get("bot_id", '1344')
                # domain = metadata.get("domain", domain)
                
                logger.info(f"Extracted parameters - chat_id: {chat_id}, bot_id: {bot_id}, domain: {domain}")
                session_id = chat_id  # Set session ID for logging
                
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            # Fallback to default values if metadata extraction fails
            # This ensures the system can still function even with missing metadata
            logger.warning(f"Metadata extraction failed: {e}, using defaults")
            chat_id = 'mynewchatID5'
            bot_id = '1344'
            domain = 'liveadmins.com'
            logger.info(f"Using default parameters - chat_id: {chat_id}, bot_id: {bot_id}, domain: {domain}")
            session_id = chat_id  # Set session ID for logging
        except Exception as e:
            # Catch any other unexpected errors
            logger.error(f"Unexpected error extracting metadata: {e}", exc_info=True)
            chat_id = 'mynewchatID5'
            bot_id = '1344'
            domain = 'liveadmins.com'
            logger.info(f"Using default parameters after error - chat_id: {chat_id}, bot_id: {bot_id}, domain: {domain}")
            session_id = chat_id  # Set session ID for logging
        
        # If no participant metadata was available yet, use safe defaults.
        # This commonly happens for some UI/Web joins where metadata arrives later.
        if not chat_id:
            logger.warning("No chat_id resolved from participant metadata. Falling back to default chat context.")
            chat_id = 'mynewchatID5'
            bot_id = '1344'
            domain = 'liveadmins.com'
            session_id = chat_id
            if not variablesForChat:
                variablesForChat = {}
            variablesForChat.setdefault("ChatId", chat_id)
            variablesForChat.setdefault("VisitorId", "")
            variablesForChat.setdefault("WebsiteId", "")
            variablesForChat.setdefault("NickName", "")
            variablesForChat.setdefault("UserId", "")
            variablesForChat.setdefault("ManagerId", "")
            variablesForChat.setdefault("visitorName", "Guest")

        # Start session logging
        if session_id:
            session_metadata = {
                'chat_id': chat_id,
                'bot_id': bot_id,
                'domain': domain,
                'room_name': ctx.room.name if hasattr(ctx, 'room') and ctx.room else 'unknown',
                **variablesForChat
            }
            session_logger.start_session(session_id, session_metadata)
        
        # ========================================================================
        # STEP 2: LOAD OR INITIALIZE CONVERSATION
        # ========================================================================
        # Attempt to load existing conversation from Redis
        # If not found, create a new conversation and fetch bot configuration
        
        logger.info(f"Loading conversation for chat_id: {chat_id}")
        redis_load_start = time.time()
        con = await safe_load_from_redis(chat_id)
        redis_load_duration = time.time() - redis_load_start
        
        if con is None:
            logger.info("No existing conversation found in Redis, creating new one")
            session_logger.log_redis_operation("LOAD", False, redis_load_duration)
        else:
            logger.info(f"Loaded existing conversation from Redis - Agent: {con.Agent}, History: {len(con.conversation_history)} messages")
            session_logger.log_redis_operation("LOAD", True, redis_load_duration)
            session_logger.log_event("CONVERSATION_LOADED", 
                                    f"Loaded conversation with {len(con.conversation_history)} messages",
                                    {"agent": con.Agent, "history_length": len(con.conversation_history)})
        # If no conversation exists, create a new one
        if con is None:
            # Create Agent identifier (format: "bot_id_domain")
            # This is used as a key for bot-specific resources (vector store, etc.)
            Agent = str(bot_id) + "_" + str(domain)
            logger.info(f"Creating new conversation with Agent: {Agent}")
            
            # Create new Conversation object
            con = Conversation(bot_id, Agent)
            logger.debug(f"Conversation object created - Agent: {con.Agent}")

            # Fetch bot conversation flow/tree from API
            # The flow defines conversation paths and decision trees
            try:
                bot_tree = await asyncio.to_thread(fetch_tree, con, bot_id)
                if bot_tree:
                    con.tree = bot_tree
                    logger.info(f"Fetched Tree/Flow for bot_id: {bot_id}")
                else:
                    logger.warning(f"No Tree/Flow found for bot_id: {bot_id}")
            except Exception as e:
                logger.error(f"Error fetching tree for bot_id {bot_id}: {e}", exc_info=True)

            # Fetch bot instructions/prompt from API
            # Instructions define how the bot should behave and respond
            try:
                BotPrompt = await asyncio.to_thread(fetch_Instructions, con, bot_id)
                if BotPrompt:
                    con.Instructions = BotPrompt
                    logger.info(f"Fetched BotPrompt for bot_id: {bot_id}")
                else:
                    logger.warning(f"No BotPrompt found for bot_id: {bot_id}")
            except Exception as e:
                logger.error(f"Error fetching instructions for bot_id {bot_id}: {e}", exc_info=True)
           
            # Save the newly created conversation object to Redis
            # This persists the bot configuration (tree, instructions) for future sessions
            save_success = await safe_save_to_redis(chat_id, con)
            if save_success:
                logger.info("New conversation saved to Redis successfully")
                session_logger.log_event("CONVERSATION_CREATED", 
                                        "New conversation created and saved",
                                        {"bot_id": bot_id, "agent": Agent, "tree_length": len(con.tree), "instructions_length": len(con.Instructions)})
            else:
                logger.error("Failed to save new conversation to Redis")

        # ========================================================================
        # STEP 3: BUILD SYSTEM PROMPT
        # ========================================================================
        # Extract bot configuration (tree/flow and instructions) from conversation object
        # These were either loaded from Redis or fetched from API
        
        tree = con.tree or ""  # Conversation flow/tree (defines conversation paths)
        instructions = con.Instructions or ""  # Bot-specific instructions/prompts
        logger.debug(f"Tree length: {len(tree)}, Instructions length: {len(instructions)}")

        # Construct the system prompt by layering our voice/style/lead-capture
        # rules and (when v3 is active) the emotion-tag instructions on top of
        # the bot-builder API's Special Instructions and Flow tree.
        prompt = build_voicebot_prompt(instructions, tree)
        logger.info(
            f"Voicebot prompt built (model={ELEVENLABS_MODEL}, v3={IS_ELEVEN_V3}, "
            f"prompt_len={len(prompt)})"
        )

        # Update conversation history with system prompt
        # If no system message exists, insert it at the beginning
        # If system message exists, update it (in case bot config changed)
        if len(con.conversation_history) == 0 or con.conversation_history[0].get('role') != 'system':
            con.conversation_history.insert(0, {'role': 'system', 'content': prompt})
            logger.info("Added System Prompt to conversation history")
        else:
            con.conversation_history[0]['content'] = prompt
            logger.info("Updated System Prompt in conversation history")

        # Log conversation state
        logger.info(f"Agent: {con.Agent}, History: {len(con.conversation_history)} messages")

        # ========================================================================
        # STEP 4: INITIALIZE CHAT CONTEXT
        # ========================================================================
        # Create LiveKit ChatContext and populate it with conversation history
        # This context is used by the LLM to maintain conversation continuity
        
        chat_ctx = ChatContext()
        
        # Populate ChatContext with all messages from conversation history
        # This includes: system prompt, previous user messages, previous assistant responses
        for m in con.conversation_history:
            role = m.get("role", "user")
            content = m.get("content", "")
            if content:  # Only add non-empty messages
                chat_ctx.add_message(role=role, content=content)

        logger.debug(f"ChatContext populated with {len(con.conversation_history)} messages")

        

        # ========================================================================
        # STEP 5: CONFIGURE AI SESSION
        # ========================================================================
        # Create AgentSession with all AI components (STT, LLM, TTS, VAD)
        # This session manages the entire voice conversation pipeline
        
        # Initialize STT with error handling
        try:
            stt_instance = assemblyai.STT(
                api_key=ASSEMBLYAI_API_KEY,
                model="u3-rt-pro",  # Use plugin-supported model name per LiveKit docs.
                min_turn_silence=100,
                max_turn_silence=1000,
                vad_threshold=0.3,
            )
            logger.info("AssemblyAI STT initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AssemblyAI STT: {e}")
            logger.error("Check your ASSEMBLYAI_API_KEY environment variable")
            raise
        
        # Initialize LLM with error handling
        try:
            llm_instance = LLM(model="gpt-4o-mini")  # OpenAI's efficient model
            logger.info("OpenAI LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI LLM: {e}")
            logger.error("Check your OPENAI_API_KEY environment variable")
            raise
        
        # Validate TTS is initialized
        if elevenlabs_tts is None:
            logger.error("ElevenLabs TTS was not initialized. Check your ELEVENLABS_API_KEY/ELEVEN_API_KEY and voice/model settings.")
            raise RuntimeError("ElevenLabs TTS initialization failed. Cannot start agent session.")
        
        session = AgentSession(
            # Preemptive generation (disabled): Generate response while user is still speaking
            # preemptive_generation=True,  # Uncomment to enable faster responses
            
            # Speech-to-Text (STT): Converts user's speech to text
            stt=stt_instance,
            
            # Large Language Model (LLM): Generates text responses
            llm=llm_instance,
            
            # Text-to-Speech (TTS): Converts assistant's text to speech
            # Uses the configured ElevenLabs TTS plugin (already initialized at module level)
            tts=elevenlabs_tts,
            
            # Voice Activity Detection (VAD): Detects when user starts/stops speaking
            vad=silero.VAD.load(
                activation_threshold=0.3,  # Align with AssemblyAI vad_threshold.
                min_silence_duration=0.3,   # Wait 0.3s of silence before considering turn ended (faster than default 0.55s)
                min_speech_duration=0.05,  # Minimum speech duration to trigger detection
            ),

            # Use the new turn_handling API (replaces deprecated turn_detection/allow_interruptions/min_endpointing_delay).
            # With AssemblyAI, STT-driven turn detection is more stable than mixing with a secondary turn detector.
            turn_handling=TurnHandlingOptions(
                turn_detection="stt",
                endpointing={"min_delay": 0},
                allow_interruptions=True,
            ),
            
            # Noise cancellation is configured in room_input_options (below)
            # Note: APM (Acoustic Processing Module) is started via room_input_options
        )

        # ========================================================================
        # STEP 6: CREATE ASSISTANT INSTANCE
        # ========================================================================
        # Instantiate the Assistant agent with all required context
        
        # Note: SignalR manager version is commented out (alternative message delivery)
        # assistant = Assistant(cid=chat_id, bot_id=bot_id, domain=domain, chat_ctx=chat_ctx, con=con, variablesForChat=variablesForChat, signalr_manager=signalr_manager)
        assistant = Assistant(
            cid=chat_id,
            bot_id=bot_id,
            domain=domain,
            chat_ctx=chat_ctx,
            con=con,
            variablesForChat=variablesForChat,
            instructions=prompt,
        )

        # Store session reference in assistant (for potential future use)
        assistant.sess = session
        
        # Hook into LLM events to track token usage
        # Note: LiveKit agents may expose token usage through events or callbacks
        # This is a best-effort approach to track LLM tokens
        def track_llm_tokens(event_data):
            """Track LLM token usage from events if available."""
            try:
                # Try to extract token information from event
                # This depends on LiveKit's event structure
                if hasattr(event_data, 'usage'):
                    usage = event_data.usage
                    input_tokens = getattr(usage, 'input_tokens', 0)
                    output_tokens = getattr(usage, 'output_tokens', 0)
                    if input_tokens > 0 or output_tokens > 0:
                        assistant.session_logger.log_llm_usage(
                            input_tokens, 
                            output_tokens,
                            model=getattr(event_data, 'model', 'gpt-4o-mini')
                        )
            except Exception as e:
                logger.debug(f"Could not extract LLM token usage: {e}")
        
        # Track LLM token usage by monitoring conversation history
        # We'll estimate tokens based on message lengths
        # Store previous history length to detect new messages
        assistant._previous_history_length = len(assistant.con.conversation_history) if assistant.con else 0

        # ========================================================================
        # STEP 7: SET UP EVENT HANDLERS
        # ========================================================================
        # Register handler for when assistant generates a response
        # This allows us to save assistant messages and post them to external API
        
        @session.on("conversation_item_added")
        def on_item(event: ConversationItemAddedEvent):
            """
            Event handler called when a new message is added to the conversation.
            This fires for both user and assistant messages, but we only process assistant messages here.
            User messages are handled in on_user_turn_completed.
            
            Args:
                event: ConversationItemAddedEvent containing the new message
            """
            try:
                msg = event.item

                # LiveKit can emit non-chat items (e.g., AgentHandoff) that don't have role/text_content.
                if not hasattr(msg, "role"):
                    logger.debug(f"Skipping non-message conversation item: {type(msg).__name__}")
                    return
                
                # Only process assistant messages (user messages handled elsewhere)
                if msg.role == "assistant":
                    logger.info(f"Processing assistant message for chat_id: {assistant.cid}")
                    logger.debug(f"History length before save: {len(assistant.con.conversation_history)}")

                    # Validate message content
                    if not msg.text_content:
                        logger.warning("Empty assistant message, skipping")
                        return
                    
                    try:
                        validated_content = validate_input(msg.text_content)
                    except ValueError as e:
                        logger.error(f"Invalid assistant message content: {e}")
                        return

                    # For v3, the LLM emits "[emotion] text" - the [emotion]
                    # is rendered as voice delivery and is NOT meant to appear
                    # in chat transcripts. Strip it before persisting/posting.
                    display_content = (
                        strip_leading_tag(validated_content) if IS_ELEVEN_V3 else validated_content
                    )
                    if not display_content:
                        display_content = validated_content
                    
                    # Step 1: Add assistant message to conversation history
                    assistant.con.conversation_history.append({
                        "role": "assistant",
                        "content": display_content,
                    })
                    
                    # Log assistant response to session file
                    assistant.session_logger.log_assistant_response(display_content)
                    
                    # Track LLM token usage
                    # Estimate tokens based on conversation history
                    try:
                        # Calculate input tokens (all messages before the assistant response)
                        input_messages = [msg for msg in assistant.con.conversation_history[:-1]]
                        input_text = ' '.join(msg.get('content', '') for msg in input_messages)
                        estimated_input_tokens = len(input_text) // 4  # Rough estimate: 4 chars per token
                        
                        # Calculate output tokens (the assistant response - the
                        # spoken/displayed text without any v3 emotion tag)
                        estimated_output_tokens = len(display_content) // 4
                        
                        # Only log if we have meaningful token counts
                        if estimated_input_tokens > 0 or estimated_output_tokens > 0:
                            assistant.session_logger.log_llm_usage(
                                estimated_input_tokens,
                                estimated_output_tokens,
                                model='gpt-4o-mini'
                            )
                    except Exception as e:
                        logger.debug(f"Error estimating LLM tokens: {e}")
                    
                    # Step 2: Post assistant message to external API (async, non-blocking with error handling)
                    # This logs the message in the chat system for UI display
                    # Capture variablesForChat in closure (safe - it's set before this handler)
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
                            operator_name="Voicebot"  # Identify as voice bot
                        )
                        duration = time.time() - api_start_time
                        assistant.session_logger.log_api_call("post_message_to_conversation", success, duration)
                        return success
                    
                    task = asyncio.create_task(post_assistant_with_logging())
                    assistant._add_background_task(task)
                    
                    # Step 3: Persist updated conversation to Redis with error handling
                    # This saves the assistant's response for future sessions
                    # Use asyncio.create_task to avoid blocking
                    redis_start_time = time.time()
                    
                    async def save_with_logging():
                        success = await safe_save_to_redis(assistant.cid, assistant.con)
                        duration = time.time() - redis_start_time
                        assistant.session_logger.log_redis_operation("SAVE", success, duration)
                        return success
                    
                    save_task = asyncio.create_task(save_with_logging())
                    assistant._add_background_task(save_task)
                    
            except Exception as e:
                logger.error(f"Error in conversation_item_added handler: {e}", exc_info=True)
                # Don't re-raise - allow conversation to continue


        # ========================================================================
        # STEP 8: START SESSION AND BEGIN CONVERSATION
        # ========================================================================
        # Start the agent session with the LiveKit room
        # This establishes audio streams and begins processing
        
        # Configure room input options
        # Note: BVC noise cancellation requires LiveKit Cloud
        # For self-hosted instances, disable noise cancellation or use a different method
        room_input_options = RoomInputOptions()
        
        # Try to enable noise cancellation, but handle gracefully if it fails
        # BVC requires LiveKit Cloud, so it will fail on self-hosted instances
        try:
            # Check if we're using LiveKit Cloud (you can set this via env var)
            use_cloud_noise_cancellation = os.getenv('LIVEKIT_USE_CLOUD_NC', 'false').lower() == 'true'
            if use_cloud_noise_cancellation:
                room_input_options = RoomInputOptions(
                    noise_cancellation=noise_cancellation.BVC(),
                )
                logger.info("Using BVC noise cancellation (requires LiveKit Cloud)")
            else:
                logger.info("Noise cancellation disabled (BVC requires LiveKit Cloud)")
        except Exception as e:
            logger.warning(f"Could not enable noise cancellation: {e}. Continuing without it.")
            room_input_options = RoomInputOptions()  # Use default options without noise cancellation
        
        # Start the session with error handling
        try:
            logger.info("Starting agent session...")
            await session.start(
                room=ctx.room,  # LiveKit room to connect to
                agent=assistant,  # Our custom Assistant instance
                room_input_options=room_input_options,
            )
            logger.info("Agent session started successfully")
            session_logger.log_event("SESSION_STARTED", "Agent session started successfully")
        except Exception as e:
            logger.error(f"Failed to start agent session: {e}", exc_info=True)
            session_logger.log_event("SESSION_START_ERROR", f"Failed to start session: {str(e)}")
            raise  # Re-raise to allow proper cleanup

        # Generate initial greeting message
        # This is the first thing the user hears when joining the conversation
        session_logger.log_event("GREETING_GENERATION", "Generating initial greeting message")
        try:
            await session.generate_reply(
                instructions="Greet the user and offer your assistance."
            )
            session_logger.log_event("GREETING_SENT", "Initial greeting sent to user")
            logger.info("Initial greeting generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate initial greeting: {e}")
            session_logger.log_event("GREETING_ERROR", f"Failed to generate greeting: {str(e)}")
            # Don't raise - allow session to continue even if greeting fails
            # The user can still interact with the agent

    except KeyboardInterrupt:
        # Allow clean shutdown on Ctrl+C
        logger.info("Received keyboard interrupt, shutting down gracefully")
        session_logger.end_session("KEYBOARD_INTERRUPT")
        raise
    except SystemExit:
        # Allow system exit to propagate
        session_logger.end_session("SYSTEM_EXIT")
        raise
    except Exception as e:
        # Handle any errors that occur during initialization or runtime
        logger.error(f"Entrypoint failed: {e}", exc_info=True)
        session_logger.end_session("ERROR")
    finally:
        # Always end session logging on exit
        if session_logger.current_session_id:
            session_logger.end_session("NORMAL")
    
    # Cleanup (currently disabled - SignalR disconnect)
    # finally:
    #     await signalr_manager.disconnect()
    #     print("SignalR connection closed on exit")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
# Run the LiveKit agent worker with our entrypoint function
# This starts the agent server that listens for incoming voice sessions

if __name__ == "__main__":
    run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
    # run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint, agent_name="voicebot-ui-aai-eleven"))