import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _get_required_str(name: str) -> str:
	value = os.getenv(name, "").strip()
	if not value:
		raise RuntimeError(f"Falta la variable de entorno obligatoria: {name}")
	return value


def _get_str(name: str, default: str = "") -> str:
	return os.getenv(name, default).strip()


def _get_int(name: str, default: int) -> int:
	return int(_get_str(name, str(default)))


def _get_float(name: str, default: float) -> float:
	return float(_get_str(name, str(default)))


def _get_bool(name: str, default: bool) -> bool:
	fallback = "true" if default else "false"
	return _get_str(name, fallback).lower() in {"1", "true", "yes", "on"}


BACKEND_URL = _get_required_str("BACKEND_URL").rstrip("/")
WORKER_SECRET_KEY = _get_required_str("WORKER_SECRET_KEY")
API_BASE = f"{BACKEND_URL}/api/v1"
POLL_INTERVAL = _get_int("POLL_INTERVAL_SECONDS", 10)

WHISPER_MODEL = _get_str("WHISPER_MODEL_SIZE", "medium")
WHISPER_DEVICE = _get_str("WHISPER_DEVICE", "auto")
COMPUTE_TYPE = _get_str("WHISPER_COMPUTE_TYPE", "int8_float16")
WHISPER_BEAM_SIZE = _get_int("WHISPER_BEAM_SIZE", 5)
WHISPER_BEST_OF = _get_int("WHISPER_BEST_OF", 5)
WHISPER_TEMPERATURE = _get_float("WHISPER_TEMPERATURE", 0.0)
WHISPER_LANGUAGE = _get_str("WHISPER_LANGUAGE", "es")
WHISPER_COMPRESSION_RATIO = _get_float("WHISPER_COMPRESSION_RATIO", 2.0)
WHISPER_NO_SPEECH_THRESHOLD = _get_float("WHISPER_NO_SPEECH_THRESHOLD", 0.7)
WHISPER_REPETITION_PENALTY = _get_float("WHISPER_REPETITION_PENALTY", 1.1)
VAD_FILTER_ENABLED = _get_bool("VAD_FILTER_ENABLED", True)
VAD_MIN_SILENCE_DURATION_MS = _get_int(
	"VAD_MIN_SILENCE_DURATION_MS",
	_get_int("VAD_MIN_SPEECH_DURATION_MS", 250),
)
VAD_SPEECH_PAD_MS = _get_int("VAD_SPEECH_PAD_MS", 400)

SUMMARY_PROVIDER = _get_str("SUMMARY_PROVIDER", "gemini").lower()
GROQ_API_KEY = _get_str("GROQ_API_KEY")
GROQ_MODEL = _get_str("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_API_KEY = _get_str("GEMINI_API_KEY")
GEMINI_MODEL = _get_str("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_REQUEST_DELAY = _get_float("GROQ_REQUEST_DELAY", 2.5)
AI_TEMPERATURE = _get_float("AI_TEMPERATURE", 0.1)
AI_REQUEST_TIMEOUT = _get_float("AI_REQUEST_TIMEOUT", 120)
MAX_TRANSCRIPT_SIZE_SINGLE = _get_int("MAX_TRANSCRIPT_SIZE_SINGLE", 15000)
AI_CHUNK_MAX_CHARS = _get_int("AI_CHUNK_MAX_CHARS", 12000)
GROQ_CHUNK_MAX_CHARS = _get_int("GROQ_CHUNK_MAX_CHARS", 8000)
GROQ_SAFE_PROMPT_CHARS = _get_int("GROQ_SAFE_PROMPT_CHARS", 10000)

LOG_LEVEL = _get_str("LOG_LEVEL", "INFO")
WORKER_HEADERS = {"X-Worker-Key": WORKER_SECRET_KEY}
