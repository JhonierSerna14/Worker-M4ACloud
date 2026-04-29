import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

LOG_PATH = PROJECT_ROOT / "worker.log"


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
GEMINI_API_KEY = _get_str("GEMINI_API_KEY")
GEMINI_MODEL = _get_str("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_FALLBACK_MODEL = _get_str("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite")
GEMINI_MAX_OUTPUT_TOKENS = _get_int("GEMINI_MAX_OUTPUT_TOKENS", 6144)
OLLAMA_BASE_URL = _get_str("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = _get_str("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_ENABLED = _get_bool("OLLAMA_ENABLED", False)
OLLAMA_REQUEST_TIMEOUT = _get_float("OLLAMA_REQUEST_TIMEOUT", 600)
OLLAMA_CONNECT_TIMEOUT = _get_float("OLLAMA_CONNECT_TIMEOUT", 10)
OLLAMA_MAX_ATTEMPTS = _get_int("OLLAMA_MAX_ATTEMPTS", 2)
OLLAMA_RETRY_DELAY = _get_float("OLLAMA_RETRY_DELAY", 2.5)
OLLAMA_SAFE_PROMPT_CHARS = _get_int("OLLAMA_SAFE_PROMPT_CHARS", 14000)
OLLAMA_NUM_PREDICT = _get_int("OLLAMA_NUM_PREDICT", 2300)
AI_TEMPERATURE = _get_float("AI_TEMPERATURE", 0.1)
AI_REQUEST_TIMEOUT = _get_float("AI_REQUEST_TIMEOUT", 120)
MAX_TRANSCRIPT_SIZE_SINGLE = _get_int("MAX_TRANSCRIPT_SIZE_SINGLE", 150000)
AI_CHUNK_MAX_CHARS = _get_int("AI_CHUNK_MAX_CHARS", 40000)
GEMINI_MAX_ATTEMPTS = _get_int("GEMINI_MAX_ATTEMPTS", 2)
SUMMARY_TARGET_WORDS_MIN = _get_int("SUMMARY_TARGET_WORDS_MIN", 2000)
SUMMARY_TARGET_WORDS_MAX = _get_int("SUMMARY_TARGET_WORDS_MAX", 3000)
# Thresholds de procesamiento por proveedor (configurables via .env)
GEMINI_MAX_SINGLE_CHARS = _get_int("GEMINI_MAX_SINGLE_CHARS", 90000)
GEMINI_CHUNK_CHARS = _get_int("GEMINI_CHUNK_CHARS", 60000)
DEFAULT_MAX_SINGLE_CHARS = _get_int("DEFAULT_MAX_SINGLE_CHARS", 30000)

LOG_LEVEL = _get_str("LOG_LEVEL", "INFO")
WORKER_HEADERS = {"X-Worker-Key": WORKER_SECRET_KEY}
