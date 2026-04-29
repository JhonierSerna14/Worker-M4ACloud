import asyncio
import ctypes
import os
import re
import time
import unicodedata
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx
from loguru import logger

from .backend_api import send_progress
from .config import (
    COMPUTE_TYPE,
    VAD_FILTER_ENABLED,
    VAD_MIN_SILENCE_DURATION_MS,
    VAD_SPEECH_PAD_MS,
    WHISPER_BEAM_SIZE,
    WHISPER_BEST_OF,
    WHISPER_COMPRESSION_RATIO,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL,
    WHISPER_NO_SPEECH_THRESHOLD,
    WHISPER_REPETITION_PENALTY,
    WHISPER_TEMPERATURE,
)

_whisper_model = None
_whisper_meta = {"device": None, "model_size": None}
_cuda_dll_configured = False
_cuda_dll_handles = []
_cuda_dll_dirs: list[Path] = []
_cuda_runtime_loaded = False

ACADEMIC_ACTIVITY_HINTS = (
    "tarea",
    "tareas",
    "taller",
    "talleres",
    "entrega",
    "entregable",
    "entregables",
    "fecha limite",
    "fecha de entrega",
    "proxima clase",
    "examen",
    "parcial",
    "quiz",
    "quices",
    "sustentacion",
    "proyecto",
)


def _configure_cuda_dll_dirs() -> None:
    global _cuda_dll_configured
    if _cuda_dll_configured:
        return
    _cuda_dll_configured = True

    if os.name != "nt":
        return

    candidates: list[Path] = []

    cuda_path = os.getenv("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")

    try:
        import site

        for base in site.getsitepackages():
            site_base = Path(base)
            candidates.extend(
                [
                    site_base / "nvidia" / "cublas" / "bin",
                    site_base / "nvidia" / "cudnn" / "bin",
                    site_base / "nvidia" / "cuda_runtime" / "bin",
                ]
            )
    except Exception:
        pass

    added_dirs: list[str] = []
    for path in candidates:
        if not path.exists():
            continue
        if path in _cuda_dll_dirs:
            continue
        try:
            handle = os.add_dll_directory(str(path))
            _cuda_dll_handles.append(handle)
            _cuda_dll_dirs.append(path)
            added_dirs.append(str(path))
        except Exception:
            continue

    # Algunas librerías resuelven dependencias por PATH; lo extendemos para este proceso.
    if _cuda_dll_dirs:
        current_path = os.environ.get("PATH", "")
        prefix = os.pathsep.join(str(p) for p in _cuda_dll_dirs)
        if prefix and prefix not in current_path:
            os.environ["PATH"] = prefix + os.pathsep + current_path

    if added_dirs:
        logger.info(f"DLL CUDA registradas: {len(added_dirs)} rutas")


def _preload_cuda_runtime_dlls() -> None:
    global _cuda_runtime_loaded
    if _cuda_runtime_loaded or os.name != "nt":
        return

    _configure_cuda_dll_dirs()

    required = [
        "cudart64_12.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cudnn64_9.dll",
    ]

    loaded = 0
    missing: list[str] = []

    for dll_name in required:
        dll_path = None
        for base in _cuda_dll_dirs:
            candidate = base / dll_name
            if candidate.exists():
                dll_path = candidate
                break

        if dll_path is None:
            missing.append(dll_name)
            continue

        try:
            _cuda_dll_handles.append(ctypes.WinDLL(str(dll_path)))
            loaded += 1
        except Exception as exc:
            raise RuntimeError(f"No se pudo cargar {dll_name} desde {dll_path}: {exc}") from exc

    if missing:
        raise RuntimeError(
            "Faltan DLL CUDA requeridas para GPU strict: "
            + ", ".join(missing)
            + f" | Rutas buscadas: {[str(p) for p in _cuda_dll_dirs]}"
        )

    _cuda_runtime_loaded = True
    logger.info(f"Runtime CUDA precargado ({loaded} DLL)")


def _is_cuda_runtime_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = (
        "cublas64_12.dll",
        "cudnn",
        "cudart",
        "cuda",
        "cannot be loaded",
        "failed to load library",
    )
    return any(marker in text for marker in markers)


def _detect_device() -> str:
    _configure_cuda_dll_dirs()
    desired = WHISPER_DEVICE.lower()
    if desired == "cpu":
        return "cpu"
    if desired == "cuda":
        return "cuda"
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _normalize_compute_type(device: str, compute_type: str) -> str:
    if device == "cuda":
        valid = {"float16", "float32", "int8_float16", "int8_float32", "int8"}
        return compute_type if compute_type in valid else "float16"
    valid = {"int8", "int8_float32", "float32"}
    return compute_type if compute_type in valid else "int8"


def _gpu_required() -> bool:
    return WHISPER_DEVICE.lower() == "cuda"


def _build_whisper_kwargs(audio_path: str) -> dict:
    return {
        "audio": audio_path,
        "language": WHISPER_LANGUAGE,
        "beam_size": WHISPER_BEAM_SIZE,
        "best_of": WHISPER_BEST_OF,
        "temperature": WHISPER_TEMPERATURE,
        "compression_ratio_threshold": WHISPER_COMPRESSION_RATIO,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": WHISPER_NO_SPEECH_THRESHOLD,
        "condition_on_previous_text": False,
        "initial_prompt": "Transcripcion de clase universitaria clara y precisa.",
        "repetition_penalty": WHISPER_REPETITION_PENALTY,
        "word_timestamps": False,
        "vad_filter": VAD_FILTER_ENABLED,
        "vad_parameters": {
            "min_silence_duration_ms": VAD_MIN_SILENCE_DURATION_MS,
            "speech_pad_ms": VAD_SPEECH_PAD_MS,
        },
    }


def _normalize_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", ascii_text).strip()


def _contains_academic_activity_marker(text: str) -> bool:
    normalized_text = _normalize_for_match(text)
    return any(hint in normalized_text for hint in ACADEMIC_ACTIVITY_HINTS)


def _is_hallucination(text: str) -> bool:
    if not text or len(text.strip()) < 10:
        return True

    # Conserva segmentos relevantes para tareas/fechas aunque tengan repeticiones.
    if _contains_academic_activity_marker(text):
        return False

    words = text.lower().split()
    if len(words) < 2:
        return False

    consecutive_repeats = sum(1 for i in range(len(words) - 1) if words[i] == words[i + 1])
    if len(words) > 0 and (consecutive_repeats / len(words)) > 0.25:
        return True

    phrase_counts = {}
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i + 1]}"
        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
    if any(count > 4 for count in phrase_counts.values()):
        return True

    unique_words = set(words)
    if len(words) > 8 and (len(unique_words) / len(words)) < 0.35:
        return True

    return False


def get_whisper_model(device_override: str | None = None, compute_type_override: str | None = None):
    global _whisper_model
    _configure_cuda_dll_dirs()
    _preload_cuda_runtime_dlls()
    if _whisper_model is None or (
        device_override is not None and _whisper_meta.get("device") != device_override
    ):
        from faster_whisper import WhisperModel

        device = device_override or _detect_device()
        if _gpu_required() and device != "cuda":
            raise RuntimeError(
                "WHISPER_DEVICE=cuda requiere GPU CUDA disponible. "
                "No se permite fallback a CPU."
            )

        compute_type = compute_type_override or _normalize_compute_type(device, COMPUTE_TYPE)
        download_root = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
        cpu_threads = max(4, os.cpu_count() or 4)

        logger.info(f"⚙️ Cargando Whisper {WHISPER_MODEL} en {device} ({compute_type})...")
        try:
            _whisper_model = WhisperModel(
                WHISPER_MODEL,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                num_workers=2 if device == "cuda" else 1,
                download_root=download_root,
            )
            _whisper_meta["device"] = device
            _whisper_meta["model_size"] = WHISPER_MODEL
            logger.success(f"✅ Whisper listo ({WHISPER_MODEL} / {device})")
        except Exception as e:
            error_msg = str(e).lower()

            if _gpu_required() and device == "cuda":
                raise RuntimeError(
                    "Whisper no pudo iniciar en GPU CUDA y el fallback a CPU está deshabilitado. "
                    f"Detalle: {e}"
                ) from e

            if "out of memory" not in error_msg and "cuda" not in error_msg:
                raise

            logger.warning("Whisper fallo por memoria/dispositivo. Intentando fallback de modelo...")
            for fallback_model in ["small", "base", "tiny"]:
                try:
                    _whisper_model = WhisperModel(
                        fallback_model,
                        device=device,
                        compute_type="int8",
                        cpu_threads=cpu_threads,
                        num_workers=1,
                        download_root=download_root,
                    )
                    _whisper_meta["device"] = device
                    _whisper_meta["model_size"] = fallback_model
                    logger.warning(f"Whisper en modo fallback: {fallback_model} / {device}")
                    break
                except Exception:
                    continue

            if _whisper_model is None:
                raise RuntimeError(
                    "No se pudo iniciar Whisper en GPU CUDA (ni siquiera con modelos fallback)."
                ) from e

    return _whisper_model


async def transcribe_file_with_progress(
    client: httpx.AsyncClient,
    nota_id: int,
    audio_path: str,
    progress_callback: Callable[[int, str], Awaitable[None]] | None = None,
) -> tuple[str, float, str]:
    loop = asyncio.get_running_loop()
    import concurrent.futures

    async def _transcribe_once(force_cpu: bool = False) -> tuple[str, float, str]:
        progress_queue: asyncio.Queue[tuple[float, float, float]] = asyncio.Queue()

        def _report_progress(ratio: float, current_sec: float, total_sec: float):
            try:
                loop.call_soon_threadsafe(progress_queue.put_nowait, (ratio, current_sec, total_sec))
            except RuntimeError:
                pass

        def _run_with_progress(reporter):
            model = get_whisper_model(
                device_override="cpu" if force_cpu else None,
                compute_type_override="int8" if force_cpu else None,
            )
            segments, info = model.transcribe(**_build_whisper_kwargs(audio_path))

            pieces = []
            total_sec = max(float(info.duration or 0.0), 1.0)

            for seg in segments:
                seg_text = (seg.text or "").strip()
                if seg_text and not _is_hallucination(seg_text):
                    pieces.append(seg_text)
                current_sec = float(getattr(seg, "end", 0.0) or 0.0)
                ratio = min(max(current_sec / total_sec, 0.0), 1.0)
                reporter(ratio, current_sec, total_sec)

            text = " ".join(pieces).strip()
            return (
                text,
                float(getattr(info, "duration", 0.0) or 0.0),
                (getattr(info, "language", None) or WHISPER_LANGUAGE),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = loop.run_in_executor(pool, _run_with_progress, _report_progress)

            last_percent = -1
            last_sent_ts = 0.0

            while True:
                if future.done() and progress_queue.empty():
                    break

                try:
                    ratio, current_sec, total_sec = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                percent = min(55, max(5, int(ratio * 50) + 5))
                now = time.monotonic()

                if percent > last_percent and (now - last_sent_ts >= 1.0 or percent - last_percent >= 2):
                    await send_progress(client, nota_id, percent, "Transcribiendo con Whisper…")
                    if progress_callback is not None:
                        await progress_callback(percent, "Transcribiendo con Whisper…")
                    last_percent = percent
                    last_sent_ts = now

            return await future

    try:
        return await _transcribe_once(force_cpu=False)
    except RuntimeError as exc:
        if _is_cuda_runtime_error(exc):
            raise RuntimeError(
                "Fallo de runtime CUDA detectado (por ejemplo cublas/cudnn). "
                "CPU fallback deshabilitado por configuración de GPU estricta."
            ) from exc
        raise
