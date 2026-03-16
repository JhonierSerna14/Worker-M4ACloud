import asyncio
import threading

import httpx
from loguru import logger

from .backend_api import claim_job, extract_queued_jobs, fail_job, get_next_job, verify_backend_connection
from .config import BACKEND_URL, COMPUTE_TYPE, POLL_INTERVAL, SUMMARY_PROVIDER, WHISPER_DEVICE, WHISPER_MODEL
from .job_processor import JobProcessor
from .logging_setup import configure_logging
from .runtime_state import WorkerRuntimeState


async def _sleep_with_stop(stop_event: threading.Event | None, seconds: float):
    if stop_event is None:
        await asyncio.sleep(seconds)
        return

    loop = asyncio.get_running_loop()
    end = loop.time() + seconds
    while not stop_event.is_set():
        remaining = end - loop.time()
        if remaining <= 0:
            return
        await asyncio.sleep(min(0.5, remaining))


async def _read_queue_count(client: httpx.AsyncClient) -> int | None:
    try:
        status = await verify_backend_connection(client)
    except Exception:
        return None
    return extract_queued_jobs(status)


async def run_worker(
    runtime_state: WorkerRuntimeState | None = None,
    stop_event: threading.Event | None = None,
) -> int:
    configure_logging()

    if runtime_state is not None:
        runtime_state.set_idle("Iniciando worker…")

    logger.info(f"🚀 M4A Worker iniciado → {BACKEND_URL}")
    logger.info(f"   Whisper: {WHISPER_MODEL} / {WHISPER_DEVICE} / {COMPUTE_TYPE}")
    logger.info(f"   IA: {SUMMARY_PROVIDER} | Poll: {POLL_INTERVAL}s")

    processor = JobProcessor()

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            status = await verify_backend_connection(client)
            if runtime_state is not None:
                runtime_state.set_backend_connected(
                    "Conexión con backend OK",
                    queued_jobs=extract_queued_jobs(status),
                )
            logger.success(f"✅ Conexión con backend OK: {status}")
        except Exception as e:
            logger.error(f"❌ No se pudo conectar con el backend: {e}")
            if runtime_state is not None:
                runtime_state.set_error(f"Sin conexión con backend: {e}", connected=False)
            return 1

        while stop_event is None or not stop_event.is_set():
            try:
                job = await get_next_job(client)

                if job is None:
                    if runtime_state is not None:
                        runtime_state.set_idle("Sin jobs pendientes", queued_jobs=0)
                    logger.debug("Sin jobs pendientes, esperando…")
                    await _sleep_with_stop(stop_event, POLL_INTERVAL)
                    continue

                nota_id = job.get("nota_id")
                if not isinstance(nota_id, int):
                    logger.error(f"Job inválido recibido (sin nota_id entero): {job}")
                    if runtime_state is not None:
                        runtime_state.set_error("Job inválido recibido desde backend")
                    await _sleep_with_stop(stop_event, POLL_INTERVAL)
                    continue

                claim_result = await claim_job(client, nota_id)
                if claim_result == "already-claimed":
                    logger.warning(f"No se pudo reclamar nota_id={nota_id}, otro worker lo tomó")
                    continue
                if claim_result == "not-claimable":
                    logger.info(f"nota_id={nota_id} ya no está en estado reclamable, se omite")
                    continue

                queued_jobs = extract_queued_jobs(job)
                if queued_jobs is None:
                    queued_jobs = await _read_queue_count(client)

                if runtime_state is not None:
                    runtime_state.start_job(
                        nota_id,
                        queued_jobs=queued_jobs,
                        message=f"Procesando nota {nota_id}",
                    )

                try:
                    async def local_progress(percent: int, message: str):
                        if runtime_state is not None:
                            runtime_state.update_progress(nota_id, percent, message)

                    await processor.process_job(client, job, progress_callback=local_progress)
                    queued_after = await _read_queue_count(client)
                    if runtime_state is not None:
                        runtime_state.finish_job(
                            nota_id,
                            success=True,
                            queued_jobs=queued_after,
                            message=f"Job {nota_id} completado",
                        )
                except Exception as e:
                    logger.exception(f"Error procesando nota_id={nota_id}: {e}")
                    queued_after = await _read_queue_count(client)
                    if runtime_state is not None:
                        runtime_state.finish_job(
                            nota_id,
                            success=False,
                            queued_jobs=queued_after,
                            message=f"Error en job {nota_id}",
                        )
                        runtime_state.set_error(f"Job {nota_id}: {e}")
                    try:
                        await fail_job(client, nota_id, str(e))
                    except Exception as fail_error:
                        logger.error(
                            f"No se pudo reportar fallo para nota_id={nota_id}; estado remoto incierto: {fail_error}"
                        )

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP {e.response.status_code}: {e.request.url}")
                if runtime_state is not None:
                    runtime_state.set_error(f"HTTP {e.response.status_code}: backend no disponible", connected=False)
                await _sleep_with_stop(stop_event, POLL_INTERVAL)
            except httpx.RequestError as e:
                logger.warning(f"Error de red con backend: {type(e).__name__}: {e}")
                if runtime_state is not None:
                    runtime_state.set_error(f"Error de red: {type(e).__name__}", connected=False)
                await _sleep_with_stop(stop_event, POLL_INTERVAL)
            except Exception as e:
                logger.exception(f"Error inesperado en loop: {e}")
                if runtime_state is not None:
                    runtime_state.set_error(f"Error inesperado: {type(e).__name__}")
                await _sleep_with_stop(stop_event, POLL_INTERVAL)

    return 0
