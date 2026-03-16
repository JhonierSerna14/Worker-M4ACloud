import httpx
from loguru import logger
import asyncio

from typing import Literal

from .config import API_BASE, WORKER_HEADERS

QUEUE_COUNT_KEYS = {
    "queued_jobs",
    "queue_jobs",
    "queue_size",
    "queue_count",
    "jobs_queued",
    "jobs_in_queue",
    "jobs_pending",
    "pending_jobs",
    "pending_count",
    "remaining_jobs",
    "remaining_count",
    "queue_remaining",
}


def _coerce_non_negative_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float) and value.is_integer():
        return max(int(value), 0)
    if isinstance(value, str) and value.strip().isdigit():
        return max(int(value.strip()), 0)
    return None


def extract_queued_jobs(payload: dict | None) -> int | None:
    if not isinstance(payload, dict):
        return None

    def _walk(node) -> int | None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key.lower() in QUEUE_COUNT_KEYS:
                    count = _coerce_non_negative_int(value)
                    if count is not None:
                        return count
                nested = _walk(value)
                if nested is not None:
                    return nested
        elif isinstance(node, list):
            for item in node:
                nested = _walk(item)
                if nested is not None:
                    return nested
        return None

    return _walk(payload)


async def get_next_job(client: httpx.AsyncClient) -> dict | None:
    resp = await client.get(f"{API_BASE}/worker/jobs/next", headers=WORKER_HEADERS)
    if resp.status_code == 204:
        return None
    resp.raise_for_status()
    payload = resp.json()
    if payload is None:
        return None
    if not isinstance(payload, dict):
        logger.warning(f"Payload inesperado en get_next_job: {type(payload).__name__}")
        return None
    return payload


async def claim_job(client: httpx.AsyncClient, nota_id: int) -> Literal["claimed", "already-claimed", "not-claimable"]:
    resp = await client.post(f"{API_BASE}/worker/jobs/{nota_id}/claim", headers=WORKER_HEADERS)
    if resp.status_code == 200:
        return "claimed"
    if resp.status_code == 409:
        return "already-claimed"
    if resp.status_code in {400, 404}:
        return "not-claimable"

    resp.raise_for_status()
    return "not-claimable"


async def send_progress(client: httpx.AsyncClient, nota_id: int, percent: int, message: str):
    try:
        await client.post(
            f"{API_BASE}/worker/jobs/{nota_id}/progress",
            headers=WORKER_HEADERS,
            json={"percent": percent, "message": message},
        )
    except Exception as e:
        logger.warning(f"No se pudo enviar progreso: {e}")


async def complete_job(
    client: httpx.AsyncClient,
    nota_id: int,
    html: str,
    transcript: str | None,
    duration: float | None,
    language: str | None,
):
    resp = await client.post(
        f"{API_BASE}/worker/jobs/{nota_id}/complete",
        headers=WORKER_HEADERS,
        json={
            "html": html,
            "transcript_text": transcript,
            "duration_seconds": duration,
            "language": language,
        },
        timeout=30.0,
    )
    resp.raise_for_status()


async def fail_job(client: httpx.AsyncClient, nota_id: int, error: str):
    last_exception: Exception | None = None
    for attempt in range(1, 4):
        try:
            resp = await client.post(
                f"{API_BASE}/worker/jobs/{nota_id}/fail",
                headers=WORKER_HEADERS,
                json={"error": error},
                timeout=20.0,
            )
            resp.raise_for_status()
            return
        except Exception as exc:
            last_exception = exc
            logger.warning(
                f"No se pudo reportar fail para nota_id={nota_id} (intento {attempt}/3): {exc}"
            )
            if attempt < 3:
                await asyncio.sleep(attempt)

    if last_exception is not None:
        raise last_exception


async def verify_backend_connection(client: httpx.AsyncClient) -> dict:
    status_resp = await client.get(f"{API_BASE}/worker/status", headers=WORKER_HEADERS)
    status_resp.raise_for_status()
    return status_resp.json()
