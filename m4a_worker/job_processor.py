import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from .ai_summary import AIServiceUnavailable, clean_html, summarize
from .backend_api import complete_job, mark_retry, save_transcript, send_progress
from .transcription import transcribe_file_with_progress


@dataclass(frozen=True)
class JobData:
    nota_id: int
    audio_url: str | None
    transcript_url: str | None
    materia: str
    is_reprocess: bool


class JobProcessor:
    async def process_job(
        self,
        client: httpx.AsyncClient,
        job: dict[str, Any],
        progress_callback: Callable[[int, str], Awaitable[None]] | None = None,
    ):
        data = JobData(
            nota_id=job["nota_id"],
            audio_url=job.get("audio_download_url"),
            transcript_url=job.get("transcript_download_url"),
            materia=job.get("materia_nombre", ""),
            is_reprocess=job.get("is_reprocess", False),
        )

        self._validate_job(data)

        logger.info(
            f"📥 Procesando job nota_id={data.nota_id} | materia='{data.materia}' | reprocess={data.is_reprocess}"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            transcript: str
            duration: float | None
            language: str | None

            if data.is_reprocess:
                transcript = await self._download_transcript(client, data, progress_callback=progress_callback)
                duration = None
                language = None
            else:
                audio_path = Path(tmpdir) / "audio.tmp"
                await self._download_audio(client, data, audio_path, progress_callback=progress_callback)
                transcript, duration, language = await self._transcribe(
                    client,
                    data,
                    audio_path,
                    progress_callback=progress_callback,
                )
                await self._persist_transcript(
                    client,
                    data,
                    transcript,
                    duration,
                    language,
                    progress_callback=progress_callback,
                )

            try:
                html = await self._summarize(client, data, transcript, progress_callback=progress_callback)
            except AIServiceUnavailable as exc:
                await self._mark_retry(client, data, str(exc), progress_callback=progress_callback)
                logger.warning(f"⚠️ Job nota_id={data.nota_id} quedó en retry manual tras agotar IA")
                return

            await self._complete(
                client,
                data,
                html,
                None,
                duration,
                language,
                progress_callback=progress_callback,
            )

        logger.success(f"✅ Job nota_id={data.nota_id} completado")

    async def _emit_progress(
        self,
        client: httpx.AsyncClient,
        nota_id: int,
        percent: int,
        message: str,
        progress_callback: Callable[[int, str], Awaitable[None]] | None = None,
    ):
        await send_progress(client, nota_id, percent, message)
        if progress_callback is not None:
            await progress_callback(percent, message)

    async def _persist_transcript(
        self,
        client: httpx.AsyncClient,
        data: JobData,
        transcript: str,
        duration: float | None,
        language: str | None,
        progress_callback: Callable[[int, str], Awaitable[None]] | None = None,
    ):
        await self._emit_progress(client, data.nota_id, 55, "Guardando transcripción...", progress_callback)
        await save_transcript(client, data.nota_id, transcript, duration, language)
        logger.info(f"Transcripción persistida para nota_id={data.nota_id} ({len(transcript)} chars)")

    async def _mark_retry(
        self,
        client: httpx.AsyncClient,
        data: JobData,
        error: str,
        progress_callback: Callable[[int, str], Awaitable[None]] | None = None,
    ):
        await self._emit_progress(client, data.nota_id, 100, "IA agotada. Queda listo para reintento manual.", progress_callback)
        await mark_retry(client, data.nota_id, error)

    async def _download_audio(
        self,
        client: httpx.AsyncClient,
        data: JobData,
        audio_path: Path,
        progress_callback: Callable[[int, str], Awaitable[None]] | None = None,
    ):
        if not data.audio_url:
            raise ValueError("No hay audio_download_url para este job")

        await self._emit_progress(client, data.nota_id, 0, "Descargando audio…", progress_callback)
        async with client.stream("GET", data.audio_url) as resp:
            resp.raise_for_status()
            with open(audio_path, "wb") as file_handle:
                async for chunk in resp.aiter_bytes(64 * 1024):
                    file_handle.write(chunk)

        logger.debug(f"Audio descargado: {audio_path.stat().st_size / 1024:.1f} KB")

    async def _download_transcript(
        self,
        client: httpx.AsyncClient,
        data: JobData,
        progress_callback: Callable[[int, str], Awaitable[None]] | None = None,
    ) -> str:
        if not data.transcript_url:
            raise ValueError("No hay transcript_download_url para reprocesar")

        await self._emit_progress(
            client,
            data.nota_id,
            1,
            "Descargando transcripción guardada…",
            progress_callback,
        )
        resp = await client.get(data.transcript_url, timeout=60.0)
        resp.raise_for_status()

        transcript = resp.text.strip()
        if not transcript:
            raise ValueError("La transcripción para reprocesar está vacía")

        logger.info(f"Reprocess: transcripción descargada ({len(transcript)} chars)")
        return transcript

    async def _transcribe(
        self,
        client: httpx.AsyncClient,
        data: JobData,
        audio_path: Path,
        progress_callback: Callable[[int, str], Awaitable[None]] | None = None,
    ) -> tuple[str, float, str]:
        await self._emit_progress(client, data.nota_id, 1, "Iniciando transcripción con Whisper…", progress_callback)
        transcript, duration, language = await transcribe_file_with_progress(
            client,
            data.nota_id,
            str(audio_path),
            progress_callback=progress_callback,
        )
        logger.info(f"Transcripción: {len(transcript)} chars | {duration:.1f}s | lang={language}")
        return transcript, duration, language

    async def _summarize(
        self,
        client: httpx.AsyncClient,
        data: JobData,
        transcript: str,
        progress_callback: Callable[[int, str], Awaitable[None]] | None = None,
    ) -> str:
        await self._emit_progress(client, data.nota_id, 60, "Generando resumen con IA", progress_callback)

        async def ai_progress(percent: float, message: str):
            mapped = min(99, max(60, int(60 + (percent / 100) * 39)))
            await self._emit_progress(client, data.nota_id, mapped, f"IA: {message}", progress_callback)

        html = await summarize(transcript, data.materia, progress_callback=ai_progress)
        return clean_html(html)

    async def _complete(
        self,
        client: httpx.AsyncClient,
        data: JobData,
        html: str,
        transcript: str | None,
        duration: float | None,
        language: str | None,
        progress_callback: Callable[[int, str], Awaitable[None]] | None = None,
    ):
        await self._emit_progress(
            client,
            data.nota_id,
            99,
            "Resumen generado. Guardando resultado…",
            progress_callback,
        )
        duration_seconds = int(duration) if duration is not None else None
        await complete_job(client, data.nota_id, html, transcript, duration_seconds, language)

    def _validate_job(self, data: JobData):
        if data.is_reprocess and not data.transcript_url:
            raise ValueError("Job inválido: is_reprocess=true pero falta transcript_download_url")
        if not data.is_reprocess and not data.audio_url:
            raise ValueError("Job inválido: falta audio_download_url para transcripción")
