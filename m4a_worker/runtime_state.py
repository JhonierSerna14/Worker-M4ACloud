from __future__ import annotations

from dataclasses import dataclass, replace
from threading import Lock


@dataclass(frozen=True)
class WorkerSnapshot:
    connected: bool = False
    processing: bool = False
    current_job_progress: int = 0
    queued_jobs: int = 0
    active_job_id: int | None = None
    status_message: str = "Iniciando worker..."
    last_error: str = ""

    @property
    def displayed_jobs(self) -> int:
        return max(self.queued_jobs + (1 if self.processing else 0), 0)


class WorkerRuntimeState:
    def __init__(self):
        self._lock = Lock()
        self._snapshot = WorkerSnapshot()

    def snapshot(self) -> WorkerSnapshot:
        with self._lock:
            return self._snapshot

    def _update(self, **changes):
        with self._lock:
            self._snapshot = replace(self._snapshot, **changes)

    def set_backend_connected(self, message: str, queued_jobs: int | None = None):
        changes = {"connected": True, "status_message": message, "last_error": ""}
        if queued_jobs is not None:
            changes["queued_jobs"] = max(queued_jobs, 0)
        self._update(**changes)

    def set_idle(self, message: str = "Sin jobs pendientes", queued_jobs: int | None = None):
        changes = {
            "processing": False,
            "current_job_progress": 0,
            "active_job_id": None,
            "status_message": message,
        }
        if queued_jobs is not None:
            changes["queued_jobs"] = max(queued_jobs, 0)
        self._update(**changes)

    def start_job(self, nota_id: int, queued_jobs: int | None = None, message: str = "Procesando job"):
        changes = {
            "connected": True,
            "processing": True,
            "current_job_progress": 0,
            "active_job_id": nota_id,
            "status_message": message,
            "last_error": "",
        }
        if queued_jobs is not None:
            changes["queued_jobs"] = max(queued_jobs, 0)
        self._update(**changes)

    def update_progress(self, nota_id: int, percent: int, message: str):
        snapshot = self.snapshot()
        if snapshot.active_job_id is not None and snapshot.active_job_id != nota_id:
            return
        self._update(
            connected=True,
            processing=True,
            active_job_id=nota_id,
            current_job_progress=max(0, min(percent, 100)),
            status_message=message,
            last_error="",
        )

    def finish_job(
        self,
        nota_id: int,
        success: bool,
        queued_jobs: int | None = None,
        message: str | None = None,
    ):
        final_message = message or (f"Job {nota_id} completado" if success else f"Job {nota_id} fallido")
        changes = {
            "processing": False,
            "current_job_progress": 0,
            "active_job_id": None,
            "status_message": final_message,
        }
        if queued_jobs is not None:
            changes["queued_jobs"] = max(queued_jobs, 0)
        self._update(**changes)

    def set_queue_count(self, queued_jobs: int):
        self._update(queued_jobs=max(queued_jobs, 0))

    def set_error(self, message: str, connected: bool | None = None):
        changes = {"status_message": message, "last_error": message}
        if connected is not None:
            changes["connected"] = connected
        self._update(**changes)