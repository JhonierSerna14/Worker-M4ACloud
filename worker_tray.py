from __future__ import annotations

import asyncio
import os
import signal
import threading
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pystray import Icon, Menu, MenuItem

from m4a_worker.config import LOG_PATH
from m4a_worker.runner import run_worker
from m4a_worker.runtime_state import WorkerRuntimeState

PROJECT_ROOT = Path(__file__).resolve().parent


def _load_font(size: int) -> ImageFont.ImageFont:
    windir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for name in ("segoeuib.ttf", "seguisb.ttf", "arialbd.ttf", "segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(str(windir / name), size=size)
        except Exception:
            continue
    return ImageFont.load_default()


class WorkerTrayApp:
    def __init__(self):
        self.runtime_state = WorkerRuntimeState()
        self.stop_event = threading.Event()
        self._shutdown_requested = threading.Event()
        self.icon = Icon(
            "m4a-worker",
            self._render_icon(),
            "M4A Worker",
            menu=Menu(
                MenuItem("Abrir log", self._open_log),
                MenuItem("Abrir carpeta", self._open_project_folder),
                MenuItem("Salir", self._quit),
            ),
        )
        self.worker_thread = threading.Thread(target=self._run_worker, name="worker-thread", daemon=True)
        self.ui_thread = threading.Thread(target=self._refresh_icon_loop, name="tray-refresh", daemon=True)

    def _run_worker(self):
        asyncio.run(run_worker(runtime_state=self.runtime_state, stop_event=self.stop_event))

    def _tooltip(self) -> str:
        snapshot = self.runtime_state.snapshot()
        status = (snapshot.status_message or "M4A Worker").replace("\n", " ").strip()
        tooltip = (
            f"M4A Worker\n"
            f"Pendientes: {snapshot.displayed_jobs}\n"
            f"Progreso actual: {snapshot.current_job_progress}%\n"
            f"{status}"
        )
        max_len = 127
        if len(tooltip) > max_len:
            return tooltip[: max_len - 3] + "..."
        return tooltip

    def _status_colors(self):
        snapshot = self.runtime_state.snapshot()
        if snapshot.last_error:
            return (220, 53, 69, 255), (248, 215, 218, 255), (127, 29, 29, 255)
        if snapshot.processing:
            return (14, 116, 144, 255), (186, 230, 253, 255), (15, 23, 42, 255)
        if snapshot.connected:
            return (22, 163, 74, 255), (220, 252, 231, 255), (20, 83, 45, 255)
        return (107, 114, 128, 255), (229, 231, 235, 255), (31, 41, 55, 255)

    def _render_icon(self) -> Image.Image:
        snapshot = self.runtime_state.snapshot()
        progress = max(0, min(snapshot.current_job_progress, 100))
        count = str(min(snapshot.displayed_jobs, 99))
        ring_color, track_color, text_color = self._status_colors()

        image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        bbox = (6, 6, 58, 58)

        draw.arc(bbox, start=0, end=359, fill=track_color, width=8)
        if snapshot.processing and progress > 0:
            draw.arc(bbox, start=-90, end=(-90 + int(360 * (progress / 100))), fill=ring_color, width=8)
        elif not snapshot.processing:
            draw.arc(bbox, start=-90, end=269, fill=ring_color, width=8)

        draw.ellipse((14, 14, 50, 50), fill=(255, 255, 255, 255))

        font = _load_font(28 if len(count) == 1 else 22)
        text_bbox = draw.textbbox((0, 0), count, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (64 - text_width) / 2 - text_bbox[0]
        text_y = (64 - text_height) / 2 - text_bbox[1]
        draw.text((text_x, text_y), count, fill=text_color, font=font)

        return image

    def _refresh_icon_loop(self):
        while not self.stop_event.is_set():
            self.icon.icon = self._render_icon()
            self.icon.title = self._tooltip()
            time.sleep(0.5)

    def _open_log(self, icon, item):
        if LOG_PATH.exists():
            os.startfile(str(LOG_PATH))
        else:
            os.startfile(str(PROJECT_ROOT))

    def _open_project_folder(self, icon, item):
        os.startfile(str(PROJECT_ROOT))

    def _quit(self, icon, item):
        self.request_shutdown()

    def request_shutdown(self):
        if self._shutdown_requested.is_set():
            return
        self._shutdown_requested.set()
        self.stop_event.set()
        try:
            self.icon.stop()
        except Exception:
            pass

    def _handle_signal(self, signum, frame):
        # Evita que Ctrl+C dispare KeyboardInterrupt dentro del callback Win32.
        self.request_shutdown()

    def _install_signal_handlers(self):
        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is not None:
                signal.signal(sig, self._handle_signal)

    def run(self):
        self._install_signal_handlers()
        self.worker_thread.start()
        self.ui_thread.start()
        try:
            self.icon.run()
        except KeyboardInterrupt:
            self.request_shutdown()
        finally:
            self.request_shutdown()
            self.worker_thread.join(timeout=5.0)
            self.ui_thread.join(timeout=2.0)


if __name__ == "__main__":
    WorkerTrayApp().run()