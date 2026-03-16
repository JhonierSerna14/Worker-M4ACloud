from __future__ import annotations

import sys
from pathlib import Path


def _startup_folder() -> Path:
    appdata = Path.home() / "AppData" / "Roaming"
    return appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def _pythonw_path(project_root: Path) -> Path:
    candidates = [
        project_root / ".venv" / "Scripts" / "pythonw.exe",
        Path(sys.executable).with_name("pythonw.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No se encontró pythonw.exe para lanzar el worker en segundo plano")


def main():
    project_root = Path(__file__).resolve().parent
    startup_dir = _startup_folder()
    startup_dir.mkdir(parents=True, exist_ok=True)

    pythonw = _pythonw_path(project_root)
    tray_script = project_root / "worker_tray.py"
    launcher_path = startup_dir / "M4A Worker Tray.vbs"

    launcher_contents = f'''Set shell = CreateObject("WScript.Shell")
shell.Run Chr(34) & "{pythonw}" & Chr(34) & " " & Chr(34) & "{tray_script}" & Chr(34), 0, False
'''
    launcher_path.write_text(launcher_contents, encoding="utf-8")

    print(f"Autoarranque instalado en: {launcher_path}")


if __name__ == "__main__":
    main()