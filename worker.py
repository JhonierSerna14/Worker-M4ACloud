"""Entrypoint del worker M4A.

Uso:
    python worker.py
"""

import asyncio

from m4a_worker.runner import run_worker


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_worker()))
