#!/usr/bin/env python3
"""Run Neraium demo using FastAPI backend + Next.js frontend.

Primary demo command:
    python run_demo.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = REPO_ROOT / "frontend"


def _ensure_repo_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def _ensure_backend_dependencies() -> None:
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        _run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_ROOT)])


def _ensure_frontend_dependencies() -> None:
    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("npm is required to run the Next.js frontend. Install Node.js 18+.")
    if not (FRONTEND_DIR / "node_modules").exists():
        _run([npm, "install"], cwd=FRONTEND_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Neraium demo (FastAPI + Next.js)")
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--frontend-port", type=int, default=3000)
    parser.add_argument("--backend-only", action="store_true")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    _ensure_repo_on_path()
    _ensure_backend_dependencies()

    print("=" * 70)
    print("Neraium Demo — FastAPI + Next.js")
    print("=" * 70)

    backend_env = os.environ.copy()
    backend_env["PORT"] = str(args.backend_port)

    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", str(args.backend_port)],
        env=backend_env,
    )
    frontend_proc: subprocess.Popen[bytes] | None = None

    try:
        if args.backend_only:
            print(f"Backend running at http://localhost:{args.backend_port}")
            print("Press Ctrl+C to stop.")
            backend_proc.wait()
            return

        _ensure_frontend_dependencies()
        frontend_env = os.environ.copy()
        frontend_env["NEXT_PUBLIC_NERAIUM_API_BASE"] = f"http://localhost:{args.backend_port}"

        npm = shutil.which("npm") or "npm"
        frontend_proc = subprocess.Popen([npm, "run", "dev", "--", "-p", str(args.frontend_port)], cwd=str(FRONTEND_DIR), env=frontend_env)

        print(f"Backend:  http://localhost:{args.backend_port}")
        print(f"Frontend: http://localhost:{args.frontend_port}")
        print("Press Ctrl+C to stop.")

        while True:
            if backend_proc.poll() is not None:
                raise RuntimeError("Backend process exited unexpectedly.")
            if frontend_proc and frontend_proc.poll() is not None:
                raise RuntimeError("Frontend process exited unexpectedly.")
            time.sleep(0.5)

    except KeyboardInterrupt:
        pass
    finally:
        for proc in [frontend_proc, backend_proc]:
            if proc and proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    main()
