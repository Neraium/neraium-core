#!/usr/bin/env python3
"""Run Neraium demo using FastAPI backend + Next.js frontend.

Primary demo command:
    python run_demo.py

To run the demo UI on a custom port (e.g. http://localhost:3004):
    python run_demo.py --frontend-port 3004

This launcher is resilient to ports already in use:
- Backend: if the configured port is already serving Neraium, it reuses it; otherwise it picks the next free port.
- Frontend: if the configured port is already serving a frontend, it does not try to start a second copy.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

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


def _is_port_listening(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=0.35):
            return True
    except OSError:
        return False


def _probe_neraium_backend(port: int) -> bool:
    """Return True if localhost:{port} looks like Neraium FastAPI."""
    p = int(port)
    # Fast path: OpenAPI is cheap and includes the app title.
    openapi_url = f"http://127.0.0.1:{p}/openapi.json"
    try:
        with urlrequest.urlopen(openapi_url, timeout=0.8) as resp:
            if int(getattr(resp, "status", 0) or 0) != 200:
                return False
            body = resp.read(8192) or b""
        text = body.decode("utf-8", errors="ignore")
        if "Neraium SII API" in text:
            return True
    except (urlerror.URLError, TimeoutError, ValueError):
        pass

    # Fallback: /health is more semantically precise but can be slower depending on persistence.
    health_url = f"http://127.0.0.1:{p}/health"
    try:
        with urlrequest.urlopen(health_url, timeout=2.0) as resp:
            if int(getattr(resp, "status", 0) or 0) != 200:
                return False
            body = resp.read(8192) or b""
        text = body.decode("utf-8", errors="ignore")
        return "\"status\"" in text and ("ok" in text or "degraded" in text)
    except (urlerror.URLError, TimeoutError, ValueError):
        return False


def _find_next_free_port(start_port: int, *, host: str = "127.0.0.1", tries: int = 30) -> int:
    start = int(start_port)
    for p in range(start, start + max(1, int(tries))):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range [{start}, {start + tries})")


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

    backend_port = int(args.backend_port)
    backend_proc: subprocess.Popen[bytes] | None = None

    # Backend: reuse if Neraium already running, otherwise pick a free port.
    if _is_port_listening(backend_port):
        if _probe_neraium_backend(backend_port):
            print(f"Reusing existing Neraium backend at http://localhost:{backend_port}")
        else:
            new_port = _find_next_free_port(backend_port + 1)
            print(f"Backend port {backend_port} is in use; switching to {new_port}.")
            backend_port = int(new_port)

    if not _is_port_listening(backend_port):
        backend_env = os.environ.copy()
        backend_env["PORT"] = str(backend_port)
        backend_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "apps.api.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(backend_port),
            ],
            env=backend_env,
        )

    frontend_port = int(args.frontend_port)
    frontend_proc: subprocess.Popen[bytes] | None = None

    try:
        if args.backend_only:
            print(f"Backend running at http://localhost:{backend_port}")
            print("Press Ctrl+C to stop.")
            if backend_proc is not None:
                backend_proc.wait()
            else:
                while True:
                    time.sleep(1.0)
            return

        # Frontend: if already running, don't start a second instance.
        if _is_port_listening(frontend_port):
            print(f"Frontend already running at http://localhost:{frontend_port}")
            print(f"Backend:  http://localhost:{backend_port}")
            print("Press Ctrl+C to stop.")
            while True:
                if backend_proc is not None and backend_proc.poll() is not None:
                    raise RuntimeError(f"Backend process exited unexpectedly (code={backend_proc.returncode}).")
                time.sleep(0.5)

        _ensure_frontend_dependencies()
        frontend_env = os.environ.copy()
        frontend_env["NEXT_PUBLIC_NERAIUM_API_BASE"] = f"http://localhost:{backend_port}"

        npm = shutil.which("npm") or "npm"
        frontend_proc = subprocess.Popen(
            [npm, "run", "dev", "--", "-p", str(frontend_port)],
            cwd=str(FRONTEND_DIR),
            env=frontend_env,
        )

        print(f"Backend:  http://localhost:{backend_port}")
        print(f"Frontend: http://localhost:{frontend_port}")
        print("Press Ctrl+C to stop.")

        while True:
            if backend_proc is not None and backend_proc.poll() is not None:
                raise RuntimeError(f"Backend process exited unexpectedly (code={backend_proc.returncode}).")
            if frontend_proc is not None and frontend_proc.poll() is not None:
                raise RuntimeError(f"Frontend process exited unexpectedly (code={frontend_proc.returncode}).")
            time.sleep(0.5)

    except KeyboardInterrupt:
        pass
    finally:
        for proc in [frontend_proc, backend_proc]:
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    main()
