#!/usr/bin/env python3
"""
Launch the Neraium stack with uvicorn: REST API + MVP web UI in one process.

There is no separate npm/React dev server. The browser app lives under
apps/api/static/ (index.html, app.js, styles.css) and is mounted by FastAPI
(apps/api/web.py) at /, /dashboard, /upload, etc. Same origin as the API.

- Dependency install via pip if fastapi/uvicorn/multipart are missing.
- Binds 0.0.0.0 by default; port from PORT / WEB_PORT env or --port (default 7860).
- Optional --share: cloudflared or ngrok quick tunnel.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
REQUIREMENTS = REPO_ROOT / "requirements.txt"

# FastAPI ASGI app: apps/api/main.py defines `app = create_app()`
APP_IMPORT = "apps.api.main:app"

# Match apps/api/main.py default for large CSV uploads (see DEFAULT_UVICORN_H11_* there)
_DEFAULT_H11_MAX = 64 * 1024 * 1024


def _ensure_repo_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def ensure_dependencies() -> None:
    """Install from requirements.txt if any runtime deps are missing."""
    need_install = False
    try:
        import fastapi  # noqa: F401
    except ImportError:
        need_install = True
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        need_install = True
    # FastAPI form/file routes require python-multipart (import name: multipart)
    try:
        import multipart  # noqa: F401
    except ImportError:
        need_install = True

    if not need_install:
        return

    if REQUIREMENTS.is_file():
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(REQUIREMENTS)]
        )
    else:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "numpy",
                "pydantic>=2",
                "fastapi>=0.110",
                "uvicorn[standard]>=0.29",
                "python-multipart",
            ]
        )


def _parse_args() -> argparse.Namespace:
    # WEB_PORT alias: same server serves UI + API; one port for both.
    default_port = int(os.environ.get("PORT") or os.environ.get("WEB_PORT") or "7860")
    parser = argparse.ArgumentParser(
        description="Run Neraium: MVP web UI + REST API (one FastAPI/uvicorn server).",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("NERAIUM_DEMO_HOST", "0.0.0.0"),
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help="Port for web UI + API (default: PORT or WEB_PORT env, else 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Try cloudflared or ngrok quick tunnel (if installed)",
    )
    return parser.parse_args()


def _h11_max_incomplete_event_size() -> int:
    raw = os.getenv("NERAIUM_UVICORN_H11_MAX_INCOMPLETE_EVENT_SIZE")
    if not raw:
        return _DEFAULT_H11_MAX
    try:
        return max(_DEFAULT_H11_MAX, int(raw))
    except (TypeError, ValueError):
        return _DEFAULT_H11_MAX


def _print_urls(host: str, port: int) -> None:
    base_local = f"http://127.0.0.1:{port}"
    print()
    print("  --- Web app (MVP UI) ---")
    print(f"    Home / dashboard:    {base_local}/")
    print(f"    Dashboard:           {base_local}/dashboard")
    print(f"    Upload:              {base_local}/upload")
    print()
    print("  --- API (same server) ---")
    print(f"    OpenAPI / Swagger:   {base_local}/docs")
    print(f"    ReDoc:               {base_local}/redoc")
    print(f"    Health:              {base_local}/health")
    print()
    if host in {"0.0.0.0", "::"}:
        print(f"  Localhost alias:       http://localhost:{port}/")
        print("  Other devices (LAN): http://<this-machine-ip>:%d/" % port)
    else:
        print(f"  Bound host UI:         http://{host}:{port}/")
    print()
    print("  Static UI files: apps/api/static/index.html (see apps/api/web.py)")
    print()


def _tunnel_worker(port: int) -> None:
    time.sleep(2.0)
    url = f"http://127.0.0.1:{port}"
    cf = shutil.which("cloudflared")
    ng = shutil.which("ngrok")
    if cf:
        print("[demo] Starting cloudflared (Ctrl+C stops server and tunnel)...", flush=True)
        try:
            subprocess.run([cf, "tunnel", "--url", url], check=False)
        except OSError as e:
            print(f"[demo] cloudflared failed: {e}", flush=True)
        return
    if ng:
        print("[demo] Starting ngrok (Ctrl+C stops server and tunnel)...", flush=True)
        try:
            subprocess.run([ng, "http", str(port)], check=False)
        except OSError as e:
            print(f"[demo] ngrok failed: {e}", flush=True)
        return
    print(
        "[demo] --share set but neither cloudflared nor ngrok found on PATH.",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    os.chdir(REPO_ROOT)
    _ensure_repo_on_path()
    ensure_dependencies()

    import uvicorn  # after ensure_dependencies

    host, port = args.host, args.port
    h11 = _h11_max_incomplete_event_size()

    print("=" * 60)
    print("Neraium demo - Web UI + API (single uvicorn process)")
    print("=" * 60)
    print(f"  Working directory: {REPO_ROOT}")
    print(f"  ASGI app:          {APP_IMPORT}")
    print(f"  Binding:           {host}:{port}")
    _print_urls(host, port)

    if args.share:
        threading.Thread(target=_tunnel_worker, args=(port,), daemon=True).start()

    try:
        uvicorn.run(
            APP_IMPORT,
            host=host,
            port=port,
            log_level="info",
            h11_max_incomplete_event_size=h11,
        )
    except KeyboardInterrupt:
        print("\n[demo] Shut down.")


if __name__ == "__main__":
    main()
