#!/usr/bin/env python3
"""
Public demo launcher for the Neraium SII FastAPI app (apps.api.main:app).

- Ensures core dependencies are installed (pip).
- Serves with uvicorn on 0.0.0.0 by default; PORT env or --port (default 7860).
- Optional --share: tries cloudflared or ngrok for a temporary public URL (if installed).
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

# ASGI app path (must be run with cwd == REPO_ROOT on sys.path)
APP_IMPORT = "apps.api.main:app"


def _ensure_repo_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def ensure_dependencies() -> None:
    """Install from requirements.txt if FastAPI/uvicorn are not importable."""
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        if not REQUIREMENTS.is_file():
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
                ]
            )
        else:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "-r", str(REQUIREMENTS)]
            )


def _parse_args() -> argparse.Namespace:
    default_port = int(os.environ.get("PORT", "7860"))
    parser = argparse.ArgumentParser(

        description="Run the Neraium API demo (FastAPI + uvicorn).",

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

        help="Port (default: PORT env or 7860)",

    )

    parser.add_argument(

        "--share",

        action="store_true",

        help="Try to open a public tunnel via cloudflared or ngrok if available",

    )

    return parser.parse_args()





def _print_access_urls(host: str, port: int) -> None:

    local = f"http://127.0.0.1:{port}"

    print()

    print("  Local API:")

    print(f"    {local}")

    print(f"    Health: {local}/health")

    print(f"    Docs:   {local}/docs")

    if host in {"0.0.0.0", "::"}:

        print()

        print("  On this machine, same as:")

        print(f"    http://localhost:{port}")

    print()





def _tunnel_worker(port: int) -> None:

    """After a short delay, start cloudflared or ngrok so the server is already listening."""

    time.sleep(2.0)

    url = f"http://127.0.0.1:{port}"

    cf = shutil.which("cloudflared")

    ng = shutil.which("ngrok")

    if cf:

        print("[demo] Starting cloudflared quick tunnel (Ctrl+C to stop)...", flush=True)

        try:

            subprocess.run([cf, "tunnel", "--url", url], check=False)

        except OSError as e:

            print(f"[demo] cloudflared failed: {e}", flush=True)

        return

    if ng:

        print("[demo] Starting ngrok (Ctrl+C to stop)...", flush=True)

        try:

            subprocess.run([ng, "http", str(port)], check=False)

        except OSError as e:

            print(f"[demo] ngrok failed: {e}", flush=True)

        return

    print(

        "[demo] --share was set but neither 'cloudflared' nor 'ngrok' was found on PATH.",

        flush=True,

    )

    print(

        "       Install one of them, or expose the port with your platform's firewall / reverse proxy.",

        flush=True,

    )





def main() -> None:

    args = _parse_args()

    os.chdir(REPO_ROOT)

    _ensure_repo_on_path()

    ensure_dependencies()



    import uvicorn  # after ensure_dependencies



    host, port = args.host, args.port



    print("=" * 60)

    print("Neraium demo - FastAPI SII API")

    print("=" * 60)

    print(f"  Working directory: {REPO_ROOT}")

    print(f"  Binding: {host}:{port}")

    _print_access_urls(host, port)



    if args.share:

        t = threading.Thread(target=_tunnel_worker, args=(port,), daemon=True)

        t.start()



    try:

        uvicorn.run(

            APP_IMPORT,

            host=host,

            port=port,

            log_level="info",

        )

    except KeyboardInterrupt:

        print("\n[demo] Shut down.")





if __name__ == "__main__":

    main()

