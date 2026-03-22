#!/usr/bin/env python3
"""
<<<<<<< HEAD
Public demo launcher for the Neraium SII FastAPI app (apps.api.main:app).

- Ensures core dependencies are installed (pip).
- Serves with uvicorn on 0.0.0.0 by default; PORT env or --port (default 7860).
- Optional --share: tries cloudflared or ngrok for a temporary public URL (if installed).
=======
One-command demo launcher for neraium-core.

Installs dependencies if needed, starts the FastAPI API + MVP web app,
binds to 0.0.0.0, and optionally creates a public tunnel for sharing.

Usage:
  python run_demo.py
  python run_demo.py --port 7860
  python run_demo.py --share
  python run_demo.py --host 127.0.0.1 --port 8000
>>>>>>> 60fa49889f50cad2076a616030c6ce50a645ad07
"""

from __future__ import annotations

import argparse
import os
<<<<<<< HEAD
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

=======
import subprocess
import sys


def _ensure_deps() -> None:
    """Install required packages via pip if not already available."""
    for pkg in ["fastapi", "uvicorn"]:
        try:
            __import__(pkg)
        except ImportError:
            break
    else:
        return  # All key deps present

    script_dir = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        cwd=script_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        required = ["numpy", "pydantic>=2", "fastapi>=0.110", "uvicorn>=0.29", "python-multipart"]
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install"] + required,
            cwd=script_dir,
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        print("Failed to install dependencies:", result.stderr or result.stdout, file=sys.stderr)
        sys.exit(1)


def _print_urls(host: str, port: int, share_url: str | None = None) -> None:
    host_display = "localhost" if host == "0.0.0.0" else host
    print()
    print("=" * 60)
    print("  Neraium SII API + MVP Web App")
    print("=" * 60)
    print()
    print(f"  Local:   http://{host_display}:{port}/")
    print(f"  API:     http://{host_display}:{port}/health")
    if host == "0.0.0.0":
        print(f"  Network: http://<this-machine-ip>:{port}/")
    if share_url:
        print(f"  Public:  {share_url}")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    print()


def _spawn_tunnel(port: int) -> str | None:
    """Try ngrok or cloudflared to create a public URL. Returns URL or None."""
    # Try ngrok first (runs in background)
    try:
        subprocess.Popen(
            ["ngrok", "http", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print("  Started ngrok tunnel. Check ngrok dashboard for public URL.")
        return "https://<ngrok-url> (see ngrok output)"
    except FileNotFoundError:
        pass

    # Try cloudflared (runs in background)
    try:
        subprocess.Popen(
            ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print("  Started cloudflared tunnel. Check output for public URL.")
        return "https://<cloudflared-url> (see cloudflared output)"
    except FileNotFoundError:
        pass

    print(
        "  No tunnel tool found. Install 'ngrok' or 'cloudflared' for --share."
    )
    return None


def main() -> int:
    # Run from project root so apps.api and neraium_core resolve
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir and os.getcwd() != script_dir:
        os.chdir(script_dir)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

    parser = argparse.ArgumentParser(description="Run Neraium demo (API + web UI)")
    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "0.0.0.0"),
        help="Bind host (default: 0.0.0.0, or HOST env)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Bind port (default: 8000, or PORT env)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public tunnel (ngrok or cloudflared) for sharing",
    )
    args = parser.parse_args()

    _ensure_deps()

    os.environ["HOST"] = args.host
    os.environ["PORT"] = str(args.port)

    share_url: str | None = None
    if args.share:
        share_url = _spawn_tunnel(args.port)

    _print_urls(args.host, args.port, share_url)

    import uvicorn

    # Match main.py's h11 parser limit for large CSV uploads
    h11_limit = 64 * 1024 * 1024
    try:
        raw = os.getenv("NERAIUM_UVICORN_H11_MAX_INCOMPLETE_EVENT_SIZE")
        if raw:
            h11_limit = max(h11_limit, int(raw))
    except (TypeError, ValueError):
        pass

    uvicorn.run(
        "apps.api.main:app",
        host=args.host,
        port=args.port,
        log_level="info",
        h11_max_incomplete_event_size=h11_limit,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
>>>>>>> 60fa49889f50cad2076a616030c6ce50a645ad07
