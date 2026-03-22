#!/usr/bin/env python3
"""
One-command demo launcher for neraium-core.

Installs dependencies if needed, starts the FastAPI API + MVP web app,
binds to 0.0.0.0, and optionally creates a public tunnel for sharing.

Usage:
  python run_demo.py
  python run_demo.py --port 7860
  python run_demo.py --share
  python run_demo.py --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import os
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
