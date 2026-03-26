#!/usr/bin/env python3
"""
Launch the Neraium stack with uvicorn: REST API + MVP web UI in one process.

There is no separate npm/React dev server. The browser app lives under
apps/api/static/ (index.html, app.js, styles.css) and is mounted by FastAPI
(apps/api/web.py) at /, /dashboard, /upload, etc. Same origin as the API.

- Dependency install via pip if fastapi/uvicorn/multipart are missing.
- Uses pyproject.toml as the single Python dependency source of truth.
- Binds 0.0.0.0 by default; port from PORT / WEB_PORT env or --port (default 7860).
- Optional --share: cloudflared or ngrok quick tunnel.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
# FastAPI ASGI app: apps/api/main.py defines `app = create_app()`
APP_IMPORT = "apps.api.main:app"

# Match apps/api/main.py default for large CSV uploads (see DEFAULT_UVICORN_H11_* there)
_DEFAULT_H11_MAX = 64 * 1024 * 1024


def _ensure_repo_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def ensure_dependencies() -> None:
    """Install from pyproject.toml if required runtime deps are missing."""
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

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_ROOT)])


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
    print(f"    three-init (module): {base_local}/web/three-init.mjs  (Three.js loads from jsDelivr CDN)")
    print()
    if host in {"0.0.0.0", "::"}:
        print(f"  Localhost alias:       http://localhost:{port}/")
        print("  Other devices (LAN): http://<this-machine-ip>:%d/" % port)
    else:
        print(f"  Bound host UI:         http://{host}:{port}/")
    print()
    print("  Static UI files: apps/api/static/index.html (see apps/api/web.py)")
    print()


def _print_share_banner(public_url: str, local_port: int) -> None:
    dash = "/dashboard?demo=1&prepare=1"
    demo_url = public_url.rstrip("/") + dash
    print()
    print("  " + "=" * 56)
    print("  SHARE THIS URL (works while this process is running)")
    print("  " + "=" * 56)
    print(f"    Public base:  {public_url}")
    print(f"    Demo deep link (auto-prepares runs):")
    print(f"    {demo_url}")
    print("  " + "=" * 56)
    print(f"    Local same UI: http://127.0.0.1:{local_port}{dash}")
    print()


def _extract_public_url_from_line(line: str) -> str | None:
    line = line.strip()
    m = re.search(r"https://[a-zA-Z0-9][-a-zA-Z0-9.]*\.trycloudflare\.com/?", line)
    if m:
        return m.group(0).rstrip("/")
    m = re.search(r"https://[a-zA-Z0-9][-a-zA-Z0-9.]*\.ngrok(?:-free)?\.app/?", line)
    if m:
        return m.group(0).rstrip("/")
    m = re.search(r"https://[a-zA-Z0-9][-a-zA-Z0-9.]*\.ngrok\.io/?", line)
    if m:
        return m.group(0).rstrip("/")
    return None


def _stream_tunnel_output(proc: subprocess.Popen, label: str) -> None:
    """Read tunnel stdout; print lines and stop when a public HTTPS URL is found."""
    assert proc.stdout is not None
    public_url: str | None = None
    try:
        for raw in iter(proc.stdout.readline, ""):
            if not raw:
                break
            sys.stdout.write(raw)
            sys.stdout.flush()
            if public_url is None:
                found = _extract_public_url_from_line(raw)
                if found:
                    public_url = found
                    port = int(os.environ.get("_NERAIUM_TUNNEL_LOCAL_PORT", "7860"))
                    _print_share_banner(found, port)
    except Exception as exc:  # noqa: BLE001
        print(f"[demo] {label} output reader: {exc}", flush=True)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass


def _tunnel_worker(port: int) -> None:
    time.sleep(1.2)
    os.environ["_NERAIUM_TUNNEL_LOCAL_PORT"] = str(port)
    local = f"http://127.0.0.1:{port}"
    cf = shutil.which("cloudflared")
    ng = shutil.which("ngrok")
    if cf:
        print("[demo] Starting cloudflared quick tunnel (Ctrl+C stops server + tunnel)...", flush=True)
        print("[demo] Tip: install from https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/", flush=True)
        try:
            proc = subprocess.Popen(
                [cf, "tunnel", "--url", local],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            t = threading.Thread(target=_stream_tunnel_output, args=(proc, "cloudflared"), daemon=True)
            t.start()
            proc.wait()
        except OSError as e:
            print(f"[demo] cloudflared failed: {e}", flush=True)
        return
    if ng:
        print("[demo] Starting ngrok (Ctrl+C stops server + tunnel)...", flush=True)
        print("[demo] Tip: https://ngrok.com/download — then `ngrok config add-authtoken ...` once.", flush=True)
        try:
            # Ngrok v3+: prints URL to console / API; stream stdout for https lines.
            proc = subprocess.Popen(
                [ng, "http", str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            t = threading.Thread(target=_stream_tunnel_output, args=(proc, "ngrok"), daemon=True)
            t.start()
            proc.wait()
        except OSError as e:
            print(f"[demo] ngrok failed: {e}", flush=True)
        return
    print(
        "[demo] --share needs cloudflared OR ngrok on PATH.",
        flush=True,
    )
    print(
        "[demo] Install cloudflared (recommended): https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/",
        flush=True,
    )
    print(
        "[demo] Or install ngrok: https://ngrok.com/download",
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
    print()
    print("  Leave this terminal window open while you use the app.")
    print("  If you close it or press Ctrl+C, the site stops (browser: page not working / refused).")
    print("  Use http://127.0.0.1 in the address bar — not https:// (this server is HTTP only).")
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
