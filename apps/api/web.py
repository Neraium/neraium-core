from __future__ import annotations

import hashlib
import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse


def build_web_router() -> APIRouter:
    router = APIRouter(tags=["web"])
    static_dir = Path(__file__).resolve().parent / "static"
    index_file = static_dir / "index.html"
    html_headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    asset_version = _resolve_asset_version(static_dir=static_dir)
    index_html = index_file.read_text(encoding="utf-8").replace("__NERAIUM_ASSET_VERSION__", asset_version)

    def _render_index() -> HTMLResponse:
        return HTMLResponse(content=index_html, headers=html_headers)

    @router.get("/", include_in_schema=False)
    def web_index() -> HTMLResponse:
        return _render_index()

    @router.get("/dashboard", include_in_schema=False)
    def web_dashboard() -> HTMLResponse:
        return _render_index()

    @router.get("/pilot", include_in_schema=False)
    def web_pilot() -> HTMLResponse:
        return _render_index()

    @router.get("/operations", include_in_schema=False)
    def web_operations() -> HTMLResponse:
        return _render_index()

    @router.get("/upload", include_in_schema=False)
    def web_upload() -> HTMLResponse:
        return _render_index()

    @router.get("/validation", include_in_schema=False)
    def web_validation() -> HTMLResponse:
        return _render_index()

    @router.get("/onboarding", include_in_schema=False)
    def web_onboarding() -> HTMLResponse:
        return _render_index()

    @router.get("/reference", include_in_schema=False)
    def web_reference() -> RedirectResponse:
        return RedirectResponse(url="/validation", status_code=307)

    @router.get("/historical-validation", include_in_schema=False)
    def web_historical_validation() -> RedirectResponse:
        return RedirectResponse(url="/validation", status_code=307)

    @router.get("/app/runs", include_in_schema=False)
    def web_runs() -> HTMLResponse:
        return _render_index()

    @router.get("/app/runs/{run_id}", include_in_schema=False)
    def web_run_detail(run_id: str) -> HTMLResponse:
        _ = run_id
        return _render_index()

    @router.get("/app/results/{result_id}", include_in_schema=False)
    def web_result_detail(result_id: int) -> HTMLResponse:
        _ = result_id
        return _render_index()

    @router.get("/operator", include_in_schema=False)
    def web_operator_redirect() -> RedirectResponse:
        return RedirectResponse(url="/dashboard", status_code=307)

    @router.get("/operator/workflow", include_in_schema=False)
    def web_operator_workflow_redirect() -> RedirectResponse:
        return RedirectResponse(url="/dashboard", status_code=307)

    @router.get("/demo/sii", include_in_schema=False)
    def web_demo_sii_redirect() -> RedirectResponse:
        return RedirectResponse(url="/dashboard", status_code=307)

    @router.get("/demo", include_in_schema=False)
    def web_demo_entry_redirect() -> RedirectResponse:
        return RedirectResponse(url="/dashboard?replay=1", status_code=307)

    @router.get("/demo/full", include_in_schema=False)
    def web_demo_full_entry_redirect() -> RedirectResponse:
        return RedirectResponse(url="/dashboard?replay=1&autoplay=1", status_code=307)

    return router


def _resolve_asset_version(*, static_dir: Path) -> str:
    configured = str(os.getenv("NERAIUM_WEB_ASSET_VERSION", "")).strip()
    if configured:
        return configured

    hasher = hashlib.sha256()
    fingerprinted_exts = {".js", ".mjs", ".css"}
    for file_path in sorted(static_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in fingerprinted_exts:
            continue
        rel = file_path.relative_to(static_dir).as_posix()
        stat = file_path.stat()
        hasher.update(rel.encode("utf-8"))
        hasher.update(str(stat.st_size).encode("ascii"))
        hasher.update(str(stat.st_mtime_ns).encode("ascii"))
    return hasher.hexdigest()[:12]
