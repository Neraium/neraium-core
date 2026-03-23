from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse


def _safe_file_under_static_root(static_root: Path, relative: str) -> Path | None:
    """Resolve relative path under static_root; reject traversal outside the tree."""
    try:
        candidate = (static_root / relative).resolve()
        candidate.relative_to(static_root.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _media_type_for_static_file(path: Path) -> str:
    if path.suffix.lower() == ".mjs":
        return "text/javascript"
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def build_web_router() -> APIRouter:
    router = APIRouter(tags=["web"])
    static_dir = Path(__file__).resolve().parent / "static"
    index_file = static_dir / "index.html"

    @router.get("/web/{resource_path:path}", include_in_schema=False)
    def web_static_file(resource_path: str) -> FileResponse:
        """Serve apps/api/static/* at /web/* (Three.js, app.js, etc.). Path-safe; no CDN."""
        path = _safe_file_under_static_root(static_dir, resource_path)
        if path is None:
            raise HTTPException(status_code=404, detail="Not Found")
        return FileResponse(path, media_type=_media_type_for_static_file(path))

    @router.get("/", include_in_schema=False)
    def web_index() -> FileResponse:
        return FileResponse(index_file)

    @router.get("/dashboard", include_in_schema=False)
    def web_dashboard() -> FileResponse:
        return FileResponse(index_file)

    @router.get("/upload", include_in_schema=False)
    def web_upload() -> FileResponse:
        return FileResponse(index_file)

    @router.get("/app/runs", include_in_schema=False)
    def web_runs() -> FileResponse:
        return FileResponse(index_file)

    @router.get("/app/runs/{run_id}", include_in_schema=False)
    def web_run_detail(run_id: str) -> FileResponse:
        _ = run_id
        return FileResponse(index_file)

    @router.get("/app/results/{result_id}", include_in_schema=False)
    def web_result_detail(result_id: int) -> FileResponse:
        _ = result_id
        return FileResponse(index_file)

    @router.get("/demo/sii", include_in_schema=False)
    def web_demo_sii() -> FileResponse:
        return FileResponse(index_file)

    @router.get("/demo", include_in_schema=False)
    def web_demo_entry() -> FileResponse:
        """Entry point for share links; client redirects to dashboard with demo query flags."""
        return FileResponse(index_file)

    return router
