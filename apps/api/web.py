from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse


def build_web_router() -> APIRouter:
    router = APIRouter(tags=["web"])
    static_dir = Path(__file__).resolve().parent / "static"

    @router.get("/", include_in_schema=False)
    def web_index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @router.get("/runs/{run_id}", include_in_schema=False)
    def web_run_page(run_id: str) -> FileResponse:
        _ = run_id
        return FileResponse(static_dir / "index.html")

    @router.get("/results/{result_id}", include_in_schema=False)
    def web_result_page(result_id: int) -> FileResponse:
        _ = result_id
        return FileResponse(static_dir / "index.html")

    @router.get("/web/app.js", include_in_schema=False)
    def web_app_js() -> FileResponse:
        return FileResponse(static_dir / "app.js")

    @router.get("/web/styles.css", include_in_schema=False)
    def web_styles() -> FileResponse:
        return FileResponse(static_dir / "styles.css")

    return router
