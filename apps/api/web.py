from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, RedirectResponse


def build_web_router() -> APIRouter:
    router = APIRouter(tags=["web"])
    static_dir = Path(__file__).resolve().parent / "static"
    index_file = static_dir / "index.html"
    html_headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    @router.get("/", include_in_schema=False)
    def web_index() -> FileResponse:
        return FileResponse(index_file, headers=html_headers)

    @router.get("/dashboard", include_in_schema=False)
    def web_dashboard() -> FileResponse:
        return FileResponse(index_file, headers=html_headers)

    @router.get("/upload", include_in_schema=False)
    def web_upload() -> FileResponse:
        return FileResponse(index_file, headers=html_headers)

    @router.get("/app/runs", include_in_schema=False)
    def web_runs() -> FileResponse:
        return FileResponse(index_file, headers=html_headers)

    @router.get("/app/runs/{run_id}", include_in_schema=False)
    def web_run_detail(run_id: str) -> FileResponse:
        _ = run_id
        return FileResponse(index_file, headers=html_headers)

    @router.get("/app/results/{result_id}", include_in_schema=False)
    def web_result_detail(result_id: int) -> FileResponse:
        _ = result_id
        return FileResponse(index_file, headers=html_headers)

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
        return RedirectResponse(url="/dashboard?demo=1", status_code=307)

    @router.get("/demo/full", include_in_schema=False)
    def web_demo_full_entry_redirect() -> RedirectResponse:
        return RedirectResponse(url="/dashboard?demo=1&autoplay=1", status_code=307)

    return router
