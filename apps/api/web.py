from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse


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
    def web_operator() -> FileResponse:
        return FileResponse(static_dir / "operator.html", headers=html_headers)

    @router.get("/operator/workflow", include_in_schema=False)
    def web_operator_workflow() -> FileResponse:
        return FileResponse(static_dir / "operator.html", headers=html_headers)

    @router.get("/demo/sii", include_in_schema=False)
    def web_demo_sii() -> FileResponse:
        return FileResponse(index_file, headers=html_headers)

    @router.get("/demo", include_in_schema=False)
    def web_demo_entry() -> FileResponse:
        """Entry point for share links; client redirects to dashboard with demo query flags."""
        return FileResponse(index_file, headers=html_headers)

    @router.get("/demo/full", include_in_schema=False)
    def web_demo_full_entry() -> FileResponse:
        """Full product demo entry; client normalizes to dashboard demo flow and opens run detail."""
        return FileResponse(index_file, headers=html_headers)

    return router
