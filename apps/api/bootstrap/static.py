from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from urllib.parse import parse_qs

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.types import Scope


logger = logging.getLogger(__name__)

# Windows / older Python may omit .mjs; browsers refuse module scripts with wrong MIME.
mimetypes.add_type("text/javascript", ".mjs", strict=False)


class CacheControlStaticFiles(StaticFiles):
    """Static files with cache headers tuned for cloud delivery.

    HTML stays non-cacheable to allow clean deploy updates.
    Versionable assets (js/css/images/fonts) get long-lived public caching.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers.setdefault("Vary", "Accept-Encoding")
        ext = Path(path).suffix.lower()
        query_string = scope.get("query_string", b"")
        query_params = parse_qs(query_string.decode("utf-8", errors="ignore")) if query_string else {}
        has_asset_version = bool(query_params.get("v", [None])[0])
        if ext in {".html"}:
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        elif ext in {".js", ".mjs", ".css"}:
            if has_asset_version:
                response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            else:
                response.headers["Cache-Control"] = "public, max-age=300, must-revalidate"
        elif ext in {".csv", ".txt", ".json", ".map"}:
            response.headers["Cache-Control"] = "public, max-age=3600, stale-while-revalidate=300"
        else:
            response.headers["Cache-Control"] = "public, max-age=604800, stale-while-revalidate=86400"
        return response


def _mount_web_static(app: FastAPI) -> None:
    """Serve `apps/api/static` at `/web` (app.js, styles, three-init, …).

    Uses Path(__file__) so the directory is correct regardless of process cwd.
    Registered after the web router so explicit HTML routes win; /web/* is fully static.

    If this mount is skipped, GET /web/... falls through to FastAPI's default 404
    (JSON ``{"detail":"Not Found"}``), which is easy to mistake for an API error.
    """
    static_dir = Path(__file__).resolve().parent.parent / "static"
    if not static_dir.is_dir():
        logger.error(
            "Web static directory missing: %s — /web/* will 404. Clone or sync apps/api/static.",
            static_dir,
        )
        return
    app.mount(
        "/web",
        CacheControlStaticFiles(directory=str(static_dir)),
        name="web",
    )
    logger.info("Serving static files at /web from %s", static_dir)
    # Front-end modules are loaded via CDN import map in static/index.html.
    # Keep startup resilient when /web/vendor/three is not deployed.
