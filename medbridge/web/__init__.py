"""
MedBridge web front-end (vanilla HTML/CSS/ES-module SPA, no build step).

``mount_web(app)`` serves:

* ``GET /``          -> ``index.html`` (chat SPA)
* ``GET /dashboard`` -> ``dashboard.html`` (admin / evaluation dashboard)
* ``/static/*``      -> files under ``css/``, ``js/`` and ``img/`` only

Static assets get explicit, platform-independent content types (Windows
registries sometimes map ``.js`` to ``text/plain``, which browsers refuse for
module scripts) and cache headers (ETag/Last-Modified revalidation from
Starlette plus a short ``max-age``). HTML is served with ``no-cache`` so a new
deployment is picked up immediately. Security headers (CSP etc.) are added by
the application's security middleware, not here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

from fastapi import FastAPI
from fastapi.responses import FileResponse
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

WEB_DIR: Final[Path] = Path(__file__).resolve().parent
STATIC_SUBDIRS: Final[frozenset[str]] = frozenset({"css", "js", "img"})
STATIC_CACHE_CONTROL: Final[str] = "public, max-age=300, must-revalidate"
HTML_CACHE_CONTROL: Final[str] = "no-cache"
CONTENT_TYPES: Final[dict[str, str]] = {
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".json": "application/json",
}


class WebStaticFiles(StaticFiles):
    """StaticFiles restricted to css/js/img with explicit types and cache headers."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        """Serve ``path`` only from an allowed sub-directory."""
        top = Path(path).parts[0] if Path(path).parts else ""
        if top not in STATIC_SUBDIRS:
            raise HTTPException(status_code=404)
        response = await super().get_response(path, scope)
        content_type = CONTENT_TYPES.get(Path(path).suffix.lower())
        if content_type and response.status_code == 200:
            response.headers["content-type"] = content_type
        response.headers["cache-control"] = STATIC_CACHE_CONTROL
        response.headers["x-content-type-options"] = "nosniff"
        return response


def _page(filename: str) -> FileResponse:
    """Return an HTML page with no-cache headers."""
    return FileResponse(
        WEB_DIR / filename,
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": HTML_CACHE_CONTROL, "X-Content-Type-Options": "nosniff"},
    )


def mount_web(app: FastAPI) -> None:
    """Register the SPA routes (``/``, ``/dashboard``) and the ``/static`` mount."""

    @app.get("/", include_in_schema=False)
    async def web_index() -> FileResponse:
        """Chat single-page application."""
        return _page("index.html")

    @app.get("/dashboard", include_in_schema=False)
    async def web_dashboard() -> FileResponse:
        """Admin / evaluation dashboard (data endpoints require an admin token)."""
        return _page("dashboard.html")

    app.mount("/static", WebStaticFiles(directory=WEB_DIR), name="static")


__all__ = ["WEB_DIR", "WebStaticFiles", "mount_web"]
