"""Mounts the dashboard.

The dashboard is plain HTML and JavaScript with no build step. It talks to the same
``/admin`` API the CLI uses — there is no private endpoint behind it — so everything
it can do is equally scriptable, and the admin key it holds is one the operator typed
in themselves.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from localgate import __version__

STATIC_DIR = Path(__file__).parent / "static"


def mount_dashboard(app: FastAPI) -> bool:
    """Serve the dashboard at ``/dashboard``. Returns whether it was mounted.

    ``html=True`` makes StaticFiles serve ``index.html`` for the directory root and
    redirect the bare ``/dashboard`` to ``/dashboard/``, which is what keeps the
    page's relative asset paths resolving.
    """
    if not STATIC_DIR.is_dir():
        return False
    app.mount("/dashboard", StaticFiles(directory=str(STATIC_DIR), html=True), name="dashboard")
    return True


def mount_home(app: FastAPI) -> None:
    """A landing page at ``/``: what this server is, whether it's healthy, and how
    to start using it. Browsers get the page; an API client asking for JSON gets
    a short pointer document instead of HTML it can't use."""
    page = STATIC_DIR / "home.html"

    @app.get("/", include_in_schema=False)
    async def home(request: Request) -> Response:
        accept = request.headers.get("accept", "")
        if not page.is_file() or ("text/html" not in accept and "json" in accept):
            return JSONResponse(
                {
                    "name": "localgate",
                    "version": __version__,
                    "api": "/v1",
                    "docs": "/docs",
                    "dashboard": "/dashboard/",
                    "health": "/health",
                }
            )
        return FileResponse(page, media_type="text/html")


def mount_connect(app: FastAPI) -> None:
    """``/connect``: the page a QR code from the Share page opens on a phone or
    laptop. Public, like the landing page — it reads the gateway address and API
    key from the URL fragment, which the browser never sends to any server."""
    page = STATIC_DIR / "connect.html"

    @app.get("/connect", include_in_schema=False)
    async def connect() -> Response:
        return FileResponse(page, media_type="text/html")
