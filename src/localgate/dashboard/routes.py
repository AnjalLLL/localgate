"""Mounts the dashboard.

The dashboard is plain HTML and JavaScript with no build step. It talks to the same
``/admin`` API the CLI uses — there is no private endpoint behind it — so everything
it can do is equally scriptable, and the admin key it holds is one the operator typed
in themselves.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

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
