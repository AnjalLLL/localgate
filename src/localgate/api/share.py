"""``/admin/share`` — what the dashboard's Share page needs to hand a device access.

The addresses other devices can reach this gateway at, and QR codes for the
connect links built from them. The QR is made here (SVG via `segno`) rather
than in the browser so the dashboard keeps its no-third-party-script CSP, and
it is requested with POST so the API key inside the link never lands in a URL
or an access log.
"""

from __future__ import annotations

import io
from typing import Any

import segno
from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, Field

from localgate import __version__
from localgate.api.deps import require_admin
from localgate.core.netinfo import is_loopback_host, share_urls
from localgate.db.repositories.audit import audit

router = APIRouter(tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/share")
async def share_info(request: Request) -> dict[str, Any]:
    settings = request.app.state.settings
    port = request.url.port or (443 if request.url.scheme == "https" else 80)
    urls = share_urls(port, settings.public_url)
    # The address the owner opened the dashboard at works for other devices too,
    # unless it's this machine's loopback.
    here = str(request.base_url).rstrip("/")
    hostname = request.url.hostname
    if hostname and not is_loopback_host(hostname) and all(entry["url"] != here for entry in urls):
        urls.insert(0, {"label": "This address", "url": here})
    tunnel = request.app.state.tunnel
    if tunnel.url and all(entry["url"] != tunnel.url for entry in urls):
        urls.insert(0, {"label": "Internet", "url": tunnel.url})
    return {"urls": urls, "internet": tunnel.status(), "version": __version__}


class InternetSwitch(BaseModel):
    enabled: bool


@router.post("/share/internet")
async def switch_internet(body: InternetSwitch, request: Request) -> dict[str, Any]:
    """Turn the gateway's internet address on or off. Opening one runs in the
    background (the first time it downloads cloudflared); poll GET /admin/share."""
    tunnel = request.app.state.tunnel
    if body.enabled:
        tunnel.start()
    else:
        tunnel.stop()
    await audit(request, "share.internet_on" if body.enabled else "share.internet_off")
    return tunnel.status()  # type: ignore[no-any-return]


class QRRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1500)


@router.post("/share/qr")
async def share_qr(body: QRRequest) -> Response:
    """An SVG QR code for ``text`` (a connect link)."""
    buffer = io.BytesIO()
    segno.make(body.text, error="m").save(
        buffer, kind="svg", scale=6, border=2, dark="#1a1a1a", light="#ffffff", xmldecl=False
    )
    return Response(buffer.getvalue(), media_type="image/svg+xml")
