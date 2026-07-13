"""Structured logging with per-request correlation IDs.

Logs are emitted through structlog so that every line carries the same
machine-readable fields. In production (``LOCALGATE_LOG_FORMAT=json``) they render
as JSON for ingestion; in development they render as coloured key-value pairs.

The request id lives in a :class:`~contextvars.ContextVar` rather than being
threaded through call signatures, so a repository or backend adapter deep in the
stack logs the id of the request that reached it without knowing anything about
HTTP.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from contextvars import ContextVar
from typing import Any

import structlog

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def bind_request_id(request_id: str) -> None:
    request_id_var.set(request_id)


def _inject_request_id(
    _logger: Any, _name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    event_dict["request_id"] = request_id_var.get()
    return event_dict


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    """Configure structlog and route stdlib logging (uvicorn, sqlalchemy) through it."""
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _inject_request_id,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s", stream=sys.stderr, level=logging.getLevelName(level.upper())
    )


def get_logger(name: str = "localgate") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
