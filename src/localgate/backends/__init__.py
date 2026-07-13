"""Backend registry and factory.

Backends are looked up through the ``localgate.backends`` entry-point group, so a
third-party package can add one without touching this file: declare the entry
point in its own ``pyproject.toml``, install it alongside localgate, and set
``LOCALGATE_BACKEND_TYPE`` to the name it registered.

    [project.entry-points."localgate.backends"]
    my-server = "my_package.backend:MyBackend"

The built-in backends are registered through that same mechanism (see localgate's
own ``pyproject.toml``) — deliberately, so there is no privileged path a plugin
can't take. :func:`register_backend` exists for tests and programmatic embedding,
where declaring an entry point isn't practical.
"""

from __future__ import annotations

import inspect
from importlib.metadata import entry_points
from typing import Any

from localgate.backends.base import InferenceBackend

ENTRY_POINT_GROUP = "localgate.backends"

_registry: dict[str, type[InferenceBackend]] = {}


def register_backend(name: str, backend_cls: type[InferenceBackend]) -> None:
    """Register a backend under ``name``, taking precedence over entry points."""
    if not (isinstance(backend_cls, type) and issubclass(backend_cls, InferenceBackend)):
        raise TypeError(f"{backend_cls!r} does not implement InferenceBackend")
    _registry[name] = backend_cls


def _discover() -> dict[str, type[InferenceBackend]]:
    """Load every backend advertised via entry points, plus explicit registrations.

    A broken third-party plugin must not take the whole gateway down, so a plugin
    that fails to import is skipped with a warning naming it, rather than raised.
    """
    discovered: dict[str, type[InferenceBackend]] = {}
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        try:
            loaded = entry_point.load()
        except Exception as exc:  # noqa: BLE001 — see docstring
            import structlog

            structlog.get_logger(__name__).warning(
                "backend_plugin_load_failed", plugin=entry_point.name, error=str(exc)
            )
            continue
        if isinstance(loaded, type) and issubclass(loaded, InferenceBackend):
            discovered[entry_point.name] = loaded
    discovered.update(_registry)
    return discovered


def available_backends() -> list[str]:
    """Names accepted by :func:`get_backend`, including any installed plugins."""
    return sorted(_discover())


def get_backend(
    backend_type: str,
    base_url: str | None = None,
    timeout: float = 120.0,
    api_key: str | None = None,
) -> InferenceBackend:
    """Instantiate the backend registered under ``backend_type``.

    Only the constructor arguments a backend actually declares are passed to it.
    That keeps the plugin contract as small as it can be — a plugin whose
    ``__init__`` takes just ``base_url`` stays valid even as localgate grows new
    per-backend options.
    """
    backends = _discover()
    backend_cls = backends.get(backend_type)
    if backend_cls is None:
        raise ValueError(
            f"Unknown backend_type {backend_type!r}. "
            f"Available: {', '.join(sorted(backends)) or '(none)'}"
        )

    offered: dict[str, Any] = {"base_url": base_url, "timeout": timeout, "api_key": api_key}
    accepted = inspect.signature(backend_cls).parameters
    kwargs = {k: v for k, v in offered.items() if k in accepted}
    return backend_cls(**kwargs)


__all__ = [
    "ENTRY_POINT_GROUP",
    "InferenceBackend",
    "available_backends",
    "get_backend",
    "register_backend",
]
