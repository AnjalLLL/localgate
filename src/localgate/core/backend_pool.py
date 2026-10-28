"""Several inference servers behind one gateway — several GPUs, or machines.

`LOCALGATE_BACKENDS` lists them (name, type, url, optionally the models each
serves and how many requests each runs at once). Every request goes to a
backend that serves its model — configured, or discovered from the backend's
own model list — picking the one with the least work in flight, and fails over
to the next on a connection error. Each backend has its own fair queue
(`core/scheduler.py`), sized to that machine.

With `LOCALGATE_BACKENDS` unset the pool holds exactly the one backend from
`LOCALGATE_BACKEND_TYPE` / `LOCALGATE_BACKEND_URL`, so a single-GPU setup
behaves as it always has.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

import httpx

from localgate.backends import get_backend
from localgate.backends.base import InferenceBackend
from localgate.config import BackendSpec, Settings
from localgate.core import metrics
from localgate.core.errors import BackendError, describe_backend_failure
from localgate.core.logging import get_logger
from localgate.core.scheduler import InferenceScheduler

T = TypeVar("T")
logger = get_logger(__name__)

#: How long a backend's discovered model list is trusted.
DISCOVERY_SECONDS = 60.0

#: Errors that mean "this backend isn't there", worth trying another one for.
FAILOVER_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)


@dataclass(eq=False)
class PooledBackend:
    spec: BackendSpec
    backend: InferenceBackend
    scheduler: InferenceScheduler
    _discovered: set[str] = field(default_factory=set)
    _discovered_at: float = float("-inf")

    @property
    def name(self) -> str:
        return self.spec.name

    def load(self) -> float:
        """Work in flight relative to capacity — lower is a better pick."""
        s = self.scheduler
        return (s.active + s.queued) / s.max_concurrent

    async def models(self) -> set[str]:
        if self.spec.models is not None:
            return set(self.spec.models)
        if time.monotonic() - self._discovered_at > DISCOVERY_SECONDS:
            # On failure keep the last good list; the backend may be restarting.
            with contextlib.suppress(httpx.HTTPError):
                self._discovered = set(await self.backend.list_models())
            self._discovered_at = time.monotonic()
        return self._discovered


class BackendPool:
    def __init__(self, members: list[PooledBackend]) -> None:
        if not members:
            raise ValueError("a backend pool needs at least one backend")
        self.members = members

    @property
    def primary(self) -> PooledBackend:
        return self.members[0]

    @classmethod
    def from_settings(cls, settings: Settings) -> BackendPool:
        specs = settings.backends or [
            BackendSpec(
                name=settings.backend_type,
                type=settings.backend_type,
                url=settings.backend_url,
                api_key=settings.backend_api_key,
            )
        ]
        members = []
        for spec in specs:
            backend = get_backend(
                spec.type, spec.url, timeout=settings.backend_timeout, api_key=spec.api_key
            )
            scheduler = InferenceScheduler(
                spec.max_concurrent or settings.max_concurrent_requests,
                settings.queue_timeout,
                settings.max_queue,
                name=spec.name,
            )
            members.append(PooledBackend(spec, backend, scheduler))
        return cls(members)

    async def route(self, model: str) -> list[PooledBackend]:
        """Backends to try for ``model``, best first. Never empty: if no backend
        claims the model, the primary gets the request and reports its own error."""
        if len(self.members) == 1:
            return list(self.members)
        serving = [m for m in self.members if model in await m.models()]
        if not serving:
            return [self.primary]
        return sorted(serving, key=lambda m: m.load())

    def stats(self) -> list[dict[str, object]]:
        return [m.scheduler.stats() for m in self.members]

    async def aclose(self) -> None:
        for member in self.members:
            await member.backend.aclose()


async def call_with_failover(
    targets: list[PooledBackend],
    key_id: str,
    per_key_limit: int | None,
    call: Callable[[PooledBackend], Awaitable[T]],
) -> T:
    """Run ``call`` on the first target, holding one of its queue slots; on a
    connection error move on to the next. Any other failure becomes a 502 that
    names the backend that failed."""
    for attempt, target in enumerate(targets):
        lease = await target.scheduler.acquire(key_id, per_key_limit)
        try:
            return await call(target)
        except FAILOVER_ERRORS as exc:
            metrics.backend_errors_total.labels(backend=target.name).inc()
            if attempt + 1 < len(targets):
                logger.warning(
                    "backend_failover", failed=target.name, next=targets[attempt + 1].name
                )
                continue
            raise BackendError(
                describe_backend_failure(exc, target.spec.url, target.spec.type)
            ) from exc
        except httpx.HTTPError as exc:
            metrics.backend_errors_total.labels(backend=target.name).inc()
            raise BackendError(
                describe_backend_failure(exc, target.spec.url, target.spec.type)
            ) from exc
        finally:
            lease.release()
    raise BackendError("No inference backend is configured.")  # pragma: no cover
