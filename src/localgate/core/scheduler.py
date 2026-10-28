"""Fair sharing of a backend's inference slots between API keys.

A GPU runs a few generations at once (Ollama's ``OLLAMA_NUM_PARALLEL``); more
than that only makes every caller slower until they time out. So each backend
gets `max_concurrent` slots, and requests beyond that wait in a queue:

* **Fair:** waiters are kept per key and slots are handed out round-robin
  across keys, so one device sending 50 requests can't starve another
  sending one.
* **Bounded:** a request waits at most `queue_timeout` seconds, and at most
  `max_queue` may wait at all. Past either, it gets 503 with a Retry-After
  estimate instead of hanging until the client gives up.
* **Per-key concurrency:** a key's ``max_concurrent`` caps how many slots it
  can hold at once, however idle the GPU is.

Streaming responses take their slot *before* the 200 goes out — once headers
are sent a 503 can't be — so `acquire` returns a `Lease` the stream releases
when it ends.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

from localgate.core import metrics
from localgate.core.errors import ServiceBusyError


@dataclass(eq=False)
class _Waiter:
    key_id: str
    per_key_limit: int | None
    future: asyncio.Future[None]


@dataclass(eq=False)
class Lease:
    """A held slot. `release` is idempotent, so every exit path may call it."""

    scheduler: InferenceScheduler
    key_id: str
    started: float = field(default_factory=time.monotonic)
    released: bool = False

    def release(self) -> None:
        if not self.released:
            self.released = True
            self.scheduler._release(self.key_id, time.monotonic() - self.started)


class InferenceScheduler:
    def __init__(
        self,
        max_concurrent: int,
        queue_timeout: float,
        max_queue: int,
        name: str = "default",
    ) -> None:
        self.max_concurrent = max_concurrent
        self.queue_timeout = queue_timeout
        self.max_queue = max_queue
        self.name = name
        self._active = 0
        self._active_by_key: dict[str, int] = {}
        self._waiting: dict[str, deque[_Waiter]] = {}
        #: Keys with waiters, in the order they get their next turn.
        self._turns: deque[str] = deque()
        #: Smoothed seconds a slot is held — for Retry-After estimates.
        self._avg_hold = 5.0

    # -- public -----------------------------------------------------------------

    async def acquire(self, key_id: str, per_key_limit: int | None = None) -> Lease:
        """Wait (fairly) for a slot. Raises `ServiceBusyError` when the queue is
        full or the wait passes `queue_timeout`."""
        if self.queued >= self.max_queue:
            metrics.queue_rejected_total.labels(self.name, "queue_full").inc()
            raise ServiceBusyError(
                f"The GPU is busy — {self.queued} requests are already waiting. Try again shortly.",
                code="queue_full",
                retry_after=self.retry_after(),
            )
        waiter = _Waiter(key_id, per_key_limit, asyncio.get_running_loop().create_future())
        self._waiting.setdefault(key_id, deque()).append(waiter)
        if key_id not in self._turns:
            self._turns.append(key_id)
        self._dispatch()
        try:
            await asyncio.wait_for(asyncio.shield(waiter.future), self.queue_timeout)
        except asyncio.TimeoutError:
            if waiter.future.done():  # granted in the same instant it timed out
                return Lease(self, key_id)
            self._forget(waiter)
            metrics.queue_rejected_total.labels(self.name, "timeout").inc()
            raise ServiceBusyError(
                f"The GPU stayed busy for {self.queue_timeout:.0f} s. Try again shortly.",
                code="queue_timeout",
                retry_after=self.retry_after(),
            ) from None
        except asyncio.CancelledError:
            # The caller went away while waiting; hand back a slot it may have just won.
            if waiter.future.done() and not waiter.future.cancelled():
                self._release(key_id, 0.0)
            else:
                self._forget(waiter)
            raise
        return Lease(self, key_id)

    @property
    def active(self) -> int:
        return self._active

    @property
    def queued(self) -> int:
        return sum(len(q) for q in self._waiting.values())

    def retry_after(self) -> float:
        """A rough wait estimate: the queue ahead, drained `max_concurrent` at a time."""
        rounds = (self.queued + 1) / self.max_concurrent
        return max(1.0, min(self.queue_timeout, rounds * self._avg_hold))

    def stats(self) -> dict[str, object]:
        keys = set(self._active_by_key) | {k for k, q in self._waiting.items() if q}
        return {
            "backend": self.name,
            "max_concurrent": self.max_concurrent,
            "active": self._active,
            "queued": self.queued,
            "by_key": {
                k: {
                    "active": self._active_by_key.get(k, 0),
                    "queued": len(self._waiting.get(k, ())),
                }
                for k in sorted(keys)
            },
        }

    # -- internals --------------------------------------------------------------

    def _eligible(self, waiter: _Waiter) -> bool:
        held = self._active_by_key.get(waiter.key_id, 0)
        return waiter.per_key_limit is None or held < waiter.per_key_limit

    def _dispatch(self) -> None:
        """Hand free slots to waiting keys, one per key per round."""
        while self._active < self.max_concurrent and self._turns:
            granted = False
            for _ in range(len(self._turns)):
                key_id = self._turns[0]
                self._turns.rotate(-1)  # this key goes to the back either way
                queue = self._waiting.get(key_id)
                if not queue:
                    self._turns.remove(key_id)
                    continue
                if not self._eligible(queue[0]):
                    continue
                waiter = queue.popleft()
                if not queue:
                    del self._waiting[key_id]
                    self._turns.remove(key_id)
                self._active += 1
                self._active_by_key[key_id] = self._active_by_key.get(key_id, 0) + 1
                waiter.future.set_result(None)
                granted = True
                break
            if not granted:
                break  # everyone waiting is at their own per-key limit
        self._report()

    def _release(self, key_id: str, held_for: float) -> None:
        self._active -= 1
        remaining = self._active_by_key.get(key_id, 1) - 1
        if remaining > 0:
            self._active_by_key[key_id] = remaining
        else:
            self._active_by_key.pop(key_id, None)
        if held_for > 0:
            self._avg_hold = 0.8 * self._avg_hold + 0.2 * held_for
        self._dispatch()

    def _forget(self, waiter: _Waiter) -> None:
        queue = self._waiting.get(waiter.key_id)
        if queue and waiter in queue:
            queue.remove(waiter)
            if not queue:
                del self._waiting[waiter.key_id]
                if waiter.key_id in self._turns:
                    self._turns.remove(waiter.key_id)
        self._report()

    def _report(self) -> None:
        metrics.inference_active.labels(self.name).set(self._active)
        metrics.inference_queued.labels(self.name).set(self.queued)
