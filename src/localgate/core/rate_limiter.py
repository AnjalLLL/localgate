"""Per-key rate limiting — sliding-window counter, in-memory.

Sized for a single-process deployment. Behind multiple workers each process would
keep its own counters, so the effective limit multiplies by the worker count; a
shared limit needs a Redis-backed counter behind the same
``allow(key_id, limit) -> bool`` interface.
"""

from __future__ import annotations

import time


class RateLimiter:
    """Sliding-window limiter with a bounded memory footprint.

    The obvious implementation — a dict of key_id -> timestamps — never forgets a
    key it has seen, so a gateway that issues and revokes keys over months leaks
    one entry per key forever. Windows that have fully expired are swept on write,
    which keeps the dict proportional to *active* keys rather than to every key
    that has ever called.
    """

    def __init__(self, window_seconds: int = 60, sweep_every: int = 256) -> None:
        self.window_seconds = window_seconds
        self.sweep_every = sweep_every
        self._hits: dict[str, list[float]] = {}
        self._calls_since_sweep = 0

    def allow(self, key_id: str, limit_per_window: int) -> bool:
        """Record a request against ``key_id`` and report whether it is permitted."""
        now = time.monotonic()
        window_start = now - self.window_seconds

        hits = [t for t in self._hits.get(key_id, ()) if t > window_start]

        self._calls_since_sweep += 1
        if self._calls_since_sweep >= self.sweep_every:
            self._sweep(window_start)

        if len(hits) >= limit_per_window:
            self._hits[key_id] = hits
            return False

        hits.append(now)
        self._hits[key_id] = hits
        return True

    def remaining(self, key_id: str, limit_per_window: int) -> int:
        """How many more requests ``key_id`` may make in the current window."""
        window_start = time.monotonic() - self.window_seconds
        used = sum(1 for t in self._hits.get(key_id, ()) if t > window_start)
        return max(limit_per_window - used, 0)

    def reset(self, key_id: str | None = None) -> None:
        if key_id is None:
            self._hits.clear()
        else:
            self._hits.pop(key_id, None)

    def _sweep(self, window_start: float) -> None:
        """Drop keys with no hits left inside the window."""
        self._calls_since_sweep = 0
        stale = [key for key, hits in self._hits.items() if not hits or hits[-1] <= window_start]
        for key in stale:
            del self._hits[key]
