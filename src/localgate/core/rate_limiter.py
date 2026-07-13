"""Per-key rate limiting — fixed-window counter, in-memory.

Good enough for a single-process deployment. If you run localgate behind
multiple worker processes, swap the in-memory dict for a Redis-backed
counter (same interface: `allow(key_id, limit) -> bool`) so limits are
shared across workers.
"""
import time
from collections import defaultdict


class RateLimiter:
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)

    def allow(self, key_id: str, limit_per_window: int) -> bool:
        now = time.monotonic()
        window_start = now - self.window_seconds
        hits = [t for t in self._hits[key_id] if t > window_start]
        if len(hits) >= limit_per_window:
            self._hits[key_id] = hits
            return False
        hits.append(now)
        self._hits[key_id] = hits
        return True
