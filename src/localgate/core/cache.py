"""In-process prompt cache.

An identical prompt sent twice costs a full inference pass the second time, which
on CPU-hosted local models is measured in seconds. Caching the response makes the
repeat free.

Two design points worth stating plainly:

* **The key is the fully-augmented payload** that would be sent to the backend —
  after model aliasing and after RAG context injection. Keying on the client's raw
  body instead would let a cache hit serve a response built from stale memory
  context.
* **Caching makes sampling deterministic.** Two identical requests at
  ``temperature=0.8`` are *supposed* to produce different completions; a cache
  returns the first one twice. That is a real semantic change, which is why the
  cache is opt-in (``LOCALGATE_CACHE_ENABLED=true``) rather than on by default.

Entries are held in this process only. Running multiple workers means multiple
independent caches — correct, just less effective. A shared cache would need Redis
behind the same ``get``/``set`` interface.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any


def cache_key(payload: dict[str, Any]) -> str:
    """A stable digest of the outgoing backend payload.

    ``sort_keys`` matters: two dicts equal in content but built in a different
    order must produce the same key, or the cache would miss on every request.
    ``stream`` is excluded because a streamed and a non-streamed request for the
    same prompt yield the same content, just framed differently.
    """
    material = {k: v for k, v in payload.items() if k != "stream"}
    encoded = json.dumps(material, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


class PromptCache:
    """A bounded, TTL-expiring LRU cache of chat completions."""

    def __init__(self, max_entries: int = 512, ttl_seconds: int = 300) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> dict[str, Any] | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None

        stored_at, value = entry
        if self.ttl_seconds and (time.monotonic() - stored_at) > self.ttl_seconds:
            del self._entries[key]
            self.misses += 1
            return None

        self._entries.move_to_end(key)  # refresh recency
        self.hits += 1
        return value

    def set(self, key: str, value: dict[str, Any]) -> None:
        self._entries[key] = (time.monotonic(), value)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)  # evict least recently used

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict[str, int | float]:
        looked_up = self.hits + self.misses
        return {
            "entries": len(self._entries),
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / looked_up, 4) if looked_up else 0.0,
        }
