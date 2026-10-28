"""Rate limits and per-key concurrency shared by every worker process, via Redis.

With one `localgate serve` process (the default, and the right choice on a GPU
box) the in-memory limiter is exact. Several workers each keep their own
counters, so a key's 60 requests/minute quietly becomes 60 × workers. Setting
`LOCALGATE_REDIS_URL` (and installing `localgate[redis]`) moves both the
per-minute window and each key's in-flight count into Redis, so limits hold
across processes and machines. The GPU queue itself stays per process.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

#: Sliding-window limiter: drop hits older than the window, then either refuse
#: (returning the oldest hit's time, for Retry-After) or record this one.
_WINDOW_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
if redis.call('ZCARD', key) >= limit then
  local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
  return {0, oldest[2]}
end
redis.call('ZADD', key, now, ARGV[4])
redis.call('PEXPIRE', key, math.ceil(window * 1000))
return {1, '0'}
"""


def _client(url: str) -> Any:
    try:
        import redis.asyncio as aioredis  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover — depends on the optional extra
        raise RuntimeError(
            "LOCALGATE_REDIS_URL is set but the redis client isn't installed: "
            "pip install 'localgate[redis]'"
        ) from exc
    return aioredis.from_url(url, decode_responses=True)


class RedisRateLimiter:
    """The same sliding window as `RateLimiter`, kept in Redis."""

    def __init__(self, url: str, window_seconds: int = 60, prefix: str = "localgate:rl:") -> None:
        self.redis = _client(url)
        self.window_seconds = window_seconds
        self.prefix = prefix
        self._script = self.redis.register_script(_WINDOW_SCRIPT)

    async def check(self, key_id: str, limit_per_window: int) -> tuple[bool, float]:
        now = time.time()
        allowed, oldest = await self._script(
            keys=[self.prefix + key_id],
            args=[now, self.window_seconds, limit_per_window, f"{now}:{uuid.uuid4().hex}"],
        )
        if int(allowed):
            return True, 0.0
        return False, max(0.0, float(oldest) + self.window_seconds - now)

    async def aclose(self) -> None:
        await self.redis.aclose()


class RedisKeyLeases:
    """Requests each key has in flight, counted across processes. A lease
    expires on its own after `ttl` seconds, so a crashed worker can't strand
    a key at its limit forever."""

    def __init__(self, url: str, ttl: float = 900.0, prefix: str = "localgate:inflight:") -> None:
        self.redis = _client(url)
        self.ttl = ttl
        self.prefix = prefix

    async def try_acquire(self, key_id: str, limit: int) -> bool:
        name = self.prefix + key_id
        count = await self.redis.incr(name)
        await self.redis.expire(name, int(self.ttl))
        if count > limit:
            await self.redis.decr(name)
            return False
        return True

    async def release(self, key_id: str) -> None:
        name = self.prefix + key_id
        if await self.redis.decr(name) <= 0:
            await self.redis.delete(name)

    async def aclose(self) -> None:
        await self.redis.aclose()
