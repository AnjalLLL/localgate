"""Rate limits and per-key concurrency shared across worker processes via Redis.
Runs only when a Redis is reachable (LOCALGATE_TEST_REDIS_URL, default
redis://localhost:6379/15); skipped otherwise, e.g. in CI."""

import asyncio
import os
import uuid

import pytest

redis = pytest.importorskip("redis.asyncio")

from localgate.core.errors import ServiceBusyError  # noqa: E402
from localgate.core.scheduler import InferenceScheduler  # noqa: E402
from localgate.core.shared_limits import RedisKeyLeases, RedisRateLimiter  # noqa: E402

URL = os.environ.get("LOCALGATE_TEST_REDIS_URL", "redis://localhost:6379/15")


@pytest.fixture
async def redis_url():
    client = redis.from_url(URL)
    try:
        await asyncio.wait_for(client.ping(), 1)
    except Exception:  # noqa: BLE001
        pytest.skip("no Redis reachable")
    finally:
        await client.aclose()
    return URL


async def test_two_workers_share_one_window(redis_url):
    """Two limiter instances (two worker processes) count against the same 3/min."""
    key = f"key-{uuid.uuid4().hex}"
    worker_a, worker_b = RedisRateLimiter(redis_url), RedisRateLimiter(redis_url)
    results = [
        (await worker_a.check(key, 3))[0],
        (await worker_b.check(key, 3))[0],
        (await worker_a.check(key, 3))[0],
    ]
    allowed, retry_after = await worker_b.check(key, 3)
    assert results == [True, True, True]
    assert not allowed and 0 < retry_after <= 60
    await worker_a.aclose()
    await worker_b.aclose()


async def test_a_keys_concurrency_holds_across_workers(redis_url):
    leases = RedisKeyLeases(redis_url)
    key = f"key-{uuid.uuid4().hex}"
    worker_a = InferenceScheduler(4, queue_timeout=0.5, max_queue=10, name="a")
    worker_b = InferenceScheduler(4, queue_timeout=0.5, max_queue=10, name="b")
    worker_a.remote = worker_b.remote = leases

    held = await worker_a.acquire(key, per_key_limit=1)
    with pytest.raises(ServiceBusyError) as err:  # the other worker has free slots…
        await worker_b.acquire(key, per_key_limit=1)
    assert err.value.code == "key_busy"  # …but the key is at its limit everywhere

    held.release()
    await asyncio.sleep(0.05)  # the remote release runs as a background task
    (await asyncio.wait_for(worker_b.acquire(key, per_key_limit=1), 2)).release()
    await asyncio.sleep(0.05)
    await leases.aclose()
