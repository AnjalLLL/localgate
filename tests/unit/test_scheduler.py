"""The fair inference queue: slots, round-robin across keys, per-key caps,
timeouts and a full queue turning into 503 + Retry-After."""

import asyncio

import pytest

from localgate.core.errors import ServiceBusyError
from localgate.core.scheduler import InferenceScheduler


async def _settle():
    for _ in range(5):
        await asyncio.sleep(0)


async def test_requests_beyond_the_slots_wait_then_run():
    scheduler = InferenceScheduler(max_concurrent=1, queue_timeout=5, max_queue=10)
    first = await scheduler.acquire("a")
    waiting = asyncio.create_task(scheduler.acquire("b"))
    await _settle()
    assert not waiting.done() and scheduler.queued == 1
    first.release()
    second = await asyncio.wait_for(waiting, 1)
    assert scheduler.active == 1 and scheduler.queued == 0
    second.release()
    assert scheduler.active == 0


async def test_slots_go_round_robin_across_keys_not_first_come():
    """Key "busy" queues three requests before "quiet" asks once — "quiet"
    still gets the second slot, not the fourth."""
    scheduler = InferenceScheduler(max_concurrent=1, queue_timeout=5, max_queue=10)
    holder = await scheduler.acquire("busy")
    order: list[str] = []

    async def run(key: str) -> None:
        lease = await scheduler.acquire(key)
        order.append(key)
        await asyncio.sleep(0)
        lease.release()

    tasks = [asyncio.create_task(run("busy")) for _ in range(3)]
    await _settle()
    tasks.append(asyncio.create_task(run("quiet")))
    await _settle()
    holder.release()
    await asyncio.wait_for(asyncio.gather(*tasks), 2)
    assert order.index("quiet") == 1


async def test_a_keys_own_concurrency_cap_holds_even_with_free_slots():
    scheduler = InferenceScheduler(max_concurrent=4, queue_timeout=5, max_queue=10)
    first = await scheduler.acquire("phone", per_key_limit=1)
    capped = asyncio.create_task(scheduler.acquire("phone", per_key_limit=1))
    other = await asyncio.wait_for(scheduler.acquire("laptop"), 1)  # free slots still go to others
    await _settle()
    assert not capped.done()
    first.release()
    (await asyncio.wait_for(capped, 1)).release()
    other.release()


async def test_waiting_too_long_is_a_503_with_retry_after():
    scheduler = InferenceScheduler(max_concurrent=1, queue_timeout=0.05, max_queue=10)
    holder = await scheduler.acquire("a")
    with pytest.raises(ServiceBusyError) as err:
        await scheduler.acquire("b")
    assert err.value.code == "queue_timeout"
    assert int(err.value.headers["Retry-After"]) >= 1
    assert scheduler.queued == 0  # the timed-out waiter left the queue
    holder.release()


async def test_a_full_queue_is_refused_at_once():
    scheduler = InferenceScheduler(max_concurrent=1, queue_timeout=5, max_queue=1)
    holder = await scheduler.acquire("a")
    waiter = asyncio.create_task(scheduler.acquire("b"))
    await _settle()
    with pytest.raises(ServiceBusyError) as err:
        await scheduler.acquire("c")
    assert err.value.code == "queue_full"
    holder.release()
    (await asyncio.wait_for(waiter, 1)).release()


async def test_a_cancelled_waiter_never_leaks_a_slot():
    scheduler = InferenceScheduler(max_concurrent=1, queue_timeout=5, max_queue=10)
    holder = await scheduler.acquire("a")
    waiter = asyncio.create_task(scheduler.acquire("b"))
    await _settle()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    holder.release()
    assert (scheduler.active, scheduler.queued) == (0, 0)
    (await asyncio.wait_for(scheduler.acquire("c"), 1)).release()


async def test_release_is_idempotent():
    scheduler = InferenceScheduler(max_concurrent=2, queue_timeout=5, max_queue=10)
    lease = await scheduler.acquire("a")
    lease.release()
    lease.release()
    assert scheduler.active == 0
    assert scheduler.stats()["by_key"] == {}
