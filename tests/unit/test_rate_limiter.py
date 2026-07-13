"""core.rate_limiter."""

import time

from localgate.core.rate_limiter import RateLimiter


def test_allows_requests_under_the_limit():
    limiter = RateLimiter()
    for _ in range(5):
        assert limiter.allow("key-1", limit_per_window=5) is True


def test_blocks_requests_over_the_limit():
    limiter = RateLimiter()
    for _ in range(5):
        limiter.allow("key-1", limit_per_window=5)
    assert limiter.allow("key-1", limit_per_window=5) is False


def test_limits_are_independent_per_key():
    limiter = RateLimiter()
    for _ in range(5):
        limiter.allow("key-1", limit_per_window=5)
    assert limiter.allow("key-2", limit_per_window=5) is True


def test_the_window_slides():
    limiter = RateLimiter(window_seconds=60)
    for _ in range(3):
        limiter.allow("key-1", limit_per_window=3)
    assert limiter.allow("key-1", limit_per_window=3) is False

    # Age the recorded hits out of the window rather than sleeping for a minute.
    limiter._hits["key-1"] = [t - 61 for t in limiter._hits["key-1"]]
    assert limiter.allow("key-1", limit_per_window=3) is True


def test_remaining_reports_the_budget_left():
    limiter = RateLimiter()
    limiter.allow("key-1", limit_per_window=5)
    limiter.allow("key-1", limit_per_window=5)
    assert limiter.remaining("key-1", limit_per_window=5) == 3


def test_expired_keys_are_swept_so_memory_stays_bounded():
    """The naive dict-of-timestamps never forgets a key it has seen, so a gateway that
    issues and revokes keys over months leaks one entry per key, forever."""
    limiter = RateLimiter(window_seconds=60, sweep_every=10)

    for i in range(20):
        limiter.allow(f"key-{i}", limit_per_window=100)

    aged = time.monotonic() - 120
    for key in list(limiter._hits):
        limiter._hits[key] = [aged]

    for _ in range(10):  # enough calls to trigger a sweep
        limiter.allow("active-key", limit_per_window=100)

    assert set(limiter._hits) == {"active-key"}


def test_an_active_key_is_never_swept():
    limiter = RateLimiter(window_seconds=60, sweep_every=2)
    for _ in range(10):
        limiter.allow("busy", limit_per_window=100)
    assert limiter.remaining("busy", limit_per_window=100) == 90
