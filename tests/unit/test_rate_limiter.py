"""Unit tests for core.rate_limiter."""
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
