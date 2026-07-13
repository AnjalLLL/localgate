"""The prompt cache."""

import time

from localgate.core.cache import PromptCache, cache_key


def test_key_is_stable_regardless_of_dict_ordering():
    """Two payloads equal in content but built in a different order must hash the
    same, or the cache would miss on every single request."""
    a = {"model": "llama3", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.2}
    b = {"temperature": 0.2, "messages": [{"role": "user", "content": "hi"}], "model": "llama3"}
    assert cache_key(a) == cache_key(b)


def test_streaming_flag_does_not_change_the_key():
    """A streamed and a non-streamed request for the same prompt produce the same
    content, just framed differently — so they should share a cache entry."""
    base = {"model": "llama3", "messages": [{"role": "user", "content": "hi"}]}
    assert cache_key({**base, "stream": True}) == cache_key({**base, "stream": False})


def test_different_prompts_get_different_keys():
    a = {"model": "llama3", "messages": [{"role": "user", "content": "hi"}]}
    b = {"model": "llama3", "messages": [{"role": "user", "content": "bye"}]}
    assert cache_key(a) != cache_key(b)


def test_different_temperature_gets_a_different_key():
    a = {"model": "llama3", "messages": [], "temperature": 0.0}
    b = {"model": "llama3", "messages": [], "temperature": 0.9}
    assert cache_key(a) != cache_key(b)


def test_hit_and_miss():
    cache = PromptCache()
    assert cache.get("k") is None
    cache.set("k", {"answer": 42})
    assert cache.get("k") == {"answer": 42}
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1


def test_entries_expire_after_the_ttl():
    cache = PromptCache(ttl_seconds=1)
    cache.set("k", {"v": 1})

    # Rather than sleeping, age the entry by rewriting its timestamp.
    stored_at, value = cache._entries["k"]
    cache._entries["k"] = (stored_at - 2, value)

    assert cache.get("k") is None


def test_ttl_of_zero_means_entries_never_expire():
    cache = PromptCache(ttl_seconds=0)
    cache.set("k", {"v": 1})
    cache._entries["k"] = (time.monotonic() - 10_000, {"v": 1})
    assert cache.get("k") == {"v": 1}


def test_eviction_is_least_recently_used():
    """The cache is bounded, so it must throw something away — and the thing it keeps
    should be the one that is actually being used."""
    cache = PromptCache(max_entries=2)
    cache.set("a", {"v": "a"})
    cache.set("b", {"v": "b"})

    cache.get("a")  # 'a' is now the more recently used of the two
    cache.set("c", {"v": "c"})  # forces an eviction

    assert cache.get("a") == {"v": "a"}
    assert cache.get("c") == {"v": "c"}
    assert cache.get("b") is None  # 'b' was the least recently used


def test_hit_rate_reported():
    cache = PromptCache()
    cache.set("k", {"v": 1})
    cache.get("k")
    cache.get("missing")
    assert cache.stats()["hit_rate"] == 0.5
