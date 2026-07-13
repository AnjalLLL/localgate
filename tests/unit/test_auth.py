"""Unit tests for core.auth — key generation and hashing."""

from localgate.core.auth import generate_key, hash_key


def test_generate_key_has_prefix():
    key = generate_key()
    assert key.startswith("lg_")


def test_generate_key_is_unique():
    assert generate_key() != generate_key()


def test_hash_key_is_deterministic():
    key = "lg_sometestkey"
    assert hash_key(key) == hash_key(key)


def test_hash_key_differs_for_different_keys():
    assert hash_key("lg_one") != hash_key("lg_two")


def test_hash_key_does_not_return_the_raw_key():
    key = "lg_sometestkey"
    assert hash_key(key) != key
