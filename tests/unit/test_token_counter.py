"""Unit tests for core.token_counter."""

from localgate.core.token_counter import count_message_tokens, count_tokens


def test_count_tokens_empty_string():
    assert count_tokens("") == 0


def test_count_tokens_nonzero_for_text():
    assert count_tokens("hello world") > 0


def test_count_tokens_scales_with_length():
    short = count_tokens("hi")
    long = count_tokens("hi " * 50)
    assert long > short


def test_count_message_tokens_sums_all_messages():
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi there"}]
    total = count_message_tokens(messages)
    assert total == count_tokens("hello") + count_tokens("hi there")


def test_count_message_tokens_handles_missing_content():
    messages = [{"role": "system"}]  # no "content" key
    assert count_message_tokens(messages) == 0
