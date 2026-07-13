"""Token counting for prompts and completions.

Uses tiktoken's cl100k_base encoding as a universal approximation. It won't be
byte-exact for every local model's own tokenizer, but it's consistent and good
enough for usage accounting and rate limiting — the same tradeoff most
OpenAI-compatible gateways make.
"""
import tiktoken

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def count_message_tokens(messages: list[dict]) -> int:
    return sum(count_tokens(m.get("content", "")) for m in messages)
