"""Token counting for prompts and completions.

Uses tiktoken's ``cl100k_base`` as a universal approximation. It is not byte-exact
for a local model's own tokenizer — Llama, Qwen and Phi all tokenize differently —
but it is *consistent*, needs no model download, and is the same tradeoff every
OpenAI-compatible gateway makes. Where a backend reports its own usage numbers
those win; this is the fallback (see ``api/chat.py::_reported``).
"""

from __future__ import annotations

import functools
from typing import Any

import tiktoken


@functools.lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    """Built lazily and cached.

    Importing tiktoken is cheap; *constructing* an encoding is not, and doing it at
    module import would put that cost on every `localgate --help`.
    """
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str | None) -> int:
    if not text:
        return 0
    return len(_encoding().encode(text))


def count_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Approximate the prompt tokens for a list of chat messages.

    Ignores the framing a chat template adds around each message (a few tokens of
    role markers and separators), so this undercounts by a small constant per
    message. Acceptable for accounting, and precisely why a backend's own count is
    preferred whenever one is reported.
    """
    return sum(count_tokens(_content_text(message)) for message in messages)


def _content_text(message: dict[str, Any]) -> str:
    """Flatten content, which may be absent, plain text, or multimodal parts."""
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)
