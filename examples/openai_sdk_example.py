"""Use localgate with the official OpenAI SDK.

    pip install openai
    python examples/openai_sdk_example.py

localgate speaks the OpenAI API, so the only two things that change are the base URL and
the key. Everything else — streaming, sampling parameters, error handling — works exactly
as it does against OpenAI.
"""

import os

from openai import OpenAI

API_KEY = os.environ.get("LOCALGATE_API_KEY", "lg_your_key_here")
BASE_URL = os.environ.get("LOCALGATE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("LOCALGATE_MODEL", "llama3")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def basic_chat() -> None:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "In one sentence: what is an API gateway?"}],
    )
    print("Reply: ", response.choices[0].message.content)
    print("Tokens:", response.usage.total_tokens)


def streaming() -> None:
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


def memory_across_requests() -> None:
    """The part that isn't a proxy.

    Both calls carry the same X-Session-ID, and the *second* sends no history at all —
    yet the model answers correctly, because the gateway retrieved it. This is what lets
    a small-context model hold a long conversation.
    """
    memo = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        default_headers={"X-Session-ID": "example-session"},
    )

    memo.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "My name is Ana and I prefer Postgres."}],
    )

    recalled = memo.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What database do I prefer?"}],
    )
    print("Recalled:", recalled.choices[0].message.content)


if __name__ == "__main__":
    print("--- basic chat");            basic_chat()
    print("\n--- streaming");           streaming()
    print("\n--- memory across calls"); memory_across_requests()
