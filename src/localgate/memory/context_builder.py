"""Merges retrieved memory chunks into the outgoing prompt as a system message."""


def build_augmented_messages(messages: list[dict], retrieved_chunks: list[str]) -> list[dict]:
    if not retrieved_chunks:
        return messages

    context_block = "\n---\n".join(retrieved_chunks)
    memory_message = {
        "role": "system",
        "content": (
            "Relevant context from earlier in this conversation (retrieved from memory):\n"
            f"{context_block}"
        ),
    }
    # Insert after any existing system message(s), before the rest of the conversation.
    insert_at = 0
    for i, m in enumerate(messages):
        if m.get("role") != "system":
            insert_at = i
            break
    else:
        insert_at = len(messages)

    return messages[:insert_at] + [memory_message] + messages[insert_at:]
