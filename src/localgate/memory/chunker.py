"""Text splitting for conversation history.

Word-based sliding window — simple, fast, no extra dependencies, and good
enough for chat turns (which are usually short). Swap for a token-aware
splitter later if you start chunking long documents rather than chat turns.
"""


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks
