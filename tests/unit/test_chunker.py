"""Unit tests for memory.chunker."""
from localgate.memory.chunker import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []


def test_short_text_returns_single_chunk():
    text = "just a few words"
    assert chunk_text(text, chunk_size=100, overlap=10) == [text]


def test_long_text_splits_into_multiple_chunks():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 1


def test_chunks_have_overlap():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    # last `overlap` words of chunk 1 should reappear at the start of chunk 2
    assert first_words[-20:] == second_words[:20]


def test_no_words_lost_across_chunks():
    words = [f"word{i}" for i in range(250)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=100, overlap=0)
    rejoined = " ".join(chunks).split()
    assert rejoined == words
