"""Rolling conversation summarization."""

import pytest

from localgate.backends.fake import FakeBackend
from localgate.config import Settings
from localgate.db.repositories.conversations import ConversationRepository, SummaryRepository
from localgate.memory.summarizer import KEEP_RECENT_MESSAGES, maybe_summarize


@pytest.fixture
def summarizing_settings():
    return Settings(_env_file=None, summarize_after_messages=10, backend_type="fake")


async def _fill(session, session_id: str, count: int) -> None:
    repo = ConversationRepository(session)
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        await repo.add_message(session_id, "k1", role, f"message number {i}")


async def test_a_short_session_is_left_alone(db_session, summarizing_settings):
    await _fill(db_session, "short", 4)

    result = await maybe_summarize(db_session, FakeBackend(), summarizing_settings, "short", "k1")

    assert result is None
    assert await SummaryRepository(db_session).latest("short") is None


async def test_a_long_session_is_summarized(db_session, summarizing_settings):
    await _fill(db_session, "long", 12)

    summary = await maybe_summarize(db_session, FakeBackend(), summarizing_settings, "long", "k1")

    assert summary is not None
    assert summary.content  # the FakeBackend echoes, so the content is whatever it echoed
    # The live tail stays out of the summary: it is still in the model's own context
    # window, so summarizing it would only duplicate what the model can already see.
    assert summary.message_count == 12 - KEEP_RECENT_MESSAGES


async def test_summarization_is_incremental(db_session, summarizing_settings):
    """Re-summarizing the whole history each time would make cost grow with the square
    of session length — which is the exact problem summarization exists to solve."""
    backend = FakeBackend()
    await _fill(db_session, "growing", 12)
    first = await maybe_summarize(db_session, backend, summarizing_settings, "growing", "k1")
    assert first is not None

    await _fill(db_session, "growing", 12)  # 12 more messages arrive
    second = await maybe_summarize(db_session, backend, summarizing_settings, "growing", "k1")
    assert second is not None

    # The second pass only covered messages newer than the first summary...
    assert second.covers_until > first.covers_until
    assert second.message_count < 24
    # ...and it folded the previous summary back in as context.
    last_prompt = backend.calls[-1]["messages"][-1]["content"]
    assert "Previous summary:" in last_prompt


async def test_the_summary_is_stored_as_a_retrievable_chunk(db_session, summarizing_settings):
    """A query matching the *gist* of an old exchange should surface the summary even
    when no verbatim chunk scores well."""
    from localgate.db.repositories.embeddings import EmbeddingRepository

    await _fill(db_session, "chunked", 12)
    await maybe_summarize(db_session, FakeBackend(), summarizing_settings, "chunked", "k1")

    chunks = await EmbeddingRepository(db_session).search(
        "chunked", await FakeBackend().embed("anything", "fake"), top_k=10
    )
    assert any(chunk.kind == "summary" for chunk in chunks)


async def test_summarization_is_disabled_by_setting_the_threshold_to_zero(db_session):
    settings = Settings(_env_file=None, summarize_after_messages=0)
    await _fill(db_session, "off", 30)

    assert await maybe_summarize(db_session, FakeBackend(), settings, "off", "k1") is None


async def test_a_backend_failure_during_summarization_does_not_raise(
    db_session, summarizing_settings
):
    """Summarization runs after the user already has their answer. Failing it would
    turn a successful request into an error for no benefit to anyone."""

    class BrokenBackend(FakeBackend):
        async def chat(self, request):
            raise RuntimeError("backend exploded")

    await _fill(db_session, "broken", 12)

    result = await maybe_summarize(
        db_session, BrokenBackend(), summarizing_settings, "broken", "k1"
    )
    assert result is None  # degraded, not failed
