"""Model details for /v1/models and the default num_ctx sent to Ollama."""

from localgate.backends.fake import FakeBackend
from localgate.core.model_info import ModelInfoCache, apply_default_num_ctx


class WindowBackend(FakeBackend):
    """A backend that, like Ollama, takes `options` and reports a context window."""

    accepts_options = True

    def __init__(self, window: int | None) -> None:
        super().__init__("")
        self.window = window
        self.lookups = 0

    async def context_window(self, model: str) -> int | None:
        self.lookups += 1
        return self.window

    async def capabilities(self, model: str) -> list[str] | None:
        return ["completion", "tools"]


async def test_num_ctx_defaults_and_is_capped_by_the_model():
    payload: dict = {"model": "m", "messages": []}
    await apply_default_num_ctx(payload, WindowBackend(4096), "m", 8192, ModelInfoCache())
    assert payload["options"] == {"num_ctx": 4096}

    payload = {"model": "m", "messages": []}
    await apply_default_num_ctx(payload, WindowBackend(131072), "m", 8192, ModelInfoCache())
    assert payload["options"] == {"num_ctx": 8192}


async def test_a_callers_num_ctx_wins():
    payload: dict = {"model": "m", "options": {"num_ctx": 32768, "temperature": 0.1}}
    await apply_default_num_ctx(payload, WindowBackend(4096), "m", 8192, ModelInfoCache())
    assert payload["options"] == {"num_ctx": 32768, "temperature": 0.1}


async def test_backends_that_reject_options_are_left_alone():
    payload: dict = {"model": "m"}
    await apply_default_num_ctx(payload, FakeBackend(""), "m", 8192, ModelInfoCache())
    assert "options" not in payload


async def test_details_are_cached_per_model():
    backend, cache = WindowBackend(4096), ModelInfoCache()
    for _ in range(3):
        details = await cache.get(backend, "m")
    assert backend.lookups == 1
    assert details.as_card_fields() == {
        "context_window": 4096,
        "capabilities": ["completion", "tools"],
    }
