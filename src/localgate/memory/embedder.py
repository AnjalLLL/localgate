"""Embedding generation — delegates to whatever inference backend is configured."""

from localgate.backends.base import InferenceBackend


async def embed_text(backend: InferenceBackend, text: str, model: str) -> list[float]:
    return await backend.embed(text, model)
