"""vLLM adapter.

vLLM's OpenAI server is the reference implementation of the API the generic
backend targets, so this is a pass-through with vLLM's default port. Embeddings
only work if vLLM was started with an embedding model (``--task embed``);
otherwise ``/v1/embeddings`` returns 400 and the memory layer surfaces that.
"""

from __future__ import annotations

from localgate.backends.openai_compat import OpenAICompatBackend


class VLLMBackend(OpenAICompatBackend):
    name = "vllm"
    default_base_url = "http://localhost:8000"
