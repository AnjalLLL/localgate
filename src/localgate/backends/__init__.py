"""Backend factory — picks the right adapter based on config."""
from localgate.backends.base import InferenceBackend
from localgate.backends.ollama import OllamaBackend


def get_backend(backend_type: str, base_url: str) -> InferenceBackend:
    if backend_type == "ollama":
        return OllamaBackend(base_url=base_url)
    if backend_type == "vllm":
        from localgate.backends.vllm import VLLMBackend
        return VLLMBackend(base_url=base_url)
    if backend_type == "llamacpp":
        from localgate.backends.llamacpp import LlamaCppBackend
        return LlamaCppBackend(base_url=base_url)
    if backend_type == "openai_compat":
        from localgate.backends.openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(base_url=base_url)
    raise ValueError(f"Unknown backend_type: {backend_type}")
