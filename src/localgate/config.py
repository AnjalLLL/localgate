"""Typed configuration. Every value is overridable via a ``LOCALGATE_*`` env var or ``.env``.

Misconfiguration is the most common way a self-hosted service fails, and the worst
version of that failure is the quiet one — a gateway that starts happily with the
placeholder admin key and is reachable from the network. Validation here is
therefore deliberately loud: anything that would be unsafe or non-functional in
production raises at construction, before the server ever binds a port.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

#: The documented placeholder. Its whole purpose is to be recognised and rejected.
INSECURE_ADMIN_KEY = "change-me-in-production"  # noqa: S105


class Settings(BaseSettings):
    """Runtime configuration for the gateway."""

    model_config = SettingsConfigDict(
        env_prefix="LOCALGATE_",
        env_file=".env",
        extra="ignore",
        # `model_aliases` is a legitimate field name here; without this, pydantic
        # reserves the whole `model_*` namespace for its own API.
        protected_namespaces=(),
    )

    # --- Deployment ---
    environment: Literal["development", "production"] = "development"

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    # NoDecode: pydantic-settings would otherwise JSON-decode this before any field
    # validator runs, so `LOCALGATE_CORS_ORIGINS=http://a,http://b` — the form people
    # actually type into a .env file — would raise a parse error instead of reaching
    # `_parse_origins` below.
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # --- Inference backend ---
    backend_type: str = "ollama"  # ollama | llamacpp | vllm | openai_compat | <plugin>
    backend_url: str = "http://localhost:11434"
    backend_timeout: float = Field(default=120.0, gt=0)
    backend_api_key: str | None = None  # for upstreams that require their own auth
    default_model: str = "llama3"

    #: Friendly name -> real backend model id, e.g. {"fast": "phi4-mini"}.
    #: Set as JSON: LOCALGATE_MODEL_ALIASES='{"fast": "phi4-mini"}'
    model_aliases: dict[str, str] = Field(default_factory=dict)

    # --- Database (SQLite by default; any SQLAlchemy async URL works) ---
    database_url: str = "sqlite+aiosqlite:///./localgate.db"

    # --- Memory / RAG (context extension) ---
    memory_enabled: bool = True
    embedding_model: str = "nomic-embed-text"
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)
    max_retrieved_chunks: int = Field(default=5, ge=0)
    #: Cosine-similarity floor for injecting a retrieved chunk. 0.0 means "no floor",
    #: which is the safe default because the right threshold depends entirely on the
    #: embedding model. With a real one (nomic-embed-text, mxbai-embed-large) 0.3–0.5
    #: is the usual range; see docs/rag-memory.md for how to tune it against your logs.
    memory_min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Summarize a session's older turns once it exceeds this many stored messages.
    #: 0 disables summarization.
    summarize_after_messages: int = Field(default=20, ge=0)

    # --- Prompt cache ---
    cache_enabled: bool = False
    cache_ttl_seconds: int = Field(default=300, ge=0)
    cache_max_entries: int = Field(default=512, ge=1)

    # --- Auth & limits ---
    admin_key: str = Field(default=INSECURE_ADMIN_KEY)
    default_rate_limit_per_min: int = Field(default=60, ge=1)

    # --- Observability ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "console"
    metrics_enabled: bool = True

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_origins(cls, value: Any) -> Any:
        """Accept both JSON (`["a","b"]`) and the comma-separated form people
        actually type into a .env file."""
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            return json.loads(text)
        return [item.strip() for item in text.split(",") if item.strip()]

    @field_validator("model_aliases", mode="before")
    @classmethod
    def _parse_aliases(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        return json.loads(text) if text else {}

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_below_chunk_size(cls, value: int, info: ValidationInfo) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({value}) must be smaller than chunk_size ({chunk_size}) — "
                "an overlap at or above the chunk size never advances the window, so "
                "chunking would not terminate."
            )
        return value

    @model_validator(mode="after")
    def _production_is_actually_safe(self) -> Settings:
        """Refuse to serve production traffic with a key that is printed in the docs."""
        if self.environment == "production" and self.admin_key == INSECURE_ADMIN_KEY:
            raise ValueError(
                "LOCALGATE_ADMIN_KEY is still the placeholder from .env.example while "
                "LOCALGATE_ENVIRONMENT=production. Anyone who has read the docs could mint "
                "API keys against this gateway. Generate a real one (`openssl rand -hex 32`) "
                "and restart."
            )
        return self

    @property
    def uses_insecure_admin_key(self) -> bool:
        return self.admin_key == INSECURE_ADMIN_KEY

    def resolve_model(self, requested: str | None) -> str:
        """Map a requested model name through the alias table, defaulting when absent."""
        name = requested or self.default_model
        return self.model_aliases.get(name, name)
