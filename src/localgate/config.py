"""Typed configuration. Every value is overridable via a ``LOCALGATE_*`` env var or ``.env``.

Misconfiguration is the most common way a self-hosted service fails, and the worst
version of that failure is the quiet one — a gateway that starts happily with the
placeholder admin key and is reachable from the network. Validation here is
therefore deliberately loud: anything that would be unsafe or non-functional in
production raises at construction, before the server ever binds a port.

**Construct settings with :func:`load_settings`, not ``Settings()``.** The env
files to read are resolved at call time there; putting a computed path in
``model_config`` below would freeze it at class-definition (import) time, which
both hides the per-user config file from anything that adjusts the environment
afterwards and makes the test suite unable to isolate itself. ``Settings()``
remains available for tests that want an explicitly-constructed object.
"""

from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from localgate import paths

#: The documented placeholder. Its whole purpose is to be recognised and rejected.
INSECURE_ADMIN_KEY = "change-me-in-production"  # noqa: S105

#: Networks allowed to reach the dashboard, /admin, the API docs and /metrics by
#: default: this machine, private LANs, Tailscale (CGNAT range) and IPv6 ULA/link-local.
#: The public internet is not on it — /v1 stays reachable from anywhere the port is.
DEFAULT_ADMIN_NETWORKS = (
    "127.0.0.0/8",
    "::1/128",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "100.64.0.0/10",
    "fc00::/7",
    "fe80::/10",
)


class BackendSpec(BaseModel):
    """One inference server in `LOCALGATE_BACKENDS` (several GPUs or machines)."""

    name: str
    type: str = "ollama"
    url: str
    api_key: str | None = None
    #: Models this backend serves; None = discover them from the backend.
    models: list[str] | None = None
    #: Concurrent requests this backend runs; None = LOCALGATE_MAX_CONCURRENT_REQUESTS.
    max_concurrent: int | None = Field(default=None, ge=1)


#: The project-local dotenv, relative to the current directory.
PROJECT_ENV_FILE = ".env"


def env_file_chain() -> tuple[Path, ...]:
    """The dotenv files to read, lowest precedence first.

    pydantic-settings gives later files priority and still takes values the later
    file omits from earlier ones, so this yields:
    real ``LOCALGATE_*`` env vars > ``./.env`` > the per-user file > field defaults.

    Putting ``./.env`` last is deliberate: someone working in a checkout with a
    ``.env`` expects it to win over whatever they once ran ``localgate init`` with.
    """
    return (paths.user_env_file(), Path(PROJECT_ENV_FILE))


def load_settings(**overrides: Any) -> Settings:
    """Build :class:`Settings` from the env-file chain resolved *now*.

    The default ``database_url`` is resolved here so that it points at
    ``data_dir()/localgate.db`` rather than the CWD — a user who runs
    ``localgate`` from different directories should always get the same database.
    The per-user data directory is only used when no explicit
    ``LOCALGATE_DATABASE_URL`` is set (env vars still win via pydantic-settings).
    """
    # Inject the data_dir default when no explicit URL override is given and the env
    # doesn't override it. We do this before constructing Settings so there is only
    # one parse pass — not two — and so the production validator on admin_key fires
    # exactly once.
    if "database_url" not in overrides and not os.environ.get("LOCALGATE_DATABASE_URL"):
        default_db = paths.ensure_data_dir() / "localgate.db"
        overrides = {**overrides, "database_url": f"sqlite+aiosqlite:///{default_db}"}
    # `_env_file` is pydantic-settings' documented per-instantiation override; it
    # isn't in the generated __init__ signature, hence the ignore.
    return Settings(_env_file=env_file_chain(), **overrides)  # type: ignore[call-arg]


class Settings(BaseSettings):
    """Runtime configuration for the gateway."""

    model_config = SettingsConfigDict(
        env_prefix="LOCALGATE_",
        # Intentionally NOT set here — see the module docstring and `load_settings`.
        # A path computed at class-definition time can never be redirected later.
        env_file=None,
        extra="ignore",
        # `model_aliases` is a legitimate field name here; without this, pydantic
        # reserves the whole `model_*` namespace for its own API.
        protected_namespaces=(),
    )

    # --- Deployment ---
    environment: Literal["development", "production"] = "development"

    # --- Server ---
    host: str = "127.0.0.1"
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

    # --- caniollama registry integration (read-only) ---
    # Default assumes registry running locally:
    # cd ~/caniollama/compat-registry && uv run uvicorn compat_registry.main:app --port 8001
    # Or via Docker: docker run -p 8001:8000 ghcr.io/anjallll/caniollama-registry:latest
    caniollama_registry_url: str = "http://localhost:8001"
    caniollama_enabled: bool = True
    caniollama_timeout: float = Field(default=1.5, gt=0)

    # --- Web search (see agent/websearch.py) ---
    # Defaults to "duckduckgo" (free, no key needed) when unset. Set explicitly to
    # override: "openserp" (self-hosted) or "tavily" (paid, needs API key).
    search_provider: str | None = None  # None = duckduckgo; or "openserp" / "tavily"
    search_api_key: str | None = None  # required for "tavily" only
    search_base_url: str | None = None  # self-hosted "openserp" URL; defaults to :7000
    # Provider-specific settings, for a fallback chain such as "searxng,brave,duckduckgo".
    search_searxng_url: str | None = None  # defaults to http://localhost:8888
    search_brave_api_key: str | None = None
    # Chat-side web search (`"localgate": {"web_search": true}` on a chat request).
    web_search_enabled: bool = True
    web_search_max_sources: int = Field(default=4, ge=1, le=10)
    # Let fetch_url / chat search reach private-network hosts. Off: refuses anything
    # resolving to loopback, RFC 1918, link-local or cloud-metadata addresses.
    fetch_allow_private: bool = False

    # --- Database (SQLite by default; any SQLAlchemy async URL works) ---
    # The default is resolved at load time via load_settings(); a bare Settings()
    # call still gets this string, which tests override explicitly anyway.
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

    # --- Tool execution ---
    tool_timeout_read: float = Field(default=30.0, gt=0)  # timeout for read-only tools
    tool_timeout_write: float = Field(default=60.0, gt=0)  # timeout for write operations
    tool_timeout_search: float = Field(default=20.0, gt=0)  # timeout for web search
    tool_timeout_delegate: float = Field(default=180.0, gt=0)  # timeout for sub-agents
    tool_logging_enabled: bool = True  # log all tool calls for debugging
    # Context window requested for the coding agent (Ollama `num_ctx`), capped at the
    # model's own maximum. Larger windows use more VRAM.
    agent_num_ctx: int = Field(default=16384, ge=2048)
    # Sampling temperature for the coding agent. Low: exact snippets and well-formed tool
    # calls matter far more than variety (Ollama's own default is 0.8).
    agent_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    # Offer the todo_write planning tool. Off by default: in recorded runs it pulled 7B
    # models into planning prose instead of acting. Worth enabling for larger models.
    agent_todo_tool: bool = False
    # Sandboxed command execution for the agent: "auto" picks the strongest sandbox
    # available on this OS (sandbox-exec, bubblewrap, docker/podman); "off" disables
    # the run_command tool. See agent/sandbox.py.
    agent_sandbox: str = "auto"
    agent_sandbox_image: str = "python:3.12-slim"
    tool_timeout_command: float = Field(default=120.0, gt=0)
    # Local voice (`pip install 'localgate[voice]'`): Piper text-to-speech and
    # faster-whisper speech-to-text, served at /v1/audio/* and used by `localgate code --voice`.
    voice_enabled: bool = True
    voice_tts_voice: str = "en_US-lessac-medium"
    voice_dir: str | None = None  # defaults to <data dir>/voices
    voice_auto_download: bool = True  # fetch a missing Piper voice on first use
    voice_stt_model: str = "base"  # faster-whisper model: tiny, base, small, medium, turbo, *.en
    voice_stt_device: str = "auto"  # auto, cpu, cuda

    # --- Prompt cache ---
    cache_enabled: bool = False
    cache_ttl_seconds: int = Field(default=300, ge=0)
    cache_max_entries: int = Field(default=512, ge=1)

    # --- Auth & limits ---
    admin_key: str = Field(default=INSECURE_ADMIN_KEY)
    default_rate_limit_per_min: int = Field(default=60, ge=1)
    # Keep an encrypted copy of new API keys so an admin can view them again. The
    # copy is encrypted with a key derived from admin_key (core/key_vault.py).
    api_key_reveal_enabled: bool = True

    # --- Sharing with other devices ---
    #: Context window sent to Ollama for requests that don't set one (capped by the
    #: model's own maximum). Ollama's server default is far smaller.
    default_num_ctx: int = Field(default=8192, ge=512)
    #: Inference requests run at once per backend; the rest wait in a fair queue.
    max_concurrent_requests: int = Field(default=2, ge=1)
    #: Seconds a request may wait for a slot before getting 503 + Retry-After.
    queue_timeout: float = Field(default=60.0, gt=0)
    #: Requests allowed to wait at once per backend; more get 503 straight away.
    max_queue: int = Field(default=100, ge=1)
    #: Who may reach /admin, /dashboard, the API docs and /metrics (CIDRs, or "*").
    admin_allowed_networks: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_ADMIN_NETWORKS)
    )
    #: The address other devices use, when it isn't this machine's own (a tunnel,
    #: a reverse proxy with a domain). Shown first on the Share page.
    public_url: str | None = None
    #: Several inference servers behind one gateway (JSON list of BackendSpec).
    #: Empty: one backend from LOCALGATE_BACKEND_TYPE / LOCALGATE_BACKEND_URL.
    backends: Annotated[list[BackendSpec], NoDecode] = Field(default_factory=list)
    #: Share rate limits and per-key concurrency across worker processes.
    redis_url: str | None = None
    #: Advertise this gateway on the local network (needs `localgate[discovery]`).
    discovery_enabled: bool = False

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

    @field_validator("admin_allowed_networks", mode="before")
    @classmethod
    def _parse_networks(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                value = json.loads(text)
            else:
                value = [item.strip() for item in text.split(",") if item.strip()]
        networks: list[str] = []
        for item in value or []:
            if item in ("*", "any"):
                networks.extend(["0.0.0.0/0", "::/0"])
                continue
            try:
                networks.append(str(ipaddress.ip_network(item, strict=False)))
            except ValueError as exc:
                raise ValueError(f"admin_allowed_networks: {item!r} is not a network") from exc
        return networks

    @field_validator("backends", mode="before")
    @classmethod
    def _parse_backends(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return json.loads(text) if text else []
        return value

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

    def is_trusted_address(self, host: str | None) -> bool:
        """Whether a client address is inside `admin_allowed_networks`."""
        if not host:
            return False
        try:
            address = ipaddress.ip_address(host.split("%", 1)[0])
        except ValueError:
            # ASGI test clients report "testclient"; unix sockets report nothing.
            return host in ("testclient", "localhost")
        if address.version == 6 and address.ipv4_mapped is not None:
            address = address.ipv4_mapped
        return any(address in ipaddress.ip_network(n) for n in self.admin_allowed_networks)

    @property
    def uses_insecure_admin_key(self) -> bool:
        return self.admin_key == INSECURE_ADMIN_KEY

    def resolve_model(self, requested: str | None) -> str:
        """Map a requested model name through the alias table, defaulting when absent."""
        name = requested or self.default_model
        return self.model_aliases.get(name, name)
