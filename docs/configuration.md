# Configuration

Every setting is read from an environment variable prefixed with `LOCALGATE_`, or from
a `.env` file in the working directory. Anything unsafe or non-functional is rejected
at startup rather than at the first request — see [Validation](#validation).

## Deployment

| Setting | Default | Description |
|---|---|---|
| `LOCALGATE_ENVIRONMENT` | `development` | `development` or `production`. Production refuses to start with the placeholder admin key. |

## Server

| Setting | Default | Description |
|---|---|---|
| `LOCALGATE_HOST` | `0.0.0.0` | Interface to bind. |
| `LOCALGATE_PORT` | `8000` | Port to bind. |
| `LOCALGATE_CORS_ORIGINS` | *(empty)* | Origins allowed to call the API from a browser. Accepts `http://a,http://b` or a JSON array. Empty means no CORS headers, which is right for server-to-server use. |

## Inference backend

| Setting | Default | Description |
|---|---|---|
| `LOCALGATE_BACKEND_TYPE` | `ollama` | `ollama`, `vllm`, `llamacpp`, `openai_compat`, or any installed plugin. `localgate backends` lists what's available. |
| `LOCALGATE_BACKEND_URL` | `http://localhost:11434` | Where that backend listens. |
| `LOCALGATE_BACKEND_TIMEOUT` | `120` | Seconds to wait for a response. Raise it for large models on CPU. |
| `LOCALGATE_BACKEND_API_KEY` | *(none)* | Sent as a bearer token to the backend, for upstreams that require their own auth. |
| `LOCALGATE_DEFAULT_MODEL` | `llama3` | Used when a request names no model. |
| `LOCALGATE_MODEL_ALIASES` | `{}` | JSON map of friendly name → real model id. |

### Model aliases

Aliases let callers ask for a capability instead of a model id, so you can swap the
model underneath without touching a single client:

```bash
LOCALGATE_MODEL_ALIASES='{"fast": "phi4-mini", "smart": "llama3:70b", "code": "qwen2.5-coder"}'
```

A request for `model: "fast"` is forwarded as `phi4-mini`. Aliases also appear in
`GET /v1/models`, so a caller can discover that "fast" is a name they're allowed to use.

## Database

| Setting | Default | Description |
|---|---|---|
| `LOCALGATE_DATABASE_URL` | `sqlite+aiosqlite:///./localgate.db` | Any SQLAlchemy **async** URL. |

The driver must be an async one — `postgresql+asyncpg://`, not `postgresql://`. See
[Database Setup](database-setup.md), which also covers the Neon-specific gotchas.

A database established through the admin UI is stored in `localgate.config.json` and
**takes precedence over this variable**, because it was connection-tested when it was
saved. `GET /admin/config` always reports which one is actually in use.

## Memory / RAG

| Setting | Default | Description |
|---|---|---|
| `LOCALGATE_MEMORY_ENABLED` | `true` | Master switch for retrieval and storage. |
| `LOCALGATE_EMBEDDING_MODEL` | `nomic-embed-text` | Must be pulled on the backend. |
| `LOCALGATE_CHUNK_SIZE` | `512` | Words per chunk. |
| `LOCALGATE_CHUNK_OVERLAP` | `50` | Words shared between adjacent chunks. Must be less than `CHUNK_SIZE`. |
| `LOCALGATE_MAX_RETRIEVED_CHUNKS` | `5` | Top-K chunks injected per request. |
| `LOCALGATE_MEMORY_MIN_SCORE` | `0.0` | Cosine-similarity floor for injecting a chunk. |
| `LOCALGATE_SUMMARIZE_AFTER_MESSAGES` | `20` | Summarize older turns past this many messages. `0` disables. |

`MEMORY_MIN_SCORE` defaults to `0.0` (no floor) because the right threshold depends
entirely on the embedding model. With `nomic-embed-text`, `0.3`–`0.5` is the usual
range. [RAG Memory](rag-memory.md) explains how to tune it against your own logs.

## Prompt cache

| Setting | Default | Description |
|---|---|---|
| `LOCALGATE_CACHE_ENABLED` | `false` | Serve identical prompts from cache. |
| `LOCALGATE_CACHE_TTL_SECONDS` | `300` | Entry lifetime. `0` means never expire. |
| `LOCALGATE_CACHE_MAX_ENTRIES` | `512` | Bound on cache size; least-recently-used entries are evicted. |

**Caching is off by default, and that is deliberate.** Two identical requests at
`temperature=0.8` are *supposed* to produce different completions; a cache returns the
first one twice. That is a real change in behaviour, so you opt into it. It is a large
win for deterministic workloads (`temperature=0`, classification, extraction) and the
wrong choice for creative ones.

**Memory and the cache interact, in a way that surprises people.** The cache key is the
*fully augmented* payload — after model aliasing and after memory injection. So the same
prompt asked twice **in the same session** is a cache *miss* the second time: the first
turn is now in memory, so the second request carries retrieved context the first one
didn't. It is genuinely a different prompt, and treating it as the same one would serve a
response built from stale context.

The practical consequence: caching pays off most for **stateless** requests (no
`X-Session-ID`, or `MEMORY_ENABLED=false`) — classification, extraction, embedding-style
workloads. In a long conversation it will mostly miss, and that is correct.

A cache hit still **bills the key and still records the turn**. It saves the inference, not
the bookkeeping.

The cache is per-process. Multiple workers means multiple independent caches — correct,
just less effective.

## Coding agent

| Setting | Default | Description |
|---|---|---|
| `LOCALGATE_AGENT_NUM_CTX` | `16384` | Context window the agent requests from Ollama (`num_ctx`), capped at the model's maximum. Larger windows use more VRAM. |
| `LOCALGATE_AGENT_TEMPERATURE` | `0.2` | Sampling temperature for the agent. Low values make tool calls and exact edit snippets more reliable. |
| `LOCALGATE_AGENT_TODO_TOOL` | `false` | Offer the `todo_write` planning tool. Helpful with larger models; in testing it pulled 7B models into planning prose instead of acting. |
| `LOCALGATE_AGENT_SANDBOX` | `auto` | `auto` picks the strongest available sandbox for `run_command` (`seatbelt` on macOS, `bubblewrap` on Linux, then `docker`/`podman`). Name one to require it, or `off`. |
| `LOCALGATE_AGENT_SANDBOX_IMAGE` | `python:3.12-slim` | Image for the Docker/Podman sandbox. Use one with your project's toolchain. |
| `LOCALGATE_TOOL_TIMEOUT_COMMAND` | `120` | Default seconds before a sandboxed command is stopped. |
| `LOCALGATE_TOOL_TIMEOUT_SEARCH` / `_READ` / `_WRITE` / `_DELEGATE` | `20` / `30` / `60` / `180` | Per-tool timeouts. |
| `LOCALGATE_TOOL_LOGGING_ENABLED` | `true` | Store a redacted log of every agent tool call (shown in the dashboard's Agent runs page). |

## Web search

| Setting | Default | Description |
|---|---|---|
| `LOCALGATE_SEARCH_PROVIDER` | `duckduckgo` | One provider or a comma-separated fallback chain: `searxng`, `openserp`, `brave`, `tavily`, `duckduckgo`. |
| `LOCALGATE_SEARCH_SEARXNG_URL` | `http://localhost:8888` | Your SearXNG instance (JSON format must be enabled). |
| `LOCALGATE_SEARCH_BASE_URL` | `http://localhost:7000` | OpenSERP URL (also used for SearXNG if the specific setting is unset). |
| `LOCALGATE_SEARCH_BRAVE_API_KEY` | — | Brave Search API key. |
| `LOCALGATE_SEARCH_API_KEY` | — | Tavily API key. |
| `LOCALGATE_WEB_SEARCH_ENABLED` | `true` | Allow chat requests to ask for web search (`"localgate": {"web_search": true}`). |
| `LOCALGATE_WEB_SEARCH_MAX_SOURCES` | `4` | Sources fetched and cited per chat answer. |
| `LOCALGATE_FETCH_ALLOW_PRIVATE` | `false` | Let `fetch_url` and chat search reach loopback/private-network hosts. Leave off unless you need it. |

## Voice

Needs `pip install 'localgate[voice]'`.

| Setting | Default | Description |
|---|---|---|
| `LOCALGATE_VOICE_ENABLED` | `true` | Serve `/v1/audio/*` and allow `/voice`/`/listen` in `localgate code`. |
| `LOCALGATE_VOICE_TTS_VOICE` | `en_US-lessac-medium` | Default Piper voice. |
| `LOCALGATE_VOICE_DIR` | `<data dir>/voices` | Where Piper voices (`.onnx` + `.onnx.json`) live. |
| `LOCALGATE_VOICE_AUTO_DOWNLOAD` | `true` | Download a missing voice on first use. |
| `LOCALGATE_VOICE_STT_MODEL` | `base` | faster-whisper model: `tiny`, `base`, `small`, `medium`, `turbo`, or a `.en` variant. |
| `LOCALGATE_VOICE_STT_DEVICE` | `auto` | `auto`, `cpu` or `cuda`. |

## Auth and limits

| Setting | Default | Description |
|---|---|---|
| `LOCALGATE_ADMIN_KEY` | `change-me-in-production` | Guards every `/admin` route. |
| `LOCALGATE_DEFAULT_RATE_LIMIT_PER_MIN` | `60` | Applied to new keys that don't specify one. |

Generate a real admin key with `openssl rand -hex 32`. Per-key limits are stored on the
key and changed with `localgate keys update` or `PATCH /admin/keys/{id}`.

Rate limits are per-process. Running N workers means the effective limit is N times what
you configured; see [Deployment](deployment.md).

## Observability

| Setting | Default | Description |
|---|---|---|
| `LOCALGATE_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `LOCALGATE_LOG_FORMAT` | `console` | `console` for humans, `json` for log ingestion. |
| `LOCALGATE_METRICS_ENABLED` | `true` | Serve Prometheus metrics at `/metrics`. |

## Validation

Configuration errors are the most common way a self-hosted service fails, and the worst
version is the silent one. localgate refuses to start when:

- `ENVIRONMENT=production` and `ADMIN_KEY` is still the documented placeholder — anyone
  who has read these docs could otherwise mint API keys against your gateway;
- `CHUNK_OVERLAP >= CHUNK_SIZE` — the chunking window would never advance;
- `BACKEND_TYPE` names a backend that isn't installed (the error lists the ones that are);
- `PORT` is outside 1–65535.

In development the placeholder admin key is allowed, because local use has to stay
zero-config — but it is logged as a warning at startup and reported in `GET /health`.

## Full example

```bash
# .env
LOCALGATE_ENVIRONMENT=production
LOCALGATE_ADMIN_KEY=63a1c8f4e29b7d5a1e4f8c2b9d6a3e7f0b5c8d1a4e7f2b9c6d3a0e5f8b1c4d7a

LOCALGATE_BACKEND_TYPE=ollama
LOCALGATE_BACKEND_URL=http://localhost:11434
LOCALGATE_DEFAULT_MODEL=llama3
LOCALGATE_MODEL_ALIASES={"fast":"phi4-mini","smart":"llama3:70b"}

LOCALGATE_DATABASE_URL=postgresql+asyncpg://user:pass@host/localgate?ssl=require

LOCALGATE_MEMORY_ENABLED=true
LOCALGATE_EMBEDDING_MODEL=nomic-embed-text
LOCALGATE_MEMORY_MIN_SCORE=0.35

LOCALGATE_LOG_FORMAT=json
```
