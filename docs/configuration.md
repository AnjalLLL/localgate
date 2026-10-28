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
| `LOCALGATE_HOST` | `127.0.0.1` | Interface to bind. The default keeps the gateway on this machine; `localgate serve --share` listens on the network (the Docker image sets `0.0.0.0`). |
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
| `LOCALGATE_ADMIN_KEY` | *(generated)* | The server's own secret: signs dashboard sessions and authenticates scripts calling `/admin`. The first `localgate serve` generates one and saves it to the per-user env file; `localgate admin-key` shows it, `--rotate` replaces it. |
| `LOCALGATE_DEFAULT_RATE_LIMIT_PER_MIN` | `60` | Applied to new keys that don't specify one. |
| `LOCALGATE_API_KEY_REVEAL_ENABLED` | `true` | Keep an encrypted copy of new API keys so an admin can show them again. |

People sign in to the dashboard with an email and password (`localgate users create`);
the admin key is for scripts. Per-key limits — allowed models, reply cap, token quota,
expiry, concurrency, whether conversations are stored — live on each key and are changed
with `localgate keys update`, the dashboard, or `PATCH /admin/keys/{id}`.

Rate limits are per-process unless `LOCALGATE_REDIS_URL` is set (see below).

## Sharing with other devices

| Setting | Default | Description |
|---|---|---|
| `LOCALGATE_DEFAULT_NUM_CTX` | `8192` | Context window sent to Ollama for requests that don't set one, capped at the model's own maximum. Ollama's server default is far smaller and silently drops the start of long prompts. |
| `LOCALGATE_MAX_CONCURRENT_REQUESTS` | `2` | Generations run at once per backend; the rest wait in a fair queue (round-robin across keys). Match it to Ollama's `OLLAMA_NUM_PARALLEL`. |
| `LOCALGATE_QUEUE_TIMEOUT` | `60` | Seconds a request may wait for a slot before getting `503` with `Retry-After`. |
| `LOCALGATE_MAX_QUEUE` | `100` | Requests allowed to wait per backend; more get `503` at once. |
| `LOCALGATE_ADMIN_ALLOWED_NETWORKS` | loopback, private LANs, Tailscale, IPv6 ULA | Who may reach the dashboard, `/admin`, `/docs` and `/metrics` (CIDRs, comma-separated, or `*`). The `/v1` API is reachable wherever the port is. Behind a reverse proxy, run `serve --proxy-headers --forwarded-allow-ips <proxy>` so real client addresses are seen. |
| `LOCALGATE_PUBLIC_URL` | *(none)* | The address devices use when it isn't this machine's (a tunnel or a domain). Shown first on the Share page. |
| `LOCALGATE_BACKENDS` | *(empty)* | Several inference servers behind one gateway, as JSON: `[{"name": "big-gpu", "type": "ollama", "url": "http://10.0.0.5:11434", "models": ["llama3:70b"], "max_concurrent": 4}]`. Requests go to a backend that serves the model, least busy first, with failover. Empty: one backend from `BACKEND_TYPE`/`BACKEND_URL`. |
| `LOCALGATE_REDIS_URL` | *(none)* | Share rate limits and per-key concurrency across worker processes. Needs `pip install 'localgate[redis]'`. |
| `LOCALGATE_DISCOVERY_ENABLED` | `false` | Announce the gateway on the local network (on with `serve --share`). Needs `pip install 'localgate[discovery]'`. |

See [Sharing your GPU](sharing.md) for the whole setup.

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
- `BACKEND_TYPE` (or a `BACKENDS` entry's `type`) names a backend that isn't installed
  (the error lists the ones that are);
- the gateway would listen on a non-loopback address with the placeholder admin key;
- `ADMIN_ALLOWED_NETWORKS` contains something that isn't a network;
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
