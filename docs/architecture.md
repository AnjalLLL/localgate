# Architecture

## The shape of the thing

localgate sits between your application and your inference server, and adds the management
layer that Ollama, LM Studio and LocalAI deliberately don't have.

```
┌──────────────────────────────────────────────────────────────┐
│  Client (OpenAI SDK, curl, LangChain, anything HTTP)         │
└───────────────────────────┬──────────────────────────────────┘
                            │  OpenAI-compatible HTTP
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                      localgate                                │
│                                                               │
│  middleware/   request id, access log, metrics                │
│  api/          routes — thin; validate, delegate, respond     │
│  core/         business logic — auth, limits, tokens, cache   │
│  memory/       chunk, embed, retrieve, summarize              │
│  db/           repositories — the only code that writes SQL   │
│  backends/     one adapter per inference server               │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  Ollama │ llama.cpp │ vLLM │ any OpenAI-compatible server     │
└──────────────────────────────────────────────────────────────┘
```

## Layers

Requests flow downward, and no layer reaches past the one below it.

```
api/ ──▶ core/ ──▶ backends/
  │        │
  └────────┴──▶ db/repositories/ ──▶ db/models.py
              └▶ memory/
```

**Route handlers never touch the database.** They validate the request, call into `core/`
or a repository, and shape the response. That keeps HTTP concerns out of the logic and
makes the logic testable without constructing a request.

**Repositories are the only code that writes SQL.** Every query lives in
`db/repositories/`, which means changing a table means changing one file, and the
Postgres/SQLite differences are contained.

**Backends are adapters, and nothing above them knows which one is loaded.**

## Key decisions

The reasoning behind the choices that shaped this codebase is recorded as ADRs:

- [0001 — Auth, rate limiting and token accounting are dependencies, not middleware](decisions/0001-dependencies-over-middleware.md)
- [0002 — JSON embeddings over pgvector](decisions/0002-json-embeddings-over-pgvector.md)
- [0003 — SHA-256, not bcrypt, for API keys](decisions/0003-sha256-for-api-keys.md)

## The request path

A single `POST /v1/chat/completions`, in order:

1. **`RequestContextMiddleware`** assigns a request id (or adopts an inbound one), starts
   the timer.
2. **`require_api_key`** resolves the bearer token to an `APIKey` row. A revoked key and a
   nonexistent key are indistinguishable — both simply fail to match a hash.
3. **`enforce_rate_limit`** charges the request against that key's per-minute budget. It
   depends on `require_api_key`, because the limit is *per key* and can't be evaluated
   before the key is known.
4. **Pydantic** validates the body. A malformed request is a 422, not a crash.
5. **The model name is resolved** through the alias table.
6. **Memory is retrieved**: embed the user's message, similarity-search this session's
   chunks, inject the top-K plus the rolling summary as a labelled system message.
7. **The cache is checked**, keyed on the *fully augmented* payload — after aliasing and
   after memory injection, so a hit can't serve a response built from stale context.
8. **The backend is called.**
9. **The turn is persisted**: history, then chunked and embedded into memory, then
   summarized if the session has grown long.
10. **Usage is recorded** against the key, and the token metrics are incremented.

Steps 6 and 9 are best-effort. A failure in either degrades the answer; it never withholds it.

## Streaming

The streaming path has one structural constraint worth stating, because it dictates the code.

By the time the backend fails mid-stream, the `200 OK` and its headers are already on the
wire. **The response cannot retroactively become a 502.** So a mid-stream failure is
reported *inside* the stream, in the same `{"error": {...}}` envelope as everywhere else,
and the stream still terminates with `data: [DONE]` — because the client is blocking on
that sentinel and would otherwise hang.

The second consequence: the turn is persisted *after* the response body has been sent, by
which point FastAPI has already torn down the request's dependencies. The injected database
session is closed. So `_record_turn` opens its own session from the factory rather than
reusing the request-scoped one. This is the kind of thing that works in testing and fails
in production, so it has a dedicated test:
`test_streaming_still_records_usage_after_the_body_is_sent`.

## Backends

Every backend implements one interface (`backends/base.py`): `chat`, `chat_stream`, `embed`,
`list_models`, `health`.

In practice vLLM, llama.cpp, LM Studio and Ollama all speak the same OpenAI HTTP contract,
so they share one implementation — `OpenAICompatBackend` — and differ only declaratively:

```python
class LlamaCppBackend(OpenAICompatBackend):
    name = "llamacpp"
    default_base_url = "http://localhost:8080"
```

Ollama overrides exactly one method (`embed`), because its native `/api/embeddings` route
has been more reliable across releases than its OpenAI shim.

### Plugin system

Backends are discovered through the `localgate.backends` entry-point group. A third-party
package adds one without touching this repository:

```toml
[project.entry-points."localgate.backends"]
my-server = "my_package.backend:MyBackend"
```

Install it alongside localgate, set `LOCALGATE_BACKEND_TYPE=my-server`, and it's live.
`localgate backends` lists everything installed.

The built-in backends are registered through that same mechanism — deliberately, so there is
no privileged path a plugin can't take. A plugin that fails to import is skipped with a
warning naming it, rather than taking the gateway down with it.

## Data model

Five tables, one engine behind all of them — so establishing a database moves *all* of
localgate's data, never an arbitrary subset.

| Table | Holds |
|---|---|
| `api_keys` | Key hash, prefix, rate limit, revocation flag |
| `usage_records` | One row per request: tokens, latency, model, cached flag |
| `conversation_messages` | Chat history — the audit log and the raw material for memory |
| `conversation_summaries` | Rolling summaries, with `covers_until` so the next pass is incremental |
| `memory_chunks` | Chunk text + embedding vector, `kind` in (`turn`, `summary`) |

Schema is versioned with Alembic and there is no `create_all` path — see
[Database Setup](database-setup.md).

## What is deliberately single-process

Two things live in process memory and do **not** coordinate across workers:

- **Rate limits.** N workers means the effective limit is N × what you configured.
- **The prompt cache.** N workers means N independent caches — correct, just less effective.

Both are the honest tradeoff for a tool whose primary deployment is one process on one
machine. Both are behind interfaces (`allow(key_id, limit) -> bool`, `get`/`set`) that a
Redis implementation would satisfy without touching anything above them. See
[Deployment](deployment.md).

## Testing

- **`FakeBackend`** — deterministic echo, and embeddings from a hash of the input. Same text
  gives the same vector, so retrieval tests can assert on exact behaviour instead of guessing
  at a real model's output. No test needs a live Ollama.
- **`test_every_route_is_guarded`** — auth is applied per route (ADR 0001), which means a new
  route that forgets its dependency would be silently public. This test asserts that every
  route requires a key, or is on a short, explicit list of endpoints that are meant to be open.
- **`test_pre_migrations_database_is_adopted_without_losing_data`** — the upgrade path for
  anyone already running localgate, which is exactly the thing you can't test by hand twice.
