# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/).

## [0.6.0] — 2026-07-14

The release where the advertised features actually work.

### Fixed

- **The vLLM, llama.cpp and generic OpenAI backends did not work at all.** They declared
  `InferenceBackend` subclasses that implemented none of its abstract methods, so
  `get_backend("vllm")` raised `TypeError` at startup. Three of the four advertised backends
  were unusable. They now share a real OpenAI-compatible HTTP implementation.
- **Malformed chat requests returned 500 instead of 422.** The handler read `body["messages"]`
  off a raw dict, so a missing field was a crash rather than a client error. Every request
  body is now validated by Pydantic.
- **The admin key was compared with `==`**, which short-circuits at the first differing byte
  and leaks the secret's prefix length through timing. Now `hmac.compare_digest`.
- **The rate limiter never forgot a key it had seen**, leaking one dict entry per key
  forever. Expired windows are now swept.
- **Errors did not use the OpenAI envelope**, so the OpenAI SDK could not parse them. All
  errors — including validation failures and mid-stream backend failures — now return
  `{"error": {"message", "type", "code"}}`.
- **A mid-stream backend failure left clients hanging.** It now reports the error inside the
  stream and still terminates with `[DONE]`.

### Added

- **Migrations.** The schema is versioned with Alembic; `create_all` is gone. Databases from
  0.1–0.2 (tables, but no `alembic_version`) are detected and adopted automatically without
  data loss.
- **A real CLI.** `keys create/list/revoke/usage`, `db upgrade/current/init`, `health`,
  `backends` — the commands the README had been advertising all along.
- **`/v1/models`, `/v1/embeddings`, `/v1/completions`** — previously empty files. The legacy
  completions route is implemented by translating to chat, so it works against backends that
  never implemented it.
- **Rolling conversation summarization.** Past a threshold, older turns are incrementally
  summarized and injected alongside retrieved chunks.
- **Model aliasing.** `LOCALGATE_MODEL_ALIASES='{"fast":"phi4-mini"}'` — swap models without
  touching clients.
- **Prompt caching** (opt-in). Identical prompts skip inference. Off by default because it
  makes sampling deterministic.
- **Plugin system.** Backends are discovered via the `localgate.backends` entry point. The
  built-ins use the same mechanism, so there's no privileged path a plugin can't take.
- **Production hardening.** Structured JSON logs with correlation IDs, Prometheus `/metrics`,
  a liveness/readiness split, graceful shutdown, and fail-fast config validation — production
  now refuses to start with the placeholder admin key.
- **`GET /admin/export`** — every row as JSON, so nobody is locked in.
- **Retrieval score logging and `MEMORY_MIN_SCORE`** — the floor that stops irrelevant chunks
  from filling the context window with noise.
- **`key_prefix`** on keys, so a key can be identified in a listing.
- **`PATCH /admin/keys/{id}`** — change a rate limit without reissuing the key.
- **Documentation.** All seven docs were one-line stubs; they are now written, along with
  three ADRs covering the decisions that shaped the codebase.

### Changed

- `GET /v1/conversations/{id}` now returns an object (messages, summary, chunk count) rather
  than a bare array.
- `POST /admin/keys` returns 201, not 200.
- Usage records now carry latency and a cached flag.

### Security

- SECURITY.md claimed keys were bcrypted; they are SHA-256, which is the *correct* choice for
  high-entropy random keys verified on every request. The document now says so and explains
  why ([ADR 0003](docs/decisions/0003-sha256-for-api-keys.md)).

## [0.2.0] — 2026-07-13

### Added
- Auth, token accounting, RAG memory, dashboard UI, established-database config.
- Test suite with a deterministic `FakeBackend`.

## [0.1.0] — 2026-07-13

### Added
- Initial scaffolding: FastAPI app, Ollama backend, `/v1/chat/completions`, config, CI.
