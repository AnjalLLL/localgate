# localgate

**Turn any local LLM into a managed API — real API keys, token accounting, and RAG memory
that makes a small model remember far more than its context window holds.**

[![CI](https://github.com/AnjalLLL/localgate/actions/workflows/ci.yml/badge.svg)](https://github.com/AnjalLLL/localgate/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Why

Ollama, LM Studio and LocalAI solve model *serving*. They deliberately don't solve anything
around it:

- **No API key management.** No per-user keys, no revocation, no usage tracking.
- **No memory past the context window.** Your 8K model forgets everything beyond 8K tokens.
- **No token accounting.** You guess at what you've spent.
- **No database story.** You wire up Postgres yourself.

localgate is the management layer. It sits between your app and your inference server and
adds all four — without touching how you serve models.

## Quick start

```bash
git clone https://github.com/AnjalLLL/localgate.git && cd localgate
uv sync --all-extras

ollama serve                     # your inference backend
ollama pull llama3
ollama pull nomic-embed-text     # enables RAG memory

uv run localgate db upgrade
uv run localgate keys create --name my-app      # prints your key, once
uv run localgate serve
```

Now use it like OpenAI, because it *is* the OpenAI API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="lg_9f3a...")

response = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

Full walkthrough: **[Getting Started](docs/getting-started.md)**.

## The memory bit

This is the part that isn't a proxy. Send an `X-Session-ID` and the gateway stores each
turn, embeds it, and retrieves what's relevant on later turns:

```python
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="lg_9f3a...",
    default_headers={"X-Session-ID": "conversation-1"},
)

client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "My name is Ana and I prefer Postgres."}],
)

# A separate request. No history sent. The model still knows.
client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "What database do I prefer?"}],
)
# → "You prefer Postgres."
```

The model answers correctly not because you sent the history, but because the gateway
retrieved it. Past a threshold, older turns are folded into a rolling summary, so the
context window holds the *useful* part of a long conversation rather than the most recent
part of it. See **[RAG Memory](docs/rag-memory.md)**.

## Features

- **OpenAI-compatible** — works with any OpenAI SDK, LangChain, or curl. Unknown fields are
  forwarded to the backend, so your sampling knobs keep working.
- **API key management** — create, revoke, and rate-limit per key. Hashed, never stored raw.
- **RAG memory** — automatic chunking, embedding, retrieval, and rolling summarization.
- **Token accounting** — prompt/completion tokens per key, per model, over time.
- **Any database** — SQLite with zero config; Postgres or Neon with one env var.
- **Any backend** — Ollama, vLLM, llama.cpp, or any OpenAI-compatible server. Third parties
  can add more via an entry point, no fork required.
- **Model aliasing** — map `fast` → `phi4-mini` and swap models without touching clients.
- **Prompt caching** — opt-in; identical prompts skip inference entirely.
- **Production-ready** — structured JSON logs with correlation IDs, Prometheus metrics,
  liveness/readiness split, graceful shutdown, fail-fast config validation.
- **Dashboard** — keys, usage and conversations in the browser, at `/dashboard/`.

## CLI

```bash
localgate serve                          # start the gateway
localgate health                         # is the backend up? is the DB migrated?
localgate backends                       # what adapters are installed

localgate keys create --name my-app      # create a key (printed once)
localgate keys list                      # every key, active and revoked
localgate keys revoke <id>               # revoke (history is kept)
localgate keys usage <id>                # token usage for one key

localgate db upgrade                     # apply migrations
localgate db current                     # current schema revision
```

The CLI talks to the database directly, not to a running server — because `keys create` has
to work before you have a key, and `db upgrade` has to work when the server won't start.

## Documentation

| | |
|---|---|
| [Getting Started](docs/getting-started.md) | Zero to working gateway |
| [Configuration](docs/configuration.md) | Every setting |
| [API Reference](docs/api-reference.md) | Every endpoint |
| [Database Setup](docs/database-setup.md) | SQLite → Postgres → Neon |
| [RAG Memory](docs/rag-memory.md) | How memory works, and how to tune it |
| [Architecture](docs/architecture.md) | How it's built, and why |
| [Deployment](docs/deployment.md) | Running it somewhere real |
| [Decisions](docs/decisions/) | ADRs for the choices that shaped the codebase |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Adding a backend means writing one class. Issues
tagged `good-first-issue` are a good place to start.

## License

[MIT](LICENSE)
