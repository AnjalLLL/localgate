# LocalLLM Gateway — Complete Project Roadmap

> **One-liner:** An open-source API gateway that turns any local LLM into a fully managed API with authentication, RAG-powered memory extension, database connectors, and token accounting.

---

## Table of Contents

1. [Project Vision & Unique Value](#1-project-vision--unique-value)
2. [Architecture Overview](#2-architecture-overview)
3. [Tech Stack](#3-tech-stack)
4. [Folder Structure](#4-folder-structure)
5. [Coding Patterns & Conventions](#5-coding-patterns--conventions)
6. [Development Phases & Milestones](#6-development-phases--milestones)
7. [All Required Markdown Files](#7-all-required-markdown-files)
8. [Open-Source Repo Setup Guide](#8-open-source-repo-setup-guide)
9. [Package Release Strategy](#9-package-release-strategy)
10. [Additional Features (Beyond Your Core Idea)](#10-additional-features-beyond-your-core-idea)
11. [Community & Growth Playbook](#11-community--growth-playbook)

---

## 1. Project Vision & Unique Value

### The Gap You're Filling

Ollama, LM Studio, and LocalAI solve model serving. Nobody has built the **management layer** on top. Your project fills four specific gaps:

1. **Real API key management** — issue, revoke, rate-limit, and track per-user keys (Ollama has zero concept of this)
2. **Context extension via RAG** — make a 4K/8K model "remember" 100K+ tokens by automatically chunking, embedding, and retrieving conversation history from a vector database
3. **Database connector abstraction** — plug-and-play support for Neon, local Postgres, SQLite, with a unified schema for embeddings and chat history
4. **Token accounting dashboard** — prompt tokens, completion tokens, cost estimation, per-key usage stats

### Suggested Project Name

**`localgate`** (or `localgate`, `llm-gateway` — pick one and own it)

Throughout this document I'll use **`localgate`**.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     Client Application                   │
│         (uses OpenAI SDK, curl, any HTTP client)         │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP (OpenAI-compatible)
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    localgate API Gateway                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ Auth &   │  │ Context  │  │ Token    │  │ Rate    │ │
│  │ Key Mgmt │  │ Extension│  │ Counter  │  │ Limiter │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
│  ┌──────────────────────────────────────────────────────┐│
│  │              RAG / Memory Layer                      ││
│  │   Chunking → Embedding → Storage → Retrieval        ││
│  └──────────────────────────────────────────────────────┘│
│  ┌──────────────────────────────────────────────────────┐│
│  │           Database Connector Abstraction             ││
│  │   SQLite │ PostgreSQL │ Neon │ (pluggable drivers)   ││
│  └──────────────────────────────────────────────────────┘│
└──────────────────────────┬──────────────────────────────┘
                           │ Forward to inference backend
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Inference Backend (pick one)                 │
│         Ollama │ llama.cpp │ vLLM │ HF Transformers      │
└─────────────────────────────────────────────────────────┘
```

### Data Flow (Single Request)

1. Client sends `POST /v1/chat/completions` with an API key in the `Authorization` header
2. Gateway validates the key, checks rate limits
3. If RAG is enabled for this key/session, the gateway retrieves relevant context chunks from the vector DB and injects them into the system prompt
4. The augmented request is forwarded to the configured inference backend (Ollama, llama.cpp, etc.)
5. Response streams back through the gateway, which counts tokens, logs usage, and stores the conversation turn for future RAG retrieval
6. Client receives a standard OpenAI-shaped response

---

## 3. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| **Language** | Python 3.10+ | Largest LLM ecosystem, FastAPI is battle-tested, lowest barrier for contributors |
| **API Framework** | FastAPI + Uvicorn | Async, streaming SSE support, auto OpenAPI docs, easy middleware |
| **Database ORM** | SQLAlchemy 2.0 (async) | Supports SQLite, Postgres, Neon out of the box with one codebase |
| **Vector Storage** | pgvector (Postgres) or sqlite-vss (SQLite) | No separate vector DB to install — reuse the user's existing database |
| **Embedding** | Local embedding via Ollama or sentence-transformers | No external API needed — stays fully local |
| **Migrations** | Alembic | Industry standard for SQLAlchemy |
| **CLI** | Typer (built on Click) | Feels like a proper CLI tool, auto help generation |
| **Config** | Pydantic Settings + YAML/TOML | Type-safe config with env var overrides |
| **Testing** | pytest + pytest-asyncio + httpx | Async test client for FastAPI |
| **Packaging** | pyproject.toml (PEP 621) | Modern Python packaging, single source of truth |
| **Containerization** | Docker + docker-compose | One-command setup for users who want Postgres |

---

## 4. Folder Structure

```
localgate/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   └── config.yml
│   ├── PULL_REQUEST_TEMPLATE.md
│   ├── workflows/
│   │   ├── ci.yml                  # lint + test on every PR
│   │   ├── release.yml             # publish to PyPI on tag
│   │   └── docker.yml              # build & push Docker image
│   ├── FUNDING.yml
│   └── CODEOWNERS
│
├── docs/
│   ├── getting-started.md
│   ├── configuration.md
│   ├── api-reference.md
│   ├── database-setup.md
│   ├── rag-memory.md
│   ├── architecture.md
│   └── deployment.md
│
├── src/
│   └── localgate/
│       ├── __init__.py             # version string
│       ├── __main__.py             # `python -m localgate` entry point
│       ├── cli.py                  # Typer CLI commands
│       ├── config.py               # Pydantic Settings model
│       ├── app.py                  # FastAPI app factory
│       │
│       ├── api/                    # Route handlers (thin controllers)
│       │   ├── __init__.py
│       │   ├── chat.py             # /v1/chat/completions
│       │   ├── completions.py      # /v1/completions (legacy)
│       │   ├── embeddings.py       # /v1/embeddings
│       │   ├── models.py           # /v1/models (list available)
│       │   ├── keys.py             # /admin/keys CRUD
│       │   └── usage.py            # /admin/usage stats
│       │
│       ├── core/                   # Business logic (no HTTP awareness)
│       │   ├── __init__.py
│       │   ├── auth.py             # API key validation, hashing
│       │   ├── rate_limiter.py     # Token bucket / sliding window
│       │   ├── token_counter.py    # tiktoken / model-specific counting
│       │   └── streaming.py        # SSE stream wrapper
│       │
│       ├── backends/               # Inference backend adapters
│       │   ├── __init__.py
│       │   ├── base.py             # Abstract base class
│       │   ├── ollama.py           # Ollama HTTP adapter
│       │   ├── llamacpp.py         # llama.cpp server adapter
│       │   ├── vllm.py             # vLLM adapter
│       │   └── openai_compat.py    # Generic OpenAI-compatible adapter
│       │
│       ├── memory/                 # Context extension / RAG
│       │   ├── __init__.py
│       │   ├── chunker.py          # Text splitting strategies
│       │   ├── embedder.py         # Local embedding generation
│       │   ├── retriever.py        # Similarity search
│       │   ├── context_builder.py  # Prompt augmentation logic
│       │   └── summarizer.py       # Conversation summarization
│       │
│       ├── db/                     # Database layer
│       │   ├── __init__.py
│       │   ├── engine.py           # Async engine factory
│       │   ├── models.py           # SQLAlchemy ORM models
│       │   ├── migrations/         # Alembic migrations
│       │   │   ├── env.py
│       │   │   ├── versions/
│       │   │   └── alembic.ini
│       │   └── repositories/       # Data access layer
│       │       ├── __init__.py
│       │       ├── keys.py
│       │       ├── usage.py
│       │       ├── conversations.py
│       │       └── embeddings.py
│       │
│       ├── middleware/              # FastAPI middleware
│       │   ├── __init__.py
│       │   ├── auth_middleware.py
│       │   ├── rate_limit_middleware.py
│       │   ├── token_counting_middleware.py
│       │   └── logging_middleware.py
│       │
│       └── dashboard/              # Optional web UI
│           ├── __init__.py
│           ├── routes.py
│           └── static/             # Minimal HTML/JS dashboard
│               ├── index.html
│               └── dashboard.js
│
├── tests/
│   ├── conftest.py                 # Shared fixtures
│   ├── unit/
│   │   ├── test_auth.py
│   │   ├── test_token_counter.py
│   │   ├── test_chunker.py
│   │   ├── test_rate_limiter.py
│   │   └── test_config.py
│   ├── integration/
│   │   ├── test_chat_endpoint.py
│   │   ├── test_key_management.py
│   │   ├── test_rag_pipeline.py
│   │   └── test_database_connectors.py
│   └── e2e/
│       └── test_full_flow.py
│
├── examples/
│   ├── openai_sdk_example.py       # Use with OpenAI Python SDK
│   ├── curl_examples.sh            # Raw curl commands
│   ├── langchain_example.py        # Integrate with LangChain
│   └── docker-compose.postgres.yml # Full stack with Postgres
│
├── scripts/
│   ├── dev-setup.sh                # One-command dev environment
│   └── benchmark.py                # Performance benchmarking
│
├── pyproject.toml                  # Package metadata + deps + tools
├── Dockerfile
├── docker-compose.yml
├── Makefile                        # make test, make lint, make run
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
│
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── CHANGELOG.md
├── LICENSE                         # MIT or Apache-2.0
├── SECURITY.md
└── ROADMAP.md                      # Public-facing roadmap
```

---

## 5. Coding Patterns & Conventions

### 5.1 Layered Architecture

Every request flows through clean layers. No layer skips another.

```
Route Handler (api/) → calls → Core Logic (core/) → calls → Backend Adapter (backends/)
                                     ↕
                              Database Repository (db/repositories/)
                                     ↕
                              Memory/RAG Layer (memory/)
```

**Rule: Route handlers never touch the database directly.** They call core logic functions, which call repositories.

### 5.2 Backend Adapter Pattern

Every inference backend implements one abstract interface:

```python
# src/localgate/backends/base.py
from abc import ABC, abstractmethod
from typing import AsyncIterator
from localgate.core.types import ChatRequest, ChatResponse, ChatChunk

class InferenceBackend(ABC):
    """All backends implement this interface."""

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Non-streaming chat completion."""
        ...

    @abstractmethod
    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        """Streaming chat completion (yields SSE chunks)."""
        ...

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Return available model names."""
        ...

    @abstractmethod
    async def health(self) -> bool:
        """Check if the backend is reachable."""
        ...
```

Adding a new backend (e.g., HuggingFace TGI) means writing one file that implements this class. Nothing else changes.

### 5.3 Repository Pattern for Database

```python
# src/localgate/db/repositories/keys.py
class APIKeyRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, name: str, rate_limit: int = 60) -> APIKey:
        key = APIKey(name=name, key_hash=hash_key(generate_key()), rate_limit=rate_limit)
        self.session.add(key)
        await self.session.commit()
        return key

    async def get_by_hash(self, key_hash: str) -> APIKey | None:
        stmt = select(APIKey).where(APIKey.key_hash == key_hash)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def revoke(self, key_id: str) -> None:
        stmt = update(APIKey).where(APIKey.id == key_id).values(revoked=True)
        await self.session.execute(stmt)
        await self.session.commit()
```

### 5.4 Dependency Injection via FastAPI

```python
# src/localgate/app.py
from fastapi import FastAPI, Depends
from localgate.config import Settings
from localgate.backends import get_backend
from localgate.db.engine import get_session

def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="localgate", version=settings.version)

    # Register backend as app state
    app.state.backend = get_backend(settings.backend_type, settings.backend_url)
    app.state.settings = settings

    # Include routers
    from localgate.api import chat, models, keys, usage
    app.include_router(chat.router)
    app.include_router(models.router)
    app.include_router(keys.router, prefix="/admin")
    app.include_router(usage.router, prefix="/admin")

    return app
```

### 5.5 Config via Pydantic Settings

```python
# src/localgate/config.py
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Backend
    backend_type: str = "ollama"          # ollama | llamacpp | vllm
    backend_url: str = "http://localhost:11434"
    default_model: str = "llama3"

    # Database
    database_url: str = "sqlite+aiosqlite:///./localgate.db"

    # Memory / RAG
    memory_enabled: bool = True
    embedding_model: str = "nomic-embed-text"
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_retrieved_chunks: int = 5

    # Auth
    admin_key: str = Field(default="change-me-in-production")

    class Config:
        env_prefix = "LOCALGATE_"
        env_file = ".env"
```

Every setting is overridable via environment variable (`LOCALGATE_BACKEND_TYPE=vllm`), `.env` file, or YAML config.

### 5.6 Naming & Style Conventions

- **Files:** `snake_case.py` always
- **Classes:** `PascalCase`
- **Functions/variables:** `snake_case`
- **Constants:** `UPPER_SNAKE_CASE`
- **Type hints:** Required everywhere, use `X | None` over `Optional[X]`
- **Docstrings:** Google style
- **Line length:** 100 chars (configured in ruff)
- **Formatter:** ruff format
- **Linter:** ruff check
- **Pre-commit:** ruff + mypy + pytest on every commit

---

## 6. Development Phases & Milestones

### Phase 1 — Foundation (v0.1.0) — Weeks 1–3

**Goal:** A working proxy that passes requests to Ollama with API key auth.

| Task | Details |
|---|---|
| Project scaffolding | pyproject.toml, folder structure, Makefile, CI |
| Config system | Pydantic Settings with .env support |
| Ollama backend adapter | Implement `InferenceBackend` for Ollama |
| `/v1/chat/completions` | Streaming + non-streaming, OpenAI-compatible |
| `/v1/models` | List models from the backend |
| API key auth (basic) | Single admin key, stored in config |
| Token counting | Count prompt + completion tokens per request |
| Basic tests | Unit tests for auth, token counting; integration test for chat |
| README + LICENSE | Enough to make the repo usable |

**Release: `v0.1.0` — "It works, it proxies, it counts tokens."**

### Phase 2 — API Key Management (v0.2.0) — Weeks 4–5

| Task | Details |
|---|---|
| Database setup | SQLAlchemy models + Alembic migrations |
| SQLite default | Zero-config database that works out of the box |
| Key CRUD endpoints | `POST/GET/DELETE /admin/keys` |
| Key hashing | Store bcrypt hashes, never plaintext |
| Per-key rate limiting | Token bucket per key, configurable limits |
| Usage logging | Log every request with key_id, model, token counts, latency |
| `/admin/usage` endpoint | Query usage stats by key, model, time range |
| CLI commands | `localgate keys create`, `localgate keys list`, `localgate keys revoke` |

**Release: `v0.2.0` — "Real multi-user API key management."**

### Phase 3 — Database Connectors (v0.3.0) — Weeks 6–7

| Task | Details |
|---|---|
| PostgreSQL support | Auto-detect and configure for Postgres connection strings |
| Neon support | Neon uses standard Postgres protocol, so test + document |
| Connection pooling | asyncpg + SQLAlchemy async pool |
| pgvector extension | Auto-create vector columns when Postgres is detected |
| sqlite-vss fallback | Vector search for SQLite users |
| Migration system | Alembic auto-detects DB type, runs appropriate migrations |
| CLI: `localgate db init` | Initialize the database schema |
| CLI: `localgate db migrate` | Run pending migrations |

**Release: `v0.3.0` — "Plug in any database."**

### Phase 4 — RAG Memory Layer (v0.4.0) — Weeks 8–11

This is the core differentiator. Build it carefully.

| Task | Details |
|---|---|
| Chunker | Recursive character splitter with configurable chunk_size and overlap |
| Local embedder | Use Ollama's embedding endpoint or sentence-transformers |
| Vector storage | Store embeddings in pgvector or sqlite-vss via the DB abstraction |
| Retriever | Cosine similarity search, return top-K chunks |
| Context builder | Inject retrieved chunks into system prompt with clear formatting |
| Conversation history storage | Auto-save each turn to the DB, linked to session_id |
| Session management | `X-Session-ID` header or auto-generated session IDs |
| Summarizer | When history exceeds a threshold, summarize older turns |
| Config toggles | Enable/disable RAG per-key, per-request, globally |
| Retrieval quality metrics | Log retrieval scores, allow tuning |

**Release: `v0.4.0` — "Small models now remember everything."**

### Phase 5 — Dashboard & DX (v0.5.0) — Weeks 12–14

| Task | Details |
|---|---|
| Web dashboard | Minimal HTML/JS dashboard served by the gateway |
| Key management UI | Create, view, revoke keys from the browser |
| Usage charts | Token usage over time, per-key breakdown |
| Session explorer | Browse conversation history and retrieved chunks |
| Health monitor | Show backend status, model availability, DB connection |
| Quickstart wizard | Interactive setup that walks through first-time config |
| OpenAPI docs | Auto-generated and customized Swagger UI at `/docs` |

**Release: `v0.5.0` — "You can see everything happening."**

### Phase 6 — Multi-Backend & Polish (v0.6.0) — Weeks 15–17

| Task | Details |
|---|---|
| llama.cpp backend | Direct llama.cpp server adapter |
| vLLM backend | vLLM adapter with batching support |
| Generic OpenAI adapter | Connect to any OpenAI-compatible server |
| Backend health checks | Auto-failover if a backend goes down |
| Model aliasing | Map friendly names to backend-specific model IDs |
| Request queuing | Queue requests when backend is overloaded |
| Docker image | Multi-stage Dockerfile, published to GHCR |
| docker-compose examples | One-click setups for various configurations |

**Release: `v0.6.0` — "Works with everything."**

### Phase 7 — Production Hardening (v1.0.0) — Weeks 18–22

| Task | Details |
|---|---|
| Security audit | Review auth flow, key storage, input validation |
| Load testing | Benchmark with locust/k6, document performance |
| Structured logging | JSON logs with correlation IDs |
| Prometheus metrics | `/metrics` endpoint for monitoring |
| Graceful shutdown | Drain in-flight requests on SIGTERM |
| Config validation | Fail fast with clear errors on bad config |
| Comprehensive docs | Full documentation site (MkDocs or similar) |
| Plugin system | Allow third-party backends and memory stores |
| Stable API contract | Semantic versioning promise from v1.0.0 |

**Release: `v1.0.0` — "Production-ready."**

---

## 7. All Required Markdown Files

Below is the full content for every markdown file your repo needs.

---

### 7.1 README.md

```markdown
# localgate

Turn any local LLM into a fully managed API — with real API keys, token accounting,
RAG-powered memory extension, and plug-and-play database support.

## Why localgate?

Ollama, LM Studio, and LocalAI are great at serving models. But they give you zero
infrastructure around that serving:

- **No API key management.** No per-user keys, no usage tracking, no revocation.
- **No memory beyond the context window.** Your 8K model forgets everything after 8K tokens.
- **No database integration.** You wire up Postgres/Neon yourself.
- **No token accounting.** You guess how much you've used.

localgate sits between your app and your inference backend, adding everything that's missing.

## Quick Start

### Install

pip install localgate

### Start (with Ollama as the backend)

# Make sure Ollama is running: ollama serve
localgate start

### That's it. Now use it like OpenAI:

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "Hello!"}]
)

## Features

- **OpenAI-compatible API** — works with any OpenAI SDK or tool
- **API key management** — create, revoke, and rate-limit keys via CLI or API
- **Token accounting** — track prompt/completion tokens per key, per session
- **RAG memory** — automatically extends your model's effective memory using retrieval
- **Database flexibility** — SQLite (zero-config), PostgreSQL, or Neon
- **Multiple backends** — Ollama, llama.cpp, vLLM, or any OpenAI-compatible server
- **Minimal hardware** — runs on CPU; uses GPU automatically when available
- **Dashboard** — web UI for key management, usage stats, and session browsing

## Configuration

Copy `.env.example` to `.env` and customize:

LOCALGATE_BACKEND_TYPE=ollama
LOCALGATE_BACKEND_URL=http://localhost:11434
LOCALGATE_DATABASE_URL=sqlite+aiosqlite:///./localgate.db
LOCALGATE_MEMORY_ENABLED=true
LOCALGATE_ADMIN_KEY=your-secure-admin-key

See [docs/configuration.md](docs/configuration.md) for all options.

## CLI Reference

localgate start                        # Start the gateway server
localgate keys create --name "my-app"  # Create a new API key
localgate keys list                    # List all keys
localgate keys revoke <key-id>         # Revoke a key
localgate db init                      # Initialize the database
localgate db migrate                   # Run pending migrations
localgate health                       # Check backend connectivity

## Documentation

- [Getting Started](docs/getting-started.md)
- [Configuration](docs/configuration.md)
- [API Reference](docs/api-reference.md)
- [Database Setup](docs/database-setup.md)
- [RAG Memory](docs/rag-memory.md)
- [Architecture](docs/architecture.md)
- [Deployment](docs/deployment.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). We welcome all contributions.

## License

[MIT](LICENSE)
```

---

### 7.2 CONTRIBUTING.md

```markdown
# Contributing to localgate

Thank you for wanting to contribute! This guide will help you get started.

## Development Setup

1. **Fork and clone the repo**

   git clone https://github.com/YOUR_USERNAME/localgate.git
   cd localgate

2. **Create a virtual environment**

   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. **Install in development mode**

   pip install -e ".[dev]"

4. **Install pre-commit hooks**

   pre-commit install

5. **Run tests**

   make test

6. **Start the dev server**

   make run

## Project Structure

- `src/localgate/api/` — Route handlers (thin HTTP layer)
- `src/localgate/core/` — Business logic (no HTTP awareness)
- `src/localgate/backends/` — Inference backend adapters
- `src/localgate/memory/` — RAG and context extension
- `src/localgate/db/` — Database models and repositories
- `tests/` — Tests (unit, integration, e2e)

## Coding Standards

- **Type hints** on all function signatures
- **Docstrings** in Google style on all public functions
- **Tests** for every new feature or bug fix
- **Ruff** for linting and formatting (runs in pre-commit)
- **No print statements** — use `structlog` logger

## Pull Request Process

1. Create a feature branch from `main`: `git checkout -b feat/my-feature`
2. Make your changes with tests
3. Run `make lint` and `make test` locally
4. Push and open a PR against `main`
5. Fill out the PR template
6. Wait for CI to pass and a maintainer review

## Types of Contributions

**Good first issues:** Look for the `good-first-issue` label.

**Backend adapters:** Implement `InferenceBackend` in `src/localgate/backends/`.

**Database drivers:** Add support for new databases in `src/localgate/db/`.

**Documentation:** Improve docs, add examples, fix typos.

**Bug reports:** Open an issue with reproduction steps.

## Code of Conduct

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md).
All contributors are expected to uphold it.
```

---

### 7.3 CODE_OF_CONDUCT.md

```markdown
# Code of Conduct

## Our Pledge

We are committed to making participation in this project a harassment-free
experience for everyone, regardless of age, body size, disability, ethnicity,
gender identity, level of experience, nationality, personal appearance, race,
religion, or sexual identity and orientation.

## Our Standards

**Positive behavior includes:**

- Using welcoming and inclusive language
- Respecting differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community

**Unacceptable behavior includes:**

- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

## Enforcement

Instances of abusive behavior may be reported by contacting the project team
at [your-email@example.com]. All complaints will be reviewed and investigated
promptly and fairly.

## Attribution

This Code of Conduct is adapted from the Contributor Covenant, version 2.1.
```

---

### 7.4 CHANGELOG.md

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Initial project scaffolding
- Ollama backend adapter
- OpenAI-compatible /v1/chat/completions endpoint
- Basic API key authentication
- Token counting per request

## [0.1.0] - YYYY-MM-DD

### Added
- First public release
- Proxy requests to Ollama with OpenAI-compatible API
- Single admin API key authentication
- Token counting (prompt + completion)
- CLI: `localgate start`
- Configuration via .env and environment variables
```

---

### 7.5 SECURITY.md

```markdown
# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.x.x   | Yes               |
| 0.x.x   | Best-effort       |

## Reporting a Vulnerability

**Do not open a public issue for security vulnerabilities.**

Instead, email [your-email@example.com] with:

1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if any)

You will receive a response within 48 hours. We will work with you to
understand the issue and coordinate a fix before any public disclosure.

## Security Practices

- API keys are stored as bcrypt hashes, never in plaintext
- Admin endpoints require a separate admin key
- Rate limiting is enforced per-key to prevent abuse
- Database connections use parameterized queries (SQLAlchemy ORM)
- All user input is validated through Pydantic models
```

---

### 7.6 ROADMAP.md (public-facing)

```markdown
# Roadmap

This is the public roadmap for localgate. Priorities may shift based on
community feedback.

## Now (Current Focus)
- [ ] Core proxy with Ollama backend
- [ ] API key CRUD and rate limiting
- [ ] SQLite + PostgreSQL support
- [ ] Token counting and usage stats

## Next
- [ ] RAG memory layer (auto-extend context)
- [ ] Conversation history storage and retrieval
- [ ] Web dashboard for key/usage management
- [ ] llama.cpp and vLLM backend adapters

## Later
- [ ] Plugin system for custom backends and stores
- [ ] Multi-model routing (send different requests to different models)
- [ ] Prompt caching layer
- [ ] Webhooks for usage alerts
- [ ] Cluster mode (multiple gateway instances)

## Community Requests
Open a feature request issue to suggest additions!
```

---

### 7.7 LICENSE (MIT)

```
MIT License

Copyright (c) 2026 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

### 7.8 .github/ISSUE_TEMPLATE/bug_report.md

```markdown
---
name: Bug Report
about: Report something that isn't working
title: "[BUG] "
labels: bug
---

## Describe the bug
A clear description of what the bug is.

## To Reproduce
Steps to reproduce the behavior:
1. Run `localgate ...`
2. Send request to `...`
3. See error

## Expected behavior
What you expected to happen.

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.11]
- localgate version: [e.g., 0.2.0]
- Backend: [e.g., Ollama 0.3.x]
- Database: [e.g., SQLite / PostgreSQL 16]

## Logs
Paste relevant log output (redact any API keys).
```

---

### 7.9 .github/ISSUE_TEMPLATE/feature_request.md

```markdown
---
name: Feature Request
about: Suggest an idea for localgate
title: "[FEATURE] "
labels: enhancement
---

## Problem
What problem does this solve? What's frustrating today?

## Proposed Solution
Describe what you'd like to happen.

## Alternatives Considered
Any alternative solutions or workarounds you've thought about.

## Additional Context
Any other context, screenshots, or examples.
```

---

### 7.10 .github/PULL_REQUEST_TEMPLATE.md

```markdown
## What does this PR do?
Brief description of the change.

## Type of change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Checklist
- [ ] Tests added/updated
- [ ] `make lint` passes
- [ ] `make test` passes
- [ ] Documentation updated (if applicable)
- [ ] CHANGELOG.md updated

## Related Issues
Closes #
```

---

## 8. Open-Source Repo Setup Guide

### Step 1: Create the repository

```bash
# Create the project directory
mkdir localgate && cd localgate
git init

# Create the folder structure
mkdir -p src/localgate/{api,core,backends,memory,db/{migrations/versions,repositories},middleware,dashboard/static}
mkdir -p tests/{unit,integration,e2e}
mkdir -p docs examples scripts .github/{ISSUE_TEMPLATE,workflows}

# Create __init__.py files
find src/localgate -type d -exec touch {}/__init__.py \;
touch tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py tests/e2e/__init__.py
```

### Step 2: Create pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "localgate"
version = "0.1.0"
description = "Local LLM API gateway with auth, RAG memory, and token accounting"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [{ name = "Your Name", email = "you@example.com" }]
keywords = ["llm", "api", "gateway", "ollama", "rag", "local-ai"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "httpx>=0.27.0",
    "pydantic>=2.6.0",
    "pydantic-settings>=2.1.0",
    "sqlalchemy[asyncio]>=2.0.25",
    "aiosqlite>=0.19.0",
    "alembic>=1.13.0",
    "typer>=0.9.0",
    "structlog>=24.1.0",
    "bcrypt>=4.1.0",
    "tiktoken>=0.6.0",
]

[project.optional-dependencies]
postgres = [
    "asyncpg>=0.29.0",
    "pgvector>=0.2.4",
]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "httpx>=0.27.0",
    "ruff>=0.3.0",
    "mypy>=1.8.0",
    "pre-commit>=3.6.0",
    "coverage>=7.4.0",
]
all = ["localgate[postgres,dev]"]

[project.scripts]
localgate = "localgate.cli:app"

[project.urls]
Homepage = "https://github.com/YOUR_USERNAME/localgate"
Documentation = "https://github.com/YOUR_USERNAME/localgate/tree/main/docs"
Repository = "https://github.com/YOUR_USERNAME/localgate"
Issues = "https://github.com/YOUR_USERNAME/localgate/issues"

[tool.hatch.build.targets.wheel]
packages = ["src/localgate"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "TCH"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.mypy]
python_version = "3.10"
strict = true
```

### Step 3: Create the Makefile

```makefile
.PHONY: run test lint format install dev

install:
	pip install -e ".[all]"

dev:
	pip install -e ".[dev]"
	pre-commit install

run:
	uvicorn localgate.app:create_app --factory --reload --host 0.0.0.0 --port 8000

test:
	pytest -v --tb=short

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

coverage:
	coverage run -m pytest
	coverage report -m
	coverage html
```

### Step 4: Create the .env.example

```bash
# Backend
LOCALGATE_BACKEND_TYPE=ollama
LOCALGATE_BACKEND_URL=http://localhost:11434
LOCALGATE_DEFAULT_MODEL=llama3

# Database (SQLite by default — no setup needed)
LOCALGATE_DATABASE_URL=sqlite+aiosqlite:///./localgate.db
# For Postgres: LOCALGATE_DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/localgate
# For Neon:     LOCALGATE_DATABASE_URL=postgresql+asyncpg://user:pass@ep-xxx.us-east-2.aws.neon.tech/localgate?sslmode=require

# Memory / RAG
LOCALGATE_MEMORY_ENABLED=true
LOCALGATE_EMBEDDING_MODEL=nomic-embed-text
LOCALGATE_CHUNK_SIZE=512
LOCALGATE_MAX_RETRIEVED_CHUNKS=5

# Auth
LOCALGATE_ADMIN_KEY=change-me-in-production

# Server
LOCALGATE_HOST=0.0.0.0
LOCALGATE_PORT=8000
```

### Step 5: Create the .gitignore

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/
*.egg

# Virtual environments
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Environment
.env
!.env.example

# Database
*.db
*.sqlite

# Coverage
htmlcov/
.coverage

# OS
.DS_Store
Thumbs.db

# Distribution
*.tar.gz
*.whl
```

### Step 6: Create CI workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Lint
        run: ruff check src/ tests/
      - name: Format check
        run: ruff format --check src/ tests/
      - name: Test
        run: pytest -v
```

### Step 7: Push to GitHub

```bash
# Add all files
git add .
git commit -m "feat: initial project scaffolding"

# Create GitHub repo (via gh CLI or web UI)
gh repo create localgate --public --source=. --push
```

---

## 9. Package Release Strategy

### Versioning

Follow **Semantic Versioning (SemVer)**:
- `0.x.y` — pre-stable, breaking changes allowed between minors
- `1.0.0` — first stable release, public API contract locked
- After 1.0: MAJOR.MINOR.PATCH

### Release Process

```bash
# 1. Update version in pyproject.toml and src/localgate/__init__.py
# 2. Update CHANGELOG.md with release notes
# 3. Commit and tag
git add .
git commit -m "release: v0.1.0"
git tag v0.1.0
git push origin main --tags
```

### PyPI Release Workflow

```yaml
# .github/workflows/release.yml
name: Release to PyPI

on:
  push:
    tags: ["v*"]

jobs:
  publish:
    runs-on: ubuntu-latest
    permissions:
      id-token: write  # Trusted publisher (no API token needed)
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Build
        run: |
          pip install build
          python -m build
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

### Docker Release Workflow

```yaml
# .github/workflows/docker.yml
name: Docker

on:
  push:
    tags: ["v*"]

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.ref_name }}
```

### Release Checklist (copy this for each release)

```
- [ ] All tests pass on main
- [ ] Version bumped in pyproject.toml and __init__.py
- [ ] CHANGELOG.md updated with all changes
- [ ] Documentation updated for new features
- [ ] Tag created and pushed
- [ ] PyPI package published successfully
- [ ] Docker image built and pushed
- [ ] GitHub Release created with release notes
- [ ] Announcement posted (Reddit, HN, Twitter/X)
```

---

## 10. Additional Features (Beyond Your Core Idea)

These are features I'm suggesting based on what would make the project significantly more useful and differentiated.

### 10.1 Prompt Caching

Cache identical or near-identical prompts and return cached responses. Saves inference time on repeated queries. Use a hash of the messages array as the cache key, store in the same database.

### 10.2 Model Aliasing & Routing

Let users define friendly names that map to specific backends:

```yaml
models:
  fast: { backend: ollama, model: phi4-mini }
  smart: { backend: vllm, model: llama3-70b }
  code: { backend: ollama, model: codellama }
```

Requests to `model: "fast"` auto-route to the right backend.

### 10.3 Conversation Branching

Store conversation trees, not just linear history. Let users fork a conversation at any point, like git branches for chat sessions.

### 10.4 Export & Portability

Export all data (conversations, keys, usage stats) as JSON or SQLite dump. Users should never feel locked into your tool.

### 10.5 Health & Diagnostics Endpoint

`GET /health` returns backend status, DB connectivity, memory usage, queue depth, and model availability in one call. Essential for production deployments.

### 10.6 Webhook Notifications

Fire webhooks when usage exceeds thresholds, keys are created/revoked, or backends go down. Lets users integrate with their monitoring stack.

### 10.7 Multi-User Web UI

Beyond the admin dashboard, provide a simple chat interface where multiple users can test their keys and see their usage without writing code.

---

## 11. Community & Growth Playbook

### Launch Strategy

1. **Day 1:** Post on r/LocalLLaMA, r/selfhosted, Hacker News. These are your people.
2. **Week 1:** Record a 3-minute demo video showing the setup → first API call → usage dashboard flow.
3. **Month 1:** Write a blog post: "Why your local LLM needs an API gateway" explaining the gap.
4. **Ongoing:** Tag issues as `good-first-issue` generously. Respond to every issue within 24 hours. Merge PRs quickly.

### Community Health

- **Label issues well:** `bug`, `enhancement`, `good-first-issue`, `help-wanted`, `documentation`
- **Use GitHub Discussions** for questions and ideas (keep Issues for actionable items)
- **Write ADRs (Architecture Decision Records)** in a `docs/decisions/` folder for significant choices, so contributors understand the "why" behind the codebase

### Metrics to Track

- GitHub stars (vanity, but real signal)
- PyPI downloads per week
- Number of unique contributors
- Issues opened vs. closed ratio
- Time to first response on issues

---

## Quick Reference: First Day Commands

```bash
# 1. Scaffold the project
mkdir localgate && cd localgate
git init

# 2. Create the structure (use the folder structure above)
# 3. Create pyproject.toml, Makefile, .env.example, .gitignore (content above)

# 4. Create your first working file
# Start with src/localgate/__init__.py → config.py → app.py → api/chat.py → backends/ollama.py

# 5. Install and verify
pip install -e ".[dev]"
make test

# 6. Push to GitHub
gh repo create localgate --public --source=. --push

# 7. Your first release
git tag v0.1.0
git push origin v0.1.0
```

---

*This roadmap is a living document. Update it as the project evolves and community feedback shapes priorities.*
