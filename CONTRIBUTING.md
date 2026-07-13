# Contributing

Thanks for wanting to help. This is meant to be an easy project to contribute to — the
architecture is deliberately shaped so that the most common contributions (a new backend, a
new database) are one file each.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/localgate.git
cd localgate
uv sync --all-extras
uv run pre-commit install

make test      # 160 tests, no live Ollama needed
make lint      # ruff + mypy
```

Tests run against a deterministic `FakeBackend` and an in-memory SQLite database, so the
suite is fast and needs nothing installed.

## The codebase in one minute

```
api/         routes — thin. Validate, delegate, respond. Never touch the database.
core/        business logic — auth, rate limiting, tokens, caching. No HTTP awareness.
backends/    one adapter per inference server.
memory/      chunking, embedding, retrieval, summarization.
db/          models + repositories. The only code that writes SQL.
middleware/  request id, access log, metrics.
```

Read [docs/architecture.md](docs/architecture.md) before a substantial change, and the
[ADRs](docs/decisions/) for *why* things are the way they are. If you're about to change
something an ADR covers, that's fine — but say so in the PR, and update the ADR.

## Adding a backend

This is the highest-value contribution and it's genuinely small. If the server speaks the
OpenAI API — most do — it's a subclass and a default port:

```python
# src/localgate/backends/my_server.py
from localgate.backends.openai_compat import OpenAICompatBackend

class MyServerBackend(OpenAICompatBackend):
    name = "my-server"
    default_base_url = "http://localhost:9000"
```

Register it in `pyproject.toml`:

```toml
[project.entry-points."localgate.backends"]
my-server = "localgate.backends.my_server:MyServerBackend"
```

Add a case to `tests/unit/test_backends.py::test_every_advertised_backend_can_actually_be_instantiated`.
That's it.

If the server *doesn't* speak OpenAI, implement `InferenceBackend` directly (see
`backends/base.py` — five methods) and translate in the adapter. Nothing above the backend
layer should ever learn that your server is different.

You can also ship a backend as **your own package** without touching this repo at all —
declare the same entry point in your `pyproject.toml`. See
[docs/architecture.md](docs/architecture.md#plugin-system).

## Standards

- **Type hints** on every function signature. `mypy src/` must pass.
- **Comments explain *why*, never *what*.** The code already says what it does. A comment
  earns its place by capturing the constraint, the tradeoff, or the bug that isn't visible
  from reading the line. If it restates the code, delete it.
- **Tests for every change.** A bug fix without a test that fails before it is not a fix.
- **`ruff check` and `ruff format`** — pre-commit runs both.
- **No `print`.** Use the structlog logger (`core/logging.py`).

## Tests

Write the test that would have caught the bug. The ones in here that earn their keep are:

- `test_every_route_is_guarded` — auth is per-route, so a new route that forgets its
  dependency would be silently public. This catches that.
- `test_pre_migrations_database_is_adopted_without_losing_data` — the upgrade path for
  existing users, which is exactly the thing you can't test by hand twice.
- `test_streaming_still_records_usage_after_the_body_is_sent` — works in dev, breaks in
  prod, if you get the session lifecycle wrong.

Aim for that. Not coverage of lines, coverage of *failure modes*.

## Pull requests

1. Branch from `main`: `git checkout -b feat/my-thing`
2. Make the change, with tests.
3. `make lint && make test`
4. Update `CHANGELOG.md` under `[Unreleased]`.
5. Open the PR and fill in the template.

Small PRs get reviewed fast. A 2000-line PR touching six subsystems will sit.

## Good first issues

Look for the `good-first-issue` label. Reliably useful things that are always open:

- A backend adapter for a server that isn't supported yet
- Better error messages — if something confused you, it will confuse the next person, and
  the fix is a sentence
- Documentation gaps, especially in [getting-started](docs/getting-started.md)
- Anything in [ROADMAP.md](ROADMAP.md) under "Later"

## Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). It applies everywhere the
project does.
