# Contributing to localgate

1. Fork and clone the repo.
2. `bash scripts/dev-setup.sh` to set up a dev environment (installs [uv](https://docs.astral.sh/uv/) if needed, then runs `uv sync --all-extras`).
3. Make your change; run `make test` and `make lint` before opening a PR (or `uv run pytest` / `uv run ruff check` directly).
4. Keep the layering rule: route handlers (`api/`) never touch the database directly —
   go through `core/` and `db/repositories/`.
5. Add tests for new behavior (`tests/unit` for isolated logic, `tests/integration` for
   endpoint-level behavior).

Good first issues are labeled `good-first-issue`.
