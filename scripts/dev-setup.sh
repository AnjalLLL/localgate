#!/usr/bin/env bash
set -e
# Installs uv if missing, creates the venv, syncs all deps (incl. dev extras),
# and installs the pre-commit hook.
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv sync --all-extras
uv run pre-commit install
echo "Dev environment ready. Run: make test  (or: uv run pytest)"
