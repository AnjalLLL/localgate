.PHONY: install test lint format run lock

install:
	uv sync --all-extras

lock:
	uv lock

test:
	uv run pytest -v

lint:
	uv run ruff check src/ tests/
	uv run mypy src/

format:
	uv run ruff format src/ tests/

run:
	uv run localgate serve --reload
