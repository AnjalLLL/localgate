.PHONY: help install test lint format coverage run serve clean

help:
	@echo "install    Install all dependencies and pre-commit hooks"
	@echo "test       Run the test suite"
	@echo "coverage   Run tests with a coverage report"
	@echo "lint       ruff check + mypy"
	@echo "format     ruff format + autofix"
	@echo "run        Start the gateway with reload"
	@echo "clean      Remove caches and build artifacts"

install:
	uv sync --all-extras
	uv run pre-commit install

test:
	uv run pytest -q

coverage:
	uv run pytest --cov --cov-report=term-missing --cov-report=html

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/
	uv run mypy src/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

run: serve
serve:
	uv run localgate serve --reload

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
