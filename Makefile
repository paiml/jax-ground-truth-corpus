.PHONY: lint test fmt typecheck coverage all clean sync

# uv-only workflow

sync:
	uv sync

lint:
	uv run ruff check src/ tests/

test:
	uv run pytest --cov=jax_gtc --cov-report=term-missing -v

fmt:
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/

coverage:
	uv run pytest --cov=jax_gtc --cov-report=html --cov-report=term-missing

all: lint typecheck test

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache htmlcov .coverage .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
