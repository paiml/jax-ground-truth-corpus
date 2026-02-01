.PHONY: lint test fmt typecheck coverage all clean

lint:
	ruff check src/ tests/

test:
	pytest --cov=jax_gtc --cov-report=term-missing -v

fmt:
	ruff format src/ tests/

typecheck:
	mypy src/

coverage:
	pytest --cov=jax_gtc --cov-report=html --cov-report=term-missing

all: lint typecheck test

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
