.PHONY: install dev-install run test lint format typecheck clean

install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements-dev.txt

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest

test-unit:
	pytest tests/unit/

test-integration:
	pytest tests/integration/

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/

ingest:
	python scripts/ingest_data.py

embed:
	python scripts/generate_embeddings.py

evaluate:
	python scripts/evaluate_classifier.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
