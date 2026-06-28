.PHONY: help install install-dev lint format type test test-mem-capped serve serve-dev load-test docker-build-serve docker-run-serve clean

help:
	@echo "Available targets:"
	@echo "  install            Install package and runtime dependencies"
	@echo "  install-dev        Install package with dev + serving extras"
	@echo "  lint               Run ruff lint"
	@echo "  format             Auto-format with ruff"
	@echo "  type               Run mypy type checks"
	@echo "  test               Run the test suite"
	@echo "  test-mem-capped    Run the test suite under a 2G memory cap"
	@echo "  serve              Run the FastAPI inference server locally"
	@echo "  load-test          Run a load test against a running server"
	@echo "  docker-build-serve Build the serving Docker image"
	@echo "  docker-run-serve   Run the serving Docker image"
	@echo "  clean              Remove caches and build artifacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,serving,tracking]"
	pre-commit install

lint:
	ruff check .

format:
	ruff check --fix .
	ruff format .

type:
	mypy .

test:
	./scripts/run_tests.sh

test-mem-capped:
	systemd-run --user --scope -p MemoryMax=2G ./scripts/run_tests.sh

serve:
	uvicorn serving.api:app --host 0.0.0.0 --port 8000 --workers 1

serve-dev:
	uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload

load-test:
	python scripts/load_test.py --url http://localhost:8000 --requests 20 --concurrency 1

docker-build-serve:
	docker build -f Dockerfile.serve -t structured-extraction-api:latest .

docker-run-serve:
	docker run --rm -p 8000:8000 --gpus all structured-extraction-api:latest

clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -not -path "./.venv/*" -exec rm -rf {} +
	rm -rf build dist *.egg-info
