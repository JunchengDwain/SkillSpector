.PHONY: help install install-dev langgraph-dev test test-unit test-integration test-cov test-ci lint lint-fix format format-check clean build docker-build

# Prefer uv if available, else use pip (set when Makefile is parsed)
UV := $(shell command -v uv 2>/dev/null)

# LangGraph Studio URL for `make langgraph-dev`.  Defaults to the hosted
# LangSmith UI.  Override per invocation with:
#   make langgraph-dev LANGGRAPH_STUDIO_URL=https://your-studio.example
LANGGRAPH_STUDIO_URL = https://smith.langchain.com

# Default target. All targets assume the virtual env is already created and activated.
help:
	@echo "Available targets (venv must be created and activated first):"
	@echo "  make install        - Install the package (uses uv if available, else pip)"
	@echo "  make install-dev    - Install with dev dependencies (uses uv if available, else pip)"
	@echo "  make langgraph-dev  - Run LangGraph dev server (Studio at \$$LANGGRAPH_STUDIO_URL)"
	@echo "  make test           - Run unit + integration tests"
	@echo "  make test-unit      - Run unit tests only (no LLM calls)"
	@echo "  make test-integration - Run integration tests only (invokes full graph, may call LLMs)"
	@echo "  make test-cov       - Run tests with coverage report"
	@echo "  make lint           - Run linters (ruff only)"
	@echo "  make lint-fix       - Auto-fix lint errors with ruff"
	@echo "  make format         - Format code with ruff"
	@echo "  make format-check   - Check code formatting with ruff"
	@echo "  make clean          - Remove build artifacts and cache files"
	@echo "  make build          - Build the package"
	@echo "  make docker-build   - Build the Docker image"

install:
	@if [ -n "$(UV)" ]; then uv sync; else pip install -e .; fi

install-dev:
	@if [ -n "$(UV)" ]; then uv sync --all-extras; else pip install -e ".[dev]"; fi

# Run LangGraph dev server, opening Studio at LANGGRAPH_STUDIO_URL.
langgraph-dev:
	langgraph dev --studio-url $(LANGGRAPH_STUDIO_URL)

# Run unit + integration tests
test: test-unit test-integration

# Run unit tests only (excludes integration marker)
test-unit:
	pytest -m "not integration" tests/

# Run integration tests only (invokes full graph, may call LLMs)
test-integration:
	pytest -m integration tests/

# Run tests with coverage
test-cov:
	pytest --cov=src/skillspector --cov-report=html --cov-report=term-missing tests/

# Run tests with coverage for CI (Cobertura XML + terminal)
test-ci:
	pytest --cov=src/skillspector --cov-report=term-missing --cov-report=xml tests/

# Run linters (fast: ruff only)
lint:
	@echo "Running ruff..."
	ruff check src/ tests/

# Auto-fix lint errors with ruff
lint-fix:
	@echo "Running ruff with auto-fix..."
	ruff check --fix src/ tests/

# Format code
format:
	@echo "Formatting with ruff..."
	ruff check --fix src/ tests/
	ruff format src/ tests/

# Check code formatting without modifying files
format-check:
	@echo "Checking formatting with ruff..."
	ruff format --check src/ tests/

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf src/*.egg-info
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete!"

# Build the package
build: clean
	python -m build

# Build the Docker image
docker-build:
	docker build -t skillspector .

