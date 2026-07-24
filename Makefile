.PHONY: help test lint typecheck format check frontend-build frontend-test openapi all clean

help:
	@echo "Available targets:"
	@echo "  make check          - lint + typecheck + test (mirrors CI)"
	@echo "  make test           - run pytest with coverage"
	@echo "  make lint           - run ruff check"
	@echo "  make typecheck      - run mypy on nightmarenet/"
	@echo "  make format         - auto-fix formatting with ruff format"
	@echo "  make openapi        - regenerate docs/api/openapi.json"
	@echo "  make frontend-build - build the Next.js frontend"
	@echo "  make frontend-test  - run frontend tests"
	@echo "  make all            - check + frontend-build (full CI equivalent)"

# Mirrors: .github/workflows/ci.yml -> "Run tests with coverage"
test:
	PYTHONPATH=. pytest -m "not slow" --cov=nightmarenet --cov-report=xml --cov-report=term-missing

# Mirrors: .github/workflows/ci.yml -> "Lint with ruff"
lint:
	ruff check nightmarenet/ scripts/ tests/

# Mirrors: .github/workflows/ci.yml -> "Type check with mypy"
typecheck:
	mypy nightmarenet/ --ignore-missing-imports --disable-error-code import-untyped --disable-error-code operator --python-version 3.12

format:
	ruff format .

openapi:
	PYTHONPATH=. python scripts/export_openapi.py

check: lint typecheck test
	@echo "All checks passed."

# Mirrors: .github/workflows/ci.yml -> "Build frontend"
frontend-build:
	cd frontend && npm ci && npm run build

frontend-test:
	cd frontend && npm run test

all: check frontend-build
	@echo "Full check complete."

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml
	rm -rf nightmarenet.egg-info dist build
	rm -rf checkpoints logs results/multi_seed
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned build artifacts."


.PHONY: dev
dev: ## Start API and frontend dev servers in parallel with hot-reload (Ctrl+C to stop)
	@chmod +x scripts/dev.sh
	@./scripts/dev.sh
