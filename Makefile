# NightmareNet — Unified developer commands
# Run `make <target>` from the repo root.
# All targets are .PHONY (they don't produce files).

.PHONY: test lint typecheck format check \
        frontend-build frontend-lint frontend-test \
        all help

# ── Python ──────────────────────────────────────────────────────────

test:
	python -m pytest --cov=nightmarenet tests/ -v --tb=short

lint:
	python -m ruff check .

typecheck:
	python -m mypy nightmarenet/ --ignore-missing-imports

format:
	python -m ruff format .
	python -m ruff check --fix .

# ── Frontend ────────────────────────────────────────────────────────

frontend-build:
	cd frontend && npm run build

frontend-lint:
	cd frontend && npm run lint

frontend-test:
	cd frontend && npm run test

# ── Aggregates ──────────────────────────────────────────────────────

check: lint typecheck test
	@echo "✅ All Python checks passed."

all: check frontend-lint frontend-build frontend-test
	@echo "✅ Full check complete (Python + Frontend)."

# ── Help ────────────────────────────────────────────────────────────

help:
	@echo "NightmareNet Makefile targets:"
	@echo ""
	@echo "  Python:"
	@echo "    test            Run pytest with coverage"
	@echo "    lint            Run ruff check"
	@echo "    typecheck       Run mypy on nightmarenet/"
	@echo "    format          Auto-fix formatting with ruff"
	@echo ""
	@echo "  Frontend:"
	@echo "    frontend-build  Build the Next.js app"
	@echo "    frontend-lint   Lint the Next.js app"
	@echo "    frontend-test   Run frontend tests (vitest)"
	@echo ""
	@echo "  Aggregates:"
	@echo "    check           lint + typecheck + test (mirrors CI)"
	@echo "    all             check + frontend-lint + frontend-build + frontend-test"
