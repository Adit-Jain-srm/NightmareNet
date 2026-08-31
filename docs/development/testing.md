# Testing Guide

This guide covers how to run and write tests for NightmareNet. The suite mirrors what CI runs on every pull request, so a green local run is the best predictor of a green PR.

## Running the Python test suite

Install the dev dependencies once, then run pytest from the repo root:

```bash
pip install -e ".[dev,api]"
pytest
```

The default options are defined in `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
testpaths = ["tests"]
addopts = "-v --tb=short -m 'not slow' --ignore=tests/test_distortion_vision.py"
markers = [
    "slow: marks slow tests (deselect with '-m not slow')",
]
```

So a bare `pytest` already runs verbosely, uses short tracebacks, **skips slow tests**, and ignores the vision distortion test module.

### Preferred commands

Use the cross-platform developer CLI or the Makefile — both mirror CI:

| Task | Command |
|------|---------|
| Run tests (`not slow`, with coverage) | `nightmarenet dev test` or `make test` |
| Run the full local check (lint + typecheck + test) | `nightmarenet dev check` or `make check` |
| Frontend tests | `nightmarenet dev test --frontend` or `make frontend-test` |

`make test` runs exactly what CI runs:

```bash
PYTHONPATH=. pytest -m "not slow" --cov=nightmarenet --cov-report=xml --cov-report=term-missing
```

## Markers

The `slow` marker separates fast unit tests from long-running ones (training loops, large fuzz suites).

```bash
# Default: skip slow tests
pytest -m "not slow"

# Run ONLY slow tests
pytest -m "slow"

# Run everything
pytest -m ""
```

Mark a slow test in code:

```python
import pytest


@pytest.mark.slow
def test_full_training_cycle():
    ...
```

## Coverage

Coverage is collected with `pytest-cov` against the `nightmarenet` package:

```bash
# Terminal report showing missing lines
pytest --cov=nightmarenet --cov-report=term-missing

# XML report (what CI uploads to Codecov)
pytest --cov=nightmarenet --cov-report=xml
```

CI uploads `coverage.xml` to Codecov on the Python 3.12 job.

## Selecting tests

```bash
# A single file
pytest tests/test_pipeline.py

# A single test function
pytest tests/test_pipeline.py::test_pipeline_runs

# By keyword
pytest -k "distortion and not vision"
```

## Frontend tests

The Next.js dashboard lives in `frontend/`. From that directory:

```bash
cd frontend
npm ci
npm run lint        # ESLint
npx tsc --noEmit    # TypeScript type-check
npm test            # unit tests
npm run build       # production build
npm run test:a11y   # Playwright accessibility tests
```

## What CI runs

The workflow in [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) runs on every push and pull request to `main`, across a Python `3.9`–`3.12` matrix:

1. `pip install -e ".[dev,api]"`
2. **mypy** (Python 3.12 only) — strict, baseline-enforced (see [Code Style](code-style.md)).
3. **pytest** — `pytest -m "not slow" --cov=nightmarenet --cov-report=xml --cov-report=term-missing`.
4. **Codecov** upload (3.12 only).
5. **OpenAPI drift check** (3.12 only) — `python scripts/export_openapi.py --check`.
6. **Frontend** (3.12 only) — `npm ci`, lint, `tsc --noEmit`, unit tests, build, and Playwright accessibility tests.

Ruff lint/format runs in a separate workflow (`.github/workflows/pre-commit.yml`) to avoid duplicate checks.

## Writing tests

Tests live in `tests/` and mirror the package structure (e.g. `tests/test_distortions.py`, `tests/test_pipeline.py`, `tests/test_metrics.py`). A few conventions from `CONTRIBUTING.md`:

- Use `monkeypatch` for environment variables; never mutate `os.environ` directly.
- Keep the non-slow suite fast; mark anything heavy with `@pytest.mark.slow`.
- Never reduce the test count without explaining why in the PR description.
- New distortions, metrics, and phases must be tested for determinism, edge inputs, and registry round-trip.

See [Adding Distortions](adding-distortions.md) and [Adding Metrics](adding-metrics.md) for the specific test patterns those subsystems expect.

## Related Documentation

- [Code Style](code-style.md) — lint, formatting, and the mypy baseline policy.
- [Local Stack](local-stack.md) — running the API and frontend locally.
- [Architecture](architecture.md) — how the pieces fit together.
