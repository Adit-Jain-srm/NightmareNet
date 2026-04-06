# NightmareNet — Agent Instructions

## Project Overview

NightmareNet is an **Autonomous AI Self-Improvement Platform** using a sleep-inspired training paradigm. It forces neural networks to learn invariant structures through 4-phase cycles: Wake → Dream → Nightmare → Compress.

**Stack**: Python ≥3.9, PyTorch, Transformers, FastAPI, Next.js 14 (frontend)

## Repository Structure

```
nightmarenet/           # Core Python package
  api/                  # FastAPI endpoints (app.py, schemas.py)
  compression/          # Knowledge distillation
  data/                 # Dataset loading & processing
  distortions/          # Dream & nightmare distortion engines
  evaluation/           # Robustness evaluation & metrics
  training/             # Training loops, DDP, experiment tracking
  utils/                # Config, early stopping, type hints
frontend/               # Next.js 14 + Framer Motion + Tailwind
  src/components/       # React components (cyberpunk UI)
  src/lib/api.ts        # TypeScript API client
tests/                  # pytest suite (206+ tests)
configs/                # YAML training configs
scripts/                # CLI entry points
```

## Critical Build & Test Commands

```bash
# Activate venv
.venv\Scripts\Activate.ps1          # Windows
source .venv/bin/activate            # Unix

# Run all tests
pytest tests/ -v --tb=short

# Run specific test module
pytest tests/test_distortions.py -v

# Lint (must pass with 0 errors)
ruff check .

# Type check
mypy nightmarenet/

# Start API server
uvicorn nightmarenet.api.app:app --reload --port 8000

# Frontend
cd frontend && npm run dev           # Dev server on :3000
cd frontend && npm run build         # Production build
```

## Code Style & Conventions

- **Line length**: 100 chars
- **Ruff rules**: E, F, W, I, N, UP, B (ignore UP007, UP045)
- **Imports**: isort via ruff — stdlib, third-party, local (alphabetical within groups)
- **Type hints**: Use `Union[X, Y]` not `X | Y` (Python 3.9 compat)
- **DO NOT** use `from __future__ import annotations` in any file that uses FastAPI `Body(...)` — it breaks Pydantic v2 at runtime
- **Docstrings**: Google style, only on public APIs
- **Error handling**: Raise with context (`raise X from e`), not bare `raise`

## Architecture Constraints

1. **No `from __future__ import annotations`** in `nightmarenet/api/` — Pydantic v2 + FastAPI incompatibility
2. **Python 3.9 minimum** — no walrus operator assumptions, use `Union[X, Y]` over `X | Y`
3. **Optional deps are optional** — guard imports with try/except: `accelerate`, `wandb`, `tensorboard`
4. **FastAPI Body singletons** — `_DISTORTION_BODY` and `_ROBUSTNESS_BODY` are module-level to avoid B008
5. **CORS** — configured via `NIGHTMARENET_CORS_ORIGINS` env var (default: `*`)

## Testing Requirements

- **206+ tests passing** — never commit code that reduces the test count
- Run `pytest` before claiming any task is complete
- Run `ruff check .` before claiming lint cleanliness
- Tests live in `tests/` — mirror the package structure
- Use `monkeypatch` for env vars, never modify `os.environ` directly

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v1/health` | Health check (status, version, tests) |
| POST | `/api/v1/generate/dream` | Dream distortion `{text, strength, seed}` |
| POST | `/api/v1/generate/nightmare` | Nightmare distortion `{text, strength, seed}` |
| POST | `/api/v1/evaluate/robustness` | Multi-strength eval `{text, strengths[]}` |

## Frontend Notes

- **Framework**: Next.js 14 (App Router), TypeScript, Tailwind CSS v4
- **Design**: Cyberpunk dark theme — void (#020617), dream (#818CF8), nightmare (#EF4444), neural (#06B6D4)
- **Animations**: Framer Motion throughout
- **API client**: `frontend/src/lib/api.ts` — typed client for all 4 endpoints
- **Key constraint**: Tailwind v4 uses `@theme inline` block, not `tailwind.config.js`

## Git Workflow

- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`
- Run tests + lint before any commit
- Keep commits atomic — one concern per commit

## Known Issues & Gotchas

- `Accelerator` import is optional — guarded with `Accelerator: Any = None` fallback
- `evaluator._log_eval()` guards against `self.tracker is None`
- Rate limiting on API uses `slowapi` — 10 req/min on generate, 5 req/min on evaluate
