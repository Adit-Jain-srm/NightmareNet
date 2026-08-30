# Local Stack Development

NightmareNet's local stack covers the FastAPI backend and Next.js frontend.
Prefer the unified developer CLI for day-to-day work; Docker Compose remains
available for the full hosted profile.

## Preferred workflow (`nightmarenet dev`)

Install once:

```bash
pip install -e ".[dev,api]"
```

| Task | Command |
|------|---------|
| Help | `nightmarenet dev --help` |
| Lint (ruff + mypy + ESLint) | `nightmarenet dev lint` |
| Python-only lint (CI core) | `nightmarenet dev lint --python-only` |
| Tests (pytest, `not slow`) | `nightmarenet dev test` |
| Frontend tests | `nightmarenet dev test --frontend` |
| Format | `nightmarenet dev format` |
| Migrations | `nightmarenet dev migrate` |
| API + frontend hot reload | `nightmarenet dev serve` |
| Docker Compose + health | `nightmarenet dev docker` |
| Benchmark script | `nightmarenet dev benchmark` |
| Lint + test (like `make check`) | `nightmarenet dev check` |

These commands use Python `subprocess` (no bash), so they work on Windows PowerShell,
macOS, and Linux. Makefile targets (`make check`, `make test`, …) still work.

## Architecture

- **api**: FastAPI backend on port 8000
- **worker**: Celery worker for background pipeline jobs (hosted profile)
- **db**: PostgreSQL
- **redis**: Redis for Celery / cache
- **frontend**: Next.js on port 3000 (proxies `/api/*` and `/ws/*`)

## Configuration

Copy `.env.example` to `.env` at the repo root. Frontend rewrites in
`frontend/next.config.ts` proxy `/api/v1/*` to the backend.

## Docker Compose

```bash
# Full stack with health verification
nightmarenet dev docker

# Or classic compose
docker compose up

# Hosted profile (worker + db + redis)
docker compose --profile hosted up
# equivalent:
nightmarenet dev docker --profile hosted
```
