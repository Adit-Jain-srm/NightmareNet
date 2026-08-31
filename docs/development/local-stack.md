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

Optional production error reporting: set `NEXT_PUBLIC_SENTRY_DSN` to a
Sentry-compatible DSN (see `.env.example`). When unset, the frontend skips
remote reporting and logs errors locally during development.

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

## Verifying the Stack

Once the stack is up, confirm everything is actually healthy instead of guessing from log output:

```bash
make verify-stack
```

This runs `scripts/verify_docker_compose.py`, which sequentially checks the API's
`/api/v1/health` endpoint and prints a PASS/FAIL/SKIP summary with per-check timing.
It exits non-zero if any required check fails, so it's safe to use in CI.

If you started the stack with the `hosted` profile, also verify Redis, Postgres, the
worker, and API-to-service network reachability by passing `--profile hosted`:

```bash
make verify-stack VERIFY_STACK_ARGS="--profile hosted"
```

Under the default profile, a down `redis`/`db`/`worker` is reported as `SKIP` (not a
failure) since those services aren't part of the default stack — see the note above.