# Local Stack Development

NightmareNet's local stack encompasses both the backend APIs and the Next.js frontend, orchestrated via Docker Compose.

## Architecture

- **api**: FastAPI backend serving the main application logic and endpoints on port 8000.
- **worker**: Celery worker for background pipeline jobs.
- **db**: PostgreSQL database for relational persistence.
- **redis**: Redis for Celery message broker and caching.
- **frontend**: Next.js frontend running on port 3000, which proxies `/api/*` and `/ws/*` requests to the `api` service.

## Configuration

Ensure you have a `.env` file in the root directory (or `frontend/.env.example` in the frontend directory for standalone UI work).
By default, the stack works out-of-the-box using the defaults provided in `docker-compose.yml`.

### Connecting Frontend to Backend
The Next.js frontend is configured to route all API calls transparently to the backend.
- **HTTP**: Fetches to `/api/v1/*` are proxied to the backend via Next.js `next.config.ts` rewrites.
- **WebSockets**: WebSocket connections to `/ws/*` are dynamically routed either to `NEXT_PUBLIC_API_URL` or directly via same-origin Next.js proxying depending on your deployment setup.

## Running the Stack

To start the full stack:
```bash
docker compose up
```

To run backend services with the hosted profile (enables Worker + DB + Redis alongside API):
```bash
docker compose --profile hosted up
```
