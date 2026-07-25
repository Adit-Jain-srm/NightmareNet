# Tutorial 5: API Deployment and Production Configuration

NightmareNet features a containerized FastAPI service for model distortion orchestration and robustness auditing. In this tutorial, we will explain how to configure, secure, deploy, and scale the NightmareNet API in local and production environments.

---

## 1. Local Development Server

To run the API server locally, install the core package with `api` extras, and start it via Uvicorn:

```bash
# Install required server dependencies
pip install -e ".[api]"

# Start the server (with auto-reload enabled for development)
uvicorn nightmarenet.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive documentation will be accessible at:
*   Swagger UI: `http://localhost:8000/docs`
*   ReDoc: `http://localhost:8000/redoc`

---

## 2. Production Environment Variables

Configure the API server using environment variables or a `.env` file in your project root:

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `NIGHTMARENET_API_KEY` | string | `null` | API Key for authorization. If set, requests must pass this key in the `X-API-Key` header. If unset, authentication is disabled (dev mode). |
| `NIGHTMARENET_CORS_ORIGINS` | string | `*` | Comma-separated list of allowed origins. |
| `NIGHTMARENET_HEALTH_TEST_COUNT` | boolean | `0` | If set to `1`/`true`, `/api/v1/health` runs a subprocess pytest collection check to report total tests passing. |
| `NIGHTMARENET_MAX_PIPELINE_RUNNERS` | integer | `64` | Maximum concurrent in-memory pipeline runner jobs before evicting completed runs. |

---

## 3. Docker Container Deployment

For production deployments, utilize the multi-stage Docker configurations included in the project.

### Step 1: Build the API Image

Build the official API server container image:

```bash
docker build -t nightmarenet/api:latest -f docker/Dockerfile.api .
```

### Step 2: Run the Container

Run the API service detached on port 8000, passing security credentials:

```bash
docker run -d \
  -p 8000:8000 \
  -e NIGHTMARENET_API_KEY="your-prod-secret-key-1234" \
  -e NIGHTMARENET_CORS_ORIGINS="https://dashboard.nightmarenet.ai" \
  --name nightmarenet-api \
  nightmarenet/api:latest
```

---

## 4. Docker Compose Orchestration

The repository includes a ready-to-run `docker-compose.yml` for unified local setup. By default, it orchestrates the functional API and frontend dashboard:

```bash
# Start API and Frontend Services
docker compose up api frontend
```

### Running with Optional Infrastructure (Hosted Profile)

If you require compliance audit tracking, async queues, or database persistence, run with the `hosted` profile:

```bash
docker compose --profile hosted up
```

This starts the auxiliary services:
*   `db`: PostgreSQL database.
*   `redis`: Celery broker for async job management.
*   `worker`: Celery worker running heavy training cycles.

---

## 5. Health Checks and Monitoring

Production reverse proxies (like Nginx, Traefik, or AWS ALB) should target the health check endpoint to monitor service status.

### Health Check Request

```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

#### Response Payload:
```json
{
  "status": "ok",
  "version": "0.2.0",
  "tests_passing": null
}
```

---

## 6. Example API Requests

### 1. Generating a Nightmare Distortion

Send a text generation request, passing your custom API key:

```bash
curl -X POST "http://localhost:8000/api/v1/generate/nightmare" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your-prod-secret-key-1234" \
     -d '{
       "text": "The training pipeline consolidates model memory.",
       "strength": 0.8,
       "seed": 42
     }'
```

#### Response:
```json
{
  "original_text": "The training pipeline consolidates model memory.",
  "distorted_text": "The training *ipeline *onsolidates *odel *emory.",
  "distortion_type": "nightmare",
  "strength": 0.8,
  "seed": 42
}
```

### 2. Testing a Webhook Alert

Trigger a mock webhook event to verify external integrations:

```bash
curl -X POST "http://localhost:8000/api/v1/notifications/test-webhook" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your-prod-secret-key-1234" \
     -d '{
       "url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
       "event_type": "regression_detected"
     }'
```

---

## 7. Scaling and Performance Considerations

*   **VRAM Isolation**: Heavy adversarial generation requires significant GPU resources. In production, run the `worker` service on GPU-enabled instances (like AWS `g4dn.xlarge`), while keeping the stateless `api` containers on standard CPU instances.
*   **Gunicorn Workers**: In CPU environments, run Uvicorn behind Gunicorn to manage multi-process request distribution:
    ```bash
    gunicorn nightmarenet.api.app:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
    ```
