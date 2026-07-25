# --- Builder stage ---
FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml README.md ./
COPY nightmarenet/ nightmarenet/
COPY scripts/ scripts/

RUN pip install --no-cache-dir --prefix=/install ".[api]"

# --- Runtime stage ---
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source
COPY nightmarenet/ nightmarenet/
COPY scripts/ scripts/
COPY configs/ configs/

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --create-home appuser && \
    mkdir -p /app/checkpoints /app/logs /app/data && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8000/api/v1/health || exit 1

ENTRYPOINT ["uvicorn", "nightmarenet.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
