# Observability

NightmareNet supports optional OpenTelemetry-based observability for API requests,
pipeline execution, and runtime metrics.

## Features

- API request tracing
- Pipeline phase tracing
- GPU utilization metrics (best effort)
- OTLP exporter support
- Jaeger integration for local development

Telemetry is completely optional. If no OpenTelemetry endpoint is configured,
NightmareNet falls back to its existing no-op implementation and incurs
effectively zero runtime overhead.

---

## Running Jaeger

Start the hosted profile:

```bash
docker compose --profile hosted up
```

Jaeger UI is available at:

```
http://localhost:16686
```

---

## Enabling OpenTelemetry

Configure the OTLP endpoint in your NightmareNet configuration:

```yaml
observability:
  otel_endpoint: http://localhost:4317
```

Telemetry will only be enabled when `observability.otel_endpoint`
is configured. If omitted, NightmareNet automatically falls back to
its built-in no-op telemetry implementation.

---

## API Tracing

Every incoming HTTP request creates an OpenTelemetry span with attributes such as:

- http.method
- http.target
- http.status_code

These spans become the parent trace for downstream pipeline operations.

---

## Structured Logging & Request Tracing

NightmareNet automatically generates or accepts an `X-Request-ID` HTTP header for correlation. 

- The `X-Request-ID` header is attached to API responses.
- It is automatically injected into all JSON and plain-text log records via the `RequestIdFilter`.
- When dispatched to the Celery worker, the correlation ID is propagated via task headers (`x-correlation-id`) so background processes continue the same trace.
- A summary log indicating `method`, `path`, `status`, and `duration_ms` is logged upon completion of every HTTP request.

---

## Pipeline Tracing

Pipeline execution creates child spans for major phases including:

- ingest
- prepare
- train
- evaluate

This allows complete request-to-training trace visualization inside Jaeger.

---

## GPU Metrics

When `pynvml` is installed and a supported NVIDIA GPU is available,
NightmareNet periodically records GPU utilization during training.

If no GPU or NVML installation is present, GPU metrics are silently skipped.

---

## Viewing Traces

1. Start Jaeger.
2. Enable an OTLP endpoint.
3. Run the API.
4. Trigger any pipeline request.
5. Open http://localhost:16686.
6. Select the `nightmarenet.pipeline` service.
7. Inspect request and pipeline spans.

---

## Notes

Telemetry is intentionally fail-safe.

Any exporter failures automatically fall back to the built-in no-op
implementation and never interrupt pipeline execution.

---

## Structured Logging Namespaces

NightmareNet utilizes standard Python logging namespaces across all core library modules instead of print() statements:

| Namespace | Subsystem | Description |
|-----------|-----------|-------------|
| 
ightmarenet.training.trainer | Training Loop | Cycle, epoch, loss, and checkpoint metrics |
| 
ightmarenet.evaluation.evaluator | Evaluation | Robustness testing, Pareto analysis, degradation curves |
| 
ightmarenet.data.adaption | Data Adaption | Chunking, dataset tokenization, transformation stats |
| 
ightmarenet.pipeline | Pipeline Lifecycle | Ingest, prepare, train, evaluate, export phase coordination |
| 
ightmarenet.pipeline_runner | Pipeline Runner | Background execution, thread pooling, heartbeat updates |
| 
ightmarenet.utils.logging_config | Logging Root | Formatter configuration (Plain text / JSON) |

Log levels can be configured via LOG_LEVEL environment variable (e.g. DEBUG, INFO, WARNING, ERROR) or in configs/default.yaml under observability.log_level.
