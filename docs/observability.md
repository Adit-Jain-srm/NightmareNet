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