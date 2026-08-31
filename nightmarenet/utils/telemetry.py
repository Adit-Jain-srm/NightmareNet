"""OpenTelemetry observability for NightmareNet pipeline stages.

Provides distributed tracing (spans) and metrics export (OTLP) for each
pipeline stage: ingest, optimize, prepare, train, evaluate.

All public APIs degrade gracefully to no-ops when either:
- ``opentelemetry-sdk`` is not installed (install via ``pip install nightmarenet[otel]``), or
- ``observability.otel_endpoint`` is not configured in the YAML config.

This ensures zero impact on existing deployments and test suites that do not
opt in to observability.

Compatible backends (via OTLP gRPC):
    - Jaeger, Grafana Tempo, Honeycomb, Datadog Agent, Azure Monitor,
      Prometheus (via OTel Collector)

Typical config snippet::

    observability:
      otel_endpoint: http://localhost:4317   # gRPC OTLP receiver
      otel_service_name: nightmarenet
      otel_export_interval_ms: 5000
"""

from __future__ import annotations

import contextlib
import logging
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal singletons — populated by setup_telemetry()
# ---------------------------------------------------------------------------

_tracer: Any = None  # opentelemetry.trace.Tracer | _NoOpTracer
_meter: Any = None  # opentelemetry.metrics.Meter | _NoOpMeter
_phase_duration_histogram: Any = None
_robustness_gauge: Any = None
_gpu_gauge: Any = None

_OTEL_AVAILABLE = False
_SETUP_DONE = False


# ---------------------------------------------------------------------------
# No-op fallbacks — used when OTel is not installed / not configured
# ---------------------------------------------------------------------------


class _NoOpSpan:
    """Minimal span that accepts attribute sets without error."""

    def set_attribute(self, key: str, value: Any) -> None:  # noqa: D401
        pass

    def record_exception(self, exc: BaseException) -> None:
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    def start_as_current_span(self, name: str, **kwargs: Any) -> Any:  # noqa: D401
        return _NoOpSpan()

    @contextmanager
    def start_as_current_span_cm(self, name: str) -> Generator[_NoOpSpan, None, None]:
        yield _NoOpSpan()


class _NoOpHistogram:
    def record(self, value: float, attributes: Optional[dict] = None) -> None:
        pass


class _NoOpGauge:
    # OTel Python SDK uses Observable Gauge (callback-based); for simplicity
    # we expose a record() interface here that maps to an UpDownCounter internally.
    def record(self, value: float, attributes: Optional[dict] = None) -> None:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_telemetry(config: dict) -> None:
    """Initialise OpenTelemetry tracing and metrics from the NightmareNet config.

    Reads the ``observability`` section of *config*. If ``otel_endpoint`` is
    ``null`` / missing, or if the OTel SDK is not installed, all telemetry
    calls become no-ops — no exceptions are raised.

    This function is idempotent; subsequent calls are ignored.

    Args:
        config: Full NightmareNet YAML configuration dictionary.
    """
    global _tracer, _meter, _phase_duration_histogram, _robustness_gauge, _gpu_gauge
    global _OTEL_AVAILABLE, _SETUP_DONE

    if _SETUP_DONE:
        return
    _SETUP_DONE = True

    obs = config.get("observability", {})
    endpoint: Optional[str] = obs.get("otel_endpoint") or None
    service_name: str = obs.get("otel_service_name", "nightmarenet")
    export_interval_ms: int = int(obs.get("otel_export_interval_ms", 5000))

    if not endpoint:
        logger.debug("OTel endpoint not configured; telemetry disabled (no-op).")
        _tracer = _NoOpTracer()
        _meter = None
        _phase_duration_histogram = _NoOpHistogram()
        _robustness_gauge = _NoOpGauge()
        _gpu_gauge = _NoOpGauge()
        return

    try:
        from opentelemetry import metrics as otel_metrics
        from opentelemetry import trace as otel_trace
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import (
            PeriodicExportingMetricReader,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
        )

        resource = Resource.create({"service.name": service_name})

        # --- Tracing ---
        tracer_provider = TracerProvider(resource=resource)
        span_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        otel_trace.set_tracer_provider(tracer_provider)
        _tracer = otel_trace.get_tracer("nightmarenet.pipeline")

        # --- Metrics ---
        metric_exporter = OTLPMetricExporter(endpoint=endpoint, insecure=True)
        reader = PeriodicExportingMetricReader(
            metric_exporter, export_interval_millis=export_interval_ms
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        otel_metrics.set_meter_provider(meter_provider)
        _meter = otel_metrics.get_meter("nightmarenet.pipeline")

        _phase_duration_histogram = _meter.create_histogram(
            name="nightmarenet.phase.duration_seconds",
            description="Wall-clock duration of each pipeline phase in seconds.",
            unit="s",
        )
        _robustness_gauge = _meter.create_up_down_counter(
            name="nightmarenet.robustness.score",
            description="Robustness score delta after the evaluation phase.",
        )
        _gpu_gauge = _meter.create_up_down_counter(
            name="nightmarenet.gpu.utilization_pct",
            description="GPU utilisation percentage (best-effort via pynvml).",
            unit="%",
        )

        _OTEL_AVAILABLE = True
        logger.info(
            "OpenTelemetry initialised",
            extra={"endpoint": endpoint, "service": service_name},
        )

    except ImportError:
        logger.warning(
            "opentelemetry-sdk not installed; telemetry disabled. "
            "Install with: pip install nightmarenet[otel]"
        )
        _tracer = _NoOpTracer()
        _phase_duration_histogram = _NoOpHistogram()
        _robustness_gauge = _NoOpGauge()
        _gpu_gauge = _NoOpGauge()

    except Exception as exc:
        logger.warning("Failed to initialise OpenTelemetry: %s; continuing without tracing.", exc)
        _tracer = _NoOpTracer()
        _phase_duration_histogram = _NoOpHistogram()
        _robustness_gauge = _NoOpGauge()
        _gpu_gauge = _NoOpGauge()


def get_tracer() -> Any:
    """Return the active tracer (OTel or no-op).

    Returns:
        An ``opentelemetry.trace.Tracer`` or :class:`_NoOpTracer` instance.
    """
    global _tracer
    if _tracer is None:
        _tracer = _NoOpTracer()
    return _tracer


@contextmanager
def trace_phase(
    phase_name: str,
    attributes: Optional[dict[str, Any]] = None,
) -> Generator[Any, None, None]:
    """Context manager that wraps a pipeline phase in an OTel span.

    Also records ``nightmarenet.phase.duration_seconds`` as a histogram metric.

    Args:
        phase_name: Name of the pipeline phase (e.g. ``"ingest"``, ``"train"``).
        attributes: Optional dict of span / metric attributes.

    Yields:
        The active span (OTel span or :class:`_NoOpSpan`).

    Example::

        with trace_phase("ingest", {"source": "huggingface"}) as span:
            dataset = ingestor.from_huggingface(...)
    """
    tracer = get_tracer()
    attrs = attributes or {}
    start = time.perf_counter()

    if _OTEL_AVAILABLE:
        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.trace import StatusCode

            with tracer.start_as_current_span(  # type: ignore[union-attr]
                f"nightmarenet.{phase_name}", kind=otel_trace.SpanKind.INTERNAL
            ) as span:
                for k, v in attrs.items():
                    span.set_attribute(k, v)
                try:
                    yield span
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(StatusCode.ERROR, str(exc))
                    raise
                finally:
                    elapsed = time.perf_counter() - start
                    _record_duration(phase_name, elapsed, attrs)
        except Exception:
            # Never let telemetry crash the pipeline
            yield _NoOpSpan()
    else:
        try:
            yield _NoOpSpan()
        finally:
            elapsed = time.perf_counter() - start
            _record_duration(phase_name, elapsed, attrs)


def record_metric(name: str, value: float, attributes: Optional[dict[str, Any]] = None) -> None:
    """Emit a named metric value.

    Currently supports:
    - ``"robustness_score"`` → recorded on the robustness UpDownCounter.
    - ``"gpu_utilization"`` → recorded on the GPU utilisation UpDownCounter.

    Silently ignored if telemetry is not configured.

    Args:
        name: Short metric name.
        value: Numeric value to record.
        attributes: Optional dimension labels.
    """
    attrs = attributes or {}
    with contextlib.suppress(Exception):
        if name == "robustness_score" and _robustness_gauge is not None:
            _robustness_gauge.record(value, attrs)  # type: ignore[union-attr]
        elif name == "gpu_utilization" and _gpu_gauge is not None:
            _gpu_gauge.record(value, attrs)  # type: ignore[union-attr]


def _record_duration(phase_name: str, elapsed: float, attrs: dict) -> None:
    """Record phase duration histogram. Internal helper."""
    with contextlib.suppress(Exception):
        if _phase_duration_histogram is not None:
            merged = {"phase": phase_name, **attrs}
            _phase_duration_histogram.record(elapsed, merged)  # type: ignore[union-attr]


def _sample_gpu_utilization() -> Optional[float]:
    """Return GPU utilisation % from pynvml if available, else None."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:
        return None


def reset_telemetry() -> None:
    """Reset telemetry state (primarily for testing)."""
    global _tracer, _meter, _phase_duration_histogram, _robustness_gauge, _gpu_gauge
    global _OTEL_AVAILABLE, _SETUP_DONE
    _tracer = None
    _meter = None
    _phase_duration_histogram = None
    _robustness_gauge = None
    _gpu_gauge = None
    _OTEL_AVAILABLE = False
    _SETUP_DONE = False
