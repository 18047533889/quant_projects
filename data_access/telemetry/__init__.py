"""
Telemetry subsystem for factor_engine.

Provides opt-in, privacy-safe monitoring:
- metrics.py: counter/gauge/histogram for key operations
- traces.py: distributed tracing support
- health.py: health check endpoints

Supports Prometheus and OpenTelemetry exporters as optional dependencies.
All telemetry is disabled by default and requires explicit opt-in.
"""

from telemetry.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    get_metrics_registry,
    configure_metrics,
)
from telemetry.traces import (
    Tracer,
    Span,
    TracerProvider,
    get_tracer,
    trace,
    configure_tracing,
)
from telemetry.health import (
    HealthChecker,
    HealthStatus,
    ComponentHealth,
    get_health_checker,
    configure_health_checks,
)

__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "get_metrics_registry",
    "configure_metrics",
    "Tracer",
    "Span",
    "TracerProvider",
    "get_tracer",
    "trace",
    "configure_tracing",
    "HealthChecker",
    "HealthStatus",
    "ComponentHealth",
    "get_health_checker",
    "configure_health_checks",
]
