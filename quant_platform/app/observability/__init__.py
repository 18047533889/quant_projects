"""Platform observability package (QRP-P11-OBS).

Pure in-memory, stdlib-only: ``MetricsRegistry`` (thread-safe counters/gauges),
layer collectors, and the ``assess`` / ``HealthReport`` health surface.
"""

from .metrics import (
    INVALID_METRIC_NAME_RE,
    InvalidMetricNameError,
    MetricsRegistry,
    MetricValue,
)
from .collectors import (
    collect_job_metrics,
    collect_outbox_metrics,
    collect_registry_metrics,
    collect_reconcile_metrics,
)
from .health import HealthCheck, HealthReport, HealthStatus, assess

__all__ = [
    "MetricsRegistry",
    "MetricValue",
    "InvalidMetricNameError",
    "INVALID_METRIC_NAME_RE",
    "collect_job_metrics",
    "collect_outbox_metrics",
    "collect_registry_metrics",
    "collect_reconcile_metrics",
    "HealthCheck",
    "HealthReport",
    "HealthStatus",
    "assess",
]
