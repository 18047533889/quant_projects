"""
Optional Prometheus exporter for metrics.

This module provides Prometheus export functionality if prometheus_client is installed.
It is an optional dependency and gracefully degrades if not available.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from factor_engine.telemetry.metrics import MetricValue

try:
    import prometheus_client
    from prometheus_client import Counter as PromCounter
    from prometheus_client import Gauge as PromGauge
    from prometheus_client import Histogram as PromHistogram
    from prometheus_client import REGISTRY, CollectorRegistry

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class PrometheusExporter:
    """Exporter for Prometheus metrics."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        if not PROMETHEUS_AVAILABLE:
            raise RuntimeError(
                "prometheus_client is not installed. "
                "Install it with: pip install prometheus-client"
            )

        self._registry = registry or REGISTRY
        self._metrics: Dict[str, object] = {}
        self._lock = threading.Lock()

    def export(self, metrics: Dict[str, List[MetricValue]]) -> None:
        """Export metrics to Prometheus."""
        from factor_engine.telemetry.metrics import MetricType, get_metrics_registry

        registry = get_metrics_registry()

        with self._lock:
            for metric_name, values in metrics.items():
                # Get the metric object to determine type
                metric_obj = registry._metrics.get(metric_name)
                if metric_obj is None:
                    continue

                metric_type = metric_obj.get_type()

                # Create or get Prometheus metric
                if metric_name not in self._metrics:
                    self._metrics[metric_name] = self._create_prometheus_metric(
                        metric_name,
                        metric_obj.description,
                        metric_type,
                        metric_obj.label_names,
                    )

                prom_metric = self._metrics[metric_name]

                # Update metric values
                if metric_type == MetricType.COUNTER:
                    self._update_counter(prom_metric, values)
                elif metric_type == MetricType.GAUGE:
                    self._update_gauge(prom_metric, values)
                elif metric_type == MetricType.HISTOGRAM:
                    self._update_histogram(prom_metric, values)

    def _create_prometheus_metric(
        self,
        name: str,
        description: str,
        metric_type: MetricType,
        label_names: tuple,
    ) -> object:
        """Create a Prometheus metric."""
        from factor_engine.telemetry.metrics import MetricType

        if metric_type == MetricType.COUNTER:
            return PromCounter(
                name,
                description,
                labelnames=label_names,
                registry=self._registry,
            )
        elif metric_type == MetricType.GAUGE:
            return PromGauge(
                name,
                description,
                labelnames=label_names,
                registry=self._registry,
            )
        elif metric_type == MetricType.HISTOGRAM:
            return PromHistogram(
                name,
                description,
                labelnames=label_names,
                registry=self._registry,
            )
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

    def _update_counter(self, prom_counter: PromCounter, values: List[MetricValue]) -> None:
        """Update a Prometheus counter."""
        for value in values:
            if value.labels:
                prom_counter.labels(**value.labels).inc(value.value)
            else:
                prom_counter.inc(value.value)

    def _update_gauge(self, prom_gauge: PromGauge, values: List[MetricValue]) -> None:
        """Update a Prometheus gauge."""
        for value in values:
            if value.labels:
                prom_gauge.labels(**value.labels).set(value.value)
            else:
                prom_gauge.set(value.value)

    def _update_histogram(self, prom_histogram: PromHistogram, values: List[MetricValue]) -> None:
        """Update a Prometheus histogram."""
        # Note: This is a simplified implementation
        # In production, you'd want to track individual observations
        for value in values:
            # Extract count and reconstruct observations
            labels_copy = dict(value.labels)
            count_str = labels_copy.pop("_count", "0")
            buckets_str = labels_copy.pop("_buckets", "[]")

            try:
                count = int(count_str)
                if count > 0:
                    # Approximate: observe the mean value
                    mean = value.value / count
                    for _ in range(count):
                        if labels_copy:
                            prom_histogram.labels(**labels_copy).observe(mean)
                        else:
                            prom_histogram.observe(mean)
            except (ValueError, ZeroDivisionError):
                pass

    def start_http_server(self, port: int = 9090, addr: str = "") -> None:
        """Start Prometheus HTTP server."""
        if not PROMETHEUS_AVAILABLE:
            raise RuntimeError("prometheus_client is not installed")

        prometheus_client.start_http_server(port, addr, registry=self._registry)


def create_prometheus_exporter(
    registry: Optional[CollectorRegistry] = None,
) -> Optional[PrometheusExporter]:
    """Create a Prometheus exporter if available."""
    if not PROMETHEUS_AVAILABLE:
        return None

    return PrometheusExporter(registry)
