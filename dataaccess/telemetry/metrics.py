"""
Metrics collection for factor_engine.

Provides counter, gauge, and histogram metrics with optional Prometheus/OpenTelemetry export.
All metrics are privacy-safe and opt-in only.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class MetricValue:
    """A single metric observation."""
    value: float
    labels: Dict[str, str]
    timestamp: float


class Metric(ABC):
    """Base class for all metrics."""

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        label_names: Sequence[str] = (),
    ):
        self.name = name
        self.description = description
        self.unit = unit
        self.label_names = tuple(label_names)
        self._lock = threading.Lock()

    @abstractmethod
    def collect(self) -> List[MetricValue]:
        """Collect current metric values."""
        pass

    @abstractmethod
    def get_type(self) -> MetricType:
        """Return the metric type."""
        pass

    def _validate_labels(self, labels: Dict[str, str]) -> None:
        """Validate that labels match the declared label names."""
        if set(labels.keys()) != set(self.label_names):
            raise ValueError(
                f"Labels {set(labels.keys())} do not match declared "
                f"label names {set(self.label_names)}"
            )


class Counter(Metric):
    """A counter that only increases."""

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        label_names: Sequence[str] = (),
    ):
        super().__init__(name, description, unit, label_names)
        self._values: Dict[Tuple[Tuple[str, str], ...], float] = defaultdict(float)

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the counter."""
        if amount < 0:
            raise ValueError("Counter can only increase")

        labels = labels or {}
        if self.label_names and not labels:
            raise ValueError(f"Labels required: {self.label_names}")
        if labels:
            self._validate_labels(labels)

        label_key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[label_key] += amount

    def get_type(self) -> MetricType:
        return MetricType.COUNTER

    def collect(self) -> List[MetricValue]:
        with self._lock:
            now = time.time()
            return [
                MetricValue(
                    value=value,
                    labels=dict(label_key),
                    timestamp=now,
                )
                for label_key, value in self._values.items()
            ]


class Gauge(Metric):
    """A gauge that can go up and down."""

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        label_names: Sequence[str] = (),
    ):
        super().__init__(name, description, unit, label_names)
        self._values: Dict[Tuple[Tuple[str, str], ...], float] = defaultdict(float)

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set the gauge to a specific value."""
        labels = labels or {}
        if self.label_names and not labels:
            raise ValueError(f"Labels required: {self.label_names}")
        if labels:
            self._validate_labels(labels)

        label_key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[label_key] = value

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the gauge."""
        labels = labels or {}
        if self.label_names and not labels:
            raise ValueError(f"Labels required: {self.label_names}")
        if labels:
            self._validate_labels(labels)

        label_key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[label_key] += amount

    def dec(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrement the gauge."""
        self.inc(-amount, labels)

    def get_type(self) -> MetricType:
        return MetricType.GAUGE

    def collect(self) -> List[MetricValue]:
        with self._lock:
            now = time.time()
            return [
                MetricValue(
                    value=value,
                    labels=dict(label_key),
                    timestamp=now,
                )
                for label_key, value in self._values.items()
            ]


@dataclass
class HistogramBucket:
    """A histogram bucket with upper bound and count."""
    le: float  # Less than or equal to
    count: int = 0


class Histogram(Metric):
    """A histogram that tracks distribution of values."""

    DEFAULT_BUCKETS = (
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5,
        0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float("inf")
    )

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        label_names: Sequence[str] = (),
        buckets: Optional[Sequence[float]] = None,
    ):
        super().__init__(name, description, unit, label_names)
        self._buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        if self._buckets[-1] != float("inf"):
            self._buckets.append(float("inf"))

        self._observations: Dict[Tuple[Tuple[str, str], ...], List[float]] = defaultdict(list)
        self._sum: Dict[Tuple[Tuple[str, str], ...], float] = defaultdict(float)
        self._count: Dict[Tuple[Tuple[str, str], ...], int] = defaultdict(int)

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record an observation."""
        labels = labels or {}
        if self.label_names and not labels:
            raise ValueError(f"Labels required: {self.label_names}")
        if labels:
            self._validate_labels(labels)

        label_key = tuple(sorted(labels.items()))
        with self._lock:
            self._observations[label_key].append(value)
            self._sum[label_key] += value
            self._count[label_key] += 1

    def get_type(self) -> MetricType:
        return MetricType.HISTOGRAM

    def collect(self) -> List[MetricValue]:
        """Collect histogram values with bucket counts."""
        with self._lock:
            now = time.time()
            results = []
            for label_key, observations in self._observations.items():
                labels_dict = dict(label_key)

                # Count observations in each bucket
                bucket_counts = []
                for bucket_le in self._buckets:
                    count = sum(1 for v in observations if v <= bucket_le)
                    bucket_counts.append((bucket_le, count))

                # Store as custom attributes (exporters will handle differently)
                results.append(
                    MetricValue(
                        value=self._sum[label_key],
                        labels={
                            **labels_dict,
                            "_count": str(self._count[label_key]),
                            "_buckets": str(bucket_counts),
                        },
                        timestamp=now,
                    )
                )

            return results


class MetricsRegistry:
    """Central registry for all metrics."""

    def __init__(self, enabled: bool = False):
        self._enabled = enabled
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()
        self._exporters: List[MetricsExporter] = []

    def enable(self) -> None:
        """Enable metrics collection (opt-in)."""
        self._enabled = True

    def disable(self) -> None:
        """Disable metrics collection."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if metrics are enabled."""
        return self._enabled

    def counter(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        label_names: Sequence[str] = (),
    ) -> Counter:
        """Create or retrieve a counter metric."""
        return self._get_or_create(
            name, lambda: Counter(name, description, unit, label_names)
        )

    def gauge(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        label_names: Sequence[str] = (),
    ) -> Gauge:
        """Create or retrieve a gauge metric."""
        return self._get_or_create(
            name, lambda: Gauge(name, description, unit, label_names)
        )

    def histogram(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        label_names: Sequence[str] = (),
        buckets: Optional[Sequence[float]] = None,
    ) -> Histogram:
        """Create or retrieve a histogram metric."""
        return self._get_or_create(
            name, lambda: Histogram(name, description, unit, label_names, buckets)
        )

    def _get_or_create(self, name: str, factory: Callable[[], Metric]) -> Metric:
        """Get existing metric or create new one."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = factory()
            return self._metrics[name]

    def collect_all(self) -> Dict[str, List[MetricValue]]:
        """Collect all metric values."""
        if not self._enabled:
            return {}

        with self._lock:
            return {
                name: metric.collect()
                for name, metric in self._metrics.items()
            }

    def add_exporter(self, exporter: MetricsExporter) -> None:
        """Add a metrics exporter."""
        with self._lock:
            self._exporters.append(exporter)

    def export(self) -> None:
        """Export metrics to all registered exporters."""
        if not self._enabled:
            return

        metrics = self.collect_all()
        for exporter in self._exporters:
            try:
                exporter.export(metrics)
            except Exception:
                pass  # Don't let exporter failures break the application


class MetricsExporter(ABC):
    """Base class for metrics exporters."""

    @abstractmethod
    def export(self, metrics: Dict[str, List[MetricValue]]) -> None:
        """Export collected metrics."""
        pass


# Global registry
_global_registry: Optional[MetricsRegistry] = None
_registry_lock = threading.Lock()


def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry."""
    global _global_registry
    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = MetricsRegistry(enabled=False)
    return _global_registry


def configure_metrics(enabled: bool = False) -> MetricsRegistry:
    """Configure the global metrics registry."""
    registry = get_metrics_registry()
    if enabled:
        registry.enable()
    else:
        registry.disable()
    return registry
