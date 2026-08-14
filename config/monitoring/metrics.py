"""
Prometheus metrics for monitoring.

Provides performance, resource, error, and business metrics.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import psutil
import functools
from contextlib import contextmanager

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, Info,
        generate_latest, REGISTRY, CollectorRegistry,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Define dummy types for when prometheus is not available
    Counter = None
    Gauge = None
    Histogram = None
    Summary = None
    Info = None
    CollectorRegistry = None
    REGISTRY = None


class MetricType(str, Enum):
    """Metric type enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricDefinition:
    """Metric definition."""
    name: str
    type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None


class MetricsCollector:
    """
    Centralized metrics collector using Prometheus client.

    Provides methods for tracking performance, errors, resources, and business metrics.
    """

    def __init__(self, namespace: str = "quant", registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics collector.

        Args:
            namespace: Metric namespace prefix
            registry: Prometheus registry (defaults to default registry)
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("prometheus_client not installed. Install with: pip install prometheus-client")

        self.namespace = namespace
        self.registry = registry or REGISTRY
        self._metrics: Dict[str, Any] = {}

        # Initialize standard metrics
        self._init_performance_metrics()
        self._init_error_metrics()
        self._init_resource_metrics()
        self._init_business_metrics()

    def _init_performance_metrics(self):
        """Initialize performance metrics."""
        # Request latency histogram
        self._metrics['request_latency'] = Histogram(
            f'{self.namespace}_request_duration_seconds',
            'Request duration in seconds',
            ['operation', 'status'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry,
        )

        # Operation counter
        self._metrics['operation_total'] = Counter(
            f'{self.namespace}_operations_total',
            'Total number of operations',
            ['operation', 'status'],
            registry=self.registry,
        )

        # Throughput gauge
        self._metrics['throughput'] = Gauge(
            f'{self.namespace}_throughput',
            'Operations per second',
            ['operation'],
            registry=self.registry,
        )

        # Active requests gauge
        self._metrics['active_requests'] = Gauge(
            f'{self.namespace}_active_requests',
            'Number of requests currently being processed',
            ['operation'],
            registry=self.registry,
        )

    def _init_error_metrics(self):
        """Initialize error metrics."""
        # Error counter
        self._metrics['errors_total'] = Counter(
            f'{self.namespace}_errors_total',
            'Total number of errors',
            ['operation', 'error_type'],
            registry=self.registry,
        )

        # Error rate gauge
        self._metrics['error_rate'] = Gauge(
            f'{self.namespace}_error_rate',
            'Error rate (errors per second)',
            ['operation'],
            registry=self.registry,
        )

    def _init_resource_metrics(self):
        """Initialize resource metrics."""
        # Memory usage
        self._metrics['memory_bytes'] = Gauge(
            f'{self.namespace}_memory_bytes',
            'Memory usage in bytes',
            ['type'],
            registry=self.registry,
        )

        # CPU usage
        self._metrics['cpu_percent'] = Gauge(
            f'{self.namespace}_cpu_percent',
            'CPU usage percentage',
            registry=self.registry,
        )

        # Disk usage
        self._metrics['disk_bytes'] = Gauge(
            f'{self.namespace}_disk_bytes',
            'Disk usage in bytes',
            ['type'],
            registry=self.registry,
        )

        # File descriptors
        self._metrics['open_fds'] = Gauge(
            f'{self.namespace}_open_file_descriptors',
            'Number of open file descriptors',
            registry=self.registry,
        )

    def _init_business_metrics(self):
        """Initialize business metrics."""
        # Factor metrics
        self._metrics['factors_evaluated'] = Counter(
            f'{self.namespace}_factors_evaluated_total',
            'Total number of factors evaluated',
            ['factor_type'],
            registry=self.registry,
        )

        self._metrics['factor_computation_time'] = Histogram(
            f'{self.namespace}_factor_computation_seconds',
            'Factor computation time in seconds',
            ['factor_type'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry,
        )

        # Evaluation metrics
        self._metrics['evaluations_total'] = Counter(
            f'{self.namespace}_evaluations_total',
            'Total number of evaluations',
            ['evaluation_type'],
            registry=self.registry,
        )

        # Data points processed
        self._metrics['data_points_processed'] = Counter(
            f'{self.namespace}_data_points_processed_total',
            'Total number of data points processed',
            ['data_type'],
            registry=self.registry,
        )

        # Cache metrics
        self._metrics['cache_hits'] = Counter(
            f'{self.namespace}_cache_hits_total',
            'Total number of cache hits',
            ['cache_type'],
            registry=self.registry,
        )

        self._metrics['cache_misses'] = Counter(
            f'{self.namespace}_cache_misses_total',
            'Total number of cache misses',
            ['cache_type'],
            registry=self.registry,
        )

    @contextmanager
    def track_operation(self, operation: str):
        """
        Context manager for tracking operation duration and status.

        Args:
            operation: Operation name

        Example:
            with metrics.track_operation("compute_factor"):
                result = compute_factor()
        """
        # Increment active requests
        self._metrics['active_requests'].labels(operation=operation).inc()

        start_time = time.perf_counter()
        status = "success"

        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            # Record duration
            duration = time.perf_counter() - start_time
            self._metrics['request_latency'].labels(operation=operation, status=status).observe(duration)

            # Increment operation counter
            self._metrics['operation_total'].labels(operation=operation, status=status).inc()

            # Decrement active requests
            self._metrics['active_requests'].labels(operation=operation).dec()

    def track_performance(self, operation: str):
        """
        Decorator for tracking function performance.

        Args:
            operation: Operation name

        Example:
            @metrics.track_performance("compute_factor")
            def compute_factor():
                ...
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.track_operation(operation):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def record_error(self, operation: str, error_type: str):
        """
        Record an error occurrence.

        Args:
            operation: Operation where error occurred
            error_type: Type of error
        """
        self._metrics['errors_total'].labels(
            operation=operation,
            error_type=error_type
        ).inc()

    def record_factor_evaluation(self, factor_type: str, duration_seconds: float):
        """
        Record factor evaluation.

        Args:
            factor_type: Type of factor
            duration_seconds: Computation duration
        """
        self._metrics['factors_evaluated'].labels(factor_type=factor_type).inc()
        self._metrics['factor_computation_time'].labels(factor_type=factor_type).observe(duration_seconds)

    def record_cache_hit(self, cache_type: str):
        """Record cache hit."""
        self._metrics['cache_hits'].labels(cache_type=cache_type).inc()

    def record_cache_miss(self, cache_type: str):
        """Record cache miss."""
        self._metrics['cache_misses'].labels(cache_type=cache_type).inc()

    def record_data_points(self, data_type: str, count: int):
        """
        Record data points processed.

        Args:
            data_type: Type of data
            count: Number of data points
        """
        self._metrics['data_points_processed'].labels(data_type=data_type).inc(count)

    def update_resource_metrics(self):
        """
        Update resource usage metrics.

        Should be called periodically to track system resources.
        """
        try:
            # Get process
            process = psutil.Process()

            # Memory metrics
            mem_info = process.memory_info()
            self._metrics['memory_bytes'].labels(type='rss').set(mem_info.rss)
            self._metrics['memory_bytes'].labels(type='vms').set(mem_info.vms)

            # CPU metrics
            cpu_percent = process.cpu_percent(interval=0.1)
            self._metrics['cpu_percent'].set(cpu_percent)

            # File descriptors (Unix only)
            try:
                num_fds = process.num_fds()
                self._metrics['open_fds'].set(num_fds)
            except (AttributeError, psutil.AccessDenied):
                pass

            # Disk usage
            try:
                disk_usage = psutil.disk_usage('/')
                self._metrics['disk_bytes'].labels(type='used').set(disk_usage.used)
                self._metrics['disk_bytes'].labels(type='free').set(disk_usage.free)
            except Exception:
                pass

        except Exception as e:
            # Don't crash on metrics collection failure
            import logging
            logging.getLogger(__name__).warning(f"Failed to update resource metrics: {e}")

    def get_metrics(self) -> bytes:
        """
        Get current metrics in Prometheus format.

        Returns:
            Metrics in Prometheus exposition format
        """
        return generate_latest(self.registry)


# Global metrics instance
_metrics_instance: Optional[MetricsCollector] = None


def get_metrics_collector(namespace: str = "quant") -> MetricsCollector:
    """
    Get global metrics collector instance.

    Args:
        namespace: Metric namespace

    Returns:
        MetricsCollector instance
    """
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = MetricsCollector(namespace=namespace)
    return _metrics_instance


def configure_metrics(config) -> MetricsCollector:
    """
    Configure metrics from ProductionConfig.

    Args:
        config: ProductionConfig instance

    Returns:
        Configured MetricsCollector instance
    """
    global _metrics_instance

    if not config.metrics.enabled:
        # Return a no-op metrics collector
        class NoOpMetrics:
            def __getattr__(self, name):
                def noop(*args, **kwargs):
                    return self
                return noop
            @contextmanager
            def track_operation(self, *args, **kwargs):
                yield
            def track_performance(self, *args, **kwargs):
                def decorator(func):
                    return func
                return decorator
        return NoOpMetrics()

    _metrics_instance = MetricsCollector(namespace=config.service_name.replace('-', '_'))
    return _metrics_instance
