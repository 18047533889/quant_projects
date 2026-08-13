"""
Example usage of the telemetry system.

This demonstrates how to use metrics, traces, and health checks
in a typical factor engine workflow.
"""

import time
from telemetry import (
    get_metrics_registry,
    get_tracer,
    get_health_checker,
    configure_metrics,
    configure_tracing,
    configure_health_checks,
    ComponentHealth,
    HealthStatus,
    trace,
)


def setup_telemetry():
    """Initialize telemetry systems (opt-in)."""
    configure_metrics(enabled=True)
    configure_tracing(enabled=True)
    configure_health_checks(enabled=True)

    print("Telemetry enabled")


def setup_metrics():
    """Create metrics for factor computation."""
    registry = get_metrics_registry()

    # Counter for tracking computations
    factor_counter = registry.counter(
        "factors_computed_total",
        description="Total factors computed",
        label_names=["factor_name", "status"],
    )

    # Gauge for current queue depth
    queue_gauge = registry.gauge(
        "factor_queue_depth",
        description="Current factor computation queue depth",
    )

    # Histogram for computation latency
    latency_histogram = registry.histogram(
        "factor_compute_duration_seconds",
        description="Factor computation duration",
        label_names=["factor_name"],
        buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    )

    return factor_counter, queue_gauge, latency_histogram


def setup_health_checks():
    """Register health checks for critical components."""
    checker = get_health_checker()

    def check_data_source():
        """Check if data source is available."""
        try:
            # In real code, check actual data source
            return ComponentHealth(
                "data_source",
                HealthStatus.HEALTHY,
                "All data sources reachable",
            )
        except Exception as e:
            return ComponentHealth(
                "data_source",
                HealthStatus.UNHEALTHY,
                f"Data source error: {e}",
            )

    def check_memory():
        """Check memory usage."""
        import psutil
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            if memory_mb > 8000:
                return ComponentHealth(
                    "memory",
                    HealthStatus.DEGRADED,
                    f"High memory usage: {memory_mb:.1f} MB",
                )
            else:
                return ComponentHealth(
                    "memory",
                    HealthStatus.HEALTHY,
                    f"Memory usage: {memory_mb:.1f} MB",
                )
        except ImportError:
            # psutil not available
            return ComponentHealth(
                "memory",
                HealthStatus.HEALTHY,
                "Memory check skipped (psutil not installed)",
            )
        except Exception as e:
            return ComponentHealth(
                "memory",
                HealthStatus.UNHEALTHY,
                f"Memory check failed: {e}",
            )

    # Register checks with different intervals
    checker.register("data_source", check_data_source, interval=30.0)
    checker.register("memory", check_memory, interval=60.0)

    return checker


@trace(name="compute_factor")
def compute_factor(factor_name: str, data: dict, metrics: tuple):
    """
    Compute a factor with full telemetry.

    This function is automatically traced via the @trace decorator.
    """
    factor_counter, queue_gauge, latency_histogram = metrics
    tracer = get_tracer(__name__)

    # Manual span for detailed tracking
    with tracer.trace("load_data") as span:
        span.set_attribute("factor_name", factor_name)
        span.set_attribute("data_size", len(data))

        # Simulate data loading
        time.sleep(0.01)
        span.add_event("data_loaded")

    # Compute factor
    start_time = time.time()

    with tracer.trace("compute") as span:
        span.set_attribute("factor_name", factor_name)

        try:
            # Simulate computation
            time.sleep(0.05)
            result = {"factor_name": factor_name, "value": 42.0}

            span.set_status("ok")
            span.add_event("computation_complete")

            # Record success metrics
            factor_counter.inc(labels={
                "factor_name": factor_name,
                "status": "success",
            })

        except Exception as e:
            span.set_status("error", str(e))
            span.add_event("computation_failed", {"error": str(e)})

            # Record failure metrics
            factor_counter.inc(labels={
                "factor_name": factor_name,
                "status": "error",
            })
            raise

    # Record latency
    duration = time.time() - start_time
    latency_histogram.observe(duration, labels={"factor_name": factor_name})

    return result


def main():
    """Main example workflow."""
    print("=== Telemetry Example ===\n")

    # Step 1: Enable telemetry
    setup_telemetry()

    # Step 2: Set up metrics
    print("\nSetting up metrics...")
    metrics = setup_metrics()
    factor_counter, queue_gauge, latency_histogram = metrics

    # Step 3: Set up health checks
    print("Setting up health checks...")
    checker = setup_health_checks()

    # Step 4: Simulate factor computation
    print("\nComputing factors...")

    queue_gauge.set(5)  # 5 factors in queue

    for i in range(3):
        factor_name = f"factor_{i}"
        data = {"date": "2026-08-14", "symbols": ["AAPL", "GOOGL"]}

        print(f"  Computing {factor_name}...")
        result = compute_factor(factor_name, data, metrics)

        queue_gauge.dec()  # One less in queue

    # Step 5: Collect and display metrics
    print("\n=== Metrics ===")
    registry = get_metrics_registry()
    all_metrics = registry.collect_all()

    for metric_name, metric_values in all_metrics.items():
        print(f"  {metric_name}:")
        for metric_value in metric_values:
            print(f"    {metric_value.value} (labels: {metric_value.labels})")

    # Step 6: Check system health
    print("\n=== Health Status ===")
    system_health = checker.check_all()
    print(f"Overall status: {system_health.status.value}")

    for component_name, component in system_health.components.items():
        message = f" - {component.message}" if component.message else ""
        print(f"  {component_name}: {component.status.value}{message}")

    # Step 7: Get HTTP-ready status
    print("\n=== HTTP Status (for endpoints) ===")
    status_dict = checker.get_status_dict()
    import json
    print(json.dumps(status_dict, indent=2))

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
