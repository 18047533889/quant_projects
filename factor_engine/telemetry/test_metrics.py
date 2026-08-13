"""
Tests for telemetry.metrics module.
"""

import time
from telemetry.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    MetricType,
    get_metrics_registry,
    configure_metrics,
)


def test_counter_basic():
    """Test basic counter operations."""
    counter = Counter("test_counter", "A test counter")

    counter.inc()
    assert counter.collect()[0].value == 1.0

    counter.inc(5.0)
    assert counter.collect()[0].value == 6.0


def test_counter_with_labels():
    """Test counter with labels."""
    counter = Counter("test_counter", "A test counter", label_names=["method", "status"])

    counter.inc(labels={"method": "GET", "status": "200"})
    counter.inc(labels={"method": "GET", "status": "200"})
    counter.inc(labels={"method": "POST", "status": "201"})

    values = counter.collect()
    assert len(values) == 2

    # Find the GET/200 value
    get_200 = next(v for v in values if v.labels == {"method": "GET", "status": "200"})
    assert get_200.value == 2.0

    post_201 = next(v for v in values if v.labels == {"method": "POST", "status": "201"})
    assert post_201.value == 1.0


def test_counter_negative_increment_raises():
    """Test that negative increments raise an error."""
    counter = Counter("test_counter")

    try:
        counter.inc(-1.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "only increase" in str(e)


def test_gauge_basic():
    """Test basic gauge operations."""
    gauge = Gauge("test_gauge", "A test gauge")

    gauge.set(42.0)
    assert gauge.collect()[0].value == 42.0

    gauge.inc(8.0)
    assert gauge.collect()[0].value == 50.0

    gauge.dec(10.0)
    assert gauge.collect()[0].value == 40.0


def test_gauge_with_labels():
    """Test gauge with labels."""
    gauge = Gauge("test_gauge", "A test gauge", label_names=["host"])

    gauge.set(10.0, labels={"host": "host1"})
    gauge.set(20.0, labels={"host": "host2"})

    values = gauge.collect()
    assert len(values) == 2

    host1 = next(v for v in values if v.labels == {"host": "host1"})
    assert host1.value == 10.0

    host2 = next(v for v in values if v.labels == {"host": "host2"})
    assert host2.value == 20.0


def test_histogram_basic():
    """Test basic histogram operations."""
    histogram = Histogram("test_histogram", "A test histogram")

    histogram.observe(0.1)
    histogram.observe(0.5)
    histogram.observe(1.5)
    histogram.observe(5.0)

    values = histogram.collect()
    assert len(values) == 1

    value = values[0]
    assert value.value == 7.1  # sum
    assert value.labels["_count"] == "4"


def test_histogram_custom_buckets():
    """Test histogram with custom buckets."""
    histogram = Histogram(
        "test_histogram",
        "A test histogram",
        buckets=[0.1, 0.5, 1.0, 5.0, 10.0],
    )

    histogram.observe(0.05)
    histogram.observe(0.3)
    histogram.observe(0.8)
    histogram.observe(3.0)

    values = histogram.collect()
    assert len(values) == 1
    assert values[0].labels["_count"] == "4"


def test_histogram_with_labels():
    """Test histogram with labels."""
    histogram = Histogram("test_histogram", "A test histogram", label_names=["endpoint"])

    histogram.observe(0.1, labels={"endpoint": "/api/v1"})
    histogram.observe(0.2, labels={"endpoint": "/api/v1"})
    histogram.observe(0.5, labels={"endpoint": "/api/v2"})

    values = histogram.collect()
    assert len(values) == 2

    v1 = next(v for v in values if v.labels.get("endpoint") == "/api/v1")
    assert v1.labels["_count"] == "2"

    v2 = next(v for v in values if v.labels.get("endpoint") == "/api/v2")
    assert v2.labels["_count"] == "1"


def test_metrics_registry_disabled_by_default():
    """Test that metrics registry is disabled by default."""
    registry = MetricsRegistry()
    assert not registry.is_enabled()

    counter = registry.counter("test_counter")
    counter.inc()

    # Metrics should not be collected when disabled
    metrics = registry.collect_all()
    assert metrics == {}


def test_metrics_registry_enabled():
    """Test that metrics are collected when enabled."""
    registry = MetricsRegistry(enabled=True)
    assert registry.is_enabled()

    counter = registry.counter("test_counter")
    counter.inc(5.0)

    metrics = registry.collect_all()
    assert "test_counter" in metrics
    assert metrics["test_counter"][0].value == 5.0


def test_metrics_registry_enable_disable():
    """Test enabling and disabling metrics collection."""
    registry = MetricsRegistry()

    registry.enable()
    assert registry.is_enabled()

    counter = registry.counter("test_counter")
    counter.inc()

    metrics = registry.collect_all()
    assert "test_counter" in metrics

    registry.disable()
    assert not registry.is_enabled()

    metrics = registry.collect_all()
    assert metrics == {}


def test_metrics_registry_reuses_metrics():
    """Test that registry reuses existing metrics."""
    registry = MetricsRegistry(enabled=True)

    counter1 = registry.counter("test_counter")
    counter1.inc()

    counter2 = registry.counter("test_counter")
    counter2.inc()

    # Should be the same instance
    assert counter1 is counter2

    metrics = registry.collect_all()
    assert metrics["test_counter"][0].value == 2.0


def test_global_registry():
    """Test global registry singleton."""
    registry1 = get_metrics_registry()
    registry2 = get_metrics_registry()

    assert registry1 is registry2


def test_configure_metrics():
    """Test configuring global metrics."""
    registry = configure_metrics(enabled=True)
    assert registry.is_enabled()

    counter = registry.counter("test_counter")
    counter.inc()

    metrics = registry.collect_all()
    assert "test_counter" in metrics

    # Clean up
    configure_metrics(enabled=False)


def test_metric_types():
    """Test metric type identification."""
    counter = Counter("test_counter")
    gauge = Gauge("test_gauge")
    histogram = Histogram("test_histogram")

    assert counter.get_type() == MetricType.COUNTER
    assert gauge.get_type() == MetricType.GAUGE
    assert histogram.get_type() == MetricType.HISTOGRAM


def test_label_validation():
    """Test that label validation works correctly."""
    counter = Counter("test_counter", label_names=["method", "status"])

    # Should work with correct labels
    counter.inc(labels={"method": "GET", "status": "200"})

    # Should fail with missing labels
    try:
        counter.inc(labels={"method": "GET"})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "do not match" in str(e)

    # Should fail with extra labels
    try:
        counter.inc(labels={"method": "GET", "status": "200", "extra": "value"})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "do not match" in str(e)


def test_metric_timestamp():
    """Test that metrics have timestamps."""
    counter = Counter("test_counter")
    counter.inc()

    before = time.time()
    values = counter.collect()
    after = time.time()

    assert len(values) == 1
    # Timestamp is set at collection time
    assert before <= values[0].timestamp <= after


if __name__ == "__main__":
    import sys

    # Run all test functions
    test_functions = [
        test_counter_basic,
        test_counter_with_labels,
        test_counter_negative_increment_raises,
        test_gauge_basic,
        test_gauge_with_labels,
        test_histogram_basic,
        test_histogram_custom_buckets,
        test_histogram_with_labels,
        test_metrics_registry_disabled_by_default,
        test_metrics_registry_enabled,
        test_metrics_registry_enable_disable,
        test_metrics_registry_reuses_metrics,
        test_global_registry,
        test_configure_metrics,
        test_metric_types,
        test_label_validation,
        test_metric_timestamp,
    ]

    failed = []
    for test_fn in test_functions:
        try:
            test_fn()
            print(f"✓ {test_fn.__name__}")
        except Exception as e:
            print(f"✗ {test_fn.__name__}: {e}")
            failed.append(test_fn.__name__)

    if failed:
        print(f"\n{len(failed)} tests failed:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print(f"\nAll {len(test_functions)} tests passed!")
        sys.exit(0)
