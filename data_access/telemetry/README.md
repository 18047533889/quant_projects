# Telemetry System

Privacy-safe, opt-in monitoring for factor_engine.

## Overview

The telemetry subsystem provides three core capabilities:

1. **Metrics** - counter/gauge/histogram for key operations
2. **Traces** - distributed tracing support with span tracking
3. **Health** - health check endpoints for component monitoring

All telemetry is **disabled by default** and requires explicit opt-in. No data is collected unless enabled.

## Quick Start

### Metrics

```python
from telemetry import get_metrics_registry, configure_metrics

# Enable metrics (opt-in)
configure_metrics(enabled=True)

registry = get_metrics_registry()

# Create metrics
request_counter = registry.counter(
    "requests_total",
    description="Total requests",
    label_names=["method", "status"],
)

latency_histogram = registry.histogram(
    "request_duration_seconds",
    description="Request latency",
    label_names=["endpoint"],
)

# Use metrics
request_counter.inc(labels={"method": "GET", "status": "200"})
latency_histogram.observe(0.123, labels={"endpoint": "/api/v1"})

# Collect all metrics
metrics = registry.collect_all()
```

### Traces

```python
from telemetry import get_tracer, configure_tracing

# Enable tracing (opt-in)
configure_tracing(enabled=True)

tracer = get_tracer(__name__)

# Manual span
with tracer.trace("operation_name") as span:
    span.set_attribute("key", "value")
    span.add_event("checkpoint")
    # ... do work ...

# Decorator
from telemetry import trace

@trace(name="compute_factor")
def compute_factor(data):
    # ... computation ...
    return result
```

### Health Checks

```python
from telemetry import get_health_checker, configure_health_checks, ComponentHealth, HealthStatus

# Enable health checks (opt-in)
configure_health_checks(enabled=True)

checker = get_health_checker()

# Register health checks
def check_database():
    try:
        # ... check database connection ...
        return ComponentHealth("database", HealthStatus.HEALTHY)
    except Exception as e:
        return ComponentHealth("database", HealthStatus.UNHEALTHY, str(e))

checker.register("database", check_database, interval=30.0)

# Check health
system_health = checker.check_all()
print(f"Status: {system_health.status.value}")

# Get as dict (for HTTP endpoints)
status_dict = checker.get_status_dict()
```

## Optional Exporters

### Prometheus

Requires: `pip install prometheus-client`

```python
from telemetry.prometheus_exporter import PrometheusExporter

exporter = PrometheusExporter()
registry.add_exporter(exporter)

# Start HTTP server
exporter.start_http_server(port=9090)
```

### OpenTelemetry

Requires: `pip install opentelemetry-api opentelemetry-sdk`

```python
from telemetry.opentelemetry_bridge import (
    create_opentelemetry_bridge,
    OpenTelemetrySpanExporter,
)
from telemetry.traces import SimpleSpanProcessor

# Create bridge
bridge = create_opentelemetry_bridge(service_name="factor-engine")

# Add to tracer provider
exporter = OpenTelemetrySpanExporter(bridge)
processor = SimpleSpanProcessor(exporter)
provider.add_span_processor(processor)
```

#### Jaeger Integration

Requires: `pip install opentelemetry-exporter-jaeger`

```python
from telemetry.opentelemetry_bridge import create_jaeger_exporter

jaeger_exporter = create_jaeger_exporter(
    agent_host="localhost",
    agent_port=6831,
)
```

#### OTLP Integration

Requires: `pip install opentelemetry-exporter-otlp`

```python
from telemetry.opentelemetry_bridge import create_otlp_exporter

otlp_exporter = create_otlp_exporter(
    endpoint="http://localhost:4317",
)
```

## Privacy & Security

- All telemetry is **opt-in only** - disabled by default
- No personal or sensitive data is collected
- Metric names and labels are controlled by the application
- Health checks can include details but should not expose secrets
- Trace attributes should be sanitized before recording

## Testing

Run the test suite:

```bash
# Test metrics
python telemetry/test_metrics.py

# Test traces
python telemetry/test_traces.py

# Test health checks
python telemetry/test_health.py
```

## Architecture

### Metrics

- **Counter**: Monotonically increasing value (requests, errors)
- **Gauge**: Value that can go up or down (memory usage, queue depth)
- **Histogram**: Distribution of values (latency, request size)

All metrics support labels for dimensional data.

### Traces

- **Span**: A single unit of work with start/end time
- **Tracer**: Creates and manages spans
- **SpanProcessor**: Handles completed spans
- **SpanExporter**: Exports spans to backend

Supports nested spans for distributed tracing.

### Health

- **ComponentHealth**: Health status of a single component
- **SystemHealth**: Aggregate health of all components
- **HealthCheck**: Reusable check with caching and timeout

Health checks run periodically with configurable intervals.

## Design Principles

1. **Opt-in by default** - Nothing runs unless explicitly enabled
2. **Zero overhead when disabled** - No performance impact
3. **Graceful degradation** - Missing exporters don't break the app
4. **Thread-safe** - Safe for concurrent use
5. **Privacy-first** - No sensitive data by default
