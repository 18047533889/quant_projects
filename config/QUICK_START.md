# Quick Start: Production Configuration

This guide helps you get started with the production configuration, logging, and monitoring system.

## 5-Minute Quick Start

### 1. Install Dependencies

```bash
pip install prometheus-client psutil pyyaml
```

### 2. Run the Example HTTP Server

```bash
# Using environment variables (default)
python scripts/run_production_server.py

# Using configuration file
python scripts/run_production_server.py --config config/examples/development.yaml

# Override settings
python scripts/run_production_server.py --log-level DEBUG --port 8081
```

### 3. Test the Endpoints

```bash
# Health check (startup)
curl http://localhost:8080/health

# Liveness probe
curl http://localhost:8080/health/live

# Readiness probe
curl http://localhost:8080/health/ready

# Prometheus metrics
curl http://localhost:8080/metrics
```

## Basic Usage in Your Code

### Minimal Setup

```python
from config.production_config import ProductionConfig
from config.logging.structured_logger import configure_from_config
from config.monitoring import configure_metrics, configure_health_checks

# Load configuration
config = ProductionConfig.from_env()
config.validate_and_raise()

# Setup logging
logger = configure_from_config(config)

# Setup metrics
metrics = configure_metrics(config)

# Setup health checks
health = configure_health_checks(config)

# Use them
logger.info("Service started", context={"version": "1.0.0"})

with metrics.track_operation("process_data"):
    # Your code here
    pass
```

### Adding to Existing Service

```python
# Add to your main service class
class MyService:
    def __init__(self):
        config = ProductionConfig.from_env()
        self.logger = configure_from_config(config)
        self.metrics = configure_metrics(config)
        self.health = configure_health_checks(config)
        
    def run(self):
        self.health.mark_startup_complete()
        self.logger.info("Service ready")
        
        while True:
            with self.metrics.track_operation("main_loop"):
                self.process()
```

## Environment Variables

Create a `.env` file:

```bash
# Basic settings
QUANT_ENV=production
SERVICE_NAME=my-service
VERSION=1.0.0

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_DIR=/var/log/quant

# Monitoring
METRICS_ENABLED=true
METRICS_PORT=9090
HEALTH_CHECK_PORT=8080
```

## Configuration File

Create `config/my-service.yaml`:

```yaml
environment: production
service_name: my-service

logging:
  level: INFO
  format: json
  output_dir: /var/log/quant

metrics:
  enabled: true
  export_port: 9090

health_check:
  enabled: true
  port: 8080
```

Load it:

```python
import yaml
from config.production_config import ProductionConfig

with open("config/my-service.yaml") as f:
    config_dict = yaml.safe_load(f)

config = ProductionConfig(config_dict)
config.validate_and_raise()
```

## Common Patterns

### 1. Logging with Context

```python
logger.info(
    "Processing batch",
    context={
        "batch_id": "12345",
        "size": 1000,
        "type": "factors"
    }
)

# Automatic sanitization of sensitive data
logger.info(
    "Database connection established",
    context={
        "host": "db.example.com",
        "password": "secret123"  # Automatically redacted
    }
)
```

### 2. Performance Tracking

```python
# As decorator
@metrics.track_performance("compute_factor")
def compute_factor(data):
    return process(data)

# As context manager
with metrics.track_operation("database_query"):
    results = db.query(...)

# Manual tracking
from config.logging.performance import log_performance

@log_performance("heavy_computation")
def compute():
    pass
```

### 3. Custom Health Checks

```python
from config.monitoring.health import CustomCheck, HealthCheckResult, HealthStatus

def check_my_component():
    try:
        # Check your component
        if component.is_healthy():
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Component OK"
            )
        else:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message="Component failed"
            )
    except Exception as e:
        return HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            message=str(e)
        )

health.add_check(CustomCheck("my_component", check_my_component))
```

### 4. Custom Metrics

```python
# Record business metrics
metrics.record_factor_evaluation("momentum", duration_seconds=0.123)
metrics.record_data_points("timeseries", count=10000)

# Record cache performance
metrics.record_cache_hit("memory")
metrics.record_cache_miss("disk")

# Record errors
metrics.record_error("process_batch", "ValueError")
```

### 5. Graceful Shutdown

```python
from config.monitoring.shutdown import configure_graceful_shutdown

def cleanup_database():
    db.close()

def cleanup_cache():
    cache.flush()
    cache.close()

shutdown = configure_graceful_shutdown(
    config,
    shutdown_hooks=[cleanup_cache, cleanup_database]
)

# Hooks are called in reverse order on SIGTERM/SIGINT
```

## Testing

### Run Tests

```bash
# All tests
python -m pytest tests/test_production_config.py -v

# Specific test
python -m pytest tests/test_production_config.py::TestProductionConfig -v

# With coverage
python -m pytest tests/test_production_config.py --cov=config --cov-report=html
```

### Test Your Configuration

```python
def test_my_config():
    config = ProductionConfig.from_env()
    
    # Should not raise
    config.validate_and_raise()
    
    # Check specific values
    assert config.service_name == "my-service"
    assert config.logging.level == "INFO"
```

## Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Create log directory
RUN mkdir -p /var/log/quant

# Expose ports
EXPOSE 8080 9090

# Run service
CMD ["python", "-m", "my_service"]
```

```bash
docker build -t my-service:latest .
docker run -p 8080:8080 -p 9090:9090 \
  -e QUANT_ENV=production \
  -e SERVICE_NAME=my-service \
  my-service:latest
```

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-service
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: my-service
        image: my-service:latest
        ports:
        - containerPort: 8080
          name: health
        - containerPort: 9090
          name: metrics
        env:
        - name: QUANT_ENV
          value: "production"
        - name: SERVICE_NAME
          value: "my-service"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
```

## Monitoring Setup

### Prometheus Configuration

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'my-service'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 15s
```

### Grafana Dashboard

1. Go to Grafana
2. Import dashboard from `monitoring/grafana-dashboards/quant-overview.json`
3. Select Prometheus data source
4. View metrics

## Troubleshooting

### Logs Not Appearing

```python
# Check log directory exists
from pathlib import Path
log_dir = Path(config.logging.output_dir)
assert log_dir.exists(), f"Log directory {log_dir} does not exist"

# Check permissions
assert log_dir.is_dir()
```

### Metrics Not Working

```bash
# Check if prometheus-client is installed
pip show prometheus-client

# Check metrics endpoint
curl http://localhost:9090/metrics
```

### Health Checks Failing

```bash
# Check health endpoint
curl -v http://localhost:8080/health

# Check logs for errors
tail -f /var/log/quant/my-service.log | grep ERROR
```

## Next Steps

1. Read the full [Production Deployment Guide](../docs/PRODUCTION_DEPLOYMENT.md)
2. Review [example implementations](../examples/)
3. Customize configuration for your service
4. Add custom metrics and health checks
5. Set up monitoring dashboard
6. Configure alerting rules

## Resources

- **Configuration**: `config/production_config.py`
- **Logging**: `config/logging/`
- **Monitoring**: `config/monitoring/`
- **Examples**: `examples/`
- **Tests**: `tests/test_production_config.py`
- **Docs**: `docs/PRODUCTION_DEPLOYMENT.md`

## Support

- Check logs first
- Review configuration validation errors
- Test health check endpoints
- Verify environment variables
- Consult full documentation

---

**Quick Start Version**: 1.0.0  
**Last Updated**: 2024-08-14
