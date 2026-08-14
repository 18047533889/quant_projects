# Production Deployment Guide

This guide covers deploying the Quant Platform in production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Configuration](#configuration)
3. [Deployment Options](#deployment-options)
4. [Monitoring and Observability](#monitoring-and-observability)
5. [Health Checks](#health-checks)
6. [Logging](#logging)
7. [Security](#security)
8. [Performance Tuning](#performance-tuning)
9. [Troubleshooting](#troubleshooting)
10. [Disaster Recovery](#disaster-recovery)

## Prerequisites

### System Requirements

- **CPU**: 8+ cores recommended
- **Memory**: 32GB+ RAM recommended
- **Storage**: 500GB+ SSD with high IOPS
- **OS**: Linux (Ubuntu 20.04+, CentOS 8+, RHEL 8+)
- **Python**: 3.9+

### Dependencies

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    python3.9 \
    python3-pip \
    postgresql-client \
    redis-tools \
    build-essential

# Install Python packages
pip install -r requirements.txt
pip install prometheus-client psutil
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Environment
QUANT_ENV=production
SERVICE_NAME=quant-platform
VERSION=1.0.0

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_DIR=/var/log/quant

# Metrics
METRICS_ENABLED=true
METRICS_PORT=9090

# Health Checks
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_PORT=8080

# Resource Limits
MAX_MEMORY_MB=30720
MAX_CONCURRENT_EVALUATIONS=100

# Security
REQUIRE_AUTHENTICATION=true
ENABLE_TLS=true
TLS_CERT_FILE=/etc/quant/certs/server.crt
TLS_KEY_FILE=/etc/quant/certs/server.key

# Database (if applicable)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=quant
DB_USER=quant_user
DB_PASSWORD_FILE=/run/secrets/db_password

# Cache (if applicable)
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Configuration File

Create `config/production.yaml`:

```yaml
environment: production
service_name: quant-platform
version: ${VERSION}

logging:
  level: INFO
  format: json
  output_dir: /var/log/quant
  max_file_size_mb: 100
  backup_count: 10
  enable_console: true
  enable_file: true
  enable_syslog: false
  sanitize_sensitive: true
  include_context: true

metrics:
  enabled: true
  export_port: 9090
  export_path: /metrics
  collect_interval_seconds: 60
  enable_latency_metrics: true
  enable_throughput_metrics: true
  enable_error_metrics: true
  enable_memory_metrics: true
  enable_cpu_metrics: true
  enable_factor_metrics: true

health_check:
  enabled: true
  port: 8080
  path: /health
  check_disk_space: true
  disk_space_threshold_percent: 90.0
  check_memory: true
  memory_threshold_percent: 95.0

resource_limits:
  max_memory_mb: 30720
  warning_memory_threshold_percent: 80.0
  max_concurrent_evaluations: 100
  max_concurrent_io_operations: 50
  max_queue_size: 1000
  default_operation_timeout_seconds: 300
  enable_rate_limiting: true
  requests_per_minute: 1000

security:
  require_authentication: true
  enable_tls: true
  tls_cert_file: /etc/quant/certs/server.crt
  tls_key_file: /etc/quant/certs/server.key
  validate_input: true
  max_request_size_mb: 10
  enable_audit_log: true
  audit_log_path: /var/log/quant/audit.log

shutdown_timeout_seconds: 30
enable_graceful_shutdown: true
```

### Loading Configuration

```python
from config.production_config import ProductionConfig
from config.logging.structured_logger import configure_from_config
from config.monitoring import configure_metrics, configure_health_checks
from config.monitoring.shutdown import configure_graceful_shutdown

# Load configuration
config = ProductionConfig.from_env()
config.validate_and_raise()

# Setup logging
logger = configure_from_config(config)
logger.info("Service starting", context={"version": config.version})

# Setup metrics
metrics = configure_metrics(config)

# Setup health checks
health = configure_health_checks(config)

# Setup graceful shutdown
shutdown_handler = configure_graceful_shutdown(config, shutdown_hooks=[
    cleanup_resources,
    close_connections,
])
```

## Deployment Options

### Option 1: Systemd Service

Create `/etc/systemd/system/quant-platform.service`:

```ini
[Unit]
Description=Quant Platform Service
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=quant
Group=quant
WorkingDirectory=/opt/quant
EnvironmentFile=/etc/quant/environment
ExecStart=/opt/quant/.venv/bin/python -m quant_platform.main
ExecReload=/bin/kill -HUP $MAINPID
Restart=on-failure
RestartSec=5s
TimeoutStopSec=30

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log/quant /var/lib/quant

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable quant-platform
sudo systemctl start quant-platform
sudo systemctl status quant-platform
```

### Option 2: Docker

See `Dockerfile` and `docker-compose.production.yml` in the project root.

```bash
# Build image
docker build -t quant-platform:latest .

# Run with docker-compose
docker-compose -f docker-compose.production.yml up -d

# Check status
docker-compose -f docker-compose.production.yml ps
docker-compose -f docker-compose.production.yml logs -f
```

### Option 3: Kubernetes

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quant-platform
  labels:
    app: quant-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: quant-platform
  template:
    metadata:
      labels:
        app: quant-platform
    spec:
      containers:
      - name: quant-platform
        image: quant-platform:latest
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: QUANT_ENV
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
          limits:
            memory: "32Gi"
            cpu: "8"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 3
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        startupProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 30
```

Deploy:

```bash
kubectl apply -f k8s/
kubectl get pods -l app=quant-platform
kubectl logs -f deployment/quant-platform
```

## Monitoring and Observability

### Prometheus Metrics

Metrics are exposed at `http://localhost:9090/metrics`.

**Key Metrics:**

- `quant_request_duration_seconds` - Request latency histogram
- `quant_operations_total` - Total operations counter
- `quant_errors_total` - Error counter by type
- `quant_memory_bytes` - Memory usage
- `quant_cpu_percent` - CPU usage
- `quant_factors_evaluated_total` - Factors evaluated
- `quant_cache_hits_total` / `quant_cache_misses_total` - Cache efficiency

### Prometheus Configuration

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'quant-platform'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 15s
    scrape_timeout: 10s
```

### Grafana Dashboards

Import pre-built dashboards from `monitoring/grafana-dashboards/`:

1. `quant-overview.json` - Service overview
2. `quant-performance.json` - Performance metrics
3. `quant-resources.json` - Resource usage
4. `quant-business.json` - Business metrics

### Alerting

Configure alerts in `monitoring/alerts.yml`. Critical alerts:

- High error rate (>5% over 5 minutes)
- High memory usage (>90%)
- High disk usage (>90%)
- Service unavailable
- Slow response times (p99 > 5s)

## Health Checks

### Endpoints

- **Startup**: `GET /health` - Returns `ready` when startup complete
- **Liveness**: `GET /health/live` - Checks if service is alive
- **Readiness**: `GET /health/ready` - Checks if service can handle traffic

### Response Format

```json
{
  "status": "healthy",
  "service": "quant-platform",
  "timestamp": 1692345678.123,
  "checks": [
    {
      "name": "disk_space",
      "status": "healthy",
      "message": "Disk space OK: 45.2% used",
      "details": {
        "percent_used": 45.2,
        "free_gb": 250.5
      }
    },
    {
      "name": "memory",
      "status": "healthy",
      "message": "Memory OK: 65.3% used",
      "details": {
        "percent_used": 65.3,
        "available_gb": 11.2
      }
    }
  ]
}
```

### Custom Health Checks

```python
from config.monitoring.health import CustomCheck, HealthCheckResult, HealthStatus

def check_database_connection():
    try:
        db.ping()
        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="Database connection OK"
        )
    except Exception as e:
        return HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            message=f"Database connection failed: {e}"
        )

health.add_check(CustomCheck("database", check_database_connection))
```

## Logging

### Log Location

- Console: stdout (JSON format)
- File: `/var/log/quant/quant-platform.log`
- Rotated logs: `/var/log/quant/quant-platform.log.1.gz`, etc.
- Audit log: `/var/log/quant/audit.log`

### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages (default)
- **WARNING**: Warning messages for recoverable issues
- **ERROR**: Error messages for failures
- **CRITICAL**: Critical failures requiring immediate attention

### Log Format

JSON structured logs:

```json
{
  "timestamp": "2024-08-14T10:30:45.123Z",
  "level": "INFO",
  "logger": "quant-platform.factor_engine",
  "message": "Factor computed successfully",
  "service": "quant-platform",
  "module": "compute",
  "function": "compute_factor",
  "line": 145,
  "duration_ms": 123.45,
  "context": {
    "factor_id": "momentum_1d",
    "symbols": 500
  }
}
```

### Centralized Logging

Forward logs to ELK stack, Splunk, or similar:

```bash
# Filebeat configuration
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/quant/*.log
  json.keys_under_root: true
  json.add_error_key: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
```

## Security

### TLS Configuration

Generate certificates:

```bash
# Self-signed (development only)
openssl req -x509 -newkey rsa:4096 \
  -keyout server.key -out server.crt \
  -days 365 -nodes

# Production: Use Let's Encrypt or internal CA
```

### Authentication

API key authentication:

```python
# In requests
headers = {
    'X-API-Key': 'your-api-key-here'
}
```

### Secrets Management

Use environment variables or secret managers:

```bash
# Kubernetes secrets
kubectl create secret generic quant-secrets \
  --from-literal=db-password=<password> \
  --from-literal=api-key=<key>

# Docker secrets
echo "db_password" | docker secret create db_password -
```

### Audit Logging

All sensitive operations are logged to `/var/log/quant/audit.log`:

```json
{
  "timestamp": "2024-08-14T10:30:45.123Z",
  "user": "admin",
  "action": "factor_evaluated",
  "resource": "momentum_1d",
  "result": "success",
  "ip_address": "192.168.1.100"
}
```

## Performance Tuning

### Memory Optimization

```yaml
resource_limits:
  max_memory_mb: 30720  # Adjust based on available RAM
  warning_memory_threshold_percent: 80.0
```

### Concurrency Tuning

```yaml
resource_limits:
  max_concurrent_evaluations: 100  # Adjust based on CPU cores
  max_concurrent_io_operations: 50
  max_queue_size: 1000
```

### Cache Configuration

Enable caching for frequently accessed data:

```python
from factor_optimizer.cache_manager import CacheManager

cache = CacheManager(
    max_size_mb=10240,  # 10GB
    ttl_seconds=3600,   # 1 hour
)
```

### Database Connection Pooling

```python
# Adjust pool size based on concurrency
pool_size = 20
max_overflow = 10
pool_timeout = 30
```

## Troubleshooting

### High Memory Usage

1. Check current usage: `docker stats` or `top`
2. Review memory metrics: Check Grafana dashboard
3. Identify memory leaks: `python -m memory_profiler script.py`
4. Adjust limits in configuration

### High CPU Usage

1. Check CPU metrics in Grafana
2. Profile code: `python -m cProfile script.py`
3. Check for infinite loops or inefficient algorithms
4. Scale horizontally if needed

### Slow Performance

1. Check latency metrics (p50, p95, p99)
2. Review logs for slow operations
3. Check database query performance
4. Review cache hit rate
5. Profile critical paths

### Service Crashes

1. Check logs: `journalctl -u quant-platform -n 100`
2. Review health check failures
3. Check resource limits
4. Analyze core dumps if available

### Common Issues

**Issue**: Service won't start
- Check configuration validity
- Verify dependencies are installed
- Check port availability
- Review systemd logs

**Issue**: Health checks failing
- Verify thresholds are appropriate
- Check dependent services (DB, cache)
- Review resource usage

**Issue**: High error rate
- Check error logs for patterns
- Review recent deployments
- Verify data quality
- Check external dependencies

## Disaster Recovery

### Backup Strategy

**What to backup:**
- Configuration files: `/etc/quant/`
- Log files: `/var/log/quant/`
- Data files: `/var/lib/quant/`
- Database dumps

**Backup frequency:**
- Configurations: Before every change
- Logs: Retained for 30 days
- Data: Daily incremental, weekly full

### Restore Procedure

1. Stop service: `systemctl stop quant-platform`
2. Restore configuration: Copy from backup
3. Restore data: Restore from latest backup
4. Verify integrity: Run validation scripts
5. Start service: `systemctl start quant-platform`
6. Verify health: Check health endpoints

### High Availability

For HA deployments:

1. Run multiple replicas (3+ recommended)
2. Use load balancer for traffic distribution
3. Configure service mesh for resilience
4. Implement circuit breakers
5. Use distributed caching

### Monitoring Checklist

- [ ] Prometheus scraping metrics successfully
- [ ] Grafana dashboards showing data
- [ ] Alerts configured and tested
- [ ] Health checks passing
- [ ] Logs flowing to centralized system
- [ ] Backups running successfully
- [ ] Resource usage within limits
- [ ] Error rate below threshold

## Support

For issues or questions:

- Check logs first: `/var/log/quant/`
- Review metrics in Grafana
- Consult troubleshooting section
- File issue with full context

---

**Last Updated**: 2024-08-14
**Version**: 1.0.0
