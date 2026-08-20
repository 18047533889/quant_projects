# Production Deployment Guide

Comprehensive deployment guide for the quantitative research platform consisting of **data_access** (unified data layer) and **factor_engine** (factor computation DSL).

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Production Installation](#production-installation)
- [Environment Configuration](#environment-configuration)
- [Performance Tuning](#performance-tuning)
- [Monitoring Setup](#monitoring-setup)
- [Backup and Disaster Recovery](#backup-and-disaster-recovery)
- [Scaling Strategies](#scaling-strategies)
- [Common Troubleshooting](#common-troubleshooting)
- [Production Checklist](#production-checklist)

---

## Architecture Overview

### Core Components

| Component | Purpose | Port | Dependencies |
|-----------|---------|------|--------------|
| **data_access** | Unified data read/write (DuckDB + Parquet + COS) | 8765 | Python 3.10+, DuckDB, PyArrow |
| **factor_engine** | Factor DSL compilation and materialization | 8766 | data_access, NumPy, Pandas, Polars |

### Data Flow

```
COS/S3 Storage → data_access (mirror/remote) → DuckDB Query Engine
                                              ↓
                                         factor_engine DSL
                                              ↓
                                    Parquet Factor Lake (partitioned)
```

---

## Production Installation

### Prerequisites

- **OS**: Linux (Ubuntu 20.04+ or RHEL 8+)
- **Python**: 3.10 or 3.11 (3.12 not tested)
- **Memory**: Minimum 16GB RAM (32GB+ recommended for large panels)
- **Storage**: SSD with 500GB+ for data cache
- **Network**: Stable connection to COS/S3 endpoints

### Step 1: System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    python3.10 python3.10-venv python3.10-dev \
    build-essential git curl \
    libssl-dev libffi-dev

# RHEL/CentOS
sudo yum install -y \
    python310 python310-devel \
    gcc gcc-c++ make git curl \
    openssl-devel libffi-devel
```

### Step 2: Clone Repository

```bash
cd /opt
git clone https://github.com/18047533889/quant_projects.git
cd quant_projects
```

### Step 3: Python Virtual Environment

```bash
python3.10 -m venv /opt/quant_projects/venv
source /opt/quant_projects/venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### Step 4: Install Dependencies

```bash
# Core runtime dependencies
pip install -r requirements.txt

# data_access with all features
pip install -e "./dataaccess[all]"

# factor_engine with performance optimizations
pip install -e "./factor_engine[full,performance,service]"
```

### Step 5: Verify Installation

```bash
bash scripts/setup_quant_projects.sh --verify-only
```

---

## Environment Configuration

### Production Environment Variables

Create `/opt/quant_projects/.env.production`:

```bash
# ========================================
# Core Paths
# ========================================
QUANT_PROJECTS_ROOT=/opt/quant_projects
QUANTSOCIETY_WORKSPACE_DATA_ROOT=/data/quant_workspace
FACTOR_LAKE_ROOT=/data/quant_workspace/factors/lake

# ========================================
# data_access Configuration
# ========================================
# API Security
DATA_ACCESS_API_KEY=<STRONG_SECRET_KEY_FROM_VAULT>
DATA_ACCESS_API_HOST=0.0.0.0
DATA_ACCESS_API_PORT=8765
DATA_ACCESS_API_MAX_CONCURRENCY=8

# COS/S3 Backend
DATA_ACCESS_COS_READ_MODE=remote
DATA_ACCESS_COS_S3_ENDPOINT=cos.ap-guangzhou.myqcloud.com
DATA_ACCESS_COS_CACHE_ROOT=/data/quant_workspace/cos_cache
DATA_ACCESS_SKIP_COS_MIRROR=0

# Credentials (use secret manager in production)
COS_SECRET_ID=<FROM_SECRET_MANAGER>
COS_SECRET_KEY=<FROM_SECRET_MANAGER>

# Data Source Roots
ASHARE_PARQUET_ROOT=/data/quant_workspace/a_share/lqtp_data
US_MASSIVE_ROOT=/data/quant_workspace/us_stock/massive_data
US_CLEAN_ROOT=/data/quant_workspace/us_stock/clean_data

# ========================================
# DuckDB Resource Limits
# ========================================
DUCKDB_MEMORY_LIMIT=24GB
DUCKDB_THREADS=12
DUCKDB_TEMP_DIRECTORY=/tmp/duckdb

# ========================================
# factor_engine Configuration
# ========================================
FACTOR_ENGINE_SERVICE_PORT=8766
FACTOR_ENGINE_MAX_WORKERS=4
FACTOR_ENGINE_USE_NUMBA=1

# ========================================
# Observability
# ========================================
QUANT_PRODUCTION_MODE=1
QUANT_LOG_LEVEL=INFO
QUANT_METRICS_PORT=9090

# ========================================
# Security
# ========================================
QUANT_SCHEMA_CHECK=strict
DATA_ACCESS_EXTRA_ALLOWED_ROOTS=/data/quant_workspace,/data/backup
```

### Loading Configuration

```bash
# Source before starting services
source /opt/quant_projects/env.sh
source /opt/quant_projects/.env.production
```

### Secret Management

**Never commit secrets to version control**. Use a secret manager:

```bash
# AWS Secrets Manager
export COS_SECRET_ID=$(aws secretsmanager get-secret-value \
    --secret-id prod/quant/cos_id --query SecretString --output text)

# Kubernetes Secrets
kubectl create secret generic quant-cos-credentials \
    --from-literal=secret_id=<value> \
    --from-literal=secret_key=<value>

# HashiCorp Vault
export COS_SECRET_ID=$(vault kv get -field=id secret/quant/cos)
```

---

## Performance Tuning

### DuckDB Optimization

```bash
# Memory sizing: 60-70% of available RAM
# For 32GB machine:
export DUCKDB_MEMORY_LIMIT=20GB

# Thread count: leave 2-4 cores for system
# For 16-core machine:
export DUCKDB_THREADS=12

# Temporary directory on fast SSD
export DUCKDB_TEMP_DIRECTORY=/mnt/nvme/duckdb_temp
```

### data_access Concurrency

```python
# dataaccess/config/performance.yaml
api:
  max_concurrency: 8  # Concurrent read requests
  request_timeout: 300  # Seconds
  
cache:
  ttl_seconds: 3600
  max_size_gb: 50
  
duckdb:
  pool_size: 4  # Connection pool
  query_timeout: 600
```

### factor_engine Parallelism

```yaml
# factor_engine/config/execution.yaml
execution:
  backend: duckdb  # duckdb | polars | pandas
  max_parallel_factors: 4
  adaptive_batch_size: true
  
resource_broker:
  target_memory_utilization: 0.75
  spill_threshold_gb: 2.0
  
performance:
  use_numba: true
  use_bottleneck: true
  blas_threads: 4
```

### Kernel Parameters (Linux)

```bash
# /etc/sysctl.d/99-quant.conf
# Increase file descriptors
fs.file-max = 2097152
fs.nr_open = 2097152

# Network tuning
net.core.somaxconn = 32768
net.ipv4.tcp_max_syn_backlog = 8192

# Memory management
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5

# Apply
sudo sysctl -p /etc/sysctl.d/99-quant.conf
```

### Process Limits

```bash
# /etc/security/limits.d/quant.conf
quantuser soft nofile 65536
quantuser hard nofile 1048576
quantuser soft nproc 32768
quantuser hard nproc 32768
```

---

## Monitoring Setup

### Health Check Endpoints

```bash
# data_access service
curl http://localhost:8765/health
# Expected: {"status": "healthy", "duckdb": "ok", "cos": "reachable"}

# factor_engine service
curl http://localhost:8766/health
# Expected: {"status": "ready", "operators": 554, "backends": ["pandas", "duckdb"]}
```

### Prometheus Metrics

Add to `/opt/quant_projects/metrics_exporter.py`:

```python
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import time

# Metrics
duckdb_query_duration = Histogram('duckdb_query_seconds', 'Query execution time')
cache_hit_ratio = Gauge('cache_hit_ratio', 'Cache hit rate')
active_requests = Gauge('active_requests', 'Active API requests')
cos_download_bytes = Counter('cos_download_bytes_total', 'COS download volume')

if __name__ == '__main__':
    start_http_server(9090)
    while True:
        time.sleep(60)
```

Run as systemd service (see systemd section below).

### Log Aggregation

```bash
# Ship logs to centralized logging
# Example: Fluent Bit configuration
# /etc/fluent-bit/fluent-bit.conf

[INPUT]
    Name tail
    Path /var/log/quant/*.log
    Tag quant

[OUTPUT]
    Name es
    Match quant
    Host elasticsearch.internal
    Port 9200
    Index quant-logs
```

### Alerting Rules

```yaml
# prometheus_alerts.yml
groups:
  - name: quant_platform
    interval: 30s
    rules:
      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes{job="data-access"} > 28e9
        for: 5m
        annotations:
          summary: "data_access memory usage > 28GB"
          
      - alert: DuckDBQuerySlow
        expr: histogram_quantile(0.95, duckdb_query_seconds) > 60
        for: 3m
        annotations:
          summary: "95th percentile query time > 60s"
          
      - alert: CacheHitRateLow
        expr: cache_hit_ratio < 0.6
        for: 10m
        annotations:
          summary: "Cache hit ratio < 60%"
```

---

## Backup and Disaster Recovery

### Data Backup Strategy

```bash
#!/bin/bash
# /opt/quant_projects/scripts/backup_production.sh

BACKUP_ROOT=/mnt/backup/quant
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 1. Backup factor lake (incremental)
rsync -av --link-dest="${BACKUP_ROOT}/latest" \
    /data/quant_workspace/factors/lake/ \
    "${BACKUP_ROOT}/factor_lake_${TIMESTAMP}/"

# 2. Backup configurations
tar czf "${BACKUP_ROOT}/config_${TIMESTAMP}.tar.gz" \
    /opt/quant_projects/dataaccess/config/ \
    /opt/quant_projects/factor_engine/config/ \
    /opt/quant_projects/.env.production

# 3. Backup DuckDB catalog metadata
sqlite3 /data/quant_workspace/duckdb/catalog.db ".backup '${BACKUP_ROOT}/catalog_${TIMESTAMP}.db'"

# 4. Update latest symlink
ln -sfn "${BACKUP_ROOT}/factor_lake_${TIMESTAMP}" "${BACKUP_ROOT}/latest"

# 5. Retain last 7 days, monthly archives
find "${BACKUP_ROOT}" -name "factor_lake_*" -mtime +7 -delete
```

### COS Sync (Disaster Recovery)

```bash
# Upload critical factor lake to COS
rclone sync /data/quant_workspace/factors/lake/ \
    cos:qs-cold/backup/factor_lake/ \
    --transfers=8 \
    --checkers=16 \
    --fast-list
```

### Recovery Procedure

```bash
# 1. Restore from backup
rsync -av /mnt/backup/quant/latest/ /data/quant_workspace/factors/lake/

# 2. Restore configuration
tar xzf /mnt/backup/quant/config_latest.tar.gz -C /

# 3. Restore DuckDB catalog
cp /mnt/backup/quant/catalog_latest.db /data/quant_workspace/duckdb/catalog.db

# 4. Verify integrity
cd /opt/quant_projects
python3 -c "
from data_access import get_store
store = get_store()
assert store.load_columns('ashare_stock_daily', columns=['Close'], time_range=('2024-01-01', '2024-01-02'))
print('Recovery verified')
"

# 5. Restart services
sudo systemctl restart data-access factor-engine
```

---

## Scaling Strategies

### Horizontal Scaling (Multi-Node)

```yaml
# kubernetes_deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: data-access
spec:
  replicas: 3  # Scale to 3 instances
  selector:
    matchLabels:
      app: data-access
  template:
    metadata:
      labels:
        app: data-access
    spec:
      containers:
      - name: data-access
        image: quant/data-access:0.10.2
        env:
        - name: DATA_ACCESS_API_MAX_CONCURRENCY
          value: "8"
        - name: DUCKDB_MEMORY_LIMIT
          value: "20GB"
        resources:
          requests:
            memory: "24Gi"
            cpu: "8"
          limits:
            memory: "30Gi"
            cpu: "12"
        volumeMounts:
        - name: cos-cache
          mountPath: /var/lib/dataaccess/cos-cache
      volumes:
      - name: cos-cache
        persistentVolumeClaim:
          claimName: data-access-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: data-access-lb
spec:
  type: LoadBalancer
  selector:
    app: data-access
  ports:
  - port: 8765
    targetPort: 8765
```

### Vertical Scaling (Resource Allocation)

```yaml
# Increase resources for single instance
resources:
  requests:
    memory: "48Gi"
    cpu: "16"
  limits:
    memory: "60Gi"
    cpu: "20"
    
env:
- name: DUCKDB_MEMORY_LIMIT
  value: "40GB"
- name: DUCKDB_THREADS
  value: "16"
- name: DATA_ACCESS_API_MAX_CONCURRENCY
  value: "16"
```

### Read Replica Pattern

```python
# Load balancing across read replicas
from data_access.service.client import DataAccessClient
import random

REPLICAS = [
    "http://data-access-0:8765",
    "http://data-access-1:8765",
    "http://data-access-2:8765",
]

def get_client():
    endpoint = random.choice(REPLICAS)
    return DataAccessClient(endpoint, api_key=API_KEY)
```

### Cache Tier Architecture

```
┌─────────────────┐
│  factor_engine  │
└────────┬────────┘
         │
    ┌────▼─────┐
    │  Redis   │ ← Hot cache (factor metadata, small panels)
    │  Cluster │
    └────┬─────┘
         │
┌────────▼─────────┐
│  data_access API │ ← Warm cache (COS mirror, DuckDB query cache)
└────────┬─────────┘
         │
    ┌────▼─────┐
    │ COS/S3   │ ← Cold storage (full historical data)
    └──────────┘
```

---

## Common Troubleshooting

### Issue: Out of Memory (OOM)

**Symptoms**: Process killed by OOM killer, `duckdb.OutOfMemoryException`

**Solution**:
```bash
# Check current memory usage
free -h
ps aux --sort=-%mem | head -10

# Reduce DuckDB memory limit
export DUCKDB_MEMORY_LIMIT=16GB  # Lower from 24GB

# Enable spill-to-disk
export DUCKDB_TEMP_DIRECTORY=/mnt/nvme/duckdb_spill

# Reduce concurrent requests
export DATA_ACCESS_API_MAX_CONCURRENCY=4
```

### Issue: Slow Query Performance

**Symptoms**: Queries taking > 60s, high CPU usage

**Diagnosis**:
```python
# Enable query profiling
import duckdb
conn = duckdb.connect()
conn.execute("SET enable_profiling = 'query_tree'")
conn.execute("SET profiling_output = '/tmp/profile.json'")

# Run slow query
result = conn.execute("SELECT ... FROM ...")

# Analyze profile
import json
with open('/tmp/profile.json') as f:
    profile = json.load(f)
    print(profile['children'])  # Look for bottlenecks
```

**Solution**:
```python
# Add indices to frequent filters
# Partition data by time
# Use columnar projections (select only needed columns)
# Enable aggressive pushdown

store.load_columns(
    "ashare_stock_daily",
    columns=["Close"],  # Not SELECT *
    time_range=("2024-01-01", "2024-01-31"),  # Narrow range
    filters={"Symbol": ["000001.SZ"]},  # Pushdown filters
)
```

### Issue: COS Connection Timeout

**Symptoms**: `COSClientError: Connection timeout`, slow data loading

**Solution**:
```bash
# Check network connectivity
ping cos.ap-guangzhou.myqcloud.com

# Test bandwidth
wget --spider http://cos.ap-guangzhou.myqcloud.com

# Use local mirror mode for repeated queries
export DATA_ACCESS_COS_READ_MODE=mirror
export DATA_ACCESS_SKIP_COS_MIRROR=0

# Increase timeout
export DATA_ACCESS_COS_TIMEOUT=600

# Use CDN endpoint if available
export DATA_ACCESS_COS_S3_ENDPOINT=cos.accelerate.myqcloud.com
```

### Issue: Factor Materialization Fails

**Symptoms**: `FactorComputationError`, incomplete parquet files

**Diagnosis**:
```bash
# Check logs
tail -f /var/log/quant/factor_engine.log

# Verify data availability
python3 -c "
from data_access import get_store
store = get_store()
df = store.load_columns('ashare_stock_daily', columns=['Close'], time_range=('2024-01-01', '2024-01-31'))
print(df.shape)
"

# Check disk space
df -h /data/quant_workspace
```

**Solution**:
```bash
# Clean up partial writes
rm -rf /data/quant_workspace/factors/lake/factors/broken_factor_id/

# Retry with smaller batch
python3 scripts/materialize_factor.py \
    --factor-id=my_factor \
    --batch-size=100 \
    --time-range=2024-01-01:2024-01-31

# Check operator contract
python3 -c "
from cleaned_operators.registry import get_operator
op = get_operator('my_operator')
print(op.contract)  # Verify inputs/outputs
"
```

### Issue: API Authentication Failures

**Symptoms**: `HTTP 401 Unauthorized`, `Invalid API key`

**Solution**:
```bash
# Verify API key is set
echo $DATA_ACCESS_API_KEY

# Test authentication
curl -H "X-API-Key: ${DATA_ACCESS_API_KEY}" http://localhost:8765/health

# Rotate API key (update in secret manager)
export DATA_ACCESS_API_KEY=<new_key>
sudo systemctl restart data-access

# Check service logs
journalctl -u data-access -n 100 --no-pager
```

### Issue: DuckDB Lock Contention

**Symptoms**: `Database is locked`, concurrent write failures

**Solution**:
```python
# Use WAL mode for concurrent reads
import duckdb
conn = duckdb.connect('/data/quant_workspace/catalog.db')
conn.execute("PRAGMA journal_mode=WAL")

# Serialize writes through queue
from service.queue import JobQueue
queue = JobQueue()
queue.submit(write_job)

# Use separate databases per job
conn = duckdb.connect(f'/tmp/job_{job_id}.db')
```

---

## Production Checklist

### Pre-Deployment

- [ ] All tests passing: `pytest dataaccess/tests/ factor_engine/tests/`
- [ ] Secrets stored in vault (not `.env` files)
- [ ] Resource limits configured (`DUCKDB_MEMORY_LIMIT`, `DUCKDB_THREADS`)
- [ ] Monitoring endpoints exposed (`:9090/metrics`)
- [ ] Backup script scheduled (daily cron)
- [ ] Alert rules configured (Prometheus/Grafana)
- [ ] Network firewall rules applied (8765, 8766, 9090)
- [ ] COS credentials validated
- [ ] Disk space allocated (500GB+ for cache)
- [ ] Log rotation configured (`/etc/logrotate.d/quant`)

### Security

- [ ] Non-root user for services (`useradd quantuser`)
- [ ] Read-only filesystem where possible
- [ ] API key authentication enforced (`DATA_ACCESS_API_KEY`)
- [ ] TLS/HTTPS for external endpoints
- [ ] Network segmentation (no direct internet access)
- [ ] Audit logging enabled
- [ ] Secrets rotation policy (90 days)
- [ ] Dependency vulnerability scanning (Snyk/Trivy)

### Performance

- [ ] Kernel parameters tuned (`/etc/sysctl.d/99-quant.conf`)
- [ ] Process limits raised (`/etc/security/limits.conf`)
- [ ] DuckDB temp directory on fast SSD
- [ ] COS cache on high-IOPS volume
- [ ] BLAS/OpenMP thread limits set
- [ ] Benchmark baseline established (see `benchmarks/`)

### Operational

- [ ] Systemd units installed and enabled
- [ ] Health check passing
- [ ] Runbook documented (this guide)
- [ ] On-call rotation defined
- [ ] Escalation path established
- [ ] Disaster recovery tested
- [ ] Rollback procedure documented
- [ ] Change management process followed

---

## Containerized Deployment

### Docker Compose (Single Node)

Create `/opt/quant_projects/docker-compose.production.yml`:

```yaml
version: '3.8'

services:
  data-access:
    build:
      context: ./dataaccess
      dockerfile: Dockerfile
    image: quant/data-access:0.10.2
    container_name: data-access-prod
    restart: unless-stopped
    ports:
      - "8765:8765"
    environment:
      DATA_ACCESS_API_KEY: ${DATA_ACCESS_API_KEY:?Required}
      DATA_ACCESS_API_MAX_CONCURRENCY: "8"
      DUCKDB_MEMORY_LIMIT: "20GB"
      DUCKDB_THREADS: "12"
      DUCKDB_TEMP_DIRECTORY: /tmp/duckdb
      DATA_ACCESS_COS_CACHE_ROOT: /var/lib/dataaccess/cos-cache
      DATA_ACCESS_COS_READ_MODE: remote
      DATA_ACCESS_COS_S3_ENDPOINT: ${COS_S3_ENDPOINT}
      COS_SECRET_ID: ${COS_SECRET_ID:?Required}
      COS_SECRET_KEY: ${COS_SECRET_KEY:?Required}
      QUANT_PRODUCTION_MODE: "1"
    volumes:
      - dataaccess-cache:/var/lib/dataaccess/cos-cache:rw
      - dataaccess-tmp:/tmp/duckdb:rw
      - /data/quant_workspace:/data/quant_workspace:ro
    tmpfs:
      - /tmp:size=4g,mode=1777
    read_only: true
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8765/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "10"
    deploy:
      resources:
        limits:
          cpus: '12'
          memory: 28G
        reservations:
          cpus: '8'
          memory: 20G

  factor-engine:
    build:
      context: ./factor_engine
      dockerfile: Dockerfile
    image: quant/factor-engine:0.3.1
    container_name: factor-engine-prod
    restart: unless-stopped
    ports:
      - "8766:8766"
    environment:
      FACTOR_ENGINE_SERVICE_PORT: "8766"
      FACTOR_ENGINE_MAX_WORKERS: "4"
      FACTOR_ENGINE_USE_NUMBA: "1"
      DATA_ACCESS_API_URL: "http://data-access:8765"
      DATA_ACCESS_API_KEY: ${DATA_ACCESS_API_KEY}
      QUANT_PRODUCTION_MODE: "1"
    volumes:
      - /data/quant_workspace:/data/quant_workspace:rw
      - factor-tmp:/tmp:rw
    depends_on:
      data-access:
        condition: service_healthy
    read_only: true
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8766/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "10"
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 16G
        reservations:
          cpus: '4'
          memory: 8G

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus-quant
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus:rw
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'

volumes:
  dataaccess-cache:
    driver: local
    driver_opts:
      type: none
      device: /data/quant_workspace/cos_cache
      o: bind
  dataaccess-tmp:
    driver: local
  factor-tmp:
    driver: local
  prometheus-data:
    driver: local
```

### Deploy with Docker Compose

```bash
# Set secrets in environment
export DATA_ACCESS_API_KEY=$(vault kv get -field=api_key secret/quant/prod)
export COS_SECRET_ID=$(vault kv get -field=id secret/quant/cos)
export COS_SECRET_KEY=$(vault kv get -field=key secret/quant/cos)
export COS_S3_ENDPOINT=cos.ap-guangzhou.myqcloud.com

# Deploy
cd /opt/quant_projects
docker-compose -f docker-compose.production.yml up -d

# Check status
docker-compose -f docker-compose.production.yml ps
docker-compose -f docker-compose.production.yml logs -f data-access

# Health check
curl http://localhost:8765/health
curl http://localhost:8766/health
```

### Systemd Units (Non-Docker)

Create `/etc/systemd/system/data-access.service`:

```ini
[Unit]
Description=Quantitative Data Access Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=quantuser
Group=quantuser
WorkingDirectory=/opt/quant_projects

EnvironmentFile=/opt/quant_projects/.env.production
ExecStartPre=/bin/bash -c 'source /opt/quant_projects/venv/bin/activate'
ExecStart=/opt/quant_projects/venv/bin/data-access-server --host 0.0.0.0 --port 8765

Restart=on-failure
RestartSec=10s
StandardOutput=journal
StandardError=journal
SyslogIdentifier=data-access

# Resource limits
LimitNOFILE=65536
LimitNPROC=32768

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/data/quant_workspace /tmp/duckdb
CapabilityBoundingSet=CAP_NET_BIND_SERVICE

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/factor-engine.service`:

```ini
[Unit]
Description=Quantitative Factor Engine Service
After=network-online.target data-access.service
Wants=network-online.target
Requires=data-access.service

[Service]
Type=simple
User=quantuser
Group=quantuser
WorkingDirectory=/opt/quant_projects

EnvironmentFile=/opt/quant_projects/.env.production
ExecStartPre=/bin/bash -c 'source /opt/quant_projects/venv/bin/activate'
ExecStart=/opt/quant_projects/venv/bin/factor-engine-serve --host 0.0.0.0 --port 8766

Restart=on-failure
RestartSec=10s
StandardOutput=journal
StandardError=journal
SyslogIdentifier=factor-engine

# Resource limits
LimitNOFILE=65536
LimitNPROC=32768

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/data/quant_workspace
CapabilityBoundingSet=CAP_NET_BIND_SERVICE

[Install]
WantedBy=multi-user.target
```

Enable and start services:

```bash
sudo systemctl daemon-reload
sudo systemctl enable data-access factor-engine
sudo systemctl start data-access factor-engine

# Check status
sudo systemctl status data-access
sudo systemctl status factor-engine

# View logs
journalctl -u data-access -f
journalctl -u factor-engine -f
```

---

## Support and Escalation

### Internal Documentation

- Main documentation: `docs/量化平台使用总览.md`
- data_access manual: `dataaccess/docs/用户使用手册.md`
- factor_engine guide: `factor_engine/docs/FactorEngine完全指南.md`

### Common Commands

```bash
# Health check
curl http://localhost:8765/health
curl http://localhost:8766/health

# Restart services
sudo systemctl restart data-access factor-engine

# View logs
journalctl -u data-access -n 100 --no-pager
journalctl -u factor-engine -n 100 --no-pager

# Check resource usage
htop
df -h /data/quant_workspace

# Test data access
python3 -c "from data_access import get_store; print(get_store().load_columns('ashare_stock_daily', columns=['Close'], time_range=('2024-01-01', '2024-01-02')))"
```

---

**Last Updated**: 2026-08-14  
**Version**: 1.0  
**Maintained by**: Quantitative Platform Team
