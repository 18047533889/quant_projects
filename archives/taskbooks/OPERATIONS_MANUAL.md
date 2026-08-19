# Factor Engine Operations Manual

**Version:** 1.0  
**Date:** 2026-08-14  
**Audience:** Operations Team, On-Call Engineers  
**Status:** PRODUCTION

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Daily Operations](#3-daily-operations)
4. [Monitoring & Alerting](#4-monitoring--alerting)
5. [Troubleshooting Guide](#5-troubleshooting-guide)
6. [Performance Tuning](#6-performance-tuning)
7. [Backup & Recovery](#7-backup--recovery)
8. [Security Operations](#8-security-operations)
9. [Incident Response](#9-incident-response)
10. [Runbooks](#10-runbooks)

---

## 1. Overview

### 1.1 What is Factor Engine?

Factor Engine is a quantitative factor computation platform that:
- Computes financial factors from market data
- Supports 620+ operators across multiple backends (Pandas, Polars, DuckDB, ClickHouse)
- Materializes factor results to a factor lake
- Provides HTTP API for factor computation and validation

### 1.2 System Components

```
┌─────────────────────────────────────────────────────────┐
│                    Factor Engine                         │
├─────────────────────────────────────────────────────────┤
│  HTTP Service (FastAPI)                                  │
│    ├─ /health                                            │
│    ├─ /factor-engine/operators                           │
│    ├─ /factor-engine/validate-spec                       │
│    ├─ /factor-engine/compute                             │
│    └─ /factor-engine/materialize                         │
├─────────────────────────────────────────────────────────┤
│  Execution Engine                                        │
│    ├─ CSE (Common Subexpression Elimination)             │
│    ├─ AdaptiveBatchScheduler                             │
│    ├─ ResourceBroker                                     │
│    └─ Multi-backend execution                            │
├─────────────────────────────────────────────────────────┤
│  Storage Layer                                           │
│    ├─ Factor Lake (Parquet)                              │
│    ├─ Catalog (SQLite)                                   │
│    ├─ Cache (Persistent)                                 │
│    └─ Data Access (via data_access module)               │
└─────────────────────────────────────────────────────────┘
```

### 1.3 Key Directories

| Path | Purpose | Owner | Permissions |
|------|---------|-------|-------------|
| `/opt/factor_engine/` | Application code | factoreng | 755 |
| `/data/factor_lake/` | Factor storage | factoreng | 750 |
| `/data/factor_lake/catalog.db` | Metadata catalog | factoreng | 640 |
| `/data/factor_lake/cache/` | Persistent cache | factoreng | 750 |
| `/data/backup/factor_engine/` | Backups | factoreng | 700 |
| `/var/log/factor_engine/` | Application logs | factoreng | 750 |
| `/etc/factor_engine/` | Configuration | root | 755 |

### 1.4 Service Information

**Service Name:** `factor-engine-service`  
**Port:** 8088 (HTTP)  
**User:** `factoreng`  
**Python:** 3.10.x  
**Version:** 0.3.1

---

## 2. System Architecture

### 2.1 Process Architecture

```
systemd (PID 1)
  └─ factor-engine-serve (main process)
       ├─ Uvicorn worker threads
       ├─ Job queue workers (8 threads)
       ├─ Resource broker thread
       └─ Telemetry collector thread
```

### 2.2 Data Flow

```
Client Request
    ↓
HTTP API (FastAPI)
    ↓
Factor Engine (compilation)
    ↓
Backend Selection (auto)
    ↓
Data Access (DuckDB/ClickHouse)
    ↓
Computation (Pandas/Polars/SQL)
    ↓
Materialization (Parquet)
    ↓
Catalog Update (SQLite)
    ↓
Response to Client
```

### 2.3 Storage Layout

```
/data/factor_lake/
├── factors/                    # Materialized factors
│   ├── factor_a/
│   │   ├── year=2024/
│   │   │   └── data.parquet
│   │   └── year=2025/
│   │       └── data.parquet
│   └── factor_b/
│       └── year=2024/
│           └── data.parquet
├── cache/                      # Persistent cache
│   ├── columns/
│   └── operators/
├── catalog.db                  # SQLite catalog
├── staging/                    # Temporary staging
└── .checkpoints/              # Generation pointers
```

---

## 3. Daily Operations

### 3.1 Daily Health Check

**Run every morning at 09:00:**

```bash
#!/bin/bash
# /opt/factor_engine/scripts/daily_health_check.sh

DATE=$(date +%Y-%m-%d)
echo "=== Factor Engine Health Check - $DATE ==="

# 1. Service Status
echo -e "\n[1/8] Service Status:"
systemctl status factor-engine-service --no-pager | grep "Active:"
curl -s http://localhost:8088/health || echo "FAIL: Health check failed"

# 2. Disk Space
echo -e "\n[2/8] Disk Space:"
df -h /data/factor_lake | grep -v Filesystem

# 3. Memory Usage
echo -e "\n[3/8] Memory Usage:"
free -h | grep -E "Mem:|Swap:"

# 4. Process Status
echo -e "\n[4/8] Process Status:"
ps aux | grep factor-engine-serve | grep -v grep | awk '{print $2, $3, $4, $6}'

# 5. Recent Errors
echo -e "\n[5/8] Recent Errors (last 24h):"
ERROR_COUNT=$(grep -c "ERROR" /var/log/factor_engine/service.log)
echo "Total errors: $ERROR_COUNT"
if [ $ERROR_COUNT -gt 100 ]; then
    echo "WARNING: High error count!"
    tail -20 /var/log/factor_engine/service.log | grep ERROR
fi

# 6. Catalog Integrity
echo -e "\n[6/8] Catalog Integrity:"
sqlite3 /data/factor_lake/catalog.db "PRAGMA integrity_check;" | head -1

# 7. Backup Status
echo -e "\n[7/8] Backup Status:"
LATEST_BACKUP=$(ls -t /data/backup/factor_engine/catalog_*.db 2>/dev/null | head -1)
if [ -n "$LATEST_BACKUP" ]; then
    echo "Latest backup: $LATEST_BACKUP"
    BACKUP_AGE=$(($(date +%s) - $(stat -c %Y "$LATEST_BACKUP")))
    BACKUP_AGE_HOURS=$((BACKUP_AGE / 3600))
    echo "Backup age: ${BACKUP_AGE_HOURS}h"
    if [ $BACKUP_AGE_HOURS -gt 26 ]; then
        echo "WARNING: Backup older than 26 hours!"
    fi
else
    echo "ERROR: No backups found!"
fi

# 8. Cache Hit Rate (last 24h)
echo -e "\n[8/8] Cache Statistics:"
# Parse logs for cache hits/misses if available
# This is placeholder - implement based on actual logging
echo "Cache metrics: See monitoring dashboard"

echo -e "\n=== Health Check Complete ==="
```

**Save output:**
```bash
/opt/factor_engine/scripts/daily_health_check.sh > \
  /var/log/factor_engine/health_$(date +%Y%m%d).log
```

### 3.2 Weekly Tasks

**Every Monday at 10:00:**

```bash
#!/bin/bash
# /opt/factor_engine/scripts/weekly_tasks.sh

DATE=$(date +%Y-%m-%d)
echo "=== Factor Engine Weekly Tasks - $DATE ==="

# 1. Run quality gates
echo -e "\n[1/4] Running Quality Gates..."
cd /opt/factor_engine
python scripts/audit_r32_hard_gates.py > /tmp/r32_weekly.json
GATE_STATUS=$(jq -r '.R32_HARD_BLOCKERS_ZERO' /tmp/r32_weekly.json)
if [ "$GATE_STATUS" != "true" ]; then
    echo "ERROR: Quality gates failing!"
    jq '.' /tmp/r32_weekly.json
fi

# 2. Catalog vacuum
echo -e "\n[2/4] Vacuuming Catalog..."
sqlite3 /data/factor_lake/catalog.db "VACUUM;"
sqlite3 /data/factor_lake/catalog.db "ANALYZE;"

# 3. Clean old logs (keep 30 days)
echo -e "\n[3/4] Cleaning Old Logs..."
find /var/log/factor_engine/ -name "*.log" -mtime +30 -delete
echo "Old logs cleaned"

# 4. Cache cleanup (remove unused entries)
echo -e "\n[4/4] Cache Cleanup..."
find /data/factor_lake/cache/ -type f -atime +90 -delete
echo "Old cache entries cleaned"

echo -e "\n=== Weekly Tasks Complete ==="
```

### 3.3 Monthly Tasks

**First day of month at 02:00:**

```bash
#!/bin/bash
# /opt/factor_engine/scripts/monthly_tasks.sh

DATE=$(date +%Y-%m-%d)
echo "=== Factor Engine Monthly Tasks - $DATE ==="

# 1. Full system audit
echo -e "\n[1/5] Full System Audit..."
cd /opt/factor_engine
pytest tests/ -v --tb=short > /tmp/monthly_test_$(date +%Y%m).log 2>&1
TESTS_PASSED=$(grep -c "passed" /tmp/monthly_test_$(date +%Y%m).log || echo 0)
echo "Tests passed: $TESTS_PASSED"

# 2. Performance benchmarks
echo -e "\n[2/5] Running Performance Benchmarks..."
python scripts/benchmark_ttdc.py > /tmp/benchmark_$(date +%Y%m).log

# 3. Catalog statistics
echo -e "\n[3/5] Catalog Statistics..."
sqlite3 /data/factor_lake/catalog.db <<EOF
SELECT 'Total factors:', COUNT(*) FROM factor_metadata;
SELECT 'Total checkpoints:', COUNT(*) FROM factor_materialize_checkpoint;
SELECT 'Disk usage (MB):', SUM(page_count * page_size) / 1048576 FROM dbstat;
EOF

# 4. Generate monthly report
echo -e "\n[4/5] Generating Monthly Report..."
python scripts/generate_monthly_report.py \
    --month $(date +%Y-%m) \
    --output /var/log/factor_engine/report_$(date +%Y%m).html

# 5. Archive old backups (keep 90 days)
echo -e "\n[5/5] Archiving Old Backups..."
find /data/backup/factor_engine/ -name "catalog_*.db" -mtime +90 \
    -exec gzip {} \; -exec mv {}.gz /data/archive/ \;

echo -e "\n=== Monthly Tasks Complete ==="
```

---

## 4. Monitoring & Alerting

### 4.1 Key Metrics

#### Service Health Metrics

| Metric | Normal Range | Warning | Critical |
|--------|--------------|---------|----------|
| **Uptime** | 99.9%+ | < 99.5% | < 99% |
| **Request Rate** | 10-100 req/s | > 200 req/s | > 500 req/s |
| **Error Rate** | < 0.1% | 0.1-1% | > 1% |
| **P50 Latency** | < 200ms | 200-500ms | > 500ms |
| **P99 Latency** | < 2s | 2-5s | > 5s |
| **P999 Latency** | < 10s | 10-30s | > 30s |

#### Resource Metrics

| Metric | Normal Range | Warning | Critical |
|--------|--------------|---------|----------|
| **CPU Usage** | 20-60% | 60-80% | > 80% |
| **Memory Usage** | 40-70% | 70-85% | > 85% |
| **Disk Usage** | < 70% | 70-85% | > 85% |
| **Disk I/O Wait** | < 5% | 5-15% | > 15% |
| **Network Traffic** | < 100 MB/s | 100-500 MB/s | > 500 MB/s |

#### Business Metrics

| Metric | Normal Range | Warning | Critical |
|--------|--------------|---------|----------|
| **Factors Computed/Day** | 1000+ | 500-1000 | < 500 |
| **Cache Hit Rate** | > 70% | 50-70% | < 50% |
| **Materialization Success Rate** | > 99% | 95-99% | < 95% |
| **Data Freshness (lag)** | < 1 hour | 1-6 hours | > 6 hours |

### 4.2 Alert Definitions

#### Critical Alerts (Page Immediately)

**Service Down:**
```yaml
alert: FactorEngineDown
expr: up{job="factor-engine"} == 0
for: 2m
annotations:
  summary: "Factor Engine service is down"
  description: "Service has been unreachable for 2+ minutes"
  runbook: "RUNBOOK-001: Service Down"
```

**High Error Rate:**
```yaml
alert: FactorEngineHighErrorRate
expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
for: 5m
annotations:
  summary: "High error rate (>5%)"
  description: "Error rate has exceeded 5% for 5+ minutes"
  runbook: "RUNBOOK-002: High Error Rate"
```

**Data Corruption:**
```yaml
alert: FactorEngineDataCorruption
expr: factor_catalog_integrity_check == 0
for: 1m
annotations:
  summary: "Catalog integrity check failed"
  description: "Database corruption detected"
  runbook: "RUNBOOK-003: Data Corruption"
```

**Disk Space Critical:**
```yaml
alert: FactorEngineDiskFull
expr: disk_free_percent{mount="/data"} < 10
for: 5m
annotations:
  summary: "Disk space critically low (<10%)"
  description: "Factor lake partition running out of space"
  runbook: "RUNBOOK-004: Disk Space"
```

#### Warning Alerts (Investigate Within 1 Hour)

**High Latency:**
```yaml
alert: FactorEngineHighLatency
expr: histogram_quantile(0.99, http_request_duration_seconds) > 5
for: 10m
annotations:
  summary: "P99 latency >5s"
  description: "Performance degradation detected"
  runbook: "RUNBOOK-005: Performance Degradation"
```

**Memory Pressure:**
```yaml
alert: FactorEngineMemoryPressure
expr: memory_usage_percent > 85
for: 10m
annotations:
  summary: "Memory usage >85%"
  description: "System approaching memory limits"
  runbook: "RUNBOOK-006: Memory Pressure"
```

**Backup Failure:**
```yaml
alert: FactorEngineBackupFailed
expr: time() - backup_last_success_timestamp > 86400*2
for: 5m
annotations:
  summary: "Backup hasn't succeeded in 48+ hours"
  description: "Backup system may be failing"
  runbook: "RUNBOOK-007: Backup Failure"
```

### 4.3 Monitoring Commands

**Check service status:**
```bash
systemctl status factor-engine-service
curl http://localhost:8088/health
```

**View real-time metrics:**
```bash
# Request rate
watch -n 1 'tail -100 /var/log/factor_engine/service.log | grep -c "INFO"'

# Error rate
watch -n 5 'tail -1000 /var/log/factor_engine/service.log | grep -c "ERROR"'

# Memory usage
watch -n 2 'free -h'

# Disk I/O
watch -n 1 'iostat -x 1 2 | tail -n +4'
```

**Monitor logs in real-time:**
```bash
# All logs
tail -f /var/log/factor_engine/service.log

# Errors only
tail -f /var/log/factor_engine/service.log | grep ERROR

# Specific factor
tail -f /var/log/factor_engine/service.log | grep "factor_id=my_factor"
```

---

## 5. Troubleshooting Guide

### 5.1 Service Won't Start

**Symptoms:**
- `systemctl start factor-engine-service` fails
- Health check returns connection refused
- No process listening on port 8088

**Diagnosis:**
```bash
# Check service status
systemctl status factor-engine-service

# Check for port conflicts
sudo lsof -i :8088

# Check logs
tail -100 /var/log/factor_engine/service.log

# Check Python environment
/opt/factor_engine/venv/bin/python -c "import factor_engine; print(factor_engine.__version__)"
```

**Common Causes & Fixes:**

**1. Port already in use:**
```bash
# Find process using port
sudo lsof -i :8088
# Kill old process
sudo kill <PID>
# Or change port in config
```

**2. Missing dependencies:**
```bash
cd /opt/factor_engine
pip install -e ".[service,full,performance]"
```

**3. Permission issues:**
```bash
# Fix ownership
sudo chown -R factoreng:factoreng /opt/factor_engine
sudo chown -R factoreng:factoreng /data/factor_lake
sudo chown -R factoreng:factoreng /var/log/factor_engine

# Fix permissions
sudo chmod 750 /data/factor_lake
sudo chmod 640 /data/factor_lake/catalog.db
```

**4. Corrupt catalog:**
```bash
# Restore from backup
cp /data/backup/factor_engine/catalog_$(date -d yesterday +%Y%m%d).db \
   /data/factor_lake/catalog.db

# Verify integrity
sqlite3 /data/factor_lake/catalog.db "PRAGMA integrity_check;"
```

### 5.2 High Error Rate

**Symptoms:**
- 5xx responses in logs
- Error rate metric spiking
- Failed factor computations

**Diagnosis:**
```bash
# Count errors by type
grep ERROR /var/log/factor_engine/service.log | \
    awk -F': ' '{print $2}' | sort | uniq -c | sort -rn

# Find most common error
tail -1000 /var/log/factor_engine/service.log | \
    grep ERROR | tail -20

# Check for specific exceptions
grep -A 5 "Traceback" /var/log/factor_engine/service.log | tail -50
```

**Common Causes & Fixes:**

**1. Database connection errors:**
```bash
# Check database accessibility
sqlite3 /data/factor_lake/catalog.db "SELECT 1;"

# Check file locks
lsof /data/factor_lake/catalog.db

# If locked, kill blocking process or restart service
```

**2. Data source unavailable:**
```bash
# Test DuckDB connection
python -c "
import duckdb
conn = duckdb.connect()
conn.execute('SELECT 1')
print('DuckDB OK')
"

# Check data access module
python -c "
from data_access import scan_polars
# Test with known dataset
"
```

**3. Out of memory errors:**
```bash
# Check memory usage
free -h

# Check for memory leaks
ps aux | grep factor-engine-serve | awk '{print $2, $6}'

# If OOM, restart service
systemctl restart factor-engine-service
```

**4. Operator errors:**
```bash
# Find failing operators
grep "operator.*failed" /var/log/factor_engine/service.log | \
    awk '{print $5}' | sort | uniq -c | sort -rn

# Run operator audit
cd /opt/factor_engine
python scripts/audit_all_registered_operators.py
```

### 5.3 Performance Degradation

**Symptoms:**
- Slow response times
- High P99 latency
- Requests timing out
- CPU or I/O saturation

**Diagnosis:**
```bash
# Check system resources
top -bn 1 | head -20

# Check I/O wait
iostat -x 5 3

# Check database locks
sqlite3 /data/factor_lake/catalog.db \
    "SELECT * FROM pragma_database_list;"

# Profile current requests
# (requires instrumentation)
cd /opt/factor_engine
python scripts/profile_live_requests.py
```

**Common Causes & Fixes:**

**1. Cache cold start:**
```bash
# Check cache hit rate
# (from monitoring dashboard or logs)

# Warm cache with common factors
python scripts/warm_cache.py --factors common_factors.txt
```

**2. Large factor computation:**
```bash
# Check current computations
# (from job queue)
python -c "
from service.jobstore import JobStore
store = JobStore('/data/factor_lake/.jobstore')
running = store.list_jobs(status='RUNNING')
for job in running:
    print(f'{job.run_id}: {job.factor_id}')
"

# If stuck, consider killing long-running job
```

**3. Database contention:**
```bash
# Check for long-running queries
# (SQLite doesn't have pg_stat_activity, so check locks)

# Optimize catalog
sqlite3 /data/factor_lake/catalog.db "VACUUM; ANALYZE;"

# If severe, consider restarting service during low-traffic window
```

**4. Resource exhaustion:**
```bash
# If CPU saturated, reduce worker threads
# Edit /etc/factor_engine/production.env
echo "FACTOR_ENGINE_MAX_WORKERS=4" >> /etc/factor_engine/production.env
systemctl restart factor-engine-service

# If memory saturated, adjust cache size
echo "FACTOR_ENGINE_CACHE_SIZE_GB=16" >> /etc/factor_engine/production.env
systemctl restart factor-engine-service
```

### 5.4 Disk Space Issues

**Symptoms:**
- Disk usage >85%
- Write errors in logs
- Materialization failures

**Diagnosis:**
```bash
# Check disk usage
df -h /data/factor_lake

# Find largest directories
du -h /data/factor_lake/ | sort -rh | head -20

# Find old/unused factors
find /data/factor_lake/factors/ -type f -atime +90 -exec ls -lh {} \;
```

**Immediate Actions:**

**1. Clean staging directory:**
```bash
# Remove temp files
rm -rf /data/factor_lake/staging/*

# Remove spill store
rm -rf /data/factor_lake/.spillstore/*
```

**2. Clean old cache:**
```bash
# Remove cache entries not accessed in 90 days
find /data/factor_lake/cache/ -type f -atime +90 -delete
```

**3. Archive old factors:**
```bash
# Move factors older than 1 year to archive
find /data/factor_lake/factors/ -name "year=2023" -type d \
    -exec tar -czf /data/archive/{}.tar.gz {} \; \
    -exec rm -rf {} \;
```

**4. Vacuum database:**
```bash
sqlite3 /data/factor_lake/catalog.db "VACUUM;"
```

**Long-term Solutions:**
- Add more disk space
- Implement retention policies
- Archive to object storage (S3/COS)

### 5.5 Data Corruption

**Symptoms:**
- Catalog integrity check fails
- Factor values incorrect
- NaN/Inf mask mismatches
- SQLite corruption errors

**Diagnosis:**
```bash
# Check catalog integrity
sqlite3 /data/factor_lake/catalog.db "PRAGMA integrity_check;"

# Check for foreign key violations
sqlite3 /data/factor_lake/catalog.db "PRAGMA foreign_key_check;"

# Run quality gates
cd /opt/factor_engine
python scripts/audit_r32_hard_gates.py
```

**Recovery Procedure:**

**CRITICAL: Stop writes immediately**
```bash
# Stop service
systemctl stop factor-engine-service
```

**1. Assess damage:**
```bash
# Check catalog
sqlite3 /data/factor_lake/catalog.db "PRAGMA integrity_check;" > /tmp/integrity.log
cat /tmp/integrity.log

# Check parquet files
python scripts/verify_parquet_integrity.py --all > /tmp/parquet_check.log
```

**2. Restore from backup:**
```bash
# Identify last known good backup
ls -lt /data/backup/factor_engine/catalog_*.db

# Restore catalog
GOOD_BACKUP=/data/backup/factor_engine/catalog_20260813.db
cp $GOOD_BACKUP /data/factor_lake/catalog.db

# Verify restored catalog
sqlite3 /data/factor_lake/catalog.db "PRAGMA integrity_check;"
```

**3. Validate restored state:**
```bash
# Run quality gates
cd /opt/factor_engine
python scripts/audit_r32_hard_gates.py

# Run integration tests
pytest tests/integration/ -v
```

**4. Restart service:**
```bash
systemctl start factor-engine-service

# Verify health
sleep 5
curl http://localhost:8088/health
```

**5. Post-incident:**
- Document root cause
- Review backup procedures
- Consider increasing backup frequency
- Add additional integrity checks

---

## 6. Performance Tuning

### 6.1 Configuration Tuning

**Memory Optimization:**
```bash
# /etc/factor_engine/production.env

# Cache size (adjust based on available RAM)
FACTOR_ENGINE_CACHE_SIZE_GB=32

# Worker threads (2x CPU cores typical)
FACTOR_ENGINE_MAX_WORKERS=16

# DuckDB memory limit
DUCKDB_MEMORY_LIMIT=16GB
```

**Execution Optimization:**
```bash
# Enable Polars fast path
FACTOR_ENGINE_POLARS_EXPR=1

# Enable Numba acceleration
FACTOR_BACKTEST_EXECUTION_ENGINE=numba

# CSE optimization
FACTOR_ENGINE_ENABLE_CSE=1
```

**I/O Optimization:**
```bash
# DuckDB threads
DUCKDB_THREADS=8

# Parquet compression
FACTOR_ENGINE_PARQUET_COMPRESSION=snappy

# Staging directory on fast SSD
FACTOR_ENGINE_STAGING_ROOT=/nvme/factor_staging
```

### 6.2 Database Optimization

**Regular Maintenance:**
```bash
# Weekly: Vacuum and analyze
sqlite3 /data/factor_lake/catalog.db <<EOF
VACUUM;
ANALYZE;
PRAGMA optimize;
EOF

# Monthly: Full rebuild
sqlite3 /data/factor_lake/catalog.db <<EOF
VACUUM INTO '/tmp/catalog_rebuilt.db';
EOF
# Then replace original after verification
```

**Index Optimization:**
```sql
-- Check existing indexes
SELECT name, tbl_name FROM sqlite_master WHERE type='index';

-- Add indexes for common queries (if not exist)
CREATE INDEX IF NOT EXISTS idx_factor_metadata_factor_id 
    ON factor_metadata(factor_id);

CREATE INDEX IF NOT EXISTS idx_checkpoint_factor_partition 
    ON factor_materialize_checkpoint(factor_id, partition_key);
```

### 6.3 Cache Tuning

**Monitor Cache Effectiveness:**
```bash
# Cache hit rate from logs
grep "cache_hit" /var/log/factor_engine/service.log | \
    awk '{sum+=$NF; count++} END {print "Avg hit rate:", sum/count}'
```

**Adjust Cache Size:**
```python
# Calculate optimal cache size
import psutil
total_memory = psutil.virtual_memory().total / (1024**3)  # GB
# Use 30-40% of total memory for cache
optimal_cache_size = int(total_memory * 0.35)
print(f"Recommended cache size: {optimal_cache_size}GB")
```

**Cache Warming Strategy:**
```bash
# Warm cache with frequently used factors
cat > /tmp/warm_factors.txt <<EOF
momentum_rank_20d
volatility_std_60d
volume_rank_10d
EOF

python scripts/warm_cache.py --factors /tmp/warm_factors.txt
```

### 6.4 Monitoring Performance

**Baseline Metrics:**
```bash
# Capture baseline
cd /opt/factor_engine
python scripts/benchmark_ttdc.py > /tmp/baseline_$(date +%Y%m%d).log

# Compare after tuning
python scripts/benchmark_ttdc.py > /tmp/tuned_$(date +%Y%m%d).log

# Calculate improvement
python scripts/compare_benchmarks.py \
    /tmp/baseline_*.log \
    /tmp/tuned_*.log
```

**Continuous Profiling:**
```bash
# Profile hot paths
python -m cProfile -o /tmp/profile.stats \
    -m factor_engine.main compute --config test.yaml

# Analyze results
python -m pstats /tmp/profile.stats
# Then: sort cumulative; stats 20
```

---

## 7. Backup & Recovery

### 7.1 Backup Strategy

**Automated Daily Backups:**
```bash
#!/bin/bash
# /opt/factor_engine/scripts/daily_backup.sh
# Run via cron: 0 3 * * * /opt/factor_engine/scripts/daily_backup.sh

DATE=$(date +%Y%m%d)
BACKUP_ROOT=/data/backup/factor_engine
CATALOG_PATH=/data/factor_lake/catalog.db

# 1. Verify catalog integrity
echo "Checking catalog integrity..."
INTEGRITY=$(sqlite3 $CATALOG_PATH "PRAGMA integrity_check;")
if [ "$INTEGRITY" != "ok" ]; then
    echo "ERROR: Catalog integrity check failed: $INTEGRITY" >&2
    exit 1
fi

# 2. Backup catalog
echo "Backing up catalog..."
sqlite3 $CATALOG_PATH "VACUUM INTO '$BACKUP_ROOT/catalog_$DATE.db';"

# 3. Verify backup
echo "Verifying backup..."
BACKUP_INTEGRITY=$(sqlite3 $BACKUP_ROOT/catalog_$DATE.db "PRAGMA integrity_check;")
if [ "$BACKUP_INTEGRITY" != "ok" ]; then
    echo "ERROR: Backup verification failed" >&2
    exit 1
fi

# 4. Incremental factor lake backup
echo "Backing up factor lake..."
rsync -av --delete \
    /data/factor_lake/factors/ \
    $BACKUP_ROOT/factors_$DATE/

# 5. Backup generation pointers
echo "Backing up generation pointers..."
cp -r /data/factor_lake/.checkpoints $BACKUP_ROOT/checkpoints_$DATE/

# 6. Clean old backups (keep 30 days)
echo "Cleaning old backups..."
find $BACKUP_ROOT -name "catalog_*.db" -mtime +30 -delete
find $BACKUP_ROOT -name "factors_*" -type d -mtime +30 -exec rm -rf {} +

echo "Backup complete: $DATE"
```

**Setup Cron:**
```bash
# Add to crontab
sudo crontab -e -u factoreng

# Add line:
0 3 * * * /opt/factor_engine/scripts/daily_backup.sh >> /var/log/factor_engine/backup.log 2>&1
```

### 7.2 Recovery Procedures

**Scenario 1: Catalog Corruption**
```bash
# 1. Stop service
systemctl stop factor-engine-service

# 2. Move corrupt catalog
mv /data/factor_lake/catalog.db /data/factor_lake/catalog.db.corrupt

# 3. Restore latest backup
LATEST=$(ls -t /data/backup/factor_engine/catalog_*.db | head -1)
cp $LATEST /data/factor_lake/catalog.db

# 4. Verify integrity
sqlite3 /data/factor_lake/catalog.db "PRAGMA integrity_check;"

# 5. Restart service
systemctl start factor-engine-service

# 6. Verify health
sleep 5
curl http://localhost:8088/health
```

**Scenario 2: Factor Data Loss**
```bash
# 1. Identify affected factor
FACTOR_ID="momentum_rank_20d"

# 2. Find backup
BACKUP_DATE=20260813  # Adjust to last known good
BACKUP_PATH=/data/backup/factor_engine/factors_$BACKUP_DATE/$FACTOR_ID

# 3. Restore factor data
cp -r $BACKUP_PATH /data/factor_lake/factors/$FACTOR_ID

# 4. Update catalog checkpoint
python scripts/restore_factor_checkpoint.py --factor $FACTOR_ID --date $BACKUP_DATE

# 5. Verify factor
python scripts/verify_factor_integrity.py --factor $FACTOR_ID
```

**Scenario 3: Complete System Loss**
```bash
# Disaster recovery from backups

# 1. Provision new server
# 2. Install Factor Engine
# 3. Restore catalog
cp /backup/catalog_latest.db /data/factor_lake/catalog.db

# 4. Restore factor lake
rsync -av /backup/factors_latest/ /data/factor_lake/factors/

# 5. Restore checkpoints
cp -r /backup/checkpoints_latest/ /data/factor_lake/.checkpoints/

# 6. Run integrity checks
cd /opt/factor_engine
python scripts/audit_r32_hard_gates.py

# 7. Start service
systemctl start factor-engine-service
```

### 7.3 Disaster Recovery Testing

**Quarterly DR Test:**
```bash
#!/bin/bash
# Test disaster recovery procedure

echo "=== DR Test $(date) ==="

# 1. Create test environment
mkdir -p /tmp/dr_test
export FACTOR_ENGINE_LAKE_ROOT=/tmp/dr_test

# 2. Restore from production backup
LATEST_BACKUP=$(ls -t /data/backup/factor_engine/catalog_*.db | head -1)
cp $LATEST_BACKUP /tmp/dr_test/catalog.db

# 3. Verify catalog integrity
sqlite3 /tmp/dr_test/catalog.db "PRAGMA integrity_check;"

# 4. Start service in test mode
cd /opt/factor_engine
FACTOR_ENGINE_LAKE_ROOT=/tmp/dr_test \
  factor-engine-serve --port 9088 &
TEST_PID=$!

# 5. Run smoke tests
sleep 5
curl http://localhost:9088/health

# 6. Cleanup
kill $TEST_PID
rm -rf /tmp/dr_test

echo "=== DR Test Complete ==="
```

---

## 8. Security Operations

### 8.1 Access Control

**Service Account:**
```bash
# Factor Engine runs as 'factoreng' user
id factoreng
# uid=1001(factoreng) gid=1001(factoreng) groups=1001(factoreng)

# Key files ownership
ls -l /data/factor_lake/catalog.db
# -rw-r----- 1 factoreng factoreng ... catalog.db
```

**File Permissions:**
```bash
# Set correct permissions
sudo chown -R factoreng:factoreng /opt/factor_engine
sudo chown -R factoreng:factoreng /data/factor_lake
sudo chmod 750 /data/factor_lake
sudo chmod 640 /data/factor_lake/catalog.db
sudo chmod 700 /data/backup/factor_engine
```

### 8.2 Credential Management

**Environment Variables:**
```bash
# Credentials stored in secure environment file
# /etc/factor_engine/production.env

# Never commit credentials to git
# Never log credentials

# Rotate credentials quarterly
```

**Database Credentials:**
```bash
# If using external databases (ClickHouse, etc.)
# Store credentials in /etc/factor_engine/credentials.yaml

# Restrict access
sudo chmod 600 /etc/factor_engine/credentials.yaml
sudo chown root:factoreng /etc/factor_engine/credentials.yaml
```

### 8.3 Security Auditing

**Daily Security Check:**
```bash
#!/bin/bash
# /opt/factor_engine/scripts/security_audit.sh

# 1. Check for unauthorized access
grep "401\|403" /var/log/factor_engine/service.log | tail -20

# 2. Check for path traversal attempts
grep "\.\." /var/log/factor_engine/service.log | tail -20

# 3. Check file permissions
find /data/factor_lake -type f -perm /o+w -ls

# 4. Check for credential leaks in logs
grep -i "password\|secret\|token" /var/log/factor_engine/service.log | \
    grep -v "password=\*\*\*"

# 5. Check catalog access
sqlite3 /data/factor_lake/catalog.db \
    "SELECT COUNT(*) FROM factor_metadata;" > /dev/null
if [ $? -ne 0 ]; then
    echo "WARNING: Catalog access issue"
fi
```

### 8.4 Incident Response

**Security Incident Checklist:**

1. **Detect & Contain**
   - [ ] Identify affected systems
   - [ ] Isolate affected service
   - [ ] Stop writes to database

2. **Assess**
   - [ ] Determine scope of breach
   - [ ] Identify compromised data
   - [ ] Review access logs

3. **Remediate**
   - [ ] Patch vulnerability
   - [ ] Rotate credentials
   - [ ] Update firewall rules

4. **Recover**
   - [ ] Restore from clean backup
   - [ ] Verify integrity
   - [ ] Resume operations

5. **Post-Incident**
   - [ ] Document incident
   - [ ] Update security procedures
   - [ ] Conduct post-mortem

---

## 9. Incident Response

### 9.1 Incident Classification

**P0 - Critical (Page Immediately)**
- Service completely down
- Data corruption
- Security breach
- Data loss

**P1 - High (Respond Within 30min)**
- Severe performance degradation
- High error rate (>5%)
- Backup failure
- Resource exhaustion

**P2 - Medium (Respond Within 2h)**
- Moderate performance issues
- Elevated error rate (1-5%)
- Single component failure

**P3 - Low (Respond Next Business Day)**
- Minor issues
- Feature requests
- Documentation updates

### 9.2 Incident Response Process

```
┌─────────────────┐
│ Incident Raised │ (Alert/Report)
└────────┬────────┘
         ↓
┌─────────────────┐
│   Acknowledge   │ (On-call engineer)
└────────┬────────┘
         ↓
┌─────────────────┐
│    Assess       │ (Severity, Impact)
└────────┬────────┘
         ↓
┌─────────────────┐
│   Mitigate      │ (Stop bleeding)
└────────┬────────┘
         ↓
┌─────────────────┐
│    Resolve      │ (Root cause fix)
└────────┬────────┘
         ↓
┌─────────────────┐
│     Verify      │ (Confirm resolution)
└────────┬────────┘
         ↓
┌─────────────────┐
│  Post-Mortem    │ (Learn & Improve)
└─────────────────┘
```

### 9.3 Escalation Path

**L1 - On-Call Engineer (0-15 min)**
- Initial response
- Run standard diagnostics
- Apply known fixes
- Escalate if unresolved

**L2 - Engineering Lead (15-30 min)**
- Complex troubleshooting
- Code-level investigation
- Architecture decisions
- Escalate if critical

**L3 - Engineering Director (30+ min)**
- Critical incident coordination
- External communication
- Major architecture changes

---

## 10. Runbooks

### RUNBOOK-001: Service Down

**Symptoms:** Health check fails, service unreachable

**Response Time:** Immediate (P0)

**Steps:**
```bash
# 1. Verify service status
systemctl status factor-engine-service

# 2. Check if process crashed
ps aux | grep factor-engine-serve

# 3. Check recent logs
tail -100 /var/log/factor_engine/service.log

# 4. Attempt restart
systemctl restart factor-engine-service

# 5. Wait and verify
sleep 10
curl http://localhost:8088/health

# 6. If still down, check for:
# - Port conflicts: lsof -i :8088
# - Resource exhaustion: free -h, df -h
# - Corrupt catalog: sqlite3 catalog.db "PRAGMA integrity_check;"

# 7. If unresolvable, rollback to previous version
/opt/factor_engine/scripts/emergency_rollback.sh
```

**Escalate if:** Service doesn't start after 2 restart attempts

---

### RUNBOOK-002: High Error Rate

**Symptoms:** Error rate >1%, 5xx responses

**Response Time:** 5 minutes (P0)

**Steps:**
```bash
# 1. Check error distribution
grep ERROR /var/log/factor_engine/service.log | \
    tail -100 | awk '{print $5}' | sort | uniq -c | sort -rn

# 2. Identify most common error
tail -100 /var/log/factor_engine/service.log | grep ERROR | tail -1

# 3. Check system resources
top -bn1 | head -20
free -h
df -h

# 4. Check database
sqlite3 /data/factor_lake/catalog.db "PRAGMA integrity_check;"

# 5. If OOM: Restart service
systemctl restart factor-engine-service

# 6. If database locked: Identify and kill blocking process
lsof /data/factor_lake/catalog.db

# 7. Monitor error rate for 5 minutes
watch -n 10 'tail -100 /var/log/factor_engine/service.log | grep -c ERROR'

# 8. If not improving: Enable debug logging
# Add to env: LOG_LEVEL=DEBUG
# Restart service
```

**Escalate if:** Error rate doesn't drop below 1% within 15 minutes

---

### RUNBOOK-003: Data Corruption

**Symptoms:** Integrity check fails, incorrect factor values

**Response Time:** Immediate (P0)

**Steps:**
```bash
# CRITICAL: Stop all writes immediately
systemctl stop factor-engine-service

# 1. Assess corruption
sqlite3 /data/factor_lake/catalog.db "PRAGMA integrity_check;"

# 2. Identify last known good backup
ls -lt /data/backup/factor_engine/catalog_*.db | head -5

# 3. Restore from backup
GOOD_BACKUP=/data/backup/factor_engine/catalog_20260813.db
cp $GOOD_BACKUP /data/factor_lake/catalog.db.restored

# 4. Verify restored catalog
sqlite3 /data/factor_lake/catalog.db.restored "PRAGMA integrity_check;"

# 5. If OK, replace
mv /data/factor_lake/catalog.db /data/factor_lake/catalog.db.corrupt
mv /data/factor_lake/catalog.db.restored /data/factor_lake/catalog.db

# 6. Run quality gates
cd /opt/factor_engine
python scripts/audit_r32_hard_gates.py

# 7. Restart service
systemctl start factor-engine-service

# 8. Verify health
sleep 5
curl http://localhost:8088/health

# 9. Document incident
```

**Escalate immediately:** This is always P0, engage Engineering Lead

---

### RUNBOOK-004: Disk Space Critical

**Symptoms:** Disk usage >90%, write errors

**Response Time:** 15 minutes (P1)

**Steps:**
```bash
# 1. Check disk usage
df -h /data/factor_lake

# 2. Find largest consumers
du -h /data/factor_lake/ | sort -rh | head -20

# 3. Immediate cleanup:

# Clear staging
rm -rf /data/factor_lake/staging/*

# Clear spill store
rm -rf /data/factor_lake/.spillstore/*

# Remove old cache (>90 days)
find /data/factor_lake/cache/ -type f -atime +90 -delete

# 4. Check disk again
df -h /data/factor_lake

# 5. If still critical (>85%), archive old factors
# Move year=2023 data to archive
find /data/factor_lake/factors/ -name "year=2023" -type d \
    -exec tar -czf /data/archive/{}.tar.gz {} \; \
    -exec rm -rf {} \;

# 6. Vacuum database
sqlite3 /data/factor_lake/catalog.db "VACUUM;"

# 7. If still critical, consider:
# - Expanding disk
# - Moving data to object storage
# - Implementing retention policy
```

**Escalate if:** Cannot free 20% disk space

---

### RUNBOOK-005: Performance Degradation

**Symptoms:** P99 latency >5s, slow responses

**Response Time:** 1 hour (P1)

**Steps:**
```bash
# 1. Check system resources
top -bn1 | head -20
iostat -x 5 2

# 2. Identify bottleneck
# CPU bound: top shows high %CPU
# I/O bound: iostat shows high %iowait
# Memory: free -h shows low available

# 3. Check cache hit rate
# (from monitoring or logs)

# 4. For CPU saturation:
# Reduce workers
echo "FACTOR_ENGINE_MAX_WORKERS=4" >> /etc/factor_engine/production.env
systemctl restart factor-engine-service

# 5. For I/O saturation:
# Check for slow queries
# Optimize database
sqlite3 /data/factor_lake/catalog.db "ANALYZE;"

# 6. For memory pressure:
# Reduce cache size
echo "FACTOR_ENGINE_CACHE_SIZE_GB=16" >> /etc/factor_engine/production.env
systemctl restart factor-engine-service

# 7. For cold cache:
# Warm cache
python scripts/warm_cache.py --factors common_factors.txt

# 8. Monitor improvement
watch -n 5 'tail -100 /var/log/factor_engine/service.log | grep latency'
```

**Escalate if:** Performance doesn't improve within 1 hour

---

## Appendices

### A. Quick Command Reference

**Service Control:**
```bash
systemctl start factor-engine-service
systemctl stop factor-engine-service
systemctl restart factor-engine-service
systemctl status factor-engine-service
```

**Health Checks:**
```bash
curl http://localhost:8088/health
python scripts/daily_health_check.sh
```

**Logs:**
```bash
tail -f /var/log/factor_engine/service.log
journalctl -u factor-engine-service -f
```

**Backup/Restore:**
```bash
/opt/factor_engine/scripts/daily_backup.sh
/opt/factor_engine/scripts/emergency_rollback.sh
```

### B. Contact Information

**On-Call:** See PagerDuty schedule  
**Slack:** #factor-engine-oncall  
**Escalation:** Engineering Lead → Director

---

**Document Control:**  
Created: 2026-08-14  
Author: Claude (Kiro)  
Version: 1.0  
Status: PRODUCTION  
Next Review: 2026-11-14
