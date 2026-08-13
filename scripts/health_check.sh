#!/usr/bin/env bash
# Health check script for production monitoring
# Usage: bash scripts/health_check.sh

set -euo pipefail

EXIT_CODE=0

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }
error() { echo "[ERROR] $*" >&2; EXIT_CODE=1; }

log "Starting health check..."

# 1. Check services are running
log "Checking Docker containers..."
if ! docker ps | grep -q data-access-prod; then
    error "data-access container not running"
fi

if ! docker ps | grep -q factor-engine-prod; then
    error "factor-engine container not running"
fi

# 2. Check HTTP endpoints
log "Checking HTTP endpoints..."
if ! curl -sf -m 10 http://localhost:8765/health > /dev/null; then
    error "data-access health endpoint failed"
else
    log "data-access health: OK"
fi

if ! curl -sf -m 10 http://localhost:8766/health > /dev/null; then
    error "factor-engine health endpoint failed"
else
    log "factor-engine health: OK"
fi

# 3. Check disk space
log "Checking disk space..."
disk_usage=$(df /data | tail -1 | awk '{print $5}' | sed 's/%//')
if [[ $disk_usage -gt 90 ]]; then
    error "Disk usage critical: ${disk_usage}%"
elif [[ $disk_usage -gt 80 ]]; then
    log "WARNING: Disk usage high: ${disk_usage}%"
else
    log "Disk usage: ${disk_usage}% (OK)"
fi

# 4. Check memory usage
log "Checking memory usage..."
mem_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100}')
if [[ $mem_usage -gt 95 ]]; then
    error "Memory usage critical: ${mem_usage}%"
elif [[ $mem_usage -gt 85 ]]; then
    log "WARNING: Memory usage high: ${mem_usage}%"
else
    log "Memory usage: ${mem_usage}% (OK)"
fi

# 5. Check log errors
log "Checking recent errors in logs..."
error_count=$(docker logs data-access-prod --since 5m 2>&1 | grep -c ERROR || echo 0)
if [[ $error_count -gt 10 ]]; then
    error "High error rate in data-access logs: ${error_count} errors in last 5min"
elif [[ $error_count -gt 0 ]]; then
    log "WARNING: ${error_count} errors in data-access logs (last 5min)"
else
    log "No recent errors in data-access logs"
fi

# 6. Test data access
log "Testing data access functionality..."
if python3 - <<'PY' 2>&1 | grep -q "Functional test passed"
import os
os.environ["DATA_ACCESS_SKIP_COS_MIRROR"] = "1"
try:
    from data_access import get_store
    store = get_store()
    df = store.load_columns('ashare_stock_daily', columns=['Close'], time_range=('2024-01-01', '2024-01-02'))
    if len(df) > 0:
        print("Functional test passed")
    else:
        print("Functional test failed: empty result")
except Exception as e:
    print(f"Functional test failed: {e}")
PY
then
    log "Data access functional test: OK"
else
    error "Data access functional test failed"
fi

# Summary
log "Health check completed"
if [[ $EXIT_CODE -eq 0 ]]; then
    log "Status: HEALTHY"
else
    log "Status: UNHEALTHY"
fi

exit $EXIT_CODE
