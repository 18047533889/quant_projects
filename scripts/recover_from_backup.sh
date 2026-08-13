#!/usr/bin/env bash
# Recovery script for disaster scenarios
# Usage: bash scripts/recover_from_backup.sh <backup_timestamp>

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <backup_timestamp>"
    echo "Example: $0 20260814_120000"
    exit 1
fi

BACKUP_TIMESTAMP="$1"
BACKUP_ROOT="${BACKUP_ROOT:-/mnt/backup/quant}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }

log "Starting recovery from backup: ${BACKUP_TIMESTAMP}"

# Verify backup exists
if [[ ! -d "${BACKUP_ROOT}/factor_lake/${BACKUP_TIMESTAMP}" ]]; then
    error "Backup not found: ${BACKUP_ROOT}/factor_lake/${BACKUP_TIMESTAMP}"
fi

# Stop services
log "Stopping services..."
docker-compose -f docker-compose.production.yml down || log "No containers to stop"

# Backup current state (safety measure)
if [[ -d /data/quant_workspace/factors/lake ]]; then
    log "Creating safety backup of current state..."
    mv /data/quant_workspace/factors/lake "/data/quant_workspace/factors/lake.before_recovery.$(date +%s)" || true
fi

# 1. Restore factor lake
log "Restoring factor lake..."
mkdir -p /data/quant_workspace/factors
rsync -av "${BACKUP_ROOT}/factor_lake/${BACKUP_TIMESTAMP}/" \
    /data/quant_workspace/factors/lake/ || error "Factor lake restore failed"

# 2. Restore configurations
log "Restoring configurations..."
if [[ -f "${BACKUP_ROOT}/config/config_${BACKUP_TIMESTAMP}.tar.gz" ]]; then
    tar xzf "${BACKUP_ROOT}/config/config_${BACKUP_TIMESTAMP}.tar.gz" -C / || error "Config restore failed"
fi

# 3. Restore DuckDB catalog
log "Restoring DuckDB catalog..."
if [[ -f "${BACKUP_ROOT}/catalog/catalog_${BACKUP_TIMESTAMP}.db" ]]; then
    mkdir -p /data/quant_workspace/duckdb
    cp "${BACKUP_ROOT}/catalog/catalog_${BACKUP_TIMESTAMP}.db" \
        /data/quant_workspace/duckdb/catalog.db || error "Catalog restore failed"
fi

# 4. Verify data integrity
log "Verifying data integrity..."
cd "${PROJECT_ROOT}"

# Check factor lake structure
factor_count=$(find /data/quant_workspace/factors/lake/factors -name "*.parquet" 2>/dev/null | wc -l || echo 0)
log "Found ${factor_count} parquet files in factor lake"

if [[ $factor_count -eq 0 ]]; then
    error "Factor lake appears empty after restore"
fi

# 5. Test data access
log "Testing data access..."
source venv/bin/activate
python3 - <<'PY' || error "Data access verification failed"
import os
os.environ["DATA_ACCESS_SKIP_COS_MIRROR"] = "1"

from data_access import get_store
store = get_store()

try:
    df = store.load_columns(
        'ashare_stock_daily',
        columns=['Close'],
        time_range=('2024-01-01', '2024-01-02')
    )
    print(f"✓ Loaded {len(df)} rows from test query")
    assert len(df) > 0, "Empty result"
    print("✓ Data access verification passed")
except Exception as e:
    print(f"✗ Data access verification failed: {e}")
    raise
PY

# 6. Restart services
log "Restarting services..."
source "${PROJECT_ROOT}/.env.production"
docker-compose -f docker-compose.production.yml up -d || error "Service restart failed"

# Wait for health
log "Waiting for services to be healthy..."
max_wait=120
elapsed=0

while [[ $elapsed -lt $max_wait ]]; do
    if curl -sf http://localhost:8765/health > /dev/null && \
       curl -sf http://localhost:8766/health > /dev/null; then
        log "✓ Services are healthy"
        break
    fi
    sleep 5
    elapsed=$((elapsed + 5))
done

if [[ $elapsed -ge $max_wait ]]; then
    error "Services failed to become healthy within ${max_wait}s"
fi

# 7. Final smoke test
log "Running final smoke test..."
bash scripts/health_check.sh || error "Health check failed after recovery"

log "Recovery completed successfully"
log "Restored from backup: ${BACKUP_TIMESTAMP}"
log "Services are running and healthy"
log ""
log "Next steps:"
log "  1. Verify critical factors: python3 scripts/verify_factors.py"
log "  2. Check recent logs: docker-compose logs -f"
log "  3. Monitor dashboards: http://localhost:3000"
