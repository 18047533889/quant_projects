#!/usr/bin/env bash
# Production backup script
# Usage: bash scripts/backup_production.sh

set -euo pipefail

BACKUP_ROOT="${BACKUP_ROOT:-/mnt/backup/quant}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }

# Create backup directories
mkdir -p "${BACKUP_ROOT}"/{factor_lake,config,catalog,logs}

log "Starting backup at ${TIMESTAMP}"

# 1. Backup factor lake (incremental with hard links)
if [[ -d /data/quant_workspace/factors/lake ]]; then
    log "Backing up factor lake (incremental)..."
    rsync -av --link-dest="${BACKUP_ROOT}/factor_lake_latest" \
        /data/quant_workspace/factors/lake/ \
        "${BACKUP_ROOT}/factor_lake/${TIMESTAMP}/" || log "Factor lake backup failed"

    # Update latest symlink
    ln -sfn "${BACKUP_ROOT}/factor_lake/${TIMESTAMP}" "${BACKUP_ROOT}/factor_lake_latest"
    log "Factor lake backup completed"
fi

# 2. Backup configurations
log "Backing up configurations..."
tar czf "${BACKUP_ROOT}/config/config_${TIMESTAMP}.tar.gz" \
    -C "${PROJECT_ROOT}" \
    dataaccess/config/ \
    factor_engine/config/ \
    .env.production \
    docker-compose.production.yml \
    monitoring/ \
    2>/dev/null || log "Config backup completed with warnings"

# 3. Backup DuckDB catalog
if [[ -f /data/quant_workspace/duckdb/catalog.db ]]; then
    log "Backing up DuckDB catalog..."
    sqlite3 /data/quant_workspace/duckdb/catalog.db \
        ".backup '${BACKUP_ROOT}/catalog/catalog_${TIMESTAMP}.db'" || log "Catalog backup failed"
fi

# 4. Backup logs
log "Backing up logs..."
if [[ -d /var/log/quant ]]; then
    tar czf "${BACKUP_ROOT}/logs/logs_${TIMESTAMP}.tar.gz" \
        -C /var/log/quant . || log "Log backup failed"
fi

# 5. Create backup manifest
cat > "${BACKUP_ROOT}/manifest_${TIMESTAMP}.txt" <<EOF
Backup Timestamp: ${TIMESTAMP}
Host: $(hostname)
Factor Lake Size: $(du -sh /data/quant_workspace/factors/lake 2>/dev/null | cut -f1 || echo "N/A")
Config Files: $(tar tzf "${BACKUP_ROOT}/config/config_${TIMESTAMP}.tar.gz" 2>/dev/null | wc -l || echo "0")
Catalog Size: $(stat -f%z "${BACKUP_ROOT}/catalog/catalog_${TIMESTAMP}.db" 2>/dev/null || echo "0") bytes
Services: $(docker ps --format '{{.Names}}' | grep -E 'data-access|factor-engine' || echo "none")
EOF

log "Backup manifest created"

# 6. Cleanup old backups (keep last 7 days, plus monthly)
log "Cleaning up old backups..."
find "${BACKUP_ROOT}/factor_lake" -maxdepth 1 -type d -mtime +7 ! -name "$(date +%Y%m)01_*" -exec rm -rf {} + 2>/dev/null || true
find "${BACKUP_ROOT}/config" -name "config_*.tar.gz" -mtime +30 -delete 2>/dev/null || true
find "${BACKUP_ROOT}/catalog" -name "catalog_*.db" -mtime +14 -delete 2>/dev/null || true
find "${BACKUP_ROOT}/logs" -name "logs_*.tar.gz" -mtime +7 -delete 2>/dev/null || true

# 7. Optional: Upload to COS
if [[ "${BACKUP_TO_COS:-0}" == "1" ]] && command -v rclone &> /dev/null; then
    log "Uploading backup to COS..."
    rclone sync "${BACKUP_ROOT}/factor_lake/${TIMESTAMP}/" \
        cos:qs-cold/backup/factor_lake/${TIMESTAMP}/ \
        --transfers=4 \
        --checkers=8 \
        --fast-list || log "COS upload failed"
fi

log "Backup completed successfully"
log "Backup location: ${BACKUP_ROOT}"
log "Factor lake backup: ${BACKUP_ROOT}/factor_lake/${TIMESTAMP}"
log "Config backup: ${BACKUP_ROOT}/config/config_${TIMESTAMP}.tar.gz"
