#!/usr/bin/env bash
# Production deployment automation script
# Usage: bash scripts/deploy_production.sh [--environment prod|staging]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENVIRONMENT="${1:-prod}"

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }

# Load environment
if [[ ! -f "${PROJECT_ROOT}/.env.${ENVIRONMENT}" ]]; then
    error "Environment file .env.${ENVIRONMENT} not found"
fi

source "${PROJECT_ROOT}/.env.${ENVIRONMENT}"

log "Starting deployment to ${ENVIRONMENT}"

# Pre-deployment checks
log "Running pre-deployment checks..."

# Check required environment variables
required_vars=(
    "DATA_ACCESS_API_KEY"
    "COS_SECRET_ID"
    "COS_SECRET_KEY"
    "QUANT_PROJECTS_ROOT"
)

for var in "${required_vars[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        error "Required environment variable ${var} is not set"
    fi
done

# Check disk space
available_space=$(df /data | tail -1 | awk '{print $4}')
if [[ $available_space -lt 524288000 ]]; then  # 500GB in KB
    error "Insufficient disk space. Available: $((available_space / 1024 / 1024))GB, Required: 500GB"
fi

# Run tests
log "Running test suite..."
cd "${PROJECT_ROOT}"
source venv/bin/activate

pytest dataaccess/tests/ -q --tb=short || error "data_access tests failed"
cd factor_engine && pytest tests/ -q --tb=short || error "factor_engine tests failed"
cd "${PROJECT_ROOT}"

# Build Docker images
log "Building Docker images..."
docker-compose -f docker-compose.production.yml build || error "Docker build failed"

# Backup current deployment
if docker ps | grep -q data-access-prod; then
    log "Backing up current deployment..."
    bash scripts/backup_production.sh || log "Backup failed (continuing)"
fi

# Stop old containers
log "Stopping old containers..."
docker-compose -f docker-compose.production.yml down --remove-orphans || true

# Start new deployment
log "Starting new deployment..."
docker-compose -f docker-compose.production.yml up -d

# Wait for services to be healthy
log "Waiting for services to be healthy..."
max_wait=120
elapsed=0

while [[ $elapsed -lt $max_wait ]]; do
    if curl -sf http://localhost:8765/health > /dev/null && \
       curl -sf http://localhost:8766/health > /dev/null; then
        log "Services are healthy"
        break
    fi
    sleep 5
    elapsed=$((elapsed + 5))
done

if [[ $elapsed -ge $max_wait ]]; then
    error "Services failed to become healthy within ${max_wait}s"
fi

# Smoke tests
log "Running smoke tests..."
python3 - <<'PY'
import os
import sys
os.environ["DATA_ACCESS_SKIP_COS_MIRROR"] = "1"

try:
    from data_access import get_store
    store = get_store()
    df = store.load_columns('ashare_stock_daily', columns=['Close'], time_range=('2024-01-01', '2024-01-02'))
    assert len(df) > 0, "Empty result"
    print(f"Smoke test passed: loaded {len(df)} rows")
except Exception as e:
    print(f"Smoke test failed: {e}")
    sys.exit(1)
PY

if [[ $? -ne 0 ]]; then
    error "Smoke tests failed"
fi

log "Deployment completed successfully"
log "Services:"
log "  data_access: http://localhost:8765"
log "  factor_engine: http://localhost:8766"
log "  prometheus: http://localhost:9090"
log "  grafana: http://localhost:3000"
