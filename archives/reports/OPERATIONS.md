# Production Deployment Quick Reference

Quick reference guide for daily operations. See [DEPLOYMENT.md](DEPLOYMENT.md) for comprehensive documentation.

## Service Management

### Docker Compose

```bash
# Start all services
docker-compose -f docker-compose.production.yml up -d

# Stop all services
docker-compose -f docker-compose.production.yml down

# View logs
docker-compose -f docker-compose.production.yml logs -f

# Restart specific service
docker-compose -f docker-compose.production.yml restart data-access
```

### Systemd

```bash
# Start services
sudo systemctl start data-access factor-engine

# Stop services
sudo systemctl stop data-access factor-engine

# Restart services
sudo systemctl restart data-access factor-engine

# Check status
sudo systemctl status data-access factor-engine

# View logs
journalctl -u data-access -f
journalctl -u factor-engine -f
```

## Health Checks

```bash
# Automated health check
bash scripts/health_check.sh

# Manual checks
curl http://localhost:8765/health  # data_access
curl http://localhost:8766/health  # factor_engine
curl http://localhost:9090/        # Prometheus
curl http://localhost:3000/        # Grafana

# Container status
docker ps | grep -E 'data-access|factor-engine'
```

## Deployment

```bash
# Full deployment
bash scripts/deploy_production.sh

# Manual deployment
source .env.production
docker-compose -f docker-compose.production.yml build
docker-compose -f docker-compose.production.yml up -d
```

## Backup & Recovery

```bash
# Manual backup
bash scripts/backup_production.sh

# Scheduled backup (crontab -e)
0 2 * * * /opt/quant_projects/scripts/backup_production.sh

# Recovery
bash scripts/recover_from_backup.sh 20260814_120000
```

## Monitoring Dashboards

- **Grafana**: http://localhost:3000 (admin / $GRAFANA_ADMIN_PASSWORD)
- **Prometheus**: http://localhost:9090
- **Alerts**: http://localhost:9090/alerts

## Troubleshooting

### High Memory Usage

```bash
# Check memory
docker stats --no-stream

# Reduce DuckDB memory limit
docker-compose -f docker-compose.production.yml exec data-access \
    bash -c 'export DUCKDB_MEMORY_LIMIT=16GB && ...'

# Restart service
docker-compose -f docker-compose.production.yml restart data-access
```

### Slow Queries

```bash
# Check recent queries
docker-compose -f docker-compose.production.yml logs data-access | grep "Query time"

# Check connection pool
docker-compose -f docker-compose.production.yml exec data-access \
    python3 -c "from data_access import get_store; print(get_store().pool_stats())"
```

### Service Down

```bash
# Check logs
docker-compose -f docker-compose.production.yml logs --tail=100 data-access

# Restart service
docker-compose -f docker-compose.production.yml restart data-access

# Full restart
docker-compose -f docker-compose.production.yml down
docker-compose -f docker-compose.production.yml up -d
```

### Disk Space Low

```bash
# Check usage
df -h /data

# Clean old logs
find /var/log/quant -name "*.log" -mtime +7 -delete

# Clean Docker cache
docker system prune -a --volumes -f

# Clean old backups
find /mnt/backup/quant -mtime +30 -delete
```

## Configuration Files

| File | Purpose |
|------|---------|
| `.env.production` | Production environment variables |
| `docker-compose.production.yml` | Docker Compose orchestration |
| `monitoring/prometheus.yml` | Prometheus scrape config |
| `monitoring/alerts.yml` | Alert rules |
| `dataaccess/config/datasets.yaml` | Data source definitions |

## Key Environment Variables

```bash
# Security
DATA_ACCESS_API_KEY=<secret>
COS_SECRET_ID=<secret>
COS_SECRET_KEY=<secret>

# Performance
DUCKDB_MEMORY_LIMIT=20GB
DUCKDB_THREADS=12
DATA_ACCESS_API_MAX_CONCURRENCY=8

# Paths
QUANT_PROJECTS_ROOT=/opt/quant_projects
QUANTSOCIETY_WORKSPACE_DATA_ROOT=/data/quant_workspace
```

## Emergency Contacts

- **On-Call Engineer**: [Contact info]
- **Platform Team**: [Contact info]
- **Escalation**: [Contact info]

## Common Tasks

### Materialize New Factor

```bash
source .env.production
python3 scripts/materialize_factor.py \
    --factor-id=my_new_factor \
    --time-range=2024-01-01:2024-12-31
```

### Update Data Sources

```bash
# Sync from COS
bash scripts/sync_ashare_lqtp_cos.sh StockDailyBar
bash scripts/sync_us_stock_cos.sh StockDailyBar
```

### Rotate API Key

```bash
# 1. Generate new key
export NEW_API_KEY=$(openssl rand -hex 32)

# 2. Update secret manager
vault kv put secret/quant/prod api_key=$NEW_API_KEY

# 3. Update .env.production
sed -i "s/DATA_ACCESS_API_KEY=.*/DATA_ACCESS_API_KEY=${NEW_API_KEY}/" .env.production

# 4. Restart services
docker-compose -f docker-compose.production.yml restart
```

### Scale Services

```bash
# Increase replicas (Kubernetes)
kubectl scale deployment data-access --replicas=5

# Increase resources (Docker Compose)
# Edit docker-compose.production.yml, then:
docker-compose -f docker-compose.production.yml up -d --force-recreate
```

---

**See [DEPLOYMENT.md](DEPLOYMENT.md) for full documentation.**
