# Quant Projects - Deployment Guide

## Overview

This guide covers the complete deployment pipeline for the Quant Projects ecosystem, including DataAccess, FactorEngine, and supporting services.

## Architecture

```
┌─────────────────┐
│   Load Balancer │
└────────┬────────┘
         │
    ┌────┴────┐
    │  Nginx  │
    └────┬────┘
         │
    ┌────┴──────────────┐
    │                   │
┌───▼────────┐  ┌──────▼──────┐
│ DataAccess │  │ FactorEngine│
└────────────┘  └─────────────┘
         │              │
    ┌────┴──────────────┘
    │
┌───▼───────────┐
│   Storage     │
│  (COS/S3)     │
└───────────────┘
```

## Prerequisites

- Kubernetes cluster (v1.24+)
- kubectl configured
- Docker (v20.10+)
- GitHub Actions runner (for CI/CD)
- Helm (v3.0+) - optional
- Access to container registry (ghcr.io)

## Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/your-org/quant_projects.git
cd quant_projects

# Start services with Docker Compose
cd docker
docker-compose -f docker-compose.dev.yml up -d

# Verify services
docker-compose ps
curl http://localhost:8765/health  # DataAccess
curl http://localhost:8766/health  # FactorEngine

# View logs
docker-compose logs -f dataaccess
docker-compose logs -f factorengine

# Access Grafana dashboard
open http://localhost:3000  # admin/admin
```

### Building Docker Images

```bash
# Build all images
docker build -f docker/dataaccess/Dockerfile -t quant/dataaccess:dev ./dataaccess
docker build -f docker/factor_engine/Dockerfile -t quant/factor-engine:dev ./factor_engine

# Build with build args
docker build \
  --build-arg BUILD_SHA=$(git rev-parse HEAD) \
  --build-arg BUILD_VERSION=$(git describe --tags) \
  -f docker/dataaccess/Dockerfile \
  -t quant/dataaccess:$(git describe --tags) \
  ./dataaccess
```

## CI/CD Pipeline

### GitHub Actions Workflows

**1. Build and Push** (`.github/workflows/build-and-push.yml`)
- Triggered on: push to main/master, tags, PRs
- Builds Docker images for all services
- Pushes to GitHub Container Registry
- Uses layer caching for faster builds

**2. Deploy to Staging** (`.github/workflows/deploy-staging.yml`)
- Triggered on: push to develop branch
- Deploys to staging environment
- Runs smoke tests
- Automatic rollback on failure

**3. Deploy to Production** (`.github/workflows/deploy-production.yml`)
- Triggered on: release published, manual dispatch
- Requires approval for production environment
- Supports multiple deployment strategies
- Includes pre/post deployment checks

### Manual Deployment

```bash
# Tag a release
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0

# Or trigger manual deployment
gh workflow run deploy-production.yml \
  -f image_tag=v1.0.0 \
  -f deployment_strategy=rolling
```

## Deployment Strategies

### Rolling Update (Default)

Zero-downtime deployment that gradually replaces old pods with new ones.

```bash
kubectl set image deployment/dataaccess \
  dataaccess=ghcr.io/your-org/quant/dataaccess:v1.0.0 \
  -n quant-production

kubectl rollout status deployment/dataaccess -n quant-production
```

**Pros**: Simple, automatic rollback
**Cons**: Both versions run simultaneously
**Use when**: Standard updates, low risk changes

### Blue-Green Deployment

Maintains two identical environments, switching traffic between them.

```bash
./deploy/scripts/blue-green-deploy.sh dataaccess v1.0.0
```

**Pros**: Instant rollback, full testing before switch
**Cons**: Requires 2x resources temporarily
**Use when**: Major releases, database migrations

### Canary Deployment

Gradually rolls out to a subset of users before full deployment.

```bash
CANARY_WEIGHT=10 ./deploy/scripts/canary-deploy.sh dataaccess v1.0.0
```

**Pros**: Early detection of issues, gradual rollout
**Cons**: Complex setup, requires traffic splitting
**Use when**: High-risk changes, A/B testing needed

## Kubernetes Deployment

### Deploy to Staging

```bash
# Apply staging configuration
kubectl apply -f deploy/kubernetes/staging.yaml

# Verify deployment
kubectl get pods -n quant-staging
kubectl get services -n quant-staging

# Check logs
kubectl logs -f -l app=dataaccess -n quant-staging
```

### Deploy to Production

```bash
# Review configuration
kubectl diff -f deploy/kubernetes/production.yaml

# Apply configuration
kubectl apply -f deploy/kubernetes/production.yaml

# Monitor rollout
kubectl rollout status deployment/dataaccess -n quant-production
kubectl rollout status deployment/factorengine -n quant-production

# Verify health
./deploy/scripts/health-check.sh dataaccess http://dataaccess.quant-production:8765
./deploy/scripts/health-check.sh factorengine http://factorengine.quant-production:8766
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment/dataaccess --replicas=5 -n quant-production

# Auto-scaling is configured via HPA (Horizontal Pod Autoscaler)
kubectl get hpa -n quant-production

# View current scaling status
kubectl describe hpa dataaccess-hpa -n quant-production
```

## Rollback Procedures

### Automatic Rollback

Kubernetes automatically rolls back if readiness probes fail.

### Manual Rollback

```bash
# View rollout history
kubectl rollout history deployment/dataaccess -n quant-production

# Rollback to previous version
./deploy/scripts/rollback.sh dataaccess

# Rollback to specific revision
./deploy/scripts/rollback.sh dataaccess 5

# Or using kubectl
kubectl rollout undo deployment/dataaccess -n quant-production
```

### Emergency Rollback

```bash
# For blue-green deployment
kubectl patch service dataaccess -n quant-production \
  -p '{"spec":{"selector":{"color":"blue"}}}'

# Scale up old deployment
kubectl scale deployment/dataaccess-blue --replicas=2 -n quant-production
```

## Environment Configuration

### Secrets Management

```bash
# Create secret from file
kubectl create secret generic quant-secrets \
  --from-file=.env.production \
  -n quant-production

# Or from literals
kubectl create secret generic quant-secrets \
  --from-literal=DATABASE_PASSWORD=xxx \
  --from-literal=COS_SECRET_KEY=yyy \
  -n quant-production

# View secrets (base64 encoded)
kubectl get secret quant-secrets -n quant-production -o yaml

# Decode secret
kubectl get secret quant-secrets -n quant-production \
  -o jsonpath='{.data.DATABASE_PASSWORD}' | base64 -d
```

### ConfigMaps

```bash
# Create from file
kubectl create configmap dataaccess-config \
  --from-env-file=deploy/env/.env.production \
  -n quant-production

# Update existing configmap
kubectl create configmap dataaccess-config \
  --from-env-file=deploy/env/.env.production \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart pods to pick up new config
kubectl rollout restart deployment/dataaccess -n quant-production
```

## Monitoring and Observability

### Prometheus Metrics

Access Prometheus at `http://prometheus:9090`

**Key Metrics:**
- `up{job="dataaccess"}` - Service health
- `http_requests_total` - Request count
- `http_request_duration_seconds` - Latency
- `process_resident_memory_bytes` - Memory usage
- `process_cpu_seconds_total` - CPU usage

```bash
# Query via CLI
kubectl port-forward -n quant-production svc/prometheus 9090:9090

# Example queries
curl 'http://localhost:9090/api/v1/query?query=up'
```

### Grafana Dashboards

Access Grafana at `http://grafana:3000` (admin/admin)

**Available Dashboards:**
- Quant Services Overview - High-level service metrics
- DataAccess Performance - Detailed DataAccess metrics
- FactorEngine Performance - Detailed FactorEngine metrics
- Kubernetes Resources - Pod/Node metrics

```bash
# Port forward to local machine
kubectl port-forward -n quant-production svc/grafana 3000:3000
```

### Logs

```bash
# View recent logs
kubectl logs -l app=dataaccess -n quant-production --tail=100

# Follow logs
kubectl logs -f -l app=dataaccess -n quant-production

# View logs from previous instance (crashed pod)
kubectl logs -l app=dataaccess -n quant-production --previous

# Export logs for analysis
kubectl logs -l app=dataaccess -n quant-production --since=1h > dataaccess.log
```

### Alerts

Alerts are configured in `deploy/monitoring/alert_rules.yml`

**Critical Alerts:**
- Service Down (2min)
- High Error Rate (>5% for 5min)
- High Memory Usage (>90% for 10min)
- Pod Crash Looping

**View Active Alerts:**
```bash
# Via Prometheus
curl http://prometheus:9090/api/v1/alerts

# Via AlertManager (if configured)
curl http://alertmanager:9093/api/v2/alerts
```

## Troubleshooting

### Pods Not Starting

```bash
# Check pod status
kubectl get pods -n quant-production

# Describe pod for events
kubectl describe pod <pod-name> -n quant-production

# Check logs
kubectl logs <pod-name> -n quant-production

# Common issues:
# - Image pull errors: Check registry access
# - CrashLoopBackOff: Check application logs
# - Pending: Check resource availability
```

### High Memory Usage

```bash
# Check memory usage
kubectl top pods -n quant-production

# View OOM kills
kubectl get events -n quant-production | grep OOM

# Temporary fix: increase memory limits
kubectl set resources deployment/dataaccess \
  --limits=memory=12Gi \
  -n quant-production
```

### Network Issues

```bash
# Test service connectivity
kubectl run -it --rm debug --image=busybox --restart=Never -n quant-production -- sh
wget -O- http://dataaccess:8765/health

# Check service endpoints
kubectl get endpoints -n quant-production

# View network policies
kubectl get networkpolicies -n quant-production
```

### Performance Issues

```bash
# Check resource utilization
kubectl top nodes
kubectl top pods -n quant-production

# View HPA status
kubectl get hpa -n quant-production

# Check for throttling
kubectl describe node <node-name> | grep -A5 "Allocated resources"
```

## Maintenance

### Database Migrations

```bash
# Run migrations as a Job
kubectl create job --from=cronjob/db-migrate migrate-$(date +%s) -n quant-production

# Monitor migration
kubectl logs -f job/migrate-<timestamp> -n quant-production
```

### Backup and Restore

```bash
# Backup deployment configuration
kubectl get deployment/dataaccess -n quant-production -o yaml > backup-dataaccess.yaml

# Backup persistent data (via COS/S3)
# Handled automatically by application

# Restore from backup
kubectl apply -f backup-dataaccess.yaml
```

### Certificate Renewal

```bash
# Check certificate expiry
kubectl get certificate -n quant-production

# Cert-manager handles automatic renewal
# Manual renewal if needed:
kubectl delete certificate quant-tls -n quant-production
kubectl apply -f deploy/kubernetes/production.yaml
```

## Security Best Practices

1. **Never commit secrets to git**
   - Use Kubernetes secrets or external secret managers
   - Use `.env.template` files with placeholders

2. **Use non-root containers**
   - All Dockerfiles use dedicated users (uid 10001+)
   - Set securityContext in Kubernetes

3. **Network policies**
   - Restrict pod-to-pod communication
   - Use ingress/egress rules

4. **Image scanning**
   - GitHub Actions includes security scanning
   - Use signed images in production

5. **RBAC**
   - Use service accounts with minimal permissions
   - Separate permissions per environment

## Performance Tuning

### Resource Limits

```yaml
resources:
  requests:
    cpu: 2000m      # Guaranteed CPU
    memory: 4Gi     # Guaranteed memory
  limits:
    cpu: 4000m      # Maximum CPU
    memory: 8Gi     # Maximum memory (OOM kill threshold)
```

### Connection Pooling

Configure in `.env` files:
```bash
DATABASE_POOL_SIZE=50
CONNECTION_POOL_SIZE=100
MAX_CONCURRENT_REQUESTS=500
```

### Caching

```bash
# DataAccess cache configuration
DUCKDB_MAX_MEMORY=16GB
DUCKDB_THREADS=16

# FactorEngine cache configuration
FACTOR_ENGINE_CACHE_SIZE=8GB
FACTOR_ENGINE_MAX_WORKERS=32
```

## Support and Contacts

- **DevOps Team**: devops@example.com
- **On-Call**: oncall@example.com
- **Slack**: #quant-infrastructure
- **Runbook**: https://wiki.example.com/quant/runbook

## Appendix

### Useful Commands

```bash
# Quick health check
kubectl get all -n quant-production

# Resource usage summary
kubectl top nodes && kubectl top pods -n quant-production

# Recent events
kubectl get events -n quant-production --sort-by='.lastTimestamp' | tail -20

# Service endpoints
kubectl get svc,ep -n quant-production

# Full cluster status
kubectl cluster-info && kubectl get nodes
```

### Links

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Prometheus Querying](https://prometheus.io/docs/prometheus/latest/querying/)
- [Grafana Dashboards](https://grafana.com/docs/grafana/latest/dashboards/)
