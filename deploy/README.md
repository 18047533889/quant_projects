# Quant Projects CI/CD and Deployment

## Directory Structure

```
.
├── .github/workflows/          # GitHub Actions workflows
│   ├── build-and-push.yml      # Build and push Docker images
│   ├── deploy-staging.yml      # Deploy to staging environment
│   ├── deploy-production.yml   # Deploy to production environment
│   ├── test.yml                # Run tests and quality checks
│   └── release.yml             # Create releases
├── docker/                     # Docker configurations
│   ├── dataaccess/             # DataAccess Dockerfile
│   ├── factor_engine/          # FactorEngine Dockerfile
│   ├── factor_preprocess/      # FactorPreprocess Dockerfile
│   ├── factor_optimizer/       # FactorOptimizer Dockerfile
│   ├── quant_evaluator/        # QuantEvaluator Dockerfile
│   ├── docker-compose.dev.yml  # Development compose file
│   └── docker-compose.production.yml  # Production compose file
├── deploy/                     # Deployment configurations
│   ├── kubernetes/             # Kubernetes manifests
│   │   ├── production.yaml     # Production K8s config
│   │   └── staging.yaml        # Staging K8s config
│   ├── scripts/                # Deployment scripts
│   │   ├── blue-green-deploy.sh    # Blue-green deployment
│   │   ├── canary-deploy.sh        # Canary deployment
│   │   ├── rollback.sh             # Rollback script
│   │   ├── health-check.sh         # Health check script
│   │   ├── setup-dev.sh            # Development setup
│   │   ├── backup.sh               # Backup script
│   │   └── restore.sh              # Restore script
│   ├── monitoring/             # Monitoring configurations
│   │   ├── prometheus.yml      # Prometheus config
│   │   ├── alert_rules.yml     # Alert rules
│   │   └── grafana/            # Grafana dashboards
│   └── env/                    # Environment configurations
│       ├── .env.template       # Template for environment variables
│       ├── .env.dev            # Development environment
│       ├── .env.staging        # Staging environment
│       └── .env.production     # Production environment
└── docs/
    └── DEPLOYMENT.md           # Complete deployment documentation
```

## Quick Start

### Local Development

```bash
# Run setup script
./deploy/scripts/setup-dev.sh

# Or manual setup
cd docker
docker-compose -f docker-compose.dev.yml up -d

# Access services
open http://localhost:8765  # DataAccess
open http://localhost:8766  # FactorEngine
open http://localhost:3000  # Grafana
```

### Deploy to Staging

```bash
# Push to develop branch
git push origin develop

# Or manually trigger
gh workflow run deploy-staging.yml
```

### Deploy to Production

```bash
# Create and push a tag
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0

# Or manually trigger
gh workflow run deploy-production.yml \
  -f image_tag=v1.0.0 \
  -f deployment_strategy=rolling
```

## Features

### CI/CD Pipeline

- **Automated Testing**: Unit tests, integration tests, linting, type checking
- **Security Scanning**: Bandit, Safety checks for vulnerabilities
- **Docker Build**: Multi-stage builds with layer caching
- **Automated Deployment**: Staging on develop, production on release
- **Rollback Support**: Automatic and manual rollback capabilities

### Deployment Strategies

1. **Rolling Update**: Gradual replacement of pods (default, zero downtime)
2. **Blue-Green**: Two identical environments with instant switch
3. **Canary**: Gradual rollout with automatic rollback on errors

### Monitoring

- **Prometheus**: Metrics collection and alerting
- **Grafana**: Pre-configured dashboards for visualization
- **Alerts**: Critical alerts for service health, errors, performance

### Security

- **Non-root containers**: All services run as dedicated users
- **Secrets management**: Kubernetes secrets, never in git
- **Network policies**: Restrict pod-to-pod communication
- **Security scanning**: Automated vulnerability scanning

## Usage

### Build Docker Images

```bash
# Build single service
docker build -f docker/dataaccess/Dockerfile -t quant/dataaccess:dev ./dataaccess

# Build all services
cd docker
docker-compose -f docker-compose.dev.yml build
```

### Deploy to Kubernetes

```bash
# Deploy to staging
kubectl apply -f deploy/kubernetes/staging.yaml

# Deploy to production
kubectl apply -f deploy/kubernetes/production.yaml

# Check status
kubectl get pods -n quant-production
```

### Use Deployment Scripts

```bash
# Blue-green deployment
./deploy/scripts/blue-green-deploy.sh dataaccess v1.0.0

# Canary deployment (10% traffic)
CANARY_WEIGHT=10 ./deploy/scripts/canary-deploy.sh dataaccess v1.0.0

# Rollback
./deploy/scripts/rollback.sh dataaccess

# Health check
./deploy/scripts/health-check.sh dataaccess http://dataaccess:8765
```

### Backup and Restore

```bash
# Create backup
./deploy/scripts/backup.sh

# Restore from backup
./deploy/scripts/restore.sh /backups/quant/quant-backup-20260814_120000.tar.gz
```

## Configuration

### Environment Variables

Copy and customize environment files:

```bash
cp deploy/env/.env.template deploy/env/.env.local
# Edit .env.local with your values
```

### Kubernetes Secrets

```bash
# Create secrets
kubectl create secret generic quant-secrets \
  --from-literal=DATABASE_PASSWORD=xxx \
  --from-literal=COS_SECRET_KEY=yyy \
  -n quant-production
```

### Resource Limits

Edit `deploy/kubernetes/production.yaml`:

```yaml
resources:
  requests:
    cpu: 2000m
    memory: 4Gi
  limits:
    cpu: 4000m
    memory: 8Gi
```

## Monitoring

### Access Dashboards

```bash
# Prometheus
kubectl port-forward -n quant-production svc/prometheus 9090:9090

# Grafana
kubectl port-forward -n quant-production svc/grafana 3000:3000
```

### View Logs

```bash
# View recent logs
kubectl logs -l app=dataaccess -n quant-production --tail=100

# Follow logs
kubectl logs -f -l app=dataaccess -n quant-production

# Export logs
kubectl logs -l app=dataaccess -n quant-production --since=1h > logs.txt
```

### Check Metrics

```bash
# Service health
curl http://dataaccess:8765/metrics

# Prometheus queries
curl 'http://prometheus:9090/api/v1/query?query=up'
```

## Troubleshooting

### Common Issues

1. **Pods not starting**: Check `kubectl describe pod <pod-name>`
2. **Image pull errors**: Verify registry access and image tags
3. **Health checks failing**: Check application logs
4. **High memory usage**: Increase resource limits or optimize code

### Debug Commands

```bash
# Get pod status
kubectl get pods -n quant-production

# Describe pod
kubectl describe pod <pod-name> -n quant-production

# View events
kubectl get events -n quant-production --sort-by='.lastTimestamp'

# Execute commands in pod
kubectl exec -it <pod-name> -n quant-production -- bash
```

## Documentation

- [DEPLOYMENT.md](../docs/DEPLOYMENT.md) - Complete deployment guide
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

## Support

For issues or questions:
- Create an issue in the repository
- Contact: devops@example.com
- Slack: #quant-infrastructure
