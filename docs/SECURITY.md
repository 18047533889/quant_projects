# Security Policies for Quant Projects

## Overview

This document outlines security policies and best practices for deploying and maintaining the Quant Projects ecosystem.

## Authentication and Authorization

### API Authentication

All production APIs must use authentication:

```yaml
# Enable API key authentication in production
ENABLE_API_KEY_AUTH=true

# Rotate keys regularly (every 90 days)
# Store keys in Kubernetes secrets, never in code
```

### Service Accounts

Use dedicated service accounts with minimal permissions:

```bash
# Create service account
kubectl create serviceaccount dataaccess-sa -n quant-production

# Bind to role
kubectl create rolebinding dataaccess-binding \
  --role=dataaccess-role \
  --serviceaccount=quant-production:dataaccess-sa \
  -n quant-production
```

## Secrets Management

### Never Commit Secrets

- Use `.env.template` files with placeholders
- Store real secrets in Kubernetes secrets or external secret managers
- Use `.gitignore` to prevent accidental commits

### Kubernetes Secrets

```bash
# Create secret from literals
kubectl create secret generic quant-secrets \
  --from-literal=DATABASE_PASSWORD=xxx \
  --from-literal=COS_SECRET_KEY=yyy \
  -n quant-production

# Create secret from file
kubectl create secret generic quant-secrets \
  --from-file=.env.production \
  -n quant-production

# Use sealed secrets for GitOps
kubeseal --format=yaml < secret.yaml > sealed-secret.yaml
```

### Secret Rotation

Rotate secrets regularly:

1. Database passwords: Every 90 days
2. API keys: Every 90 days
3. TLS certificates: Automatic via cert-manager
4. Object storage credentials: Every 180 days

## Container Security

### Non-Root Containers

All containers run as non-root users:

```dockerfile
# Create dedicated user
RUN addgroup -g 10001 dataaccess && \
    adduser -D -u 10001 -G dataaccess dataaccess

USER dataaccess
```

### Image Scanning

All images are scanned for vulnerabilities:

```yaml
# GitHub Actions security scan
- name: Run Trivy vulnerability scanner
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: 'ghcr.io/${{ github.repository }}/dataaccess:${{ github.sha }}'
    format: 'sarif'
    output: 'trivy-results.sarif'
```

### Image Signing

Production images must be signed:

```bash
# Sign image with cosign
cosign sign ghcr.io/your-org/dataaccess:v1.0.0

# Verify signature
cosign verify ghcr.io/your-org/dataaccess:v1.0.0
```

## Network Security

### Network Policies

Restrict pod-to-pod communication:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: dataaccess-policy
  namespace: quant-production
spec:
  podSelector:
    matchLabels:
      app: dataaccess
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: factorengine
    ports:
    - protocol: TCP
      port: 8765
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
```

### TLS/SSL

Enable TLS for all external communication:

```yaml
# Ingress with TLS
spec:
  tls:
  - hosts:
    - dataaccess.example.com
    secretName: quant-tls
```

### Rate Limiting

Protect against DDoS:

```yaml
# Nginx ingress rate limiting
metadata:
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/limit-rps: "10"
```

## Data Security

### Encryption at Rest

- Database: Enable encryption at rest
- Object Storage: Use server-side encryption
- Persistent Volumes: Use encrypted storage classes

```yaml
# Encrypted storage class
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: encrypted-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  encrypted: "true"
```

### Encryption in Transit

- Use TLS 1.2+ for all connections
- Enforce HTTPS redirects
- Use mutual TLS for service-to-service communication

### Data Access Control

```yaml
# Row-level security in PostgreSQL
CREATE POLICY user_data_policy ON sensitive_data
  USING (user_id = current_user_id());

# Object storage bucket policies
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"AWS": "arn:aws:iam::ACCOUNT:role/dataaccess"},
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::quant-production/*"
  }]
}
```

## Code Security

### Dependency Scanning

Scan dependencies for vulnerabilities:

```bash
# Python security check
pip install safety
safety check

# Dependency vulnerability scan
pip-audit
```

### Static Code Analysis

```bash
# Security linting with bandit
bandit -r . -f json -o security-report.json

# Additional security checks
semgrep --config=p/security-audit .
```

### Code Review

- All code changes require review
- Security-sensitive changes require security team review
- Automated security checks in CI/CD

## Access Control

### RBAC (Role-Based Access Control)

```yaml
# Read-only role
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-reader
  namespace: quant-production
rules:
- apiGroups: [""]
  resources: ["pods", "pods/log"]
  verbs: ["get", "list", "watch"]

# Deployment manager role
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: deployment-manager
  namespace: quant-production
rules:
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch", "update", "patch"]
```

### Principle of Least Privilege

- Grant minimum necessary permissions
- Use separate service accounts per service
- Regular access audits

## Monitoring and Auditing

### Security Monitoring

```yaml
# Alert on suspicious activity
- alert: UnauthorizedAccessAttempt
  expr: rate(http_requests_total{status="401"}[5m]) > 10
  labels:
    severity: warning
  annotations:
    summary: "High rate of unauthorized access attempts"

- alert: PrivilegeEscalation
  expr: kube_pod_container_status_running{container="nginx"} and 
        kube_pod_security_context_run_as_user == 0
  labels:
    severity: critical
  annotations:
    summary: "Container running as root (potential privilege escalation)"
```

### Audit Logging

Enable Kubernetes audit logging:

```yaml
# Audit policy
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
- level: Metadata
  namespaces: ["quant-production"]
  verbs: ["create", "update", "delete"]
```

### Security Dashboards

Monitor security metrics:
- Failed authentication attempts
- Privilege escalation attempts
- Network policy violations
- Secret access patterns

## Incident Response

### Security Incident Procedure

1. **Detection**: Automated alerts, monitoring
2. **Containment**: Isolate affected services
3. **Investigation**: Review logs, audit trails
4. **Remediation**: Patch vulnerabilities, update configs
5. **Recovery**: Restore services, verify integrity
6. **Post-Mortem**: Document lessons learned

### Emergency Contacts

- Security Team: security@example.com
- On-Call: oncall@example.com
- Incident Response: +1-XXX-XXX-XXXX

## Compliance

### Data Residency

Ensure data stays in compliant regions:

```yaml
# Node affinity for data residency
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: topology.kubernetes.io/region
          operator: In
          values:
          - us-east-1
```

### Audit Requirements

- Maintain audit logs for 1 year
- Regular security assessments
- Penetration testing annually
- Third-party security audits

## Security Checklist

### Pre-Deployment

- [ ] All secrets stored in secret manager
- [ ] Non-root containers
- [ ] Image vulnerability scan passed
- [ ] Network policies configured
- [ ] TLS certificates valid
- [ ] RBAC roles configured
- [ ] Resource limits set

### Post-Deployment

- [ ] Health checks passing
- [ ] Security monitoring enabled
- [ ] Audit logging enabled
- [ ] Backup configured
- [ ] Incident response plan reviewed
- [ ] Access controls verified

## Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CIS Kubernetes Benchmark](https://www.cisecurity.org/benchmark/kubernetes)
- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

## Updates

This security policy should be reviewed and updated:
- Quarterly for regular updates
- Immediately after security incidents
- When new threats are identified
- When infrastructure changes
