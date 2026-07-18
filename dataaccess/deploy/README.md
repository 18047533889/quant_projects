# Enterprise deployment templates

These templates are examples only. Replace the image, secret references, data/cache
volumes, resource limits, TLS, ingress and COS credentials in the deployment system.
Never commit a real API key, COS secret, certificate or private hostname.

- `Dockerfile`: non-root container image.
- `docker-compose.yml`: local smoke deployment.
- `kubernetes.yaml`: Deployment and Service with probes and resource limits.
- `data-access.service`: systemd unit for a host installation.

Production prerequisites outside this package include an internal image/PyPI registry,
secret manager, TLS/API gateway, network policy, COS/S3 access, metrics/logging backend,
and tested CPU/memory/concurrency capacity values.
