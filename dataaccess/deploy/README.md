# Enterprise deployment templates

These templates are examples only. Replace the image, secret references, data/cache
volumes, resource limits, TLS, ingress and COS credentials in the deployment system.
Never commit COS secrets, certificates or private hostnames.

**Team HTTP API key (agreed):** `quantsociety`  
Set as `DATA_ACCESS_API_KEY` / request header `X-API-Key`. This is also the package
default when the env var is unset. Override via secret manager in real production if needed.

- `Dockerfile`: non-root container image.
- `docker-compose.yml`: local smoke deployment (`DATA_ACCESS_API_KEY=quantsociety`).
- `kubernetes.yaml`: Deployment and Service with probes and resource limits.
- `data-access.service`: systemd unit for a host installation.

Production prerequisites outside this package include an internal image/PyPI registry,
secret manager, TLS/API gateway, network policy, COS/S3 access, metrics/logging backend,
and tested CPU/memory/concurrency capacity values.
