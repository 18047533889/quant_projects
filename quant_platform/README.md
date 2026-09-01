# quant_platform

Quant Research Platform — the thin **integration/contract layer**. It holds the
pure-stdlib DTO contracts (23 modules, `__all__` 135 items), auth/RBAC, outbox/
worker/orchestrator skeletons, and storage adapters. It is the platform-side
boundary that domain packages must NOT import.

**Version:** 0.1.0 ｜ **Repo:** https://github.com/HKUST-QUANT-SOCIETY/quant_platform (private)

> Status: **DRAFT / WIP** — Control & Metadata planes not yet built (no Postgres,
> no workflow runtime, no unified registry). Contracts are NOT V1_FROZEN pending
> cross-package reconciliation.

## What this package owns

- **`app/contracts/`** — pure stdlib frozen dataclasses + `typing.Protocol`
  (PURE-DTO rule: no fastapi/sqlalchemy/pydantic). Modules: `artifact_ref`,
  `event_envelope`, `jobs`, `workflow`, `candidate`, `lifecycle` (23-state +
  health), `identities`, `cluster_library`, `feature_set`, `rbac`/`permission`/
  `principal`, `security`, `storage`, `backtest`, `model_version`, `daily_snapshot`,
  `factor_library`, `timing`, `versioning`, `admission`, `cluster`, `_contenthash`.
- **`app/db/schema.py`** — 33-table Postgres-compatible schema (principals, roles,
  artifacts, jobs, workflow_runs, outbox/inbox, factor_candidates, cluster_*...).
- **`app/security/`** — session-token auth (HMAC-SHA256 cookie), password hashing
  (scrypt), RBAC (`resolve_principal_permissions`, `effective_permissions`).
- **`app/storage/`** — COS artifact publisher/materializer, local artifact cache
  (checksum-verified), access guard, registry.
- **`app/adapters/data_access_storage.py`** — DataAccessStorageAdapter (guarded import).
- **`app/worker/`** — job runner, outbox worker, publish pump (in-memory + sqlite).
- **`app/orchestrator.py`** — candidate → evaluation → admission pipeline skeleton.
- **`app/observability/`** — health checks, metrics registry, collectors.
- **`app/api/app.py`** — FastAPI: `/auth/login`, `/auth/logout`, `/auth/me`, `/health`.

## Design rules (hard)

- Platform MUST NOT reimplement quant logic: no RankIC, no second factor lake,
  no duplicate metadata registry. Domain authority stays in domain packages.
- `quant_platform.app.contracts` imports cleanly with zero third-party deps
  (asserted by `tests/test_smoke.py`).
- Platform ↔ domain translation happens in worker/adapters: `Platform DTO ↔
  Domain Native Contract` (e.g. `FactorCandidateManifest` ↔ FA treatment artifacts,
  `BacktestRequest` ↔ vectorbt_qs `BacktestRequest`, model DTOs ↔ modeling artifacts).

## Current gaps (from `docs/GAP_ANALYSIS.md` / `docs/CURRENT_ARCHITECTURE.md`)

- Control Plane (workflow runtime, outbox in production, retry/promotion) — missing.
- Metadata Plane (Postgres registry, factor catalog unified) — missing.
- `research_platform` legacy dual-artifact authority still to migrate.

## Install & test

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/quant_platform.git
cd quant_platform
pip install -e .
python -m pytest tests/ -q    # 31 test files
```

## Related repos

- **All domain packages** — this is their platform-facing boundary
- **platform_web** — the React frontend types mirror `app/contracts/*`
- **data_access** — storage adapter source
