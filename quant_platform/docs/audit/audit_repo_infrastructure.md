# Phase-0 Repository Infrastructure Audit

- Date: 2026-08-26
- Git HEAD: `d58eae0821da2c71182e200751af97a577e1a616`
- Scope: READ-ONLY repo-level infrastructure audit for QRP (Quant Research Platform). Source untouched; only this report file written.
- Status: COMPLETE

Legend for §11/§12: EXISTS = present and usable; PARTIAL = present but does not meet spec; MISSING = absent.

---

## 1. Top-level layout snapshot

Confirmed top-level entries (repo root `/home/sunhaiwei/quant_projects`):

- Domain packages (each has its own `pyproject.toml`): `data_access`, `factor_assets`, `factor_engine`, `factor_optimizer`, `factor_preprocess`, `quant_evaluator`, `riskfolio_qs`.
- Root-umbrella packages: `modeling`, `vectorbt_qs`, `evidence`, `alphaprobe`, `lightgbm_qs`, `research_platform`.
- Build: `pyproject.toml`, `_build_backend.py`, `MANIFEST.in`, `build/`, `quant_projects.egg-info`, `Dockerfile`.
- Config/meta: `config/` (gates.json), `CLAUDE.md`, `.claude/`, `.github/`, `.gitignore`, `.pre-commit-config.yaml`, `.data_access_allowlist.yaml`, `ownership.yaml`, several `.r2_*_manifest.yaml`.
- Runtime dirs: `data/`, `runs/`, `jobs/`, `evidence/`, `weekly_backtest_output/`, `量化研究与生产报告中心/`, `scripts/`, `tests/`.
- VCS/venv: `.git/`, `.venv/` (the active interpreter is `.venv/bin/python`), `__pycache__`, `.pytest_cache`.
- Spec: `QUANT_RESEARCH_PLATFORM_MASTER_IMPLEMENTATION_SPEC_20260826.md` (untracked, new).

**Confirmations requested by coordinator:**
- `platform/` — DOES exist, but only as the directory created by THIS audit (audit report + docs/audit). There was NO `platform/` source tree before this report was written. **NO `platform_web/` anywhere.** `find . -name "platform_web"` → none.
- `factory/` — NO top-level `factory/`. Only `tests/factory/` exists (a pytest test dir: `tests/factory/__init__.py`, `operator_obligations.py`). No product `factory/` package.
- Both `platform/` and `platform_web/` and `factory/` are free names for QRP — no collision.

`git status` shows a dirty tree: 6 modified files under `lightgbm_qs/outputs/` and 4 untracked files under `lightgbm_qs/scripts/`, plus the untracked spec. Read-only audit does not touch these.

---

## 2. pyproject.toml + lock

### 2.1 Root `pyproject.toml` (184 lines)
- `[build-system]`: `setuptools>=68`, `wheel`, **custom `build-backend = "_build_backend"`** with `backend-path = ["."]` (root `_build_backend.py` is the build backend).
- `[project] name = "quant-projects"`, version `0.4.0`, `requires-python = ">=3.10"`.
- Top-level `dependencies`: PyYAML, numpy, pandas, packaging, pyarrow, scipy, duckdb, sqlglot, psutil, threadpoolctl — these are the umbrella wheel's hard deps.
- Optional extras: `pandas, accel, talib, polars, modin, backtest, data, performance, service, validation, client, clickhouse, sql, dev, full, vectorbt`.
  - **`service` extra = `["fastapi>=0.110", "uvicorn[standard]>=0.27", "pydantic>=2.0"]`** — this is the current HTTP stack surface.
  - `vectorbt` extra includes `"cos-python-sdk-v5>=1.9.0"`.
- Scripts/console entries (`[project.scripts]`): `factor-engine-serve`, `data-access-server`, `data-access-quality`, `vectorbt-qs`.
- **Package topology:** root wheel `[tool.setuptools.packages.find]` includes the *domain packages by source path* (`factor_engine*`, `data_access.*`, `vectorbt_qs.*`, `modeling*`, `evidence*`) — so the root install is an **umbrella monolith** that also ships the domain packages. Comment at `pyproject.toml:66-71` explicitly states R45 rule: source authority is `factor_engine.*` ONLY; historical root packages were consolidated into `factor_engine/`. The root wheel EXCLUDES the separately-built domain packages (factor_preprocess/factor_optimizer/factor_assets/quant_evaluator are NOT in the find include — they are separate distributions, see §3).
- `[tool.pytest.ini_options]` in root pyproject (testpaths=`tests`, markers incl. integration/slow/modin/perf). No pytest.ini file found; pytest config lives in pyproject.

### 2.2 `requirements-production.lock` (90 lines)
- R46 real-resolver lock, `lock_digest` header, produced under CPython 3.12.3 x86_64-linux-gnu; production target declared python==3.10.12 (with a NETWORK note that hashes MUST be re-resolved under 3.10 before production).
- Internal monorepo dists pinned by name but **built fresh from own pyproject** by `scripts/fresh_wheel_matrix.sh` (hashes NOT baked): `data-access==0.10.2`, `factor-engine==0.3.1`, `quant-evaluator==0.0.1a1`, `factor-optimizer==0.1.0`, `factor-assets==0.1.0`, `factor-preprocess==0.1.0`.
- Third-party key pins (exact): **fastapi 0.139.2, starlette 1.6.0, uvicorn 0.52.4, pydantic 2.13.4, pydantic-core 2.46.4, anyio 4.14.2, h11 0.16.0, duckdb 1.5.4, polars 1.42.1, numpy 2.2.6, pandas 2.3.3, sqlglot 30.17.0, numba 0.67.0, psutil 7.2.2, pyarrow 25.0.0, pyyaml 6.0.3, statsmodels 0.14.6, llvmlite 0.49.0, urllib3 2.7.0**.
- **ABSENT from lock (by grep):** sqlalchemy, alembic, psycopg/psycopg2, temporalio, cos-python-sdk-v5, jwt/pyjwt, bcrypt, argon2, redis. The Web surface is FastAPI+Starlette+Uvicorn only; **no DB ORM/migrations, no temporal, no COS SDK, no auth crypto libs are locked.** This is the biggest infra gap for QRP-P1 (spec §19 PostgreSQL, §47 migrations, §46 auth).

---

## 3. Packaging per package

Each domain package has its **own independent `pyproject.toml`** — install topology is a **multi-package monorepo, NOT one monolith**:

| Package | name/version | build backend | Deps (base) | Circular-dep guards |
|---|---|---|---|---|
| `data_access/` | data-access 0.10.2 | setuptools | numpy/pandas/pyarrow/duckdb | flat `dataaccess/` source root mapped to `data_access` dotted name |
| `factor_engine/` | factor-engine 0.3.1 | setuptools | PyYAML/numpy/pandas/packaging/pyarrow/scipy; `service` extra adds fastapi+uvicorn+pydantic | self-contained |
| `factor_preprocess/` | factor-preprocess 0.1.0 | setuptools | numpy/pandas/scipy/statsmodels/PyWavelets | standalone |
| `factor_optimizer/` | factor-optimizer 0.1.0 | **hatchling** | PyYAML/numpy; opt extras `factor_engine`, `quant_evaluator` | optional adapters only (no hard circular dep) |
| `factor_assets/` | factor_assets 0.1.0 | setuptools | typing-extensions/dataclasses-json/numpy/scipy; opt extras `quant_evaluator/factor_engine/data_access/adapters*` | optional adapters |
| `quant_evaluator/` | quant_evaluator 0.0.1a1 | setuptools | numpy/scipy/pandas/psutil | standalone |
| `riskfolio_qs/` | riskfolio-qs 0.3.0 | setuptools | numpy<2/pandas<3/pyarrow/cvxpy/exchange-calendars/data_access | depends on data_access |

- **Egg-info present:** root `quant_projects.egg-info`; package-level `factor_assets.egg-info`, `factor_preprocess.egg-info`, `quant_evaluator.egg-info`. Confirms root wheel + these packages were built/installed at some point.
- **Circular-dep safety for platform:** each domain package declares adapters to sibling packages as *optional extras* (`factor_engine`, `quant_evaluator`, `data_access`), never hard deps. The platform must therefore depend on these packages via extras / namespace only, and must NOT add a hard dep on `factor_engine` from `factor_assets` (already avoided). Platform can be a new top-level package (e.g. `platform/`) with its own pyproject depending on the named distributions; no cycle is forced.
- `factor_engine/pyproject.toml` ships a flat `py-modules` list (`logging_utils`, `pit_contract`, `stateful_contract`, `stateful_runtime`, `workspace_paths`) plus `cleaned_operators.*` subpackages — its wheel is the full engine.
- Note: `riskfolio_qs/pyproject.toml` depends on `data_access>=0.1.0` as a hard dep; `vectorbt_qs/requirements.txt` (435 B) is a compatibility profile (numpy<2, pandas<3, numba<0.61, cos-python-sdk-v5 included) — vectorbt_qs is NOT built via its own pyproject (none found), it ships inside the root umbrella wheel.

---

## 4. Dockerfile (root)

`Dockerfile` (root):
- `FROM python:3.10-slim`; env `PYTHONUNBUFFERED=1`, `FACTOR_ENGINE_SERVICE_PORT=8766`, `PYTHONPATH=/opt/factor_engine`.
- Creates user `factorengine` (uid 10002), `/opt/factor_engine`, `/data/quant_workspace`.
- `apt-get install gcc g++ gfortran libopenblas-dev`.
- `COPY . .` then `pip install --no-cache-dir '.[full,performance,service]'`.
- Runs as non-root, `EXPOSE 8766`, CMD `factor-engine-serve --host 0.0.0.0 --port 8766`.
- **Assessment:** image is the *FactorEngine HTTP service* only (not a platform image). Builds the root umbrella wheel with full/perf/service extras. No platform API/web container, no postgres, no worker. QRP Phase 1 deployment (§51 Docker Compose) will need new platform-api/platform-web/worker/postgres images — not present.

---

## 5. CI — `.github/`

- Single workflow: `.github/workflows/ci.yml` (14,184 B), name "CI" — R46 P0-AB real GitHub Actions CI.
- Triggers: push to `main`, pull_request, workflow_dispatch; `concurrency` group + cancel-in-progress; `PYTHON_VERSION=3.12`.
- **Jobs are verifier-backed structural gates** (`gate_runner` + `find_spec` from `scripts/`), each on a fresh clone installing lock deps third-party-only via `bash scripts/ci_smoke.sh`:
  `source-authority`, `supply-chain`, `evidence-current`, `submodule-reachability`, `fresh-wheel-matrix`, and a matrix of `gate (...)` jobs: NUMERICAL_ORACLE, PROPERTY, CROSS_PACKAGE, SERIALIZATION, CHECKPOINT_RESUME, UNIT, LEAKAGE, PIT, DETERMINISM, 1K_SCALE, 10K_SCALE, 100K_SCALE, plus `ashare-semantic-golden` / `ashare-real-data-shadow` (optional).
- **Notably ABSENT from CI vs spec §52:** no ruff/formatter job, no mypy/pyright, no frontend lint/tsc, no migration test, no API OpenAPI-compat check, no `pip install --require-hashes -r requirements-production.lock` full-lock path (uses `scripts/ci_smoke.sh` which is third-party-only). Fresh-install is only the wheel matrix, not a full clean-venv packaging test per §52.

---

## 6. Web / API / auth surface

HTTP layer hosts (grep `fastapi|starlette|flask|django` in product code, excluding venv/build/tests):
- **`factor_engine/service/app.py`** — the primary FastAPI HTTP service. `import fastapi` at line 38 (lazy, raises ImportError if `factor-engine[service]` not installed). R21 production service: typed Pydantic models `extra="forbid"`, `create_app`, `STORE`, `EXECUTOR`, `JobStore/JobRecord/JobStatus/JobType`, `_require_service_api_key`, `_authorized_config_path`. Runs on port 8766 via `factor-engine-serve`.
- **`data_access/service/app.py`** — FastAPI app (imports `from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request, Response` line 41). Runs via `data-access-server`, compose port 8765.
- `vectorbt_qs/vectorbt/apps/candlestick-patterns/app.py` — vendored upstream, not product API.
- `factor_engine/scripts/generate_production_lock.py` — dev script, not a service.

**Auth primitives** (grep argon2|bcrypt|jwt|oauth|session across repo, excluding `.venv`/`build`/`tests`/`.git`/`node_modules`): hits are all in `alphaprobe/.../gplearn/`, vendored `vectorbt_qs/vectorbt/`, and `scripts/*.py` (audit/build scripts using the words `session`). **No real auth crypto library (argon2/bcrypt/jwt) is imported by any product package.** The existing auth is a **plain API-key model** (`_require_service_api_key`, `DATA_ACCESS_API_KEY`) — i.e., a shared-secret header, not RBAC/sessions/JWT. Spec §24 (users/roles/permissions/audit) and §24.5 (authentication) are MISSING.

## 7. Storage / COS / local cache

- `data_access/cos/` is the COS module: `mirror.py`, `remote.py`, `s3_duckdb.py`, `serving.py` (+ `__init__.py`). `data_access/cos/remote.py` supports mirror / remote (DuckDB httpfs S3 read) / auto modes keyed by `DATA_ACCESS_COS_READ_MODE`, using `COS_SECRET_ID/COS_SECRET_KEY/COS_ENDPOINT`. `data_access/cos_contract.py` defines COS dataset contracts.
- `data_access/read/object_store.py` — **ObjectStore abstraction** (R44-P0): `ObjectStore` Protocol (head/list/range/open_reader/put/multipart/delete) + `LocalObjectStore` (atomic `os.replace`), `COSObjectStore` (thin adapter over `cos.remote`, multipart no-ops to `put_object`), `NullObjectStore` (policy-gating tests). Object keys reject `..`/absolute escapes.
- `data_access/read/local_disk_policy.py` — **LocalDiskPolicy** (R44-P0 zero-local-disk governance): `STRICT_REMOTE`/`HIGH_PERFORMANCE`, `track_local_persistent_bytes_written`, `assert_no_local_persistent_write` (fail-closed), `spill_policy_for`; authoritative from env, contextvar-based counter.
- `data_access/registry/layout_policy.py` — BucketHashRegistry / BucketLayoutPolicy (sha256, bucket_count validation; rejects MD5 for production identity).
- Exposure provider: `data_access/__init__.py` exports **`get_store()`** (line ~68) = `install_cos_runtime(_get_store())`, the hardened factory that installs COS semantics; `_store_module.get_store` is patched to the same factory so callers cannot bypass COS contracts by import path (`data_access/__init__.py:62-66`). `data_access/core/storage.py` + `core/__init__.py` expose `StorageBackend`/`StorageSpec`/`resolve_storage_for_dataset`/`to_s3_uri`. There is NO symbol literally named `create_adapter` — the exposure factory is `get_store()`.
- **`LocalArtifactCache` (spec §22) does NOT exist anywhere** — grep returned no class. Existing is `LocalObjectStore` (a byte object store), which is a different concept from the spec's cached LRU+checksum artifact cache. **MISSING**.
- No `cos-python-sdk-v5` is importable in `.venv` (see §13); COS access is via DuckDB httpfs/S3, not the vendor SDK.

## 8. modeling/ + vectorbt_qs contract boundary

`modeling/` exists as a root umbrella package (24 modules + `learners/`):
- `modeling/contracts.py` — rich typed contracts (TradingTimestamp, ModelExecutionClass, TimingKind, RichModelTiming, SampleAdequacyContract, EmbargoSpec, ApplicationWindow, LabelContract, DecisionClock, PredictionOutputContract, PredictionBatch, RegimeMetadata, FitFingerprint, ParameterSearchPolicy, ModelOperatorSpec, LegacyLocalPredictive...).
- `modeling/dataset.py` — `FeatureSchema` (frozen, ordered feature contract) + `PanelDataset` (pooled `(stock,date)`), date-authoritative split telemetry.
- `modeling/artifact.py` — `PredictionContext` + artifact/`TimestampContractError` machinery; `ledger.py`, `registry.py`, `model_catalog.py`, `artifact.py`.
- Other: `trainer.py`, `trainer_governance.py`, `split.py`, `leakage_guard.py`, `walk_forward.py`, `diagnostics.py`, `selection.py`, `evaluation.py`, `predictor.py`, `evidence.py`, `hyperparams.py`, `monitoring.py`, `dsl_bridge.py`, `sample_policy.py`, `benchmark.py`, `timing.py`, `model_semantic_registry.py`, `presets.py`, `legacy.py`, `learners/{mixture_of_experts, regime, elastic_net, pcr, pls, base}`.

**Spec §2 (lines 222-224) REQUIRES EXACT contract names `ModelDatasetRequest`, `ModelTrainingRequest`, `ModelArtifactRef`.** Grep for those exact class names in all product packages: **found only in the SPEC, not in code.** Current modeling has `FeatureSchema`/`PanelDataset`/`PredictionContract`/`ModelOperatorSpec`/`Artifact...`, but not the three named spec contracts → **PARTIAL**.

`vectorbt_qs/contracts/` is a dedicated contract boundary: `backtest.py`, `execution.py`, `ledger.py`, `orders.py`, `positions.py`, `signal.py`, `trades.py`, `invariants.py`, `reference_simulator.py`.
- `vectorbt_qs/contracts/backtest.py` — `BacktestRequest` (immutable, frozen: strategy_ref/snapshot_ref/universe_ref/policy/execution/cost/capacity/start/end/init_cash) + **`BacktestArtifact`** (immutable content-addressed result holding derived ledgers: positions/orders/trades/cash/NAV/returns/turnover/cost/unfilled/shortfall). This is the spec §3 `BacktestRequest`/`BacktestArtifactRef` contract (PARTIAL: `BacktestArtifact` exists, ref-name differs).
- `PlatformContractE2E`: grep for this exact token returned no product hit — a prior-work boundary under that name is NOT present in current code (NOT_VERIFIED/absent at HEAD d58eae08). The E2E contract in place is `vectorbt_qs/contracts/backtest.py`.

## 9. Testing infrastructure

- `tests/` root tree is large (50+ suites): `api, audit, backend, backend_parity, backend_sql, cache, da_identity, evidence, factor_recipes, factory, fields, fixtures, fundamental, group, integration, intraday, ir, label, market, mining, modeling, multibackend, multibackend_integration, operator_contracts, operator_golden, operators, perf, performance, pit, planner, planning, q_backend, r25/r27/r30-r45, relation, routing_authority, runtime, scripts, service, source, source_authority, stateful, storage, unit, util, validation`.
- **conftest chain:** `conftest.py` (root), `tests/conftest.py` (import-path wiring: inserts `_QUANT_ROOT` then `_FE_ROOT` so factor_engine shadows stale root copies; thaws OperatorRegistry at session start; autouse `_reset_production_env_leaks`), plus package-level `factor_optimizer/conftest.py`, `factor_preprocess/conftest.py`.
- pytest config lives in **root `pyproject.toml`** (`[tool.pytest.ini_options]` testpaths/tests, addopts=-ra, markers). No `pytest.ini`.
- Dedicated R-phases have their own suites (`r25`...`r45`) — consistent with the R-loop workflow. Gate verifiers in `scripts/gate_runner.py` drive the R-series gates.

## 10. Deployment

- **docker-compose**: only `data_access/docker-compose.yml` (build `.`, port 8765, env `DATA_ACCESS_API_KEY`, `QUANT_PRODUCTION_MODE=1`, COS cache root, duckdb tmp dir; read_only + tmpfs + no-new-privileges). **No root/platform compose.**
- **`.env.example`**: NONE.
- **`deployment/`**: NONE.
- **`migrations/`**: NONE. Spec §47 (Alembic/equivalent migration tool) is MISSING.
- Only root `Dockerfile` (FactorEngine service). QRP Phase 1 must author platform-api/platform-web/postgres/worker compose + migration framework from scratch.

---

## 11. Mapping to spec sections (EXISTS / PARTIAL / MISSING)

| Spec section | Requirement | Status | Evidence |
|---|---|---|---|
| §0.1 local tree as dev source | works from repo tree | EXISTS | root/package pyprojects; conftest path insert |
| §3 3-plane separation | web/API/control separated | PARTIAL | `factor_engine/service` + `data_access/service` only; no web plane, no control plane |
| §5 dependency rules (per-§) | independent packages, no monolith | EXISTS | each domain pkg own pyproject; platform can deps via extras (see §3) |
| §5 Model-boundary contracts (exact `ModelDatasetRequest`/`ModelTrainingRequest`/`ModelArtifactRef`) | reserve contracts | PARTIAL | only FeatureSchema/PanelDataset/PredictionContract in modeling; exact names MISSING |
| §7 ArtifactRef immutable + 2-phase commit | artifacts immutable | PARTIAL | `BacktestArtifact` frozen; `modeling/artifact.py` artifacts; no 2-phase artifact registry yet |
| §19 PostgreSQL schema | postgres DB | MISSING | no sqlalchemy/alembic/psycopg in lock or .venv |
| §22 local disk policy + LocalArtifactCache | ObjectStore + LocalArtifactCache | PARTIAL | ObjectStore EXISTS (`data_access/read/object_store.py`); **LocalArtifactCache MISSING**; `LocalDiskPolicy` EXISTS (`data_access/read/local_disk_policy.py`) |
| §23 API design | platform API | PARTIAL | two existing service apps (FE port 8766, DA 8765); no unified platform API |
| §24 RBAC + audit + auth (§24.5) | users/roles/permissions/audit/auth | MISSING | only API-key auth; no RBAC, no audit_log store, no JWT/session |
| §46 security | secrets/CSRF/rate-limit/XSS/etc. | PARTIAL | `DATA_ACCESS_API_KEY`, read_only compose; no rate-limit/CSRF/JWT/audit |
| §47 DB migrations (alembic) | migrations | MISSING | no alembic, no migrations/ |
| §51 deployment (compose) | platform-api/web/postgres/worker | PARTIAL | only DA compose exists; root image = factor-engine only |
| §52 CI | ruff/mypy/pytest/frontend/migration/API compat/clean-install | PARTIAL | gate_runner verifiers EXIST in ci.yml; missing formatter/type-check/frontend/OpenAPI-compat/clean-install-wire |
| §53 integration test matrix | happy-path/duplicate/event-lost/crash etc. | PARTIAL | `tests/integration/` + `tests/multibackend_integration/` exist; not scoped to platform/outbox/worker |
| §54 performance/scale | scale tests | PARTIAL | CI has 1K/10K/100K_SCALE gates; `tests/perf`,`tests/performance` |
| §56 Phase 0 | repo audit + contract freeze | IN PROGRESS | this report |

---

## 12. GAP rows (Requirement | Current | Gap | Action | Reuse/Modify/Add | Priority)

| # | Requirement | Current (files) | Gap | Action | Reuse/Modify/Add | Priority |
|---|---|---|---|---|---|---|
| 1 | QRP platform skeleton package | no `platform/` (only this audit dir) | empty name | scaffold `platform/` pyproject + app package | Add | P0 |
| 2 | PostgreSQL metadata schema (§19) | no sqlalchemy/alembic/psycopg importable | DB layer absent | add deps + schema | Add | P0 |
| 3 | Alembic migrations (§47) | no migrations/ | no migration framework | scaffold `platform/migrations` | Add | P0 |
| 4 | Platform API (§23) | `factor_engine/service/app.py`, `data_access/service/app.py` (FastAPI) | no unified API | extend/build on FastAPI+Starlette present | Modify (reuse FE/DA service patterns) | P0 |
| 5 | RBAC + audit log (§24) | only API-key auth | no RBAC/audit | add auth/RBAC middleware | Add | P0 |
| 6 | Web plane (§3) | none (no node/npm) | frontend absent | defer or add minimal web | Add | P1 |
| 7 | Temporal/workflow control (§51) | no temporalio importable | no control plane | add temporalio OR reuse task queue | Add | P1 |
| 8 | COS SDK importable | none (`cos-python-sdk-v5` in vectorbt extra only, not importable) | COS via DuckDB httpfs only | decide sdk-vs-httpfs; add import if needed | Reuse (DuckDB httpfs) / Add sdk | P1 |
| 9 | `LocalArtifactCache` (§22) | only `LocalObjectStore` | spec cache absent | implement cache over ObjectStore | Add | P1 |
| 10 | Spec §5 ModelDataset/ModelTraining/ModelArtifactRef contracts | modeling contracts partial | exact names absent | add reserved contracts in modeling | Modify (add) | P0 (contract freeze) |
| 11 | Docker compose platform stack (§51) | only data_access compose | no root compose | add root compose | Add | P1 |
| 12 | Clean-install packaging CI (§52) | `scripts/ci_smoke.sh` third-party-only | no `--require-hashes` full lock + clean venv | extend CI | Modify | P1 |
| 13 | SSO/scale-engines | CI 1K/10K/100K gates exist | platform-scale tests absent | extend | Modify | P2 |

---

## 13. Environment audit (Phase-0 env, confirmed + extended)

Runtime tools:
| tool | status |
|---|---|
| python3 | `/usr/bin/python3` (interpreter 3.12.3) |
| `.venv/bin/python` | Python **3.12.3** (report uses this for imports) |
| pip | `/home/sunhaiwei/.local/bin/pip` (reported 26.2.1) |
| git | 2.43 |
| gcc / g++ | 13.3 (both present) |
| make | present |
| node / npm | **MISSING** |
| docker / docker-compose | **MISSING** |
| psql / redis-cli | **MISSING** |

`.venv` import test (`./.venv/bin/python -c "import ..."`), AVAILABLE / MISSING for QRP-P1:

| module | AVAILABLE/MISSING | version |
|---|---|---|
| fastapi | AVAILABLE | 0.139.2 |
| starlette | AVAILABLE | 1.6.0 |
| uvicorn | AVAILABLE | 0.52.4 |
| pydantic | AVAILABLE | 2.13.4 |
| httpx | AVAILABLE | 0.28.1 |
| sqlglot | AVAILABLE | 30.17.0 |
| duckdb | AVAILABLE | 1.5.4 |
| polars | AVAILABLE | 1.42.1 |
| sqlalchemy | **MISSING** | — |
| alembic | **MISSING** | — |
| psycopg / psycopg2 | **MISSING** | — |
| cos_python_sdk_v5 / qcloud_cos | **MISSING** | — |
| jwt | **MISSING** | — |
| temporalio | **MISSING** | — |

Net: FastAPI/Starlette/Uvicorn/pydantic (web layer) and duckdb/polars/sqlglot (compute layer) are ready in `.venv`. The entire **persistence/control/auth stack (§19/§46/§47/§51) — SQLAlchemy, Alembic, psycopg, temporalio, JWT — is MISSING** and must be added for QRP-P1. No installs were performed (read-only).

---

## Appendix: key file:line anchors
- Root pyproject (184 lines): `[project]` at L, `service` extra at `pyproject.toml:59`, package `find`/R45 comment at `pyproject.toml:121-171`, `[tool.pytest]` at `pyproject.toml:175-184`.
- Lock: `requirements-production.lock:57` fastapi, `:73-74` pydantic, `:82` starlette, `:90` uvicorn.
- Dockerfile: `Dockerfile:1` FROM python:3.10-slim; `Dockerfile:28-29` CMD factor-engine-serve.
- CI: `.github/workflows/ci.yml` — jobs `source-authority` L84, `gate` matrix, `scripts/ci_smoke.sh` at L88.
- FE service: `factor_engine/service/app.py:38` fastapi import, `:1275` ImportError message; DA service `data_access/service/app.py:41-42` fastapi import.
- COS: `data_access/cos/remote.py:1-9`, `data_access/read/object_store.py:1-27`, `data_access/read/local_disk_policy.py:43-54`, `data_access/registry/layout_policy.py:1-12`, `data_access/__init__.py:61-66` (get_store/install_cos_runtime), `data_access/core/__init__.py:1-56`.
- Modeling: `modeling/dataset.py:26-27` FeatureSchema, `:38` PanelDataset; `modeling/contracts.py` (per class, see §8); `modeling/artifact.py:120-175`.
- vbt contract: `vectorbt_qs/contracts/backtest.py:29` BacktestRequest, `:57` BacktestArtifact.
- tests: `tests/conftest.py:1-40`, root `conftest.py:1-14`; pytest config at root pyproject L175.
- Compose: `data_access/docker-compose.yml:1-22`.
