# QRP Phase 0 — Verified findings ledger (2026-08-26)

Authority: repository tree + test output. Sources: baseline runs + subagent audits
(verified against filesystem, not markdown claims).

## Baselines (repo root, .venv/bin/python -m pytest, -q --no-header -p no:cacheprovider)
- quant_evaluator: `332 passed, 2 skipped` (15.8s)
- factor_optimizer: `505 passed` (28s) — cd factor_optimizer
- factor_preprocess: `103 passed, 1 xfailed` (1.5s) — cd factor_preprocess
- factor_assets: `827 passed, 43 skipped` (2.9s) — cd factor_assets
- factor_engine: **BROKEN at collection** (26 errors) — ParamRole→SimpleNamespace
  (task QRP-P0-B1, fix agent dispatched; root cause isolated to an import-time
  mutation during cleaned_operators/__init__ hardening chain)

## Verified facts (file:line)

### COS / object store (data_access)
- `data_access/cos/` — mirror.py, remote.py, s3_duckdb.py (**DuckDB httpfs/S3**, NOT vendor SDK; `cos_uri_to_s3_uri()` remote.py:135; `S3Credentials` :47)
- `data_access/cos/mirror.py:34` — `ASHARE_COS_PREFIX = "cos://qs-cold/clean_data/ashare/lqtp_data"` — **dataset-clean layout, NOT spec §21 /canonical/factor_values/**
- `data_access/read/object_store.py` — `ObjectStore` Protocol (:37) + LocalObjectStore/COSObjectStore (:82/:202)
- `data_access/read/local_disk_policy.py` — `LocalDiskPolicy` (STRICT_REMOTE/HIGH_PERFORMANCE, fail-closed)
- `data_access/__init__.py:61-66` — `get_store()` → `install_cos_runtime(_get_store())`
- No symbol named `create_adapter` in data_access
- **NO production path writes factor values to COS** — materialization is local-only (factor lake `{workspace_data}/factors/lake`, workspace_paths.py:69; lake_publish.py:217). Spec §21 canonical factor-major/feature-block layout: **MISSING**
- **LocalArtifactCache: MISSING (spec §22 gap)**

### Ingestion today (verified)
- Manifest/_READY protocol = **CONTRACT ONLY** (quant_platform/app/contracts/candidate.py — `READY_MARKER_NAME="_READY"`, FactorCandidateManifest per §10.2). No ingestion service/scanner/writer.
- No COS candidate bucket. Today's ingestion = formula gates: FE `compile`/`materialize_from_config` (runtime/engine.py:1041/:2471), mining-manifest validation (api/mining_integration.py:1107/:1230), mining/campaign.py batch, operator_catalog.py:252 admission.
- Transactional outbox **MISSING**; FE service uses in-memory BoundedJobQueue (service/queue.py:96). EventEnvelope = contract only (event_envelope.py:86, 19 event types :48-88).

### HTTP service (infra)
- `factor_engine/service/app.py` — FastAPI, `--port` default **8088** (`service/app.py:1521`, `FACTOR_ENGINE_SERVICE_PORT`), `factor-engine-serve`
- `data_access/service/app.py` — FastAPI port **8765**, `data-access-server`
- `vectorbt_qs/apps/candlestick-patterns/app.py` — vendored upstream
- **Auth: NO argon2/bcrypt/jwt/oauth; plain shared-secret API key only (`_require_service_api_key`, `DATA_ACCESS_API_KEY`)**
- **RBAC + audit log: MISSING (spec §24 = P0 gap)**

### modeling / backtest contracts
- `modeling/dataset.py:26-38` — FeatureSchema, PanelDataset
- `modeling/contracts.py` — PredictionOutputContract, EmbargoSpec, FitFingerprint, ModelOperatorSpec
- `modeling/artifact.py`
- Exact §2 names ModelDatasetRequest/ModelTrainingRequest/ModelArtifactRef NOT in code → PARTIAL
- `vectorbt_qs/contracts/backtest.py` — BacktestRequest (frozen), BacktestArtifact (content-addressed)
- PlatformContractE2E token NOT at HEAD

### P0 dependency status (.venv)
- AVAILABLE: fastapi, starlette, uvicorn, pydantic, httpx, duckdb, polars, sqlglot
- MISSING (also in requirements-production.lock): sqlalchemy, alembic, psycopg, temporalio, cos-python-sdk, jwt
- No node/npm/docker/psql system-wide (env audit)

### FA (verified, 167 lines) — QRP-P6 critical
- Identity solid: FactorIdentity/FactorDefinitionIdentity/FactorCompilerIdentity/FactorValueIdentity (identity/canonical.py:61/103/130/151). EvaluationIdentity MISSING (§8.3).
- Admission/novelty: FactorAdmissionArtifact + SHADOW/SHADOWED_COEXIST; ResidualICNoveltyProducer (adapters/residual_novelty.py:55). No corr>thresh=reject.
- Clustering: LeidenClustering (families.py:418, igraph/leidenalg, fail-closed), ClusterArtifact derived content-hash.
- FactorSet: FactorSetArtifact sole canonical; legacy FactorSet = compat `to_legacy_view()` only — NO dual-truth.
- **GAPS: NO logical_cluster_id / ClusterVersionArtifact / cross-version matching (§14); NO incremental cluster assignment (Stage 11); 8/12 §36 contracts missing (SimilarityFingerprint/Graph, IncrementalClusterAssignment, ClusterVersion, ClusterLineage, FactorLibraryDefinition/Version/Membership, FeatureSetCandidate, PromotionDecision); lifecycle only 6/~21 states; multi-view similarity missing Pearson/quantile_overlap/regime; assembly consumes external composite scores + no capacity.**

### FP/FO (verified, 198 lines)
- FP policy authority resolved: registry/policies.py canonical; contracts/policy.py deprecated compat-only (explicit docstring).
- FittedState (contracts/state.py:114), FeatureBundle (:390), FeatureManifest (:82), FactorProfileArtifact (:50), TransformLineage (treatment_lineage.py:98) exist. **TreatmentRecipe MISSING**.
- hp_filter/STL/bandpass/wavelet family all `admission="OFFLINE_ONLY"`, causal_safe=False (registry/transforms.py:588-607, exact spec rationale); smoothing family causal_safe=True. step_id uniqueness enforced (registry/policies.py:147-149).
- FO: SearchRunner (search/runner.py:881), TrialLedger append-only, purge+embargo (contracts/splits.py:407-506), DataProvider isolation (data_capabilities/data_providers), sealed test only via TestAuthorityBroker (runner.py:1645-1690), multi-fidelity. **Gap: multiple-testing corrections only in quant_evaluator/metrics/multiple_testing.py, not referenced by FO. TreatmentSelectionArtifact never emitted by runner; lacks runner-ups/failed-trials/fit-boundary.**
- 7 second-authority duplicates incl. `TransformStep` name collision (registry/policies.py:32 vs contracts/treatment_lineage.py:74), dual multi-fidelity stacks.

## Open audits (in flight)
- QE: (agent running)
- FP/FO: (agent running)
- FA: (resumed, was partial)
- Report center: (running)
- FE/DA/mining: **partial — 54-line skeleton, NEEDS RESUME**
- Contract: DRAFT (resumed, was empty)
- FE fix: running

## Second-authority risk (from infra audit)
- Two FastAPI HTTP entry points exist (FE 8766, DA 8765) — platform must not add a third.
- COS uses DuckDB/httpfs, not COS vendor SDK — adapter must bridge to that.

Last verified: 2026-08-26T18:37Z — HEAD d58eae08
