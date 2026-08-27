# Phase-0 Audit: factor_engine + data_access + mining

- date: 2026-08-26
- git rev-parse HEAD: `d58eae0821da2c71182e200751af97a577e1a616`
- repo root: `/home/sunhaiwei/quant_projects`
- scope: `factor_engine/` (api, service, export, storage, cache, mining, runtime, backend), `data_access/` (top-level, underscore), `tests/mining/`, `factor_engine/tests/mining/`, `quant_platform/app/contracts/`, `scripts/`
- status: COMPLETE
- method: read-only grep/sed/find + targeted import smoke (imports blocked by a `platform` module name-shadowing bug in the repo `.venv`; module existence verified by source inspection, full import NOT_RUN)
- master spec: `QUANT_RESEARCH_PLATFORM_MASTER_IMPLEMENTATION_SPEC_20260826.md`

## 0. Legend
- EXISTS = verified in file at `file:line` (or section)
- PARTIAL = exists but incomplete vs spec requirement
- MISSING = not found anywhere in scope
- NOT_VERIFIED = exists as a contract/DTO but not yet wired to runtime (this pass did not trace every call site)

## 1. factor_engine/ adapter surface (spec §37: validate / compile / resolve fields / materialize / incremental / batch / artifact metadata)

All §37 adapter operations EXIST and are served both as a native API and a FastAPI HTTP service.

### 1.1 api/ — definition & validation
- `factor_engine/api/factor.py:76` `class Factor` (frozen dataclass; `:113` `__post_init__` name validation). Entry: `factor_engine/api/__init__.py:38` `__all__`; dynamic operator export `factor_engine/api/__init__.py:52` `__getattr__`.
- `factor_engine/api/operator_registry.py:12` `build_dsl_allowlist()`, `:81` `build_authoring_allowlist()`, `:97` `build_research_mining_allowlist()`, `:109` `build_production_mining_allowlist()`.
- `factor_engine/api/mining_integration.py:1107` `validate_manifest_for_execution()`; `:1230` `get_mining_operators_from_manifest()` (mining manifest gate); `:1324` search-space manifest schema `"factor_engine.mining_search_space.v1"`.
- Field resolution: `factor_engine/runtime/engine.py` `Analyzer.lower` → `Lowerer.to_logical_plan` → `Optimizer.optimize` (engine.py:1041-1110); referenced columns resolved in compile `engine.py:2093` (`analyses` → `referenced_columns`).

### 1.2 service/ — execution/jobs (FastAPI)
- `factor_engine/service/app.py:1271` `create_app()`; routes: `:1328` GET `/health`, `:1336` `/livez`, `:1341` `/readyz`, `:1349` `/metrics`, `:1354` GET `/factor-engine/operators`, `:1359` POST `/factor-engine/validate-spec`, `:1365` POST `/factor-engine/research/compute`, `:1381` POST `/factor-engine/production/compute`, `:1402` POST `/factor-engine/production/materialize`, `:1418` POST `/factor-engine/jobs/compute`, `:1436` POST `/factor-engine/jobs/materialize`, `:1452` GET `/factor-engine/jobs/{run_id}`, `:1462` GET `/factor-engine/jobs/{run_id}/artifacts`, `:1471` POST cancel, `:1484` POST retry.
- Default port is **8088** (`app.py:1521`, env `FACTOR_ENGINE_SERVICE_PORT`) — NOT 8766 (8766 is not referenced anywhere in FE; the coordinator note about "FastAPI 8766" is not confirmed by code — the FE service and DA service use 8088/8765).
- `factor_engine/service/jobstore.py:173` `JobStore` (SQLite `:182-240`), `:71` `JobRecord`, `:45` `JobStatus`, `:58` `JobPhase`. `factor_engine/service/queue.py:96` `BoundedJobQueue` (in-memory FIFO, not outbox-persistent).
- `factor_engine/service/models.py:178` `ComputeRequest` (strict Pydantic; fields incl. `formula/dsl/market/calendar/decision_time_policy/run_mode/backend/universe`, `:183+`); `:78` `WriteTarget` enum `local|staging|production|clickhouse`.
- Materialize path: `app.py:951` `_execute_materialize()` → `FactorEngine.materialize_from_config` (`engine.py:2471`).
- Artifact metadata return: `app.py:1040` `_summarize_result()` (rows/factor/keys), `app.py:1068` `job.artifacts[key]`, `app.py:1462` `/jobs/{run_id}/artifacts`; job-level artifact map only. R30 `data_access/r30/artifact_meta.py:18` `FactorArtifactMetadata` (sidecar schema, `:85-89`) is the richer artifact-metadata model.

### 1.3 export/ — artifact metadata serialization
- `factor_engine/export/serializers.py:35` `ContractSerializer` (`to_dict/to_json/to_parquet/to_feather`); `:178-256` `serialize_to_json/parquet/feather/batch`.
- `factor_engine/export/converters.py:30` `DataFrameConverter` (to_pandas/polars/arrow, infer/validate schema).
- `factor_engine/export/importers.py:31` `ContractImporter` (from_dict/json/parquet/feather).
- `factor_engine/export/versioning.py:26` `SchemaVersion`, `:76` `VersionRegistry` (+ migrations, `:363` `compute_schema_hash`). Versioned-contract persistence, not yet a full ArtifactRef publish registry.

### 1.4 storage/ — materialize (batch / incremental), factor lake
- Real materializer: `factor_engine/storage/materialize/materializer.py:242` `ParquetMaterializer` (`:332` `materialize`, `:505` `materialize_block` Arrow path, `:1060` `_materialize_long_df`, `:1540` `commit_deferred_materialization`, `:1866` `_append_tombstones`, `:1907` `_upsert_to_data_access_staging`). `factor_engine/storage/materializer.py` and `storage/lake_publish.py` are **compat shims** (sys.modules redirect).
- Lake publish (staging → published, human approval gate): `factor_engine/storage/materialize/lake_publish.py:22` `is_publish_approved`, `:89` `sync_local_factor_to_staging`, `:148` `advance_published_watermark`, `:217` `publish_factor_lake`.
- Catalog + watermarks: `factor_engine/storage/catalog.py:544` `FactorCatalog` (SQLite; `:342` `factor_watermark` DDL; `:921` `batch_transaction`).
- Batch multi-factor: `factor_engine/runtime/engine.py:1722` `materialize_many_from_config`, `:1778` `materialize_many_from_config_parallel`, `:2098` `materialize_many_fast`, `:2324` `materialize_matrix`. Incremental: `:2409` `materialize_incremental_from_event`, `:2448` `materialize_incremental_many_from_config`, `:2520` `materialize_incremental_from_config`, `:3779` `materialize_incremental`.
- Result store: `factor_engine/storage/result_store.py:37` `PandasResultStore` (+ `PolarsResultStore` `:14` in `storage/__init__.py`).
- Lake root: `factor_engine/util/workspace_paths.py:69` `default_factor_lake_root()` → `FACTOR_LAKE_ROOT` or `{workspace_data}/factors/lake`.
- Cache hit: `engine.py:1279` `_build_cache` → `CacheManager`/`PersistentPlanCache` (`storage/cache.py`); `materialize_block` readback `materializer.py:1030`.
- Write-target enforcement: `factor_engine/storage/materialize/write_targets.py:43` `LocalParquetWriteTarget` (production-direct forbidden `:97-108`), `:142` `StagingWriteTarget` (data_access upsert), `:204` `ClickHouseWriteTarget`.

### 1.5 cache/
- `factor_engine/cache/__init__.py` — `CachePolicy`, `ColumnCacheScope`/`column_cache_scope`, `ExpressionCache`, `CacheHitStats`/`CacheLayer`, `PanelCache`, `ExecutionCacheSession`; unified interface `unified_cache.py` `UnifiedCache`, `cache_manager.py:38` `GlobalCacheManager` (singleton), `tiered_cache.py` `TieredCache`.

## 2. Who sets compile results / capability certification / required lookback / backend plan / formula identity

Single-authority stack (verified, no production duplicates):

- **Required lookback**: `factor_engine/ir/analyzer.py` (`AnalysisResult.lookback`); history requirements registered in `analyzer.py:275+` (e.g. `_compound_history` at `:290`); entry `Analyzer().lower(expr)` returns `ir/lookback/has_ts_op/has_cs_op/referenced_columns` (per `factor_engine/ir/README.md:7`). Lookback is returned per-factor from compile (`engine.py:2093`).
- **Capability certification**: single registry `factor_engine/backend/operator_capability.py` (`CAPABILITY_REGISTRY_VERSION = "v2.1.0"` `:52`; `BackendCapabilityRegistry`; `:277` `enumerate_physical_inventory`; `:416` `resolve_canonical`; `:422` polars-native checks). `factor_engine/backend/capability_registry.py:1-9` is a **deprecated facade** re-exporting the same classes (warns DeprecationWarning) — authority is NOT split. Per-operator production status: `factor_engine/backend/backend_certification.py:17` `BackendCertification` (pandas/polars/duckdb_sql `BackendStatus` `:13`); `factor_engine/api/lqtp_capabilities.py:11` `build_lqtp_capability_manifest()` aggregates surface/production_allowed/production_backends per canonical (state model `recognized/parseable/executable/production_certified` `:51-53`).
- **Compile gate**: `factor_engine/runtime/engine.py:1041` `FactorEngine.compile` → production gates `assert_production_plan_ops`, `assert_no_unapproved_map_groups_in_production`, `assert_production_fastpath_plan` (`:1090-1107`), PIT audit `pit_audit.assert_pit_safe` (`:1074`); precompiled plan re-certification `:738` `_assert_production_plan_gates`.
- **Backend plan**: `factor_engine/planning/hybrid_execution_planner.py` produces `PhysicalRegionPlan` with `TransferEdge`; `engine.py:73` `_assert_backend_plan_authority`; `engine.py:251` `_execute_ready_single_region_plan` (region residency enforced, `:354` TransferEdge). `ExecutionKind` in `operator_capability.py` + `factor_engine/polars_backend_kind.py`.
- **Formula identity**: `factor_engine/runtime/factor_identity.py:898` `ExecutionSemanticIdentityV2` (frozen; `:155` `decision_time_policy`); HTTP construction `service/app.py:538` `_build_execution_semantic_identity` (ir_hash/operator_contract_hash/field_contract_hash/source_contract_hash/universe_membership_hash + market/calendar/pit_policy). **One** identity type shared by FE internal + HTTP path.

## 3. data_access/ — PIT semantics, calendar, snapshot, universe, field registry, freshness

- **get_store**: `data_access/__init__.py:58` `get_store()` → `install_cos_runtime(_get_store())`; `data_access/store.py:8997` `get_store()` and `:234` `class DataAccessStore` (single class).
- **PIT contract (canonical)**: `data_access/contract/runtime_contract.py:45` `PITContract` (pit_policy, availability/event/period columns, time representation/precision, revision availability, `pit_fidelity` knowledge_date_pit/vintage_pit, dedup_tiebreaker); `:117` `RuntimeDatasetContract` (dataset+market+storage+physical_partition+temporal_axes+pit+units+cardinality+coverage+security+semantic schema). PIT read enforcement: `data_access/read/temporal_join.py` (decision_time/knowledge_time/period_time/revision_order; `:7-25` semantics), `data_access/read/contract_ir.py:42-54` (`decision_time` anchor), `data_access/store.py:3805+` (PIT join fail-closed, `:3825` exact-join warning), `data_access/read/pit_event_index.py:207` `PITEventIndex`, `:87` `PITEventRecord`. FE-side four-layer PIT gate: `factor_engine/pit_contract.py:59` `PITLayerVerdict`, `:79` `four_layer_pit_allowed`.
- **Calendar**: `data_access/read/session_calendar.py:340` `MarketCalendar` (`:399` `is_trading_day`); `data_access/r30/calendar_snapshot.py:161` `CalendarSnapshot`.
- **DataSnapshotRef / snapshot**: `data_access/read/read_contract.py:139` `DataSnapshot` (frozen; snapshot_id = registry/schema/file_manifest hashes + params canonical bytes); `data_access/snapshot/resolver.py:409` `SourceSnapshotResolver`, `snapshot/source_snapshot.py:138` `ResolvedSourceSnapshot`, `snapshot/verifier.py:46` `SnapshotVerifier`, `snapshot/fidelity.py:34` `SnapshotFidelity`; R30 `data_access/r30/experiment_snapshot.py:96` `ExperimentDataSnapshot` (whole dependency-chain identity: source_snapshot_id/execution_context_id/security_scope_id, `:113-123`). NOTE: the exact name `DataSnapshotRef` does NOT appear in data_access — the `DataSnapshot` identity object is the closest existing analog.
- **UniverseRef / universe**: `data_access/r30/universe_snapshot.py:57` `UniverseSnapshot` (frozen; `:81` `build` digest = universe_id+market+members+membership/tradability policy versions+source_snapshot; `:127` `from_store` via `_resolve_universe_instruments`; `data_access/store.py:5967` `_resolve_universe_instruments`). Exact name `UniverseRef` does NOT appear — `UniverseSnapshot` is the analog.
- **Field registry**: `data_access/read/semantic_catalog.py:118` `SemanticField`, `:702` `SemanticFieldCatalog` (`resolve_one`); `data_access/read/derived_fields.py:119` `ResolvedFieldID`, `:300` `DerivedFieldCompiler`; `data_access/r30/mining_profile.py:266` `FieldCapabilityCatalog` (from_store `:314`); `data_access/read/data_read_identity.py` + `data_access/r30/contracts.py:86` `ReadIdentity` (snapshot/build/contract/API version identity).
- **DataFreshness**: no class literally named `DataFreshness`. Closest: `data_access/read/metadata_plane.py:125` coverage/complete/staleness report (`:218` max_staleness), `data_access/read/coverage.py:46` `CoverageContract.max_staleness` + `:424` staleness verdict (`"5d"`/`"5t"`), `data_access/r30/contracts.py:77` `SourceColumnFreshness` (column/dataset/snapshot_id/vintage/maturity), `data_access/contract/runtime_contract.py:141` `CoverageContract` (`max_staleness`, `missing_partition_semantics`).
- **API**: `data_access/service/app.py:251` `/health`, `:255` `/ready`, `:318` `/version`, `:330` `/v1/metrics`, `:366` `/v1/datasets`, `:370` `/v1/datasets/{dataset_name}`, `:387` `/v1/read/arrow-stream`, `:522` `/v1/read`, `:633` `/v1/read_uri`, `:691` `/v1/factors`, `:724` `/v1/factors/read`. Default port **8765** (`data_access/service/__main__.py:39-44`, env `DATA_ACCESS_API_PORT`).

## 4. COS usage

- **Transport**: NO vendor `cos-python-sdk-v5` / `qcloud_cos` imports anywhere in data_access/factor_engine/scripts (grep verified). COS is accessed as S3-compatible via **DuckDB httpfs** and via a read-only COS CLI.
  - `data_access/cos/remote.py:1-17` — DuckDB httpfs `read_parquet('s3://...')` direct remote read; `cos_uri_to_s3_uri()` `:135`; `S3Credentials` `:47` (COS_SECRET_ID/KEY or AWS env; `DATA_ACCESS_COS_S3_ENDPOINT` e.g. `cos.ap-guangzhou.myqcloud.com`); read modes `mirror|remote|auto` (`cos_read_mode()` `:125`, `_VALID_MODES` `:46`).
  - `data_access/cos/s3_duckdb.py` — DuckDB S3 secret config (`_apply_s3_secret` `:87`, `_apply_s3_legacy` `:118`, `ensure_duckdb_s3` `:189`).
  - `data_access/cos/mirror.py` — local mirror sync (`ensure_local_mirror_for_dataset`; COS CLI `clean-cos-ro` `:31`).
- **Layout** (bucket `qs-cold`): `data_access/cos/mirror.py:34` `ASHARE_COS_PREFIX = "cos://qs-cold/clean_data/ashare/lqtp_data"`; `:45` US massive `"cos://qs-cold/clean_data/us_stock/massive_data"`; `:56` US clean `"cos://qs-cold/clean_data"`. Dataset→table mirror registry `DATASET_MIRROR_REGISTRY` `:110-146` (ashare Calendar/StockDailyBar/StockIncome/StockBalance/StockCashFlow/StockMinuteBar/TurnoverBaseDaily/StockValuationDaily/StockIndustry/…; US SecurityMaster/StockBalance period_files/…). This is **dataset-clean-data** layout — NOT the spec §21 `/canonical/factor_values/` layout.
- **ObjectStore Protocol**: `data_access/read/object_store.py:37` `ObjectStore` (Protocol: head/list/range_read/open_reader/put/begin_multipart/upload_part/complete_multipart/abort/delete), `:82` `LocalObjectStore`, `:202` `COSObjectStore` (S3 client, `:228` `_s3`, `:253` `_uri`).
- **Who writes FACTOR VALUES to COS**: NO production path found. Factor materialization is **local-only** (factor lake `{workspace_data}/factors/lake`; staging write target `write_targets.py:142` → data_access `factor_lake_staging` upsert; publish `lake_publish.py:217`). All `cos` writes in `scripts/` are in `scripts/archive/jobs/*` (e.g. `ashare_feature_pipeline.py`, `compute_all_61_factors.py`, `compute_factors_from_zip.py`) and write `cos_data/StockDailyBar` (raw kline mirror) + `weekly_backtest_output/factor_values.parquet` (local report artifact, `scripts/archive/jobs/build_index.py:34`), not a canonical factor lake. Spec §21 canonical factor-major/feature-block layout: **MISSING** (no `factor_values/ market= freq= factor_bucket= factor_id=` writer exists).
- **COS event/reconciliation**: `data_access/cos_storage_runtime.py:21-279` installs patches for mirror sync + complete-marker (`_write_complete_marker` `:70`, `_complete_marker_valid` `:88`) — local completeness markers only, not §10.3 dual-channel ingestion.

## 5. Candidate lake / ingestion

- **Manifest/_READY protocol EXISTS as a contract**: `quant_platform/app/contracts/candidate.py:1-52` — `READY_MARKER_NAME = "_READY"` (`:20`), `is_ready_marker()` `:23`, `FactorCandidateManifest` (`:29`, fields per spec §10.2: schema_version/candidate_id/submitted_at/submitted_by/generator_type/generator_version/market/frequency/formula_language/factor_spec_uri/factor_spec_sha256/parent_factor_ids/required_fields/semantic_family_hint/campaign_id/attempt_id). This is a **contract only** (frozen dataclass); no ingestion service, no scanner.
- **No COS candidate bucket**: no code writes/reads `/candidates/market=…/source=…/date=…/candidate_id=FC_xxx/manifest.json + _READY` (grep for `_READY`/`candidates/`/`factor_spec.json` outside quant_platform/app/contracts = none).
- **What ingestion exists TODAY** (not §10.3 dual-channel): factor definition ingestion via `FactorEngine.compile` from an expression/`Factor` or YAML config (`engine.py:1041`/`:2471`); mining-manifest gates `factor_engine/api/mining_integration.py:1107` (`validate_manifest_for_execution`) and `:1230` (`get_mining_operators_from_manifest`, direct-use manifest `factor_engine/mining/direct_use.py:29` R18 DirectUse Matrix manifests); `factor_engine/mining/campaign.py` additive batch mining campaign (compile_many `:245`, negative-compile cache `:124`); `factor_engine/mining/operator_catalog.py:252` `MiningOperator` + role admission (`:47` MiningRole, `:80` AdmissionState). These are **factor-formula** ingestion gates, not the §10 **candidate artifact lake** protocol. No COS Object Event ingestion; no reconciliation scanner with prefix-partition/last_modified cursor/etag (only local complete-markers `cos_storage_runtime.py:70`).
- **Incremental recompute from data events**: `factor_engine/runtime/incremental_event_service.py:39` `plan_incremental_from_event`, `:69` `materialize_incremental_from_event`; `engine.py:2409`. This is FE's own event lane (DataEvent), not §10.3 COS ingestion.
- **Events**: `quant_platform/app/contracts/event_envelope.py` `EventEnvelope` (`:86`) + 19 `EVENT_TYPE_*` constants (`:48-88`, incl. `FactorCandidateDiscovered`, `FactorCandidateValidated`, `FactorMaterialized`, `ArtifactPublished`) — contract only; transactional outbox impl MISSING (no outbox table/worker; only doc refs in `quant_platform/docs/PLATFORM_CONTRACTS_DRAFT.md` and FE service uses in-memory `BoundedJobQueue`).

## 6. Timing contract (spec §39)

- **5 unified fields EXIST as a contract**: `quant_platform/app/contracts/timing.py:18` `TimingContract` — `decision_time`, `signal_available_time`, `first_executable_time`, `label_start_time`, `label_end_time` (`:21-25`); `EvidenceStatus` `:28`. DRAFT, frozen dataclass, pure stdlib. Wired into FE HTTP compute as `decision_time_policy` only (`service/models.py:198`, `runtime/factor_identity.py:155`).
- **In data_access**: `decision_time` is a first-class PIT/event concept (`read/contract_ir.py:42-54`, `read/temporal_join.py:7-25,100-112`, `cos_event_runtime.py:183,328-417` `read_cos_events_asof`). The other four fields (`signal_available_time`, `first_executable_time`, `label_start_time`, `label_end_time`) appear ONLY in: `quant_platform/app/contracts/timing.py` and legacy report scripts (`scripts/archive/jobs/rebuild_factor_detail_pages.py:526-527`, `build_index.py:122`, `scripts/platform_contract_e2e.py:138-139`). No production runtime consumes them.
- **A-share semantics**: PARTIAL. `factor_engine/runtime/ashare_intraday.py:12` `ashare_price_limit_rate`, `:30` `ashare_limit_prices` (ST 5% `:40`), `:46` `limit_touch_fraction`, `:64` `return_path_features`, `:79` `trade_structure_features`; PIT knowledge-date clock in `cos_event_runtime.py` (`_visible`, latency minutes `:156`). ST/停牌/涨跌停/IPO/复权/T+1/公告日期 as a **unified §39 evidence contract**: NOT present as one object; only the platform DRAFT `TimingContract` + scattered FE helpers. A-share timing is the single biggest PARTIAL.

## 7. Mapping to spec §37 / §38 / §10 / §21

| Spec section | Verdict | Evidence |
|---|---|---|
| §37 FE adapter (validate/compile/resolve/materialize/incremental/batch/artifact metadata) | **EXISTS** | api/factor.py:76,76; service/app.py routes 1328-1484; engine.py:1041,2093,1722,2409,3779; materialize/materializer.py:242,332; write_targets.py:43,142; lake_publish.py:217; export/serializers.py:35 |
| §37 "FE 仍可独立运行 / 不依赖 PG schema" | **EXISTS** | FE has own SQLite catalog (catalog.py:544), no PG dependency seen |
| §38 DA canonical PIT/calendar/snapshot/universe/field registry | **EXISTS (names differ)** | runtime_contract.py:45 PITContract; session_calendar.py:340 MarketCalendar; read_contract.py:139 DataSnapshot; r30/universe_snapshot.py:57 UniverseSnapshot; semantic_catalog.py:702 SemanticFieldCatalog |
| §38 adapter DataSnapshotRef/UniverseRef/FieldAvailability/DataFreshness | **PARTIAL** | DataSnapshot + UniverseSnapshot exist; `FieldAvailability`/`DataFreshness` named types MISSING (closest: metadata_plane.py:125, r30/contracts.py:77 SourceColumnFreshness) |
| §10.1 manifest protocol + `_READY` + `/candidates/` layout | **PARTIAL** | Contract exists (quant_platform/app/contracts/candidate.py:20,29); no writer/scanner/ingestion |
| §10.2 manifest minimal fields | **EXISTS** | candidate.py:29-52 matches field list |
| §10.3 dual-channel (COS event + reconciliation) | **MISSING** | no COS event ingestion; no reconciliation scanner; only local complete markers (cos_storage_runtime.py:70) |
| §21 canonical factor-major `factor_values/` layout | **MISSING** | factor values local-only (`workspace_paths.py:69`); COS holds clean dataset mirrors (`cos/mirror.py:34`) |
| §21 feature-block derived cache | **MISSING** | no feature_blocks writer (FE `materialize_matrix` writes local lake only) |

## 8. GAP rows

| # | Requirement | Current (files) | Gap | Action | Reuse/Modify/Add | Priority |
|---|---|---|---|---|---|---|
| 1 | §10.3 COS Object Event ingestion | none; only `quant_platform/app/contracts/event_envelope.py` EventEnvelope contract | No COS event source → ingestion; event types defined but no consumer | Build ingestion service listening on COS events; map `FactorCandidateDiscovered` → manifest validation | Add (contract Reuse) | P0 |
| 2 | §10.3 reconciliation scanner (prefix-partition, last_modified cursor, etag, manifest hash) | local complete markers only (`data_access/cos_storage_runtime.py:70`) | No periodic scanner; no cursor/etag state | Add scanner over `/candidates/` prefix; maintain cursor table | Add | P0 |
| 3 | §10.1 candidate artifact lake writer | `FactorCandidateManifest` contract only (candidate.py) | No producer writes manifest.json + `_READY` to COS `/candidates/…/FC_xxx/` | Add candidate-publish step in mining campaigns (`factor_engine/mining/campaign.py`) | Add | P0 |
| 4 | §21 canonical factor-major values on COS | factor lake is local (`workspace_paths.py:69`, staging/publish local) | No `factor_values/market/freq/factor_bucket/factor_id/snapshot` writer | Add COS publish target after `publish_factor_lake` (write_targets.py pattern) | Modify/Add | P1 |
| 5 | §21 feature-block derived cache | `materialize_matrix` local only | No `feature_blocks/feature_set/date_bucket/block` writer | Add feature-block materializer (read-only for models) | Add | P1 |
| 6 | §38 named `DataSnapshotRef`/`UniverseRef` | `DataSnapshot` (read_contract.py:139), `UniverseSnapshot` (r30/universe_snapshot.py:57) | Adapter contract names differ from spec; no `FieldAvailability`/`DataFreshness` type | Alias/export adapter-facing refs + add FieldAvailability & DataFreshness DTOs in DA adapter | Modify (Reuse existing) | P1 |
| 7 | §39 five timing fields consumable by evaluation/materialization | `TimingContract` DRAFT (quant_platform/app/contracts/timing.py) + `decision_time_policy` in FE identity; DA has `decision_time` only | `signal_available_time`/`first_executable_time`/`label_start_time`/`label_end_time` not computed/consumed anywhere in runtime | Wire TimingContract into DA read contracts + FE label window (label_pit.py LabelWindowSpec) | Modify | P1 |
| 8 | §39 A-share evidence (ST/停牌/涨跌停/IPO/复权/T+1/公告) as unified contract | scattered: `ashare_intraday.py:12,30`, PIT clocks in `cos_event_runtime.py` | No single A-share timing evidence object; page/QE could guess | Build A-share timing evidence provider in DA (calendar+tradability+limits) exposing §39 fields | Add | P1 |
| 9 | §11.1 transactional outbox | EventEnvelope contract only; service queue is in-memory (`service/queue.py:96`) | No outbox table/worker; events can be lost on crash | Add outbox writer in FE service jobstore + relay to platform | Add | P2 |
| 10 | §37 Artifact metadata return (FactorValueArtifact) | job-level `job.artifacts` (app.py:1068) + `FactorArtifactMetadata` (r30/artifact_meta.py:18) | No platform `ArtifactRef` publish registry in FE; FE returns dict summaries | Publish ArtifactRef (quant_platform/app/contracts/artifact_ref.py:74) after materialize; attach FactorArtifactMetadata sidecar | Modify | P2 |

## 9. Duplicate authorities / drift

- **Backend capability registry**: intentionally unified. `factor_engine/backend/operator_capability.py` is the single source; `backend/capability_registry.py:1-9` is a deprecated re-export facade (warns DeprecationWarning). No split authority.
- **get_store**: two `get_store()` defs — `data_access/__init__.py:58` (installs COS runtime) and `data_access/store.py:8997` — but `__init__.py` explicitly re-points the store-module attr (`__init__.py:61-63` `_store_module.get_store = get_store`) so imports can't bypass COS contracts. Single `DataAccessStore` class (`store.py:234`).
- **Formula identity**: one type `ExecutionSemanticIdentityV2` shared by FE runtime (`runtime/factor_identity.py:898`) and HTTP (`service/app.py:538`). No drift observed.
- **Capacity/freshness naming drift**: spec names (`DataSnapshotRef`, `UniverseRef`, `DataFreshness`, `FieldAvailability`) vs actual (`DataSnapshot`, `UniverseSnapshot`, `SourceColumnFreshness`, coverage report). Same concepts, different names — adapter layer needed, documented honestly.
- **PIT authority**: FE four-layer gate (`factor_engine/pit_contract.py:79`) consumes DA contracts (`PITContract`, temporal_join); both exist, wiring verified only at contract level (NOT_VERIFIED end-to-end).
- **Ingestion drift**: platform DRAFT contracts (candidate.py, timing.py, event_envelope.py) exist but are NOT referenced by any runtime (FE/DA/scripts) — dead contract drift, currently harmless but must be the base for Phase-1 implementation.
- **COS writer authority**: NONE for factor values (local-only). Archived `scripts/archive/jobs/` write `cos_data/` + `weekly_backtest_output/factor_values.parquet` — historical/reporting artifacts, not the §21 canonical lake; risk of confusion if reused as a reference.

## 10. Environment notes
- Full FE test suite NOT run (collection broken, per task note; ParamRole pollution being fixed by another agent). Module existence verified by source inspection; import smoke failed on a repo-level `platform` module name-shadow (`.venv` `sitecustomize`/`platform.py` shadow masks stdlib `platform` — `uuid` import broke) — environment issue, NOT a FE defect. NOT_VERIFIED: runtime importability of every module listed above.

_End of audit._
