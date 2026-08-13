# FactorAssets Legacy Migration Audit

Scope: read-only mining of requested legacy roots under `/home/shw/quant_projects`, interpreted against `AI_GUIDE/06_FACTOR_ASSETS_SPEC.md` and `AI_GUIDE/21_LEGACY_FUNCTION_MINING_CHECKLIST.md`. Decisions use only `REUSE_AFTER_TEST`, `REWRITE`, `REFERENCE_ONLY`, `CORPUS_ONLY`, and `DISCARD`.

Audit observation: `2026-08-13T21:58:56+08:00`; observed blueprint repository HEAD: `7fc990e706f98924b0249de3c49d67530ffe2185`. The working tree was changing concurrently; this audit did not clean, restore, checkout, stash, or modify any legacy/production/settings file.

## files found

### Admission, catalog, and pool

- `/home/shw/quant_projects/factor_layer/factor_admission/catalog.py`
- `/home/shw/quant_projects/factor_layer/factor_admission/admission.py`
- `/home/shw/quant_projects/factor_layer/factor_admission/config.py`
- `/home/shw/quant_projects/factor_layer/factor_admission/config_runner.py`
- `/home/shw/quant_projects/factor_layer/factor_admission/pipeline.py`
- `/home/shw/quant_projects/factor_layer/factor_admission/run_from_config.py`
- `/home/shw/quant_projects/factor_layer/factor_admission/run_pipeline.py`
- `/home/shw/quant_projects/factor_layer/factor_admission/tests/test_factor_admission_runtime.py`
- `/home/shw/quant_projects/factor_layer/factor_admission/tests/test_pipeline.py`
- `/home/shw/quant_projects/factor_layer/factor_admission/tests/test_pipeline_cli.py`
- `/home/shw/quant_projects/factor_layer/factor_pool/README.md` (placeholder only; no `CandidatePool` implementation)
- `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/FactorAnalyzer.py` (evaluation only; no admission/catalog behavior)

### Gateway, assetization, and integration

- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/scripts/models.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/scripts/deduplicator.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/scripts/gateway_core.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/scripts/config.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/scripts/complexity.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/scripts/data_quality.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/scripts/future_scanner.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/scripts/io_utils.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/scripts/kafka_producer.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/scripts/router.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/scripts/validator.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/gateway/models.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/gateway/gateway_core.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/gateway/deduplicator.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/gateway/config.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/gateway/complexity.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/gateway/data_quality/checker.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/gateway/gateway/data_quality/runner.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/assetization/scripts/models.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/assetization/scripts/registry.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/assetization/scripts/compute.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/assetization/scripts/compute_engine.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/assetization/scripts/worker.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/assetization/scripts/router.py`
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/integrations/quant_platform.py`

The top-level `gateway/*.py`, `gateway/gateway/*.py`, and `gateway/scripts/*.py` contain compatibility copies. The matrix names the inspected implementation under `gateway/scripts`; duplicate wrappers/copies are `DISCARD` rather than separate implementations to migrate.

### Candidate corpora and taxonomy

- `/home/shw/quant_projects/gtja191/candidate_pool/manual_ashare_pv_202607041600/config.json`
- `/home/shw/quant_projects/gtja191/candidate_pool/manual_ashare_pv_202607041600/*/manifest.json` (185 manifests)
- `/home/shw/quant_projects/gtja191/scripts/convert_and_build_delivery.py`
- `/home/shw/quant_projects/gtja191/scripts/validate_manifests.py`
- `/home/shw/quant_projects/gtja191/lib/catalog.py`
- `/home/shw/quant_projects/gtja191/tests/test_gtja191.py`
- `/home/shw/quant_projects/week2_pv_factors/candidate_pool/manual_ashare_week2_pv_202606301800/config.json`
- `/home/shw/quant_projects/week2_pv_factors/candidate_pool/manual_ashare_week2_pv_202606301800/*/manifest.json` (37 manifests)
- `/home/shw/quant_projects/week2_pv_factors/scripts/build_delivery.py`
- `/home/shw/quant_projects/week2_pv_factors/scripts/validate_manifests.py`
- `/home/shw/quant_projects/factor-pool-standard/enums/domain_roots.yaml`
- `/home/shw/quant_projects/factor-pool-standard/enums/canonical_data_fields.json`
- `/home/shw/quant_projects/factor-pool-standard/scripts/check_manifest_fields.py`
- `/home/shw/quant_projects/factor-pool-standard/README.md`

### SQL/schema/migration search

- Relevant SQL DDL exists only as Python string constants in `/home/shw/quant_projects/factor_layer/factor_admission/catalog.py`.
- Relevant model/schema files are `gateway/scripts/models.py`, `gateway/gateway/models.py`, and `assetization/scripts/models.py`.
- No relevant standalone `*.sql`, SQLAlchemy model, Alembic directory, migration directory, `alembic_version`, `PRAGMA user_version`, or catalog schema-version table was found in the requested roots.
- `schema_version: disk.v1` versions JSON disk contracts only; it is not a database migration mechanism.

## function-level migration matrix

| file | symbol | signature | decision | target FA module | reason | required tests |
|---|---|---|---|---|---|---|
| `factor_admission/catalog.py` | `_BASE_SCHEMA_SQL`, `_ADMISSION_SCHEMA_SQL` | SQL constants | REFERENCE_ONLY | `factor_assets.storage.sqlite` migrations | Useful identity/run/status/decision separation; lacks versioned migrations, indexes, lifecycle events, lineage, relations, fingerprints and EvidenceRef | migration upgrade; FK integrity; index plan; no-delete invariant |
| `factor_admission/catalog.py` | `AdmissionCatalog.__init__` | `(db_path: str | Path) -> None` | REUSE_AFTER_TEST | `SQLiteRepository` | SQLite WAL is suitable for the first backend; initialization must run numbered migrations transactionally | WAL concurrency; migration rollback; FK enforcement |
| `factor_admission/catalog.py` | `ensure_factor_registered` | `(factor_id: str) -> None` | REWRITE | `Repository.require_asset` | Coupled to factor-lake registration; FA owns asset identity and references FE through public contracts | missing/existing asset; package boundary |
| `factor_admission/catalog.py` | `upsert_evaluation_run` | `(payload: dict) -> None` | REWRITE | `evidence.attach_evidence_ref` | Stores filesystem paths and overwrites run metadata; FA needs immutable QE bundle refs | idempotency; immutable bundle identity; stale ref rejection |
| `factor_admission/catalog.py` | `replace_evaluation_summary` | `(run_id, rows) -> None` | DISCARD | none | Deletes/reinserts 18 QE-owned metric columns, creating a second metrics store | schema test proving metric columns absent; QE adapter mock |
| `factor_admission/catalog.py` | `record_decision` | keyword args -> `str` | REWRITE | `DecisionLedger` | Append-only decision and policy snapshot are valuable, but need typed evidence/context and a transaction with state events | append-only; idempotency; rollback; audit ordering |
| `factor_admission/catalog.py` | `update_factor_status` | `(*, factor_id, latest_run_id, approved)` | REWRITE | `lifecycle.StateMachine.transition` | Blind overwrite collapses lifecycle to two states and writes no event | legal/illegal transition property tests; event/current consistency |
| `factor_admission/catalog.py` | `get_factor_status`, `get_evaluation_run` | `(id) -> dict | None` | REUSE_AFTER_TEST | `Repository` | Parameterized read pattern is reusable after typed models/repository abstraction | found/not-found; type round-trip |
| `factor_admission/catalog.py` | `list_factor_library` | `() -> list[dict]` | REWRITE | `AssetQueryService` | Useful read-model intent, but joins QE metrics and factor-lake watermarks into governance | pagination; state/family filters; no metric truth duplication |
| `factor_admission/admission.py` | `_load_summary_payload`, `_resolve_run_dir` | path helpers | DISCARD | none | Direct `summary.json` and factor-lake path coupling are replaced by QE bundle adapter | boundary audit; malformed/unavailable ref |
| `factor_admission/admission.py` | `_check_thresholds` | `(summary_row, thresholds) -> list[str]` | REWRITE | `admission.policy.ConfiguredGate` | Diagnostics are useful; fixed thresholds must be named/versioned policy config and only one gate | missing evidence fail-closed; boundary values; policy replay |
| `factor_admission/admission.py` | `admit_evaluation_run` | `(config) -> dict` | REWRITE | `AdmissionService` | Must orchestrate safety gates, EvidenceRef, conditional novelty, Pareto/budget context, decision and lifecycle transaction | novelty adapter; Pareto cases; rollback; replay idempotency |
| `factor_admission/config.py` | `MetaConfig`, `SourceConfig`, `OutputConfig` | dataclasses | REFERENCE_ONLY | `factor_assets.config` | Separation is useful; lake paths and decision files are legacy coupling | config validation; environment independence |
| `factor_admission/config.py` | `ThresholdConfig`, `DecisionConfig` | dataclasses | REWRITE | `admission.policy` | Convert to typed, named, versioned contextual policy with hard gates and Pareto/budget rules | round-trip; version pinning; invalid policy rejection |
| `factor_admission/config.py` | `load_config` | `(path) -> FactorAdmissionConfig` | REUSE_AFTER_TEST | `config.load_policy` | YAML and scalar validation pattern is useful after removing lake paths | relative paths; types; digest repeatability |
| `factor_admission/pipeline.py` | `run_pipeline`, `run_from_config`, `run_config_directory` | batch runners | REFERENCE_ONLY | `service.batch_admit` | Batch/error/config-snapshot ideas are useful; filesystem orchestration is not registry core | mixed batch; deterministic snapshot; no partial DB commit |
| `factor_admission/pipeline.py` | `_prepare_output_root` | `(output_root, fallback_name)` | DISCARD | none | Uses `shutil.rmtree` to clear history-like outputs | no physical-delete static test |
| `factor_pool/README.md` | placeholder pool | n/a | DISCARD | none | No implementation or stable contract | none |
| `FactorAnalyzer.py` | `FactorAnalyzer`, configs | evaluator API | CORPUS_ONLY | QE corpus | Pure metrics/evaluation; no FA admission behavior | FA import boundary; QE golden tests |
| `gateway/scripts/models.py` | `GatewayLabel` | `PASS/DUPLICATED/REJECTED` | REFERENCE_ONLY | lifecycle import mapping | Useful outcomes, not an asset lifecycle | mapping; duplicate-to-shadow/seen behavior |
| `gateway/scripts/models.py` | `GatewaySegment` | dataclass | REWRITE | `decision.GateEvent` | Preserve stage/run/timestamp/diagnostics; scores become evidence refs | JSON round-trip; immutable event |
| `gateway/scripts/models.py` | `Candidate` | candidate dataclass | REWRITE | `models.FactorCandidate` | Preserve candidate ID, campaign, batch, origin and timestamps; add FE definition ref and typed parents | disk.v1 import; provenance; FE identity parity |
| `gateway/scripts/models.py` | `GatewayResult` | dataclass | REFERENCE_ONLY | `admission.GateResult` | Typed result pattern useful; statuses/evidence need redesign | serialization; exhaustive statuses |
| `gateway/scripts/models.py` | `Manifest` | disk artifact dataclass | REFERENCE_ONLY | `importers.disk_v1` | Import contract only, not new repository model | all 222 manifests; version rejection |
| `gateway/scripts/deduplicator.py` | `DuplicateRecord`, `DeduplicationResult` | dataclasses | REWRITE | `seen.models` | Preserve first-seen/historical ref idea; add lifecycle scope, hash kind, neighbors and decision hints | lifecycle retention; serialization |
| `gateway/scripts/deduplicator.py` | `ExpressionNormalizer.normalize` | `(expr) -> str` | REFERENCE_ONLY | `seen.FEIdentityAdapter` | Whitespace/lowercase is not semantic canonicalization; FE must provide canonical AST hash | adversarial expressions; FE hash parity |
| `gateway/scripts/deduplicator.py` | `compute_expr_hash` | `(expr) -> str` | REFERENCE_ONLY | `GlobalSeenIndex` | Exact hash/history concept is sound; production implementation consumes FE canonical representation | repeatability; collision handling |
| `gateway/scripts/deduplicator.py` | `compute_candidate_hash` | `(expr, config) -> str` | REWRITE | `IdentityService` | Arbitrary JSON config includes incidental fields and lacks parameter/sign canonicalization | key order; numeric normalization; relevant-field tests |
| `gateway/scripts/deduplicator.py` | `HashDeduplicator` persistence/cache | cache API | REWRITE | `SQLiteSeenIndex` | 10k LRU evicts global history and corrupt JSON is silently ignored | restart; >10k history; corrupt store fail-closed; concurrency |
| `gateway/scripts/deduplicator.py` | `SemanticDeduplicator` vector/config methods | dedup API | DISCARD | none | Vector init/search/insert and config equivalence are stubs | ensure no stub backend registered |
| `gateway/scripts/gateway_core.py` | `StepResult`, `run_gateway`, `_finalize` | seven-stage pipeline | REFERENCE_ONLY | `admission.gates` | Short-circuit gate/event pattern useful; implementation mixes FE/QE/LLM/I/O | deterministic ordering; downstream N/A; fail-closed |
| `gateway/scripts/gateway_core.py` | `_step1_validate` | `(manifest, campaign_config)` | REWRITE | `importers.disk_v1` | Preserve schema/campaign merge with typed, immutable input validation | required fields; versions; no input mutation |
| `gateway/scripts/gateway_core.py` | `_step2_build_yaml`, `_step4_tiny_run` | FE gates | DISCARD | none | FE owns configuration, compilation and execution; FA stores refs/events | no FE private import; adapter mock |
| `gateway/scripts/gateway_core.py` | `_step3_dedup`, `_write_dedup_cache` | hash/cache | REWRITE | `GlobalSeenIndex` | First-seen idea useful; JSON read-modify-write is nontransactional and local | duplicate race; atomic insert; restart |
| `gateway/scripts/gateway_core.py` | `_step5_future_scan` | `(manifest) -> StepResult` | DISCARD | none | LLM scan is not authoritative; FE timing contracts own legality | FE rejection propagation |
| `gateway/scripts/gateway_core.py` | `_step6_complexity`, `_calc_nesting_depth` | regex score | DISCARD | none | Placeholder second planner/complexity system; FE owns complexity profile | no regex parser in FA; FE adapter mock |
| `gateway/scripts/gateway_core.py` | `_step7_data_quality` | coverage gate | DISCARD | none | Direct Parquet read and hard-coded 98%; DA/QE own data quality | no Parquet read; QE evidence mock |
| `gateway/scripts/complexity.py` | all symbols | regex weighted score API | DISCARD | none | Duplicates FE complexity/planning; optional AST parser is a stub | dependency audit |
| `gateway/scripts/data_quality.py` | all detector/checker/runner symbols | DQ APIs | DISCARD | none | Second metrics/DQ system with fixed thresholds; QE/DA ownership | no scipy/pandas DQ kernels in FA |
| `gateway/scripts/config.py` | `GatewayConfig` and subconfigs | config API | DISCARD | none | Legacy routing plus duplicated FE allowlist/future/complexity config | dependency audit |
| `assetization/scripts/models.py` | `FactorCandidate` | `(expr, config, born_timestamp, basic_info)` | REWRITE | `models.FactorCandidate` | Minimal metadata reference; needs definition refs, origin, campaign and parents | legacy import; immutable identity |
| `assetization/scripts/models.py` | `FactorAsset` | `(factor_id, coordinates, state, materialized_path)` | REWRITE | `models.FactorAsset` | Concept useful; model is incomplete and embeds materialization path/state | full round-trip; lifecycle enum; no raw path |
| `assetization/scripts/models.py` | `PhysicalPlan` | plan dataclass | DISCARD | none | Explicitly prohibited; FE owns planner/IR/execution | no `PhysicalPlan` in FA imports/API |
| `assetization/scripts/registry.py` | `generate_factor_id` | `(coordinates, seed=None) -> str` | REWRITE | `identity.IdentityService` | Stable seed idea useful; eight-hex suffix and coordinates are insufficient canonical identity | collision/property; concurrent registration |
| `assetization/scripts/registry.py` | `extract_coordinates` | `(config) -> dict[str, str]` | REUSE_AFTER_TEST | `origin.import_coordinates` | Four coordinates are useful import metadata | missing field; enum validation; round-trip |
| `assetization/scripts/compute.py` | `FormulaEvaluator`, `run_assetization`, helpers | execution/materialization | DISCARD | none | FE executes and DA materializes; returned series/daily values cannot enter FA | no execution/materialization imports; repository rejects values |
| `assetization/scripts/compute_engine.py` | all symbols | plan/execution/storage | DISCARD | none | PhysicalPlan, PIT join, Parquet and staging belong to FE/DA | package boundary; no data-frame storage dependencies |
| `assetization/scripts/worker.py` | `process_factor_dir`, `_daily_series_frame` | worker | DISCARD | none | Persists daily raw values and staging summaries in governance artifacts | no raw persistence; no `.parquet` writes |
| `assetization/scripts/router.py` | `AssetizationRouter` | directory mover | DISCARD | none | Directory moves and `shutil.rmtree` are not lifecycle governance | destructive-I/O scan; no-delete invariant |
| `integrations/quant_platform.py` | all symbols | FE/DA bridge | DISCARD | none | FA should use narrow public identity/evidence/ValueProvider adapters, not own FE/DA execution bridge | extraction and dependency DAG tests |
| `week2.../build_delivery.py` | `candidate_hash`, `load_catalog`, `build_campaign` | corpus builder | CORPUS_ONLY | disk.v1 fixtures | 37 real formulas and origin metadata; truncated hash is directory naming only; embedded metrics are not truth | 37 imports; provenance; metric stripping |
| `gtja191.../convert_and_build_delivery.py` | `_candidate_hash`, `build_dsl_catalog`, `build_campaign` | corpus builder | CORPUS_ONLY | disk.v1 fixtures | 185 delivered formulas for identity/family/graph tests; conversion belongs to FE/corpus | 185 imports; provenance; duplicate families |
| `factor-pool-standard/enums/domain_roots.yaml` | `domain_roots` | four-domain taxonomy | REUSE_AFTER_TEST | `taxonomy.domain` | Useful vocabulary: price_volume, fundamental, alternative, microstructure | enum round-trip; unknown values; multi-membership |
| `canonical_data_fields.json` | canonical registry | v3 JSON | REFERENCE_ONLY | DA/FE taxonomy adapter | Useful field/domain source but FA must not duplicate DA/FE registry truth | source-version pin; adapter contract |
| `check_manifest_fields.py` | `_validate_manifest` and helpers | validator | REFERENCE_ONLY | `importers.disk_v1` | Domain validation intent useful; direct `sys.path` mutation and FE internals cannot migrate | public FE adapter; mismatch corpus; extraction |
| mirrored gateway trees | duplicate symbols | duplicates | DISCARD | none | Avoid multiple divergent copies | duplicate-source/import audit |

## schema designs worth reusing

| table | columns | indexes/constraints | migration value |
|---|---|---|---|
| `factor_registry` | `factor_id`, `author`, `frequency`, `description`, `ast_hash`, `expression`, `created_at` | PK `factor_id`; no secondary index | Identity/catalog separation is useful. New `factor_asset` needs FE definition ref, canonical hash, origin/campaign, families, lifecycle, version and timestamps. |
| `factor_watermark` | `factor_id`, dates, `last_updated`, `row_count` | PK/FK `factor_id` | `DISCARD` from FA; factor-lake/materialization metadata belongs to FE/DA. |
| `factor_evaluation_runs` | run/factor IDs, paths, sample bounds, universe, horizon, config hash, time | PK `run_id`, FK `factor_id`; no secondary index | Replace paths with immutable `factor_evidence_ref` rows keyed to QE bundles. |
| `factor_evaluation_summary` | run/horizon plus 18 IC/rank/portfolio/stat columns | PK `(run_id,horizon)` | `DISCARD`; duplicated QE truth. Keep selected summary only beside QE bundle ref. |
| `factor_admission_status` | factor/latest run IDs, status, time | PK/FKs | Current-state projection useful only when transactionally paired with append-only state events. |
| `factor_admission_decisions` | decision/factor/run IDs, decision, actor, reason, policy name/snapshot, time | PK/FKs | Strongest reusable design: immutable decision ledger plus policy snapshot. Replace run path identity with EvidenceRefs and idempotency key. |

Representative legacy table:

```sql
CREATE TABLE IF NOT EXISTS factor_admission_decisions (
    decision_id      TEXT PRIMARY KEY,
    factor_id        TEXT NOT NULL,
    run_id           TEXT NOT NULL,
    decision         TEXT NOT NULL,
    decided_by       TEXT NOT NULL,
    reason           TEXT,
    policy_name      TEXT,
    policy_snapshot  TEXT,
    decided_at       TEXT NOT NULL,
    FOREIGN KEY (factor_id) REFERENCES factor_registry(factor_id),
    FOREIGN KEY (run_id) REFERENCES factor_evaluation_runs(run_id)
);
```

No explicit secondary indexes were defined. New FA needs unique canonical-hash/fingerprint indexes; state-event and decision indexes on `(factor_id, occurred_at)`; lineage indexes in both directions; evidence indexes and unique QE bundle refs; relation/membership indexes on both endpoints/family; lifecycle-state and campaign/origin indexes. Enable SQLite foreign keys on every connection.

## lifecycle state machines legacy

Admission catalog states are only `evaluated` and `approved`; decisions are `approved` and `rejected`. There is no transition validator, so code permits:

- absent -> `evaluated` or `approved`
- `evaluated` -> `evaluated` or `approved`
- `approved` -> `approved` or `evaluated`

A rejection after approval demotes current status while retaining `latest_approved_run_id`. No state event records the transition.

Gateway terminal labels are `PASS`, `DUPLICATED`, `REJECTED`; step statuses additionally include `TEMP` and `N/A`. Assetization comments name `landing`, `screening`, `materializing`; the worker emits `Materialized/passed`. No coded transition graph, weakening, shadow, revival, quarantine, retirement, or illegal-transition rejection exists.

FA must define and test the spec lifecycle: `DISCOVERED`, `COMPILE_VALIDATED`, `VALUE_VALIDATED`, `EVALUATED`, `ADMITTED`, `ACTIVE_CORE`, `ACTIVE_TACTICAL`, `REGIME_CONDITIONAL`, `SHADOWED`, `WEAKENING`, `DORMANT`, `QUARANTINED`, `REJECTED`, `RETIRED`. Use `PRODUCTION_CERTIFIED` as a separate certification flag. Legacy states are importer mappings only.

## lineage/origin/campaign metadata patterns

- Gateway Candidate: `candidate_id`, `campaign_id`, `batch_id`, `born_timestamp`, `basic_info`, `config`, and staged Gateway/Assetization/Purification/Evaluation sections.
- Campaign config: generator name/version, miner, market, universe, signal structure, asset class, frequency, domain root/domain, operator policy, mining periods, creation/run times and run statistics.
- Candidate manifests repeat campaign/origin fields and add formula/expression type, timestamp and description.
- Dedup records retain candidate ID, campaign ID, first-seen time and historical report ID.
- Missing everywhere: typed parent factor IDs, mutation/search operation and edge role, lineage version, search provenance ref, multiple memberships, and family split/merge history.

Map campaign/batch/generator/miner fields to origin/search provenance; candidate ID to external source ID; formula to FE definition ref and canonical hash; parents to `factor_lineage`; domains to data/economic memberships. Never infer parents from naming or expression similarity.

## dedup/seen patterns

- Campaign builders hash normalized whitespace plus universe/frequency and truncate SHA-256 to eight hex characters. This is corpus directory naming only.
- Gateway `_step3_dedup` stores full SHA-256 first-seen records in `dedup_cache.json`.
- `HashDeduplicator` stores expression and expression+config hashes plus historical report ID, but evicts after 10,000 and silently ignores corrupt persistence.
- Whitespace removal/lowercasing is not canonical DSL/AST identity.
- `SemanticDeduplicator` is a stub: vector initialization/search/insert and config equivalence are unimplemented, so it is `DISCARD`.
- New `GlobalSeenIndex` must persist FE canonical AST hashes, parameter/sign canonicalization metadata, structural fingerprints and behavioral fingerprint refs across active, shadowed, rejected, retired and failed history. SQLite uniqueness/transactions must resolve concurrent first-seen races. Behavioral metrics are QE evidence; FA stores sparse relations, never an all-pairs matrix.

## PhysicalPlan and raw-value storage to DISCARD

`PhysicalPlan` exists at `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/assetization/scripts/models.py` and is consumed by `assetization/scripts/compute_engine.py`. It carries a serialized DAG/backend and private FE plan/IR handles. It is unambiguously `DISCARD`: FactorEngine owns parser, planner, IR, CSE, execution and materialization.

Raw values that must not migrate into FA:

- `assetization/scripts/compute.py:run_assetization` returns full `series` and `daily_values` and can materialize staging.
- `assetization/scripts/worker.py:process_factor_dir` writes `data/{date}.parquet`; `_daily_series_frame` stores `TradeDate`, `Symbol`, `factor_id`, `factor_value`.
- `assetization/scripts/compute_engine.py:atomic_materialize` writes local Parquet or staging.
- `integrations/quant_platform.py:materialize_factor_to_staging` writes `factor_lake_staging` and optionally publishes to `factor_lake`.
- Admission `factor_watermark` stores value-lake coverage/row-count metadata.

These may be valid FE/DA behaviors in the legacy runtime, but they are discarded as FA migration sources. FA aggregation must use a `ValueProvider` and persist references/specifications only.

## second metrics system to DISCARD

Two legacy areas duplicate metric truth owned by QE:

1. `factor_evaluation_summary` stores 18 metric columns (`ic_*`, `rank_ic_*`, spread/monotonicity, coverage counts, long-short return/volatility/Sharpe/drawdown), and `replace_evaluation_summary` deletes/reinserts them.
2. GTJA and Week2 manifests embed train/valid/test `ic`, `icir`, `rank_ic`; Gateway DQ adds separate drift/liquidity/clock metrics and thresholds.

All are `DISCARD` as FA truth stores. Corpus import may reference original JSON as a source artifact, but FactorAsset stores a QE bundle ref and a small selected summary only. FactorAnalyzer and Gateway DQ calculations must not migrate into FA.

## fixed thresholds to convert to policy config

- Admission `ThresholdConfig`: minimum IC/rank-IC means and win rates, top-minus-bottom, monotonicity, total/annual return, Sharpe and max drawdown.
- Gateway complexity: default/environment `20.0`; config `max_complexity`, `max_ast_depth`, semantic similarity default `0.95`.
- Gateway field coverage: hard-coded `98%`.
- Assetization local null threshold: default `0.05`.
- Gateway DQ: KS `0.05`, Wasserstein `1.0`, spread spike `2.0`, LOB gap `0.05`, pass `0.7`, reason cutoffs `0.3/0.2/0.1`.
- Complexity labels hard-code `<5`, `<15`, `<30`; this regex score does not migrate.

Any retained threshold belongs in a named/versioned policy snapshot with evidence field, unit, horizon, universe, missing-value behavior and context. Core FA code must encode no universal IC, similarity, coverage or complexity cutoff.

## migrations/versioning patterns

- Catalog startup uses `CREATE TABLE IF NOT EXISTS`; there is no migration history, schema version, compatibility check, down migration or checksum.
- WAL is enabled, but `PRAGMA foreign_keys=ON` is absent and multi-step admission commits independently rather than atomically with decisions/state events.
- `disk.v1` and `canonical_data_fields.v3` are useful external artifact versions. Gateway's version rejection should survive in an importer.
- `policy_name` plus serialized `policy_snapshot` is valuable for decision replay.
- Pipeline YAML snapshots are useful audit references, but durable decisions should reference a content digest/version rather than filesystem paths.
- FA needs ordered immutable migrations, for example `schema_migration(version, name, checksum, applied_at)`, transactional application, startup compatibility checks, explicit indexes/FKs, migration tests from every supported version, and no physical-delete migration for asset history.

The reusable legacy core is narrow: SQLite WAL/repository mechanics after testing, append-only decision/policy snapshot semantics, origin/campaign metadata, exact first-seen history as a concept, and the four-domain taxonomy. Admission, lifecycle, identity and persistence require rewrites. PhysicalPlan, execution/materialization, raw matrices, embedded metrics/DQ, stub semantic dedup, duplicate gateway copies and destructive directory routing must not enter `factor_assets`.
