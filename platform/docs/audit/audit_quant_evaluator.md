# Phase-0 Audit: quant_evaluator

- **HEAD SHA:** d58eae0821da2c71182e200751af97a577e1a616
- **Baseline:** `pytest quant_evaluator/tests -q` = **332 passed, 2 skipped** (verified 2026-08-26)
- **Scope:** `/home/sunhaiwei/quant_projects/quant_evaluator/` (206 .py/.json/.toml files)
- **Spec:** `QUANT_RESEARCH_PLATFORM_MASTER_IMPLEMENTATION_SPEC_20260826.md` §33 (QE), §19 (schema), §7 (artifacts), §40 (streaming evidence)
- **Report target:** ≤400 lines. File:line refs relative to `quant_evaluator/`.

---

## 1. Artifact inventory (frozen dataclasses *Artifact/*Bundle)

All payload arrays are copied + marked read-only on construction; `provenance` is a recursively-immutable `FrozenMapping`; `__hash__` is process-stable (`stable_content_hex`). Immutability + provenance are universal across the contract family.

### contracts/artifact_types.py (derived-input contracts)
| Class | Line | Fields | content_hash / provenance / immutable |
|---|---|---|---|
| `ICSeriesArtifact` | artifact_types.py:52 | values (T,F), time_index, ic_method (pearson/spearman), factor_ids, metric_id, provenance | stable hash `__hash__`:112; FrozenMapping provenance:90; arrays frozen:72 |
| `QuantileReturnArtifact` | artifact_types.py:147 | values (n_q,F), n_quantiles, factor_ids, metric_id, provenance | hash:188; provenance:175 |
| `ProbePortfolioArtifact` | artifact_types.py:228 | values (T,F), time_index, factor_ids, metric_id, provenance | hash:269; provenance:256 |
| `ExposureArtifact` | artifact_types.py:309 | values (K,F), exposure_type, factor_ids, metric_id, provenance | hash:344; provenance:331 |

All four are `@dataclass(frozen=True, eq=False)`, serialize losslessly via `to_dict`/`from_dict` (ndarray codec). All carry **provenance** + stable **content hash** + **immutability**. No explicit `content_hash` field — identity is `__hash__`.

### contracts/metric_artifacts.py (canonical MetricArtifact family)
| Class | Line | Payload | Notes |
|---|---|---|---|
| `MetricArtifact` (base) | metric_artifacts.py:301 | metric_id, domain, artifact_kind, provenance, created_from, factor_axis, production, contract_schema_version, producer_version | frozen; deep-immutable via `FrozenMapping`:74; stable hash:411; production mode forces copy+freeze:270; `artifact_kind` class-authoritative QE-P0-03:335 |
| `ScalarMetricArtifact` | :516 | values (F,) | |
| `SeriesMetricArtifact` | :542 | values (T,F), time_index, time_axis | axis hash-consistent |
| `VectorMetricArtifact` | :594 | values (K,F), quantile_axis | |
| `MatrixMetricArtifact` | :637 | values (K,K,F), matrix_axis | |
| `DistributionMetricArtifact` | :680 | samples (B,F), stat_names | |

Provenance + immutability + process-stable content hash all present (QE-P0-01..04). `contract_schema_version`/`producer_version` ride the artifact (R46 P0-T).

### contracts/treatment_evaluation.py
| Class | Line | Fields | Notes |
|---|---|---|---|
| `DeltaMetrics` | :85 | 7 delta_* float fields, None=fail-closed | frozen; finite-validation |
| `TreatmentEvaluationArtifact` | :125 | factor_id, treatment_id, raw_baseline_evidence_ref, treatment_evidence_ref, absolute_metrics, delta_vs_raw, snapshot/universe/split_ref, created_at, content_hash | frozen; **DERIVED-ONLY content_hash** computed over semantic fields, caller cannot forge (:241-245). Raw baseline = NO_OP/RAW candidate 0 invariant (:16-21, :248) |

### contracts/evidence_status.py
| `MetricEvidence` | :98 | status, reason_code, observations, minimum_required, artifact, reason_detail | frozen, hashable; serialize via to_dict |

### api/requests.py (public request/result bundles)
| `EvaluationRequest` | :13 | batch_or_factor_ids, label_bundle, metric_ids, slices, context, tier, cost_budget, metadata, split_ref | frozen |
| `MetricValue` | :69 | metric_id, value, valid, observation_count, metric_version, warnings | frozen |
| `FactorDiagnosis` | :80 | factor_id, obs counts, coverage, constant/nan/inf flags, min/max/mean, warnings | frozen |
| `EvaluationBundle` | :96 | request_id, factor_ids, label_id, timestamp, schema_version, metric_values, diagnostics, grouped_metrics, series_refs, metric_versions, config_hash, warnings, metadata, split_ref | frozen; NO content_hash field; NO provenance field |

### contracts/label_bundle.py
| `LabelBundle` | label_bundle.py:11 | target_id, values, horizon, execution_delay, decision/execution/label timing, validity, source_ref, calendar_ref, metadata | frozen |

### reporting/tear_sheet.py
| `MetricArtifact` (reporting wrapper) | tear_sheet.py:31 | name, metric_type, data, title, x/y_label, dimensions, computed, status, source_artifact, reason_code | `@dataclass(frozen=True, slots=True)`. **Second distinct `MetricArtifact` class** — thin wrapper over canonical `contracts.metric_artifacts.MetricArtifact` (imported as `CanonicalMetricArtifact`). `from_canonical`:90, `not_computed`:140, `wrap_artifact`:202 |
| `EvaluationResult` | tear_sheet.py:484 | ~40 scalar/series fields + `artifacts: Dict[str, MetricArtifact]` + `status: EvidenceStatus` | **mutable** dataclass, backward-compat container |
| `INSTITUTIONAL_PANELS` | tear_sheet.py:225 | 22-panel registry (panel_key, title, artifact_key, chart_type) | — |

### reporting/artifacts.py (durable store)
| `ArtifactInfo` | :36 | frozen metadata record | — |
| `ChartArtifactStore` | :158 | durable JSON store; content-hash dedup, atomic fsync, file lock, checksum | persists ChartSpec only |

**Notable:** There is **no class named `EvaluationArtifact`** anywhere in QE (only `EvaluationBundle`). Spec §33 req 1 names `EvaluationArtifact / EvaluationBundle` as the single source of truth.

---

## 2. Registry / catalog authority

**ONE MetricSpec/MetricRegistry/Domain class definition** lives in `registry/metrics.py` (`MetricSpec`:116, `MetricRegistry`:224). `metrics/catalog.py` re-exports those classes — no second class definition. **However, two SEPARATE populated catalogs / authorities exist** (see Second-Authority section):

1. `registry/metrics.py` module-level `_METRIC_CATALOG` + `register_metric`/`get_metric`/`seal_metric_registry`:340 — populated with core metrics `rank_ic`, `ic_std`, `ic_ir`, `mean_ic`, `coverage`, `pearson_ic`, `pearson_ic_series/std/ir`, `rank_ic_series`, `ic_median`, `hac_pvalue/tstat`, `turnover`, `quantile_spread`, `subsample_stability`, `ic_autocorr_lag1`, `rank_stability`, `half_life`, `block_bootstrap_ci`, `factor_turnover_rate`, `quantile_returns_full` (:528-839). Canonical dotted aliases `CANONICAL_METRIC_ALIASES` (:490).
2. `metrics/catalog.py` catalog-scoped `MetricRegistry()` instance `_REGISTRY` + `CATALOG`, sealed at import — registered ~10-domain specs (`pearson_ic`, `spearman_ic`, `rank_ic`, `rank_ic_time_series`, `rank_ic_cross_section`, `quantile_returns`, `quantile_spread`, `quantile_stability`, `max_drawdown`, `drawdown_duration`, `calmar_ratio`, `turnover_rate`, `turnover_cost`, `turnover_adjusted_ic`, `var_95/99`, `cvar_95/99`, `skewness`, `kurtosis`, `factor_coverage`, `return_coverage`, `joint_coverage`, `hhi_concentration`, `hhi_effective_n`, `ic_stability`, `turnover_stability`, `coverage_stability`, `rolling_ic`, `ic_decay`, `autocorrelation_ic`).

Both keyed by `metric_id`, overlapping on `pearson_ic`, `rank_ic`, `quantile_spread`. The two catalogs diverge on naming, `ic_method` defaults, and domain assignment — see Second-Authority.

### RankIC vs PearsonIC dependency separation
- Registry specs declare `ic_method` per metric: `rank_ic`→"spearman" (registry/metrics.py:545), `pearson_ic`→"pearson" (:628). Separate specs `pearson_ic_std` vs `ic_std`, `pearson_ic_ir` vs `ic_ir` — each with its own `ic_method`. Separated in registry.
- Runtime adapter picks the method from the spec: `_ic_method_for_metric` (runtime/evaluator.py:881-892), used in `evaluate` to build the wrapper IC series (evaluator.py:1010-1039).
- Formal artifact contract enforces it: `ICSeriesArtifact.ic_method` validated to pearson/spearman (artifact_types.py:80-84) so a Spearman series is never consumed as Pearson. Requirement: **EXISTS** for the registry/metrics authority.

### Evidence status enums (spec §40 requires)
`contracts/evidence_status.py: EvidenceStatus` (:32) = COMPUTED, NOT_COMPUTED, UNAVAILABLE, UNSUPPORTED, INSUFFICIENT_DATA, FAILED.
- **MISSING** `INVALID_EVIDENCE` and `LABEL_NOT_MATURE` (spec §40 explicitly demands all four: NOT_COMPUTED / UNAVAILABLE / INVALID_EVIDENCE / LABEL_NOT_MATURE).
- tear_sheet.py defines module string constants `NOT_COMPUTED="NOT_COMPUTED"`, `UNAVAILABLE="UNAVAILABLE"` (:173-174) — separate from the enum, not enum members. Minor second source.

---

## 3. Storage / large-series handling

- **In-memory:** metric series/matrices are numpy arrays embedded in artifacts; frozen + copied. For 100k+ factors there is `runtime/streaming_evaluator.py` (chunked, constant-memory accumulators), returning `StreamingEvaluationResult`.
- **Intermediate cache:** `runtime/cache_v2.py` `MultiLevelCache` (MemoryCacheLayer:352, DiskCacheLayer:600). Persistence: pickle payload + JSON `.meta.json` sidecars (:694, :711), atomic writes, checksum/TTL/dependency-invalidate. Wrapped by `runtime/cache_v2_adapter.py:V2IntermediateCacheAdapter` (:19).
- **ChartSpec store:** `reporting/artifacts.py` `ChartArtifactStore` persists ChartSpec JSON files with SHA-256 content dedup (:158,216).
- **ArtifactRef:** QE has **NO** `ArtifactRef` type. Spec §19.1 says DB only stores summary + ArtifactRef, large matrices in COS. The platform repo has `platform/app/contracts/artifact_ref.py` (ArtifactRef with artifact_id/type/schema_version/content_hash/storage_uri/...), but QE does not reference it; QE exposes raw artifacts/series via `.to_dict()` + evaluation_summary in `EvaluationBundle.series_refs` (strings, not typed ArtifactRef).

## 4. API / adapters

- `api/requests.py`: typed DTOs only (request + result bundles). **No FastAPI/service layer, no HTTP handler** inside QE.
- `adapters/`: `data_access.py` (`DataAccessAdapter`), `factor_engine.py` (`FactorEngineAdapter`), `pandas.py` (`PandasAdapter`). Protocol-based, optional deps, raise `OptionalDependencyMissing`. Explicitly not imported by core (adapters/__init__.py).
- `backends/registry.py` `BackendRegistry` + `register_backend`/`auto_backend` for compute backends (numba/cupy/polars), capability gating is backend-per-operation.
- **No `create_adapter` factory, no explicit capability-check protocol method, no `reports export` on adapters.** Adapter boundary is per-file protocol methods; capability expressed as availability-of-optional-dependency.

## 5. Reporting

- `reporting/tear_sheet.py`: `generate_tear_sheet` renders **only pre-computed** artifacts (`_spec_from_artifact`:323-351); never recomputes metrics; absent artifact → `NOT_COMPUTED` placeholder (never 0.0). Confirmed: no `compute_`/`np.mean`/`np.std` aggregation in panel renderers; only serialization `_plain`/`tolist`.
- **Does not read the metric registry/catalog** — reads `contracts.metric_artifacts` and `contracts.evidence_status` only. No `metric-registry.json` file exists in QE.
- `reporting/library_reports.py` `generate_library_report`/`compare_libraries`/`generate_adversarial_report` exist but `reporting/__init__.py` stubs them as `NotImplementedError` placeholders (lazy).

## 5. Grade / health

- **No `Grade`/`Health`/scoring subsystem** anywhere in QE. Closest: `diagnosis/factor.py::diagnose_factor` (data-quality warnings: missing, constant, NaN/Inf, z-score outliers) and `diagnosis/warnings.py` `WarningSystem`/`WarningSeverity`. `reporting/library_reports.py:257` exposes an `stability_scores` in the adversarial report only.
- Spec §33 asks QE to provide `grade`, `health evidence`, `data quality`, `rolling health updates`. **Grade and health = MISSING.** Evidence status mechanism exists but no grade/health producer.

---

## 6. Mapping to master spec §33 (8 points)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | `EvaluationArtifact / EvaluationBundle` become single source of truth | **PARTIAL** | `EvaluationBundle` EXISTS (api/requests.py:96). `EvaluationArtifact` **MISSING** (no such class). |
| 2 | metric registry single authority | **PARTIAL — TWO catalogs** | One class (registry/metrics.py) but two populated authorities (`registry/metrics.py` module catalog + `metrics/catalog.py` `_REGISTRY`), diverging metric_ids/domains. |
| 3 | typed dependency, no magic string | **EXISTS** | `requires` names artifact-class (`ICSeriesArtifact`) — registry/metrics.py:555, evaluator.py:983-1007; `_ic_method_for_metric`::881. |
| 4 | reporting reads artifact only (never computes) | **EXISTS** | tear_sheet.py:_spec_from_artifact:323; placeholders:306. |
| 5 | summary support platform read model | **PARTIAL** | `EvaluationBundle` summary + `series_refs` (api/requests.py:115); no dedicated typed `factor evaluation summary` read-model contract, no ArtifactRef. |
| 6 | large series/matrix use ArtifactRef | **MISSING** | No ArtifactRef in QE (only in platform/app/contracts/artifact_ref.py); series via in-memory ndarray / `series_refs` strings. |
| 7 | provenance immutable | **EXISTS** | `FrozenMapping` (metric_artifacts.py:74,353); artifact_types all use `FrozenMapping`. |
| 8 | content hash includes data semantics | **EXISTS** | process-stable hash over payload+provenance (metric_artifacts:411; artifact_types `__hash__`); `TreatmentEvaluationArtifact` derived-only content_hash (treatment_evaluation.py:241). |

Additional spec §40: Evidence status must distinguish NOT_COMPUTED / UNAVAILABLE / **INVALID_EVIDENCE** / **LABEL_NOT_MATURE** — currently only NOT_COMPUTED + UNAVAILABLE exist; the two extra states are **MISSING**.

---

## 7. GAP analysis (GAP_ANALYSIS format)

| Requirement | Current implementation (files) | Gap | Action | Reuse/Modify/Add | Priority |
|---|---|---|---|---|---|
| EvaluationArtifact single truth | `EvaluationBundle` only (api/requests.py:96) | no `EvaluationArtifact` type | add canonical frozen `EvaluationArtifact` = typed metric bundle w/ ArtifactRefs + provenance + content_hash | Add | P0 |
| metric registry single authority | `registry/metrics.py` module `_METRIC_CATALOG` vs `metrics/catalog.py` `_REGISTRY`/`CATALOG` (two populated registries) | 2 authorities, overlapping ids (pearson_ic, rank_ic, quantile_spread) diverge | unify to one canonical registry; make `metrics/catalog.py` a read-only view of the same instance | Modify | P0 |
| typed dependency (magic string) | artifact-class `requires` + `ic_method` (artifact_types, registry, evaluator:1010) | catalog `_register` still passes `implementation_id` strings (catalog.py:57) but classes present | keep; verify catalog `requires` is artifact-typed too | Reuse | P2 |
| reporting reads artifact only | `tear_sheet.py:_spec_from_artifact` | legacy `EvaluationResult` flat-field panels (tear_sheet:584-892) still read raw fields | route all 22 panels exclusively through `artifacts` dict | Modify | P1 |
| summary for platform read model | `EvaluationBundle` + `series_refs` (api/requests.py:115) | no typed `factor_summary` read-model DTO; series_refs are bare strings | add summary contract (grade + top metrics + ArtifactRef) | Add | P1 |
| large series/matrix via ArtifactRef | in-memory ndarray / cache_v2 / `series_refs` | QE has no ArtifactRef | add `ArtifactRef` to QE contracts and use for series/matrix payloads | Add | P1 |
| provenance immutable | `FrozenMapping` everywhere (metric_artifacts, artifact_types) | — | keep | Reuse | — |
| content hash includes semantics | stable hash over payload+provenance; derived-only content_hash in treatment | — | keep | Reuse | — |
| Evidence status set (§40) | NOT_COMPUTED/UNAVAILABLE present; tear_sheet string consts | missing `INVALID_EVIDENCE`, `LABEL_NOT_MATURE`; tear_sheet `NOT_COMPUTED`/`UNAVAILABLE` are ad-hoc strings | add the two enum members to `EvidenceStatus`, migrate tear_sheet string consts | Modify | P0 |
| Grade | none | no grade/health evidence, no `grade` in any contract | Add `Grade`/`Health` contracts + rolling health update producer | Add | P1 |

---

## 8. Second-authority / duplicate-implementation findings (documented, NOT resolved)

1. **Two metric catalogs:** `registry/metrics.py` module-level `_METRIC_CATALOG` (register_metric) and `metrics/catalog.py` catalog-scoped `MetricRegistry()` `_REGISTRY` + `CATALOG`. Same `MetricSpec` class, but **two populated authorities** with overlapping (`pearson_ic`, `rank_ic`, `quantile_spread`) yet divergent specs (`ic_method`, domain, tier, implementation_id). `metrics/catalog.py` is explicitly labelled "read-only view over the single authority" but builds a *second independent registry instance* and registers a *different* set of 10-domain specs — so it is a **second catalog**, not a view.
2. **Two `MetricArtifact` classes:** canonical `contracts.metric_artifacts.MetricArtifact` (typed, array payloads, artifact_kind) and reporting `tear_sheet.py.MetricArtifact` (dict `data` wrapper, `computed`+`status`). The reporting one is a deliberate thin wrapper (`from_canonical`), but they share the name and are distinct types — a dual-authority on the artifact name.
3. **Evidence status two sources:** enum `EvidenceStatus` (evidence_status.py:32) vs module string constants `NOT_COMPUTED`/`UNAVAILABLE` in `tear_sheet.py:173-174` — separate from the enum, not enum members; `computed` flag also sits on the reporting wrapper.
4. **`_METRIC_CATALOG` picklability special-casing** — `CANONICAL_METRIC_ALIASES` mappingproxy guarded (registry/metrics.py:506-524) and provenance MappingProxy guarded in `StreamingEvaluationResult.__getstate__` (streaming_evaluator.py:164); consistent but indicates the canonical-mapping authority is embedded in multiple places.
5. `build/lib/quant_evaluator/**` is a stale build mirror of the source (same modules). Tests run against `build/lib` (the 2 skips are from `test_metamorphic.py` referencing kernels "in build/lib"). Should be ignored / excluded from authority analysis.
