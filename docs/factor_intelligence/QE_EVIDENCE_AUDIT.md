# QE Evidence Audit (R61-FI-020, plan section 12 Workstream D D0)

Audit date: 2026-09-04. Owner: R61-FI-020 (QE capability audit + named
EvidenceProfiles). Scope: `quant_evaluator/` only — no GPU kernels touched,
no metric semantics changed, no metric computation added (bindings/naming
only).

## D0 Audit findings

### Single MetricRegistry authority (105 metric ids)

- `quant_evaluator/registry/metrics.py` holds the ONE populated
  `MetricRegistry` (`_REGISTRY`, `MetricRegistry` class at :295,
  `MetricSpec` frozen dataclass at :186). `metrics/catalog.py` is a read-only
  `MappingProxyType` view over that same instance (QE-P0-R-C2) — there is no
  second populated catalog.
- Registered ids at audit time: **105**, across 15 `Domain` values
  (IC/RANK_IC/QUANTILE/DRAWDOWN/TURNOVER/TAIL_RISK/COVERAGE/HHI/STABILITY/
  TEMPORAL/PREDICTIVE/QUANTILE_SHAPE/REGIME/DATA_QUALITY/
  RESEARCH_INTEGRITY). Status split: 59 STABLE / 46 EXPERIMENTAL
  (28 STABLE ids are catalog-only with `compute_fn=None` — those are the
  10-domain catalog entries; they are NOT bound by any R61 profile).
- `MetricTier` CORE/EXTENDED/RESEARCH; `MetricStatus`
  STABLE/EXPERIMENTAL/DEPRECATED. Lifecycle BUILDING -> SEALED
  (`seal_metric_registry()`), fail-closed after seal.
- Registry query surface: `list_metrics()`, `get_metric(id)` (KeyError on
  unknown), `list_metrics_by_status/tier`, `resolve_alias`, frozen
  `CANONICAL_METRIC_ALIASES` mappingproxy (pickle-safe via copyreg).

### Evidence status vocabulary (reused, not invented)

- `quant_evaluator/contracts/evidence_status.py`:
  - `EvidenceStatus` — exactly **8** values: `computed`, `not_computed`,
    `unavailable`, `unsupported`, `insufficient_data`, `label_not_mature`,
    `invalid_evidence`, `failed`. Plan section 8 vocabulary maps onto these
    (NOT_APPLICABLE -> UNSUPPORTED path, STALE -> INVALID_EVIDENCE + reason,
    UNKNOWN -> NOT_COMPUTED); **no STALE/UNKNOWN/NOT_APPLICABLE literals
    exist as statuses** and none were added.
  - `EvidenceReasonCode` (str Enum, 30+ machine-readable codes incl.
    P0-QE-002 timing/PIT/hash-mismatch codes) with a fail-closed
    `STATUS_REASON_ALLOWED` matrix.
  - `MetricEvidence` frozen dataclass: COMPUTED requires a canonical
    artifact; LABEL_NOT_MATURE must carry artifact=None; missing evidence is
    a status, **never a fabricated 0.0**.
- Profile/evidence consumers must emit only these statuses; test
  `test_missing_metric_status_vocabulary_is_existing_evidence_status` pins
  the 8-token vocabulary and the absence of near-duplicate inventions.

### Pre-existing profile/preset mechanism (audited before building)

| Mechanism | Where | Shape | Gap that R61-FI-020 fills |
|---|---|---|---|
| `MetricPreset` (legacy) | `registry/presets.py` | plain mutable class; 3 id-lists (`factor_core` 6 / `factor_extended` 11 / `production_daily` 4); `get_preset` raises bare KeyError | no cost class, no target market/frequency, no policy version, no required-inputs, no anti-fabrication check |
| `evaluation_profile_ref` / `ArtifactDomain.EVALUATION_PROFILE` | `contracts/domain_refs.py`, `contracts/evaluation_artifact.py` | typed ref edge (domain/kind/identity/content_hash) | reference slot exists but nothing produced a named profile to point at |
| `EvaluationRequest.metric_ids` default | `api/requests.py:35` | plain tuple `("pearson_ic","rank_ic","coverage")` | no named binding; FO/FP could silently fall back |
| `DomainArtifactRef.of(..., "profile:core")` | adapters e2e tests | ad-hoc string identity | pre-dates named profile registry |

Decision: **EXTEND existing** — no new package; `registry/presets.py`
gains a versioned `EvidenceProfile` registry next to (never replacing) the
legacy `MetricPreset` mechanism, and `registry/__init__.py` re-exports both.
The existing `EvaluationProfile` artifact-domain reference slot is now
addressable by real profile ids.

### Cost classes (new vocabulary, audited to be absent)

No cost/budget class existed at registry level (`runtime/budgets.py` is
numeric execution budgets). R61-FI-020 adds `CostClass` CHEAP < MODERATE <
EXPENSIVE and `GPUPreference` (auto/cpu_reference/cuda, reusing
`BackendPolicy` semantics) and `TargetFrequency` (`cn_1d`) — declared in
`registry/presets.py` beside the profiles.

## D1 Delivery — the five named EvidenceProfiles

All profiles: `policy_id="QE_EVIDENCE_PROFILE"`, `policy_version="1.0.0"`
(SemVer), `target=cn_1d` (CN A-share daily), frozen dataclass, every bound
metric id verified to exist in the 105-id MetricRegistry AND to have a bound
`compute_fn` (no catalog-only placeholder bound). R61-FI-020 adds **no**
metric computation.

| profile_id | cost class | #metric ids | bound metric ids | required_inputs |
|---|---|---|---|---|
| `CHEAP_SCREEN_CN_1D` | cheap | 23 | rank_ic, pearson_ic, ic_ir, ic_median, ic_std, rank_ic_series, pearson_ic_series, pearson_ic_std, pearson_ic_ir, coverage, turnover, quantile_spread, missing_ratio, staleness, outlier_ratio, effective_n, distinct_level_ratio, tie_ratio, label_maturity, bonferroni_correction, benjamini_hochberg_correction, holm_bonferroni_correction, sidak_correction | factor_batch, label_bundle, ICSeriesArtifact, p_values |
| `SHAPE_DIAGNOSTIC_CN_1D` | cheap | 8 | quantile_returns_full, quantile_monotonicity, quantile_curvature, quantile_tail_asymmetry, quantile_adjacent_spread, quantile_extreme_cliff, top_quantile_cliff, bottom_quantile_cliff | factor_batch, label_bundle, QuantileReturnArtifact |
| `FULL_VALIDATION_CN_1D` | moderate | 39 | CHEAP_SCREEN set + SHAPE_DIAGNOSTIC set + long_short_returns, sharpe_ratio, sortino_ratio, win_rate, rank_stability, factor_turnover_rate, missing_timeline, universe_churn | all five input kinds |
| `EXPENSIVE_STATISTICAL_CN_1D` | expensive | 5 | hac_tstat, hac_pvalue, block_bootstrap_ci, subsample_stability, ic_autocorr_lag1 | ICSeriesArtifact |
| `MODEL_FEATURE_DIAGNOSTIC_CN_1D` | moderate | 28 | ic_positive_ratio, rank_ic_positive_ratio, yearly/monthly/quarterly_rank_ic, rolling_rank_ic_mean, rolling_rank_ic_ir, recent_3m/6m/12m_rank_ic, worst_year/quarter_rank_ic, rank_ic_decay_h01_h05_h10_h20, ic_sign_consistency, ic_recent_vs_history_delta, year/quarter/month_consistency, rolling_ic_volatility, rolling_ic_drawdown, ic_sign_flip_rate, change_point_score, cusum_break_score, recent_degradation_score, regime_conditional_ic, regime_worst_ic, regime_dispersion, regime_sign_consistency | ICSeriesArtifact |

Subset ladder (pinned by tests): CHEAP_SCREEN ids ⊂ FULL_VALIDATION ids;
SHAPE_DIAGNOSTIC ids ⊂ FULL_VALIDATION ids; CHEAP_SCREEN ∩
EXPENSIVE_STATISTICAL = ∅; CostClass ordering CHEAP < MODERATE < EXPENSIVE.

Honest gaps (recorded, NOT stubbed): DSR/PBO/CSCV/SPA/White-reality-check
are absent from the registry (capability must say NOT_IMPLEMENTED — the
EXPENSIVE profile description states this rather than fabricating ids);
shape 10/20-adaptive named ids and capacity/cost_drag ids remain out of
scope of R61-FI-020 (later Wave 2 tasks add metrics).

## Test evidence

`quant_evaluator/tests/test_evidence_profiles.py` — **23 passed**:
registration/set equality of the 5 profiles; instance typing; non-empty
unique bindings; cn_1d target + policy version; every bound id exists in the
registry + has compute_fn (anti-fabrication); construction rejects
unregistered metric (UnsupportedMetricError) / duplicates / empty binding;
duplicate registration fails; frozen + hashable; SemVer shape; to_dict
round-trip; CHEAP_SCREEN ⊆ FULL_VALIDATION (strict); SHAPE ⊆ FULL_VALIDATION
(strict); cost ordering; screen ∩ expensive = ∅; unknown profile -> typed
`UnknownEvidenceProfileError` (KeyError subclass); get_profile_or_none;
required_inputs closed vocabulary + covers bound specs' requires; evidence
status vocabulary is exactly the existing 8-token `EvidenceStatus` (no
STALE/UNKNOWN/NOT_APPLICABLE inventions); legacy 3 presets unchanged and
name-disjoint from profiles.

Full-suite regression: repo-root run
`python -m pytest -q quant_evaluator/tests --timeout=300` =
**523 passed, 2 skipped, 0 failed** (baseline 500 passed / 2 skipped;
delta +23 = the new profile tests; the 2 pre-existing skips are documented
build/lib metamorphic contract tests).
