# Loop Engineering Status (compact)

**Loop home:** [`README.md`](README.md) · [`orchestration.md`](orchestration.md)
**Updated:** 2026-08-20

## Hard rules

- **LOCAL ONLY:** no GitHub/`gh`/fetch/push/remotes.
- No `git checkout|restore|stash|clean|reset`; working tree is truth.
- ≤2 disjoint Writers + Finder/Tester/Reviewer per [`orchestration.md`](orchestration.md); Coordinator does not bulk-edit.
- Never claim PASS / production-ready for an unrun gate; broad QE/full repo/root CI stay `NOT_RUN` unless recorded.
- Never load/prompt `docs/R2_HISTORY_ARCHIVE.md`; pass only owned files + ID/evidence pointers.

## Truth matrix

| Surface | Status |
|---|---|
| Q / K | `NOT_PRODUCTION_CERTIFIED` (fail-closed) |
| Polars | `PARTIALLY_VERIFIED` |
| DuckDB | `PARTIALLY_VERIFIED` |
| DataAccess | local focused evidence only; live COS/S3 `NOT_RUN` |
| FactorAssets | `NOT_PRODUCTION_CERTIFIED` |
| Modeling | `NOT_PRODUCTION_CERTIFIED` |
| QuantEvaluator | local focused evidence only; real Redis / broad QE `NOT_RUN` |
| Root CI | `NOT_RUN` |
| Cache v2 | `PARTIALLY_VERIFIED` (dependency invalidation + absolute TTL implemented) |
| Operator Tests | `PARTIALLY_VERIFIED` (132 passed, 96 failed - pre-existing operator implementation issues) |

## Still open / NOT_RUN

- Broad QuantEvaluator suite: previously exit **137** → `NOT_RUN / RESOURCE_BLOCKED`
- Real Redis, root CI, full compile, registry bootstrap, backend parity, PIT poison
- Live COS/S3 DataAccess integration
- Independent review agents: repeatedly fail with **prompt too long** if archive/status is loaded whole
- Residual LQTP-unrunnable RankIC>2% formulas (ADX / RSI_WILDER / pow⅓ / UInt64 mix): 20 left in aug18 materialize
- FactorPreprocess polars_native syntax errors (10 files)
- FactorOptimizer compiler optimization algorithms (R42-124~149)
- Evidence staleness and freshness gates

## Latest local evidence (pointers only)

| ID / topic | Result | Evidence |
|---|---|---|
| FE-CACHE-V2-DEPENDENCY-INVALIDATION | Dependency-based cache invalidation implemented with级联失效 support; absolute TTL support added; 10/10 cache decorator tests pass | `factor_engine/cache/unified_cache.py`, `tests/cache/test_cache_decorator.py` |
| FE-MODELING-NAMESPACE | Namespace migration validated; 20/20 contract tests pass | `modeling/tests/test_contracts.py` |
| FE-OPERATOR-TEST-SUITE-HEALTH | 132 passed, 96 failed (pre-existing operator implementation issues), 21 skipped; test infrastructure fixes applied (registry lifecycle reset, mutation token bypass, mutable dict reset) | `factor_engine/tests/operators/` |
| R21-MATRIX-REGEN-V2 | regen picks up 16 event_state canonicals + polars kernel changes: 1612 canonicals / 4009 rows (no count decreased), direct-use 1553/1553/1380/0/0; R21-MT-P2a pipe-break fixed at the emitter (`_cell` escapes `\|`, flattens \r\n\t, all cells both tables routed through it); polars 0→77 / duckdb 0→79 parity correctly attributed to regenerated primitive_verified.json (provenance SHA 724cf28 = HEAD); 5 new guard tests + 4 pre-existing generator tests = 9 passed serial; review PASS (3 P2s: untracked-file diff caveat, stale FAIL-CLOSED legend text, `\|` downstream contract) → CLOSED_VERIFIED | `factor_engine/evidence/r2/R21-MATRIX-REGEN-V2.yaml` |
| R21-DEF4-TECHMISC-SUPERTRND | live-registry polars_tech_misc Supertrend warmup (row w vs pandas w+1, polars ewm min_samples + ffill vs pandas min_periods=window), post-gap hardcoded bullish re-seed -> UNKNOWN+re-assert, PLUS latent 3rd defect: PSAR kernel mirrored a stale pre-R30 §23 reference (row-0 bull seed, SAR on indeterminate move, raw t-1/t-2 clamps) — all aligned to audited pandas via NumPy kernels; 3 new pandas-direct guards incl. Red Team dead-parameter; owned gate 5/5, adjacent 20/20; review PASS (3 P2/notes, no P0/P1) → CLOSED_VERIFIED | `factor_engine/evidence/r2/R21-DEF4-TECHMISC-SUPERTRND.yaml` |
| R21-DEF5-MANIFEST-SERIALIZE | certify-pipeline refresh_operator_manifest crash fixed: _sql_contract hashed the RAW catalog entry (20 canonicals carry live ParamSpec objects) — new _json_contract_value typed serializer (ParamSpec 11 fields, dtype __name__, ParamRole .value, MISSING sentinel), fail-closed on unknown types, consistent across all 3 consumers; export_operator_manifest exit 0; 7 new tests; pre-existing operator_capability/r40/fastpath failures honestly disclosed (other writers' surfaces); review PASS → CLOSED_VERIFIED | `factor_engine/evidence/r2/R21-DEF5-MANIFEST-SERIALIZE.yaml` |
| R21-P0-EWM-PAIRWISE | ts_ewm_corr/ts_ewm_cov nested-ewm_mean recursion (wrong cov from row 0, 0.197 vs pandas 0.674) rewritten as pandas-delegate kernel: pairwise-finite cohort, Series.ewm(span,adjust=False,ignore_na=False,min_periods=2).corr/.cov, span= alias honored, honest POLARS_PANDAS_DELEGATE; independent pandas oracles, 46 tests passed serial; review PASS (2 P2s: shadowed duplicate test name, inaccurate carry-forward comment) → CLOSED_VERIFIED | `factor_engine/evidence/r2/R21-P0-EWM-PAIRWISE.yaml` |
| R21-P0-EVENTSTATE-FRAMEWORK | 16 new typed canonicals (EVENT_BOOL 3 / SIGNED_EVENT 8 / STATE 5) in event_state_derivations_v1.py; duplicates skipped (event_age/event_decay = existing aliases; state_persistence = documented dwell_pct twin, asserted equal); brute-loop oracles, prefix invariance all 16, causality mutation, sparseness counterexample (finite age/decay where event_recency_z NaN); gates 107 passed serial; review PASS (1 P1 deferral recorded: CategoricalEvent family; 3 P2s) → CLOSED_VERIFIED | `factor_engine/evidence/r2/R21-P0-EVENTSTATE-FRAMEWORK.yaml` |
| R21-DEF1-SUPERTRND | polars SupertrendNative dead-param kernel (returned (h+l)/2 exactly) rewritten as NumPy ratchet state machine: Wilder ATR bit-matched to pandas ewm(adjust=False) incl. gap re-blend, band ratchet, close-cross flip, post-gap UNKNOWN; honest `_SupertrendSpec` POLARS_NUMPY_KERNEL stateful; Red Team 8-bar counterexample + window/multiplier sensitivity tests; 11 new tests, gate 24 passed serial; review PASS (2 P2s) → CLOSED_VERIFIED. Live-registry polars_tech_misc.py Supertrend parity failures remain open (different file, pre-existing) | `factor_engine/evidence/r2/R21-DEF1-SUPERTRND.yaml` |
| R21-P1-POLARS-STATIC-AUDIT | ts_corr native block genuinely rewritten: `_rolling_corr_centered_map_groups` (group_by().map_groups python loop) deleted → pure-Expr `_rolling_corr_centered_expr` (finite-pair mask, shift(k) window `.over(_INST,order_by=_TS)`, anchor+mean double centering); both dispatch sites rewired; audit rules/tier sets byte-identical; gates 38+1skip serial, 200-trial randomized oracle NULL-exact / |diff|≤1.5e-6; unblocks certify pipeline stage; review PASS (2 P2 notes) → CLOSED_VERIFIED | `factor_engine/evidence/r2/R21-P1-POLARS-STATIC-AUDIT.yaml` |
| R21-P0-MATRIX-TRUTH | machine-generated single-truth backend matrix: `scripts/generate_physical_implementation_matrix.py` walks live registry → 1596 canonicals / 3977 rows (pandas 1594 · polars 1595 · duckdb 394 · clickhouse 394); direct-use 1537/1537/1364/0/0; parity=0 recorded as fail-closed stale-evidence truth; docs/BACKEND_COVERAGE.md derived only from generated rows; 4 tests passed serial; review PASS (2 P2s) → CLOSED_VERIFIED | `factor_engine/evidence/r2/R21-P0-MATRIX-TRUTH.yaml` |
| R20-P0-BETA-INTERCEPT | `_prior_beta_stats` -> (alpha, beta, resid_std); beta_residual_z / beta_divergence_pct residuals now r_t-(alpha+beta*m_t); oracle rewritten numpy.polyfit + zero-noise counterexample r=0.01+1.5m kills old impl (old z max 37145); gate 46 passed serial; review PASS (2 P2 notes, no P0/P1) → CLOSED_VERIFIED | `factor_engine/evidence/r2/R20-P0-BETA-INTERCEPT.yaml` |
| R20-RELALPHA-PROMOTION | 6 dimensionless ex-self/prior-beta/group-relative canonicals promoted into `_RELATIVE_ALPHA_OPS` explicit table; ex_self_mean_gap pinned OUT (unit-bearing); 4 tests + 208-passed 10-module R20 merge; review PASS (2 P2s, no P0/P1) → CLOSED_VERIFIED | `factor_engine/evidence/r2/R20-RELALPHA-PROMOTION.yaml` |
| R20-EXEC-CONTRACT-STATEFUL | 23 recursive derivative canonicals declared via `declare_stateful`; 7 tests + 204-passed 9-module merge; review PASS-with-findings (3 P2s fixed/documented) → CLOSED_VERIFIED; DirectUse matrix regen NOT_RUN (queued) | `factor_engine/evidence/r2/R20-EXEC-CONTRACT-STATEFUL.yaml` |
| 8.18 RankIC combo | 197 factors used | `data/cogalpha_lqtp_production/factor_lake_aug18_window/_combo/` |
| Q PlanNode ID collision | fail-closed; commit `19d2ff95` | `factor_engine/evidence/r2/Q2-P0-019-q-plan-node-abi.yaml` |
| Q resident release / batch | focused pass | `Q2-P0-009.yaml`, `Q2-P0-020*`, `Q2-P0-021-022*` |
| Q shared direct namespace | 27 focused tests pass; broader Q gates `NOT_RUN` | `factor_engine/backend/q_backend/test_q_residency.py`, `factor_engine/evidence/r2/Q2-P0-023.yaml` |
| R19 backend/direct-use inventories | current-SHA inventory only; certification `NOT_RUN` | `factor_engine/evidence/r2/BACKEND_GAP_MATRIX.yaml`, `DIRECTUSE_REHABILITATION_MATRIX.yaml` |
| Q `ts_beta` | quarantined (no lowering) | `Q2-P0-008.yaml` |
| POL2 moment/kurt | no defect; oracles added | `POL2-P0-004.yaml` |
| DA semantic catalog identity | latency in correctness key | `dataaccess/DA2_P0_002_MANIFEST.yaml` |
| DA PIT latest revision | focused repair | `dataaccess/R2-P0-039_MANIFEST.yaml` |
| FE multi-region physical plan | local repair | `evidence/r2/R2-P0-059-062-063_manifest.yaml` |
| FA adapter / assembly | focused pass | `FA2-P0-003.yaml`, `FA2-P0-004.yaml` |
| QE cache / streaming / mean_ic | focused pass | `quant_evaluator/tests/test_cache_v2.py` et al. |
| QE-P2-b quick_reference | CLOSED (no module exists; examples standalone) | `quant_evaluator/evidence/qe-full-suite-verify.yaml` |
| QE-P2-e full suite | CLOSED (1540 passed, polars perf reduced) | `quant_evaluator/evidence/qe-full-suite-verify.yaml` |
| FactorAssets legacy packaging | focused pass; 2 tests | `factor_assets/tests/test_legacy_packaging.py` |
| FactorPreprocess registry admission | fail-closed; 44 tests | `factor_preprocess/tests/registry/test_transforms_registry.py` |
| FactorOptimizer sealed test boundary | focused pass; 160 search tests; production remains fail-closed | `factor_optimizer/tests/search/` |
| FactorPreprocess exposure shape | fail-closed; 741 local pass | `tests/adapters/test_data_access.py` |
| QE cache dead-test repair | 2 orphaned regression bodies promoted to tests; 131 pass | `quant_evaluator/tests/test_cache_v2.py` |
| QE-METRIC V2 | 120/120 tests passed, independent review PASS, bug fix applied | `quant_evaluator/VERIFICATION_MANIFEST.yaml` |
| QE-FULL-SUITE-VERIFY | 1540 passed, 25 skipped; registry-spec leak fixed; polars perf reduced; no PIT leakage | `quant_evaluator/evidence/qe-full-suite-verify.yaml` |
| FE-MOCK-IMPORT | 111/111 operators with complete metadata; MISSING + all ParamRole attrs mocked | `factor_engine/test_ts_batch1_metadata_evidence.yaml` |
| Modeling clean wheel gate | 1 pass (venv install, namespace purity) | `modeling/tests/test_wheel_clean_install_smoke.py` |

## How to append

Add **one** short bullet under "Latest local evidence" (ID · result · path).
Move narrative detail into `docs/R2_HISTORY_ARCHIVE.md` or an evidence YAML. Keep this file under ~120 lines.

## Auto-context

Claude/Cursor load repo `CLAUDE.md` → `loop/care.md` only. History → `archives/` (do not load).
