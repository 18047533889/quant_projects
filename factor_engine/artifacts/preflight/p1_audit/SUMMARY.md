# GO_PROMPT P1 — operator family correctness + alias/DSL/causality/signature audit (read-only)

Auditor: agent40 · Date: 2026-09-01 · Scope: 1673 canonical (load_all include_research=False) from artifacts/operator_audit/current_head/operator_matrix.csv (live truth).

## Severity ranking

### P0 — genuine look-ahead bug: **NONE found**
No canonical in the audited set shows a real future-reference leak in source. (See P1-audit-improvement: the 16 `leak_detected` flags are heuristic false positives, not bugs.)

### P1 — inconsistency requiring a code-level fix / dedup
| # | Item | Detail | Where |
|---|------|--------|-------|
| P1-1 | `fin_cash_earnings_gap` polars signature is stale/duplicated | Catalog `backend_signatures.polars` = `(earnings,cashflow,scale_base,flow_type)` (4) but the ACTUAL registered polars class `FinCashEarningsGapPolarsNative` has metadata `param_names=['operating_cf','net_income']` (2). Two polars impls for one canonical (`PolarsFundamental_*` vs `FinCashEarningsGapPolarsNative`); the signature audit captured one, runtime binds the other. Recommend dedup / recompute signature. Not a numeric divergence (both compute ocf−ni). | `polars_native/fin_advanced.py:479` |
| P1-2 | `backend_signature_consistent=False` flags are metadata over-flags, not true divergence | All 7 (benchmark_relative_price, float_share_ratio, free_float_share_ratio, fin_cash_earnings_gap, intra_segment_amount/volume_share, ts_sharpe) differ only by variable name or a trailing defaulted extension; none shrinks the pandas reference. The 3 share-ratio/benchmark polars impls use generic `(numerator,denominator)` names. | registration_audit.py:279 |

### P2 — documentation drift (hard doc/code-sync mandate): **MUST fix in a doc-sync pass**
| # | Item | Detail | Where |
|---|------|--------|-------|
| P2-1 | `docs/dsl_allowlist.json` is 68 under live | Static tags `dsl_surface='daily'`, 1421 ops; live daily/native = 1483. 68 real daily canonicals were never backfilled (atr_*/donchian_*/keltner_*/psar_*/supertrend_*/reg_*/state_*/spectral_*/vwap_*/bvc_*/vpin_* etc). Backfill list in `dsl_allowlist_drift.json`. | docs/dsl_allowlist.json |
| P2-2 | `docs/dsl_allowlist.json` is 6 over live | WMA, ewm_corr, ewm_cov, fin_mad are EXTENDED-only (classify=extended; verified present in build_dsl_allowlist('extended'), absent in daily) → remove from daily-tagged static. **ts_ewm_corr / ts_ewm_cov are classified daily but have NO default (pandas) backend (`get()`→None, polars-only), so build_dsl_allowlist('daily') skips them** → keep in static but they are unreachable at daily runtime until a pandas backend lands, or retag as polars-only. These are two distinct root causes, not one "extended-only" bucket. | docs/dsl_allowlist.json |
| P2-3 | Alias→allowlist coverage gap | 29 aliases resolve to extended-only canonicals (e.g. ts_wma→WMA, fin_mad→fin_mean_abs_deviation, fp_beta→rolling_beta_to_market). Functional (registry-lookup works) but undocumented at daily surface. | docs/dsl_allowlist.json |
| P2-4 | README static count drift (pre-existing) | README 1737 vs live 1673. Reconfirm at next doc pass. | README / BACKEND_COVERAGE |
| P2-5 | causality classifier name-token heuristic | See P1-audit-improvement; produces false `leak_detected` that keeps 16 PIT-safe ops out of PRODUCTION_ADMITTED. | mining/direct_use.py:2211 |

## What the audit confirms (no action)
- **Alias graph**: 270 aliases, no dangling/collision/non-flattened (finalize hard-fails). 49 canonicals with ≥2 aliases are benign synonyms. Genuine ambiguity: 0.
- **Causality 16 leak_detected**: all false positives of the `_lead`/`forecast_` name-token rule. Evidence pack per canonical in `causality_findings.json` (reg_forecast_error_pct, ts_*_forecast_error family, group_leader_laggard_exposure, group_tail_lead_score, intra_*_lead_lag_ex_self all PIT-safe).
- **Causality 91 unknown**: all conservative markers for recursive/episode/session_state + required_full_history (86/3/2). No positive leak.
- **Family parity**: pandas vs polars EXACT on all sampled dual-backend ops that ran (ts_rolling, cross_section, fundamental binary, state_since_count). Markov/intra-state/fundamental-period could not be numeric-parity confirmed on an unstructured panel (identical exceptions both backends — harness input restriction, not divergence).

## Report files
- artifacts/preflight/p1_audit/alias_consistency.json
- artifacts/preflight/p1_audit/dsl_allowlist_drift.json
- artifacts/preflight/p1_audit/causality_findings.json
- artifacts/preflight/p1_audit/signature_findings.json
- artifacts/preflight/p1_audit/family_parity_matrix.json
