# Backend Coverage Report

Generated: 2026-08-05

> **Data-status warning:** This document is a hand-maintained historical snapshot. Its backend totals and production-status statements are internally contradictory (including the 626/528 Pandas totals and the differing Polars/DuckDB totals), so every numeric count and status claim below is quarantined and **must not be treated as current inventory or certification evidence**. No production completeness claim follows from this document.
>
> For authoritative, current-head results, use the generated inventories and evidence artifacts: `docs/operator_manifest.json`, `benchmarks/operator_manifest.json`, `operator_comprehensive_inventory.csv`, `evidence/operator_upgrade_matrix.yaml`, `evidence/factor_operator_verified.json`, and `evidence/primitive_verified.json`. Check each artifact's recorded commit SHA (`generated_commit_sha`, `commit_sha`, or equivalent) against `git rev-parse HEAD`; a SHA mismatch makes the artifact stale. `evidence/scm_manifest.json` records the repository evidence binding. Do not infer or manually add replacement totals here.

## Quarantined historical snapshot (not current)

| Backend | Coverage | Count |
|---------|----------|-------|
| **Pandas** | 626 (100%) | ✅ Full |
| **Polars** | 554 (88.5%) | ✅ Strong |
| **DuckDB** | 141 (22.5%) | ⚠️ Partial |

## Status

### ✅ Production Ready
- **All 626 operators work in Pandas** (reference implementation)
- **544 operators (86.9%) support Polars** with certified parity
- **132 operators (24.4%) support DuckDB/SQL** pushdown
- Production infrastructure complete (field catalog, DSL gate, service security, evidence)

## Status

### ✅ Production Ready
- **All 528 operators work in Pandas** (reference implementation)
- **All 86 primitive operators support Polars & DuckDB** with certified parity
- Production infrastructure complete (field catalog, DSL gate, service security, evidence)

### ⚠️ Remaining Pandas-only
- **64 operators (10.2%) are Pandas-only**, comprising:
  - **25 `intra_*` minute-aggregation operators** (`intraday_agg`) - source-blocked,
    require a certified minute dataset, excluded from production targets
  - **11 strict `fiscal_*` primitives** (inventory/AR/regression/elasticity) - covered
    by the fiscal_v2 SQL infrastructure
  - **11 `relation_*` snapshot operators** - require relation-snapshot PIT semantics
  - **8 index/event/fin operators** and a few session-aware/scalar/complex-pivot
    operators (`intraday_vwap_deviation`, `arg`/`atan2`, `candlestick_pattern`)

## Primitives (86) - Full Backend Support ✅

All primitive operators support Pandas, Polars, and DuckDB with certified parity:

- Arithmetic: `add`, `subtract`, `multiply`, `divide`, `power`, `abs`, `neg`, `sign`
- Math: `log`, `sqrt`, `exp`, `ceil`, `floor`, `round`, `clip`
- Logic: `and_`, `or_`, `not_`, `xor`, `where`, `coalesce`
- Comparison: `eq`, `ne`, `lt`, `le`, `gt`, `ge`
- Null handling: `is_null`, `is_not_null`, `fill_null`, `safe_div_null`
- Time series: `ts_delay`, `ts_delta`, `ts_pct`, `ts_mean`, `ts_std`, `ts_sum`, `ts_min`, `ts_max`, `ts_rank`, `ts_zscore`
- Cross-sectional: `rank`, `zscore`, `cs_demean`, `cs_std`, `cs_sum`, `cs_count`, `cs_pct_rank`, `cs_mad`, `cs_mad_zscore`
- Statistical: `correlation`, `covariance`, `beta`

## Factor Operators (442)

### Pandas + Polars (92) ✅

These operators have both Pandas reference and Polars fast-path:

- Technical indicators: `EMA`, `SMA`, `WMA`, `MACD_line`, `MACD_signal`, `MACD_hist`, `RSI`, `bollinger_upper`, `bollinger_lower`, `bollinger_mid`
- Volume indicators: Some volume-based operators
- Extended operations: Various composite operators

### Pandas Only (350) ⚠️

**Technical Indicators (16):**
- `CMF`, `CMO`, `DEMA`, `TEMA`, `DMI_plus`, `DMI_minus`, `DX`, `MFI`, `NATR`
- `PPO`, `PPO_hist`, `PPO_signal`, `PVO`, `PVO_hist`, `PVO_signal`
- `TSI`, `TSI_signal`, `UltimateOscillator`, `VortexMinus`, `VortexPlus`

**Candle Patterns (76):**
- All candle pattern operators (`candle_*`, `cdl_*`, `pattern_*`)
- Examples: `candle_body`, `candle_gap`, `cdl_doji`, `cdl_engulfing`, `pattern_hammer`

**Structure Patterns (35):**
- All structure/pivot operators (`ts_pivot_*`, `ts_support_*`, `ts_resistance_*`)
- Examples: `ts_breakout_high`, `ts_confirmed_pivot_high`, `ts_channel_position`

**Fundamental/Fiscal (77):**
- All fiscal operators (`fin_*`)
- Examples: `fin_growth`, `fin_ttm`, `fin_yoy`, `fin_accrual_quality`

**Volume Indicators (42):**
- Most volume-based operators
- Examples: `abnormal_volume`, `volume_zscore`, `turnover_momentum`

**Volatility (9):**
- `NATR`, `parkinson_vol`, `overnight_volatility`, `range_volatility`

**Other (95):**
- Various composite and specialized operators
- Examples: `ADL`, `ChaikinOscillator`, `ForceIndex`, `Supertrend`

## Performance Implications

### Fast Path Available (Polars)
- **544 operators (86.9%)** can use the Polars fast path
- Technical indicators, candle geometry/patterns, structure patterns,
  volume/liquidity, PIT-safe fundamental operators, robust statistics,
  conditional/state-event, group ex-self, limit-state and rolling-regression
  operators all have native backends
- Typical speedup: **2-10x** vs Pandas on large datasets (10M+ rows)

### Pandas Fallback Required
- **70 operators (11.2%)** remain Pandas-only (source-blocked minute
  aggregation, strict fiscal primitives, relation-snapshot, session-aware)
- Fallback is **automatic and safe** - no correctness issues

## Testing Status

- **Full test suite**: 2788 passed, 1044 skipped, 0 failed
- **Parity suites** (each verifies Polars vs Pandas output incl. NaN warmup):
  `test_polars_indicators_v2`, `test_polars_liquidity_v2`, `test_polars_candle`,
  `test_polars_structure`, `test_polars_fundamental`, `test_polars_tech_misc`,
  `test_polars_misc_v2`, `test_polars_misc_utils`, `test_polars_robust_stats`,
  `test_polars_state_event`, `test_polars_cs_misc`, `test_polars_limit_misc`,
  `test_polars_group`, `test_polars_fiscal_v2`
- **Evidence regenerated**: `factor_operator_verified.json`,
  `primitive_verified.json`, `benchmarks/operator_manifest.json`,
  `evidence/operator_upgrade_matrix.yaml`

## Notes

- All operators are **production-ready** - they work correctly in Pandas
- Polars/DuckDB support is a **performance optimization**, not a correctness requirement
- Backend selection is **automatic** - users don't need to know which operators support which backends
- Fallback to Pandas is **transparent and logged** in production mode
