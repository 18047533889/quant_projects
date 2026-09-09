# G12 duplicate top-level binding review

Date: 2026-09-09. Formal tree: `/home/sunhaiwei/quant_projects`; scan snapshot HEAD `d561be0670ae5eb7af8a1113d2d9fc7900fc1861`.

This was a read-only AST scan of Python source under FactorEngine, DataAccess, and FactorPreprocess. Tests, virtual environments, caches, distributions, and egg metadata were excluded. No definition was mechanically deleted.

## Authoritative source result

Excluding generated `build/` trees, 1,626 files parsed with zero errors: 1,323 FactorEngine, 230 DataAccess, and 73 FactorPreprocess files. There were **zero same-module duplicate top-level function/class definitions**.

Targeted bindings each have one authoritative definition:

- `factor_engine/backend/sql_tiers.py`: `is_sql_production_safe` at line 1112; this is the stricter static-allowlist plus certified DuckDB parameter-domain gate.
- `factor_engine/backend/sql_pushdown/emitter.py`: `_int_attr` at line 1406.
- `factor_engine/storage/sources/field_plan.py`: `_table_current_snapshot_only` at line 219.
- `data_access/read/scan_cost.py`: `estimate_scan_cost` 166, `conservative_scan_cost` 353, `assert_scan_cost_usable` 377, and `record_scan_actual` 488.
- `factor_preprocess/factor_preprocess/errors.py`: `MissingFittedStateError` at line 116.

SHA-256:

- `sql_tiers.py`: `a08b3d83debddb861ec0485b3a94ea7647e2aa3986ad6632c6eb6e0950fa5393`
- `sql_pushdown/emitter.py`: `5308e12d0cd62d8567d01fc0883ba19a77e3932f88dca5b457c57ba3543bbd35`
- `field_plan.py`: `0bed8eb69c37b3fc3056a45b10670d2df3418ee8fd7730f844d39723ce8cb4d8`
- `scan_cost.py`: `db743e37d446fda5e90e7cb0df4b1c7ecd11effc58133c4dc21843e60f5d0931`
- `errors.py`: `cd0bdb8b6fa31096b849e7e3bf3954a5a4a259743295058cc56fc569f718cc4`

## Generated build-tree candidates

Including stale/generated `build/lib` copies expands the scan to 3,034 files (FactorEngine 2,455; DataAccess 434; FactorPreprocess 145) and yields 15 duplicate bindings, all in build outputs:

- `factor_engine/build/lib/factor_engine/backend/sql_tiers.py`: `is_sql_production_safe` (973, 1127). The later definition is the stricter parameter-domain-certified replacement; the earlier static-only definition is shadowed inside the stale build copy.
- `factor_engine/build/lib/factor_engine/backend/sql_pushdown/emitter.py`: `_int_attr` (419, 1421).
- `factor_engine/build/lib/factor_engine/storage/sources/field_plan.py`: `_table_current_snapshot_only` (219, 237).
- `data_access/build/lib/data_access/read/scan_cost.py`: `record_scan_actual` (431, 509); the later definition adds shape-key recording while preserving EMA behavior.
- `factor_preprocess/build/lib/factor_preprocess/errors.py`: `MissingFittedStateError` (116, 121).
- `factor_engine/build/lib/factor_engine/cleaned_operators/auto_polars_all.py`: `BetaPolars` (196, 1697), `InterceptPolars` (441, 5647), `ResidualPolars` (765, 9162), and `SlopePolars` (819, 9882).
- `factor_engine/build/lib/factor_engine/cleaned_operators/common/time_series.py`: `TSStdDev` (1632, 2547).
- `factor_engine/build/lib/factor_engine/cleaned_operators/common/cross_sectional.py`: `CrossSectionalRank` (336, 1277), `CrossSectionalScale` (821, 1424), and `CrossSectionalZscore` (859, 1524).
- `factor_engine/build/lib/factor_engine/cleaned_operators/common/elementwise.py`: `Reciprocal` (1129, 1743).
- `factor_engine/build/lib/factor_engine/cleaned_operators/technical/signal.py`: `IfElse` (810, 1157).

These build-tree copies are not the imported authoritative paths in the current source checkout. They remain packaging-hygiene candidates, not grounds for deleting current source implementations.

## Boundaries

The scan detects duplicate top-level `def`, `async def`, and `class` names within one module. It does not prove semantic equivalence, inspect historical refs or unscanned languages, certify package artifacts, or certify old production-worker imports. No business source was changed.
