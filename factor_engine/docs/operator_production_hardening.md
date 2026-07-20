# Operator production hardening

The public daily DSL is fail-closed. Registering a runtime does not make it
available to factor authors. Every canonical must be assigned explicitly to one
of: daily, research, unsafe, legacy, or internal. Unknown registrations are
`unclassified` and fail CI.

Key production rules:

- `bfill`, `causal_bfill`, lead/next, random operators and full-sample norms are removed from the runtime.
- `causal_linear_extrapolate` is research-only and unavailable to daily formulas.
- Global diagnostics and model-fitting utilities are research-only.
- `constant` is an internal IR/runtime helper, not a public operator.
- Recursive indicators are stateful, not native Polars expressions.
- Primitive evidence is stored under the final post-dedup canonical name.
- High-risk statistical operators cannot be marked Polars production-safe until
  their required DuckDB NaN and Inf edge evidence exists.
- Polars backends that materialise via `to_numpy` / `np.` / `rolling_map` are
  stripped at finalize. Daily production ops that need Polars must ship
  **expression-native** adapters (see `cleaned_operators/common/polars_daily_native.py`).
- Research-only names must not leak into the `daily` DSL allowlist; use
  `surface="compat"` or an explicit research import instead.

## Speed-oriented backend policy

Optimize for wall-clock speed, not backend completeness theatre.

| Path | When | Why |
|------|------|-----|
| DuckDB SQL pushdown | Production-safe daily trees / maximal SQL subtrees | Avoid Python materialisation |
| `auto` → `hybrid_long` | Data source exposes `scan_polars_long` | Long LazyFrame stay; skip wide unpivot |
| Expression-native Polars | Daily ops in `POLARS_PRODUCTION_SAFE` | True Polars exprs, not pandas bridges |
| Pandas | Research / stripped / unverified | Correctness fallback |

Hard rules:

1. **SQL lowerer** never extracts bare `column` / `literal` leaves under a
   Python parent (that only adds DuckDB IO). A bare-column *root* remains
   fully SQL.
2. **`c_*` → `cs_*` rename** rebuilds `POLARS_PRODUCTION_SAFE` *after* rename
   so evidence keyed as `cs_mean` is not dropped.
3. **Do not force** native Polars or production SQL for extended cheap ops
   (`cbrt`, `truncate`, `is_nan`, `scale`) unless they join the daily surface
   and pass six-way dual evidence. They already SQL-compile in research mode
   (`SQL_RESEARCH_SPEED_CANDIDATES`).
4. Prefer DuckDB long hybrid for cross-sectional work when available; wide-panel
   Polars CS via unpivot is a fallback, not the first choice for large panels.
