# Operator surfaces

`factor_engine` separates operator **runtime availability** from operator
**DSL submission eligibility**.

| Surface | Purpose | Normal manifest access |
|---|---|---|
| `daily` | Causal scalar operators that generate daily factor panels (= **production core**) | Yes |
| `compat` | `daily ∪ extended` for already-published packs (e.g. GTJA185) | Published only |
| `extended` | Runtime-available but not daily-certified | No for new manifests |
| `research` | Statistical tests, distributions, matrix/PCA and signal processing | No; import `research_operators` explicitly |
| `unsafe` | Reserved for explicitly reviewed unsafe compatibility tools | No; explicit unsafe opt-in only |
| `legacy` | Redundant historical names retained for direct runtime compatibility | No; migrate to canonical names |

The runtime registry still contains compatibility implementations so old
results can be reproduced.  New formulas are validated only against the
`daily` surface. Production admission additionally requires
`allow_in_production` + dual-backend / DuckDB evidence (see
[`operator_production_hardening.md`](operator_production_hardening.md) and
[`sql_pushdown_coverage.md`](sql_pushdown_coverage.md)).

**Research → production**: the 14 `RESEARCH_ONLY` operators stay research until
they gain dual-backend evidence **and** an explicit surface promotion. Do not
leak them into `daily` via allowlist exceptions.

**Speed**: prefer DuckDB + expression-native Polars (see
[`operator_production_hardening.md`](operator_production_hardening.md)). Extended
ops with SQL emitters (`cbrt` / `truncate` / `is_nan` / `scale`) stay research
SQL-capable; do not promote solely for completeness.

## Canonical replacements

| Removed public DSL name | Use instead |
|---|---|
| `inv`, `reciprocal` | `inverse` |
| `fmax` | `maximum` |
| `fmin` | `minimum` |
| `sqr` | `square` or `power(x, 2)` |
| `cube` | `power(x, 3)` |
| `cumulative_max` | `expanding_max` |
| `cumulative_min` | `expanding_min` |
| `cumulative_mean` | `expanding_mean` |

`Lead`, `next`, `bfill`, misleading interpolation helpers, random sampling,
random-number operators, and full-sample matrix norms have been removed from
the factor runtime. `causal_linear_extrapolate` remains research-only and uses
only historical observations.
