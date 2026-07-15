# Operator surfaces

`factor_engine` now separates operator **runtime availability** from operator
**DSL submission eligibility**.

| Surface | Purpose | Normal manifest access |
|---|---|---|
| `daily` | Causal scalar operators that generate daily factor panels | Yes |
| `research` | Statistical tests, distributions, matrix/PCA and signal processing | No; import `research_operators` explicitly |
| `unsafe` | Lead/backfill/random/non-deterministic helpers | No; explicit unsafe opt-in only |
| `legacy` | Redundant historical names retained for direct runtime compatibility | No; migrate to canonical names |

The runtime registry still contains compatibility implementations so old
results can be reproduced.  New formulas are validated only against the
`daily` surface.

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

`Lead`, `next`, `bfill`, interpolation using future observations, random
sampling and random-number operators are never available through the normal
factor DSL.
