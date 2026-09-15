# R07 backend parity and candidate-math ledger — 2026-09-12

Status: **IN_PROGRESS — final frozen-source rerun required.**

## Certified project environment run

Environment: `/home/sunhaiwei/quant_projects/.venv/bin/python_p3_12`; Python 3.12.3, pandas 2.3.3, NumPy 2.2.6, Polars 1.42.1, DuckDB 1.5.4, pytest 9.1.1.

Command: `.venv/bin/python_p3_12 -m pytest tests/backend_parity -ra --junitxml=evidence/r3/R07-backend-parity.xml`

Observed: 1,364 collected; 1,024 passed; 340 skipped; 0 failed; 329.46 seconds. The JUnit artifact preserves every node. `R07-candidate-math-ledger.json` preserves every node's status, exact skip reason, and canonical/backend mapping only when an exact parameter token resolves in the live registry. Unresolved mappings are null and are not inferred.

The full run began with FactorEngine Python source-set hash `dc0b43430f57d0c9de9ccf70c57a818c1474f501cf01fb2ca096154e75bf66a0` and ended with `eec32adb4fa0c42518f7d2a4799a630d0136445320d95fe33fd5a6910ebbe179`; concurrent owned changes occurred. Therefore this is useful node evidence but not a final frozen-source PASS.

Largest exact skip reasons:

- 105: not on the production daily surface
- 47: migrated to recipe or research layer
- 40: legacy broad PolarsLong matrix included removed/non-production canonicals
- 14: legacy protected aliases removed from active surface
- 14: pending extended operators intentionally not production certified
- 13: legacy parameter family outside daily production
- 11: legacy rollout semantics superseded by canonical edge certification
- 10: extended rolling rollout outside certified daily surface
- 10: `ts_quantile` not in daily surface

The JSON ledger contains all remaining reasons and all 340 skipped nodes rather than collapsing them into these summary buckets.

## Candidate math lane

Four actual unverified `pandas_numpy` bindings were resolved from the live registry and tested against independent pandas references on a bounded NaN-bearing panel. The same `_impl_source_hash` used by `math_certificate` and execution identity is recorded.

| Canonical | Actual class | Implementation hash | Candidate result | Production capability |
|---|---|---|---|---|
| `ts_skew` | `MovingSkew` | `3dd6e1745969b6ae` | PASS_LIMITED | research/rejected |
| `ts_kurt` | `StableTsKurt` | `64da960efbc7d9a9` | PASS_LIMITED | research/rejected |
| `ts_quantile` | `TSQuantile` | `33ce48359b9ac37a` | PASS_LIMITED | research/rejected |
| `is_nan` | `IsNaN` | `dc6491d639aec1ec` | PASS_LIMITED | research/rejected |

For every row, catalog production certification and six-gate production certification remain false. Polars and SQL bindings are listed with exact live mappings but explicitly `NOT_RUN`; the limited pandas result does not certify them.

## Dependency-version divergence

The system interpreter is a different matrix: `/usr/bin/python3`, pandas 3.0.5, NumPy 2.5.2, Polars 1.44.1, DuckDB 1.5.5. A separate full run by the SQL owner reported 449 failures, 567 passes and 348 skips; representative node `test_bounded_triple_backend_parity[add-<lambda>]` fails because pandas 3 produces `StringDtype(str)` for the instrument index while the Polars result has `object` dtype. The project-venv result and pandas-3 result must not be merged. Pandas-3 compatibility remains unresolved, not skipped or treated as equivalent.

Artifacts:

- `evidence/r3/R07-backend-parity.xml`
- `evidence/r3/R07-candidate-math-ledger.json`
- `evidence/r3/R01-full-backend-parity.log` (separate system-interpreter run)
