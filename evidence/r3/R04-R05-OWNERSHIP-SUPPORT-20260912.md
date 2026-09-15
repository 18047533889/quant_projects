# R04/R05 verification — 2026-09-12

Scope: server-c `/home/sunhaiwei/quant_projects`, branch `main`, direct working-tree edits only.

## Runtime

- Python 3.12.3
- pandas 2.3.3
- NumPy 2.2.6
- Polars 1.42.1
- PyArrow 25.0.0
- pytest 9.1.1

## Focused results

The production test files were copied temporarily into `evidence/r3/` solely to avoid an unrelated parent `conftest.py` bootstrap failure, executed, and immediately removed.

```text
pytest evidence/r3/test_r04_tmp.py -k lazy_
12 passed, 13 deselected in 0.51s

pytest evidence/r3/test_r05_tmp.py -k "weighted_mean_scalar_vector_joint_positive_support or two_panel_parity"
10 passed, 19 deselected in 1.35s
```

An independent in-process smoke used an actual Polars `LazyFrame` → Arrow table → pandas `Series` path. It mutated returned values, ndarray views, index names and attrs under pandas Copy-on-Write both disabled and enabled. The second alias read preserved `[1.0, 2.0]` and performed one collect total. Result: `R04_SMOKE_OK 1 ['A']`.

The R05 smoke checked both VWAP and amount-weighted mean for `min_finite=1/2/3` and joint positive support counts 0/1/2/3, including NaN, zero, negative and positive-infinity weights. Scalar and bound vector outputs matched the independent hand calculations. Result: `R05_SCALAR_VECTOR_SMOKE_OK`.

## Full-file limitation

Running both production test files at their normal paths did not reach the tests: the existing session-wide operator certification fixture failed during registry bootstrap on `cs_huber_resid` with `TypeError: _calculate_series has no certifiable implementation identity`. This is outside R04/R05 and was not modified. The focused copies preserve the exact test file bytes while excluding only that unrelated ancestor fixture.

`git diff --check` passed for all five changed files. No commit, push, deployment, dependency installation, branch, worktree, repository copy, or production publication was performed.
