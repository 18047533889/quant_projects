# KAMA rolling-window performance audit (2026-09-22)

## Scope

This audit covers only the rolling volatility and direction calculations inside
`factor_preprocess.transforms.smoothing.kama`.  The recursive KAMA state
update, strict-past default, warm-up, missing-data hold rule, and result
alignment are unchanged.

The old implementation scanned each efficiency-ratio window in Python.  The
new implementation uses a NumPy sliding-window **view** for the path sum and
aligned NumPy slices for net direction.  The view does not copy the full
`rows x period_er` matrix.  The remaining recursive state update is still
forward-only.

## Correctness evidence

The focused suite covers `period_er` values 1, 3, 10, and 100; both
`use_current` modes; float32 and float64; two interleaved assets; irregular
rows; duplicate external indices; NaN, infinity, and flat sections; output
index preservation; and prefix invariance.

Command:

```text
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  factor_preprocess/tests/transforms/test_kama_window_equivalence.py \
  factor_preprocess/tests/transforms/test_smoothing_kama_oracle.py \
  factor_preprocess/tests/eligibility/test_executable_smoothing.py -k kama
```

Result: **22 passed, 1 deselected**.

An in-memory old/new comparison used the committed implementation from
`git show HEAD:factor_preprocess/factor_preprocess/transforms/smoothing.py`;
it did not create a repository copy.  On a deterministic 256 asset x 500 date
panel containing missing, infinite, and flat segments, old and new results were
bit-for-bit equal (including the missing mask) for:

- `period_er=1, 10, 100` with `use_current=False`;
- `period_er=10` with `use_current=True`.

## Bounded A/B timings

Both measurements used `OPENBLAS_NUM_THREADS=1`, one warm-up call per
implementation, and the alternating order old/new/new/old repeated three
times.  Values are the median of six calls.

| Input | Old | New | Kernel speedup |
| --- | ---: | ---: | ---: |
| Deterministic stress panel, 256 x 500, `period_er=10` | 0.903650 s | 0.256906 s | 3.52x |
| DataAccess COS factor, TRAIN only, 256 x 267, `period_er=10` | 0.509380 s | 0.163070 s | 3.12x |

The real-data input was
`weekly_4cbd7ca6dccc61dc`, selected by the existing bounded DataAccess loader
from its content-addressed landing manifest.  Asset selection used TRAIN
coverage, only the 267 TRAIN dates were passed to KAMA, and TEST was not
evaluated.  Old and new real-factor outputs were bit-for-bit equal, including
the missing mask.

These timings establish a local KAMA-kernel improvement on the stated hardware
and inputs.  They are not a claim that the complete optimizer is 3.12x faster:
candidate construction, evaluator metrics, other transforms, and data access
remain outside this measurement.
