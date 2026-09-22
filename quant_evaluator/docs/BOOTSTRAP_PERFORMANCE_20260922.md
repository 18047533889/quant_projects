# Block-bootstrap kernel check (2026-09-22)

## Change

`compute_block_bootstrap_ci` now constructs bootstrap draw indices in bounded
batches and evaluates each batch with NumPy. The flattened draw order is
unchanged from the former Python list implementation, preserving floating-point
reduction order. Batch sizing uses the padded draw length
`ceil(T / block_length) * block_length`, so even the temporary untrimmed index
matrix is limited to 1,000,000 elements (unless one individual draw itself is
larger than that unavoidable bound).

## Exactness

`quant_evaluator/tests/test_bootstrap_vectorized_equivalence.py` compares the
kernel bit-for-bit with a frozen copy of the former implementation. It covers
six parameter sets, non-divisible lengths (`T=503`), a multi-batch case
(`T=2003`), NaN masking, factor permutation, and strong cancellation using
`1e16, 1, -1e16`.

Focused result:

```
21 passed in 1.70s
```

This includes the new oracle and the existing legacy statistical/calendar
bootstrap tests.

## Bounded A/B timing

Single server-c run in the project virtual environment. Workload:
`T=500`, `F=8`, block length 13, 1,000 draws, seed 20260922. Five rounds
used the alternating order old/new/new/old, producing ten timings per arm.

| Kernel | Median |
| --- | ---: |
| Former Python-list kernel | 0.492342 s |
| Bounded NumPy kernel | 0.015678 s |

Median speedup for this workload and run: **31.403x**. Both arms returned
bit-exact confidence bounds. This is a bounded microbenchmark result, not a
global performance claim.

The main-agent replay used the committed source at 5767f2e41 in memory,
three old/new/new/old rounds and OPENBLAS_NUM_THREADS=1. Medians were
0.495446 s and 0.015760 s (31.44x), with exact confidence-bound equality.
The QE full test suite was concurrently running, so this replay corroborates
the local speedup but is not an isolated-machine latency guarantee.
