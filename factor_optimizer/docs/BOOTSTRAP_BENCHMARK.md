# Moving-block bootstrap scratch benchmark

This bounded CPU microbenchmark compares the previous moving-block lower-bound loop with `factor_optimizer.research_bootstrap.moving_block_lower_bound`. It uses deterministic synthetic float64 inputs only; it does not invoke FactorEngine, GPU code, COS, project datasets, or a complete optimization workflow. These measurements are helper-level evidence, not end-to-end performance claims.

From the server-c project root, the default matrix is:

```bash
cd /home/sunhaiwei/quant_projects
.venv/bin/python factor_optimizer/benchmarks/benchmark_bootstrap_scratch_oct04.py
```

The defaults are `n=1000,2500`, `bootstrap_draws=499,1999`, and three ABBA rounds. For a quick bounded smoke run:

```bash
.venv/bin/python factor_optimizer/benchmarks/benchmark_bootstrap_scratch_oct04.py \
  --sizes 60 --draws 7 --rounds 1
```

`--sizes` and `--draws` accept comma-separated values; the script runs their Cartesian product. Each round times complete old-helper and scratch-helper calls in this order: old, scratch, scratch, old. Fixture generation is outside the timed region. The fixture is seeded with `default_rng(20261004 + n)`, drawn as normal float64 values, with NaN at indices `[::37]`.

The benchmark first checks exact seeded output equality. It also checks every individual timed result against the old implementation and exits with an error on any mismatch; a completed JSON record therefore has `parity: true`. The JSON includes raw timings, per-helper medians, and `speedup_old_over_scratch` (old median divided by scratch median). The reported source SHA256 covers only `factor_optimizer/factor_optimizer/research_bootstrap.py`; it does not fingerprint the benchmark script, tests, Python/NumPy, or host environment.

The SHA256 is a checksum of the on-disk helper file read at run end; it does not prove the loaded implementation used identical bytes. A concurrent source edit could make the digest differ from the code already imported in the benchmark process. For reproducible results, keep the source version fixed throughout each run. The benchmark report is not production or automatic qualification authorization.

A case is rejected before timing if either implementation has no finite lower-bound result. For example, `n=15` is too small to meet the helper's 30-observation per-draw validation minimum, so the CLI exits nonzero without emitting a JSON record. Work is bounded before synthetic arrays are allocated: `n <= 10000`, draws `<= 5000`, rounds `<= 5`, at most 16 Cartesian cases, and aggregate budget `sum(sizes) * sum(draws) * (4 * rounds + 2) <= 400000000`. This is the configured upper-bound proxy for n-by-draw helper sampling work, counting two preflight helper calls plus four timed calls per round; JSON retains the legacy `max_total_operations` field name. It is not a bound on all FLOPs, peak RSS, or elapsed time. The benchmark test suite uses only tiny settings and checks successful output, invalid-result rejection, and budget rejection:

```bash
.venv/bin/pytest -q factor_optimizer/tests/test_benchmark_bootstrap_scratch_oct04.py
```

## Recorded bounded smoke result

One server-c run used `sizes=1000,2500`, `draws=499`, `rounds=1`. Both cases had exact parity. The measured old/scratch times and ratios were:

| n | Old median (s) | Scratch median (s) | Old / scratch |
|---:|---:|---:|---:|
| 1000 | 0.03991079697 | 0.00878381653 | 4.54367379× |
| 2500 | 0.09246394498 | 0.01285571401 | 7.19243948× |

The helper source SHA256 for that run was `d28a409658027209553ef229f08cae9af81a5c0a7c03cc1db73caa3c1371b811`. This is one bounded synthetic CPU smoke result, not a general speed guarantee or evidence of end-to-end optimizer improvement.
