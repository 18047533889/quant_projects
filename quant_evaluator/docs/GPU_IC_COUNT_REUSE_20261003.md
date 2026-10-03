# GPU IC observation-count reuse (2026-10-03)

## Scope and calculation

For one factor j and one IC family with threshold m, let I[t,j;m] be the
daily correlation produced by the existing pairwise-finite correlation
kernel. Its observation count is

$$
C_j(m)=\sum_t \mathbf{1}\{I[t,j;m]\text{ is finite}\}.
$$

GPUExecutor now calculates this reduction and transfers its result once per
family and min_assets value within one _compute_metrics invocation.
Spearman and Pearson have separate caches; different thresholds do not share
counts. No cache survives into another executor invocation or label horizon.
Each requested metric receives its own host-array copy, so modifying one
metric's count array cannot change another metric's evidence.

The correlation kernels, formulas, parameter defaults and backend admission
rules are unchanged. This is a device-executor optimization, not new auto
routing evidence. It does not broaden the certified shape or metric envelopes.

## Matched executor A/B

Reproduce from the formal repository:

```sh
.venv/bin/python -m quant_evaluator.scripts.benchmark_gpu_ic_count_cache \
  --baseline-ref 1c5629e6bd5ac8d03051b97b9debe5561f1ac061 \
  --repeats 3 --output /tmp/qe-ic-count-ab.json
```

The baseline module is read from the fixed commit and executed in memory;
there is no copied repository or alternate source tree. Both versions call
the same currently installed correlation kernels on the same resident CUDA
arrays. These timings therefore isolate the executor difference and do not
compare complete historical software environments.

The harness uses two warm-up calls per version and three ABBA blocks, with
GPU stream synchronization before and after each timed call. Transfer and
reduction instrumentation runs outside timing and is restored in finally.
The eight metrics include rank/Pearson series, means, standard deviations and
IR, with explicit min_assets values 20 for series/means and 10 for std/IR.
Outputs, finite masks and integer counts are compared. CUDA allocation is
bounded by a pool-growth limit of 4 GiB including inputs and a free-memory
preflight of 5 GiB.

Receipt: [gpu_ic_count_cache_ab_20261003.json](benchmarks/gpu_ic_count_cache_ab_20261003.json).
NVIDIA L20, Python 3.12.3, CuPy 14.2.0:

| Dates / assets / factors | Old median ms | New median ms |
|---|---:|---:|
| 128 / 256 / 32 | 8.223797 | 7.931722 |
| 128 / 256 / 48 | 7.688818 | 7.695519 |
| 512 / 1000 / 48 | 169.164228 | 168.581231 |

Every shape passed parity. Count reductions and count D2H calls changed from
8 to 4; all D2H calls changed from 16 to 12. For 48 factors, count-transfer
bytes changed from 3072 to 1536. The larger shape's pool growth was
2,020,269,056 bytes, below the 4 GiB guard.

These measurements show the redundant operations were removed. They do not
establish a robust throughput improvement: the changes are small and one
shape was slightly slower. No claim of globally fastest execution, full-market
speedup, or zero bugs follows from this microbenchmark.

## Public and real-source checks

The public CUDA/CPU parity test uses 32 factors, 24 dates and 48 assets, missing
values and explicit validity masks. It compares eight IC metrics at
rtol=1e-8 and atol=1e-10. The largest separately observed absolute difference
was 1.665e-16. A focused cache test also checks separate family/threshold
counts and independent writable host copies. Both tests skip on an unavailable
CUDA driver instead of breaking collection in a CPU-only environment.

Independent exposure tests cover beta, liquidity, volatility and momentum
against an orthogonal-design formula; a typed cost-drag fixture checks that
turnover_cost can coexist with rank_ic in one public request.

The real COS ordinary-route receipt
[f48_cap16_ic_count_reuse_ordinary_20261003.json](benchmarks/f48_cap16_ic_count_reuse_ordinary_20261003.json)
covers 2586 dates, 5461 stocks and 48 factors. Requested tile cap 16 selected
the already-certified GPU width 2. Auto took 59.943705 seconds and explicit
CUDA width 2 took 66.986451 seconds; each completed 24 tiles with zero OOM
retries and a 2,586,991,616-byte reported VRAM peak. Route, coverage, factor
identity, reference-value comparison, direct comparison and source-provenance
checks passed. The current source aggregate was
93de2d963d0c03e6b23c80e764dfd4940bc786066464e53f3a73073fcdd9188f.

This real request contains rank_ic, quantile_spread and factor_turnover_rate,
not the eight-metric cache microbenchmark. It is a current-route correctness
and memory check, not matched old/new pipeline timing. Stored CPU reference
receipts do not attest current source-code equivalence. Prior source-axis
benchmarks attest their recorded source snapshots, not this new executor.

## Regression gates

After the final runtime and test edits, the QE suite passed 5,031 tests with
26 skips and 53 warnings in 200.20 seconds. The metric reference/formula
check passed. The four newly added focused tests also passed independently.
