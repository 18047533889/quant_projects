# Factor-source batch evaluation API

`evaluate_factor_source_batch(source, label_bundle, *, metrics, backend="auto",
max_tile_size=None, gpu_policy=None)` is a public, bounded one-call API. It
covers every declared factor exactly once and returns one columnar
`BatchEvaluationBundle` with arrays indexed by the source's ordered factor IDs.
The caller owns the source and closes it in `finally`.

Current supported metrics are `rank_ic`, `rank_ic_series`, `ic_ir`, `ic_std`,
`ic_median`, `pearson_ic`, `pearson_ic_series`, `pearson_ic_std`,
`pearson_ic_ir`, `coverage`, `quantile_spread`, `quantile_monotonicity`,
`daily_quantile_monotonicity_rate`, `turnover`, and
`factor_turnover_rate`. These scalar/series metrics were checked against the
public CPU evaluator in a combined CPU/CUDA source-batch request and in
missing-value, all-missing, tied-value and factor-masked cases. Unsupported
metrics or special inputs fail closed. This API
does **not** replace the richer `evaluate()` / `EvaluationBundle` contract:
portfolio trajectories, diagnostics, domain artifacts, qualification receipts,
and durable request identity are not present in the columnar result. Do not
publish it as such.

```python
from quant_evaluator import evaluate_factor_source_batch

try:
    result = evaluate_factor_source_batch(
        source, label_bundle,
        metrics=("rank_ic", "rank_ic_series"),
        backend="auto",          # or "cpu" / "cuda_strict"
        max_tile_size=2,         # upper bound on simultaneously loaded factors
    )
    print(result.metadata["backend_used"])
    print(result.scalar_metrics["rank_ic"])          # (F,)
    print(result.series_metrics["rank_ic_series"])   # (T, F)
    print(result.observation_counts["rank_ic"])      # (F,)
finally:
    source.close()
```

`gpu_policy` is an optional `GPUExecutionPolicy`; it controls device IDs,
VRAM fraction, OOM retiling and maximum host result bytes. Explicit
`cuda_strict` never falls back. Explicit `cpu` processes bounded tiles with
the existing public CPU evaluator and aggregates only factor-separable typed
artifacts. `auto` is deterministic: it selects CUDA only for exact
2586-date × 5461-asset float64 panels with either 8 factors or 32 factors
(the latter requires effective source tile width 2), the two-metric
`rank_ic`/`rank_ic_series` set, a single NVIDIA L20, default-compatible
precision policy, and at least 14 GiB effective free VRAM. Otherwise it uses
CPU and records the reason. The F8 route has prior real-COS in-memory and
synthetic source-path evidence; the F32 source route has full-shape synthetic
A/B/A evidence below plus separately certified real-COS in-memory F32 ranks.
Neither result certifies arbitrary source snapshots or metric combinations.

The source snapshot ID is caller-supplied, not an independently verified COS
digest. Source adapters must bind it to a trusted manifest and object set.
Metadata checks before and after every tile reject source drift, axis changes,
reordering, gaps, dtype changes and oversized tiles. A `max_tile_size` override
can only reduce the source's declared maximum. An OOM retry may re-read a
failed range at a smaller width. Results retain the ordered factor axis and
expose backend choice and route reason in `metadata`.

`metadata["source_request_fingerprint"]` is stable across CPU/CUDA and tile
width for the same ordered factor IDs, full axes, factor dtype, supplied
snapshot ID, label content hash and metric order. It is a semantic request
fingerprint **relative to the caller-supplied snapshot ID**, not a cryptographic
verification of COS values. `metadata["execution_receipt"]` records the chosen
backend and GPU/tile options with its own hash; execution choices can change
that receipt without changing the request fingerprint. `request_id` is a fresh
per-call identifier, not a durable content ID.

## Full-history source-path A/B/A (synthetic)

A deterministic on-demand source, not a materialized factor cube, was tested
at 2586 × 5461 × 8 with four reads of two factors. The metrics were
`rank_ic` and `rank_ic_series`:

| Run | Requested → used | Wall time |
| --- | --- | ---: |
| A1 | CPU → CPU | 19.390 s |
| B | auto → CUDA | 2.975 s |
| A2 | CPU → CPU | 19.249 s |

Scalar, series and observation-count arrays matched CPU across all runs.
The GPU reported 2,991,313,408 peak VRAM bytes. Process RSS high-water marks
rose from 1,089,024 to 1,838,616 to 1,879,892 KiB; those are cumulative
whole-process marks, not per-run isolated peaks. The synthetic source includes
random factor generation in every run. The exact same-route real-COS source
test remains pending: this execution context has no usable COS/S3 credential.
The F8 route was not widened based on this result alone.

## Full-history F32 source-path A/B/A (synthetic)

The same deterministic on-demand approach was run at 2586 × 5461 × 32,
with 16 reads of two factors and the `rank_ic`/`rank_ic_series` pair.
Each factor is regenerated from a fixed independent seed for every run.
No complete factor cube or COS object was materialized.

| Run | Requested → used | Wall time |
| --- | --- | ---: |
| A1 | CPU → CPU | 77.855 s |
| B | CUDA strict → CUDA | 10.789 s |
| A2 | CPU → CPU | 77.660 s |

Both metric arrays and their observation-count arrays matched CPU at
rtol=1e-8, atol=1e-10. The CUDA run reported 2,991,313,408 peak VRAM
bytes. A separate exact-shape `auto` run selected CUDA with reason
`bounded_f32_rank_pair_gpu` and took 10.556 s. The source generated random
factors during each read, so these wall times include source generation.
This is not real COS source-path certification and does not establish
the fastest backend for other metrics, dtypes, tile widths or factor counts.

The reproducible [F32 synthetic benchmark report](benchmarks/synthetic_f32_source_aba_20260929.json)
uses the checked-in `benchmark_f32_factor_source_synthetic.py` script.
Its CPU/CUDA/CPU/auto sequence took 77.068/9.978/76.550/9.459 s; `auto`
selected CUDA. Metric values and observation counts matched the first CPU run.
Run the script without flags for a RAM/L20 preflight, or pass `--run-full`
to repeat the bounded benchmark. Do not interpret the synthetic report as
COS I/O performance or full `EvaluationBundle` parity.
