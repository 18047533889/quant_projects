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
artifacts. `backend="auto"` selects a compute backend; it does not create,
replace, or configure `source`, and it does not implicitly enable COS reads or
prefetch. The caller chooses and owns the source adapter, including its read
policy.

Auto routing is deterministic. The historical route envelopes below do not
by themselves prove that the current implementation is the fastest for the
current source contents, package versions, active threads, or device state.
Shape/metric matching is a routing heuristic unless a current, context-bound
qualification receipt is independently validated. Resource and unsupported
input checks remain fail-closed; that is separate from performance evidence.
The F48 default-route report bound to source digest `804aeb378aaacb83033909dc0cfdeb79477da912ad996f666ec897db51996323`
is a historical snapshot, not certification of later code changes.

The historical CUDA envelopes require float64 factors and labels, the default-compatible precision
policy, one NVIDIA L20 device with no extra capability requirements, and at
least 14 GiB effective free VRAM. They are:

- 2586 × 5461 × 8 factors with exactly `rank_ic` and `rank_ic_series`.
- 2586 × 5461 × 32 factors with that same metric pair and an effective source
  tile width of 2, or with the exact three-metric set `rank_ic`,
  `quantile_spread`, and `factor_turnover_rate` at tile width 2.
- 2586 × 5461 × 48 factors with exactly `rank_ic`, `quantile_spread`, and
  `factor_turnover_rate`. Requested tile cap must be exactly 2 or 16; both
  routes execute width 2. A source declaring cap 16 can omit the public API
  `max_tile_size` argument. See [F48 default-route evidence](benchmarks/F48_AUTO_ADMISSION_20261002.md).
- 2586 × 5461 × 61 factors with the exact 15-metric set listed above, with
  `rank_ic` plus `quantile_spread` plus `factor_turnover_rate`, with the single
  metric `pearson_ic`, or with the four-metric Pearson chain
  (`pearson_ic`, `pearson_ic_series`, `pearson_ic_std`, `pearson_ic_ir`).
  The 15-metric route and Pearson routes require a requested tile width of at
  least 16 and use effective width 16. The three-metric route requires a
  requested width of at least 8 and uses width 16 when available, otherwise 8.
- 2400 × 5000 × 61 factors with the exact 15-metric set; requested tile width
  must be at least 16 and the effective width is 16.

Other shapes, metrics, dtypes, precision policies, devices, or resource
conditions use CPU and record the rejection reason. These are narrow profiles,
not a claim that CUDA is best for arbitrary source snapshots. The F8 route has
real-COS in-memory and synthetic source-path evidence; the F32 source route
has full-shape synthetic A/B/A evidence below plus separately certified
real-COS in-memory F32 ranks. The F61 all-source route also has a real-COS
source-adapter comparison for its exact 2400 × 5000 × 61 profile; this does
not certify other source snapshots or combinations.

For COS input, the optional
`quant_evaluator.adapters.cos_factor_tile_source.CosFactorTileSource.from_data_access`
factory binds reads through caller-injected DataAccess context factories and
verified manifest helpers. Its `prefetch="auto"` default uses at most two
independent reads in flight, preserves source order, and enforces its source
and prefetch memory budgets. `prefetch="off"` reads serially. This adapter
default applies only when the caller explicitly constructs and passes this
COS source; `evaluate_factor_source_batch(..., backend="auto")` never
discovers COS credentials, selects a manifest, or switches another source to
this adapter. The caller must close the source in `finally` as shown above.
The measured F61 COS adapter comparison is documented in
[`PUBLIC_RUNTIME_CAPABILITIES.md`](PUBLIC_RUNTIME_CAPABILITIES.md#f61-全部-15-项来源指标整批请求-2026-09-30)
and its machine-readable [prefetch-off report](benchmarks/real_cos_f61_all_source_15_2400d_5000a_cos_adapter_off_20260929.json)
and [prefetch-auto report](benchmarks/real_cos_f61_all_source_15_2400d_5000a_cos_adapter_auto_counts_20260929.json).

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
random factor generation in every run. The exact same-route real-COS source test
for this F8 source profile remains pending in this synthetic benchmark context.
The F8 route was not widened based on this result alone; separate F61 COS
evidence does not extend the F8 profile.

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

## Source-axis assembly measurements

Read [bounded source-axis assembly](SOURCE_AXIS_MATERIALIZATION_20261003.md)
for contiguous/exact-axis paths, overlap behavior, matched local A/B results,
and the 48-factor real COS default-auto verification.
