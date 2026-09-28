# Factor-source batch evaluation API

`evaluate_factor_source_batch(source, label_bundle, *, metrics, backend="auto",
max_tile_size=None, gpu_policy=None)` is a public, bounded one-call API. It
covers every declared factor exactly once and returns one columnar
`BatchEvaluationBundle` with arrays indexed by the source's ordered factor IDs.
The caller owns the source and closes it in `finally`.

Current supported metrics are `rank_ic`, `rank_ic_series`, `coverage`, and
`quantile_spread`. Unsupported metrics or special inputs fail closed. This API
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
artifacts. `auto` is deterministic: it selects CUDA only for the exact
2586-date × 5461-asset × 8-factor float64 shape, the two-metric
`rank_ic`/`rank_ic_series` set, a single NVIDIA L20, default-compatible
precision policy, and at least 14 GiB effective free VRAM. Otherwise it uses
CPU and records the reason. This is a bounded source-path extrapolation from prior
real-COS in-memory F8 A/B plus the source-path synthetic A/B below; it is
**not** a claim that all source shapes or metrics have a fastest GPU route.

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
random factor generation in every time. The exact same-route real-COS source
test remains pending because the current server account's read-only gateway
rejects the existing factor objects. No route is widened based on this result.
