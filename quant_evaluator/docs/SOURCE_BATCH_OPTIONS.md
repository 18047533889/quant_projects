# Bounded batch evaluation options

## Current-context measured selection

`source_qualification` is an optional tuple of two typed
`CounterbalancedABRecord` objects: one CPU-then-CUDA run and one
CUDA-then-CPU run. It is consumed only by `backend="auto"`.
The records are caller-trusted measurements, not signed producer attestation.
A constructor or a manifest verification callback alone does not authenticate
DataAccess provenance. Never fabricate records from historical timings.

```python
# records comes from your independently verified counterbalanced benchmark.
# Open a fresh source for every evaluation; COS sources are sequential.
with open_source() as source:
    result = evaluate_factor_source_batch(
        source, label_bundle, metrics=metrics, backend="auto",
        max_tile_size=measured_cap, gpu_policy=measured_policy,
        source_qualification=records)
print(result.metadata["execution_receipt"])
```

A valid supplied pair enters a process-local cache (32 entries, one-hour TTL).
Later matching `auto` requests may omit the argument; cache misses do not
launch calibration or read pilot panels. Every cache hit is revalidated.
Forked children clear inherited cache state.

Evidence must match the exact request, selected COS objects and manifest,
all Python source under QE (`quant_evaluator_python_content_v1`), loaded
runtime packages, active thread configuration, initialized CUDA device and
execution policy. Disk source identity does not attest loaded code closure.
The producer can use `capture_source_route_context` from
`quant_evaluator.runtime.source_qualified_router` before and after its pair;
contexts must match. Both measured backends must use the same effective
width, matching the request's current admitted maximum. Smaller-width
measurements do not establish the fastest route for a larger default cap.
Compare every scalar/series output position, masks and observation counts
against an independent numerical reference, not only another backend.

Receipts distinguish missing/rejected evidence and legacy fallback from
`qualified_current_source`. Inspect `source_qualification_applied` and
`source_qualification_cache_status`, not merely the backend name.
If CUDA retiles or retries after OOM, qualification is cleared and the cache
entry evicted; valid output is retained with `execution_configuration_deviated`.
Historical envelope routing remains a heuristic when qualification is absent.
This integration alone is not a current real-COS fastest-backend certificate.

Use `evaluate_factor_source_batch(source, label_bundle, metrics=..., backend="auto",
max_tile_size=16, gpu_policy=GPUExecutionPolicy())` for factor-separable source metrics.
The caller owns `source.close()`; use its context manager where supported.
The source must preserve the declared snapshot, factor order, complete axes and dtype.
COS sources are sequential: never read a pilot tile and restart the same source at zero.

Backend choices are `auto`, `cpu`, and `cuda_strict`.
`auto` deterministically matches historical measured request envelopes, then checks
precision, device capabilities and available VRAM. Matching a historical shape and
metric set does not validate current code, source contents or actual thread settings.
Without current context-bound qualification, treat this route as a heuristic, not
a current fastest-backend certificate. An unknown request falls back to CPU.
This is not a guarantee of the fastest backend for every possible input.
`cuda_strict` explicitly requests CUDA; it does not silently promise numerical support
for unsupported metrics. `max_tile_size` is a cap, not a guarantee of actual read width.
GPU policy controls device selection, VRAM fraction, host-result budget and OOM retiling.
`GPUExecutionPolicy.max_factor_tile_size` is an optional positive integer cap on each
GPU factor tile. It is a GPU execution limit; CPU source execution is unchanged. The
session selects a fitting power-of-two tile below the cap and may reduce it further
after OOM. Static batch auto uses the default tile policy, so custom caps route
materialized requests to CPU. Source auto retains a certified width when the cap is at
least that width and falls back to CPU when the cap would reduce it. Measured auto can
measure capped policies.
Source assembly budget and GPU/result budgets are distinct.

For example, keep the existing numerical policy and request bounded execution:

```python
policy = GPUExecutionPolicy(device_ids=(0,), max_vram_fraction=0.75,
                            max_host_result_bytes=256 * 1024**2, oom_retile=True)
result = evaluate_factor_source_batch(source, label_bundle,
    metrics=("pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir"),
    backend="auto", max_tile_size=16, gpu_policy=policy)
print(result.metadata["execution_receipt"])
```

The real COS benchmark accepts `--max-source-memory-mib` (default 4096).
It forwards this cap to every COS run and isolated worker, and checks available RAM
with at least 32 GiB and at least the source cap plus 4 GiB of headroom.
This is admission control, not a hard RSS guarantee. See `COS_SOURCE_MEMORY_BUDGET.md`.
Its private tile callback consumes payload-list entries one at a time so previously
assembled frames are not retained through FactorBatch freezing; public callback
contracts and immutable result ownership are unchanged.

For the existing 61-factor Pearson-chain experiment, use the established source
adapter/axis index, `--factors 61 --tile-size 16 --max-total-mib 6144`,
`--max-source-memory-mib 12288 --source-adapter cos --verify-auto`, and metrics
`pearson_ic,pearson_ic_series,pearson_ic_std,pearson_ic_ir`.
Do not treat this explicit example budget as a default or a proof of optimality.
Report whole-request and source-read timings separately, compare values/masks/counts,
and alternate run order before extending any automatic routing envelope.

For explicit four-worker IO experiments, additionally pass
`--cos-prefetch-workers 4 --max-prefetch-memory-mib 1024`.
Default values remain two workers and 512 MiB. These options are COS-only;
the harness rejects nondefault worker/budget settings on the legacy adapter.
The options also propagate into isolated CUDA benchmark workers.
A larger allowed worker count does not guarantee an IO speedup; measure the
complete request and keep resource headroom before changing defaults.
