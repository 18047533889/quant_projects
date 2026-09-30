# Bounded batch evaluation options

Use `evaluate_factor_source_batch(source, label_bundle, metrics=..., backend="auto",
max_tile_size=16, gpu_policy=GPUExecutionPolicy())` for factor-separable source metrics.
The caller owns `source.close()`; use its context manager where supported.
The source must preserve the declared snapshot, factor order, complete axes and dtype.
COS sources are sequential: never read a pilot tile and restart the same source at zero.

Backend choices are `auto`, `cpu`, and `cuda_strict`.
`auto` deterministically matches the measured request envelopes, then checks precision,
device capabilities and available VRAM. An unknown request falls back to CPU.
This is not a guarantee of the fastest backend for every possible input.
`cuda_strict` explicitly requests CUDA; it does not silently promise numerical support
for unsupported metrics. `max_tile_size` is a cap, not a guarantee of actual read width.
GPU policy controls device selection, VRAM fraction, host-result budget and OOM retiling.
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
