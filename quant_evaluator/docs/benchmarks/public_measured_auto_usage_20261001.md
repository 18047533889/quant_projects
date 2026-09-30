# Public exact-batch measured auto selection

Use `evaluate(batch, label, backend="auto")` for the certified workload routes.
You can also reuse accepted exact-batch measurements in that same process, as
described below. A request with neither a certified CUDA route nor a valid
measured candidate uses CPU. Ordinary auto does not run cold calibration, so
its first call on an unknown workload does not establish which backend is fastest.
You can choose `backend="cpu"` or `"cuda_strict"` to bypass auto selection.

To measure the complete current materialized batch instead of relying on a
predefined shape certificate, use the explicit public option below:

```python
from quant_evaluator import AutoCalibrationOptions, GPUExecutionPolicy, evaluate
from quant_evaluator.runtime.backend_calibration import (
    CalibrationPolicy, BoundedCalibrationCache,
)

# Reuse this object within one process; do not construct a new cache each call.
options = AutoCalibrationOptions(
    policy=CalibrationPolicy(
        repetitions=3,
        warmups=1,
        max_wall_time_seconds=120,
        source_check_mode="strict_full_content",  # default assurance
    ),
    cache=BoundedCalibrationCache(max_entries=32, ttl_seconds=3600),
)
bundle = evaluate(
    batch, label_bundle,
    metrics=("pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir"),
    backend="auto",
    auto_calibration=options,
    gpu_policy=GPUExecutionPolicy(max_vram_fraction=0.75),
)
print(bundle.metadata["backend_used"])
print(bundle.metadata["auto_calibration"])
```

The first cache miss executes both CPU and admitted strict CUDA, compares every
paired result (including typed artifacts, counts, masks and axes), and returns
the measured steady-state median winner; timing ties choose CPU. Divergence
returns CPU and is not cached. No CUDA admission also returns CPU. Cache hits
recheck device admission and recompute a fresh bundle on the recorded backend.
The execution receipt records `backend_strategy="calibrated_auto"`; semantic
`config_hash` and result artifacts are unchanged by this execution annotation.

This does **not** reduce initial latency: calibration runs multiple whole calls,
and the budget is checked between calls, not a hard interruption timeout. The
winner is evidence for this exact input/runtime, not a proof of fastest behavior
for future inputs or changed device occupancy. Inputs are content-addressed;
new factor values/masks/axes or metrics cause a miss. Advanced contexts, metric
parameters, portfolio inputs, sealed splits and `EvaluationRequest` are currently
rejected with this option rather than silently dropped. For those requests use
ordinary certified auto or explicit backend selection.

## Reuse a measured route through ordinary auto

Keep `options` alive after the calibration call and use the same GPU policy:

```python
bundle = evaluate(
    batch, label_bundle,
    metrics=("pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir"),
    backend="auto",
    gpu_policy=GPUExecutionPolicy(max_vram_fraction=0.75),
)
print(bundle.metadata["execution_receipt"])
```

QE accepts candidates from strict-content calibration with at least one warmup,
two measured repetitions, passing paired parity, and tolerances no looser than
`rtol=1e-10`, `atol=1e-12`. The measured latency gain must exceed the recorded
identity/setup cost. You may use a different repetition count or wall-time
budget; QE rechecks the candidate with its original calibration policy.

The registry holds up to 32 candidates for one hour, with weak references to
their caches. Keep the options/cache owner alive. QE clears these candidates
after fork and retains no factor arrays or result bundles in the registry.
A cheap descriptor lookup reads existing metadata. A missing candidate, or a
candidate choosing the same backend as the certified route, adds no input
fingerprint or source-content scan. For a different route, QE checks exact
values, masks, axes, metric IDs/order, disk source identity, runtime and device
admission before execution. Changed contents, expired caches and failed checks
fall back to the certified route. QE annotates an adopted route with
`auto_backend_reason="measured_auto_exact_candidate"` and
`auto_backend_profile="exact_content_process_local"`.
Advanced requests retain their existing routing. Manual backend choices remain explicit.
