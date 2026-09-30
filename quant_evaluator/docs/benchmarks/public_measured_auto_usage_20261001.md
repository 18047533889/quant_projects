# Public exact-batch measured auto selection

Ordinary `evaluate(batch, label, backend="auto")` continues to use the deterministic
certified workload router, without replaying the evaluation. Its coverage is not
universal: unknown shape/metric combinations conservatively use CPU. Manual
`backend="cpu"` / `"cuda_strict"` remains available.

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
