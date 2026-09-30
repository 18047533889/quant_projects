# Materialized batch calibration

`quant_evaluator.runtime.backend_calibration.evaluate_calibrated_batch` is an opt-in wrapper for a materialized `FactorBatch` and `LabelBundle`. It runs complete public evaluations with `backend="cpu"` and `backend="cuda_strict"`, checks result parity, and returns the bundle from the measured fastest passing route. A parity failure returns the CPU bundle and is not cached.

The cache stores only route and timing records. Its key includes the exact materialized inputs and axes, requested metrics, QE source fingerprint, process nonce, Python/package/runtime/device versions, GPU execution policy, and calibration tolerances. Entries use bounded LRU capacity and TTL and are process-local; they are never shared across workers. If source files drift after the process's first calibration, route records are no longer read or written. A cache hit still rechecks CUDA device admission and invokes the selected explicit backend for the current request.

Example:

```python
from quant_evaluator.runtime.backend_calibration import (
    BoundedCalibrationCache,
    CalibrationPolicy,
    evaluate_calibrated_batch,
)

# Reuse this process-local cache across calls; it does not cache result values.
cache = BoundedCalibrationCache(max_entries=32, ttl_seconds=3600)
result = evaluate_calibrated_batch(
    batch,
    label_bundle,
    metrics=("pearson_ic", "pearson_ic_series"),
    calibration_policy=CalibrationPolicy(
        max_wall_time_seconds=120,
        repetitions=3,
        warmups=1,
    ),
    cache=cache,
)
bundle = result.bundle
route = result.metadata["winner"]
```

The soft budget is checked between complete public calls; an in-flight evaluation cannot be safely interrupted. Warmup calls are excluded from steady-state medians, and first-call timings are reported separately. The default auto router does not use this wrapper. See [the bounded synthetic run](calibrated_batch_public32_20260930.json) for parity, timing, bundle-shape, and cache-hit evidence. That run preceded the final cache fingerprint hardening, so its measured outcomes are smoke evidence rather than a validation of the final cache-key implementation.

The final hardened module was also exercised through real public CPU/CUDA
bundles on a 64 × 512 × 8 panel: parity passed, CPU was selected (25.6 ms
versus 173.6 ms for the first CUDA call), and a second call reused that CPU
route while recomputing a fresh bundle. These are small-request validation
timings, not large-batch speed evidence. Fingerprinting/device/source checks
still cost time (38.3 ms setup on this cache hit); calibration is not a promise
of lower end-to-end latency than an already known explicit backend.
