# QE automatic backend selection and runtime identity

This note distinguishes QE's conservative, evidence-bounded `auto` route from
explicit whole-request calibration. Neither mode is a general promise that every
metric, shape, or future request will use its globally fastest implementation.

## What ordinary `auto` does

For the public `evaluate(...)` facade, omitting `backend` (or passing `None`) is
equivalent to `backend="auto"`. The normal selector in
`runtime/evaluator.py` makes one decision for the complete request. It recognizes
only registered metric/shape/dtype/input envelopes with corresponding public
CPU/CUDA correctness and performance evidence. A request outside those exact
certificates, with unsupported extra parameters, or without GPU admission uses
CPU. An admitted certified CUDA request is dispatched to `cuda_strict`; the
execution receipt and bundle metadata describe the selected route and reason.

These certificates are deliberately narrow, not a general search across every
metric or every possible shape. Consult
[`PUBLIC_RUNTIME_CAPABILITIES.md`](PUBLIC_RUNTIME_CAPABILITIES.md) for the
currently documented public envelopes and their evidence. Passing a subset,
adding an unmeasured metric, changing dtype/shape, or adding special inputs does
not inherit a certificate unless the selector explicitly covers it.

Ordinary `auto` does not cold-run CPU and CUDA to discover a winner. It can,
however, consult a process-local measured-auto candidate registered by an
earlier explicit calibration call. That is an optional exact-candidate reuse
path layered on the certified selector: if no admissible candidate is available,
or its winner is not worth its validation cost, the certified decision remains
in force. A cache miss in this path is not a calibration run. When no measured
candidate matches, ordinary auto does not compute the full calibration identity
or probe thread-runtime identity; its normal certified-route and device checks
still apply.

## Opt-in measured calibration

For a materialized `FactorBatch` and `LabelBundle`, a caller may explicitly pass
`AutoCalibrationOptions` through `evaluate(..., auto_calibration=options)` with
`backend=None` or `backend="auto"`. This is a separate operation from the
ordinary certified selector:

```python
from quant_evaluator import AutoCalibrationOptions, GPUExecutionPolicy, evaluate
from quant_evaluator.runtime.backend_calibration import (
    BoundedCalibrationCache, CalibrationPolicy,
)

options = AutoCalibrationOptions(
    policy=CalibrationPolicy(repetitions=3, warmups=1),
    cache=BoundedCalibrationCache(max_entries=32, ttl_seconds=3600),
)
result = evaluate(
    batch, labels, metrics=("pearson_ic", "pearson_ic_series"),
    backend="auto", gpu_policy=GPUExecutionPolicy(),
    auto_calibration=options,
)
```

On a cold admitted calibration, QE alternates complete explicit CPU and
`cuda_strict` evaluations, compares each paired result under the requested
parity tolerance, and selects by steady-state median; ties choose CPU. A parity
failure is not cached and returns CPU. If CUDA admission fails, QE returns CPU
without attempting the CUDA measurements. Reuse the same options/cache object
within the process: the bounded cache stores route/timing records, not output
bundles. The wall-time budget is soft because an in-flight public evaluation
cannot be safely interrupted. Calibration adds several whole evaluations to
the cold call; it is not a cold-latency optimization.

The facade intentionally rejects advanced inputs with `auto_calibration`
instead of silently omitting them. The current opt-in path is for explicit
materialized batches and labels plus metric IDs and GPU policy; it does not
accept a request object, custom metric parameters/context/evaluator, portfolio
or holding-return inputs, sealed-split inputs, or other specialized evaluation
panels. Use ordinary certified `auto` or an explicit backend for those requests.

After explicit calibration, a later ordinary `auto` request may reuse a
registered candidate only when the exact request, metric IDs/order, policy,
source/runtime identity and GPU admission revalidate. The documented measured
candidate gate requires strict-content calibration, at least one warmup, at
least two measured repetitions, passing parity, and a measured gain exceeding
identity/setup validation cost. The candidate registry is bounded, process
local, uses weak references to its cache owner, and does not retain factor arrays
or result bundles. See
[`benchmarks/public_measured_auto_usage_20261001.md`](benchmarks/public_measured_auto_usage_20261001.md)
for the usage and acceptance details.

## Explicit choices and nested CPU kernels

`backend="cpu"` explicitly fixes the public whole-request route to CPU;
`backend="cuda"`, `"cuda_strict"`, or the accepted `"gpu"` alias requests GPU
execution. Explicit GPU requests do not silently turn into successful CPU
results when device execution is unavailable or fails. `"numba"` and
`"polars"` are not public `evaluate` backend choices; the facade rejects them.

This whole-request CPU/CUDA choice is separate from a metric's own CPU-kernel
dispatch. For example, `compute_quantile_returns_fast(..., use_numba=True)`
uses Numba when it is available and falls back to NumPy if it is not; passing
`use_numba=False` selects NumPy. The public CPU path for daily IC uses the
`compute_daily_ic` exact/NumPy default; its lower-level `numba`, `polars`, and
`gpu` alternatives are explicit metric-kernel choices, not auto candidates.
Consequently, `backend="auto"` does not compare or promise the fastest of
NumPy, Numba, Polars, and CUDA for every individual kernel. Optional package
availability and the selected metric path affect CPU execution time.

The evaluator does not set a universal CPU thread count. Effective BLAS,
OpenMP, and JIT thread behavior depends on the installed libraries and process
environment. Some benchmark records pin thread-related environment variables;
those are properties of those particular measurements, not defaults imposed by
`evaluate`. Do not extrapolate a measured winner across different thread
settings or CPU contention.

## What identity means—and what it does not

Measured route records are keyed by exact materialized input identity, requested
metrics, source identity, runtime/device details, and calibration/GPU policy.
Identity checks and device admission are repeated before adopting a measured
candidate. The normal documented calibration policy uses `strict_full_content`
source checks. `stat_guarded` is a faster opt-in with known metadata-preserving
edit limitations; neither mode identifies Python functions already loaded in
memory. Source receipts describe files on disk, so hot reloads and runtime
monkeypatches are outside that identity guarantee. See
[`benchmarks/source_identity_20261001.md`](benchmarks/source_identity_20261001.md)
and [`benchmarks/calibrated_batch_usage_20260930.md`](benchmarks/calibrated_batch_usage_20260930.md).

The updated `_runtime_fingerprint` in `runtime/backend_calibration.py` records
Python/platform and selected package versions (NumPy, SciPy, CuPy CUDA variants,
Numba, llvmlite and threadpoolctl, when installed), plus the admitted device.
It also records `numba.get_num_threads()` and Numba's active threading layer
when Numba is installed; selected thread-related environment variables; and a
normalized list of active BLAS/OpenMP pools (including API, version,
architecture, threading layer and thread count, but no filesystem paths).
When threadpoolctl is absent, the identity records that availability explicitly.
When it is present but its runtime probe fails, identity creation fails closed
rather than silently dropping pool information.

The runtime probe is a metadata query: it does not execute a JIT kernel or set
global thread masks. It may initialize or query a library's thread-runtime
metadata. During explicit calibration, QE samples this identity immediately
before and after each full public CPU/CUDA call, outside the timed interval. An
observed change invalidates the comparison and its cache write; the result uses
CPU and reports `runtime_changed_cpu_fallback` with
`calibration_record["runtime_stable"] == false`. A passing stable run records
`runtime_stable == true`.

These samples do not detect a runtime setting that changes and reverts entirely
inside a public evaluation call. Runtime identity also does not identify already
loaded Python semantics or guarantee immunity to runtime monkeypatching/hot
reloads. Absence of threadpoolctl means actual pools are not inspected, not
proof that no threaded libraries exist. Treat calibration evidence as specific
to the effective runtime inspected and recalibrate after materially changing it.

## Verification snapshot (2026-10-03)

The final related suite passed **143 tests in 2.79 seconds**, including the
small admitted public CPU/CUDA calibration and cache-receipt tests. There were
no skips in this run. One warning is the existing deliberately multithreaded
fork-cache reset regression's Python deprecation warning. This is not a
multi-year real-COS or all-metric performance certificate.

New regressions cover observed changes during warmups and measured calls,
stable calibration admission, real registry lookup hits and runtime mismatch
rejection, normalized duplicate pools/path omission, optional package absence,
and failure propagation from installed probes. Documentation API imports and
Python syntax were independently checked.

A fresh server-c process without explicit thread env settings observed Numba
32 threads with the `omp` layer, and BLAS/OpenMP pools at 32 threads. The earlier
128×5461×48 quantile benchmarks pinned 2 threads: their 1.48–1.49× CPU kernel
and 6.17× request-cache gains must not be extrapolated to this default setting.

In short, certified `auto` means “use CUDA only inside a narrowly evidenced
route envelope and otherwise use CPU,” not “benchmark every choice and always
pick the fastest.” Explicit calibration can learn and reuse a route for an exact
request/runtime, with added cold-call cost and the identity limitations above.
