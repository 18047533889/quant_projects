"""Small matched GPU A/B for the pinned pre-change rank source and current code.

The pinned source is compiled in memory from Git; no checkout or source copy is
created. Timings use CUDA events after warmup in ABBA order. Pool-memory checks
run separately from timing. Example: python -m quant_evaluator.tests.gpu_rank_ab_harness
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from types import SimpleNamespace

import numpy as np

try:
    import cupy as cp
except ImportError as exc:  # pragma: no cover - depends on optional GPU runtime
    raise SystemExit("CuPy is required for this GPU harness") from exc

from quant_evaluator.kernels.gpu import rank as current_rank


BASELINE = "249113d554f2a88c03879e0a776042af26db002e"
RANK_PATH = "quant_evaluator/kernels/gpu/rank.py"
WORKSPACE_LIMIT = 4 << 30


def _load_baseline():
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:{RANK_PATH}"], text=True
    )
    namespace = {
        "__name__": "quant_evaluator.kernels.gpu._rank_ab_baseline",
        "__package__": "quant_evaluator.kernels.gpu",
    }
    exec(compile(source, f"{BASELINE}:{RANK_PATH}", "exec"), namespace)
    return SimpleNamespace(**namespace)


def _rank_bytes_bound(t, factors, assets, dtype):
    cells = t * factors * assets
    input_bytes = cells * np.dtype(dtype).itemsize
    output_bytes = cells * np.dtype(np.float64).itemsize
    # rank.py caps its conservative per-chunk sort workspace at 1 GiB.
    estimate = input_bytes + output_bytes + (1 << 30)
    if estimate > WORKSPACE_LIMIT:
        raise MemoryError(
            f"estimated rank working set {estimate} exceeds 4 GiB hardguard"
        )
    return estimate


def _event_ms(fn, values, *, return_distinct):
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    result = fn(values, return_distinct=return_distinct)
    stop.record()
    stop.synchronize()
    elapsed = cp.cuda.get_elapsed_time(start, stop)
    del result
    return elapsed


def _pool_delta(fn, values, *, return_distinct):
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    before = pool.total_bytes()
    result = fn(values, return_distinct=return_distinct)
    cp.cuda.get_current_stream().synchronize()
    allocated = max(pool.total_bytes() - before, 0)
    del result
    pool.free_all_blocks()
    if allocated > WORKSPACE_LIMIT:
        raise MemoryError(f"observed CuPy pool growth {allocated} exceeds 4 GiB")
    return allocated


def run(*, t=512, factors=48, assets=1000, repeats=4):
    estimate = _rank_bytes_bound(t, factors, assets, np.float32)
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    if free_bytes < 5 * (1 << 30):
        raise MemoryError("GPU free memory is below the 5 GiB preflight headroom")
    baseline = _load_baseline()
    rng = np.random.default_rng(20261003)
    host = rng.standard_normal((t, factors, assets), dtype=np.float32)
    host[:, 0, :] = np.round(host[:, 0, :], 0)  # exercise ties
    host[..., ::53] = np.nan                    # deterministic invalid positions
    values = cp.asarray(host)
    del host
    cp.cuda.get_current_stream().synchronize()

    report = {
        "baseline": BASELINE,
        "baseline_rank_sha256": hashlib.sha256(subprocess.check_output(
            ["git", "show", f"{BASELINE}:{RANK_PATH}"])).hexdigest(),
        "current_rank_sha256": hashlib.sha256(open(RANK_PATH, "rb").read()).hexdigest(),
        "sorted_rank_runs_sha256": hashlib.sha256(open(
            "quant_evaluator/kernels/gpu/sorted_rank_runs.py", "rb").read()).hexdigest(),
        "cupy_version": cp.__version__,
        "numpy_version": np.__version__,
        "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        "gpu_name": str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
        "free_vram_bytes_preflight": int(free_bytes),
        "total_vram_bytes": int(total_bytes),
        "timing_scope": "warm CUDA event time including stream work, not CPU wall time",
        "memory_scope": "CuPy pool growth in separate passes; not process total peak",
        "shape": [t, factors, assets],
        "dtype": "float32",
        "conservative_estimate_bytes": estimate,
        "workspace_limit_bytes": WORKSPACE_LIMIT,
        "modes": {},
    }
    for with_distinct in (False, True):
        old_fn = baseline.batched_rank
        new_fn = current_rank.batched_rank
        # Warm both implementations and validate exact rank/count parity.
        old = old_fn(values, return_distinct=with_distinct)
        new = new_fn(values, return_distinct=with_distinct)
        if with_distinct:
            cp.testing.assert_array_equal(old[0], new[0])
            cp.testing.assert_array_equal(old[1], new[1])
        else:
            cp.testing.assert_array_equal(old, new)
        del old, new
        cp.cuda.get_current_stream().synchronize()
        _event_ms(old_fn, values, return_distinct=with_distinct)
        _event_ms(new_fn, values, return_distinct=with_distinct)

        # Memory instrumentation is a separate pass from event timing.
        old_pool = _pool_delta(old_fn, values, return_distinct=with_distinct)
        new_pool = _pool_delta(new_fn, values, return_distinct=with_distinct)
        # The memory passes flush the pool: restore symmetric warm allocation
        # state before collecting samples, so A does not pay a cold allocation.
        _event_ms(old_fn, values, return_distinct=with_distinct)
        _event_ms(new_fn, values, return_distinct=with_distinct)
        samples = {"baseline_ms": [], "current_ms": []}
        for _ in range(repeats):
            for impl, fn in (("baseline_ms", old_fn), ("current_ms", new_fn),
                             ("current_ms", new_fn), ("baseline_ms", old_fn)):
                samples[impl].append(
                    _event_ms(fn, values, return_distinct=with_distinct)
                )
        report["modes"]["return_distinct" if with_distinct else "rank_only"] = {
            "baseline_median_ms": statistics.median(samples["baseline_ms"]),
            "current_median_ms": statistics.median(samples["current_ms"]),
            "baseline_samples_ms": samples["baseline_ms"],
            "current_samples_ms": samples["current_ms"],
            "baseline_pool_growth_bytes_separate_pass": old_pool,
            "current_pool_growth_bytes_separate_pass": new_pool,
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors", type=int, choices=(32, 48), default=48)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    print(json.dumps(run(factors=args.factors, repeats=args.repeats), indent=2))


if __name__ == "__main__":
    main()
