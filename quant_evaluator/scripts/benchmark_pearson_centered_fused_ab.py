"""Synchronous A/B benchmark against pinned Pearson implementation 249113d55."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import types
import gc

import numpy as np
import cupy as cp


BASE = "249113d554f2a88c03879e0a776042af26db002e"
SOURCE = "quant_evaluator/kernels/gpu/correlation.py"
REPAIR_SOURCE = "quant_evaluator/kernels/gpu/pearson_repair.py"
MEMORY_BUDGET = 4 * 1024**3
MIN_FREE_MEMORY = 5 * 1024**3


def _pinned_module():
    source = subprocess.check_output(["git", "show", f"{BASE}:{SOURCE}"])
    module = types.ModuleType("quant_evaluator.kernels.gpu._pearson_pinned_249113d55")
    module.__package__ = "quant_evaluator.kernels.gpu"
    exec(compile(source, f"{BASE}:{SOURCE}", "exec"), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def _sha256(path):
    with open(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def _sync():
    cp.cuda.Stream.null.synchronize()


def _call(fn):
    _sync()
    started = time.perf_counter()
    result = fn()
    _sync()
    return result, time.perf_counter() - started


def _one_shape(old, t, f, n, seed):
    free_before, device_total = cp.cuda.runtime.memGetInfo()
    if free_before < MIN_FREE_MEMORY:
        raise MemoryError(f"need 5 GiB free before case; only {free_before} bytes available")
    row_bytes = t * f * n * np.dtype(np.float64).itemsize
    input_bytes = row_bytes + t * n * np.dtype(np.float64).itemsize
    mask_bytes = t * f * n
    # Deliberately overcount full-size old-path reduction temporaries.
    conservative_working_set_estimate = input_bytes + mask_bytes + 10 * row_bytes + 16 * t * f * 8
    if conservative_working_set_estimate > MEMORY_BUDGET:
        raise MemoryError(f"conservative working-set estimate {conservative_working_set_estimate} exceeds 4 GiB")
    rng = np.random.default_rng(seed)
    x_host = rng.normal(size=(t, f, n)).astype(np.float64)
    y_host = rng.normal(size=(t, n)).astype(np.float64)
    x_host[rng.random(x_host.shape) < 0.015] = np.nan
    y_host[rng.random(y_host.shape) < 0.012] = np.inf
    x, y = cp.asarray(x_host), cp.asarray(y_host)
    _sync()

    old_fn = lambda: old._pairwise_finite_sums(x, y, 20)
    new_fn = lambda: correlation._pairwise_finite_sums(x, y, 20)
    old_result, _ = _call(old_fn)
    new_result, _ = _call(new_fn)
    old_ic, old_count = old_result
    new_ic, new_count = new_result
    cp.testing.assert_allclose(new_ic, old_ic, rtol=1e-8, atol=1e-10, equal_nan=True)
    if bool(cp.any(old_count != new_count)):
        raise AssertionError("A/B valid-count mismatch")
    if bool(cp.any(cp.isnan(old_ic) != cp.isnan(new_ic))):
        raise AssertionError("A/B finite-mask mismatch")
    valid = cp.isfinite(old_ic) & cp.isfinite(new_ic)
    max_abs_error = float(cp.max(cp.abs(old_ic[valid] - new_ic[valid]))) if bool(cp.any(valid)) else 0.0

    pool = cp.get_default_memory_pool()
    memory_before = int(pool.total_bytes())
    # Compile and warm both variants before any recorded sample.
    for _ in range(2):
        _call(old_fn)
        _call(new_fn)
    warm_memory = int(pool.total_bytes())
    if warm_memory > MEMORY_BUDGET:
        raise MemoryError(f"CuPy pool high-water {warm_memory} exceeds 4 GiB budget")

    samples = {"old_seconds": [], "fused_seconds": []}
    for _ in range(3):
        for variant in ("old", "fused", "fused", "old"):
            _, seconds = _call(old_fn if variant == "old" else new_fn)
            samples[f"{variant}_seconds"].append(seconds)
    result = {
        "shape": [t, f, n],
        "input_seed": seed,
        "dtype": "float64",
        "valid_count_equal": True,
        "finite_nan_mask_equal": True,
        "max_absolute_error": max_abs_error,
        "warmup_runs_each": 2,
        "timed_order": "ABBA x 3",
        "timed_samples": samples,
        "median_seconds": {
            "old": float(np.median(samples["old_seconds"])),
            "fused": float(np.median(samples["fused_seconds"])),
        },
        "speedup_old_over_fused": float(np.median(samples["old_seconds"]) / np.median(samples["fused_seconds"])),
        "pool_bytes_before_warmup": memory_before,
        "pool_bytes_after_warmup": warm_memory,
        "memory_budget_bytes": MEMORY_BUDGET,
        "conservative_working_set_estimate_bytes": conservative_working_set_estimate,
        "free_memory_before_input_bytes": int(free_before),
        "device_total_bytes": int(device_total),
    }
    del old_fn, new_fn, old_result, new_result, old_ic, old_count, new_ic, new_count
    del x, y, x_host, y_host, valid
    gc.collect()
    pool.free_all_blocks()
    free_after, _ = cp.cuda.runtime.memGetInfo()
    result["free_memory_after_cleanup_bytes"] = int(free_after)
    return result


def main():
    global correlation
    from quant_evaluator.kernels.gpu import correlation

    old, old_hash = _pinned_module()
    device = cp.cuda.runtime.getDeviceProperties(0)
    result = {
        "baseline_commit": BASE,
        "baseline_correlation_sha256": old_hash,
        "current_correlation_sha256": _sha256(SOURCE),
        "current_fused_module_sha256": _sha256("quant_evaluator/kernels/gpu/pearson_centered_fused.py"),
        "repair_module_sha256": _sha256(REPAIR_SOURCE),
        "gpu_name": device["name"].decode() if isinstance(device["name"], bytes) else str(device["name"]),
        "cupy_version": cp.__version__,
        "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver_version": cp.cuda.runtime.driverGetVersion(),
        "shapes": [],
    }
    for f in (32, 48):
        result["shapes"].append(_one_shape(old, 512, f, 1000, 610030 + f))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
