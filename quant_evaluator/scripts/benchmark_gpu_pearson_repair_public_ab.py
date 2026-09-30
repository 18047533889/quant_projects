"""Bounded alternating A/B of the full public Pearson API."""
from __future__ import annotations
import argparse
import json
import statistics
import subprocess
import time
import types
from pathlib import Path

import cupy as cp
import numpy as np

from quant_evaluator.kernels.gpu.correlation import batched_pearson_ic

BASELINE_REV = "6984c311f"
BASELINE_PATH = "quant_evaluator/kernels/gpu/correlation.py"
T, F, N = 128, 8, 5000


def _baseline_api():
    src = subprocess.check_output(
        ["git", "show", f"{BASELINE_REV}:{BASELINE_PATH}"], text=True)
    module = types.ModuleType("_pearson_baseline")
    exec(compile(src, f"{BASELINE_REV}:{BASELINE_PATH}", "exec"), module.__dict__)
    return module.batched_pearson_ic


def _measure_pair(old_fn, new_fn, x, y, repeats):
    old_fn(x, y, min_obs=20)
    new_fn(x, y, min_obs=20)
    cp.cuda.Stream.null.synchronize()
    samples = {k: {"wall": [], "event": []} for k in ("legacy", "new")}
    outputs = {}
    for turn in range(repeats):
        order = (("legacy", old_fn), ("new", new_fn))
        if turn & 1:
            order = order[::-1]
        for name, fn in order:
            start, stop = cp.cuda.Event(), cp.cuda.Event()
            t0 = time.perf_counter()
            start.record()
            outputs[name] = fn(x, y, min_obs=20)
            stop.record()
            cp.cuda.Stream.null.synchronize()
            samples[name]["wall"].append((time.perf_counter() - t0) * 1000)
            samples[name]["event"].append(cp.cuda.get_elapsed_time(start, stop))
    medians = {
        key: (statistics.median(samples[key]["wall"]),
              statistics.median(samples[key]["event"]))
        for key in samples
    }
    return medians, outputs["legacy"], outputs["new"]


def _panel(fraction, dtype, seed):
    rng = cp.random.RandomState(seed)
    x = rng.standard_normal((T, F, N), dtype=dtype)
    y = rng.standard_normal((T, N), dtype=dtype)
    if fraction and dtype == cp.float64:
        rows = int(T * F * fraction)
        x.reshape(-1, N)[:rows] = 1e16 + 2 * cp.arange(N, dtype=cp.float64)
        y[:rows // F] = cp.arange(N, dtype=cp.float64)
    return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path(
        "quant_evaluator/docs/benchmarks/gpu_pearson_repair_ab_20260930.json"))
    args = parser.parse_args()
    if not 3 <= args.repeats <= 20:
        parser.error("repeats must be in 3..20")
    old_fn = _baseline_api()
    cases = []
    for idx, (name, fraction, dtype) in enumerate((
        ("safe_float64", 0.0, cp.float64),
        ("mixed_50pct_float64", 0.5, cp.float64),
        ("all_unsafe_float64", 1.0, cp.float64),
        ("float32_bypass", 0.0, cp.float32),
    )):
        x, y = _panel(fraction, dtype, idx + 100)
        timing, (old_ic, old_n), (new_ic, new_n) = _measure_pair(
            old_fn, batched_pearson_ic, x, y, args.repeats)
        old_ic, new_ic = cp.asnumpy(old_ic), cp.asnumpy(new_ic)
        nan_equal = bool(np.array_equal(np.isnan(old_ic), np.isnan(new_ic)))
        counts_equal = bool(np.array_equal(cp.asnumpy(old_n), cp.asnumpy(new_n)))
        error = float(np.nanmax(np.abs(old_ic - new_ic)))
        if not nan_equal or not counts_equal or error > 1e-12:
            raise AssertionError(f"full public API parity failed for {name}")
        old_wall, old_event = timing["legacy"]
        new_wall, new_event = timing["new"]
        cases.append({
            "case": name, "shape": [T, F, N], "dtype": str(dtype),
            "legacy_wall_median_ms": old_wall, "new_wall_median_ms": new_wall,
            "legacy_cuda_event_median_ms": old_event,
            "new_cuda_event_median_ms": new_event,
            "wall_speedup_pct": (old_wall - new_wall) / old_wall * 100.0,
            "max_abs_error": error, "nan_mask_equal": nan_equal,
            "valid_counts_equal": counts_equal,
        })
    result = {
        "schema": "qe_gpu_pearson_selective_repair_public_ab.v1",
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "baseline_revision": BASELINE_REV, "repeats": args.repeats,
        "measurement": "warm, alternating full public API calls; wall and CUDA event medians",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
