"""Matched CUDA microbenchmark for GPUExecutor IC count reuse.

The A path executes GPUExecutor.run loaded in memory from a fixed baseline
commit. The B path uses the working-tree implementation. This isolates the
executor count-reduction / D2H behavior; it is not a public/full-pipeline test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
import types
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
import numpy as np

from quant_evaluator.runtime import gpu_executor as new_module

METRICS = (
    "rank_ic_series", "rank_ic", "ic_std", "ic_ir",
    "pearson_ic_series", "pearson_ic", "pearson_ic_std", "pearson_ic_ir",
)
DEFAULT_BASELINE_REF = "1c5629e6bd5ac8d03051b97b9debe5561f1ac061"
MAX_POOL_GROWTH_BYTES = 4 * 1024**3


def _git(repo_root, *args):
    return subprocess.check_output(["git", "-C", str(repo_root), *args], text=True).strip()


def _load_baseline_module(repo_root, baseline_ref):
    resolved = _git(repo_root, "rev-parse", "--verify", f"{baseline_ref}^{{commit}}")
    source = subprocess.check_output(
        ["git", "-C", str(repo_root), "show",
         f"{resolved}:quant_evaluator/runtime/gpu_executor.py"], text=True,
    )
    module = types.ModuleType("quant_evaluator.runtime.gpu_executor_baseline")
    module.__package__ = "quant_evaluator.runtime"
    exec(compile(source, f"gpu_executor.py@{resolved}", "exec"), module.__dict__)
    return module, source, resolved


def _instrument(module, factor_count, time_count):
    counters = {"count_reductions": 0, "count_d2h_calls": 0,
                "count_d2h_bytes": 0, "all_d2h_calls": 0,
                "all_d2h_bytes": 0}
    original_to_cpu = module._to_cpu

    class CupyProxy:
        def __getattr__(self, name):
            return getattr(cp, name)

        def sum(self, values, *args, **kwargs):
            axis = kwargs.get("axis", args[0] if args else None)
            result = cp.sum(values, *args, **kwargs)
            if (getattr(values, "ndim", None) == 2 and axis == 0
                    and tuple(values.shape) == (time_count, factor_count)
                    and getattr(result, "ndim", None) == 1
                    and result.dtype.kind in "iu"):
                counters["count_reductions"] += 1
            return result

    def measured_to_cpu(value):
        host = np.asarray(original_to_cpu(value))
        counters["all_d2h_calls"] += 1
        counters["all_d2h_bytes"] += host.nbytes
        if (tuple(getattr(value, "shape", ())) == (factor_count,)
                and getattr(getattr(value, "dtype", None), "kind", None) in "iu"):
            counters["count_d2h_calls"] += 1
            counters["count_d2h_bytes"] += host.nbytes
        return host

    module._import_cp = lambda: CupyProxy()
    module._to_cpu = measured_to_cpu
    return counters


def _make_session(factors, labels):
    return SimpleNamespace(
        _staged_factors={"__all__": factors},
        _staged_labels={"next_ret": labels},
        metadata=lambda: {},
    )


def _assert_equal(left, right):
    for field in ("scalar_metrics", "series_metrics", "vector_metrics", "observation_counts"):
        lhs, rhs = getattr(left, field), getattr(right, field)
        if lhs.keys() != rhs.keys():
            raise AssertionError(f"{field} keys differ: {lhs.keys()} != {rhs.keys()}")
        for key in lhs:
            np.testing.assert_array_equal(np.isfinite(lhs[key]), np.isfinite(rhs[key]))
            np.testing.assert_allclose(lhs[key], rhs[key], rtol=1e-10, atol=1e-12,
                                       equal_nan=True, err_msg=f"{field}.{key}")
            if field == "observation_counts":
                np.testing.assert_array_equal(lhs[key], rhs[key], err_msg=f"{field}.{key}")


def _validated_run(module, executor, counters, factor_count):
    counters.update(count_reductions=0, count_d2h_calls=0, count_d2h_bytes=0,
                    all_d2h_calls=0, all_d2h_bytes=0)
    cp.cuda.get_current_stream().synchronize()
    result = executor.run(tuple(f"f{i}" for i in range(factor_count)), METRICS)
    cp.cuda.get_current_stream().synchronize()
    return result, dict(counters)


def _run_timed(executor, factor_count):
    cp.cuda.get_current_stream().synchronize()
    started = time.perf_counter()
    result = executor.run(tuple(f"f{i}" for i in range(factor_count)), METRICS)
    cp.cuda.get_current_stream().synchronize()
    return result, time.perf_counter() - started


def _gpu_info():
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = props.get("name", b"unknown")
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    return {"name": name, "free_bytes_before": int(free_bytes),
            "total_bytes": int(total_bytes)}


def _benchmark_case(old_module, source_hashes, baseline_commit,
                    time_count, asset_count, factor_count, repeats):
    required_free = MAX_POOL_GROWTH_BYTES + 1024**3
    free_before, _ = cp.cuda.runtime.memGetInfo()
    if int(free_before) < required_free:
        raise MemoryError(f"preflight requires {required_free} free GPU bytes; got {int(free_before)}")

    pool = cp.get_default_memory_pool()
    pool_before = int(pool.total_bytes())
    pool.set_limit(size=pool_before + MAX_POOL_GROWTH_BYTES)
    rng = np.random.default_rng(20261003 + factor_count + time_count)
    host_factors = rng.normal(size=(time_count, factor_count, asset_count)).astype(np.float64)
    host_labels = rng.normal(size=(time_count, asset_count)).astype(np.float64)
    host_factors[rng.random(host_factors.shape) < 0.04] = np.nan
    host_labels[rng.random(host_labels.shape) < 0.03] = np.nan
    factors, labels = cp.asarray(host_factors), cp.asarray(host_labels)
    del host_factors, host_labels

    old_executor = old_module.GPUExecutor(_make_session(factors, labels))
    new_executor = new_module.GPUExecutor(_make_session(factors, labels))
    metric_parameters = {
        "rank_ic_series": {"min_assets": 20}, "rank_ic": {"min_assets": 20},
        "ic_std": {"min_assets": 10}, "ic_ir": {"min_assets": 10},
        "pearson_ic_series": {"min_assets": 20}, "pearson_ic": {"min_assets": 20},
        "pearson_ic_std": {"min_assets": 10}, "pearson_ic_ir": {"min_assets": 10},
    }
    old_executor.metric_parameters = metric_parameters
    new_executor.metric_parameters = metric_parameters

    # Counters run once outside timing; original executor module functions are
    # restored even when validation fails.
    old_originals = (old_module._import_cp, old_module._to_cpu)
    new_originals = (new_module._import_cp, new_module._to_cpu)
    old_counts = _instrument(old_module, factor_count, time_count)
    new_counts = _instrument(new_module, factor_count, time_count)
    try:
        old_reference, old_validation = _validated_run(
            old_module, old_executor, old_counts, factor_count)
        new_reference, new_validation = _validated_run(
            new_module, new_executor, new_counts, factor_count)
    finally:
        old_module._import_cp, old_module._to_cpu = old_originals
        new_module._import_cp, new_module._to_cpu = new_originals
    _assert_equal(old_reference, new_reference)
    expected_counts = {"old": 8, "new": 4}
    for variant, counters in (("old", old_validation), ("new", new_validation)):
        if counters["count_reductions"] != expected_counts[variant]:
            raise AssertionError(f"unexpected {variant} count reductions: {counters}")
        if counters["count_d2h_calls"] != expected_counts[variant]:
            raise AssertionError(f"unexpected {variant} count transfers: {counters}")

    # Two warm-up calls per variant; instrumentation has been removed.
    for variant in ("old", "new", "new", "old"):
        executor = old_executor if variant == "old" else new_executor
        result = executor.run(tuple(f"f{i}" for i in range(factor_count)), METRICS)
        cp.cuda.get_current_stream().synchronize()
        _assert_equal(old_reference if variant == "old" else new_reference, result)

    samples = {"old": [], "new": []}
    for _ in range(repeats):
        for variant in ("old", "new", "new", "old"):
            executor = old_executor if variant == "old" else new_executor
            result, elapsed = _run_timed(executor, factor_count)
            _assert_equal(old_reference if variant == "old" else new_reference, result)
            samples[variant].append(elapsed)

    pool_growth = max(0, int(pool.total_bytes()) - pool_before)
    if pool_growth > MAX_POOL_GROWTH_BYTES:
        raise MemoryError(f"CuPy pool growth {pool_growth} exceeds 4 GiB guard")
    old_median = statistics.median(samples["old"])
    new_median = statistics.median(samples["new"])
    return {
        "shape_t_f_n": [time_count, factor_count, asset_count],
        "metrics": list(METRICS),
        "min_assets_by_metric": metric_parameters,
        "warmup_calls_per_variant": 2,
        "abba_blocks": repeats,
        "timing_seconds": samples,
        "median_seconds": {"old": old_median, "new": new_median},
        "median_new_over_old": new_median / old_median,
        "untimed_validation_counters": {"old": old_validation, "new": new_validation},
        "count_values_equal": True,
        "all_outputs_equal": True,
        "cupy_pool_growth_bytes_including_inputs": pool_growth,
        "pool_growth_guard_bytes": MAX_POOL_GROWTH_BYTES,
        "free_gpu_bytes_before_case": int(free_before),
        "input_gpu_bytes": int((time_count * factor_count * asset_count
                                 + time_count * asset_count) * 8),
        "baseline_commit": baseline_commit,
        "source_sha256": source_hashes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-ref", default=DEFAULT_BASELINE_REF,
                        help="fixed Git commit supplying the uncached GPUExecutor implementation")
    parser.add_argument("--repeats", type=int, default=3,
                        help="ABBA blocks per input shape; minimum 2")
    parser.add_argument("--output", type=Path, help="optional compact JSON receipt path")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")

    repo_root = Path(__file__).resolve().parents[2]
    old_module, old_source, baseline_commit = _load_baseline_module(repo_root, args.baseline_ref)
    current_path = repo_root / "quant_evaluator/runtime/gpu_executor.py"
    current_source = current_path.read_bytes()
    old_hash = hashlib.sha256(old_source.encode()).hexdigest()
    current_hash = hashlib.sha256(current_source).hexdigest()
    if old_hash == current_hash:
        raise RuntimeError("baseline and working runtime sources are identical; no A/B comparison")
    source_hashes = {"baseline_gpu_executor_sha256": old_hash,
                     "working_gpu_executor_sha256": current_hash}

    gpu = _gpu_info()
    pool_baseline = int(cp.get_default_memory_pool().total_bytes())
    cases = ((128, 256, 32), (128, 256, 48), (512, 1000, 48))
    results = [
        _benchmark_case(old_module, source_hashes, baseline_commit, t, n, f, args.repeats)
        for t, n, f in cases
    ]
    payload = {
        "benchmark": "GPUExecutor IC count-cache matched microbenchmark",
        "date_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "baseline_ref_requested": args.baseline_ref,
        "baseline_commit_resolved": baseline_commit,
        "python": platform.python_version(),
        "cupy": cp.__version__,
        "gpu": gpu,
        "timing_method": "perf_counter around executor.run, current-stream synchronized immediately before and after; instrumentation is untimed",
        "transfer_interpretation": "int64 count cp.asnumpy calls and bytes; each blocking D2H is a host-visible sync point; direct CUDA sync-call count is not intercepted",
        "memory": {"pool_total_before": pool_baseline,
                   "max_pool_growth_bytes_including_inputs": MAX_POOL_GROWTH_BYTES},
        "scope_limit": "resident-array GPUExecutor.run only; excludes public routing, source load/staging, output assembly, and all other full-pipeline costs",
        "results": results,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
