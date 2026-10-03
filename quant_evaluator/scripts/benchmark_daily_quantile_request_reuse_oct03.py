"""Bounded public-evaluator A/B for daily quantile request-local reuse.

Only evaluator.py is pinned to e8fcc4b0a in memory. Both arms share current
adapters and quantile kernels; this does not compare all source modules.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
import types

import numpy as np
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
import quant_evaluator.metrics.quantile as quantile_module
import quant_evaluator.metrics.registry_adapters as adapters

BASELINE_COMMIT = "e8fcc4b0a"
MAX_FACTOR_CELLS = 40_000_000
MAX_ESTIMATED_PEAK_BYTES = 3 * 1024**3
METRICS = (
    "quantile_returns_daily", "quantile_returns_full", "quantile_spread",
    "quantile_monotonicity", "daily_quantile_monotonicity_series",
    "daily_quantile_monotonicity_rate",
)


def _check_shape(time_count, asset_count, factor_count):
    for name, value in (("time_count", time_count), ("asset_count", asset_count),
                        ("factor_count", factor_count)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive builtin integer")
    cells = time_count * asset_count * factor_count
    if cells > MAX_FACTOR_CELLS:
        raise ValueError("factor panel exceeds the 40M-cell cap")
    # FactorBatch freezes/copies the input; reserve for that copy, labels,
    # daily return/count/mask panels, runtime temporaries, and fixed overhead.
    estimated = (cells * 8 * 4 + time_count * asset_count * 8 * 4
                 + time_count * 5 * factor_count * 24 + 256 * 1024**2)
    if estimated > MAX_ESTIMATED_PEAK_BYTES:
        raise ValueError("estimated peak memory exceeds the 3 GiB cap")
    return cells, estimated


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _load_pinned_evaluator():
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE_COMMIT}:quant_evaluator/runtime/evaluator.py"],
        text=False,
    )
    current = importlib.import_module("quant_evaluator.runtime.evaluator")
    name = "quant_evaluator.runtime._pinned_daily_quantile_evaluator"
    module = types.ModuleType(name)
    module.__dict__.update(vars(current))
    module.__name__ = name
    sys.modules[name] = module
    try:
        exec(compile(source, f"<{BASELINE_COMMIT}:runtime/evaluator.py>", "exec"),
             module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.evaluate, source


def _source_hashes(pinned_source):
    evaluator = importlib.import_module("quant_evaluator.runtime.evaluator")
    daily_cache = importlib.import_module("quant_evaluator.runtime.daily_quantile_cache")
    return {
        "baseline_evaluator_sha256": _sha256(pinned_source),
        "optimized_evaluator_sha256": _sha256(Path(evaluator.__file__).read_bytes()),
        "shared_registry_adapters_sha256": _sha256(Path(adapters.__file__).read_bytes()),
        "shared_quantile_kernel_sha256": _sha256(Path(quantile_module.__file__).read_bytes()),
        "daily_quantile_cache_sha256": _sha256(Path(daily_cache.__file__).read_bytes()),
    }


def _make_inputs(time_count, asset_count, factor_count, seed):
    rng = np.random.default_rng(seed)
    shape = (time_count, asset_count, factor_count)
    values = rng.normal(size=shape)
    values[::13, ::19, :] = np.nan
    validity = np.ones(shape, dtype=bool)
    validity[::17, ::23, :] = False
    labels = rng.normal(size=shape[:2])
    labels[::19, ::29] = np.nan
    times = AxisRef("time", "int64", time_count, np.arange(time_count, dtype=np.int64))
    assets = AxisRef("asset", "int64", asset_count, np.arange(asset_count, dtype=np.int64))
    batch = FactorBatch(tuple(f"f{i}" for i in range(factor_count)),
                        times, assets, values, validity=validity)
    del values, validity
    bundle = LabelBundle(
        "daily-quantile-reuse-benchmark", labels, 1,
        decision_time=tuple(range(time_count)),
        label_start_time=tuple(range(1, time_count + 1)),
        label_end_time=tuple(range(2, time_count + 2)),
        asset_axis=assets,
    )
    del labels
    return batch, bundle


def _payload(artifact):
    if not hasattr(artifact, "values"):
        return {"values": np.asarray(artifact)}
    payload = {"values": np.asarray(artifact.values)}
    for field in ("counts", "valid_mask"):
        value = getattr(artifact, field, None)
        if value is not None:
            payload[field] = np.asarray(value)
    provenance = getattr(artifact, "provenance", {})
    if "observation_counts" in provenance:
        payload["observation_counts"] = np.asarray(provenance["observation_counts"])
    return payload


def _compare(baseline, optimized):
    if set(baseline.artifacts) != set(optimized.artifacts):
        raise AssertionError("baseline and optimized artifact IDs differ")
    for metric_id in METRICS:
        left, right = _payload(baseline.artifacts[metric_id]), _payload(optimized.artifacts[metric_id])
        if set(left) != set(right):
            raise AssertionError(f"{metric_id} fields differ: {set(left)} != {set(right)}")
        for field in left:
            left_array, right_array = np.asarray(left[field]), np.asarray(right[field])
            if field == "values" and left_array.dtype.kind in "fc" and right_array.dtype.kind in "fc":
                np.testing.assert_allclose(left_array, right_array, rtol=1e-9, atol=1e-12,
                                           equal_nan=True, err_msg=f"{metric_id}.{field}")
            else:
                np.testing.assert_array_equal(left_array, right_array,
                                              err_msg=f"{metric_id}.{field}")


@contextmanager
def _count_kernel_calls(cpu_kernel):
    force_numba = cpu_kernel == "numba"
    calls = []
    original_q = quantile_module.compute_quantile_returns_fast
    original_a = adapters.compute_quantile_returns_fast

    def counted(*args, **kwargs):
        from inspect import signature
        bound = signature(original_q).bind_partial(*args, **kwargs)
        bound.arguments["use_numba"] = force_numba
        calls.append((bound.arguments.get("n_quantiles", 5),
                      bound.arguments.get("min_assets", 10)))
        return original_q(**bound.arguments)

    quantile_module.compute_quantile_returns_fast = counted
    adapters.compute_quantile_returns_fast = counted
    try:
        yield calls
    finally:
        quantile_module.compute_quantile_returns_fast = original_q
        adapters.compute_quantile_returns_fast = original_a


def _run_one(evaluate_fn, batch, labels, cpu_kernel):
    with _count_kernel_calls(cpu_kernel) as calls:
        started = time.perf_counter()
        result = evaluate_fn(
            batch, labels, metrics=METRICS, backend="cpu",
            quantile_builder_parameters={"n_quantiles": 5, "min_assets": 10},
            metric_parameters={
                "quantile_returns_full": {"min_periods": 20},
                "quantile_spread": {"min_periods": 20},
                "daily_quantile_monotonicity_rate": {"min_periods": 20},
            },
        )
        elapsed = time.perf_counter() - started
    return result, elapsed, calls


def _versions():
    result = {"python": sys.version.split()[0], "platform": platform.platform(),
              "numpy": np.__version__}
    for name in ("numba", "llvmlite", "scipy"):
        try:
            mod = importlib.import_module(name)
            result[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            result[name] = None
    return result


def _thread_info():
    info = {"environment": {name: os.environ.get(name) for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
    try:
        import threadpoolctl
        info["threadpools"] = threadpoolctl.threadpool_info()
    except Exception:
        info["threadpools"] = None
    try:
        import numba
        info["numba_threads"] = numba.get_num_threads()
    except Exception:
        info["numba_threads"] = None
    return info


def _rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def _input_sha256(batch, labels):
    digest = hashlib.sha256()
    for array in (batch.values, batch.validity, labels.values, labels.validity):
        if array is not None:
            digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    digest.update(repr((batch.factor_ids, batch.time_axis.name, batch.asset_axis.name,
                        labels.target_id, labels.decision_time)).encode())
    return digest.hexdigest()


def benchmark(time_count=128, asset_count=5461, factor_count=48, *,
              rounds=3, seed=82109, cpu_kernel="numpy"):
    """Measure cold calls, warmups, then at least three ABBA rounds."""
    cells, estimated = _check_shape(time_count, asset_count, factor_count)
    if type(rounds) is not int or not 3 <= rounds <= 20:
        raise ValueError("rounds must be a builtin integer from 3 through 20")
    if cpu_kernel not in {"numpy", "numba"}:
        raise ValueError("cpu_kernel must be numpy or numba")
    if cpu_kernel == "numba" and not getattr(quantile_module, "_NUMBA_AVAILABLE", False):
        raise RuntimeError("numba kernel requested, but Numba is unavailable")
    pinned_evaluate, pinned_source = _load_pinned_evaluator()
    from quant_evaluator.runtime.evaluator import evaluate as optimized_evaluate
    batch, labels = _make_inputs(time_count, asset_count, factor_count, seed)

    # Baseline is the process first evaluation; optimized is this arm's first
    # call, but follows the baseline call. Both timings remain outside ABBA.
    cold, first_results = {}, {}
    for name, fn in (("baseline", pinned_evaluate), ("optimized", optimized_evaluate)):
        result, elapsed, calls = _run_one(fn, batch, labels, cpu_kernel)
        cold[name] = {"seconds": elapsed, "kernel_calls": len(calls)}
        first_results[name] = result
    reference_result = first_results["baseline"]
    _compare(reference_result, first_results["optimized"])
    del first_results["optimized"]

    warmups = {}
    for name, fn in (("baseline", pinned_evaluate), ("optimized", optimized_evaluate)):
        result, elapsed, calls = _run_one(fn, batch, labels, cpu_kernel)
        _compare(reference_result, result)
        warmups[name] = {"seconds": elapsed, "kernel_calls": len(calls)}
        del result

    samples = {"baseline_seconds": [], "optimized_seconds": []}
    call_hist = {"baseline": [], "optimized": []}
    for _ in range(rounds):
        for name, fn in (("baseline", pinned_evaluate), ("optimized", optimized_evaluate),
                         ("optimized", optimized_evaluate), ("baseline", pinned_evaluate)):
            result, elapsed, calls = _run_one(fn, batch, labels, cpu_kernel)
            _compare(reference_result, result)
            samples[name + "_seconds"].append(elapsed)
            call_hist[name].append(len(calls))
            del result

    medians = {name: int(statistics.median(values)) for name, values in call_hist.items()}
    del reference_result
    report = {
        "shape": [time_count, asset_count, factor_count], "factor_cells": cells,
        "seed": seed, "input_sha256": _input_sha256(batch, labels),
        "sources": _source_hashes(pinned_source),
        "baseline_commit": BASELINE_COMMIT,
        "baseline_scope": "pinned evaluator.py only; adapters and quantile kernels shared/current",
        "metrics": list(METRICS), "cpu_kernel": cpu_kernel,
        "correctness_tolerance": {"rtol": 1e-9, "atol": 1e-12},
        "baseline": {"kernel_calls_per_evaluate": medians["baseline"],
                     "kernel_call_history": call_hist["baseline"],
                     "outputs_match_optimized": True,
                     "median_seconds": statistics.median(samples["baseline_seconds"])},
        "optimized": {"kernel_calls_per_evaluate": medians["optimized"],
                      "kernel_call_history": call_hist["optimized"],
                      "outputs_match_baseline": True,
                      "median_seconds": statistics.median(samples["optimized_seconds"])},
        "cold_first_invocation": {"arm": "baseline", **cold["baseline"]},
        "first_invocation_per_arm": cold, "warmups": warmups,
        "rounds": rounds, "order": "ABBA per round", "samples": samples,
        "estimated_peak_bytes": estimated, "process_peak_rss_bytes": _rss_bytes(),
        "limits": {"factor_cells": MAX_FACTOR_CELLS,
                   "estimated_peak_bytes": MAX_ESTIMATED_PEAK_BYTES},
        "thread_settings": _thread_info(),
        "library_versions": _versions(),
    }
    if medians != {"baseline": 7, "optimized": 1}:
        raise AssertionError("expected per-call quantile kernel counts 7 and 1; got " + repr(medians))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-count", type=int, default=128)
    parser.add_argument("--asset-count", type=int, default=5461)
    parser.add_argument("--factor-count", type=int, default=48)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=82109)
    parser.add_argument("--cpu-kernel", choices=("numpy", "numba"), default="numpy")
    args = parser.parse_args(argv)
    print(json.dumps(benchmark(
        args.time_count, args.asset_count, args.factor_count,
        rounds=args.rounds, seed=args.seed, cpu_kernel=args.cpu_kernel,
    ), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
