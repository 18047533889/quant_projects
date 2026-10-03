"""Small CPU A/B for TRAIN research decay; never invoked by pytest."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib
import statistics
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from factor_optimizer.research_batch import BatchOptimizationConfig


OLD_COMMIT = "000eadc70"
IGNORED_CACHE_FIELDS = {"assignment_cache_status"}


def _pinned_module(path, name):
    source = subprocess.check_output(["git", "show", f"{OLD_COMMIT}:{path}"], text=True)
    namespace = {"__name__": name}
    exec(compile(source, f"{OLD_COMMIT}:{path}", "exec"), namespace)
    return SimpleNamespace(source=source, namespace=namespace)


def _pinned_old_decay_module():
    pinned = _pinned_module(
        "factor_optimizer/factor_optimizer/research_decay.py",
        "_pinned_old_research_decay",
    )
    return SimpleNamespace(
        diagnose_layer_decay=pinned.namespace["diagnose_layer_decay"],
        source=pinned.source,
    )


def _pinned_old_quantile_module():
    return _pinned_module(
        "quant_evaluator/metrics/quantile.py", "_pinned_old_quantile"
    )


def _legacy_quantile_compute(factor_batch, label_bundle, n_quantiles=5, min_assets=10):
    """Pinned scalar percentile + bincount oracle matching pre-reuse QE."""
    values = np.asarray(factor_batch.values)
    if factor_batch.validity is not None:
        values = np.where(factor_batch.validity, values, np.nan)
    labels = np.asarray(label_bundle.values, dtype=float)
    if label_bundle.validity is not None:
        labels = np.where(label_bundle.validity, labels, np.nan)
    t_count, n_assets, n_factors = values.shape
    out = np.full((t_count, n_quantiles, n_factors), np.nan)
    counts_out = np.zeros((t_count, n_quantiles, n_factors), dtype=np.int32)
    for f in range(n_factors):
        for t in range(t_count):
            row = np.asarray(values[t, :, f], dtype=float)
            finite = np.isfinite(row)
            sorted_values = np.sort(row[finite])
            if len(sorted_values) < n_quantiles:
                continue
            boundaries = []
            for b in range(n_quantiles - 1):
                pos = (b + 1) / n_quantiles * (len(sorted_values) - 1)
                lo, frac = int(pos), pos - int(pos)
                if lo >= len(sorted_values)-1:
                    boundary = sorted_values[-1]
                elif frac < 1e-9:
                    boundary = sorted_values[lo]
                elif frac > 1.-1e-9:
                    boundary = sorted_values[lo+1]
                else:
                    left, right = float(sorted_values[lo]), float(sorted_values[lo+1])
                    delta = right-left
                    boundary = left+frac*delta if np.isfinite(delta) else (1-frac)*left+frac*right
                boundaries.append(boundary)
            ids = np.full(n_assets, -1, dtype=np.int32)
            ids[finite] = np.clip(np.searchsorted(boundaries, row[finite], side="right"),
                                  0, n_quantiles-1)
            good = (ids >= 0) & np.isfinite(labels[t])
            counts = np.bincount(ids[good], minlength=n_quantiles)
            sums = np.bincount(ids[good], weights=labels[t, good], minlength=n_quantiles)
            enough = counts >= min_assets
            counts_out[t, :, f] = counts
            out[t, enough, f] = sums[enough]/counts[enough]
    return out, counts_out


def _case(t, n, seed):
    rng = np.random.default_rng(seed)
    time_axis = AxisRef("time", "int", t, np.arange(t))
    asset_axis = AxisRef("asset", "str", n, np.array([f"a{i}" for i in range(n)]))
    x = rng.normal(size=(t, n))
    x[rng.random((t, n)) < .01] = np.nan
    y = np.roll(x, 1, axis=0) * .002 + rng.normal(0, .01, (t, n))
    y[~np.isfinite(y)] = np.nan
    batch = FactorBatch(("benchmark",), time_axis, asset_axis, x[:, :, None])
    labels = LabelBundle(
        "benchmark", y, 1, decision_time=tuple(range(t)),
        label_start_time=tuple(range(1, t+1)),
        label_end_time=tuple(range(2, t+2)), asset_axis=asset_axis,
    )
    split = SimpleNamespace(train_indices=np.arange(t, dtype=np.int64))
    config = replace(BatchOptimizationConfig(), minimum_train_days=1,
                     minimum_coverage=.1)
    return batch, labels, split, config


def _run(fn, args):
    start = time.perf_counter()
    record = fn(*args, 0, minimum_assets_per_quantile=2)
    return time.perf_counter() - start, record


def _without_cache_metadata(record):
    return {k: v for k, v in record.items() if k not in IGNORED_CACHE_FIELDS}


def benchmark(t, n, seed, repeats=3):
    old_module = _pinned_old_decay_module()
    old_quantile = _pinned_old_quantile_module()
    new_module = importlib.import_module("factor_optimizer.research_decay")
    quantile_module = importlib.import_module("quant_evaluator.metrics.quantile")
    old_compute = quantile_module.compute_quantile_returns
    try:
        args = _case(t, n, seed)
        # Scalar oracle is correctness-only; it is never included in timing samples.
        quantile_module.compute_quantile_returns = _legacy_quantile_compute
        _, oracle_record = _run(old_module.diagnose_layer_decay, args)
        oracle_record = _without_cache_metadata(oracle_record)

        # The timed old path uses the full pre-change QE module from git, including
        # its optimized assigner and every helper referenced by the old compute.
        quantile_module.compute_quantile_returns = old_quantile.namespace[
            "compute_quantile_returns"
        ]
        # One untimed invocation per timed implementation for warmup.
        _, old_warmup_record = _run(old_module.diagnose_layer_decay, args)
        if _without_cache_metadata(old_warmup_record) != oracle_record:
            raise AssertionError("pinned optimized baseline disagrees with scalar oracle")
        quantile_module.compute_quantile_returns = old_compute
        _run(new_module.diagnose_layer_decay, args)
        timings = {"old": [], "new": []}
        # ABBA ordering each round limits thermal/frequency drift bias.
        for _ in range(repeats):
            for name, fn in (("old", old_module.diagnose_layer_decay),
                             ("new", new_module.diagnose_layer_decay),
                             ("new", new_module.diagnose_layer_decay),
                             ("old", old_module.diagnose_layer_decay)):
                if name == "old":
                    quantile_module.compute_quantile_returns = old_quantile.namespace[
                        "compute_quantile_returns"
                    ]
                else:
                    quantile_module.compute_quantile_returns = old_compute
                elapsed, record = _run(fn, args)
                timings[name].append(elapsed)
                if _without_cache_metadata(record) != oracle_record:
                    raise AssertionError(
                        f"{name} output record differs from scalar oracle at sample "
                        f"{len(timings[name])}"
                    )
        old_med = statistics.median(timings["old"])
        new_med = statistics.median(timings["new"])
        return {
            "T": t, "N": n, "seed": seed,
            "old_baseline_commit": OLD_COMMIT,
            "old_quantile_source_sha256": hashlib.sha256(
                old_quantile.source.encode("utf-8")).hexdigest(),
            "old_decay_source_sha256": hashlib.sha256(
                old_module.source.encode("utf-8")).hexdigest(),
            "oracle": "scalar percentile + bincount; correctness only",
            "correctness": "pass for scalar oracle, pinned baseline, and every cached sample",
            "old_seconds_samples": timings["old"],
            "cached_seconds_samples": timings["new"],
            "old_seconds_median": old_med,
            "cached_seconds_median": new_med,
            "speedup_median": old_med/new_med,
            "record_status": oracle_record["status"],
        }
    finally:
        quantile_module.compute_quantile_returns = old_compute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-large", action="store_true",
                        help="also run the 1200 x 5461 single-factor CPU case")
    args = parser.parse_args()
    cases = [(300, 200, 81030), (500, 1000, 81031)]
    if args.include_large:
        cases.append((1200, 5461, 81032))
    for t, n, seed in cases:
        print(benchmark(t, n, seed))


if __name__ == "__main__":
    main()
