"""Controlled CPU A/B of QE's standard quantile value API.

Uses a pinned optimized baseline in memory, never copies a checkout or data.
Inputs are synthetic; timings are not COS/GPU or whole-evaluator timings.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
import quant_evaluator.metrics.quantile as quantile_module

BASELINE_COMMIT = "000eadc70"
MAX_FACTOR_CELLS = 60_000_000


def benchmark(time_count=512, asset_count=5461, factor_count=16, seed=81033):
    """Compare exact returns/counts over one warmup and three ABBA rounds."""
    for name, value in (("time_count", time_count), ("asset_count", asset_count),
                        ("factor_count", factor_count)):
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be a positive builtin integer")
    if time_count * asset_count * factor_count > MAX_FACTOR_CELLS:
        raise ValueError("benchmark is bounded to 60 million factor cells")
    raw = subprocess.check_output(
        ["git", "show", BASELINE_COMMIT + ":quant_evaluator/metrics/quantile.py"],
        text=True,
    )
    namespace = {"__name__": "_pinned_quantile_ab"}
    exec(compile(raw, "pinned_quantile", "exec"), namespace)
    rng = np.random.default_rng(seed)
    shape = (time_count, asset_count, factor_count)
    values = rng.normal(size=shape)
    values[::13, ::19, :] = np.nan
    returns = rng.normal(size=shape[:2])
    returns[::17, ::29] = np.nan
    time_axis = AxisRef("time", "int64", time_count,
                       np.arange(time_count, dtype=np.int64))
    asset_axis = AxisRef("asset", "int64", asset_count,
                        np.arange(asset_count, dtype=np.int64))
    batch = FactorBatch(tuple(f"f{i}" for i in range(factor_count)),
                        time_axis, asset_axis, values)
    del values
    labels = LabelBundle(
        "ab", returns, 1, decision_time=tuple(range(time_count)),
        label_start_time=tuple(range(1, time_count + 1)),
        label_end_time=tuple(range(2, time_count + 2)), asset_axis=asset_axis,
    )
    old = namespace["compute_quantile_returns"]
    new = quantile_module.compute_quantile_returns
    expected = old(batch, labels, 20, 2)
    actual = new(batch, labels, 20, 2)
    for current, reference in zip(actual, expected):
        np.testing.assert_array_equal(current, reference)
    del actual
    samples = {"old": [], "new": []}
    for _ in range(3):
        for name, function in (("old", old), ("new", new),
                               ("new", new), ("old", old)):
            start = time.perf_counter()
            result = function(batch, labels, 20, 2)
            elapsed = time.perf_counter() - start
            for current, reference in zip(result, expected):
                np.testing.assert_array_equal(current, reference)
            samples[name].append(elapsed)
            del result
    return {
        "shape": list(shape), "seed": seed, "baseline_commit": BASELINE_COMMIT,
        "old_quantile_source_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "new_quantile_source_sha256": hashlib.sha256(
            Path(quantile_module.__file__).read_bytes()).hexdigest(),
        "n_quantiles": 20, "min_assets": 2, "warmups": 1,
        "rounds": 3, "order": "ABBA", "samples": samples,
        "medians": {name: statistics.median(times)
                    for name, times in samples.items()},
        "correctness": "all returns and counts exactly equal",
    }


def main():
    print(json.dumps(benchmark()), flush=True)


if __name__ == "__main__":
    main()
