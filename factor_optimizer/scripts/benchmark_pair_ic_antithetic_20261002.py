"""Bounded exact-backend A/B for paired TRAIN RankIC sign reuse.

Run from the server-c repository root with:
    .venv/bin/python factor_optimizer/scripts/benchmark_pair_ic_antithetic_20261002.py

The benchmark exercises `_pair_ic` with its normal RAW/RAW_FULL reference and
candidate caches. It is a synthetic microbenchmark, not a full optimizer run.
"""
from __future__ import annotations

import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
import quant_evaluator.metrics.ic as ic_module
from factor_optimizer.research_batch import BatchOptimizationConfig, PairICCache, _pair_ic


ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "factor_optimizer/factor_optimizer/research_batch.py",
    "factor_optimizer/factor_optimizer/research_fitness.py",
    "quant_evaluator/metrics/ic.py",
    "factor_optimizer/factor_optimizer/research_ic_antithetic.py",
)
T, N, SEED = 128, 2000, 20261002
REPEATS = 3


def source_state():
    return {
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCES
        },
    }


def fixture():
    rng = np.random.default_rng(SEED)
    times = np.arange(T)
    assets = np.array([f"a{i}" for i in range(N)])
    raw = rng.normal(size=(T, N))
    raw[::17, 2] = np.nan
    raw[:, ::11] = np.round(raw[:, ::11], 1)
    candidate = raw + .3 * rng.normal(size=(T, N))
    candidate[::23, 1] = np.nan
    candidate[:, ::13] = np.round(candidate[:, ::13], 1)
    labels_values = rng.normal(size=(T, N))
    labels_values[::19, 3] = np.nan
    label_validity = rng.random((T, N)) > .025
    time_axis = AxisRef("time", "int", T, times)
    asset_axis = AxisRef("asset", "str", N, assets)
    batch = FactorBatch(("raw",), time_axis, asset_axis, raw[:, :, None])
    labels = LabelBundle(
        "pair-ic-antithetic-ab", labels_values, 1,
        decision_time=tuple(times), label_start_time=tuple(times + 1),
        label_end_time=tuple(times + 2), validity=label_validity,
        asset_axis=asset_axis,
    )
    return raw, candidate, batch, labels


def run_pair(raw, candidate, batch, labels, *, reuse_opposite):
    indices = tuple(range(T))
    config = BatchOptimizationConfig(minimum_assets=20)
    reference_cache, candidate_cache = PairICCache(), PairICCache()
    positive = _pair_ic(
        raw, candidate, batch, labels, indices, config,
        reference_cache=reference_cache, candidate_cache=candidate_cache,
        cache_opposite_candidate=reuse_opposite,
    )
    negative = _pair_ic(
        raw, -candidate, batch, labels, indices, config,
        reference_cache=reference_cache, candidate_cache=candidate_cache,
    )
    return positive, negative


def exact_equal(left, right):
    for left_series, right_series in zip(left, right):
        for a, b in zip(left_series, right_series):
            if isinstance(a, np.ndarray):
                if a.dtype != b.dtype or a.shape != b.shape:
                    return False
                if a.dtype.kind == "f":
                    if not np.array_equal(a.view(np.uint8), b.view(np.uint8)):
                        return False
                elif not np.array_equal(a, b):
                    return False
            elif a != b:
                return False
    return True


def call_counts(raw, candidate, batch, labels):
    original = ic_module.compute_daily_ic
    observed = []

    def count(factors, *args, **kwargs):
        observed.append(factors.num_factors)
        return original(factors, *args, **kwargs)

    ic_module.compute_daily_ic = count
    try:
        run_pair(raw, candidate, batch, labels, reuse_opposite=False)
        baseline = len(observed)
        observed.clear()
        run_pair(raw, candidate, batch, labels, reuse_opposite=True)
        reused = len(observed)
    finally:
        ic_module.compute_daily_ic = original
    return {"baseline": baseline, "reuse": reused}


def timed_case(raw, candidate, batch, labels, *, reuse_opposite):
    run_pair(raw, candidate, batch, labels, reuse_opposite=reuse_opposite)
    samples = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        run_pair(raw, candidate, batch, labels, reuse_opposite=reuse_opposite)
        samples.append(time.perf_counter() - started)
    return {"samples_seconds": samples, "median_seconds": statistics.median(samples)}


def main():
    before = source_state()
    raw, candidate, batch, labels = fixture()
    baseline = run_pair(raw, candidate, batch, labels, reuse_opposite=False)
    reused = run_pair(raw, candidate, batch, labels, reuse_opposite=True)
    parity = exact_equal(baseline, reused)
    if not parity:
        raise AssertionError("paired outputs differ at the byte level")
    calls = call_counts(raw, candidate, batch, labels)

    # Warmups and measurement order are fixed to match the 2026-10-02 receipt.
    baseline_timing = timed_case(raw, candidate, batch, labels, reuse_opposite=False)
    reuse_timing = timed_case(raw, candidate, batch, labels, reuse_opposite=True)
    after = source_state()
    if before != after:
        raise RuntimeError("source hashes or HEAD changed during the benchmark")
    output = {
        "head": before["head"],
        "source_sha256": before["sha256"],
        "source_unchanged_during_run": True,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "fixture": {"time_rows": T, "assets": N, "dtype": "float64", "seed": SEED,
                    "candidates": 2, "mask": "candidate-specific with label validity"},
        "warmup": "one baseline run, then one reuse run; excluded from samples",
        "measurement_order": "baseline three runs, then reuse three runs",
        "repeats": REPEATS,
        "bitwise_pair_output_parity": parity,
        "compute_daily_ic_calls": calls,
        "baseline_two_full_pair_calls": baseline_timing,
        "positive_plus_antithetic_cache": reuse_timing,
        "speedup_fraction_from_medians": 1.0 - reuse_timing["median_seconds"] / baseline_timing["median_seconds"],
        "estimated_memory_mib": {
            "one_float64_panel": T * N * 8 / 1024**2,
            "three_column_pair_panel": T * N * 3 * 8 / 1024**2,
            "three_column_validity_mask": T * N * 3 / 1024**2,
            "extra_antithetic_peak_estimate": 3 * T * N * 8 / 1024**2,
            "peak_estimate_scope": "helper and key serializer arrays; excludes QE rank scratch and process RSS",
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
