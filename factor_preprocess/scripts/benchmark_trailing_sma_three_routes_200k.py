#!/usr/bin/env python3
"""Bounded three-route trailing-SMA parity/performance check on 200k long rows.

This is a research benchmark, not a production qualification.  It separates the
first FE composite call (which can include cold FactorEngine/Polars setup) from
all warm, order-alternated measurements.  FP and FE cover min_periods 0, 1 and
window; the FO value-repair contract intentionally covers complete-window SMA
only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
import pandas as pd

from factor_preprocess.registry.transforms import get_default_registry
from factor_preprocess.transforms.smoothing import trailing_sma
from factor_optimizer.adapters.fe_smoothing import execute_lagged_sma


ROW_COUNT = 200_000
WINDOWS = (3, 10, 30)
ROUND_COUNT = 5


def make_panel() -> pd.DataFrame:
    """Create a deterministic, sparse-date 200k panel with awkward identities."""
    asset_count = 1_000
    rows_per_asset = ROW_COUNT // asset_count
    asset_codes = np.repeat(np.arange(asset_count, dtype=np.int32), rows_per_asset)
    offsets = np.tile(np.arange(rows_per_asset, dtype=np.int64) * 3, asset_count)
    # Offset assets by parity while retaining monotone dates within each asset.
    dates = offsets + np.repeat(np.arange(asset_count, dtype=np.int64) % 2,
                                rows_per_asset)
    categories = [f"asset-{i:04d}" for i in range(asset_count)] + ["unused"]
    asset_values = np.asarray(categories[:-1], dtype=object)[asset_codes]
    asset_values[::997] = None
    asset_ids = pd.Categorical(asset_values, categories=categories)

    rng = np.random.default_rng(20260930)
    values = rng.normal(size=ROW_COUNT).astype(np.float64)
    values[::101] = np.nan
    values[::997] = np.inf
    index = pd.Index((np.arange(ROW_COUNT, dtype=np.int64) * 7919) % 150_003,
                     name="duplicate_row_label")
    return pd.DataFrame(
        {"asset_id": asset_ids, "date": dates, "value": values},
        index=index,
    )


def _assert_same(reference: pd.Series, actual: pd.Series, label: str) -> float:
    if not reference.index.equals(actual.index):
        raise AssertionError(f"{label} changed the duplicate input index")
    left = reference.to_numpy(dtype=np.float64, copy=False)
    right = actual.to_numpy(dtype=np.float64, copy=False)
    np.testing.assert_allclose(left, right, rtol=0.0, atol=1e-12, equal_nan=True)
    finite = np.isfinite(left) & np.isfinite(right)
    return float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0



def run(rounds: int = ROUND_COUNT) -> dict:
    if rounds < 2:
        raise ValueError("at least two warm rounds are required for order alternation")
    frame = make_panel()

    # Keep cold FE initialization/first evaluation out of every warm statistic.
    cold_start = time.perf_counter()
    registry = get_default_registry()
    fe_execute = registry.get_execution("trailing_sma")
    cold_result = fe_execute(frame, window=3, min_periods=3)
    cold_fe_first_call_s = time.perf_counter() - cold_start
    cold_fp = trailing_sma(frame, window=3, min_periods=3)
    cold_max_abs_diff = _assert_same(cold_fp, cold_result, "FE cold first call")

    results = []
    for window in WINDOWS:
        for min_periods in (0, 1, window):
            functions = {
                "FP_native": lambda w=window, m=min_periods: trailing_sma(
                    frame, window=w, min_periods=m),
                "FE_registry": lambda w=window, m=min_periods: fe_execute(
                    frame, window=w, min_periods=m),
            }
            if min_periods == window:
                functions["FO_adapter"] = lambda w=window: execute_lagged_sma(
                    frame, window=w)

            # Warm each path before parity assertions and timed repetitions.
            first_outputs = {name: fn() for name, fn in functions.items()}
            reference = first_outputs["FP_native"]
            max_diffs = {
                name: _assert_same(reference, output, name)
                for name, output in first_outputs.items()
            }

            timings = {name: [] for name in functions}
            names = list(functions)
            for repeat in range(rounds):
                order = names[repeat % len(names):] + names[:repeat % len(names)]
                if repeat % 2:
                    order.reverse()
                for name in order:
                    started = time.perf_counter()
                    output = functions[name]()
                    timings[name].append(time.perf_counter() - started)
                    max_diffs[name] = max(
                        max_diffs[name], _assert_same(reference, output, name)
                    )

            results.append({
                "window": window,
                "min_periods": min_periods,
                "causal_semantics": "per-asset prior observed rows [t-window, t-1]; current excluded; delay=1",
                "routes": {
                    name: {
                        "warm_median_seconds": statistics.median(samples),
                        "warm_samples_seconds": samples,
                        "max_abs_diff_vs_fp": max_diffs[name],
                    }
                    for name, samples in timings.items()
                },
            })

    return {
        "status": "passed",
        "workload": {
            "rows": len(frame),
            "assets": 1000,
            "observations_per_asset_before_null_asset_rows": 200,
            "date_spacing": "3 integer ticks per per-asset observation (sparse calendar)",
            "asset_dtype": "category with unused category and null asset rows",
            "duplicate_index": True,
            "null_values": True,
            "infinite_values": True,
            "seed": 20260930,
        },
        "timing_method": {
            "cold_fe_first_call_seconds": cold_fe_first_call_s,
            "cold_fe_max_abs_diff_vs_fp": cold_max_abs_diff,
            "warm_rounds": rounds,
            "warm_order": "rotating route order, reversed on alternating rounds",
            "cold_fe_time_excluded_from_warm_statistics": True,
        },
        "fo_contract": (
            "FO FE-smoothing adapter keeps complete-window SMA (min_periods=window); "
            "min_periods=0/1 are compared only between FP-native and FE registry."
        ),
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=ROUND_COUNT)
    parser.add_argument("--output", help="optional JSON output path; stdout is always emitted")
    args = parser.parse_args()
    report = run(args.rounds)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
