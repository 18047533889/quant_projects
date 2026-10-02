#!/usr/bin/env python
"""Reproduce the bounded, alternating robust-EWMA A/B timing on server-c.

Run with the project environment and explicit ``--run`` consent. The default
case recreates the deterministic analytic 600 x 400 fixture in the saved
2026-10-02 receipt; it uses no random source or external data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from factor_preprocess.transforms.smoothing import robust_ewma as old_impl
from factor_engine.backend.long_robust_ewm import lagged_robust_ewma as new_impl

_MAX_ROWS = 300_000
_MIN_HEADROOM_BYTES = 768 * 1024 * 1024
_SOURCE_PATHS = (
    Path("factor_preprocess/factor_preprocess/transforms/smoothing.py"),
    Path("factor_engine/backend/long_robust_ewm.py"),
    Path("factor_engine/backend/native_long_robust_ewma.py"),
)


def _mem_available_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    raise RuntimeError("cannot read /proc/meminfo; refusing an unguarded benchmark")


def _resource_guard(times: int, assets: int) -> int:
    if times <= 0 or assets <= 0:
        raise ValueError("times and assets must be positive")
    rows = times * assets
    if rows > _MAX_ROWS:
        raise ValueError(f"refusing {rows:,} rows; hard limit is {_MAX_ROWS:,}")
    estimated_peak = _MIN_HEADROOM_BYTES + rows * 3 * 512
    available = _mem_available_bytes()
    if available < estimated_peak:
        raise RuntimeError(
            f"refusing benchmark: {available} bytes available; "
            f"estimated requirement is {estimated_peak} bytes"
        )
    return rows


def _source_state() -> dict:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in _SOURCE_PATHS
    }
    return {"head": head, "sha256": hashes}


def _fixture(times: int, assets: int) -> pd.DataFrame:
    asset_codes = np.tile(np.arange(assets, dtype=np.int16), times)
    time_values = np.repeat(np.arange(times, dtype=np.int32), assets)
    asset_phase = asset_codes.astype(np.float64) / 11.0
    values = (
        np.sin(time_values / 15.0)
        + 0.2 * np.cos(asset_phase)
        + (asset_codes % 7) * 0.001
    )
    values[((time_values % 79 == 0) & (asset_codes % 11 == 0))] += 6.0
    values[((time_values + asset_codes) % 53 == 0)] = np.nan
    categories = [f"A{i:04d}" for i in range(assets)]
    return pd.DataFrame({
        "asset_id": pd.Categorical.from_codes(asset_codes, categories=categories),
        "date": time_values,
        "value": values,
    })


def _run(times: int, assets: int, repetitions: int) -> dict:
    rows = _resource_guard(times, assets)
    frame = _fixture(times, assets)
    params = {"halflife": 10.0, "winsor_std": 4.0, "min_periods": 1}
    before = _source_state()

    old_impl(frame, **params)
    new_impl(frame, **params)

    records = []
    expected = None
    for repetition in range(1, repetitions + 1):
        order = ("old", "new") if repetition % 2 else ("new", "old")
        pair = {}
        for label in order:
            started = time.perf_counter()
            result = old_impl(frame, **params) if label == "old" else new_impl(frame, **params)
            seconds = time.perf_counter() - started
            values = result.to_numpy(copy=False)
            if label == "old" and expected is None:
                expected = values.copy()
            if expected is not None:
                if not np.array_equal(np.isnan(values), np.isnan(expected)):
                    raise AssertionError(f"NaN mask mismatch at repetition {repetition} ({label})")
                finite = np.isfinite(values) & np.isfinite(expected)
                max_error = (
                    float(np.max(np.abs(values[finite] - expected[finite])))
                    if finite.any() else 0.0
                )
                if not np.allclose(
                    values, expected, rtol=2e-13, atol=2e-13, equal_nan=True,
                ):
                    raise AssertionError(
                        f"numeric mismatch at repetition {repetition} ({label}): {max_error}"
                    )
                pair[label + "_max_abs_error"] = max_error
            pair[label + "_seconds"] = seconds
        records.append({"repetition": repetition, "order": list(order), **pair})

    after = _source_state()
    old_times = [record["old_seconds"] for record in records]
    new_times = [record["new_seconds"] for record in records]
    receipt = {
        "case": "robust_ewma_alternating_ab_v1",
        "host": platform.node(),
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "polars": pl.__version__,
            "cwd": str(Path.cwd()),
        },
        "fixture": {
            "times": times,
            "assets": assets,
            "rows": rows,
            "generation": "deterministic analytic; no RNG or seed",
            "same_as_saved_measurement": times == 600 and assets == 400,
        },
        "params": params,
        "warmup_order": ["old", "new"],
        "runs": records,
        "old_median_s": float(np.median(old_times)),
        "new_median_s": float(np.median(new_times)),
        "speedup_old_over_new": float(np.median(old_times) / np.median(new_times)),
        "parity": "exact NaN mask; allclose rtol=2e-13, atol=2e-13",
        "source_state_before": before,
        "source_state_after": after,
        "source_state_unchanged": before == after,
    }
    if before != after:
        raise AssertionError("source state changed during timing")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--times", type=int, default=600)
    parser.add_argument("--assets", type=int, default=400)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--run", action="store_true",
        help="execute the bounded benchmark (otherwise only print usage)",
    )
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    if not 1 <= args.repetitions <= 9:
        raise ValueError("repetitions must be between 1 and 9")
    print(json.dumps(
        _run(args.times, args.assets, args.repetitions),
        sort_keys=True, separators=(",", ":"),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
