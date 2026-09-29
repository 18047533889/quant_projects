#!/usr/bin/env python3
"""Bounded RSS sampling for lagged vectorized transforms.

Each operation/shape runs in a fresh subprocess with a 3 GiB address-space
limit, 30 CPU-second limit, and 35-second wall timeout. Maximum is 512,000 rows.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OPS = ("rolling_mean", "rolling_std", "rolling_zscore", "ewma",
       "trailing_median", "robust_ewma")
SHAPES = ((256, 500), (1024, 500))


def child(op, n_assets, n_dates):
    import resource
    limit = 3 * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 31))
    import numpy as np
    import pandas as pd
    from factor_preprocess.transforms import rolling as R
    from factor_preprocess.transforms import smoothing as S
    funcs = {
        "rolling_mean": (R.rolling_mean, {"window": 20, "min_periods": 10}),
        "rolling_std": (R.rolling_std, {"window": 20, "min_periods": 10}),
        "rolling_zscore": (R.rolling_zscore, {"window": 20, "min_periods": 10}),
        "ewma": (R.ewma, {"halflife": 10}),
        "trailing_median": (S.trailing_median, {"window": 20, "min_periods": 10}),
        "robust_ewma": (S.robust_ewma, {"halflife": 10}),
    }
    if op not in funcs:
        raise ValueError(f"unknown operation: {op}")
    count = n_assets * n_dates
    rng = np.random.default_rng(20260929 + n_assets)
    values = rng.standard_normal(count)
    values[rng.random(count) < 0.03] = np.nan
    frame = pd.DataFrame({
        "asset_id": np.repeat(np.arange(n_assets, dtype=np.int32), n_dates),
        "date": np.tile(pd.date_range("2010-01-01", periods=n_dates, freq="W"), n_assets),
        "value": values,
    })
    fn, kwargs = funcs[op]

    def rss():
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        raise RuntimeError("VmRSS unavailable")

    baseline = rss()
    peak = [baseline]
    stopped = threading.Event()
    def sample():
        while not stopped.is_set():
            peak[0] = max(peak[0], rss())
            stopped.wait(0.005)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        result = fn(frame, **kwargs)
    finally:
        elapsed = time.perf_counter() - started
        stopped.set()
        monitor.join(timeout=1)
    peak[0] = max(peak[0], rss())
    print(json.dumps({
        "operation": op, "assets": n_assets, "dates": n_dates, "rows": count,
        "baseline_rss_mb": round(baseline / 1024**2, 1),
        "peak_rss_mb": round(peak[0] / 1024**2, 1),
        "incremental_peak_rss_mb": round(max(0, peak[0] - baseline) / 1024**2, 1),
        "elapsed_seconds": round(elapsed, 3), "output_rows": len(result),
    }, sort_keys=True))


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "--child":
        child(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
        return
    env = os.environ.copy()
    env.update({k: "1" for k in (
        "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS")})
    deadline = time.monotonic() + 240
    print(json.dumps({
        "scope": "RSS sampled in fresh subprocesses",
        "memory_limit_gib": 3, "cpu_limit_seconds_per_child": 30,
        "wall_timeout_seconds_per_child": 35, "max_rows": 512000,
        "sampling_interval_ms": 5,
    }, sort_keys=True), flush=True)
    for assets, dates in SHAPES:
        for op in OPS:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("overall 240-second budget exceeded")
            cmd = [sys.executable, __file__, "--child", op, str(assets), str(dates)]
            result = subprocess.run(cmd, check=True, capture_output=True,
                                    text=True, timeout=min(35, remaining), env=env)
            print(result.stdout.strip(), flush=True)


if __name__ == "__main__":
    main()
