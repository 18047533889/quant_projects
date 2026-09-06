"""Synthetic, bounded-panel CUDA acceptance; no real data or driver changes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import subprocess
import tempfile
import time

import cupy as cp
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--factors", type=int, default=100_000)
    args = parser.parse_args()
    if not 1 <= args.factors <= 100_000:
        parser.error("factors must be between 1 and 100000")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    T, N, F = 8, 32, args.factors
    rng = np.random.default_rng(62026)
    y = rng.normal(size=(T, N))
    y[:, 2] = np.nan
    times = tuple(pd.date_range("2024-01-01", periods=T))
    lb = LabelBundle("synthetic_next_ret", y, 1, decision_time=times,
                     label_start_time=times,
                     label_end_time=tuple(t + pd.Timedelta(days=1) for t in times))
    total_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="memmap-", dir=output) as scratch:
        values = np.memmap(Path(scratch) / "factors.dat", mode="w+", dtype="float64", shape=(T, N, F))
        for start in range(0, F, 128):
            stop = min(start + 128, F)
            tile = rng.integers(0, 17, size=(T, N, stop - start)).astype(float)
            tile[:, 5, :] = np.nan
            values[:, :, start:stop] = tile
        values.flush()
        fb = FactorBatch(tuple(f"synthetic_{i}" for i in range(F)),
                         AxisRef("time", "datetime", T), AxisRef("asset", "str", N), values)
        small = FactorBatch(fb.factor_ids[:2], fb.time_axis, fb.asset_axis, values[:, :, :2])
        evaluate(small, lb, backend="cuda_strict", metrics=("rank_ic_series",))
        cp.cuda.Device().synchronize()
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        started = time.perf_counter()
        result = evaluate(fb, lb, backend="cuda_strict", metrics=("rank_ic_series",))
        cp.cuda.Device().synchronize()
        gpu_seconds = time.perf_counter() - started
        print(f"CUDA done: factors={F}, seconds={gpu_seconds:.3f}, tiles={result.metadata['factor_tiles_processed']}", flush=True)
        actual = result.series_metrics["rank_ic_series"]
        oracle_start = time.perf_counter()
        max_abs_error = 0.0
        quality_rejected = 0
        for start in range(0, F, 128):
            stop = min(start + 128, F)
            x = values[:, :, start:stop]
            valid = np.isfinite(x) & np.isfinite(y[:, :, None])
            xr = rankdata(np.where(valid, x, np.nan), axis=1, method="average", nan_policy="omit")
            yr = rankdata(np.where(valid, y[:, :, None], np.nan), axis=1, method="average", nan_policy="omit")
            xr -= np.nanmean(xr, axis=1, keepdims=True)
            yr -= np.nanmean(yr, axis=1, keepdims=True)
            expected = np.nansum(xr * yr, axis=1) / np.sqrt(np.nansum(xr * xr, axis=1) * np.nansum(yr * yr, axis=1))
            # Canonical CPU metrics/ic.py rejects fewer than max(20//2,2)
            # distinct levels, even if a raw SciPy correlation is defined.
            sx = np.sort(np.where(valid, x, np.inf), axis=1)
            sy = np.sort(np.where(valid, y[:, :, None], np.inf), axis=1)
            nx = np.isfinite(sx[:, 0, :]).astype(int) + np.sum((sx[:, 1:, :] != sx[:, :-1, :]) & np.isfinite(sx[:, 1:, :]), axis=1)
            ny = np.isfinite(sy[:, 0, :]).astype(int) + np.sum((sy[:, 1:, :] != sy[:, :-1, :]) & np.isfinite(sy[:, 1:, :]), axis=1)
            rejected = (nx < 10) | (ny < 10) | (valid.sum(axis=1) < 20)
            expected[rejected] = np.nan
            quality_rejected += int(rejected.sum())
            np.testing.assert_allclose(actual[:, start:stop], expected, rtol=0, atol=1e-12, equal_nan=True)
            max_abs_error = max(max_abs_error, float(np.nanmax(np.abs(actual[:, start:stop] - expected))))
        oracle_seconds = time.perf_counter() - oracle_start
        assert result.factor_ids == fb.factor_ids
        assert result.metadata["h2d_bytes"] == values.nbytes + y.nbytes
        assert result.metadata["d2h_bytes"] == actual.nbytes
        report = {
            "status": "PASS", "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "synthetic_only": True, "shape_T_N_F": [T, N, F], "dtype": "float64",
            "input_memmap_bytes": values.nbytes, "output_bytes": actual.nbytes,
            "cuda_seconds_after_warmup": gpu_seconds,
            "independent_cpu_oracle_seconds": oracle_seconds,
            "total_seconds_including_input_creation": time.perf_counter() - total_start,
            "all_factor_times_checked": int(actual.size), "max_abs_error": max_abs_error,
            "quality_rejected_factor_times": quality_rejected,
            "oracle": "SciPy average-tie rankdata plus NumPy pairwise-finite Pearson and canonical CPU min20/min10-distinct gates; all columns checked in bounded blocks",
            "max_rss_before_cuda_bytes": rss_before,
            "max_rss_process_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "cupy_version": cp.__version__, "metadata": result.metadata,
            "limitations": ["Small synthetic 8-time x 32-asset panel, not a real-data or production throughput certification",
                            "Only rank_ic_series requested; broader metric mixes and full production/PIT gates NOT_RUN",
                            "RSS is process high-water mark; mapped pages are included",
                            "peak_vram is session pool reserved-byte high-water observation, not whole-device occupancy"],
        }
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        del small, fb, values


if __name__ == "__main__":
    main()
