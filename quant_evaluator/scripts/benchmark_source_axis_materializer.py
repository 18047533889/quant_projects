"""Matched in-memory ABBA benchmark for bounded QE axis materialization.

No COS/DataAccess reads are performed. The reference is the pre-fast-path
row-chunk algorithm: fancy date index selection followed by fancy asset
selection. RSS deltas are process high-water marks relative to the prepared
baseline and therefore report transient peak growth, not total RSS.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import statistics
import time

import numpy as np
import pandas as pd

from quant_evaluator.adapters.source_axis_materializer import (
    AXIS_REINDEX_CHUNK_BYTES,
    write_axis_aligned_float64,
)


def _reference(frame, dates, assets, output, chunk_bytes):
    date_positions = frame.index.get_indexer(dates)
    asset_positions = frame.columns.get_indexer(assets)
    bytes_per_row = (len(frame.columns) + len(asset_positions)) * 8
    rows_per_chunk = chunk_bytes // bytes_per_row
    for row_start in range(0, len(date_positions), rows_per_chunk):
        row_stop = min(len(date_positions), row_start + rows_per_chunk)
        row_frame = frame.iloc[date_positions[row_start:row_stop]]
        chunk = row_frame.iloc[:, asset_positions].to_numpy(
            dtype=np.float64, copy=False)
        output[row_start:row_stop, :] = chunk
        del chunk, row_frame


def _peak_rss_kib():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _worker(method: str, width: int, seed: int):
    rows, source_rows, cols = 2586, 2588, 5461
    dates = pd.date_range("2016-01-04", periods=source_rows, freq="B")
    assets = tuple(f"{i:06d}.SZ" for i in range(cols))
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((source_rows, cols), dtype=np.float64)
    values.reshape(-1)[::100] = np.nan
    frame = pd.DataFrame(values, index=dates, columns=assets)
    target_dates = dates[:rows]
    backing = np.empty((rows, cols, width), dtype=np.float64)
    backing.fill(0.0)
    outputs = [backing[:, :, i] for i in range(width)]
    peak_before = _peak_rss_kib()
    started = time.perf_counter()
    for output in outputs:
        if method == "A":
            _reference(frame, target_dates, assets, output,
                       AXIS_REINDEX_CHUNK_BYTES)
        else:
            write_axis_aligned_float64(
                frame, target_dates, assets, output, memory_bounded=True,
                chunk_bytes=AXIS_REINDEX_CHUNK_BYTES)
    elapsed = time.perf_counter() - started
    peak_after = _peak_rss_kib()
    expected = frame.reindex(index=target_dates, columns=assets).to_numpy(
        dtype=np.float64, copy=False)
    return {
        "method": method,
        "seconds": elapsed,
        "peak_rss_delta_kib": max(0, peak_after - peak_before),
        "all_values_and_nan_masks_match": all(
            np.array_equal(output, expected, equal_nan=True)
            for output in outputs),
        "output_sha256": [
            hashlib.sha256(
                np.ascontiguousarray(output).view(np.uint8)).hexdigest()
            for output in outputs
        ],
        "output_slice_strides_bytes": list(outputs[0].strides),
    }


def run(repeats: int = 2, seed: int = 20261003):
    if type(repeats) is not int or not 1 <= repeats <= 8:
        raise ValueError("repeats must be an integer in 1..8")
    import subprocess
    import sys
    rows, source_rows, cols = 2586, 2588, 5461
    order = tuple("ABBA" * repeats)
    results = {}
    module = "quant_evaluator.scripts.benchmark_source_axis_materializer"
    for width in (2, 4):
        measurements = []
        for method in order:
            completed = subprocess.run(
                [sys.executable, "-m", module, "--worker", method, "--width", str(width),
                 "--seed", str(seed)],
                check=True, capture_output=True, text=True, timeout=120)
            measurements.append(json.loads(completed.stdout))
        methods = {
            method: [item["seconds"] for item in measurements
                     if item["method"] == method]
            for method in ("A", "B")
        }
        results[str(width)] = {
            "shape": [rows, cols, width],
            "order": list(order),
            "seconds": methods,
            "median_seconds": {
                method: statistics.median(samples)
                for method, samples in methods.items()},
            "peak_rss_delta_kib": {
                method: [item["peak_rss_delta_kib"] for item in measurements
                         if item["method"] == method]
                for method in ("A", "B")},
            "all_values_and_nan_masks_match": all(
                item["all_values_and_nan_masks_match"]
                for item in measurements),
            "output_sha256": {
                method: [item["output_sha256"] for item in measurements
                         if item["method"] == method]
                for method in ("A", "B")},
            "output_slice_strides_bytes": measurements[0][
                "output_slice_strides_bytes"],
        }
    module_path = __import__("pathlib").Path(
        "quant_evaluator/adapters/source_axis_materializer.py")
    harness_path = __import__("pathlib").Path(__file__)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "status": "complete",
        "kind": "source_axis_materializer_bounded_abba.v1",
        "scope": "synthetic in-memory materialization only; no COS, Arrow, FactorBatch, or GPU work",
        "seed": seed,
        "repeats_per_method_width": repeats * 2,
        "source_shape": [source_rows, cols],
        "target_shape": [rows, cols],
        "factor_widths": [2, 4],
        "factor_output_slice_strides": "T x N x width backing; each output is a noncontiguous [:,:,i] slice",
        "chunk_bytes": AXIS_REINDEX_CHUNK_BYTES,
        "reference": "original bounded fancy date and fancy column iloc chunk algorithm",
        "candidate": "bounded helper with contiguous slice selection where positions form unit-stride runs",
        "peak_rss_note": "each measurement is a fresh process; inputs and outputs are allocated and touched before baseline; ru_maxrss is a high-water mark that earlier pandas/NumPy allocations may dominate, so this is only a limited incremental-peak indicator, not total process RSS",
        "comparison_includes_axis_indexers": True,
        "source_sha256": {
            str(module_path): sha(module_path),
            str(harness_path): sha(harness_path),
        },
        "runs": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=4,
                        help="ABBA repetitions per method and width")
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--output")
    parser.add_argument("--worker", choices=("A", "B"))
    parser.add_argument("--width", type=int, choices=(2, 4))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(_worker(args.worker, args.width, args.seed)))
        return
    receipt = run(args.repeats, args.seed)
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        from pathlib import Path
        with Path(args.output).open("x", encoding="utf-8") as stream:
            stream.write(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
