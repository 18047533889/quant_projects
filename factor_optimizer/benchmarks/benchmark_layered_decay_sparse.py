"""Reproduce the layered-decay sparse-vs-HEAD end-to-end benchmark.

Run from the repository root with .venv/bin/python. The historical recurrence is
loaded directly from git into memory; this script does not create a repository copy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from factor_optimizer.adapters.layered_decay import LayeredDecayPlan, _assign_daily_quantiles
from factor_optimizer.adapters.repair_execution import _validate_frame


ROOT = Path(__file__).resolve().parents[2]
HEAD_RECURRENCE = "factor_preprocess/factor_preprocess/transforms/layered_decay.py"
CANDIDATE_PATHS = (
    "factor_optimizer/factor_optimizer/adapters/layered_decay.py",
    "factor_optimizer/factor_optimizer/adapters/repair_execution.py",
    "factor_optimizer/factor_optimizer/adapters/layered_decay_long.py",
    "factor_preprocess/factor_preprocess/transforms/layered_decay.py",
    "factor_preprocess/factor_preprocess/transforms/layered_decay_state.py",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_head_recurrence(baseline_ref: str):
    source = subprocess.check_output(
        ["git", "show", f"{baseline_ref}:{HEAD_RECURRENCE}"], cwd=ROOT, text=True
    )
    namespace = {"np": np, "math": math, "numbers": numbers}
    exec(compile(source, f"<HEAD:{HEAD_RECURRENCE}>", "exec"), namespace)
    return namespace["layered_decay"], _sha256(source.encode())


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-ref", default="120f51720da822bf7f2a6fa47d435af6440e2027")
    parser.add_argument("--write-json", type=Path)
    args = parser.parse_args()
    baseline_commit = subprocess.check_output(
        ["git", "rev-parse", args.baseline_ref], cwd=ROOT, text=True
    ).strip()
    frozen_ref = "120f51720da822bf7f2a6fa47d435af6440e2027"
    if baseline_commit != frozen_ref:
        raise ValueError(f"unexpected historical baseline {baseline_commit}; expected {frozen_ref}")
    old_decay, head_source_sha = _load_head_recurrence(baseline_commit)
    candidate_source_hashes_before = {
        path: _sha256((ROOT / path).read_bytes()) for path in CANDIDATE_PATHS
    }
    script_relpath = str(Path(__file__).resolve().relative_to(ROOT))
    script_sha = _sha256(Path(__file__).read_bytes())
    rng = np.random.default_rng(8087)
    n_dates, n_assets = 480, 3000
    dates = np.repeat(np.arange(n_dates), n_assets)
    assets = np.tile(np.arange(n_assets), n_dates)
    values = rng.normal(size=n_dates * n_assets)
    half_lives = tuple(np.linspace(1.0, 60.0, 20))
    plan = LayeredDecayPlan(half_lives, "benchmark:layered-decay-head-ab")
    cases = []

    for density in (1.0, 0.55, 0.25, 0.10):
        if density == 1.0:
            keep = np.ones(n_dates * n_assets, dtype=bool)
        else:
            keep = ((dates * 7 + assets * 11) % 100) < int(density * 100)
        frame = pd.DataFrame({
            "date": dates[keep], "asset_id": assets[keep], "value": values[keep]
        })

        def historical_head():
            valid = _validate_frame(frame)
            if not all(g["date"].is_monotonic_increasing
                       for _, g in valid.groupby("asset_id")):
                raise ValueError("per-asset dates must be monotone increasing")
            panel = valid.pivot(
                index="date", columns="asset_id", values="value"
            ).sort_index()
            x = panel.to_numpy(dtype=float)
            x = np.where(np.isfinite(x), x, np.nan)
            bins = _assign_daily_quantiles(x)
            y = old_decay(x, bins, half_lives, allow_research=True)
            return y[
                panel.index.get_indexer(valid["date"]),
                panel.columns.get_indexer(valid["asset_id"]),
            ]

        def candidate():
            return plan.execute(frame, allow_research=True).to_numpy()

        expected = historical_head()
        actual = candidate()
        if not np.array_equal(expected, actual, equal_nan=True):
            raise AssertionError(f"candidate diverged at density={density}")
        output_hash = _sha256(np.nan_to_num(expected, nan=1234567.0).tobytes())
        finite_mask_hash = _sha256(np.isfinite(expected).tobytes())
        finite_values_hash = _sha256(expected[np.isfinite(expected)].astype("<f8", copy=False).tobytes())
        raw = {"historical_head_seconds": [], "candidate_seconds": []}
        for repetition in range(3):
            order = ("historical_head", "candidate") if repetition % 2 == 0 else (
                "candidate", "historical_head"
            )
            for name in order:
                start = time.perf_counter()
                result = historical_head() if name == "historical_head" else candidate()
                raw[f"{name}_seconds"].append(time.perf_counter() - start)
                if not np.array_equal(expected, result, equal_nan=True):
                    raise AssertionError(f"{name} drift at density={density}")
        head_median = statistics.median(raw["historical_head_seconds"])
        candidate_median = statistics.median(raw["candidate_seconds"])
        cases.append({
            "density": density,
            "rows": len(frame),
            "raw_seconds": raw,
            "median_speedup_candidate_vs_head": head_median / candidate_median,
            "output_sha256": output_hash,
            "finite_mask_sha256": finite_mask_hash,
            "finite_values_sha256": finite_values_hash,
        })

    candidate_source_hashes_after = {
        path: _sha256((ROOT / path).read_bytes()) for path in CANDIDATE_PATHS
    }
    if candidate_source_hashes_after != candidate_source_hashes_before:
        raise RuntimeError("candidate source changed while the benchmark was running")
    source_hashes = dict(candidate_source_hashes_before)
    source_hashes[script_relpath] = script_sha
    report = {
        "benchmark": "layered-decay-sparse-stream-vs-historical-head",
        "candidate_checkout_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "baseline_commit": baseline_commit,
        "historical_recurrence": {
            "path": HEAD_RECURRENCE,
            "source_sha256": head_source_sha,
            "loaded_in_memory_from_git_show": True,
        },
        "candidate_source_sha256": source_hashes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "method": {
            "shape": [n_dates, n_assets],
            "seed": 8087,
            "half_lives": 20,
            "warmups_per_case": 1,
            "timed_repetitions": 3,
            "alternation": "HEAD,candidate; candidate,HEAD; HEAD,candidate",
            "parity": "exact including NaN positions on warmup and every timed result",
            "head_boundary": "frame validation + per-asset monotonic check + pivot + bounded QE quantiles + frozen historical recurrence + output mapping",
            "validation_scope": "matched for valid benchmark fixtures; duplicate-column rejection guard is not exercised because fixture columns are unique",
            "shared_live_helpers": "Both paths use the current _validate_frame and _assign_daily_quantiles implementations; their owner files are hashed in candidate_source_sha256.",
            "candidate_boundary": "LayeredDecayPlan.execute including frame validation + sparse quantile blocks + shared state + output mapping",
            "mask": "keep if (date*7 + asset_id*11) % 100 < int(density*100); density=1 uses all rows",
        },
        "cases": cases,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.write_json:
        args.write_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    _main()
