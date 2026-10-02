#!/usr/bin/env python3
"""Reproducible bounded A/B for rolling-Sharpe precision behavior.

Uses a pinned HEAD implementation extracted from Git in memory and compares it
with the working-tree candidate. Oracle work is deliberately outside timings.
"""
from __future__ import annotations

import ast
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quant_evaluator.metrics.portfolio_stats import (
    _rolling_sharpe_per_window as candidate,
)
BASELINE_COMMIT = "278233d458d566f7b4d771916ea2e1eb2823e906"
SEED = 301
SIZE = 3000
PERIODS_PER_YEAR = 252
REPEATS = 10
WINDOWS = (20, 252)
GAP_STRIDE = 11


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def candidate_source_hashes() -> dict[str, str]:
    paths = (
        "quant_evaluator/metrics/portfolio_stats.py",
        "quant_evaluator/metrics/rolling_sharpe_moments.py",
        "quant_evaluator/metrics/rolling_window_constancy.py",
    )
    return {
        path: sha256_bytes((ROOT / path).read_bytes())
        for path in paths
    }


def canonical_array_hash(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f8").copy()
    canonical[np.isnan(canonical)] = np.nan
    return sha256_bytes(canonical.tobytes(order="C"))


def finite_mask_hash(values: np.ndarray) -> str:
    return sha256_bytes(np.isfinite(values).astype(np.uint8).tobytes())


def load_baseline():
    source = subprocess.check_output(
        [
            "git", "-C", str(ROOT), "show",
            f"{BASELINE_COMMIT}:quant_evaluator/metrics/portfolio_stats.py",
        ],
        text=True,
    )
    tree = ast.parse(source)
    function = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_rolling_sharpe_per_window"
    )
    namespace = {"np": np}
    module = ast.Module(body=[function], type_ignores=[])
    exec(
        compile(module, f"{BASELINE_COMMIT}/portfolio_stats.py", "exec"),
        namespace,
    )
    function_source = ast.get_source_segment(source, function)
    if function_source is None:
        raise RuntimeError("could not extract pinned baseline function source")
    return namespace[function.name], function_source


def make_cases() -> dict[str, tuple[np.ndarray, bool]]:
    rng = np.random.default_rng(SEED)
    ordinary = 0.01 + rng.normal(0.0, 0.02, SIZE)
    constant = np.full(SIZE, 0.01, dtype=np.float64)
    missing_constant = constant.copy()
    missing_constant[::GAP_STRIDE] = np.nan
    near_flat = 0.01 + rng.normal(0.0, 1e-5, SIZE)
    noisy_near_flat_gaps = near_flat.copy()
    noisy_near_flat_gaps[::GAP_STRIDE] = np.nan
    return {
        "ordinary": (ordinary, False),
        "constant": (constant, False),
        "missing_constant": (missing_constant, True),
        "near_flat": (near_flat, False),
        "noisy_near_flat_with_gaps": (noisy_near_flat_gaps, True),
    }


def direct_oracle(
    values: np.ndarray, window: int, min_periods: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Compute direct finite-slice results, one window at a time."""
    expected = np.full(values.size, np.nan, dtype=np.float64)
    eligible = 0
    eligible_with_missing = 0
    finite_outputs = 0
    for start in range(values.size - window + 1):
        row = values[start:start + window]
        row = row[np.isfinite(row)]
        if row.size < min_periods:
            continue
        eligible += 1
        if row.size < window:
            eligible_with_missing += 1
        std = np.std(row, ddof=1)
        if np.isfinite(std) and std > 1e-10:
            expected[start + window - 1] = (
                np.mean(row) / std * np.sqrt(PERIODS_PER_YEAR)
            )
            finite_outputs += 1
    return expected, {
        "eligible_windows": eligible,
        "eligible_windows_with_missing_values": eligible_with_missing,
        "oracle_finite_outputs": finite_outputs,
    }


def error_summary(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    actual_finite = np.isfinite(actual)
    expected_finite = np.isfinite(expected)
    common = actual_finite & expected_finite
    if np.any(common):
        absolute = np.abs(actual[common] - expected[common])
        denominators = np.maximum(np.abs(expected[common]), np.finfo(np.float64).tiny)
        relative = absolute / denominators
        max_abs = float(np.max(absolute))
        max_rel = float(np.max(relative))
        oracle_close = bool(
            np.allclose(actual[common], expected[common], rtol=1e-12, atol=1e-11)
        )
    else:
        max_abs = None
        max_rel = None
        oracle_close = True
    return {
        "finite_mask_matches_oracle": bool(np.array_equal(actual_finite, expected_finite)),
        "finite_mask_mismatch_count": int(np.count_nonzero(actual_finite != expected_finite)),
        "max_absolute_error_common_finite": max_abs,
        "max_relative_error_common_finite": max_rel,
        "allclose_common_finite_rtol_1e-12_atol_1e-11": oracle_close,
    }


def result_fingerprints(values: np.ndarray) -> dict[str, object]:
    return {
        "finite_outputs": int(np.isfinite(values).sum()),
        "finite_mask_sha256": finite_mask_hash(values),
        "output_sha256_canonical_little_endian_f64": canonical_array_hash(values),
    }


def benchmark() -> dict[str, object]:
    baseline, baseline_function_source = load_baseline()
    candidate_hashes_before = candidate_source_hashes()
    cases = make_cases()
    case_results = []
    for case_name, (values, has_gaps) in cases.items():
        for window in WINDOWS:
            min_periods = (
                max(2, int(np.floor(0.9 * window)))
                if has_gaps
                else window
            )
            expected, counts = direct_oracle(values, window, min_periods)
            baseline_result = baseline(
                values, window, PERIODS_PER_YEAR, min_periods,
            )
            candidate_result = candidate(
                values, window, PERIODS_PER_YEAR, min_periods,
            )
            timings: dict[str, list[float]] = {
                "baseline": [],
                "candidate": [],
            }
            for repeat in range(REPEATS):
                order = (
                    (("baseline", baseline), ("candidate", candidate))
                    if repeat % 2 == 0
                    else (("candidate", candidate), ("baseline", baseline))
                )
                for label, function in order:
                    started = time.perf_counter()
                    function(values, window, PERIODS_PER_YEAR, min_periods)
                    timings[label].append(time.perf_counter() - started)

            case_results.append({
                "case": case_name,
                "window": window,
                "min_periods": min_periods,
                "input_size": int(values.size),
                "input_finite_count": int(np.isfinite(values).sum()),
                **counts,
                "baseline": result_fingerprints(baseline_result),
                "candidate": result_fingerprints(candidate_result),
                "oracle": result_fingerprints(expected),
                "baseline_vs_oracle": error_summary(baseline_result, expected),
                "candidate_vs_oracle": error_summary(candidate_result, expected),
                "timing_seconds_raw_alternating_10_reps": timings,
                "timing_seconds_median": {
                    label: float(np.median(samples))
                    for label, samples in timings.items()
                },
            })

    candidate_hashes_after = candidate_source_hashes()
    return {
        "schema_version": 1,
        "runner_source_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "repository_head": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Synthetic one-dimensional return-like series; not a historical "
            "market-return sample or a full factor batch/portfolio evaluation."
        ),
        "baseline_commit": BASELINE_COMMIT,
        "baseline_function_source_sha256": sha256_bytes(
            baseline_function_source.encode("utf-8")
        ),
        "candidate_source_sha256_before": candidate_hashes_before,
        "candidate_source_sha256_after": candidate_hashes_after,
        "candidate_sources_unchanged_during_run": (
            candidate_hashes_before == candidate_hashes_after
        ),
        "parameters": {
            "seed": SEED,
            "size": SIZE,
            "periods_per_year": PERIODS_PER_YEAR,
            "windows": list(WINDOWS),
            "repeats": REPEATS,
            "gap_stride": GAP_STRIDE,
            "oracle": "per-window direct finite-slice mean and sample std",
            "accuracy_tolerance": {"rtol": 1e-12, "atol": 1e-11},
            "old_baseline_accuracy_is_reported_not_asserted": True,
        },
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "longdouble": {
                "eps": float(np.finfo(np.longdouble).eps),
                "nmant": int(np.finfo(np.longdouble).nmant),
                "machep": int(np.finfo(np.longdouble).machep),
                "precision": int(np.finfo(np.longdouble).precision),
                "itemsize": int(np.dtype(np.longdouble).itemsize),
            },
        },
        "results": case_results,
    }


if __name__ == "__main__":
    report = benchmark()
    output_path = ROOT / "quant_evaluator/docs/benchmarks/rolling_sharpe_precision_ab_20261002.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output_path)
