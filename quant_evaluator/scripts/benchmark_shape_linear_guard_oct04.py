"""Bounded CPU ABBA check for quantile-shape numeric guard overhead.

This compares the in-memory HEAD version of quantile_shape.py with the current
module on two small deterministic profiles. It does not copy or modify source
files and checks bitwise output parity on every timed call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = "quant_evaluator/metrics/quantile_shape.py"
SOURCE_PATHS = (
    MODULE_PATH,
    "quant_evaluator/metrics/shape_linear_numeric.py",
)
METRICS = (
    "compute_quantile_curvature",
    "compute_quantile_adjacent_spread",
    "compute_quantile_extreme_cliff",
    "compute_quantile_tail_asymmetry",
    "compute_top_quantile_cliff",
    "compute_bottom_quantile_cliff",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {name: _sha256((ROOT / name).read_bytes()) for name in SOURCE_PATHS}


def _load_head_module():
    source = subprocess.check_output(
        ["git", "show", f"HEAD:{MODULE_PATH}"], cwd=ROOT,
    )
    module = types.ModuleType("quantile_shape_head_original")
    exec(compile(source.decode("utf-8"), f"HEAD:{MODULE_PATH}", "exec"), module.__dict__)
    return module, source


def _assert_bitwise_equal(actual, expected, context: str) -> None:
    actual_bits = np.ascontiguousarray(actual, dtype=np.float64).view(np.uint64)
    expected_bits = np.ascontiguousarray(expected, dtype=np.float64).view(np.uint64)
    if not np.array_equal(actual_bits, expected_bits):
        raise AssertionError(f"bitwise parity failed: {context}")


def _profiles() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20261004)
    normal = rng.normal(0.02, 0.004, size=(20, 48)).astype(np.float64)
    masked = normal.copy()
    masked[3, 1::4] = np.nan
    masked[11, 2::5] = np.inf
    masked[15, 3::7] = -np.inf
    return {"normal_q20_f48": normal, "masked_q20_f48": masked}


def _profile_hash(values: np.ndarray) -> str:
    metadata = f"{values.shape}:{values.dtype.str}:C".encode("ascii")
    return _sha256(metadata + np.ascontiguousarray(values).tobytes())


def run_benchmark(output: Path) -> dict:
    sys.path.insert(0, str(ROOT))
    from quant_evaluator.metrics import quantile_shape as candidate

    before = _source_hashes()
    baseline, baseline_source = _load_head_module()
    profiles = _profiles()
    results = {}
    for profile_name, values in profiles.items():
        profile_results = {}
        for metric in METRICS:
            baseline_fn = getattr(baseline, metric)
            candidate_fn = getattr(candidate, metric)
            expected = baseline_fn(values)
            _assert_bitwise_equal(candidate_fn(values), expected, f"{profile_name}/{metric}/preflight")
            samples = {"head": [], "candidate": []}
            for round_index in range(3):
                for block_index, mode in enumerate(("head", "candidate", "candidate", "head")):
                    fn = baseline_fn if mode == "head" else candidate_fn
                    durations = []
                    for _ in range(10):
                        start = time.perf_counter_ns()
                        actual = fn(values)
                        elapsed = time.perf_counter_ns() - start
                        _assert_bitwise_equal(
                            actual, expected,
                            f"{profile_name}/{metric}/round{round_index + 1}/block{block_index + 1}",
                        )
                        durations.append(elapsed / 1.0e9)
                    samples[mode].append({
                        "round": round_index + 1,
                        "block": block_index + 1,
                        "calls": 10,
                        "seconds_per_call": float(np.mean(durations)),
                    })
            head_mean = float(np.mean([row["seconds_per_call"] for row in samples["head"]]))
            candidate_mean = float(np.mean([row["seconds_per_call"] for row in samples["candidate"]]))
            profile_results[metric] = {
                "head_samples": samples["head"],
                "candidate_samples": samples["candidate"],
                "head_mean_seconds_per_call": head_mean,
                "candidate_mean_seconds_per_call": candidate_mean,
                "candidate_over_head": candidate_mean / head_mean,
                "bitwise_parity_each_call": True,
            }
        results[profile_name] = {
            "shape": list(values.shape),
            "dtype": values.dtype.str,
            "input_sha256": _profile_hash(values),
            "metrics": profile_results,
        }
    after = _source_hashes()
    if before != after:
        raise RuntimeError("quantile-shape source changed during the benchmark")
    report = {
        "benchmark": "quantile_shape linear guard CPU ABBA",
        "scope": "two deterministic Q20/F48 CPU profiles; no GPU or full-QE performance claim",
        "baseline": {
            "revision": "HEAD",
            "module": MODULE_PATH,
            "source_sha256": _sha256(baseline_source),
        },
        "candidate_source_sha256_before": before,
        "candidate_source_sha256_after": after,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "seed": 20261004,
        "normal_distribution": "normal(0.02, 0.004), float64",
        "schedule": "3 ABBA rounds; 10 calls per block; 6 samples per mode and metric",
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "quant_evaluator/docs/benchmarks/shape_linear_guard_ab_20261004.json",
    )
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
