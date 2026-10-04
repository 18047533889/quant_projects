"""Bounded CPU-only ABBA benchmark for moving-block lower-bound scratch reuse.

Run with the project virtualenv from any directory. Defaults are intentionally
small and bounded; this benchmark uses synthetic arrays only (no FE/GPU/COS).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

from factor_optimizer.research_bootstrap import moving_block_lower_bound


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = PROJECT_ROOT / "factor_optimizer/factor_optimizer/research_bootstrap.py"
MAX_N = 10_000
MAX_DRAWS = 5_000
MAX_ROUNDS = 5
MAX_CASES = 16
MAX_TOTAL_OPERATIONS = 400_000_000
BLOCK_LENGTH = 5
HORIZON = 5
MINIMUM_VALIDATION_DAYS = 30
SEED = 7341
CONFIDENCE_LEVEL = 0.95
ORDER = ("old", "scratch", "scratch", "old")


def _parse_csv_ints(value: str, label: str) -> list[int]:
    try:
        items = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError(f"{label} must be comma-separated integers") from exc
    if not items:
        raise ValueError(f"{label} must not be empty")
    return items


def _validate_budget(sizes: list[int], draws: list[int], rounds: int) -> None:
    if not sizes or any(type(n) is not int or n < 3 * max(BLOCK_LENGTH, HORIZON) for n in sizes):
        raise ValueError("sizes must be integers >= 15")
    if any(n > MAX_N for n in sizes):
        raise ValueError(f"size exceeds maximum allowed n (max {MAX_N})")
    if not draws or any(type(count) is not int or count < 1 for count in draws):
        raise ValueError("draws must be positive integers")
    if any(count > MAX_DRAWS for count in draws):
        raise ValueError(f"draw count exceeds maximum allowed bootstrap draws (max {MAX_DRAWS})")
    if type(rounds) is not int or not 1 <= rounds <= MAX_ROUNDS:
        raise ValueError(f"rounds must be between 1 and max {MAX_ROUNDS}")
    if len(sizes) * len(draws) > MAX_CASES:
        raise ValueError(f"case count exceeds maximum allowed (max {MAX_CASES} cases)")
    total_operations = sum(sizes) * sum(draws) * (rounds * len(ORDER) + 2)
    if total_operations > MAX_TOTAL_OPERATIONS:
        raise ValueError(
            "aggregate workload exceeds maximum allowed operations "
            f"({MAX_TOTAL_OPERATIONS})"
        )


def _old_lower_bound(differences: np.ndarray, *, bootstrap_draws: int) -> float | None:
    """Literal independent reference for the pre-extraction implementation."""
    length = max(BLOCK_LENGTH, int(HORIZON))
    n = len(differences)
    if n < 3 * length:
        return None
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(bootstrap_draws):
        starts = rng.integers(0, n - length + 1, size=math.ceil(n / length))
        values = np.concatenate([differences[s:s + length] for s in starts])[:n]
        values = values[np.isfinite(values)]
        if len(values) < MINIMUM_VALIDATION_DAYS:
            return None
        draws.append(float(values.mean()))
    return float(np.quantile(draws, (1 - CONFIDENCE_LEVEL) / 2))


def _scratch_lower_bound(differences: np.ndarray, *, bootstrap_draws: int) -> float | None:
    return moving_block_lower_bound(
        differences,
        block_length=BLOCK_LENGTH,
        horizon=HORIZON,
        minimum_validation_days=MINIMUM_VALIDATION_DAYS,
        bootstrap_draws=bootstrap_draws,
        seed=SEED,
        confidence_level=CONFIDENCE_LEVEL,
    )


def _benchmark_case(values: np.ndarray, *, bootstrap_draws: int, rounds: int) -> dict:
    expected = _old_lower_bound(values, bootstrap_draws=bootstrap_draws)
    if expected is None or not math.isfinite(expected):
        raise RuntimeError("reference produced no finite result before timed runs")
    scratch_result = _scratch_lower_bound(values, bootstrap_draws=bootstrap_draws)
    if scratch_result is None or not math.isfinite(scratch_result):
        raise RuntimeError("scratch produced no finite result before timed runs")
    if scratch_result != expected:
        raise RuntimeError("old and scratch results differ before timed runs")

    raw = {"old": [], "scratch": []}
    funcs = {"old": _old_lower_bound, "scratch": _scratch_lower_bound}
    for _ in range(rounds):
        for label in ORDER:
            started = time.perf_counter()
            result = funcs[label](values, bootstrap_draws=bootstrap_draws)
            elapsed = time.perf_counter() - started
            raw[label].append(elapsed)
            if result is None or not math.isfinite(result):
                raise RuntimeError(f"{label} produced no finite result during timed run")
            if result != expected:
                raise RuntimeError("old and scratch results differ during timed run")
    medians = {name: statistics.median(times) for name, times in raw.items()}
    return {
        "n": int(values.size),
        "bootstrap_draws": bootstrap_draws,
        "rounds": rounds,
        "parity": True,
        "raw_times_seconds": raw,
        "median_seconds": medians,
        "speedup_old_over_scratch": medians["old"] / medians["scratch"],
    }


def _run(sizes: list[int], draws: list[int], rounds: int) -> dict:
    # All work bounds are checked before allocating any synthetic arrays.
    _validate_budget(sizes, draws, rounds)
    cases = []
    for n in sizes:
        values = np.random.default_rng(20261004 + n).normal(size=n).astype(np.float64)
        values[::37] = np.nan
        for bootstrap_draws in draws:
            cases.append(_benchmark_case(values, bootstrap_draws=bootstrap_draws, rounds=rounds))
    return {
        "schema": "moving-block-bootstrap-abba-v1",
        "benchmark": "moving_block_lower_bound",
        "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
        "order": list(ORDER),
        "config": {
            "sizes": sizes,
            "draw_counts": draws,
            "rounds": rounds,
            "block_length": BLOCK_LENGTH,
            "horizon": HORIZON,
            "minimum_validation_days": MINIMUM_VALIDATION_DAYS,
            "seed": SEED,
            "confidence_level": CONFIDENCE_LEVEL,
            "input_seed_rule": "default_rng(20261004+n); normal float64; NaN at [::37]",
            "timed_boundary": "complete helper call; excludes fixture generation",
            "hard_limits": {"max_n": MAX_N, "max_bootstrap_draws": MAX_DRAWS,
                            "max_rounds": MAX_ROUNDS, "max_cases": MAX_CASES,
                            "max_total_operations": MAX_TOTAL_OPERATIONS},
        },
        "source_sha256_scope": "factor_optimizer/factor_optimizer/research_bootstrap.py",
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="1000,2500", help="comma-separated n values; max 10000")
    parser.add_argument("--draws", default="499,1999", help="comma-separated draw counts; max 5000")
    parser.add_argument("--rounds", type=int, default=3, help="ABBA rounds; max 5")
    args = parser.parse_args(argv)
    try:
        sizes = _parse_csv_ints(args.sizes, "sizes")
        draws = _parse_csv_ints(args.draws, "draws")
        _validate_budget(sizes, draws, args.rounds)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        payload = _run(sizes, draws, args.rounds)
    except RuntimeError as exc:
        parser.error(str(exc))
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
