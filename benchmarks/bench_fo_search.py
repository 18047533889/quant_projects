#!/usr/bin/env python3
"""
FO (Factor Optimization) search benchmark.

Tests: Factor mutation and search performance
- Small: 100 trials
- Medium: 1000 trials
- Large: 10000 trials
"""
from __future__ import annotations

import gc
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FO_ROOT = ROOT / "factor_optimizer"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FO_ROOT))


class MockSeenCache:
    """Simple seen-set based on structural hash."""

    def __init__(self):
        self.seen = set()
        self.collision_count = 0

    def mark_seen(self, candidate: dict) -> bool:
        """Return True if already seen."""
        key = self._hash(candidate)
        if key in self.seen:
            self.collision_count += 1
            return True
        self.seen.add(key)
        return False

    def _hash(self, candidate: dict) -> str:
        """Simple structural hash."""
        op = candidate.get("op", "unknown")
        params = candidate.get("params", {})
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        s = f"{op}_{param_str}"
        return hashlib.md5(s.encode()).hexdigest()[:16]

    def size(self) -> int:
        return len(self.seen)


def generate_candidate():
    """Generate random factor candidate."""
    rng = np.random.default_rng()

    ops = ["ts_mean", "ts_std", "ts_zscore", "ts_corr", "rank", "zscore",
           "log_returns", "volatility", "vwap", "ts_delta", "ts_sum"]

    op = rng.choice(ops)
    params = {}

    if op.startswith("ts_"):
        params["window"] = int(rng.choice([5, 10, 20, 60]))

    if op in ["ts_corr", "vwap"]:
        params["input2"] = rng.choice(["volume", "open", "high", "low"])

    if rng.random() < 0.3:
        params["min_periods"] = int(rng.choice([1, 5, 10]))

    return {
        "op": op,
        "params": params,
        "depth": int(rng.integers(1, 5)),
    }


def mutate_candidate(candidate: dict) -> dict:
    """Apply mutation to candidate."""
    rng = np.random.default_rng()
    new_candidate = candidate.copy()

    mutation_type = rng.choice(["param", "op", "depth"])

    if mutation_type == "param" and "params" in new_candidate:
        params = new_candidate["params"].copy()
        if "window" in params:
            params["window"] = int(params["window"] * rng.choice([0.5, 1.5, 2.0]))
        new_candidate["params"] = params

    elif mutation_type == "op":
        new_ops = ["ts_mean", "ts_std", "rank", "zscore"]
        new_candidate["op"] = rng.choice(new_ops)

    elif mutation_type == "depth":
        new_candidate["depth"] = max(1, candidate["depth"] + rng.choice([-1, 1]))

    return new_candidate


def validate_candidate(candidate: dict) -> bool:
    """Simple validation."""
    if "op" not in candidate:
        return False

    params = candidate.get("params", {})

    if "window" in params:
        if not (1 <= params["window"] <= 252):
            return False

    depth = candidate.get("depth", 1)
    if depth > 10:
        return False

    return True


def bench_mutation_generation(n_trials: int):
    """Benchmark mutation generation."""
    gc.collect()
    t0 = time.perf_counter()

    seed = generate_candidate()
    candidates = [seed]

    for _ in range(n_trials - 1):
        parent = candidates[-1]
        mutated = mutate_candidate(parent)
        candidates.append(mutated)

    elapsed = time.perf_counter() - t0
    return elapsed, len(candidates)


def bench_deduplication(n_trials: int):
    """Benchmark seen cache deduplication."""
    # Generate candidates
    candidates = [generate_candidate() for _ in range(n_trials)]

    gc.collect()
    t0 = time.perf_counter()

    cache = MockSeenCache()
    unique_count = 0

    for candidate in candidates:
        if not cache.mark_seen(candidate):
            unique_count += 1

    elapsed = time.perf_counter() - t0

    return elapsed, unique_count, cache.collision_count, cache.size()


def bench_validation(n_trials: int):
    """Benchmark candidate validation."""
    candidates = [generate_candidate() for _ in range(n_trials)]

    # Add some invalid ones
    rng = np.random.default_rng(456)
    for i in range(n_trials // 10):
        idx = rng.integers(0, len(candidates))
        candidates[idx]["params"]["window"] = 1000  # Invalid

    gc.collect()
    t0 = time.perf_counter()

    valid_count = sum(1 for c in candidates if validate_candidate(c))

    elapsed = time.perf_counter() - t0

    return elapsed, valid_count, n_trials


def bench_complexity_profile(n_trials: int):
    """Benchmark complexity profiling."""
    candidates = [generate_candidate() for _ in range(n_trials)]

    gc.collect()
    t0 = time.perf_counter()

    profiles = []
    for candidate in candidates:
        depth = candidate.get("depth", 1)
        params = candidate.get("params", {})
        window = params.get("window", 1)

        # Simple complexity heuristic
        complexity = depth * np.log(window + 1)
        memory_est = window * depth * 8  # bytes

        profiles.append({
            "complexity": complexity,
            "memory_bytes": memory_est,
        })

    elapsed = time.perf_counter() - t0

    return elapsed, len(profiles)


def run_fo_benchmark():
    """Run FO search benchmark suite."""
    scales = [
        ("small", 100),
        ("medium", 1000),
        ("large", 10000),
    ]

    results = {}

    for scale_name, n_trials in scales:
        print(f"\n=== {scale_name.upper()}: {n_trials} trials ===")

        scale_results = {
            "n_trials": n_trials,
            "operations": {},
        }

        # Test 1: Mutation generation
        try:
            elapsed, count = bench_mutation_generation(n_trials)
            throughput = count / elapsed
            scale_results["operations"]["mutation_gen"] = {
                "elapsed_s": round(elapsed, 4),
                "throughput_per_s": round(throughput, 1),
            }
            print(f"  Mutation generation:  {elapsed:6.3f}s  ({throughput:10.1f} trials/s)")
        except Exception as e:
            scale_results["operations"]["mutation_gen"] = {"error": str(e)}
            print(f"  Mutation generation:  FAILED - {e}")

        # Test 2: Deduplication
        try:
            elapsed, unique, collisions, cache_size = bench_deduplication(n_trials)
            throughput = n_trials / elapsed
            dedup_rate = collisions / n_trials * 100
            scale_results["operations"]["deduplication"] = {
                "elapsed_s": round(elapsed, 4),
                "throughput_per_s": round(throughput, 1),
                "unique_count": unique,
                "collision_count": collisions,
                "dedup_rate_pct": round(dedup_rate, 1),
                "cache_size": cache_size,
            }
            print(f"  Deduplication:        {elapsed:6.3f}s  ({throughput:10.1f} trials/s)")
            print(f"    Unique: {unique}/{n_trials}, Collisions: {collisions} ({dedup_rate:.1f}%)")
        except Exception as e:
            scale_results["operations"]["deduplication"] = {"error": str(e)}
            print(f"  Deduplication:        FAILED - {e}")

        # Test 3: Validation
        try:
            elapsed, valid, total = bench_validation(n_trials)
            throughput = total / elapsed
            valid_rate = valid / total * 100
            scale_results["operations"]["validation"] = {
                "elapsed_s": round(elapsed, 4),
                "throughput_per_s": round(throughput, 1),
                "valid_count": valid,
                "valid_rate_pct": round(valid_rate, 1),
            }
            print(f"  Validation:           {elapsed:6.3f}s  ({throughput:10.1f} trials/s)")
            print(f"    Valid: {valid}/{total} ({valid_rate:.1f}%)")
        except Exception as e:
            scale_results["operations"]["validation"] = {"error": str(e)}
            print(f"  Validation:           FAILED - {e}")

        # Test 4: Complexity profiling
        try:
            elapsed, count = bench_complexity_profile(n_trials)
            throughput = count / elapsed
            scale_results["operations"]["complexity_profile"] = {
                "elapsed_s": round(elapsed, 4),
                "throughput_per_s": round(throughput, 1),
            }
            print(f"  Complexity profile:   {elapsed:6.3f}s  ({throughput:10.1f} trials/s)")
        except Exception as e:
            scale_results["operations"]["complexity_profile"] = {"error": str(e)}
            print(f"  Complexity profile:   FAILED - {e}")

        results[scale_name] = scale_results

    return {
        "benchmark": "fo_search",
        "description": "Factor optimization search operations",
        "results": results,
    }


if __name__ == "__main__":
    result = run_fo_benchmark()

    print("\n=== FO Benchmark Summary ===")
    for scale, data in result["results"].items():
        print(f"\n{scale}: {data['n_trials']} trials")
        if data["operations"]:
            times = [v["elapsed_s"] for v in data["operations"].values() if "elapsed_s" in v]
            if times:
                print(f"  Mean time: {np.mean(times):.3f}s")
                print(f"  Total time: {sum(times):.3f}s")
