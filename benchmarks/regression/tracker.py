#!/usr/bin/env python3
"""
Performance benchmark result tracker.

Stores benchmark results with git commit metadata for regression tracking.
Results are stored in benchmarks/results/ with standardized naming.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkResult:
    """Single benchmark result with metadata."""
    commit_sha: str
    commit_date: str
    branch: str
    author: str
    benchmark_name: str
    metric_name: str
    value: float
    unit: str
    timestamp: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkTracker:
    """Track benchmark results across git commits."""

    def __init__(self, results_dir: Path | None = None):
        self.results_dir = results_dir or Path(__file__).parent.parent / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def get_git_info(self) -> dict[str, str]:
        """Extract current git commit metadata."""
        try:
            commit_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL
            ).strip()

            commit_date = subprocess.check_output(
                ["git", "show", "-s", "--format=%ci", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL
            ).strip()

            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL
            ).strip()

            author = subprocess.check_output(
                ["git", "show", "-s", "--format=%an", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL
            ).strip()

            return {
                "commit_sha": commit_sha,
                "commit_date": commit_date,
                "branch": branch,
                "author": author,
            }
        except subprocess.CalledProcessError as e:
            # Fallback for non-git environments
            return {
                "commit_sha": "unknown",
                "commit_date": datetime.now().isoformat(),
                "branch": "unknown",
                "author": "unknown",
            }

    def store_result(
        self,
        benchmark_name: str,
        metric_name: str,
        value: float,
        unit: str,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """Store a single benchmark result."""
        git_info = self.get_git_info()

        result = BenchmarkResult(
            commit_sha=git_info["commit_sha"],
            commit_date=git_info["commit_date"],
            branch=git_info["branch"],
            author=git_info["author"],
            benchmark_name=benchmark_name,
            metric_name=metric_name,
            value=value,
            unit=unit,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
        )

        # Append to results file
        results_file = self.results_dir / f"{benchmark_name}.jsonl"
        with results_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

        return result

    def store_benchmark_suite(self, suite_data: dict[str, Any]) -> list[BenchmarkResult]:
        """
        Store results from a complete benchmark suite.

        Expected format matches run_all_benchmarks.py output:
        {
            "benchmarks": {
                "qe": {"results": [...], "total_time_s": 10.5},
                "fp": {"results": {...}, "total_time_s": 5.2},
                ...
            }
        }
        """
        results = []
        git_info = self.get_git_info()

        for bench_name, bench_data in suite_data.get("benchmarks", {}).items():
            if "error" in bench_data:
                continue

            # Store total time
            if "total_time_s" in bench_data:
                result = self.store_result(
                    benchmark_name=bench_name,
                    metric_name="total_time",
                    value=bench_data["total_time_s"],
                    unit="seconds",
                    metadata={"git_info": git_info},
                )
                results.append(result)

            # Store benchmark-specific metrics
            if bench_name == "qe":
                for qe_result in bench_data.get("results", []):
                    if "error" not in qe_result:
                        n_factors = qe_result["n_factors"]
                        result = self.store_result(
                            benchmark_name=f"qe_{n_factors}",
                            metric_name="mean_time",
                            value=qe_result["mean_time_s"],
                            unit="seconds",
                            metadata={
                                "git_info": git_info,
                                "throughput": qe_result["throughput_factors_per_s"],
                                "per_factor_ms": qe_result["per_factor_ms"],
                            },
                        )
                        results.append(result)

            elif bench_name == "fp":
                for scale, scale_data in bench_data.get("results", {}).items():
                    for transform_name, transform_data in scale_data.get("transforms", {}).items():
                        if "elapsed_s" in transform_data:
                            result = self.store_result(
                                benchmark_name=f"fp_{scale}_{transform_name}",
                                metric_name="elapsed_time",
                                value=transform_data["elapsed_s"],
                                unit="seconds",
                                metadata={
                                    "git_info": git_info,
                                    "total_cells": scale_data.get("total_cells", 0),
                                },
                            )
                            results.append(result)

            elif bench_name == "fa":
                for scale, scale_data in bench_data.get("results", {}).items():
                    for op_name, op_data in scale_data.get("operations", {}).items():
                        if "elapsed_s" in op_data:
                            result = self.store_result(
                                benchmark_name=f"fa_{scale}_{op_name}",
                                metric_name="elapsed_time",
                                value=op_data["elapsed_s"],
                                unit="seconds",
                                metadata={
                                    "git_info": git_info,
                                    "n_records": scale_data.get("n_records", 0),
                                },
                            )
                            results.append(result)

            elif bench_name == "fo":
                for scale, scale_data in bench_data.get("results", {}).items():
                    for op_name, op_data in scale_data.get("operations", {}).items():
                        if "throughput_per_s" in op_data:
                            result = self.store_result(
                                benchmark_name=f"fo_{scale}_{op_name}",
                                metric_name="throughput",
                                value=op_data["throughput_per_s"],
                                unit="items_per_second",
                                metadata={
                                    "git_info": git_info,
                                    "n_trials": scale_data.get("n_trials", 0),
                                },
                            )
                            results.append(result)

        return results

    def load_results(self, benchmark_name: str) -> list[BenchmarkResult]:
        """Load all historical results for a benchmark."""
        results_file = self.results_dir / f"{benchmark_name}.jsonl"

        if not results_file.exists():
            return []

        results = []
        with results_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    results.append(BenchmarkResult(**data))

        return results

    def get_baseline(self, benchmark_name: str, commit_sha: str | None = None) -> BenchmarkResult | None:
        """Get baseline result for comparison (defaults to most recent)."""
        results = self.load_results(benchmark_name)

        if not results:
            return None

        if commit_sha:
            # Find specific commit
            for result in results:
                if result.commit_sha.startswith(commit_sha):
                    return result
            return None
        else:
            # Return most recent
            return results[-1]


def main():
    """CLI for storing benchmark results."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmark_json",
        type=Path,
        help="Path to benchmark results JSON (from run_all_benchmarks.py)"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Directory to store results (default: benchmarks/results/)"
    )
    args = parser.parse_args()

    if not args.benchmark_json.exists():
        print(f"Error: {args.benchmark_json} not found")
        return 1

    # Load benchmark data
    suite_data = json.loads(args.benchmark_json.read_text(encoding="utf-8"))

    # Store results
    tracker = BenchmarkTracker(results_dir=args.results_dir)
    results = tracker.store_benchmark_suite(suite_data)

    print(f"✓ Stored {len(results)} benchmark results")
    print(f"  Results directory: {tracker.results_dir}")

    git_info = tracker.get_git_info()
    print(f"  Commit: {git_info['commit_sha'][:8]}")
    print(f"  Branch: {git_info['branch']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
