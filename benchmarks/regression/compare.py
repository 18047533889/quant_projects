#!/usr/bin/env python3
"""
Compare current benchmark results against baseline.

Detects performance regressions by comparing current results
to a baseline (previous commit, specific SHA, or most recent).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tracker import BenchmarkTracker, BenchmarkResult


@dataclass
class Comparison:
    """Comparison between current and baseline results."""
    benchmark_name: str
    metric_name: str
    current_value: float
    baseline_value: float
    delta: float
    percent_change: float
    unit: str
    is_regression: bool
    current_commit: str
    baseline_commit: str

    @property
    def is_improvement(self) -> bool:
        """Check if this is a performance improvement."""
        # Lower is better for time metrics
        if "time" in self.metric_name.lower() or "elapsed" in self.metric_name.lower():
            return self.percent_change < 0
        # Higher is better for throughput
        elif "throughput" in self.metric_name.lower():
            return self.percent_change > 0
        else:
            return False

    def __str__(self) -> str:
        sign = "+" if self.percent_change > 0 else ""
        emoji = "🔴" if self.is_regression else "🟢" if self.is_improvement else "⚪"
        return (
            f"{emoji} {self.benchmark_name}/{self.metric_name}: "
            f"{self.current_value:.3f} vs {self.baseline_value:.3f} "
            f"({sign}{self.percent_change:.1f}%)"
        )


class BenchmarkComparator:
    """Compare benchmark results and detect regressions."""

    def __init__(
        self,
        tracker: BenchmarkTracker,
        regression_threshold_pct: float = 10.0,
    ):
        self.tracker = tracker
        self.regression_threshold_pct = regression_threshold_pct

    def compare_suite(
        self,
        current_data: dict[str, Any],
        baseline_commit: str | None = None,
    ) -> list[Comparison]:
        """
        Compare current benchmark suite against baseline.

        Args:
            current_data: Current benchmark results (from run_all_benchmarks.py)
            baseline_commit: Baseline commit SHA (None = most recent)

        Returns:
            List of comparisons for all metrics
        """
        comparisons = []
        git_info = self.tracker.get_git_info()
        current_commit = git_info["commit_sha"]

        for bench_name, bench_data in current_data.get("benchmarks", {}).items():
            if "error" in bench_data:
                continue

            # Compare total time
            if "total_time_s" in bench_data:
                baseline = self.tracker.get_baseline(bench_name, baseline_commit)
                if baseline:
                    comp = self._compare_values(
                        benchmark_name=bench_name,
                        metric_name="total_time",
                        current_value=bench_data["total_time_s"],
                        baseline_value=baseline.value,
                        unit="seconds",
                        current_commit=current_commit,
                        baseline_commit=baseline.commit_sha,
                    )
                    comparisons.append(comp)

            # Compare benchmark-specific metrics
            if bench_name == "qe":
                for qe_result in bench_data.get("results", []):
                    if "error" not in qe_result:
                        n_factors = qe_result["n_factors"]
                        baseline = self.tracker.get_baseline(
                            f"qe_{n_factors}", baseline_commit
                        )
                        if baseline:
                            comp = self._compare_values(
                                benchmark_name=f"qe_{n_factors}",
                                metric_name="mean_time",
                                current_value=qe_result["mean_time_s"],
                                baseline_value=baseline.value,
                                unit="seconds",
                                current_commit=current_commit,
                                baseline_commit=baseline.commit_sha,
                            )
                            comparisons.append(comp)

            elif bench_name == "fp":
                for scale, scale_data in bench_data.get("results", {}).items():
                    for transform_name, transform_data in scale_data.get("transforms", {}).items():
                        if "elapsed_s" in transform_data:
                            baseline = self.tracker.get_baseline(
                                f"fp_{scale}_{transform_name}", baseline_commit
                            )
                            if baseline:
                                comp = self._compare_values(
                                    benchmark_name=f"fp_{scale}_{transform_name}",
                                    metric_name="elapsed_time",
                                    current_value=transform_data["elapsed_s"],
                                    baseline_value=baseline.value,
                                    unit="seconds",
                                    current_commit=current_commit,
                                    baseline_commit=baseline.commit_sha,
                                )
                                comparisons.append(comp)

            elif bench_name == "fa":
                for scale, scale_data in bench_data.get("results", {}).items():
                    for op_name, op_data in scale_data.get("operations", {}).items():
                        if "elapsed_s" in op_data:
                            baseline = self.tracker.get_baseline(
                                f"fa_{scale}_{op_name}", baseline_commit
                            )
                            if baseline:
                                comp = self._compare_values(
                                    benchmark_name=f"fa_{scale}_{op_name}",
                                    metric_name="elapsed_time",
                                    current_value=op_data["elapsed_s"],
                                    baseline_value=baseline.value,
                                    unit="seconds",
                                    current_commit=current_commit,
                                    baseline_commit=baseline.commit_sha,
                                )
                                comparisons.append(comp)

            elif bench_name == "fo":
                for scale, scale_data in bench_data.get("results", {}).items():
                    for op_name, op_data in scale_data.get("operations", {}).items():
                        if "throughput_per_s" in op_data:
                            baseline = self.tracker.get_baseline(
                                f"fo_{scale}_{op_name}", baseline_commit
                            )
                            if baseline:
                                comp = self._compare_values(
                                    benchmark_name=f"fo_{scale}_{op_name}",
                                    metric_name="throughput",
                                    current_value=op_data["throughput_per_s"],
                                    baseline_value=baseline.value,
                                    unit="items_per_second",
                                    current_commit=current_commit,
                                    baseline_commit=baseline.commit_sha,
                                )
                                comparisons.append(comp)

        return comparisons

    def _compare_values(
        self,
        benchmark_name: str,
        metric_name: str,
        current_value: float,
        baseline_value: float,
        unit: str,
        current_commit: str,
        baseline_commit: str,
    ) -> Comparison:
        """Compare two values and determine if regression occurred."""
        delta = current_value - baseline_value
        percent_change = (delta / baseline_value * 100) if baseline_value != 0 else 0

        # Determine regression based on metric type
        is_regression = False
        if "time" in metric_name.lower() or "elapsed" in metric_name.lower():
            # Higher time is worse
            is_regression = percent_change > self.regression_threshold_pct
        elif "throughput" in metric_name.lower():
            # Lower throughput is worse
            is_regression = percent_change < -self.regression_threshold_pct

        return Comparison(
            benchmark_name=benchmark_name,
            metric_name=metric_name,
            current_value=current_value,
            baseline_value=baseline_value,
            delta=delta,
            percent_change=percent_change,
            unit=unit,
            is_regression=is_regression,
            current_commit=current_commit,
            baseline_commit=baseline_commit,
        )

    def get_regressions(self, comparisons: list[Comparison]) -> list[Comparison]:
        """Filter for regressions only."""
        return [c for c in comparisons if c.is_regression]

    def get_improvements(self, comparisons: list[Comparison]) -> list[Comparison]:
        """Filter for improvements only."""
        return [c for c in comparisons if c.is_improvement]


def main():
    """CLI for comparing benchmark results."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "current_json",
        type=Path,
        help="Path to current benchmark results JSON"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="Baseline commit SHA (default: most recent)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Regression threshold percentage (default: 10.0)"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Directory with historical results"
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 1 if regressions detected"
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Write comparison results to JSON file"
    )
    args = parser.parse_args()

    if not args.current_json.exists():
        print(f"Error: {args.current_json} not found")
        return 1

    # Load current results
    current_data = json.loads(args.current_json.read_text(encoding="utf-8"))

    # Compare
    tracker = BenchmarkTracker(results_dir=args.results_dir)
    comparator = BenchmarkComparator(
        tracker=tracker,
        regression_threshold_pct=args.threshold,
    )

    comparisons = comparator.compare_suite(current_data, args.baseline)

    if not comparisons:
        print("No baseline found for comparison")
        return 0

    # Print results
    regressions = comparator.get_regressions(comparisons)
    improvements = comparator.get_improvements(comparisons)
    neutral = [c for c in comparisons if not c.is_regression and not c.is_improvement]

    print(f"\n{'='*80}")
    print("BENCHMARK COMPARISON REPORT")
    print(f"{'='*80}")
    print(f"Threshold: {args.threshold}%")
    print(f"Total comparisons: {len(comparisons)}")
    print(f"Regressions: {len(regressions)}")
    print(f"Improvements: {len(improvements)}")
    print(f"Neutral: {len(neutral)}")

    if regressions:
        print(f"\n{'='*80}")
        print("🔴 REGRESSIONS DETECTED")
        print(f"{'='*80}")
        for comp in regressions:
            print(f"  {comp}")

    if improvements:
        print(f"\n{'='*80}")
        print("🟢 IMPROVEMENTS")
        print(f"{'='*80}")
        for comp in improvements:
            print(f"  {comp}")

    if neutral and not regressions and not improvements:
        print(f"\n{'='*80}")
        print("⚪ ALL RESULTS WITHIN THRESHOLD")
        print(f"{'='*80}")

    # Save JSON output if requested
    if args.json_output:
        output_data = {
            "threshold_pct": args.threshold,
            "total_comparisons": len(comparisons),
            "regressions": len(regressions),
            "improvements": len(improvements),
            "neutral": len(neutral),
            "comparisons": [
                {
                    "benchmark_name": c.benchmark_name,
                    "metric_name": c.metric_name,
                    "current_value": c.current_value,
                    "baseline_value": c.baseline_value,
                    "percent_change": c.percent_change,
                    "is_regression": c.is_regression,
                    "is_improvement": c.is_improvement,
                    "current_commit": c.current_commit,
                    "baseline_commit": c.baseline_commit,
                }
                for c in comparisons
            ],
        }
        args.json_output.write_text(
            json.dumps(output_data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8"
        )
        print(f"\n✓ Comparison results written to {args.json_output}")

    # Exit with error if regressions detected and flag is set
    if args.fail_on_regression and regressions:
        print(f"\n❌ Exiting with error due to {len(regressions)} regression(s)")
        return 1

    print(f"\n{'='*80}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
