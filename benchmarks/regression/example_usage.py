#!/usr/bin/env python3
"""
Example usage of performance regression tracking system.

Demonstrates the complete workflow for tracking and comparing benchmarks.
"""
from pathlib import Path
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tracker import BenchmarkTracker
from compare import BenchmarkComparator
from report import RegressionReporter


def example_basic_workflow():
    """Basic workflow: store and compare results."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Workflow")
    print("=" * 80)

    # Initialize tracker with temp directory
    results_dir = Path(__file__).parent.parent / "results"
    tracker = BenchmarkTracker(results_dir=results_dir)

    # Simulate storing results from a benchmark run
    print("\n1. Storing baseline results...")
    for i in range(3):
        tracker.store_result(
            benchmark_name="example_bench",
            metric_name="execution_time",
            value=1.0 + i * 0.1,
            unit="seconds",
            metadata={"iteration": i},
        )
    print(f"   ✓ Stored 3 results to {results_dir}")

    # Load and display stored results
    print("\n2. Loading historical results...")
    results = tracker.load_results("example_bench")
    print(f"   ✓ Found {len(results)} historical results")
    for r in results[-3:]:
        print(f"     - {r.commit_sha[:8]}: {r.value:.2f} {r.unit}")

    # Get baseline
    print("\n3. Getting baseline for comparison...")
    baseline = tracker.get_baseline("example_bench")
    if baseline:
        print(f"   ✓ Baseline: {baseline.value:.2f} {baseline.unit}")
        print(f"     Commit: {baseline.commit_sha[:8]}")
        print(f"     Branch: {baseline.branch}")


def example_regression_detection():
    """Example: Detect performance regressions."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Regression Detection")
    print("=" * 80)

    results_dir = Path(__file__).parent.parent / "results"
    tracker = BenchmarkTracker(results_dir=results_dir)

    # Store baseline
    print("\n1. Establishing baseline (10 seconds)...")
    tracker.store_result(
        benchmark_name="regression_test",
        metric_name="total_time",
        value=10.0,
        unit="seconds",
    )

    # Simulate current run with regression
    print("\n2. Simulating current run (12 seconds - 20% regression)...")
    current_data = {
        "benchmarks": {
            "regression_test": {
                "total_time_s": 12.0,  # 20% slower
            }
        }
    }

    # Compare
    print("\n3. Running comparison...")
    comparator = BenchmarkComparator(tracker, regression_threshold_pct=10.0)
    comparisons = comparator.compare_suite(current_data)

    if comparisons:
        comp = comparisons[0]
        print(f"   Benchmark: {comp.benchmark_name}")
        print(f"   Current: {comp.current_value:.2f} {comp.unit}")
        print(f"   Baseline: {comp.baseline_value:.2f} {comp.unit}")
        print(f"   Change: {comp.percent_change:+.1f}%")
        print(f"   Status: {'🔴 REGRESSION' if comp.is_regression else '🟢 OK'}")


def example_full_suite():
    """Example: Process full benchmark suite."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Full Benchmark Suite")
    print("=" * 80)

    # Load actual benchmark results if available
    baseline_file = Path(__file__).parent.parent / "baseline.json"

    if not baseline_file.exists():
        print("\n❌ No baseline.json found. Run benchmarks first:")
        print("   cd /home/shw/quant_projects/benchmarks")
        print("   python run_all_benchmarks.py")
        return

    print(f"\n1. Loading benchmark suite from {baseline_file.name}...")
    suite_data = json.loads(baseline_file.read_text(encoding="utf-8"))

    results_dir = Path(__file__).parent.parent / "results"
    tracker = BenchmarkTracker(results_dir=results_dir)

    print("\n2. Storing results...")
    stored = tracker.store_benchmark_suite(suite_data)
    print(f"   ✓ Stored {len(stored)} benchmark results")

    # Show some examples
    print("\n3. Sample stored benchmarks:")
    for bench_name in ["qe", "fp", "fa", "fo"]:
        results = tracker.load_results(bench_name)
        if results:
            latest = results[-1]
            print(f"   - {bench_name}: {latest.value:.2f} {latest.unit}")


def example_report_generation():
    """Example: Generate reports."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Report Generation")
    print("=" * 80)

    results_dir = Path(__file__).parent.parent / "results"
    comparison_file = results_dir / "comparison.json"

    if not comparison_file.exists():
        print("\n❌ No comparison.json found. Run comparison first:")
        print("   cd /home/shw/quant_projects/benchmarks/regression")
        print("   python compare.py ../baseline.json --json-output ../results/comparison.json")
        return

    print(f"\n1. Loading comparison data...")
    comparison_data = json.loads(comparison_file.read_text(encoding="utf-8"))
    comparisons = comparison_data.get("comparisons", [])
    print(f"   ✓ Loaded {len(comparisons)} comparisons")

    print("\n2. Generating reports...")
    tracker = BenchmarkTracker(results_dir=results_dir)
    reporter = RegressionReporter(tracker)

    # Generate HTML
    html_path = results_dir / "example_report.html"
    reporter.generate_html_report(comparisons, html_path)
    print(f"   ✓ HTML report: {html_path}")

    # Generate Markdown
    md_path = results_dir / "example_report.md"
    reporter.generate_markdown_report(comparisons, md_path)
    print(f"   ✓ Markdown report: {md_path}")


def example_ci_integration():
    """Example: CI integration."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: CI Integration")
    print("=" * 80)

    print("\n1. Manual CI check:")
    print("   cd /home/shw/quant_projects/benchmarks/regression")
    print("   ./ci_check.sh --threshold 10.0")

    print("\n2. With baseline commit:")
    print("   ./ci_check.sh --threshold 5.0 --baseline abc123")

    print("\n3. In GitHub Actions:")
    print("   See .github_workflows_example.yml for complete workflow")

    print("\n4. Key CI features:")
    print("   ✓ Automatic benchmark execution")
    print("   ✓ Result storage with git metadata")
    print("   ✓ Baseline comparison")
    print("   ✓ Report generation")
    print("   ✓ Fail build on regression (configurable threshold)")
    print("   ✓ PR comments with performance summary")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "PERFORMANCE REGRESSION TRACKING EXAMPLES" + " " * 23 + "║")
    print("╚" + "═" * 78 + "╝")

    try:
        example_basic_workflow()
        example_regression_detection()
        example_full_suite()
        example_report_generation()
        example_ci_integration()

        print("\n" + "=" * 80)
        print("✅ All examples completed!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Run benchmarks: cd .. && python run_all_benchmarks.py")
        print("  2. Store results: python regression/tracker.py baseline.json")
        print("  3. Compare: python regression/compare.py baseline.json")
        print("  4. Generate report: python regression/report.py results/comparison.json --html report.html")
        print("\nOr use the CI integration script:")
        print("  cd regression && ./ci_check.sh")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
