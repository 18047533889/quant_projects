#!/usr/bin/env python3
"""
Comprehensive benchmark runner for quantitative platform.

Executes all four benchmark suites:
- QE: Quant Evaluator metrics at scale
- FP: Factor Processing transforms
- FA: Fundamental Analysis operations
- FO: Factor Optimization search

Generates baseline.json with consolidated results.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))


def run_all_benchmarks(quick_mode: bool = False):
    """Execute all benchmark suites."""
    results = {
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "platform": "quantitative_analysis",
        "benchmarks": {},
    }

    print("=" * 80)
    print("QUANTITATIVE PLATFORM COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 80)

    # Benchmark 1: FO (Factor Optimization) - lightest, no dependencies
    print("\n[1/4] Running FO (Factor Optimization) benchmark...")
    try:
        from bench_fo_search import run_fo_benchmark
        t0 = time.perf_counter()
        fo_result = run_fo_benchmark()
        fo_result["total_time_s"] = round(time.perf_counter() - t0, 2)
        results["benchmarks"]["fo"] = fo_result
        print(f"✓ FO benchmark completed in {fo_result['total_time_s']}s")
    except Exception as e:
        print(f"✗ FO benchmark failed: {e}")
        results["benchmarks"]["fo"] = {"error": str(e)}

    # Benchmark 2: FA (Fundamental Analysis)
    print("\n[2/4] Running FA (Fundamental Analysis) benchmark...")
    try:
        from bench_fa_operations import run_fa_benchmark
        t0 = time.perf_counter()
        fa_result = run_fa_benchmark()
        fa_result["total_time_s"] = round(time.perf_counter() - t0, 2)
        results["benchmarks"]["fa"] = fa_result
        print(f"✓ FA benchmark completed in {fa_result['total_time_s']}s")
    except Exception as e:
        print(f"✗ FA benchmark failed: {e}")
        results["benchmarks"]["fa"] = {"error": str(e)}

    # Benchmark 3: FP (Factor Processing)
    print("\n[3/4] Running FP (Factor Processing) benchmark...")
    try:
        sys.path.insert(0, str(ROOT.parent / "factor_engine"))
        from cleaned_operators import load_all
        load_all()

        from bench_fp_transforms import run_fp_benchmark
        t0 = time.perf_counter()
        fp_result = run_fp_benchmark()
        fp_result["total_time_s"] = round(time.perf_counter() - t0, 2)
        results["benchmarks"]["fp"] = fp_result
        print(f"✓ FP benchmark completed in {fp_result['total_time_s']}s")
    except Exception as e:
        print(f"✗ FP benchmark failed: {e}")
        results["benchmarks"]["fp"] = {"error": str(e)}

    # Benchmark 4: QE (Quant Evaluator)
    print("\n[4/4] Running QE (Quant Evaluator) benchmark...")
    try:
        from bench_qe_metrics import run_qe_benchmark
        t0 = time.perf_counter()
        qe_result = run_qe_benchmark()
        qe_result["total_time_s"] = round(time.perf_counter() - t0, 2)
        results["benchmarks"]["qe"] = qe_result
        print(f"✓ QE benchmark completed in {qe_result['total_time_s']}s")
    except Exception as e:
        print(f"✗ QE benchmark failed: {e}")
        results["benchmarks"]["qe"] = {"error": str(e)}

    return results


def generate_summary_report(results: dict) -> str:
    """Generate human-readable summary."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("BENCHMARK SUMMARY REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {results['generated_at']}")
    lines.append(f"Platform: {results['platform']}")
    lines.append("")

    for bench_name, bench_data in results["benchmarks"].items():
        lines.append(f"\n{bench_name.upper()} - {bench_data.get('description', 'N/A')}")
        lines.append("-" * 80)

        if "error" in bench_data:
            lines.append(f"  ✗ ERROR: {bench_data['error']}")
            continue

        total_time = bench_data.get("total_time_s", 0)
        lines.append(f"  Total time: {total_time}s")

        # Benchmark-specific summaries
        if bench_name == "qe":
            qe_results = bench_data.get("results", [])
            for r in qe_results:
                if "error" in r:
                    lines.append(f"  • {r['n_factors']} factors: ERROR")
                else:
                    lines.append(
                        f"  • {r['n_factors']:5d} factors: "
                        f"{r['mean_time_s']:6.3f}s "
                        f"({r['throughput_factors_per_s']:6.1f} factors/s, "
                        f"{r['per_factor_ms']:6.3f}ms/factor)"
                    )

        elif bench_name == "fp":
            fp_results = bench_data.get("results", {})
            for scale, scale_data in fp_results.items():
                n_cells = scale_data.get("total_cells", 0)
                transforms = scale_data.get("transforms", {})
                if transforms:
                    avg_time = sum(
                        t["elapsed_s"] for t in transforms.values() if "elapsed_s" in t
                    ) / len([t for t in transforms.values() if "elapsed_s" in t])
                    lines.append(
                        f"  • {scale:8s} ({n_cells:12,d} cells): "
                        f"{avg_time:6.3f}s avg per transform"
                    )

        elif bench_name == "fa":
            fa_results = bench_data.get("results", {})
            for scale, scale_data in fa_results.items():
                n_records = scale_data.get("n_records", 0)
                ops = scale_data.get("operations", {})
                if ops:
                    successful = [o for o in ops.values() if "elapsed_s" in o]
                    if successful:
                        avg_time = sum(o["elapsed_s"] for o in successful) / len(successful)
                        lines.append(
                            f"  • {scale:8s} ({n_records:8,d} records): "
                            f"{avg_time:6.3f}s avg per operation"
                        )

        elif bench_name == "fo":
            fo_results = bench_data.get("results", {})
            for scale, scale_data in fo_results.items():
                n_trials = scale_data.get("n_trials", 0)
                ops = scale_data.get("operations", {})
                if ops:
                    dedup = ops.get("deduplication", )
                    if "throughput_per_s" in dedup:
                        lines.append(
                            f"  • {scale:8s} ({n_trials:6d} trials): "
                            f"{dedup['throughput_per_s']:10.1f} trials/s, "
                            f"{dedup.get('dedup_rate_pct', 0):4.1f}% dedup rate"
                        )

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def main():
    """Main benchmark runner."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in quick mode with reduced scale"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "baseline.json",
        help="Output JSON file path"
    )
    args = parser.parse_args()

    # Run benchmarks
    results = run_all_benchmarks(quick_mode=args.quick)

    # Save results
    args.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )
    print(f"\n✓ Results written to {args.output}")

    # Generate and display summary
    summary = generate_summary_report(results)
    print(summary)

    # Save summary as text
    summary_path = args.output.with_suffix(".txt")
    summary_path.write_text(summary, encoding="utf-8")
    print(f"✓ Summary written to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
