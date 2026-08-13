#!/usr/bin/env python3
"""
benchmark_cli: Command-line interface for running benchmarks.

Execute and analyze platform benchmarks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))


@click.group()
@click.version_option(version="0.1.0")
def benchmark_cli():
    """Benchmark CLI - Run and analyze platform benchmarks."""
    pass


@benchmark_cli.command()
@click.option("--suite", "-s", multiple=True,
              type=click.Choice(["qe", "fp", "fa", "fo", "all"]),
              default=["all"], help="Benchmark suite(s) to run")
@click.option("--output", "-o", type=click.Path(),
              default="baseline.json", help="Output file path")
@click.option("--quick", is_flag=True, help="Run in quick mode (reduced scale)")
def run(suite: tuple[str, ...], output: str, quick: bool):
    """Run benchmark suite(s).

    Available suites:
      qe - Quant Evaluator metrics
      fp - Factor Preprocessing transforms
      fa - Factor Analysis operations
      fo - Factor Optimization search
      all - All suites
    """
    suites_to_run = set(suite)
    if "all" in suites_to_run:
        suites_to_run = {"qe", "fp", "fa", "fo"}

    click.echo("=" * 80)
    click.echo("QUANTITATIVE PLATFORM BENCHMARK SUITE")
    click.echo("=" * 80)
    click.echo(f"Running: {', '.join(sorted(suites_to_run))}")
    click.echo(f"Mode: {'quick' if quick else 'full'}")
    click.echo(f"Output: {output}")

    results = {
        "schema_version": "1.0",
        "mode": "quick" if quick else "full",
        "benchmarks": {},
    }

    import time
    from datetime import datetime
    results["generated_at"] = datetime.now().isoformat()

    # Run FO (lightest, no dependencies)
    if "fo" in suites_to_run:
        click.echo("\n[FO] Factor Optimization benchmark...")
        try:
            from bench_fo_search import run_fo_benchmark
            t0 = time.perf_counter()
            fo_result = run_fo_benchmark()
            fo_result["total_time_s"] = round(time.perf_counter() - t0, 2)
            results["benchmarks"]["fo"] = fo_result
            click.echo(f"✓ Completed in {fo_result['total_time_s']}s")
        except Exception as e:
            click.echo(f"✗ Failed: {e}", err=True)
            results["benchmarks"]["fo"] = {"error": str(e)}

    # Run FA
    if "fa" in suites_to_run:
        click.echo("\n[FA] Factor Analysis benchmark...")
        try:
            from bench_fa_operations import run_fa_benchmark
            t0 = time.perf_counter()
            fa_result = run_fa_benchmark()
            fa_result["total_time_s"] = round(time.perf_counter() - t0, 2)
            results["benchmarks"]["fa"] = fa_result
            click.echo(f"✓ Completed in {fa_result['total_time_s']}s")
        except Exception as e:
            click.echo(f"✗ Failed: {e}", err=True)
            results["benchmarks"]["fa"] = {"error": str(e)}

    # Run FP
    if "fp" in suites_to_run:
        click.echo("\n[FP] Factor Preprocessing benchmark...")
        try:
            sys.path.insert(0, str(ROOT / "factor_engine"))
            from cleaned_operators import load_all
            load_all()

            from bench_fp_transforms import run_fp_benchmark
            t0 = time.perf_counter()
            fp_result = run_fp_benchmark()
            fp_result["total_time_s"] = round(time.perf_counter() - t0, 2)
            results["benchmarks"]["fp"] = fp_result
            click.echo(f"✓ Completed in {fp_result['total_time_s']}s")
        except Exception as e:
            click.echo(f"✗ Failed: {e}", err=True)
            results["benchmarks"]["fp"] = {"error": str(e)}

    # Run QE
    if "qe" in suites_to_run:
        click.echo("\n[QE] Quant Evaluator benchmark...")
        try:
            from bench_qe_metrics import run_qe_benchmark
            t0 = time.perf_counter()
            qe_result = run_qe_benchmark()
            qe_result["total_time_s"] = round(time.perf_counter() - t0, 2)
            results["benchmarks"]["qe"] = qe_result
            click.echo(f"✓ Completed in {qe_result['total_time_s']}s")
        except Exception as e:
            click.echo(f"✗ Failed: {e}", err=True)
            results["benchmarks"]["qe"] = {"error": str(e)}

    # Save results
    output_path = Path(output)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    click.echo(f"\n✓ Results saved to {output}")


@benchmark_cli.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option("--format", "-f", type=click.Choice(["text", "json", "csv"]),
              default="text", help="Output format")
def report(results_file: str, format: str):
    """Generate report from benchmark results."""
    results_path = Path(results_file)
    data = json.loads(results_path.read_text())

    if format == "json":
        click.echo(json.dumps(data, indent=2))
        return

    if format == "csv":
        click.echo("suite,metric,value")
        for suite_name, suite_data in data.get("benchmarks", {}).items():
            if "error" in suite_data:
                click.echo(f"{suite_name},error,{suite_data['error']}")
                continue
            total_time = suite_data.get("total_time_s", 0)
            click.echo(f"{suite_name},total_time_s,{total_time}")
        return

    # Text format
    click.echo("=" * 80)
    click.echo("BENCHMARK REPORT")
    click.echo("=" * 80)
    click.echo(f"Generated: {data.get('generated_at', 'unknown')}")
    click.echo(f"Mode: {data.get('mode', 'unknown')}")

    for suite_name, suite_data in data.get("benchmarks", {}).items():
        click.echo(f"\n{suite_name.upper()}: {suite_data.get('description', '')}")
        click.echo("-" * 80)

        if "error" in suite_data:
            click.echo(f"  ✗ ERROR: {suite_data['error']}")
            continue

        total_time = suite_data.get("total_time_s", 0)
        click.echo(f"  Total time: {total_time:.2f}s")

        # Suite-specific details
        if suite_name == "qe":
            for r in suite_data.get("results", []):
                if "error" in r:
                    click.echo(f"  • {r['n_factors']} factors: ERROR")
                else:
                    click.echo(
                        f"  • {r['n_factors']:5d} factors: {r['mean_time_s']:6.3f}s "
                        f"({r['throughput_factors_per_s']:6.1f} factors/s)"
                    )

        elif suite_name == "fp":
            for scale, scale_data in suite_data.get("results", {}).items():
                n_cells = scale_data.get("total_cells", 0)
                click.echo(f"  • {scale:10s}: {n_cells:12,d} cells")

        elif suite_name == "fa":
            for scale, scale_data in suite_data.get("results", {}).items():
                n_records = scale_data.get("n_records", 0)
                click.echo(f"  • {scale:10s}: {n_records:8,d} records")

        elif suite_name == "fo":
            for scale, scale_data in suite_data.get("results", {}).items():
                n_trials = scale_data.get("n_trials", 0)
                click.echo(f"  • {scale:10s}: {n_trials:6d} trials")


@benchmark_cli.command()
@click.argument("baseline_file", type=click.Path(exists=True))
@click.argument("current_file", type=click.Path(exists=True))
@click.option("--threshold", "-t", type=float, default=0.1,
              help="Regression threshold (0.1 = 10% slower)")
def compare(baseline_file: str, current_file: str, threshold: float):
    """Compare two benchmark results for regressions."""
    baseline = json.loads(Path(baseline_file).read_text())
    current = json.loads(Path(current_file).read_text())

    click.echo("=" * 80)
    click.echo("BENCHMARK COMPARISON")
    click.echo("=" * 80)
    click.echo(f"Baseline: {baseline_file}")
    click.echo(f"Current:  {current_file}")
    click.echo(f"Threshold: {threshold * 100:.0f}% regression")

    regressions = []
    improvements = []

    for suite_name in baseline.get("benchmarks", {}).keys():
        if suite_name not in current.get("benchmarks", {}):
            continue

        baseline_data = baseline["benchmarks"][suite_name]
        current_data = current["benchmarks"][suite_name]

        if "error" in baseline_data or "error" in current_data:
            continue

        baseline_time = baseline_data.get("total_time_s", 0)
        current_time = current_data.get("total_time_s", 0)

        if baseline_time == 0:
            continue

        ratio = (current_time - baseline_time) / baseline_time

        if ratio > threshold:
            regressions.append((suite_name, baseline_time, current_time, ratio))
        elif ratio < -threshold:
            improvements.append((suite_name, baseline_time, current_time, ratio))

    if regressions:
        click.echo(f"\n⚠ {len(regressions)} REGRESSION(S) DETECTED:")
        click.echo("-" * 80)
        for suite, baseline_t, current_t, ratio in regressions:
            click.echo(
                f"  {suite:4s}: {baseline_t:6.2f}s → {current_t:6.2f}s "
                f"({ratio*100:+.1f}%)"
            )
    else:
        click.echo("\n✓ No regressions detected")

    if improvements:
        click.echo(f"\n✓ {len(improvements)} IMPROVEMENT(S):")
        click.echo("-" * 80)
        for suite, baseline_t, current_t, ratio in improvements:
            click.echo(
                f"  {suite:4s}: {baseline_t:6.2f}s → {current_t:6.2f}s "
                f"({ratio*100:+.1f}%)"
            )


@benchmark_cli.command()
def list_suites():
    """List available benchmark suites."""
    click.echo("Available Benchmark Suites:")
    click.echo("=" * 80)

    suites = [
        ("qe", "Quant Evaluator", "Metric computation at scale"),
        ("fp", "Factor Preprocessing", "Data transformation operations"),
        ("fa", "Factor Analysis", "Registry and operator queries"),
        ("fo", "Factor Optimization", "Search and mutation operations"),
    ]

    for code, name, desc in suites:
        click.echo(f"{code:4s} - {name:25s} - {desc}")


if __name__ == "__main__":
    benchmark_cli()
