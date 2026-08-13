#!/usr/bin/env python3
"""
fo_cli: Command-line interface for Factor Optimization.

Run factor search campaigns and track mutations.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
import numpy as np

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@click.group()
@click.version_option(version="0.1.0")
def fo_cli():
    """Factor Optimization CLI - Run factor search campaigns."""
    pass


@fo_cli.command()
@click.option("--n-trials", "-n", type=int, default=100, help="Number of trials to run")
@click.option("--budget", "-b", type=int, default=1000, help="Evaluation budget")
@click.option("--output", "-o", type=click.Path(), help="Output file for results")
@click.option("--seed", "-s", type=int, default=42, help="Random seed")
def search(n_trials: int, budget: int, output: str | None, seed: int):
    """Run a factor search campaign (simulation mode).

    This simulates a factor search process with random mutations and evaluations.
    """
    rng = np.random.default_rng(seed)

    click.echo(f"Starting factor search campaign...")
    click.echo(f"  Trials:  {n_trials}")
    click.echo(f"  Budget:  {budget}")
    click.echo(f"  Seed:    {seed}")

    results = []
    seen_cache = set()
    evaluations_used = 0

    base_factor_id = "base_factor_001"
    seen_cache.add(base_factor_id)

    click.echo("\nRunning search...")
    start_time = time.perf_counter()

    for trial_idx in range(n_trials):
        if evaluations_used >= budget:
            click.echo(f"Budget exhausted at trial {trial_idx}")
            break

        # Simulate mutation
        mutation_type = rng.choice(["param_tweak", "operator_swap", "window_adjust", "combine"])
        mutated_id = f"mutated_{trial_idx:05d}_{mutation_type}"

        # Check deduplication
        if mutated_id in seen_cache:
            results.append({
                "trial": trial_idx,
                "factor_id": mutated_id,
                "status": "duplicate",
                "metric": None,
            })
            continue

        seen_cache.add(mutated_id)

        # Simulate evaluation
        evaluations_used += 1
        simulated_ic = rng.normal(0.02, 0.05)
        simulated_ir = rng.normal(0.8, 0.3)

        results.append({
            "trial": trial_idx,
            "factor_id": mutated_id,
            "mutation_type": mutation_type,
            "status": "evaluated",
            "ic": round(float(simulated_ic), 6),
            "ir": round(float(simulated_ir), 6),
        })

        if (trial_idx + 1) % 10 == 0:
            click.echo(f"  Completed {trial_idx + 1}/{n_trials} trials, {evaluations_used} evaluations")

    elapsed = time.perf_counter() - start_time

    # Summary
    evaluated = [r for r in results if r["status"] == "evaluated"]
    duplicates = [r for r in results if r["status"] == "duplicate"]

    click.echo(f"\nSearch completed in {elapsed:.2f}s")
    click.echo(f"  Total trials:    {len(results)}")
    click.echo(f"  Evaluated:       {len(evaluated)}")
    click.echo(f"  Duplicates:      {len(duplicates)}")
    click.echo(f"  Dedup rate:      {len(duplicates) / len(results) * 100:.1f}%")
    click.echo(f"  Evaluations:     {evaluations_used}/{budget}")

    if evaluated:
        ics = [r["ic"] for r in evaluated]
        click.echo(f"  Best IC:         {max(ics):.6f}")
        click.echo(f"  Mean IC:         {np.mean(ics):.6f}")

    # Save results
    output_data = {
        "config": {
            "n_trials": n_trials,
            "budget": budget,
            "seed": seed,
        },
        "summary": {
            "total_trials": len(results),
            "evaluated": len(evaluated),
            "duplicates": len(duplicates),
            "elapsed_s": round(elapsed, 3),
        },
        "results": results,
    }

    if output:
        Path(output).write_text(json.dumps(output_data, indent=2))
        click.echo(f"\nResults saved to {output}")
    else:
        click.echo("\nTop 5 factors by IC:")
        top_5 = sorted(evaluated, key=lambda x: x["ic"], reverse=True)[:5]
        for i, r in enumerate(top_5, 1):
            click.echo(f"  {i}. {r['factor_id']:30s} IC={r['ic']:8.6f} IR={r['ir']:6.3f}")


@fo_cli.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option("--top-n", "-n", type=int, default=10, help="Number of top factors to show")
@click.option("--metric", "-m", type=click.Choice(["ic", "ir"]), default="ic",
              help="Metric to rank by")
def analyze(results_file: str, top_n: int, metric: str):
    """Analyze results from a search campaign."""
    results_path = Path(results_file)
    data = json.loads(results_path.read_text())

    click.echo(f"Analyzing results from {results_file}")
    click.echo("=" * 80)

    summary = data.get("summary", {})
    click.echo(f"Total trials:    {summary.get('total_trials', 0)}")
    click.echo(f"Evaluated:       {summary.get('evaluated', 0)}")
    click.echo(f"Duplicates:      {summary.get('duplicates', 0)}")
    click.echo(f"Elapsed:         {summary.get('elapsed_s', 0):.2f}s")

    results = data.get("results", [])
    evaluated = [r for r in results if r.get("status") == "evaluated" and metric in r]

    if not evaluated:
        click.echo(f"\nNo evaluated results found with metric '{metric}'")
        return

    # Statistics
    metric_values = [r[metric] for r in evaluated]
    click.echo(f"\n{metric.upper()} Statistics:")
    click.echo(f"  Mean:   {np.mean(metric_values):8.6f}")
    click.echo(f"  Median: {np.median(metric_values):8.6f}")
    click.echo(f"  Std:    {np.std(metric_values):8.6f}")
    click.echo(f"  Min:    {min(metric_values):8.6f}")
    click.echo(f"  Max:    {max(metric_values):8.6f}")

    # Top factors
    click.echo(f"\nTop {top_n} factors by {metric.upper()}:")
    click.echo("-" * 80)
    top_factors = sorted(evaluated, key=lambda x: x[metric], reverse=True)[:top_n]
    for i, r in enumerate(top_factors, 1):
        mutation = r.get("mutation_type", "unknown")
        ic_val = r.get("ic", 0)
        ir_val = r.get("ir", 0)
        click.echo(
            f"{i:2d}. {r['factor_id']:30s} [{mutation:15s}] "
            f"IC={ic_val:8.6f} IR={ir_val:6.3f}"
        )

    # Mutation type analysis
    if "mutation_type" in evaluated[0]:
        mutation_stats = {}
        for r in evaluated:
            mut_type = r.get("mutation_type", "unknown")
            if mut_type not in mutation_stats:
                mutation_stats[mut_type] = []
            mutation_stats[mut_type].append(r[metric])

        click.echo(f"\nPerformance by Mutation Type:")
        click.echo("-" * 80)
        for mut_type, values in sorted(mutation_stats.items(), key=lambda x: np.mean(x[1]), reverse=True):
            click.echo(
                f"{mut_type:20s} n={len(values):4d} "
                f"mean={np.mean(values):8.6f} std={np.std(values):8.6f}"
            )


@fo_cli.command()
@click.option("--size", "-s", type=int, default=1000, help="Cache size to benchmark")
def benchmark(size: int):
    """Benchmark deduplication cache performance."""
    click.echo(f"Benchmarking deduplication cache with {size} items...")

    rng = np.random.default_rng(42)
    cache = set()

    # Generate unique items
    items = [f"factor_{i:06d}" for i in range(size)]

    # Benchmark insertions
    start = time.perf_counter()
    for item in items:
        cache.add(item)
    insert_time = time.perf_counter() - start

    # Benchmark lookups (hits)
    start = time.perf_counter()
    for item in items:
        _ = item in cache
    lookup_hit_time = time.perf_counter() - start

    # Benchmark lookups (misses)
    miss_items = [f"missing_{i:06d}" for i in range(size)]
    start = time.perf_counter()
    for item in miss_items:
        _ = item in cache
    lookup_miss_time = time.perf_counter() - start

    click.echo(f"\nResults:")
    click.echo(f"  Cache size:        {len(cache):,d}")
    click.echo(f"  Insert time:       {insert_time*1000:.3f}ms ({size/insert_time:.0f} ops/s)")
    click.echo(f"  Lookup (hit):      {lookup_hit_time*1000:.3f}ms ({size/lookup_hit_time:.0f} ops/s)")
    click.echo(f"  Lookup (miss):     {lookup_miss_time*1000:.3f}ms ({size/lookup_miss_time:.0f} ops/s)")


@fo_cli.command()
def grammar():
    """Display available mutation operations."""
    click.echo("Factor Mutation Grammar:")
    click.echo("=" * 80)

    mutations = [
        ("param_tweak", "Adjust numeric parameters (window, threshold, etc.)"),
        ("operator_swap", "Replace operator with similar function"),
        ("window_adjust", "Modify rolling window size"),
        ("combine", "Combine two factors with binary operation"),
        ("transform", "Apply transformation (log, rank, zscore)"),
        ("filter", "Add filtering condition"),
    ]

    for name, desc in mutations:
        click.echo(f"{name:20s} - {desc}")


if __name__ == "__main__":
    fo_cli()
