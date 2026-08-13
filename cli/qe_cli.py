#!/usr/bin/env python3
"""
qe_cli: Command-line interface for Quant Evaluator.

Evaluate factors from the command line with various metric options.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@click.group()
@click.version_option(version="0.1.0")
def qe_cli():
    """Quant Evaluator CLI - Evaluate factor quality metrics."""
    pass


@qe_cli.command()
@click.argument("factor_file", type=click.Path(exists=True))
@click.argument("label_file", type=click.Path(exists=True))
@click.option("--metrics", "-m", multiple=True, default=["rank_ic"],
              help="Metrics to compute (rank_ic, ic, ir, turnover)")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--format", "-f", type=click.Choice(["json", "csv", "text"]), default="text",
              help="Output format")
def evaluate(factor_file: str, label_file: str, metrics: tuple[str, ...],
             output: str | None, format: str):
    """Evaluate factors against labels.

    FACTOR_FILE: Path to factor data (CSV/parquet)
    LABEL_FILE: Path to label data (CSV/parquet)
    """
    try:
        from quant_evaluator import FactorBatch, LabelBundle, EvaluationRequest
        from scipy.stats import spearmanr, pearsonr
    except ImportError as e:
        click.echo(f"Error: Missing dependency - {e}", err=True)
        click.echo("Install with: pip install scipy pandas", err=True)
        sys.exit(1)

    click.echo(f"Loading factor data from {factor_file}...")
    factor_path = Path(factor_file)
    if factor_path.suffix == ".parquet":
        factor_df = pd.read_parquet(factor_path)
    else:
        factor_df = pd.read_csv(factor_path, index_col=0, parse_dates=True)

    click.echo(f"Loading label data from {label_file}...")
    label_path = Path(label_file)
    if label_path.suffix == ".parquet":
        label_df = pd.read_parquet(label_path)
    else:
        label_df = pd.read_csv(label_path, index_col=0, parse_dates=True)

    # Align indices
    common_idx = factor_df.index.intersection(label_df.index)
    common_cols = factor_df.columns.intersection(label_df.columns)

    if len(common_idx) == 0:
        click.echo("Error: No overlapping dates between factor and label", err=True)
        sys.exit(1)

    if len(common_cols) == 0:
        click.echo("Error: No overlapping assets between factor and label", err=True)
        sys.exit(1)

    factor_aligned = factor_df.loc[common_idx, common_cols]
    label_aligned = label_df.loc[common_idx, common_cols]

    click.echo(f"Aligned data shape: {factor_aligned.shape}")
    click.echo(f"Computing metrics: {', '.join(metrics)}")

    results = {}

    for metric_name in metrics:
        if metric_name == "rank_ic":
            # Compute rank IC
            f_flat = factor_aligned.values.flatten()
            l_flat = label_aligned.values.flatten()
            mask = ~(np.isnan(f_flat) | np.isnan(l_flat))
            if mask.sum() > 10:
                ic, pval = spearmanr(f_flat[mask], l_flat[mask])
                results["rank_ic"] = round(float(ic), 6)
                results["rank_ic_pval"] = round(float(pval), 6)
            else:
                results["rank_ic"] = None
                results["rank_ic_pval"] = None

        elif metric_name == "ic":
            # Compute Pearson IC
            f_flat = factor_aligned.values.flatten()
            l_flat = label_aligned.values.flatten()
            mask = ~(np.isnan(f_flat) | np.isnan(l_flat))
            if mask.sum() > 10:
                ic, pval = pearsonr(f_flat[mask], l_flat[mask])
                results["ic"] = round(float(ic), 6)
                results["ic_pval"] = round(float(pval), 6)
            else:
                results["ic"] = None
                results["ic_pval"] = None

        elif metric_name == "ir":
            # Compute IC IR (time-series IC mean/std)
            ics = []
            for date in factor_aligned.index:
                f_row = factor_aligned.loc[date].values
                l_row = label_aligned.loc[date].values
                mask = ~(np.isnan(f_row) | np.isnan(l_row))
                if mask.sum() > 10:
                    ic, _ = spearmanr(f_row[mask], l_row[mask])
                    ics.append(ic)

            if len(ics) > 1:
                ic_mean = np.mean(ics)
                ic_std = np.std(ics, ddof=1)
                results["ic_mean"] = round(float(ic_mean), 6)
                results["ic_std"] = round(float(ic_std), 6)
                results["ir"] = round(float(ic_mean / ic_std) if ic_std > 0 else 0.0, 6)
            else:
                results["ic_mean"] = None
                results["ic_std"] = None
                results["ir"] = None

        elif metric_name == "turnover":
            # Compute average turnover (change in factor ranks)
            turnover_vals = []
            for i in range(1, len(factor_aligned)):
                prev_rank = factor_aligned.iloc[i-1].rank()
                curr_rank = factor_aligned.iloc[i].rank()
                mask = ~(np.isnan(prev_rank) | np.isnan(curr_rank))
                if mask.sum() > 10:
                    turnover = (prev_rank[mask] != curr_rank[mask]).sum() / mask.sum()
                    turnover_vals.append(turnover)

            if turnover_vals:
                results["turnover_mean"] = round(float(np.mean(turnover_vals)), 6)
                results["turnover_std"] = round(float(np.std(turnover_vals, ddof=1)), 6)
            else:
                results["turnover_mean"] = None
                results["turnover_std"] = None

    # Format output
    if format == "json":
        import json
        output_text = json.dumps(results, indent=2)
    elif format == "csv":
        output_text = "metric,value\n"
        for k, v in results.items():
            output_text += f"{k},{v}\n"
    else:  # text
        output_text = "Evaluation Results:\n"
        output_text += "=" * 50 + "\n"
        for k, v in results.items():
            output_text += f"  {k:20s}: {v}\n"

    if output:
        Path(output).write_text(output_text)
        click.echo(f"Results written to {output}")
    else:
        click.echo(output_text)


@qe_cli.command()
@click.argument("factor_file", type=click.Path(exists=True))
def info(factor_file: str):
    """Display information about a factor file."""
    factor_path = Path(factor_file)

    if factor_path.suffix == ".parquet":
        df = pd.read_parquet(factor_path)
    else:
        df = pd.read_csv(factor_path, index_col=0, parse_dates=True)

    click.echo(f"Factor File: {factor_file}")
    click.echo(f"Shape: {df.shape[0]} dates × {df.shape[1]} assets")
    click.echo(f"Date range: {df.index.min()} to {df.index.max()}")
    click.echo(f"Coverage: {(~df.isna()).sum().sum() / (df.shape[0] * df.shape[1]) * 100:.2f}%")
    click.echo(f"Stats:")
    click.echo(f"  Mean: {df.mean().mean():.6f}")
    click.echo(f"  Std:  {df.std().mean():.6f}")
    click.echo(f"  Min:  {df.min().min():.6f}")
    click.echo(f"  Max:  {df.max().max():.6f}")


if __name__ == "__main__":
    qe_cli()
