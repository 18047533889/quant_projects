#!/usr/bin/env python3
"""
fp_cli: Command-line interface for Factor Preprocessing.

Preprocess factor data with various transformations.
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
def fp_cli():
    """Factor Preprocessing CLI - Transform and prepare factor data."""
    pass


@fp_cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--transform", "-t", multiple=True,
              help="Transforms to apply: zscore, rank, winsorize, fillna, neutralize")
@click.option("--winsorize-std", type=float, default=3.0,
              help="Standard deviations for winsorization")
@click.option("--fillna-method", type=click.Choice(["zero", "mean", "median", "forward"]),
              default="zero", help="Method for filling NaN values")
@click.option("--axis", type=click.Choice(["cs", "ts", "both"]), default="cs",
              help="Axis for transformation: cs (cross-sectional), ts (time-series), both")
def transform(input_file: str, output_file: str, transform: tuple[str, ...],
              winsorize_std: float, fillna_method: str, axis: str):
    """Apply transformations to factor data.

    INPUT_FILE: Input factor data (CSV/parquet)
    OUTPUT_FILE: Output file path
    """
    click.echo(f"Loading data from {input_file}...")
    input_path = Path(input_file)
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path, index_col=0, parse_dates=True)

    click.echo(f"Input shape: {df.shape}")
    click.echo(f"Applying transforms: {', '.join(transform) if transform else 'none'}")

    result = df.copy()

    for trans_name in transform:
        if trans_name == "zscore":
            if axis == "cs":
                # Cross-sectional z-score
                result = result.sub(result.mean(axis=1), axis=0).div(result.std(axis=1), axis=0)
            elif axis == "ts":
                # Time-series z-score
                result = result.sub(result.mean(axis=0), axis=1).div(result.std(axis=0), axis=1)
            elif axis == "both":
                # Global z-score
                result = (result - result.mean().mean()) / result.std().mean()
            click.echo(f"  Applied z-score normalization (axis={axis})")

        elif trans_name == "rank":
            if axis == "cs":
                # Cross-sectional rank
                result = result.rank(axis=1, pct=True)
            elif axis == "ts":
                # Time-series rank
                result = result.rank(axis=0, pct=True)
            elif axis == "both":
                # Global rank
                result = result.stack().rank(pct=True).unstack()
            click.echo(f"  Applied rank transformation (axis={axis})")

        elif trans_name == "winsorize":
            if axis == "cs":
                # Cross-sectional winsorization
                mean = result.mean(axis=1)
                std = result.std(axis=1)
                lower = mean - winsorize_std * std
                upper = mean + winsorize_std * std
                for i, idx in enumerate(result.index):
                    result.loc[idx] = result.loc[idx].clip(lower.iloc[i], upper.iloc[i])
            elif axis == "ts":
                # Time-series winsorization
                mean = result.mean(axis=0)
                std = result.std(axis=0)
                lower = mean - winsorize_std * std
                upper = mean + winsorize_std * std
                result = result.clip(lower=lower, upper=upper, axis=1)
            elif axis == "both":
                # Global winsorization
                mean = result.mean().mean()
                std = result.std().mean()
                lower = mean - winsorize_std * std
                upper = mean + winsorize_std * std
                result = result.clip(lower, upper)
            click.echo(f"  Applied winsorization (std={winsorize_std}, axis={axis})")

        elif trans_name == "fillna":
            if fillna_method == "zero":
                result = result.fillna(0)
            elif fillna_method == "mean":
                if axis == "cs":
                    result = result.fillna(result.mean(axis=1), axis=0)
                elif axis == "ts":
                    result = result.fillna(result.mean(axis=0))
                else:
                    result = result.fillna(result.mean().mean())
            elif fillna_method == "median":
                if axis == "cs":
                    result = result.fillna(result.median(axis=1), axis=0)
                elif axis == "ts":
                    result = result.fillna(result.median(axis=0))
                else:
                    result = result.fillna(result.median().median())
            elif fillna_method == "forward":
                result = result.fillna(method="ffill")
            click.echo(f"  Filled NaN values (method={fillna_method})")

        elif trans_name == "neutralize":
            # Simple market neutralization (demean cross-sectionally)
            result = result.sub(result.mean(axis=1), axis=0)
            click.echo(f"  Applied market neutralization")

    # Save output
    output_path = Path(output_file)
    if output_path.suffix == ".parquet":
        result.to_parquet(output_path)
    else:
        result.to_csv(output_path)

    click.echo(f"Output shape: {result.shape}")
    click.echo(f"Results written to {output_file}")


@fp_cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--top-n", type=int, default=10, help="Number of top outliers to show")
def outliers(input_file: str, top_n: int):
    """Detect outliers in factor data."""
    input_path = Path(input_file)
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path, index_col=0, parse_dates=True)

    click.echo(f"Analyzing outliers in {input_file}...")
    click.echo(f"Data shape: {df.shape}")

    # Compute z-scores
    mean = df.mean().mean()
    std = df.std().mean()
    z_scores = np.abs((df - mean) / std)

    # Find top outliers
    outlier_values = []
    for date in df.index:
        for asset in df.columns:
            val = df.loc[date, asset]
            if not np.isnan(val):
                z = z_scores.loc[date, asset]
                if z > 3:
                    outlier_values.append({
                        "date": date,
                        "asset": asset,
                        "value": val,
                        "z_score": z
                    })

    outlier_values.sort(key=lambda x: x["z_score"], reverse=True)

    click.echo(f"\nTop {top_n} outliers (|z-score| > 3):")
    click.echo("-" * 70)
    for i, outlier in enumerate(outlier_values[:top_n], 1):
        click.echo(
            f"{i:2d}. {outlier['date']} {outlier['asset']:10s} "
            f"value={outlier['value']:12.6f} z={outlier['z_score']:.2f}"
        )

    click.echo(f"\nTotal outliers (|z-score| > 3): {len(outlier_values)}")


@fp_cli.command()
@click.argument("input_file", type=click.Path(exists=True))
def stats(input_file: str):
    """Display statistics about factor data."""
    input_path = Path(input_file)
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path, index_col=0, parse_dates=True)

    click.echo(f"Factor Data Statistics: {input_file}")
    click.echo("=" * 70)
    click.echo(f"Shape: {df.shape[0]} dates × {df.shape[1]} assets")
    click.echo(f"Date range: {df.index.min()} to {df.index.max()}")
    click.echo(f"Coverage: {(~df.isna()).sum().sum() / (df.shape[0] * df.shape[1]) * 100:.2f}%")
    click.echo(f"\nDescriptive Statistics:")
    click.echo(f"  Mean:   {df.mean().mean():12.6f}")
    click.echo(f"  Median: {df.median().median():12.6f}")
    click.echo(f"  Std:    {df.std().mean():12.6f}")
    click.echo(f"  Min:    {df.min().min():12.6f}")
    click.echo(f"  Max:    {df.max().max():12.6f}")
    click.echo(f"  Skew:   {df.apply(lambda x: x.skew()).mean():12.6f}")
    click.echo(f"  Kurt:   {df.apply(lambda x: x.kurtosis()).mean():12.6f}")

    # Check for issues
    click.echo(f"\nData Quality:")
    n_inf = np.isinf(df.values).sum()
    n_nan = df.isna().sum().sum()
    click.echo(f"  NaN values:      {n_nan:12,d} ({n_nan / df.size * 100:.2f}%)")
    click.echo(f"  Inf values:      {n_inf:12,d}")
    click.echo(f"  Zero values:     {(df == 0).sum().sum():12,d}")


if __name__ == "__main__":
    fp_cli()
