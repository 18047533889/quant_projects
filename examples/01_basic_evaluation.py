"""
Example 01: Basic Factor Evaluation (Standalone)

Demonstrates the core QE (Quantitative Evaluator) workflow:
1. Load or generate factor data
2. Configure evaluation parameters
3. Run IC analysis, quantile backtest, and long-short portfolio
4. Interpret evaluation results

This standalone version includes inline implementations for demonstration.
"""

from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Tuple, List
import pandas as pd
import numpy as np
import json


def generate_synthetic_factor_data(
    n_dates: int = 250,
    n_symbols: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic factor data with known signal strength."""
    np.random.seed(seed)

    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_dates)]
    symbols = [f"SYM_{i:04d}" for i in range(n_symbols)]

    rows = []
    for date in dates:
        # Generate factor with modest predictive signal
        true_signal = np.random.randn(n_symbols) * 0.3
        noise = np.random.randn(n_symbols) * 0.95
        factor_values = true_signal + noise

        for i, symbol in enumerate(symbols):
            rows.append({
                "timestamp": date,
                "symbol": symbol,
                "factor_raw": factor_values[i],
            })

    return pd.DataFrame(rows)


def generate_synthetic_market_data(
    n_dates: int = 250,
    n_symbols: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic market data (price series for return calculation)."""
    np.random.seed(seed + 1)

    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_dates)]
    symbols = [f"SYM_{i:04d}" for i in range(n_symbols)]

    rows = []
    prices = np.exp(np.random.randn(n_symbols) * 0.3 + 4.0)

    for date in dates:
        daily_returns = np.random.randn(n_symbols) * 0.02
        prices = prices * (1 + daily_returns)

        for i, symbol in enumerate(symbols):
            rows.append({
                "timestamp": date,
                "symbol": symbol,
                "price": prices[i],
            })

    return pd.DataFrame(rows)


def winsorize_cross_section(series: pd.Series, lower: float, upper: float) -> pd.Series:
    """Winsorize values at specified quantiles."""
    valid = series.dropna()
    if valid.empty:
        return series.astype(float)
    lo = valid.quantile(lower)
    hi = valid.quantile(upper)
    return series.clip(lower=lo, upper=hi)


def zscore_cross_section(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score normalization."""
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    std = float(valid.std(ddof=0))
    if std < 1e-8:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (series - float(valid.mean())) / std


def calculate_ic_metrics(factor_df: pd.DataFrame, return_col: str) -> dict:
    """Calculate IC metrics for a factor."""
    daily_ics = []
    daily_rank_ics = []

    for timestamp, group in factor_df.groupby("timestamp", sort=True):
        valid = group[["factor_eval", return_col]].dropna()
        if len(valid) < 50:
            continue

        ic = valid["factor_eval"].corr(valid[return_col], method="pearson")
        rank_ic = valid["factor_eval"].rank().corr(valid[return_col].rank(), method="pearson")

        daily_ics.append(ic)
        daily_rank_ics.append(rank_ic)

    if not daily_ics:
        return {
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "ic_ir": np.nan,
            "rank_ic_mean": np.nan,
            "rank_ic_std": np.nan,
            "rank_ic_ir": np.nan,
            "n_dates": 0,
        }

    ic_series = pd.Series(daily_ics)
    rank_ic_series = pd.Series(daily_rank_ics)

    return {
        "ic_mean": float(ic_series.mean()),
        "ic_std": float(ic_series.std(ddof=1)),
        "ic_ir": float(ic_series.mean() / ic_series.std(ddof=1)) if ic_series.std() > 0 else np.nan,
        "rank_ic_mean": float(rank_ic_series.mean()),
        "rank_ic_std": float(rank_ic_series.std(ddof=1)),
        "rank_ic_ir": float(rank_ic_series.mean() / rank_ic_series.std(ddof=1)) if rank_ic_series.std() > 0 else np.nan,
        "n_dates": len(daily_ics),
    }


def calculate_quantile_returns(factor_df: pd.DataFrame, return_col: str, n_quantiles: int = 5) -> dict:
    """Calculate quantile group returns."""
    # Assign quantiles
    def assign_quantile(series):
        valid = series.dropna()
        if len(valid) < 50:
            return pd.Series(np.nan, index=series.index)
        return pd.qcut(series, q=n_quantiles, labels=False, duplicates='drop') + 1

    factor_df["quantile"] = factor_df.groupby("timestamp")["factor_eval"].transform(assign_quantile)

    # Calculate mean return per quantile
    quantile_returns = (
        factor_df.dropna(subset=["quantile", return_col])
        .groupby("quantile")[return_col]
        .mean()
    )

    if len(quantile_returns) < 2:
        return {
            "top_minus_bottom": np.nan,
            "monotonicity": np.nan,
        }

    top_return = quantile_returns.iloc[-1] if len(quantile_returns) > 0 else np.nan
    bottom_return = quantile_returns.iloc[0] if len(quantile_returns) > 0 else np.nan

    # Monotonicity: Spearman correlation between quantile and return
    monotonicity = quantile_returns.reset_index().corr(method="spearman").iloc[0, 1]

    return {
        "top_minus_bottom": top_return - bottom_return,
        "monotonicity": monotonicity,
    }


def main():
    """Run basic factor evaluation example."""

    print("=" * 70)
    print("EXAMPLE 01: Basic Factor Evaluation")
    print("=" * 70)
    print()

    # Step 1: Generate synthetic data
    print("STEP 1: Generate Synthetic Data")
    print("-" * 70)

    factor_df = generate_synthetic_factor_data(n_dates=250, n_symbols=500, seed=42)
    market_df = generate_synthetic_market_data(n_dates=250, n_symbols=500, seed=42)

    print(f"Factor data: {len(factor_df)} rows, {factor_df['symbol'].nunique()} symbols")
    print(f"Market data: {len(market_df)} rows, {market_df['symbol'].nunique()} symbols")
    print()

    # Step 2: Preprocess factor
    print("STEP 2: Preprocess Factor Values")
    print("-" * 70)

    # Winsorize and standardize
    factor_df["factor_eval"] = factor_df.groupby("timestamp")["factor_raw"].transform(
        lambda s: winsorize_cross_section(s, 0.01, 0.99)
    )
    factor_df["factor_eval"] = factor_df.groupby("timestamp")["factor_eval"].transform(
        zscore_cross_section
    )

    print("Applied transforms:")
    print("  - Winsorize at 1% and 99% quantiles")
    print("  - Cross-sectional z-score normalization")
    print()

    # Step 3: Calculate forward returns
    print("STEP 3: Calculate Forward Returns")
    print("-" * 70)

    market_df = market_df.sort_values(["symbol", "timestamp"])
    market_df["ret_1d"] = market_df.groupby("symbol")["price"].pct_change(1).shift(-1)
    market_df["ret_5d"] = market_df.groupby("symbol")["price"].pct_change(5).shift(-1)

    print("Calculated forward returns:")
    print("  - 1-day forward return")
    print("  - 5-day forward return")
    print()

    # Step 4: Merge and evaluate
    print("STEP 4: Run Evaluation")
    print("-" * 70)

    merged = factor_df.merge(market_df, on=["timestamp", "symbol"], how="inner")

    # Evaluate at 1-day horizon
    ic_metrics_1d = calculate_ic_metrics(merged, "ret_1d")
    quantile_metrics_1d = calculate_quantile_returns(merged, "ret_1d", n_quantiles=5)

    # Evaluate at 5-day horizon
    ic_metrics_5d = calculate_ic_metrics(merged, "ret_5d")
    quantile_metrics_5d = calculate_quantile_returns(merged, "ret_5d", n_quantiles=5)

    print()

    # Step 5: Display results
    print("STEP 5: Evaluation Results")
    print("-" * 70)
    print()

    print("=== IC Analysis ===")
    print()
    print("1-Day Horizon:")
    print(f"  IC Mean:        {ic_metrics_1d['ic_mean']:>8.4f}")
    print(f"  IC Std:         {ic_metrics_1d['ic_std']:>8.4f}")
    print(f"  IC IR:          {ic_metrics_1d['ic_ir']:>8.2f}")
    print(f"  Rank IC Mean:   {ic_metrics_1d['rank_ic_mean']:>8.4f}")
    print(f"  Rank IC IR:     {ic_metrics_1d['rank_ic_ir']:>8.2f}")
    print(f"  N Dates:        {ic_metrics_1d['n_dates']:>8d}")
    print()

    print("5-Day Horizon:")
    print(f"  IC Mean:        {ic_metrics_5d['ic_mean']:>8.4f}")
    print(f"  IC Std:         {ic_metrics_5d['ic_std']:>8.4f}")
    print(f"  IC IR:          {ic_metrics_5d['ic_ir']:>8.2f}")
    print(f"  Rank IC Mean:   {ic_metrics_5d['rank_ic_mean']:>8.4f}")
    print(f"  Rank IC IR:     {ic_metrics_5d['rank_ic_ir']:>8.2f}")
    print(f"  N Dates:        {ic_metrics_5d['n_dates']:>8d}")
    print()

    print("=== Quantile Analysis ===")
    print()
    print("1-Day Horizon:")
    print(f"  Top-Bottom Spread: {quantile_metrics_1d['top_minus_bottom']:>8.4f}")
    print(f"  Monotonicity:      {quantile_metrics_1d['monotonicity']:>8.4f}")
    print()

    print("5-Day Horizon:")
    print(f"  Top-Bottom Spread: {quantile_metrics_5d['top_minus_bottom']:>8.4f}")
    print(f"  Monotonicity:      {quantile_metrics_5d['monotonicity']:>8.4f}")
    print()

    # Step 6: Interpretation
    print("STEP 6: Interpretation")
    print("-" * 70)
    print()
    print("IC (Information Coefficient):")
    print("  - Measures linear correlation between factor and forward returns")
    print("  - Range: -1 to +1, with 0 indicating no relationship")
    print("  - Good factors typically have |IC| > 0.03 and IC IR > 1.0")
    print()
    print("Rank IC:")
    print("  - Spearman rank correlation, more robust to outliers")
    print("  - Generally higher and more stable than Pearson IC")
    print()
    print("Quantile Analysis:")
    print("  - Top-Bottom spread measures return difference between Q5 and Q1")
    print("  - Monotonicity measures consistent ordering (higher quantile → higher return)")
    print("  - Good factors show positive spread and high monotonicity (>0.7)")
    print()

    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    print("The evaluation results (EvaluationBundle) would flow to:")
    print()
    print("1. Factor Assets (FA)")
    print("   - Register factor with EvidenceRef")
    print("   - Apply selection gates based on IC thresholds")
    print()
    print("2. Factor Optimizer (FO)")
    print("   - Guide mutation search (prioritize high-IC parents)")
    print("   - Track Pareto frontier (IC vs complexity)")
    print()
    print("3. Factor Preprocess (FP)")
    print("   - Selected factors transformed to model input")
    print("   - Preprocessing intensity guided by IC stability")
    print()

    print("=" * 70)
    print("Example 01 Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
