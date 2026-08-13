"""
Example 02: Factor Preprocessing Pipeline (Standalone)

Demonstrates the FP (Factor Preprocess) workflow:
1. Load raw factor values from selected factors
2. Apply cross-sectional transforms (rank, zscore, winsorize)
3. Apply time-series transforms (rolling operations)
4. Apply neutralization (OLS residualization against exposures)
5. Package results into a FeatureBundle for model consumption

This standalone version includes inline implementations for demonstration.
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass


def generate_raw_factors(n_dates: int, n_symbols: int, n_factors: int) -> Dict[str, pd.DataFrame]:
    """Generate synthetic raw factor data."""
    np.random.seed(123)

    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_dates)]
    symbols = [f"SYM_{i:04d}" for i in range(n_symbols)]

    factors = {}
    factor_names = ["momentum_20d", "value_pb", "volatility_20d", "quality_roe", "liquidity_turnover"]

    for factor_name in factor_names[:n_factors]:
        rows = []
        for date in dates:
            values = np.random.randn(n_symbols) * (0.5 + np.random.rand())
            for i, symbol in enumerate(symbols):
                rows.append({"timestamp": date, "symbol": symbol, "value": values[i]})
        factors[factor_name] = pd.DataFrame(rows)

    return factors


def generate_exposure_data(n_dates: int, n_symbols: int) -> Dict[str, pd.DataFrame]:
    """Generate exposure data for neutralization."""
    np.random.seed(456)

    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_dates)]
    symbols = [f"SYM_{i:04d}" for i in range(n_symbols)]

    exposures = {}

    # Industry (one-hot, 5 industries for simplicity)
    industry_assignments = np.random.randint(0, 5, size=n_symbols)
    for ind_idx in range(5):
        rows = []
        for date in dates:
            for i, symbol in enumerate(symbols):
                rows.append({
                    "timestamp": date,
                    "symbol": symbol,
                    "value": 1.0 if industry_assignments[i] == ind_idx else 0.0,
                })
        exposures[f"industry_{ind_idx}"] = pd.DataFrame(rows)

    # Market cap
    market_caps = np.random.lognormal(mean=10, sigma=1.5, size=n_symbols)
    rows = []
    for date in dates:
        for i, symbol in enumerate(symbols):
            rows.append({
                "timestamp": date,
                "symbol": symbol,
                "value": np.log(market_caps[i]),
            })
    exposures["log_market_cap"] = pd.DataFrame(rows)

    return exposures


def apply_winsorize(df: pd.DataFrame, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    """Apply winsorization by date."""
    df = df.copy()

    def winsorize_group(group):
        group = group.copy()
        valid = group["value"].dropna()
        if len(valid) > 10:
            lo = valid.quantile(lower_q)
            hi = valid.quantile(upper_q)
            group["value"] = group["value"].clip(lower=lo, upper=hi)
        return group

    return df.groupby("timestamp", group_keys=False).apply(winsorize_group)


def apply_cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cross-sectional z-score by date."""
    df = df.copy()

    def zscore_group(group):
        group = group.copy()
        valid = group["value"].dropna()
        if len(valid) > 10 and valid.std() > 1e-8:
            group["value"] = (group["value"] - valid.mean()) / valid.std()
        else:
            group["value"] = np.nan
        return group

    return df.groupby("timestamp", group_keys=False).apply(zscore_group)


def apply_cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cross-sectional ranking by date."""
    df = df.copy()

    def rank_group(group):
        group = group.copy()
        group["value"] = group["value"].rank(pct=True, method="average")
        return group

    return df.groupby("timestamp", group_keys=False).apply(rank_group)


def apply_rolling_zscore(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Apply rolling z-score per symbol."""
    df = df.copy().sort_values(["symbol", "timestamp"])

    def rolling_zscore_symbol(group):
        group = group.copy()
        rolling_mean = group["value"].rolling(window=window, min_periods=window//2).mean()
        rolling_std = group["value"].rolling(window=window, min_periods=window//2).std()
        group["value"] = (group["value"] - rolling_mean) / (rolling_std + 1e-8)
        return group

    return df.groupby("symbol", group_keys=False).apply(rolling_zscore_symbol)


def apply_ols_neutralization(df: pd.DataFrame, exposures: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Apply OLS neutralization against exposures."""
    df = df.copy()

    # Merge all exposures
    merged = df.copy()
    for exp_name, exp_df in exposures.items():
        merged = merged.merge(
            exp_df.rename(columns={"value": exp_name}),
            on=["timestamp", "symbol"],
            how="left"
        )

    exposure_cols = list(exposures.keys())

    # Neutralize per date
    def neutralize_date(group):
        group = group.copy()
        y = group["value"].values
        X = group[exposure_cols].values

        # Filter valid rows
        valid_mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        if valid_mask.sum() < len(exposure_cols) + 10:
            return group

        y_valid = y[valid_mask]
        X_valid = X[valid_mask]

        # OLS regression
        try:
            beta = np.linalg.lstsq(X_valid, y_valid, rcond=None)[0]
            y_pred = X @ beta
            residuals = y - y_pred
            group["value"] = residuals
        except:
            pass

        return group

    return merged.groupby("timestamp", group_keys=False).apply(neutralize_date)[["timestamp", "symbol", "value"]]


@dataclass
class FeatureBundle:
    """Container for transformed features."""
    bundle_id: str
    feature_matrix: np.ndarray  # Shape: (n_dates, n_symbols, n_features)
    factor_ids: List[str]
    dates: List[datetime]
    symbols: List[str]
    transforms_applied: List[str]
    created_at: datetime


def main():
    """Run preprocessing pipeline example."""

    print("=" * 70)
    print("EXAMPLE 02: Factor Preprocessing Pipeline")
    print("=" * 70)
    print()

    # Parameters
    N_DATES = 250
    N_SYMBOLS = 500
    N_FACTORS = 3

    # Step 1: Load raw factors
    print("STEP 1: Load Raw Factor Values")
    print("-" * 70)

    raw_factors = generate_raw_factors(N_DATES, N_SYMBOLS, N_FACTORS)

    print(f"Loaded {len(raw_factors)} factors:")
    for factor_id, df in raw_factors.items():
        print(f"  - {factor_id}: {len(df)} rows")
    print()

    # Step 2: Cross-sectional transforms
    print("STEP 2: Apply Cross-Sectional Transforms")
    print("-" * 70)
    print()

    cs_transformed = {}

    for factor_id, df in raw_factors.items():
        print(f"Processing {factor_id}...")

        # Pipeline: Winsorize → Z-score → Rank
        df = apply_winsorize(df, lower_q=0.01, upper_q=0.99)
        print("  ✓ Winsorized at 1% and 99%")

        df = apply_cs_zscore(df)
        print("  ✓ Cross-sectional z-score")

        df = apply_cs_rank(df)
        print("  ✓ Cross-sectional rank")

        cs_transformed[factor_id] = df

    print()

    # Step 3: Time-series transforms
    print("STEP 3: Apply Time-Series Transforms")
    print("-" * 70)
    print()

    ts_transformed = {}

    for factor_id, df in cs_transformed.items():
        print(f"Processing {factor_id}...")

        # Rolling z-score
        df = apply_rolling_zscore(df, window=60)
        print("  ✓ Rolling z-score (60-day window)")

        ts_transformed[factor_id] = df

    print()

    # Step 4: Neutralization
    print("STEP 4: Apply OLS Neutralization")
    print("-" * 70)
    print()

    exposures = generate_exposure_data(N_DATES, N_SYMBOLS)
    print(f"Loaded {len(exposures)} exposures:")
    for exp_name in list(exposures.keys()):
        print(f"  - {exp_name}")
    print()

    neutralized = {}

    for factor_id, df in ts_transformed.items():
        print(f"Neutralizing {factor_id}...")
        df = apply_ols_neutralization(df, exposures)
        neutralized[factor_id] = df
        print(f"  ✓ Neutralized against {len(exposures)} exposures")

    print()

    # Step 5: Package into FeatureBundle
    print("STEP 5: Package into FeatureBundle")
    print("-" * 70)
    print()

    # Get unique dates and symbols
    sample_df = neutralized[list(neutralized.keys())[0]]
    dates = sorted(sample_df["timestamp"].unique())
    symbols = sorted(sample_df["symbol"].unique())

    # Build feature matrix
    n_dates = len(dates)
    n_symbols = len(symbols)
    n_features = len(neutralized)

    feature_matrix = np.full((n_dates, n_symbols, n_features), np.nan, dtype=np.float64)

    for feat_idx, (factor_id, df) in enumerate(neutralized.items()):
        df_pivot = df.pivot(index="timestamp", columns="symbol", values="value")
        df_pivot = df_pivot.reindex(index=dates, columns=symbols)
        feature_matrix[:, :, feat_idx] = df_pivot.values

    bundle = FeatureBundle(
        bundle_id="example_02_bundle",
        feature_matrix=feature_matrix,
        factor_ids=list(neutralized.keys()),
        dates=dates,
        symbols=symbols,
        transforms_applied=[
            "winsorize_0.01_0.99",
            "cs_zscore",
            "cs_rank",
            "rolling_zscore_60d",
            "ols_neutralize_industry_mktcap",
        ],
        created_at=datetime.now(),
    )

    print(f"Bundle ID: {bundle.bundle_id}")
    print(f"Shape: {bundle.feature_matrix.shape} (dates × symbols × features)")
    print(f"Features: {bundle.factor_ids}")
    print()

    # Step 6: Validate and summarize
    print("STEP 6: Validate and Summarize")
    print("-" * 70)
    print()

    missing_rate = np.isnan(feature_matrix).mean()
    print(f"Overall missing rate: {missing_rate:.2%}")
    print()

    print("Per-feature statistics:")
    for feat_idx, factor_id in enumerate(bundle.factor_ids):
        feat_values = feature_matrix[:, :, feat_idx]
        valid = feat_values[~np.isnan(feat_values)]
        print(f"  {factor_id}:")
        print(f"    Mean:    {valid.mean():>8.4f}")
        print(f"    Std:     {valid.std():>8.4f}")
        print(f"    Missing: {np.isnan(feat_values).mean():>7.2%}")
    print()

    # Step 7: Understanding the pipeline
    print("STEP 7: Understanding the Preprocessing Pipeline")
    print("-" * 70)
    print()
    print("Transform sequence and rationale:")
    print()
    print("1. Winsorization")
    print("   - Clips extreme outliers that could distort statistics")
    print("   - Preserves relative ordering while reducing tail impact")
    print()
    print("2. Cross-Sectional Z-Score")
    print("   - Standardizes factors within each date")
    print("   - Ensures mean=0, std=1 across assets at each point in time")
    print("   - Makes factors comparable across different time periods")
    print()
    print("3. Cross-Sectional Rank")
    print("   - Converts to percentile ranks [0, 1]")
    print("   - Highly robust to outliers and non-linear relationships")
    print("   - Enforces uniform distribution")
    print()
    print("4. Rolling Z-Score (Time-Series)")
    print("   - Removes slow-moving trends per symbol")
    print("   - Highlights deviations from recent history")
    print("   - Window size trades off responsiveness vs stability")
    print()
    print("5. OLS Neutralization")
    print("   - Removes systematic factor exposures (industry, size)")
    print("   - Residuals represent pure alpha orthogonal to known risks")
    print("   - Essential for isolating idiosyncratic signal")
    print()
    print("The FeatureBundle is now ready for:")
    print("  - Model training (linear, tree, neural networks)")
    print("  - Backtesting and simulation")
    print("  - Production inference pipelines")
    print()

    print("=" * 70)
    print("Example 02 Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
