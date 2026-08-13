"""
Synthetic data generation for integration testing.

Generates realistic factor, label, and exposure data with controlled statistical properties.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
from datetime import datetime, timedelta


@dataclass
class PanelConfig:
    """Configuration for synthetic panel data generation."""

    num_times: int = 252
    num_assets: int = 100
    num_factors: int = 5
    start_date: str = "2024-01-01"
    freq: str = "D"
    seed: Optional[int] = None


@dataclass
class DataCharacteristics:
    """Statistical characteristics for generated data."""

    # Factor properties
    factor_mean: float = 0.0
    factor_std: float = 1.0
    factor_autocorr: float = 0.1
    cross_sectional_correlation: float = 0.3
    missing_rate: float = 0.05

    # Label properties
    signal_strength: float = 0.05  # IC level
    label_noise_ratio: float = 10.0
    horizon: int = 1
    execution_delay: int = 0

    # Exposure properties
    exposure_rank: int = 3  # Number of true risk factors
    exposure_loadings_std: float = 0.5


def generate_factor_panel(
    config: PanelConfig,
    characteristics: Optional[DataCharacteristics] = None,
) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """
    Generate synthetic factor panel with realistic properties.

    Returns:
        values: (T, N, F) array of factor values
        time_index: DatetimeIndex of length T
        asset_ids: array of asset identifiers of length N
    """
    if characteristics is None:
        characteristics = DataCharacteristics()

    if config.seed is not None:
        np.random.seed(config.seed)

    T, N, F = config.num_times, config.num_assets, config.num_factors

    # Generate time index
    time_index = pd.date_range(
        start=config.start_date,
        periods=T,
        freq=config.freq,
    )

    # Generate asset IDs
    asset_ids = np.arange(1000, 1000 + N, dtype=np.int64)

    # Generate base factors with autocorrelation
    values = np.zeros((T, N, F))

    for f in range(F):
        # Initial values
        values[0, :, f] = np.random.randn(N) * characteristics.factor_std

        # Add temporal autocorrelation
        for t in range(1, T):
            innovation = np.random.randn(N) * characteristics.factor_std
            values[t, :, f] = (
                characteristics.factor_autocorr * values[t-1, :, f] +
                np.sqrt(1 - characteristics.factor_autocorr**2) * innovation
            )

    # Add cross-sectional correlation structure
    if characteristics.cross_sectional_correlation > 0:
        # Generate common factor
        common_factor = np.random.randn(T, N) * characteristics.factor_std
        for f in range(F):
            weight = characteristics.cross_sectional_correlation
            values[:, :, f] = (
                np.sqrt(1 - weight**2) * values[:, :, f] +
                weight * common_factor
            )

    # Add missing values
    if characteristics.missing_rate > 0:
        mask = np.random.rand(T, N, F) < characteristics.missing_rate
        values[mask] = np.nan

    # Standardize to target mean and std
    for f in range(F):
        factor_slice = values[:, :, f]
        valid_mask = ~np.isnan(factor_slice)
        if valid_mask.any():
            current_mean = np.nanmean(factor_slice)
            current_std = np.nanstd(factor_slice)
            if current_std > 0:
                factor_slice[valid_mask] = (
                    (factor_slice[valid_mask] - current_mean) / current_std *
                    characteristics.factor_std + characteristics.factor_mean
                )
                values[:, :, f] = factor_slice

    return values, time_index, asset_ids


def generate_label_bundle(
    factor_values: np.ndarray,
    time_index: pd.DatetimeIndex,
    characteristics: Optional[DataCharacteristics] = None,
    true_weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Generate label bundle with controlled signal strength.

    Args:
        factor_values: (T, N, F) factor panel
        time_index: DatetimeIndex
        characteristics: Data characteristics
        true_weights: Optional (F,) array of true factor weights

    Returns:
        Dictionary with label_values, decision_time, label_start_time, label_end_time,
        true_weights, and metadata
    """
    if characteristics is None:
        characteristics = DataCharacteristics()

    T, N, F = factor_values.shape

    # Generate or use provided weights
    if true_weights is None:
        true_weights = np.random.randn(F)
        true_weights = true_weights / np.linalg.norm(true_weights)

    # Compute signal: weighted combination of factors
    signal = np.zeros((T, N))
    for f in range(F):
        signal += true_weights[f] * np.nan_to_num(factor_values[:, :, f], nan=0.0)

    # Scale signal to target IC
    signal_std = np.std(signal[signal != 0]) if np.any(signal != 0) else 1.0
    signal = signal / signal_std * characteristics.signal_strength

    # Add noise
    noise = np.random.randn(T, N) / characteristics.label_noise_ratio
    label_values = signal + noise

    # Generate timing information
    decision_time = tuple(time_index.strftime("%Y-%m-%d"))

    # Label start/end with horizon
    label_start_idx = np.clip(
        np.arange(T) + characteristics.execution_delay + 1,
        0, T-1
    )
    label_end_idx = np.clip(
        label_start_idx + characteristics.horizon,
        0, T-1
    )

    label_start_time = tuple(time_index[label_start_idx].strftime("%Y-%m-%d"))
    label_end_time = tuple(time_index[label_end_idx].strftime("%Y-%m-%d"))

    return {
        "values": label_values,
        "decision_time": decision_time,
        "label_start_time": label_start_time,
        "label_end_time": label_end_time,
        "true_weights": true_weights,
        "signal_strength": characteristics.signal_strength,
        "horizon": characteristics.horizon,
        "execution_delay": characteristics.execution_delay,
    }


def generate_exposure_matrix(
    num_times: int,
    num_assets: int,
    characteristics: Optional[DataCharacteristics] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate exposure matrix for risk model testing.

    Returns:
        exposures: (T, N, K) exposure matrix
        factor_names: List of factor names
    """
    if characteristics is None:
        characteristics = DataCharacteristics()

    if seed is not None:
        np.random.seed(seed)

    T, N = num_times, num_assets
    K = characteristics.exposure_rank

    # Generate factor names
    factor_names = [f"risk_factor_{i+1}" for i in range(K)]

    # Generate exposures with stable loadings
    exposures = np.zeros((T, N, K))

    # Base loadings (stable across time)
    base_loadings = np.random.randn(N, K) * characteristics.exposure_loadings_std

    # Add time variation
    for t in range(T):
        time_variation = np.random.randn(N, K) * 0.1
        exposures[t] = base_loadings + time_variation

    return exposures, factor_names


def generate_correlated_factors(
    num_times: int,
    num_assets: int,
    num_factors: int,
    correlation_matrix: np.ndarray,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate factors with specified correlation structure.

    Args:
        num_times: Number of time periods
        num_assets: Number of assets
        num_factors: Number of factors
        correlation_matrix: (F, F) target correlation matrix
        seed: Random seed

    Returns:
        (T, N, F) factor panel with target correlation
    """
    if seed is not None:
        np.random.seed(seed)

    T, N, F = num_times, num_assets, num_factors

    # Validate correlation matrix
    if correlation_matrix.shape != (F, F):
        raise ValueError(f"correlation_matrix must be ({F}, {F})")

    # Cholesky decomposition for correlation
    try:
        L = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        raise ValueError("correlation_matrix must be positive definite")

    # Generate independent factors
    independent = np.random.randn(T, N, F)

    # Apply correlation structure across factor dimension
    correlated = np.zeros((T, N, F))
    for t in range(T):
        for n in range(N):
            correlated[t, n, :] = L @ independent[t, n, :]

    return correlated


def generate_realistic_market_data(
    config: PanelConfig,
    include_fundamental: bool = True,
    include_technical: bool = True,
    include_alternative: bool = False,
) -> Dict[str, Any]:
    """
    Generate realistic multi-category market data.

    Returns comprehensive dataset with multiple factor categories.
    """
    if config.seed is not None:
        np.random.seed(config.seed)

    T, N = config.num_times, config.num_assets

    time_index = pd.date_range(
        start=config.start_date,
        periods=T,
        freq=config.freq,
    )
    asset_ids = np.arange(1000, 1000 + N, dtype=np.int64)

    factors = {}
    factor_metadata = {}

    # Fundamental factors
    if include_fundamental:
        # Value factors
        factors["book_to_market"] = generate_persistent_factor(T, N, persistence=0.9)
        factors["earnings_yield"] = generate_persistent_factor(T, N, persistence=0.8)

        factor_metadata.update({
            "book_to_market": {"category": "value", "timing": "quarterly"},
            "earnings_yield": {"category": "value", "timing": "quarterly"},
        })

    # Technical factors
    if include_technical:
        # Momentum
        factors["momentum_20d"] = generate_trending_factor(T, N, trend_strength=0.1)
        factors["momentum_60d"] = generate_trending_factor(T, N, trend_strength=0.15)

        # Volatility
        factors["volatility_20d"] = generate_volatility_factor(T, N)

        factor_metadata.update({
            "momentum_20d": {"category": "momentum", "timing": "daily"},
            "momentum_60d": {"category": "momentum", "timing": "daily"},
            "volatility_20d": {"category": "risk", "timing": "daily"},
        })

    # Alternative data
    if include_alternative:
        factors["sentiment_score"] = generate_noisy_signal(T, N, noise_level=2.0)
        factor_metadata["sentiment_score"] = {"category": "alternative", "timing": "daily"}

    # Stack factors
    factor_names = list(factors.keys())
    F = len(factor_names)
    factor_panel = np.stack([factors[name] for name in factor_names], axis=2)

    return {
        "factor_panel": factor_panel,
        "factor_names": factor_names,
        "time_index": time_index,
        "asset_ids": asset_ids,
        "metadata": factor_metadata,
    }


# Helper functions for realistic patterns

def generate_persistent_factor(T: int, N: int, persistence: float = 0.9) -> np.ndarray:
    """Generate highly persistent factor (e.g., fundamentals)."""
    values = np.zeros((T, N))
    values[0] = np.random.randn(N)

    for t in range(1, T):
        innovation = np.random.randn(N) * 0.1
        values[t] = persistence * values[t-1] + np.sqrt(1 - persistence**2) * innovation

    return values


def generate_trending_factor(T: int, N: int, trend_strength: float = 0.1) -> np.ndarray:
    """Generate factor with trending behavior (e.g., momentum)."""
    values = np.zeros((T, N))
    trends = np.random.randn(N) * trend_strength

    for t in range(T):
        values[t] = trends * t + np.random.randn(N) * 0.5

    return values


def generate_volatility_factor(T: int, N: int) -> np.ndarray:
    """Generate volatility-like factor with clustering."""
    values = np.zeros((T, N))
    vol_state = np.ones(N)

    for t in range(T):
        # Volatility clustering
        vol_state = 0.9 * vol_state + 0.1 * np.abs(np.random.randn(N))
        values[t] = vol_state * np.random.randn(N)

    return np.abs(values)


def generate_noisy_signal(T: int, N: int, noise_level: float = 1.0) -> np.ndarray:
    """Generate high-noise signal (e.g., alternative data)."""
    signal = np.random.randn(T, N) * 0.3
    noise = np.random.randn(T, N) * noise_level
    return signal + noise
