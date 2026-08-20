"""
IC computation module for quant_evaluator.

Provides functions for computing Information Coefficient (IC) metrics.
"""

import numpy as np
from typing import Optional


def compute_ic_std(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Compute standard deviation of IC series.

    Args:
        ic_series: Array of IC values
        min_periods: Minimum number of periods required

    Returns:
        Standard deviation of IC values
    """
    if len(ic_series) < min_periods:
        return np.nan
    return np.nanstd(ic_series)


def compute_mean_ic_value(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Compute mean IC value.

    Args:
        ic_series: Array of IC values
        min_periods: Minimum number of periods required

    Returns:
        Mean IC value
    """
    if len(ic_series) < min_periods:
        return np.nan
    return np.nanmean(ic_series)
