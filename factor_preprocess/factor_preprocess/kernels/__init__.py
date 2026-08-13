"""Kernels package - reference bridge for fast implementations."""

from factor_preprocess.kernels.reference_bridge import (
    reference_cs_rank,
    reference_cs_zscore,
    reference_cs_demean,
    reference_rolling_mean,
    reference_rolling_std,
)

from factor_preprocess.kernels.fast import (
    fast_cs_rank,
    fast_cs_zscore,
    fast_cs_demean,
    fast_rolling_mean,
    fast_rolling_std,
    numba_rolling_mean,
    numba_rolling_std,
    get_capabilities,
)

__all__ = [
    # Reference implementations
    "reference_cs_rank",
    "reference_cs_zscore",
    "reference_cs_demean",
    "reference_rolling_mean",
    "reference_rolling_std",
    # Fast implementations
    "fast_cs_rank",
    "fast_cs_zscore",
    "fast_cs_demean",
    "fast_rolling_mean",
    "fast_rolling_std",
    "numba_rolling_mean",
    "numba_rolling_std",
    # Utilities
    "get_capabilities",
]
