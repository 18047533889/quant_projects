"""
Preprocessing subpackage for modeling_adapters.

Provides stateless and fitted transform interfaces that can be implemented
directly or delegated to factor_preprocess via adapters.
"""
from modeling_adapters.preprocess.stateless import (
    rank_transform,
    zscore_transform,
    winsorize,
)
from modeling_adapters.preprocess.fitted import (
    FittedTransform,
    create_fitted_scaler,
)

__all__ = [
    "rank_transform",
    "zscore_transform",
    "winsorize",
    "FittedTransform",
    "create_fitted_scaler",
]
