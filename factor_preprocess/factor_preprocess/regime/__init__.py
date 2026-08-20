"""Regime-adaptive transforms for factor preprocessing."""
from factor_preprocess.regime.detector import (
    detect_variance_regime,
    detect_correlation_regime,
    RegimeState,
)
from factor_preprocess.regime.adaptive_weights import (
    regime_adaptive_weights,
    fit_regime_weights,
    RegimeWeightState,
)
from factor_preprocess.regime.switching import (
    regime_switching_transform,
    fit_regime_switching,
    RegimeSwitchingState,
)

__all__ = [
    "detect_variance_regime",
    "detect_correlation_regime",
    "RegimeState",
    "regime_adaptive_weights",
    "fit_regime_weights",
    "RegimeWeightState",
    "regime_switching_transform",
    "fit_regime_switching",
    "RegimeSwitchingState",
]
