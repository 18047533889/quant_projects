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
    UnknownRegimePolicy,
    serialize_regime_weights,
    deserialize_regime_weights,
)
from factor_preprocess.regime.switching import (
    regime_switching_transform,
    fit_regime_switching,
    RegimeSwitchingState,
)
from factor_preprocess.regime.causal_detector import CausalRegimeDetector

__all__ = [
    "detect_variance_regime",
    "detect_correlation_regime",
    "RegimeState",
    "regime_adaptive_weights",
    "fit_regime_weights",
    "RegimeWeightState",
    "UnknownRegimePolicy",
    "serialize_regime_weights",
    "deserialize_regime_weights",
    "regime_switching_transform",
    "fit_regime_switching",
    "RegimeSwitchingState",
    "CausalRegimeDetector",
]
