"""
Advanced statistical tests for quantitative finance.

This subpackage provides econometric and time series tests for:
- Granger causality (predictive relationships)
- Cointegration (common stochastic trends for pairs trading)
- Regime detection (market state identification via HMM)
- Structural breaks (parameter stability testing)
"""

from quant_evaluator.metrics.stats.granger_causality import (
    granger_causality_test,
    pairwise_granger_causality,
)
from quant_evaluator.metrics.stats.cointegration import (
    johansen_test,
    engle_granger_test,
)
from quant_evaluator.metrics.stats.regime_detection import (
    GaussianHMM,
    detect_regimes,
)
from quant_evaluator.metrics.stats.structural_breaks import (
    chow_test,
    cusum_test,
    cusum_of_squares_test,
    sup_wald_test,
    bai_perron_test,
)

__all__ = [
    "granger_causality_test",
    "pairwise_granger_causality",
    "johansen_test",
    "engle_granger_test",
    "GaussianHMM",
    "detect_regimes",
    "chow_test",
    "cusum_test",
    "cusum_of_squares_test",
    "sup_wald_test",
    "bai_perron_test",
]
