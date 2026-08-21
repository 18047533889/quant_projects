"""
Fast computational kernels for quant_evaluator.

Vectorized implementations optimized for large-scale factor evaluation (10k+ factors).
All fast kernels maintain parity with reference implementations in metrics/.
"""

from quant_evaluator.kernels.fast import (
    fast_ic_batch,
    fast_quantile_binning,
    fast_turnover_estimate,
)
from quant_evaluator.kernels.reference_bridge import (
    check_ic_parity,
    check_quantile_parity,
    check_turnover_parity,
)

__all__ = [
    "fast_ic_batch",
    "fast_quantile_binning",
    "fast_turnover_estimate",
    "check_ic_parity",
    "check_quantile_parity",
    "check_turnover_parity",
]
