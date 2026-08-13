"""Portfolio construction utilities.

Provides tools for:
- Weight calculation (equal, market-cap, risk-based, rank-based)
- Constraint specification (turnover, leverage, position limits)
- Rebalancing with transaction costs
- Performance attribution
"""

from .weights import (
    equal_weights,
    market_cap_weights,
    risk_parity_weights,
    inverse_volatility_weights,
    rank_weights,
    normalize_weights,
)
from .constraints import (
    PortfolioConstraints,
    check_constraints,
    project_to_constraints,
)
from .rebalance import (
    calculate_trades,
    apply_transaction_costs,
    rebalance_portfolio,
)
from .attribution import (
    brinson_attribution,
    factor_attribution,
    transaction_cost_attribution,
)

__all__ = [
    "equal_weights",
    "market_cap_weights",
    "risk_parity_weights",
    "inverse_volatility_weights",
    "rank_weights",
    "normalize_weights",
    "PortfolioConstraints",
    "check_constraints",
    "project_to_constraints",
    "calculate_trades",
    "apply_transaction_costs",
    "rebalance_portfolio",
    "brinson_attribution",
    "factor_attribution",
    "transaction_cost_attribution",
]
