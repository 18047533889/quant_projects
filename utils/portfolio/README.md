# Portfolio Construction Utilities

Lightweight portfolio construction utilities for quantitative research. Provides weighting schemes, constraint handling, rebalancing logic, and performance attribution.

## Features

- **Weighting schemes**: equal, market-cap, inverse volatility, risk parity, rank-based
- **Constraint specification**: position limits, leverage, turnover, group exposures
- **Rebalancing**: transaction cost modeling, trade scheduling, cost-aware optimization
- **Performance attribution**: Brinson attribution, factor-based attribution, cost attribution

## Installation

Add to your Python path:
```python
import sys
sys.path.append('/home/shw/quant_projects')
```

## Quick Start

### Weight Calculation

```python
from utils.portfolio import equal_weights, market_cap_weights, rank_weights

# Equal weight
weights = equal_weights(['AAPL', 'MSFT', 'GOOGL'])

# Market cap weighted
caps = pd.Series([2000, 1500, 1000], index=['AAPL', 'MSFT', 'GOOGL'])
weights = market_cap_weights(caps, max_weight=0.5)

# Rank-based from signals
signals = pd.Series([0.8, 0.3, 0.6], index=['AAPL', 'MSFT', 'GOOGL'])
weights = rank_weights(signals, method='linear')

# Risk parity
cov = returns.cov()
weights = risk_parity_weights(cov)
```

### Constraints

```python
from utils.portfolio import PortfolioConstraints, check_constraints, project_to_constraints

# Define constraints
constraints = PortfolioConstraints(
    min_weight=0.0,
    max_weight=0.3,
    max_leverage=1.0,
    max_turnover=0.2,
    target_sum=1.0
)

# Check if weights satisfy constraints
results = check_constraints(target_weights, constraints, current_weights)

# Project weights to satisfy constraints
adjusted = project_to_constraints(target_weights, constraints, current_weights)
```

### Rebalancing

```python
from utils.portfolio import TransactionCostModel, rebalance_portfolio

# Define cost model
cost_model = TransactionCostModel(
    linear_bps=10,      # 10 basis points
    quadratic_bps=50,   # Market impact
    fixed_cost=1.0,     # Fixed cost per trade
    min_trade_size=0.01 # Minimum 1% trade
)

# Rebalance
result = rebalance_portfolio(
    target_weights=new_weights,
    current_weights=old_weights,
    cost_model=cost_model,
    max_turnover=0.3
)

print(f"Trades: {result['trades']}")
print(f"Total cost: {result['total_cost']:.4f}")
print(f"Turnover: {result['turnover']:.2%}")
```

### Performance Attribution

```python
from utils.portfolio import brinson_attribution, factor_attribution

# Brinson attribution
attr = brinson_attribution(
    portfolio_weights=port_weights,
    benchmark_weights=bench_weights,
    asset_returns=returns
)

print(f"Active return: {attr['active_return']:.2%}")
print(f"Allocation effect: {attr['allocation_effect']:.2%}")
print(f"Selection effect: {attr['selection_effect']:.2%}")

# Factor attribution
factor_attr = factor_attribution(
    portfolio_weights=weights,
    asset_returns=returns,
    factor_exposures=exposures,
    factor_returns=factor_rets
)

print(f"Factor contributions: {factor_attr['factor_contributions']}")
print(f"Specific return: {factor_attr['specific_return']:.2%}")
```

## Module Overview

### `weights.py`
- `equal_weights()` - Equal weighted portfolio
- `market_cap_weights()` - Market cap weighted
- `inverse_volatility_weights()` - Inverse volatility weighted
- `risk_parity_weights()` - Risk parity (equal risk contribution)
- `rank_weights()` - Rank-based from signals (linear/quadratic/exponential)
- `normalize_weights()` - Normalize weights with optional bounds

### `constraints.py`
- `PortfolioConstraints` - Constraint specification dataclass
- `check_constraints()` - Check if weights satisfy constraints
- `project_to_constraints()` - Project weights to satisfy constraints
- `calculate_leverage()` - Calculate portfolio leverage
- `calculate_turnover()` - Calculate portfolio turnover
- `calculate_net_exposure()` - Net exposure
- `calculate_long_short_exposure()` - Long and short exposure

### `rebalance.py`
- `TransactionCostModel` - Cost model specification
- `calculate_trades()` - Calculate trades to reach target
- `apply_transaction_costs()` - Calculate costs for trades
- `rebalance_portfolio()` - Full rebalancing workflow
- `optimize_rebalance_with_costs()` - Cost-aware optimization
- `schedule_rebalance()` - Multi-period trade scheduling

### `attribution.py`
- `brinson_attribution()` - Brinson-Fachler attribution
- `sector_brinson_attribution()` - Sector-level Brinson
- `factor_attribution()` - Factor-based attribution
- `transaction_cost_attribution()` - Cost breakdown
- `rolling_attribution()` - Rolling attribution over time

## Design Philosophy

This is **not** a full portfolio optimizer. It provides building blocks for portfolio construction:

- **Explicit constraints**: Define and enforce constraints programmatically
- **Transaction cost awareness**: Model costs at the rebalancing stage
- **Attribution tools**: Understand sources of return
- **Research-focused**: Lightweight, transparent, easy to customize

For full mean-variance optimization with convex solvers, see dedicated libraries like `cvxpy` or `PyPortfolioOpt`.

## Testing

Run tests:
```bash
pytest utils/portfolio/ -v
```

All 71 tests pass.

## License

Part of the quant_projects research environment.
