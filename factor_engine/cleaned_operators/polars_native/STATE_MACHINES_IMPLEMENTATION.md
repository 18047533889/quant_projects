# State Machine Operators - Polars Native Implementation

## Overview

Implemented all 19 state machine operators (`state_*` family) with genuine Polars-native logic for Phase 2 of the polars backend migration.

**Implementation Date:** 2026-08-13

## Operators Implemented

### Adaptive Filters (5 operators)
1. **state_adaptive_deadband** - Adaptive deadband filter where threshold adjusts based on recent volatility
2. **state_adaptive_slew_limit** - Adaptive slew rate limiter where max change rate adjusts to volatility
3. **state_confidence_weighted_ema** - EMA where smoothing adapts to confidence level (high confidence = faster adaptation)
4. **state_cost_aware_deadband** - Deadband filter where threshold reflects transaction cost and expected benefit
5. **state_cost_aware_slew** - Slew limiter that penalizes rapid changes with higher effective cost

### Episode Metrics (5 operators)
6. **state_episode_efficiency** - Ratio of net excursion to path length in current episode
7. **state_episode_excursion_balance** - Balance between favorable and adverse excursions: (MFE - MAE) / (MFE + MAE)
8. **state_episode_mae** - Maximum adverse excursion (worst drawdown against position direction) in episode
9. **state_episode_mfe** - Maximum favorable excursion (best profit in position direction) in episode
10. **state_episode_retrace_ratio** - Retracement ratio from peak to current position

### Turnover and Adjustment (2 operators)
11. **state_l1_turnover_prox** - L1-regularized state that minimizes turnover with soft-threshold
12. **state_l2_partial_adjustment** - Partial adjustment model: new = old + alpha * (target - old)

### Quantile and Rank Based (2 operators)
13. **state_quantile_hysteresis** - Quantile-based hysteresis state machine with high/low bands
14. **state_rank_deadband** - Deadband applied to rolling rank percentile instead of raw value

### Since-Event Tracking (3 operators)
15. **state_since_mean** - Running mean of values since last reset event
16. **state_since_sum** - Cumulative sum since last reset event
17. **state_since_trend_tstat** - T-statistic of linear trend since last reset event

### Basic Filters (2 operators)
18. **state_slew_limit** - Basic slew rate limiter (max change rate per step)
19. **state_uncertainty_deadband** - Deadband threshold scales with uncertainty estimate

## Implementation Details

### File Location
```
/home/shw/quant_projects/factor_engine/cleaned_operators/polars_native/state_machines.py
```

### Key Patterns Used

1. **State Tracking with NumPy**
   - Used NumPy arrays for stateful iteration (more efficient than pure Polars for complex state machines)
   - Maintained current state across rows using loop-based logic

2. **Polars for Preprocessing**
   - Used Polars `rolling_std()`, `rolling_quantile()` for adaptive thresholds
   - Leveraged Polars lazy evaluation for efficient window operations

3. **Proper Parameter Validation**
   - All parameters use `ParamSpec` with appropriate `ParamRole`
   - Used `ParamRole.STATE_THRESHOLD` for threshold parameters
   - Used `ParamRole.HORIZON` for window parameters

4. **Causal Logic**
   - All operators are strictly causal (depend only on past data)
   - State resets properly on NaN values
   - No lookahead bias

### Registration

All 19 operators are registered with:
- `backend="polars"`
- `category="state"` (stored as "general" by decorator)
- Proper metadata with descriptions and parameter specifications

### Testing

Basic functionality tests confirm:
- ✓ All 19 operators register successfully
- ✓ `state_slew_limit` correctly limits rate of change
- ✓ `state_since_sum` properly resets and accumulates
- ✓ Episode operators track MFE/MAE correctly

## Usage Example

```python
from cleaned_operators.registry import OperatorRegistry
import pandas as pd

registry = OperatorRegistry()

# Get operator (use mode='any' for non-production operators)
op = registry.get('state_slew_limit', backend='polars', mode='any')

# Create test signal
signal = pd.Series([0.0, 0.5, 1.0, 1.5, 2.0], name='signal')

# Apply slew limit (max change = 0.3 per step)
result = op._calculate_series(signal, max_rate=0.3)
# Output: [0.0, 0.3, 0.6, 0.9, 1.2]
```

## Integration

Updated `/home/shw/quant_projects/factor_engine/cleaned_operators/polars_native/__init__.py` to import and export all 19 operators.

## Notes

- Operators are filtered in production mode (only "daily" and "extended" surface operators are accessible by default)
- Use `mode='any'` when retrieving operators programmatically
- All operators handle NaN values gracefully
- Episode operators assume state values in {-1, 0, +1} for direction tracking

## Performance Characteristics

- **Adaptive filters**: O(n) time, O(1) space per stock
- **Episode metrics**: O(n) time, O(1) space per stock
- **Since-event tracking**: O(n) time, O(window) space for trend_tstat
- **Quantile operations**: O(n * window) time for rolling quantile computation

All operators process per-stock time series independently and can be parallelized across instruments.
