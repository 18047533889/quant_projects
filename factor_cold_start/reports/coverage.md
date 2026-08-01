# Factor cold-start coverage

- Total factors: **5300**
- Catalogs: **4**
- Existing GTJA/Week2 formulas excluded structurally: **227**

## Catalog summary

| Catalog | Factors | Families | Fields | Eligible operators | Coverage |
|---|---:|---:|---:|---:|---:|
| `ashare_daily` | 819 | 20 | 13 | 79/79 | 100.0% |
| `ashare_extended` | 1699 | 23 | 23 | 377/377 | 100.0% |
| `us_daily` | 1007 | 25 | 26 | 79/79 | 100.0% |
| `us_extended` | 1775 | 23 | 25 | 376/376 | 100.0% |

## `ashare_daily`

### Families

| Family | Count |
|---|---:|
| `price_volume_interaction` | 144 |
| `liquidity_activity` | 116 |
| `return_momentum` | 95 |
| `volatility` | 84 |
| `trend_location` | 60 |
| `liquidity_impact` | 56 |
| `nonlinear_robust` | 41 |
| `candle_vwap` | 40 |
| `adjusted_price` | 32 |
| `reversal` | 28 |
| `serial_dependence` | 24 |
| `trend_quality` | 24 |
| `conditional_regime` | 22 |
| `cross_sectional_state` | 18 |
| `turnover` | 18 |
| `group_relative` | 8 |
| `adjustment_event` | 3 |
| `data_quality` | 3 |
| `risk_scaling` | 2 |
| `discrete_regime` | 1 |

### Availability tiers

| Tier | Count |
|---|---:|
| `core` | 793 |
| `enriched` | 26 |

### Intentionally excluded operators

- `period_average`: fundamental fiscal-period operator; excluded from price-volume cold start
- `period_cagr`: fundamental fiscal-period operator; excluded from price-volume cold start
- `period_change`: fundamental fiscal-period operator; excluded from price-volume cold start
- `quarter_from_cumulative`: fundamental statement transformation
- `ttm_from_cumulative`: fundamental statement transformation
- `ttm_from_quarterly`: fundamental statement transformation
- `yoy_by_period`: fundamental statement transformation

## `ashare_extended`

### Families

| Family | Count |
|---|---:|
| `distribution_tail` | 448 |
| `rolling_regression` | 305 |
| `decay_trend` | 192 |
| `conditional_history` | 174 |
| `technical` | 81 |
| `technical_extension` | 81 |
| `nonlinear_experimental` | 66 |
| `technical_structure` | 55 |
| `fundamental_transform` | 49 |
| `liquidity` | 38 |
| `price_deviation` | 32 |
| `technical_indicator` | 32 |
| `compounded_return` | 24 |
| `intraday_to_daily` | 24 |
| `production_promoted` | 20 |
| `advanced_cross_sectional` | 14 |
| `analyst_expectation` | 14 |
| `candlestick_pattern` | 14 |
| `candlestick` | 13 |
| `data_quality` | 10 |
| `cross_sectional_state` | 5 |
| `liquidity_activity` | 4 |
| `turnover` | 4 |

### Availability tiers

| Tier | Count |
|---|---:|
| `analyst` | 14 |
| `core` | 1593 |
| `enriched` | 19 |
| `fundamental` | 49 |
| `minute` | 24 |

### Intentionally excluded operators

- `arg`: complex-valued phase is not meaningful for real daily price-volume inputs
- `cosh`: explosive transform; unsuitable for unbounded production signals
- `cot`: singular trigonometric transform; unstable and economically unmotivated
- `csc`: singular trigonometric transform; unstable and economically unmotivated
- `fundamental_staleness`: fundamental availability diagnostic
- `period_lag`: fundamental fiscal-period operator; excluded from price-volume cold start
- `period_stability`: fundamental fiscal-period operator; excluded from price-volume cold start
- `revision_delta`: fundamental revision diagnostic
- `sec`: singular trigonometric transform; unstable and economically unmotivated
- `sinh`: explosive transform; unsuitable for unbounded production signals
- `tan`: unbounded periodic transform; unstable around singularities

## `us_daily`

### Families

| Family | Count |
|---|---:|
| `price_volume_interaction` | 144 |
| `liquidity_activity` | 116 |
| `return_momentum` | 95 |
| `volatility` | 84 |
| `us_derived` | 70 |
| `trend_location` | 60 |
| `liquidity_impact` | 56 |
| `overnight_intraday` | 56 |
| `nonlinear_robust` | 41 |
| `candle_vwap` | 40 |
| `microstructure` | 33 |
| `adjusted_price` | 32 |
| `reversal` | 28 |
| `short_flow` | 28 |
| `serial_dependence` | 24 |
| `trend_quality` | 24 |
| `conditional_regime` | 22 |
| `cross_sectional_state` | 18 |
| `turnover` | 18 |
| `group_relative` | 8 |
| `adjustment_event` | 3 |
| `data_quality` | 3 |
| `risk_scaling` | 2 |
| `data_consistency` | 1 |
| `discrete_regime` | 1 |

### Availability tiers

| Tier | Count |
|---|---:|
| `core` | 793 |
| `derived` | 127 |
| `enriched` | 54 |
| `microstructure` | 33 |

### Intentionally excluded operators

- `period_average`: fundamental fiscal-period operator; excluded from price-volume cold start
- `period_cagr`: fundamental fiscal-period operator; excluded from price-volume cold start
- `period_change`: fundamental fiscal-period operator; excluded from price-volume cold start
- `quarter_from_cumulative`: fundamental statement transformation
- `ttm_from_cumulative`: fundamental statement transformation
- `ttm_from_quarterly`: fundamental statement transformation
- `yoy_by_period`: fundamental statement transformation

## `us_extended`

### Families

| Family | Count |
|---|---:|
| `distribution_tail` | 448 |
| `rolling_regression` | 305 |
| `decay_trend` | 192 |
| `conditional_history` | 174 |
| `technical` | 81 |
| `technical_extension` | 81 |
| `overnight_intraday` | 80 |
| `nonlinear_experimental` | 66 |
| `technical_structure` | 55 |
| `fundamental_transform` | 49 |
| `liquidity` | 38 |
| `price_deviation` | 32 |
| `technical_indicator` | 32 |
| `compounded_return` | 24 |
| `intraday_to_daily` | 24 |
| `production_promoted` | 20 |
| `advanced_cross_sectional` | 14 |
| `analyst_expectation` | 14 |
| `candlestick_pattern` | 14 |
| `candlestick` | 13 |
| `data_quality` | 10 |
| `cross_sectional_state` | 5 |
| `liquidity_activity` | 4 |

### Availability tiers

| Tier | Count |
|---|---:|
| `analyst` | 14 |
| `core` | 1593 |
| `derived` | 80 |
| `enriched` | 15 |
| `fundamental` | 49 |
| `minute` | 24 |

### Intentionally excluded operators

- `arg`: complex-valued phase is not meaningful for real daily price-volume inputs
- `cosh`: explosive transform; unsuitable for unbounded production signals
- `cot`: singular trigonometric transform; unstable and economically unmotivated
- `csc`: singular trigonometric transform; unstable and economically unmotivated
- `fundamental_staleness`: fundamental availability diagnostic
- `period_lag`: fundamental fiscal-period operator; excluded from price-volume cold start
- `period_stability`: fundamental fiscal-period operator; excluded from price-volume cold start
- `real_turnover_rate`: requires a verified free-float share-count field; the current US contract does not guarantee one
- `revision_delta`: fundamental revision diagnostic
- `sec`: singular trigonometric transform; unstable and economically unmotivated
- `sinh`: explosive transform; unsuitable for unbounded production signals
- `tan`: unbounded periodic transform; unstable around singularities
