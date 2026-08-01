# Factor cold-start coverage

- Total factors: **5556**
- Catalogs: **6**
- Active daily/extended/research operators: **207/207 (100.0%)**
- Existing GTJA/Week2 formulas excluded structurally: **453**

## Catalog summary

| Catalog | Factors | Families | Fields | Surface operators | Coverage |
|---|---:|---:|---:|---:|---:|
| `ashare_daily` | 924 | 22 | 24 | 86/86 | 100.0% |
| `ashare_extended` | 1412 | 14 | 18 | 101/101 | 100.0% |
| `ashare_research` | 308 | 7 | 6 | 20/20 | 100.0% |
| `us_daily` | 1112 | 27 | 37 | 86/86 | 100.0% |
| `us_extended` | 1492 | 15 | 21 | 101/101 | 100.0% |
| `us_research` | 308 | 7 | 6 | 20/20 | 100.0% |

## Coverage policy

- Every active canonical on `daily`, `extended`, and `research` appears in each market's matching catalog.
- Fiscal operators are isolated behind the `fundamental` availability tier and require PIT-aligned disclosure data.
- Intraday-only operators are isolated behind the `intraday` tier.
- Singular or explosive transforms are used only on explicitly bounded, singularity-free inputs and remain research-default-off.
- Internal implementation primitives and deprecated legacy aliases are not cold-start authoring targets.

### Non-authoring exclusions

- `constant`: internal implementation primitive; not exposed for factor authoring
- `cube`: deprecated legacy alias; use power(x, 3) or signed_power explicitly
- `identity`: internal implementation primitive; not exposed for factor authoring
- `protected_div`: internal compatibility primitive; use safe_div_null in authored formulas

## `ashare_daily`

### Families

| Family | Count |
|---|---:|
| `price_volume_interaction` | 144 |
| `liquidity_activity` | 116 |
| `return_momentum` | 95 |
| `volatility` | 84 |
| `fundamental_period` | 60 |
| `trend_location` | 60 |
| `liquidity_impact` | 56 |
| `fundamental_quality` | 45 |
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
| `fundamental` | 105 |

## `ashare_extended`

### Families

| Family | Count |
|---|---:|
| `distribution_tail` | 448 |
| `rolling_regression` | 305 |
| `decay_trend` | 192 |
| `conditional_history` | 174 |
| `nonlinear_experimental` | 87 |
| `technical` | 81 |
| `fundamental_diagnostics` | 32 |
| `price_deviation` | 32 |
| `compounded_return` | 24 |
| `advanced_cross_sectional` | 14 |
| `data_quality` | 10 |
| `cross_sectional_state` | 5 |
| `liquidity_activity` | 4 |
| `turnover` | 4 |

### Availability tiers

| Tier | Count |
|---|---:|
| `core` | 1362 |
| `enriched` | 18 |
| `fundamental` | 32 |

## `ashare_research`

### Families

| Family | Count |
|---|---:|
| `path_shape_research` | 144 |
| `market_relative_research` | 84 |
| `stateful_research` | 56 |
| `tail_risk_research` | 9 |
| `conditional_research` | 7 |
| `group_relative_research` | 4 |
| `intraday_research` | 4 |

### Availability tiers

| Tier | Count |
|---|---:|
| `core` | 300 |
| `enriched` | 4 |
| `intraday` | 4 |

## `us_daily`

### Families

| Family | Count |
|---|---:|
| `price_volume_interaction` | 144 |
| `liquidity_activity` | 116 |
| `return_momentum` | 95 |
| `volatility` | 84 |
| `us_derived` | 70 |
| `fundamental_period` | 60 |
| `trend_location` | 60 |
| `liquidity_impact` | 56 |
| `overnight_intraday` | 56 |
| `fundamental_quality` | 45 |
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
| `fundamental` | 105 |
| `microstructure` | 33 |

## `us_extended`

### Families

| Family | Count |
|---|---:|
| `distribution_tail` | 448 |
| `rolling_regression` | 305 |
| `decay_trend` | 192 |
| `conditional_history` | 174 |
| `nonlinear_experimental` | 87 |
| `technical` | 81 |
| `overnight_intraday` | 80 |
| `fundamental_diagnostics` | 32 |
| `price_deviation` | 32 |
| `compounded_return` | 24 |
| `advanced_cross_sectional` | 14 |
| `data_quality` | 10 |
| `cross_sectional_state` | 5 |
| `liquidity_activity` | 4 |
| `turnover` | 4 |

### Availability tiers

| Tier | Count |
|---|---:|
| `capitalization` | 4 |
| `core` | 1362 |
| `derived` | 80 |
| `enriched` | 14 |
| `fundamental` | 32 |

## `us_research`

### Families

| Family | Count |
|---|---:|
| `path_shape_research` | 144 |
| `market_relative_research` | 84 |
| `stateful_research` | 56 |
| `tail_risk_research` | 9 |
| `conditional_research` | 7 |
| `group_relative_research` | 4 |
| `intraday_research` | 4 |

### Availability tiers

| Tier | Count |
|---|---:|
| `core` | 300 |
| `enriched` | 4 |
| `intraday` | 4 |
