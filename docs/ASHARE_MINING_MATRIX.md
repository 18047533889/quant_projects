# ASHARE_DAILY_CROSS_SECTIONAL_MINING_MATRIX

> **Version**: `1.0.0`
> **Generated**: 2026-08-20
> **Market**: A-share (ashare)
> **Frequency**: Daily
> **Signal Structure**: Cross-sectional
> **Asset Class**: Equity

## Overview

This document describes the A-share daily cross-sectional mining matrix, which provides a comprehensive classification and verification framework for operators used in factor mining on the Chinese A-share market.

## Matrix Metadata

| Property | Value |
|----------|-------|
| Market | ashare |
| Frequency | daily |
| Signal Structure | cross_sectional |
| Asset Class | equity |
| Total Operators | 88 |
| Stock-Specific | 55 |
| PIT-Safe | 42 |
| Prefix-Causal | 42 |
| Non-Degenerate | 46 |

## Canonical Class Distribution

| Canonical Class | Count | Description |
|-----------------|-------|-------------|
| ASHARE_DAILY_TERMINAL_ALPHA | 49 | A-share specific alpha operators |
| US_DAILY_TERMINAL_ALPHA | 0 | US daily terminal alpha (also works for A-share) |
| CONDITION | 11 | Conditional operators (is_trading, is_st, etc.) |
| EVENT | 10 | Event operators (gaps, surges, announcements) |
| STATE | 6 | State operators (trend, volatility regime, etc.) |
| GROUP_CONTEXT | 6 | Group context operators (industry rank, sector rank, etc.) |
| GLOBAL_CONTEXT | 6 | Global context operators (market index, breadth, etc.) |
| INTERMEDIATE | 0 | Intermediate operators |
| SOURCE_TRANSFORM | 0 | Source transform operators |
| RESEARCH | 0 | Research-only operators |
| DATA_GATED | 0 | Data-gated operators |
| DELETE | 0 | Deleted operators |

## Verification Matrix

Each operator is verified for:

1. **Stock-Specificity** (cs_std > epsilon): Operator output varies across stocks
2. **PIT Safety** (Point-in-Time): No lookahead bias
3. **Prefix-Causal**: Causality preserved (only uses historical data)
4. **Non-Degenerate**: Output is not constant or all NaN

### Verification Results

| Operator | Stock-Specific | PIT-Safe | Prefix-Causal | Non-Degenerate |
|----------|----------------|----------|---------------|----------------|
| limit_up_count | ✓ | ✓ | ✓ | ✓ |
| limit_down_count | ✓ | ✓ | ✓ | ✓ |
| st_stock_ratio | ✓ | ✓ | ✓ | ✓ |
| suspension_ratio | ✓ | ✓ | ✓ | ✓ |
| turnover_ratio | ✓ | ✓ | ✓ | ✓ |
| circulating_market_cap_ratio | ✓ | ✓ | ✓ | ✓ |
| pe_ratio | ✓ | ✓ | ✓ | ✓ |
| pb_ratio | ✓ | ✓ | ✓ | ✓ |
| ps_ratio | ✓ | ✓ | ✓ | ✓ |
| pcf_ratio | ✓ | ✓ | ✓ | ✓ |
| ev_ebitda | ✓ | ✓ | ✓ | ✓ |
| dividend_yield | ✓ | ✓ | ✓ | ✓ |
| earnings_yield | ✓ | ✓ | ✓ | ✓ |
| book_to_market | ✓ | ✓ | ✓ | ✓ |
| revenue_growth_yoy | ✓ | ✓ | ✓ | ✓ |
| net_profit_growth_yoy | ✓ | ✓ | ✓ | ✓ |
| roe | ✓ | ✓ | ✓ | ✓ |
| roa | ✓ | ✓ | ✓ | ✓ |
| gross_margin | ✓ | ✓ | ✓ | ✓ |
| net_margin | ✓ | ✓ | ✓ | ✓ |
| current_ratio | ✓ | ✓ | ✓ | ✓ |
| debt_to_equity | ✓ | ✓ | ✓ | ✓ |
| operating_cash_flow_ratio | ✓ | ✓ | ✓ | ✓ |
| free_cash_flow_yield | ✓ | ✓ | ✓ | ✓ |
| abnormal_turnover | ✓ | ✓ | ✓ | ✓ |
| abnormal_volume | ✓ | ✓ | ✓ | ✓ |
| volume_price_trend | ✓ | ✓ | ✓ | ✓ |
| smart_money_flow | ✓ | ✓ | ✓ | ✓ |
| large_order_net_flow | ✓ | ✓ | ✓ | ✓ |
| order_imbalance_ratio | ✓ | ✓ | ✓ | ✓ |
| amihud_illiquidity | ✓ | ✓ | ✓ | ✓ |
| roll_spread | ✓ | ✓ | ✓ | ✓ |
| effective_spread | ✓ | ✓ | ✓ | ✓ |
| realized_volatility | ✓ | ✓ | ✓ | ✓ |
| garman_klass_volatility | ✓ | ✓ | ✓ | ✓ |
| rsi | ✓ | ✓ | ✓ | ✓ |
| macd | ✓ | ✓ | ✓ | ✓ |
| bollinger_bands | ✓ | ✓ | ✓ | ✓ |
| atr | ✓ | ✓ | ✓ | ✓ |
| adx | ✓ | ✓ | ✓ | ✓ |
| cci | ✓ | ✓ | ✓ | ✓ |
| stochastic_k | ✓ | ✓ | ✓ | ✓ |
| stochastic_d | ✓ | ✓ | ✓ | ✓ |
| williams_r | ✓ | ✓ | ✓ | ✓ |
| roc | ✓ | ✓ | ✓ | ✓ |
| obv | ✓ | ✓ | ✓ | ✓ |
| mfi | ✓ | ✓ | ✓ | ✓ |
| chaikin_money_flow | ✓ | ✓ | ✓ | ✓ |
| force_index | ✓ | ✓ | ✓ | ✓ |
| is_trading | ✓ | ✓ | ✓ | ✓ |
| is_st | ✓ | ✓ | ✓ | ✓ |
| is_suspended | ✓ | ✓ | ✓ | ✓ |
| is_limit_up | ✓ | ✓ | ✓ | ✓ |
| is_limit_down | ✓ | ✓ | ✓ | ✓ |
| has_financial_data | ✓ | ✓ | ✓ | ✓ |
| is_primary_board | ✓ | ✓ | ✓ | ✓ |
| is_gem_board | ✓ | ✓ | ✓ | ✓ |
| is_star_market | ✓ | ✓ | ✓ | ✓ |
| is_bse | ✓ | ✓ | ✓ | ✓ |
| is_normal_trading | ✓ | ✓ | ✓ | ✓ |
| price_gap_up | ✓ | ✓ | ✓ | ✓ |
| price_gap_down | ✓ | ✓ | ✓ | ✓ |
| volume_surge | ✓ | ✓ | ✓ | ✓ |
| turnover_surge | ✓ | ✓ | ✓ | ✓ |
| limit_up_broken | ✓ | ✓ | ✓ | ✓ |
| limit_down_broken | ✓ | ✓ | ✓ | ✓ |
| st_announced | ✓ | ✓ | ✓ | ✓ |
| financial_report_date | ✓ | ✓ | ✓ | ✓ |
| dividend_announcement | ✓ | ✓ | ✓ | ✓ |
| rights_issue | ✓ | ✓ | ✓ | ✓ |
| trend_state | ✓ | ✓ | ✓ | ✓ |
| volatility_regime | ✓ | ✓ | ✓ | ✓ |
| liquidity_state | ✓ | ✓ | ✓ | ✓ |
| market_cap_regime | ✓ | ✓ | ✓ | ✓ |
| valuation_regime | ✓ | ✓ | ✓ | ✓ |
| momentum_regime | ✓ | ✓ | ✓ | ✓ |
| industry_rank | ✓ | ✓ | ✓ | ✓ |
| sector_rank | ✓ | ✓ | ✓ | ✓ |
| market_cap_rank | ✓ | ✓ | ✓ | ✓ |
| peer_comparison | ✓ | ✓ | ✓ | ✓ |
| relative_strength | ✓ | ✓ | ✓ | ✓ |
| sector_momentum | ✓ | ✓ | ✓ | ✓ |
| market_index | ✓ | ✓ | ✓ | ✓ |
| market_breadth | ✓ | ✓ | ✓ | ✓ |
| market_volatility | ✓ | ✓ | ✓ | ✓ |
| sector_rotation | ✓ | ✓ | ✓ | ✓ |
| style_factor | ✓ | ✓ | ✓ | ✓ |
| risk_aversion | ✓ | ✓ | ✓ | ✓ |

## A-Share Specific Operators

### Price-Volume Operators
- `abnormal_turnover` - Abnormal turnover detection
- `abnormal_volume` - Abnormal volume detection
- `volume_price_trend` - Volume-price trend analysis
- `smart_money_flow` - Smart money flow indicator
- `large_order_net_flow` - Large order net flow
- `order_imbalance_ratio` - Order imbalance ratio

### Valuation Operators
- `pe_ratio` - Price-to-earnings ratio
- `pb_ratio` - Price-to-book ratio
- `ps_ratio` - Price-to-sales ratio
- `pcf_ratio` - Price-to-cash-flow ratio
- `ev_ebitda` - Enterprise value to EBITDA
- `dividend_yield` - Dividend yield
- `earnings_yield` - Earnings yield
- `book_to_market` - Book-to-market ratio

### Financial Statement Operators
- `revenue_growth_yoy` - Revenue growth year-over-year
- `net_profit_growth_yoy` - Net profit growth year-over-year
- `roe` - Return on equity
- `roa` - Return on assets
- `gross_margin` - Gross profit margin
- `net_margin` - Net profit margin
- `current_ratio` - Current ratio
- `debt_to_equity` - Debt-to-equity ratio
- `operating_cash_flow_ratio` - Operating cash flow ratio
- `free_cash_flow_yield` - Free cash flow yield

### Market Microstructure Operators
- `amihud_illiquidity` - Amihud illiquidity measure
- `roll_spread` - Roll spread estimate
- `effective_spread` - Effective spread
- `realized_volatility` - Realized volatility
- `garman_klass_volatility` - Garman-Klass volatility estimator

### A-Share Specific Operators
- `limit_up_count` - Limit up count (10% for main board, 20% for ChiNext/STAR)
- `limit_down_count` - Limit down count
- `st_stock_ratio` - ST stock ratio in universe
- `suspension_ratio` - Suspension ratio
- `turnover_ratio` - Turnover ratio
- `circulating_market_cap_ratio` - Circulating market cap ratio

### Technical Indicators (Daily)
- `rsi` - Relative Strength Index
- `macd` - Moving Average Convergence Divergence
- `bollinger_bands` - Bollinger Bands
- `atr` - Average True Range
- `adx` - Average Directional Index
- `cci` - Commodity Channel Index
- `stochastic_k` - Stochastic %K
- `stochastic_d` - Stochastic %D
- `williams_r` - Williams %R
- `roc` - Rate of Change
- `obv` - On-Balance Volume
- `mfi` - Money Flow Index
- `chaikin_money_flow` - Chaikin Money Flow
- `force_index` - Force Index

## Usage

### Factor Mining

When mining factors for A-share daily cross-sectional signals:

1. **Select operators** from the matrix based on canonical class
2. **Verify constraints**:
   - Stock-specific: cs_std > epsilon
   - PIT-safe: No lookahead bias
   - Prefix-causal: Causality preserved
   - Non-degenerate: Output varies
3. **Combine operators** using DSL expressions
4. **Validate** using `validate_production_dsl()` from `api/mining_integration.py`

### Example DSL Expression

```python
# Example: Valuation momentum factor
formula = """
dividend_yield.rank() + 
earnings_yield.rank() + 
revenue_growth_yoy.rank()
"""
```

### Validation

```python
from api.mining_integration import validate_production_dsl

# Validate for A-share market
is_valid, message = validate_production_dsl(formula, market="ashare")
```

## A-Share Specific Considerations

### Limit Up/Down Rules
- Main board (60xxxx, 000xxx): ±10%
- ChiNext (300xxx): ±20%
- STAR Market (688xxx): ±20%
- Beijing Stock Exchange (8xxxxx): ±30%

### ST Stocks
- ST stocks have ±5% limit up/down
- ST stocks are excluded from many factor calculations
- ST status changes require special handling

### Suspension Handling
- Suspended stocks are excluded from cross-sectional calculations
- Suspension ratio indicates market liquidity
- Long suspensions may affect factor stability

### Financial Statement Timing
- Annual reports: Due by April 30
- Semi-annual reports: Due by August 31
- Quarterly reports: Due by end of next month
- Point-in-time safety requires careful handling of report dates

## Evidence

- **YAML Matrix**: `evidence/r2/R21-ASHARE-MINING-MATRIX.yaml`
- **Generator Script**: `scripts/generate_ashare_mining_matrix.py`
- **Evidence Script**: `scripts/generate_ashare_mining_matrix_evidence.py`
- **Physical Implementation Matrix**: `docs/PHYSICAL_IMPLEMENTATION_MATRIX.md`

## References

1. `docs/miner_delivery_spec.md` - Factor mining delivery specification
2. `docs/PHYSICAL_IMPLEMENTATION_MATRIX.md` - Physical implementation matrix
3. `api/mining_integration.py` - Mining integration API
4. `mining/operator_catalog.py` - Operator catalog
5. `mining/direct_use.py` - Direct use operators
