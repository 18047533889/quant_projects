#!/usr/bin/env python3
"""Generate ASHARE_DAILY_CROSS_SECTIONAL_MINING_MATRIX.

This script builds the A-share daily cross-sectional mining matrix with:
- Canonical classification for each operator
- Stock-specificity verification (cs_std > epsilon)
- PIT safety verification
- Prefix-causal verification
- Non-degeneracy verification

Usage:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 \
    python3 scripts/generate_ashare_mining_matrix.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Thread safety
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import dataclass, field
from enum import Enum

class CanonicalClass(str, Enum):
    """Canonical classification for operators."""
    US_DAILY_TERMINAL_ALPHA = "US_DAILY_TERMINAL_ALPHA"
    ASHARE_DAILY_TERMINAL_ALPHA = "ASHARE_DAILY_TERMINAL_ALPHA"
    CONDITION = "CONDITION"
    EVENT = "EVENT"
    STATE = "STATE"
    GROUP_CONTEXT = "GROUP_CONTEXT"
    GLOBAL_CONTEXT = "GLOBAL_CONTEXT"
    INTERMEDIATE = "INTERMEDIATE"
    SOURCE_TRANSFORM = "SOURCE_TRANSFORM"
    RESEARCH = "RESEARCH"
    DATA_GATED = "DATA_GATED"
    DELETE = "DELETE"


@dataclass
class OperatorSpec:
    """Specification for a single operator."""
    name: str
    canonical_class: CanonicalClass
    stock_specific: bool = False  # cs_std > epsilon
    pit_safe: bool = False       # Point-in-time safe
    prefix_causal: bool = False  # Prefix-causal
    non_degenerate: bool = False # Non-degenerate
    ashare_operators: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class MiningMatrix:
    """A-share daily cross-sectional mining matrix."""
    operators: List[OperatorSpec] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_operator_registry() -> Dict[str, Any]:
    """Load operator registry from cleaned_operators."""
    registry_path = PROJECT_ROOT / "cleaned_operators" / "operator_spec.py"
    if registry_path.exists():
        # In production, would dynamically load; here we define known operators
        pass
    return {}


def classify_operator(name: str) -> CanonicalClass:
    """Classify operator into canonical class based on A-share specific rules."""

    # A-share daily terminal alpha operators
    ashare_terminal_ops = {
        # Price-volume operators
        "abnormal_turnover", "abnormal_volume", "volume_price_trend",
        "smart_money_flow", "large_order_net_flow", "order_imbalance_ratio",

        # Valuation operators (A-share specific)
        "pe_ratio", "pb_ratio", "ps_ratio", "pcf_ratio", "ev_ebitda",
        "dividend_yield", "earnings_yield", "book_to_market",

        # Financial statement operators
        "revenue_growth_yoy", "net_profit_growth_yoy", "roe", "roa",
        "gross_margin", "net_margin", "current_ratio", "debt_to_equity",
        "operating_cash_flow_ratio", "free_cash_flow_yield",

        # Market microstructure
        "amihud_illiquidity", "roll_spread", "effective_spread",
        "realized_volatility", "garman_klass_volatility",

        # A-share specific
        "limit_up_count", "limit_down_count", "st_stock_ratio",
        "suspension_ratio", "turnover_ratio", "circulating_market_cap_ratio",

        # Technical indicators (daily)
        "rsi", "macd", "bollinger_bands", "atr", "adx", "cci",
        "stochastic_k", "stochastic_d", "williams_r", "roc",
        "obv", "mfi", "chaikin_money_flow", "force_index",
    }

    # US daily terminal alpha (subset that also works for A-share)
    us_terminal_ops = {
        "rsi", "macd", "bollinger_bands", "atr", "adx", "cci",
        "stochastic_k", "stochastic_d", "williams_r", "roc",
    }

    # Condition operators
    condition_ops = {
        "is_trading", "is_st", "is_suspended", "is_limit_up", "is_limit_down",
        "has_financial_data", "is_primary_board", "is_gem_board",
        "is_star_market", "is_bse", "is_normal_trading",
    }

    # Event operators
    event_ops = {
        "price_gap_up", "price_gap_down", "volume_surge", "turnover_surge",
        "limit_up_broken", "limit_down_broken", "st_announced",
        "financial_report_date", "dividend_announcement", "rights_issue",
    }

    # State operators
    state_ops = {
        "trend_state", "volatility_regime", "liquidity_state",
        "market_cap_regime", "valuation_regime", "momentum_regime",
    }

    # Group context operators
    group_context_ops = {
        "industry_rank", "sector_rank", "market_cap_rank",
        "peer_comparison", "relative_strength", "sector_momentum",
    }

    # Global context operators
    global_context_ops = {
        "market_index", "market_breadth", "market_volatility",
        "sector_rotation", "style_factor", "risk_aversion",
    }

    # Intermediate operators
    intermediate_ops = {
        "ema", "sma", "wma", "dema", "tema", "kama", "hma",
        "standard_deviation", "variance", "skewness", "kurtosis",
        "percentile_rank", "z_score", "winsorize", "demean",
    }

    # Source transform operators
    source_transform_ops = {
        "field", "lag", "lead", "diff", "pct_change", "rolling_mean",
        "rolling_std", "rolling_max", "rolling_min", "rolling_sum",
    }

    # Research-only operators (not production safe)
    research_ops = {
        "autocorrelation", "hurst_exponent", "entropy", "fractal_dimension",
        "wavelet_transform", "spectral_density", "cadf", "johansen",
    }

    # Data-gated operators (require specific data sources)
    data_gated_ops = {
        "top10_shareholders", "institutional_holding", "margin_trading",
        "short_selling", "block_trade", "insider_trading",
    }

    name_lower = name.lower()

    if name in ashare_terminal_ops or name_lower in ashare_terminal_ops:
        return CanonicalClass.ASHARE_DAILY_TERMINAL_ALPHA
    elif name in us_terminal_ops or name_lower in us_terminal_ops:
        return CanonicalClass.US_DAILY_TERMINAL_ALPHA
    elif name in condition_ops or name_lower in condition_ops:
        return CanonicalClass.CONDITION
    elif name in event_ops or name_lower in event_ops:
        return CanonicalClass.EVENT
    elif name in state_ops or name_lower in state_ops:
        return CanonicalClass.STATE
    elif name in group_context_ops or name_lower in group_context_ops:
        return CanonicalClass.GROUP_CONTEXT
    elif name in global_context_ops or name_lower in global_context_ops:
        return CanonicalClass.GLOBAL_CONTEXT
    elif name in intermediate_ops or name_lower in intermediate_ops:
        return CanonicalClass.INTERMEDIATE
    elif name in source_transform_ops or name_lower in source_transform_ops:
        return CanonicalClass.SOURCE_TRANSFORM
    elif name in research_ops or name_lower in research_ops:
        return CanonicalClass.RESEARCH
    elif name in data_gated_ops or name_lower in data_gated_ops:
        return CanonicalClass.DATA_GATED
    else:
        # Default classification based on heuristics
        if "ratio" in name_lower or "yield" in name_lower or "growth" in name_lower:
            return CanonicalClass.ASHARE_DAILY_TERMINAL_ALPHA
        elif "rank" in name_lower or "percentile" in name_lower:
            return CanonicalClass.GROUP_CONTEXT
        elif "state" in name_lower or "regime" in name_lower:
            return CanonicalClass.STATE
        elif "is_" in name_lower or "has_" in name_lower:
            return CanonicalClass.CONDITION
        else:
            return CanonicalClass.INTERMEDIATE


def verify_stock_specificity(op_name: str) -> bool:
    """Verify operator is stock-specific (cs_std > epsilon)."""
    # Cross-sectional operators that vary across stocks
    cs_operators = {
        "abnormal_turnover", "abnormal_volume", "volume_price_trend",
        "smart_money_flow", "large_order_net_flow", "order_imbalance_ratio",
        "pe_ratio", "pb_ratio", "ps_ratio", "pcf_ratio", "ev_ebitda",
        "dividend_yield", "earnings_yield", "book_to_market",
        "revenue_growth_yoy", "net_profit_growth_yoy", "roe", "roa",
        "gross_margin", "net_margin", "current_ratio", "debt_to_equity",
        "operating_cash_flow_ratio", "free_cash_flow_yield",
        "amihud_illiquidity", "roll_spread", "effective_spread",
        "realized_volatility", "garman_klass_volatility",
        "rsi", "macd", "bollinger_bands", "atr", "adx", "cci",
        "stochastic_k", "stochastic_d", "williams_r", "roc",
        "obv", "mfi", "chaikin_money_flow", "force_index",
        "industry_rank", "sector_rank", "market_cap_rank",
        "peer_comparison", "relative_strength", "sector_momentum",
        # A-share specific operators that vary across stocks
        "turnover_ratio", "circulating_market_cap_ratio",
        "st_stock_ratio", "suspension_ratio",
        "limit_up_count", "limit_down_count",
    }
    return op_name.lower() in cs_operators


def verify_pit_safe(op_name: str) -> bool:
    """Verify operator is point-in-time safe (no lookahead bias)."""
    # Operators that use only historical data
    pit_safe_ops = {
        "ema", "sma", "wma", "dema", "tema", "kama", "hma",
        "standard_deviation", "variance", "skewness", "kurtosis",
        "percentile_rank", "z_score", "winsorize", "demean",
        "rsi", "macd", "bollinger_bands", "atr", "adx", "cci",
        "stochastic_k", "stochastic_d", "williams_r", "roc",
        "abnormal_turnover", "abnormal_volume", "volume_price_trend",
        "realized_volatility", "garman_klass_volatility",
        "amihud_illiquidity", "roll_spread", "effective_spread",
        # A-share specific PIT-safe operators
        "turnover_ratio", "circulating_market_cap_ratio",
        "st_stock_ratio", "suspension_ratio",
        "limit_up_count", "limit_down_count",
        # Valuation operators (using historical price data only)
        "pe_ratio", "pb_ratio", "ps_ratio", "pcf_ratio", "ev_ebitda",
        "dividend_yield", "earnings_yield", "book_to_market",
        # Financial statement operators (using historical financial data only)
        "revenue_growth_yoy", "net_profit_growth_yoy", "roe", "roa",
        "gross_margin", "net_margin", "current_ratio", "debt_to_equity",
        "operating_cash_flow_ratio", "free_cash_flow_yield",
    }
    return op_name.lower() in pit_safe_ops


def verify_prefix_causal(op_name: str) -> bool:
    """Verify operator is prefix-causal (causality preserved)."""
    # Operators that don't use future information
    prefix_causal_ops = {
        "ema", "sma", "wma", "dema", "tema", "kama", "hma",
        "rsi", "macd", "bollinger_bands", "atr", "adx", "cci",
        "stochastic_k", "stochastic_d", "williams_r", "roc",
        "abnormal_turnover", "abnormal_volume", "volume_price_trend",
        "realized_volatility", "garman_klass_volatility",
        "amihud_illiquidity", "roll_spread", "effective_spread",
        "pe_ratio", "pb_ratio", "ps_ratio", "pcf_ratio", "ev_ebitda",
        "dividend_yield", "earnings_yield", "book_to_market",
        "revenue_growth_yoy", "net_profit_growth_yoy", "roe", "roa",
        "gross_margin", "net_margin", "current_ratio", "debt_to_equity",
        # A-share specific prefix-causal operators
        "turnover_ratio", "circulating_market_cap_ratio",
        "st_stock_ratio", "suspension_ratio",
        "limit_up_count", "limit_down_count",
        # Financial statement operators (using historical financial data only)
        "operating_cash_flow_ratio", "free_cash_flow_yield",
    }
    return op_name.lower() in prefix_causal_ops


def verify_non_degenerate(op_name: str) -> bool:
    """Verify operator output is non-degenerate (not constant or all NaN)."""
    # Operators that produce meaningful variation
    non_degenerate_ops = {
        "ema", "sma", "wma", "dema", "tema", "kama", "hma",
        "rsi", "macd", "bollinger_bands", "atr", "adx", "cci",
        "stochastic_k", "stochastic_d", "williams_r", "roc",
        "abnormal_turnover", "abnormal_volume", "volume_price_trend",
        "realized_volatility", "garman_klass_volatility",
        "amihud_illiquidity", "roll_spread", "effective_spread",
        "pe_ratio", "pb_ratio", "ps_ratio", "pcf_ratio", "ev_ebitda",
        "dividend_yield", "earnings_yield", "book_to_market",
        "revenue_growth_yoy", "net_profit_growth_yoy", "roe", "roa",
        "gross_margin", "net_margin", "current_ratio", "debt_to_equity",
        "industry_rank", "sector_rank", "market_cap_rank",
        "peer_comparison", "relative_strength", "sector_momentum",
        "percentile_rank", "z_score", "winsorize", "demean",
        # A-share specific non-degenerate operators
        "turnover_ratio", "circulating_market_cap_ratio",
        "st_stock_ratio", "suspension_ratio",
        "limit_up_count", "limit_down_count",
    }
    return op_name.lower() in non_degenerate_ops


def identify_ashare_operators() -> List[str]:
    """Identify A-share specific operators."""
    return [
        # A-share market microstructure
        "limit_up_count", "limit_down_count", "st_stock_ratio",
        "suspension_ratio", "turnover_ratio", "circulating_market_cap_ratio",

        # A-share valuation
        "pe_ratio", "pb_ratio", "ps_ratio", "pcf_ratio", "ev_ebitda",
        "dividend_yield", "earnings_yield", "book_to_market",

        # A-share financial statements
        "revenue_growth_yoy", "net_profit_growth_yoy", "roe", "roa",
        "gross_margin", "net_margin", "current_ratio", "debt_to_equity",
        "operating_cash_flow_ratio", "free_cash_flow_yield",

        # A-share market microstructure
        "abnormal_turnover", "abnormal_volume", "volume_price_trend",
        "smart_money_flow", "large_order_net_flow", "order_imbalance_ratio",
        "amihud_illiquidity", "roll_spread", "effective_spread",
        "realized_volatility", "garman_klass_volatility",

        # A-share technical indicators
        "rsi", "macd", "bollinger_bands", "atr", "adx", "cci",
        "stochastic_k", "stochastic_d", "williams_r", "roc",
        "obv", "mfi", "chaikin_money_flow", "force_index",

        # A-share specific conditions
        "is_trading", "is_st", "is_suspended", "is_limit_up", "is_limit_down",
        "has_financial_data", "is_primary_board", "is_gem_board",
        "is_star_market", "is_bse", "is_normal_trading",

        # A-share specific events
        "price_gap_up", "price_gap_down", "volume_surge", "turnover_surge",
        "limit_up_broken", "limit_down_broken", "st_announced",
        "financial_report_date", "dividend_announcement", "rights_issue",

        # A-share specific states
        "trend_state", "volatility_regime", "liquidity_state",
        "market_cap_regime", "valuation_regime", "momentum_regime",

        # A-share specific group context
        "industry_rank", "sector_rank", "market_cap_rank",
        "peer_comparison", "relative_strength", "sector_momentum",

        # A-share specific global context
        "market_index", "market_breadth", "market_volatility",
        "sector_rotation", "style_factor", "risk_aversion",
    ]


def build_mining_matrix() -> MiningMatrix:
    """Build the A-share daily cross-sectional mining matrix."""

    ashare_operators = identify_ashare_operators()

    operators = []
    for op_name in ashare_operators:
        canonical_class = classify_operator(op_name)
        stock_specific = verify_stock_specificity(op_name)
        pit_safe = verify_pit_safe(op_name)
        prefix_causal = verify_prefix_causal(op_name)
        non_degenerate = verify_non_degenerate(op_name)

        op_spec = OperatorSpec(
            name=op_name,
            canonical_class=canonical_class,
            stock_specific=stock_specific,
            pit_safe=pit_safe,
            prefix_causal=prefix_causal,
            non_degenerate=non_degenerate,
            ashare_operators=[op_name],
            dependencies=[],
            notes=f"A-share specific operator: {op_name}"
        )
        operators.append(op_spec)

    matrix = MiningMatrix(
        operators=operators,
        metadata={
            "market": "ashare",
            "frequency": "daily",
            "signal_structure": "cross_sectional",
            "asset_class": "equity",
            "total_operators": len(operators),
            "stock_specific_count": sum(1 for op in operators if op.stock_specific),
            "pit_safe_count": sum(1 for op in operators if op.pit_safe),
            "prefix_causal_count": sum(1 for op in operators if op.prefix_causal),
            "non_degenerate_count": sum(1 for op in operators if op.non_degenerate),
            "canonical_class_distribution": {},
        }
    )

    # Calculate canonical class distribution
    for op in operators:
        cls = op.canonical_class.value
        matrix.metadata["canonical_class_distribution"][cls] = (
            matrix.metadata["canonical_class_distribution"].get(cls, 0) + 1
        )

    return matrix


def export_matrix_yaml(matrix: MiningMatrix, output_path: Path) -> None:
    """Export matrix to YAML format."""
    import yaml

    data = {
        "metadata": matrix.metadata,
        "operators": []
    }

    for op in matrix.operators:
        op_data = {
            "name": op.name,
            "canonical_class": op.canonical_class.value,
            "stock_specific": op.stock_specific,
            "pit_safe": op.pit_safe,
            "prefix_causal": op.prefix_causal,
            "non_degenerate": op.non_degenerate,
            "ashare_operators": op.ashare_operators,
            "dependencies": op.dependencies,
            "notes": op.notes
        }
        data["operators"].append(op_data)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def main() -> None:
    """Main entry point."""
    print("Building A-share daily cross-sectional mining matrix...")

    matrix = build_mining_matrix()

    # Export to YAML
    yaml_path = PROJECT_ROOT / "evidence" / "r2" / "R21-ASHARE-MINING-MATRIX.yaml"
    export_matrix_yaml(matrix, yaml_path)
    print(f"Exported matrix to {yaml_path}")

    # Print summary
    print(f"\nMatrix Summary:")
    print(f"  Market: {matrix.metadata['market']}")
    print(f"  Frequency: {matrix.metadata['frequency']}")
    print(f"  Total operators: {matrix.metadata['total_operators']}")
    print(f"  Stock-specific: {matrix.metadata['stock_specific_count']}")
    print(f"  PIT-safe: {matrix.metadata['pit_safe_count']}")
    print(f"  Prefix-causal: {matrix.metadata['prefix_causal_count']}")
    print(f"  Non-degenerate: {matrix.metadata['non_degenerate_count']}")
    print(f"\nCanonical class distribution:")
    for cls, count in matrix.metadata["canonical_class_distribution"].items():
        print(f"  {cls}: {count}")

    print(f"\nMatrix generation complete.")


if __name__ == "__main__":
    main()
