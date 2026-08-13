"""
Risk metrics module for quantitative portfolio analysis.

This module provides comprehensive risk measurement tools including:
- VaR and CVaR computation (historical, parametric, Cornish-Fisher)
- Drawdown analysis and statistics
- Tail risk measures
- Stress testing and scenario analysis
"""

from .var_cvar import (
    compute_var_historical,
    compute_var_parametric,
    compute_var_cornish_fisher,
    compute_cvar,
)

from .drawdown_analysis import (
    compute_drawdown_series,
    compute_drawdown_statistics,
    identify_drawdown_periods,
    compute_drawdown_duration,
    compute_ulcer_index,
    compute_pain_index,
)

from .tail_risk import (
    compute_tail_ratio,
    compute_gain_loss_ratio,
    compute_upside_potential_ratio,
    compute_omega_ratio,
    compute_expected_shortfall_ratio,
    compute_tail_dependence,
)

from .stress_testing import (
    apply_historical_scenario,
    apply_hypothetical_scenario,
    compute_scenario_impact,
    compute_correlation_breakdown,
    compute_worst_case_scenarios,
)

__all__ = [
    # VaR/CVaR
    "compute_var_historical",
    "compute_var_parametric",
    "compute_var_cornish_fisher",
    "compute_cvar",
    # Drawdown analysis
    "compute_drawdown_series",
    "compute_drawdown_statistics",
    "identify_drawdown_periods",
    "compute_drawdown_duration",
    "compute_ulcer_index",
    "compute_pain_index",
    # Tail risk
    "compute_tail_ratio",
    "compute_gain_loss_ratio",
    "compute_upside_potential_ratio",
    "compute_omega_ratio",
    "compute_expected_shortfall_ratio",
    "compute_tail_dependence",
    # Stress testing
    "apply_historical_scenario",
    "apply_hypothetical_scenario",
    "compute_scenario_impact",
    "compute_correlation_breakdown",
    "compute_worst_case_scenarios",
]
