"""20 日 cohort 组合构建与真实 daily PnL 指标层。

核心入口 :func:`compute_cohort_pnl` 构造真实 cohort portfolio（每日新建
1/H 资本、H=20 日的 cohort，由真实 daily PnL 计算风险收益指标），
替代被规范 §7/§13 禁止的 overlapping-label 直接年化做法。

公共 API（全部可独立导入、可测试）：
- compute_cohort_pnl           : 真实 cohort portfolio → 每日真实 PnL
- compute_portfolio_metrics     : 真实 daily PnL 指标全家桶（Sharpe/Sortino/
                                  Calmar/MaxDD/DD Duration/TimeUnderWater/
                                  Rolling Sharpe Q20/Positive Month Ratio）
- build_cohort_panels           : 每日分位 → D1/D10 桶成员（复用 QE 库分位模块）
- build_quantile_masks          : 每日分位 → 全分位成员矩阵
- compute_drawdown_persistence  : 0.5*norm(MaxDDDuration)+0.5*norm(TimeUnderWater)
- compute_rolling_sharpe_quantile / compute_positive_month_ratio
- compute_overlapping_forward_returns / annualize_overlapping_label_sharpe
  （仅用于对照「错误算法」虚高效应的单测，禁止用于生产指标链路）
- compute_metrics_from_cohort   : 一步到位：cohort PnL → 指标 dict

注册建议（不在此处动 registry/metrics.py，交给 V2-A）：
- metric_id: probe_cohort_ls_sharpe / probe_cohort_ls_sortino /
  probe_cohort_ls_calmar / probe_cohort_ls_maxdd /
  probe_cohort_ls_drawdown_persistence / probe_cohort_ls_roll_sharpe_q20 /
  probe_cohort_ls_pos_month_ratio / probe_cohort_ls_pnl_net 等
- status: EXPERIMENTAL（指标层全新、尚未在真实数据上全窗验收）
- 建议先以 Diagnostic 身份接入 FactorFitness，35% 权重维度落地由
  Long-Short 真实 PnL 驱动（见 V2 总任务书）。
"""

from __future__ import annotations

from quant_evaluator.metrics.probe_portfolio._core import (
    build_cohort_panels,
    build_quantile_masks,
    compute_cohort_pnl,
    compute_overlapping_forward_returns,
    annualize_overlapping_label_sharpe,
)
from quant_evaluator.metrics.probe_portfolio.sharpe import (
    compute_portfolio_metrics,
    compute_drawdown_persistence,
    compute_rolling_sharpe_quantile,
    compute_positive_month_ratio,
    compute_annualized_return,
    compute_annualized_volatility,
    compute_active_metrics,
)
from quant_evaluator.metrics.probe_portfolio.costs import (
    COST_SCENARIOS,
    cost_scenario,
    apply_per_side_costs,
    CostEvidence,
    compute_transaction_costs,
    heuristic_cost_scan,
)

__all__ = [
    "build_cohort_panels",
    "build_quantile_masks",
    "compute_cohort_pnl",
    "compute_overlapping_forward_returns",
    "annualize_overlapping_label_sharpe",
    "compute_portfolio_metrics",
    "compute_drawdown_persistence",
    "compute_rolling_sharpe_quantile",
    "compute_positive_month_ratio",
    "compute_annualized_return",
    "compute_annualized_volatility",
    "compute_active_metrics",
    "COST_SCENARIOS",
    "cost_scenario",
    "apply_per_side_costs",
    "CostEvidence",
    "compute_transaction_costs",
    "heuristic_cost_scan",
    "compute_metrics_from_cohort",
    # Legacy flat-module re-exports (the package moved to a directory layout;
    # keep the historical names importable from the metrics package).
    "construct_long_short_portfolio",
    "compute_equal_weighted_returns",
    "compute_cap_weighted_returns",
    "compute_long_short_equal_weighted",
    "compute_long_short_cap_weighted",
    "compute_portfolio_weights",
    "compute_portfolio_concentration",
    "compute_turnover_from_positions",
]


def __getattr__(name: str):
    """Lazy shim for legacy names from the historical flat module.

    ``metrics/__init__.py`` imports ``construct_long_short_portfolio`` et al.
    from ``quant_evaluator.metrics.probe_portfolio``.  The old flat module
    (``metrics/probe_portfolio.py``) is still present on disk and provides
    these names; importing it directly would recurse into ``metrics/__init__``
    when it imports its own ``portfolio_stats`` siblings.  Load it through the
    already-imported ``metrics`` package (whose namespace is fully populated
    by the time this shim runs) and return the requested attribute.
    """
    if name not in {
        "construct_long_short_portfolio",
        "compute_equal_weighted_returns",
        "compute_cap_weighted_returns",
        "compute_long_short_equal_weighted",
        "compute_long_short_cap_weighted",
        "compute_portfolio_weights",
        "compute_portfolio_concentration",
        "compute_turnover_from_positions",
    }:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    import quant_evaluator.metrics as _metrics_pkg

    mod = importlib.import_module("quant_evaluator.metrics.probe_portfolio_legacy")
    return getattr(mod, name)


def compute_metrics_from_cohort(
    factor_values,
    next_ret,
    next_vwap,
    next_open=None,
    weight_matrix=None,
    n_quantiles: int = 10,
    holding: int = 20,
    long_weight: float = 0.5,
    short_weight: float = -0.5,
    per_side_cost: float = 0.0,
    min_bucket_size: int = 1,
    require_tradable: bool = True,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    min_periods: int = 20,
    rolling_window: int = 60,
    rolling_quantile: float = 0.20,
) -> dict:
    """一步到位：真实 cohort 组合 PnL → 全部指标。

    Args:
        前 10 个参数传递给 compute_cohort_pnl。
        后 5 个参数传递给 compute_portfolio_metrics。

    Returns:
        dict：{"cohort": {...pnl 序列...}, "metrics": {...指标...}}，
        其中 metrics 含 long_short 与 long_only_active 两套口径。
    """
    cohort = compute_cohort_pnl(
        factor_values=factor_values,
        next_ret=next_ret,
        next_vwap=next_vwap,
        next_open=next_open,
        weight_matrix=weight_matrix,
        n_quantiles=n_quantiles,
        holding=holding,
        long_weight=long_weight,
        short_weight=short_weight,
        per_side_cost=per_side_cost,
        min_bucket_size=min_bucket_size,
        require_tradable=require_tradable,
    )
    ls_metrics = compute_portfolio_metrics(
        cohort["pnl_net"],
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
        min_periods=min_periods,
        rolling_window=rolling_window,
        rolling_quantile=rolling_quantile,
        holding_days=holding,
    )
    # This legacy producer supplies only an arithmetic spread, not separate
    # long-only and benchmark NAV trajectories. It cannot support geometric
    # LO performance. Call sharpe.compute_active_metrics with both trajectories.
    loa_metrics = {
        "status": "WAIT",
        "reason": "separate portfolio and benchmark NAV return trajectories required",
        "return_basis": "arithmetic_active_diagnostic",
    }
    return {
        "cohort": cohort,
        "metrics": {
            "long_short": ls_metrics,
            "long_only_active": loa_metrics,
        },
    }
