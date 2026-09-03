"""Single report boundary around :mod:`quant_evaluator`.

Report renderers consume this DTO and must not recompute statistics.  QE owns
IC, quantile assignment, portfolio returns, risk and performance semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.ic import compute_daily_ic, compute_mean_ic, ic_significance
from quant_evaluator.metrics.portfolio_stats import (
    compute_long_short_returns,
    compute_wealth_curve,
    compute_win_rate,
)
from quant_evaluator.metrics.probe_portfolio.sharpe import compute_portfolio_metrics
from quant_evaluator.metrics.quantile import compute_quantile_returns


@dataclass(frozen=True)
class ReportEvaluation:
    direction: int
    rank_ic_series: np.ndarray
    mean_rank_ic: float
    rank_ic_std: float
    rank_ic_ir: float
    rank_ic_win_rate: float
    quantile_returns: np.ndarray
    quantile_nav: np.ndarray
    long_short_returns: np.ndarray
    long_short_nav: np.ndarray
    sharpe: float
    annualized_return: float
    cumulative_return: float
    max_drawdown: float
    win_rate: float
    valid_return_periods: int


def _contracts(factors: np.ndarray, returns: np.ndarray) -> tuple[FactorBatch, LabelBundle]:
    factors = np.asarray(factors, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    if factors.ndim != 2 or returns.shape != factors.shape:
        raise ValueError("factors and forward_returns must be matching (time, asset) arrays")
    t, n = factors.shape
    decision = tuple(range(t))
    fb = FactorBatch(
        factor_ids=("report_factor",),
        time_axis=AxisRef("time", "int64", t, np.arange(t)),
        asset_axis=AxisRef("asset", "int64", n, np.arange(n)),
        values=factors[:, :, None],
        validity=np.isfinite(factors)[:, :, None],
    )
    lb = LabelBundle(
        target_id="report_forward_vwap_return",
        values=returns,
        horizon=1,
        decision_time=decision,
        label_start_time=tuple(x + 1 for x in decision),
        label_end_time=tuple(x + 2 for x in decision),
        validity=np.isfinite(returns),
        price_convention="vwap_to_vwap",
    )
    return fb, lb


def evaluate_report_arrays(
    factors: np.ndarray,
    forward_returns: np.ndarray,
    *,
    n_quantiles: int = 10,
    min_assets: int = 10,
    min_ic_periods: int = 20,
    direction_training_periods: int | None = None,
    periods_per_year: int = 252,
) -> ReportEvaluation:
    """Evaluate one factor using only canonical quant_evaluator functions."""
    fb, lb = _contracts(factors, forward_returns)
    raw_ic, _ = compute_daily_ic(fb, lb, method="spearman", min_assets=min_assets)
    raw_ic = raw_ic[:, 0]
    train = raw_ic[:direction_training_periods] if direction_training_periods else raw_ic
    train_finite = train[np.isfinite(train)]
    train_mean, _ = compute_mean_ic(train[:, None], min_periods=min_ic_periods)
    direction = -1 if np.isfinite(train_mean[0]) and float(train_mean[0]) < 0 else 1

    directed_factors = np.asarray(factors, dtype=np.float64) * direction
    fb, lb = _contracts(directed_factors, forward_returns)
    ic_matrix, _ = compute_daily_ic(fb, lb, method="spearman", min_assets=min_assets)
    ic = ic_matrix[:, 0]
    ir, _, _ = ic_significance(ic, min_periods=min_ic_periods)
    finite_ic = ic[np.isfinite(ic)]

    quantile, _ = compute_quantile_returns(
        fb, lb, n_quantiles=n_quantiles, min_assets=min_assets
    )
    quantile = quantile[:, :, 0]
    _, _, ls = compute_long_short_returns(
        directed_factors,
        np.asarray(forward_returns, dtype=np.float64),
        long_threshold=(n_quantiles - 1) / n_quantiles,
        short_threshold=1 / n_quantiles,
        validity_mask=np.isfinite(directed_factors),
        missing_return_policy="drop",
    )
    clean_ls = ls[np.isfinite(ls)]
    perf = compute_portfolio_metrics(
        clean_ls, periods_per_year=periods_per_year, min_periods=min_ic_periods
    )
    q_nav = np.column_stack([
        compute_wealth_curve(quantile[:, q], missing_return_policy="drop")
        for q in range(n_quantiles)
    ])
    ls_nav = compute_wealth_curve(ls, missing_return_policy="drop")
    mean_ic, ic_std = compute_mean_ic(ic[:, None], min_periods=min_ic_periods)

    return ReportEvaluation(
        direction=direction,
        rank_ic_series=ic,
        mean_rank_ic=float(mean_ic[0]),
        rank_ic_std=float(ic_std[0]),
        rank_ic_ir=float(ir),
        rank_ic_win_rate=float(compute_win_rate(ic)),
        quantile_returns=quantile,
        quantile_nav=q_nav,
        long_short_returns=ls,
        long_short_nav=ls_nav,
        sharpe=float(perf["sharpe"]),
        annualized_return=float(perf["annualized_return"]),
        cumulative_return=float(ls_nav[-1] - 1.0) if ls_nav.size else np.nan,
        max_drawdown=float(perf["max_drawdown"]),
        win_rate=float(perf["win_rate"]),
        valid_return_periods=int(clean_ls.size),
    )
