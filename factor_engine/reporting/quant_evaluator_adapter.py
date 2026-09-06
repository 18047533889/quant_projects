"""Single report boundary around :mod:`quant_evaluator`.

Report renderers consume this DTO and must not recompute statistics.  QE owns
IC, quantile assignment, portfolio returns, risk and performance semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.ic import compute_daily_ic, compute_mean_ic, ic_significance
from quant_evaluator.metrics.portfolio_stats import (
    compute_long_short_returns,
    compute_aligned_wealth_curve,
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
    long_short_nav_aligned: np.ndarray
    sharpe: float
    annualized_return: float
    top_quantile_annualized_return: float
    bottom_quantile_annualized_return: float
    cumulative_return: float
    max_drawdown: float
    win_rate: float
    valid_return_periods: int


@dataclass(frozen=True)
class ReportBatchEvaluation:
    factors: Mapping[str, ReportEvaluation]
    backend_used: str
    backend_fallback_reason: str | None = None


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


def _batch_contracts(
    factors: np.ndarray,
    returns: np.ndarray,
    factor_ids: tuple[str, ...],
) -> tuple[FactorBatch, LabelBundle]:
    factors = np.asarray(factors, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    if factors.ndim != 3 or factors.shape[:2] != returns.shape:
        raise ValueError("factors must be (time, asset, factor) and labels (time, asset)")
    if factors.shape[2] != len(factor_ids) or len(set(factor_ids)) != len(factor_ids):
        raise ValueError("factor_ids must be unique and match the factor dimension")
    t, n, _ = factors.shape
    decision = tuple(range(t))
    return (
        FactorBatch(
            factor_ids=factor_ids,
            time_axis=AxisRef("time", "int64", t, np.arange(t)),
            asset_axis=AxisRef("asset", "int64", n, np.arange(n)),
            values=factors,
            validity=np.isfinite(factors),
        ),
        LabelBundle(
            target_id="report_forward_vwap_return",
            values=returns,
            horizon=1,
            decision_time=decision,
            label_start_time=tuple(x + 1 for x in decision),
            label_end_time=tuple(x + 2 for x in decision),
            validity=np.isfinite(returns),
            price_convention="vwap_to_vwap",
        ),
    )


def evaluate_report_batch(
    factors: np.ndarray,
    forward_returns: np.ndarray,
    *,
    factor_ids: tuple[str, ...],
    backend: str = "cpu",
    n_quantiles: int = 10,
    min_assets: int = 10,
    min_ic_periods: int = 20,
    direction_training_periods: int | None = None,
    periods_per_year: int = 252,
    fixed_directions: tuple[int, ...] | None = None,
) -> ReportBatchEvaluation:
    """Evaluate a bounded factor tile with shared QE IC/quantile intermediates."""
    if backend not in {"cpu", "auto", "cuda", "cuda_strict"}:
        raise ValueError("backend must be cpu, auto, cuda, or cuda_strict")
    fb, lb = _batch_contracts(factors, forward_returns, factor_ids)
    fallback_reason = None
    backend_used = "cpu"
    use_cuda = backend in {"auto", "cuda", "cuda_strict"}
    if use_cuda and min_assets != 20:
        message = "QE CUDA rank_ic currently has canonical min_assets=20"
        if backend == "cuda_strict":
            raise ValueError(message)
        fallback_reason = message
        use_cuda = False
    if use_cuda:
        try:
            from quant_evaluator.runtime.evaluator import evaluate

            gpu = evaluate(
                fb,
                lb,
                metrics=("rank_ic_series", "quantile_returns_full"),
                backend="cuda_strict" if backend == "cuda_strict" else "cuda",
            )
            raw_ic = np.asarray(gpu.series_metrics["rank_ic_series"], dtype=np.float64)
            quantile = np.asarray(gpu.vector_metrics["quantile_returns"], dtype=np.float64)
            backend_used = "cuda"
        except Exception as exc:
            if backend == "cuda_strict":
                raise
            fallback_reason = f"{type(exc).__name__}: {exc}"
            use_cuda = False
    if not use_cuda:
        raw_ic, _ = compute_daily_ic(fb, lb, method="spearman", min_assets=min_assets)
        quantile, _ = compute_quantile_returns(
            fb, lb, n_quantiles=n_quantiles, min_assets=min_assets
        )

    train = raw_ic[:direction_training_periods] if direction_training_periods else raw_ic
    train_mean, _ = compute_mean_ic(train, min_periods=min_ic_periods)
    directions = np.where(np.isfinite(train_mean) & (train_mean < 0), -1, 1)
    if fixed_directions is not None:
        if len(fixed_directions) != len(factor_ids) or any(
            isinstance(value, bool) or value not in (-1, 1) for value in fixed_directions
        ):
            raise ValueError("fixed_directions must contain one +/-1 per factor")
        directions = np.asarray(fixed_directions, dtype=int)
    directed_ic = raw_ic * directions[None, :]

    factor_results = {}
    for index, factor_id in enumerate(factor_ids):
        factor_quantiles = quantile[:, :, index]
        if directions[index] < 0:
            factor_quantiles = factor_quantiles[:, ::-1]
        ls = factor_quantiles[:, -1] - factor_quantiles[:, 0]
        clean_ls = ls[np.isfinite(ls)]
        performance = compute_portfolio_metrics(
            clean_ls,
            periods_per_year=periods_per_year,
            min_periods=min_ic_periods,
        )
        top_performance = compute_portfolio_metrics(
            factor_quantiles[:, -1][np.isfinite(factor_quantiles[:, -1])],
            periods_per_year=periods_per_year,
            min_periods=min_ic_periods,
        )
        bottom_performance = compute_portfolio_metrics(
            factor_quantiles[:, 0][np.isfinite(factor_quantiles[:, 0])],
            periods_per_year=periods_per_year,
            min_periods=min_ic_periods,
        )
        ic = directed_ic[:, index]
        mean_ic, ic_std = compute_mean_ic(ic[:, None], min_periods=min_ic_periods)
        ir = ic_significance(ic, min_periods=min_ic_periods)[0]
        quantile_nav = np.column_stack([
            compute_aligned_wealth_curve(factor_quantiles[:, group])
            for group in range(n_quantiles)
        ])
        ls_nav = compute_wealth_curve(ls, missing_return_policy="drop")
        ls_nav_aligned = compute_aligned_wealth_curve(ls)
        factor_results[factor_id] = ReportEvaluation(
            direction=int(directions[index]),
            rank_ic_series=ic,
            mean_rank_ic=float(mean_ic[0]),
            rank_ic_std=float(ic_std[0]),
            rank_ic_ir=float(ir),
            rank_ic_win_rate=float(compute_win_rate(ic)),
            quantile_returns=factor_quantiles,
            quantile_nav=quantile_nav,
            long_short_returns=ls,
            long_short_nav=ls_nav,
            long_short_nav_aligned=ls_nav_aligned,
            sharpe=float(performance["sharpe"]),
            annualized_return=float(performance["annualized_return"]),
            top_quantile_annualized_return=float(top_performance["annualized_return"]),
            bottom_quantile_annualized_return=float(bottom_performance["annualized_return"]),
            cumulative_return=float(ls_nav[-1] - 1.0) if ls_nav.size else np.nan,
            max_drawdown=float(performance["max_drawdown"]),
            win_rate=float(performance["win_rate"]),
            valid_return_periods=int(clean_ls.size),
        )
    return ReportBatchEvaluation(
        factors=factor_results,
        backend_used=backend_used,
        backend_fallback_reason=fallback_reason,
    )


def evaluate_report_arrays(
    factors: np.ndarray,
    forward_returns: np.ndarray,
    *,
    n_quantiles: int = 10,
    min_assets: int = 10,
    min_ic_periods: int = 20,
    direction_training_periods: int | None = None,
    periods_per_year: int = 252,
    fixed_direction: int | None = None,
) -> ReportEvaluation:
    """Evaluate one factor through the same bounded batch authority."""
    values = np.asarray(factors, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("factors must be a (time, asset) array")
    batch = evaluate_report_batch(
        values[:, :, None],
        forward_returns,
        factor_ids=("report_factor",),
        backend="cpu",
        n_quantiles=n_quantiles,
        min_assets=min_assets,
        min_ic_periods=min_ic_periods,
        direction_training_periods=direction_training_periods,
        periods_per_year=periods_per_year,
        fixed_directions=None if fixed_direction is None else (fixed_direction,),
    )
    return batch.factors["report_factor"]
