"""Portfolio, benchmark, and active exposure analysis using B and f."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .result import ExposureAnalysisResult
from .risk_model import RiskModelStore
from .weights import (
    extract_portfolio_weights,
    load_benchmark_weights,
    load_industry_name_map,
)


def _single_return_series(value: Any, name: str) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        if value.shape[1] != 1:
            raise ValueError(f"{name} 必须是 Series 或单列 DataFrame")
        value = value.iloc[:, 0]
    return pd.Series(value, copy=False).astype(float)


def _validate_coverage(
    coverage: pd.DataFrame,
    minimum: float,
    label: str,
) -> None:
    if not 0.0 <= float(minimum) <= 1.0:
        raise ValueError("最小覆盖率必须位于 [0, 1]")
    invested = coverage["total_abs_weight"] > 1e-12
    failures = coverage.loc[invested & (coverage["coverage_weight"] < float(minimum))]
    if not failures.empty:
        date = failures.index[0]
        value = failures.iloc[0]["coverage_weight"]
        raise ValueError(
            f"{label} {date.date()} 风险暴露覆盖率 {value:.4%} "
            f"低于门槛 {float(minimum):.4%}"
        )


def analyze_portfolio_exposure(
    pf: Any,
    *,
    risk_model_root: str | Path,
    benchmark_index: Optional[str] = None,
    benchmark_data_root: Optional[str | Path] = None,
    weight_source: str = "realized_close",
    missing_policy: str = "report_unknown",
    min_portfolio_coverage: float = 0.98,
    min_benchmark_coverage: float = 0.995,
    include_attribution: bool = True,
) -> ExposureAnalysisResult:
    """Analyze realized portfolio exposure without using factor covariance F/D."""
    store = RiskModelStore(risk_model_root)
    portfolio_weights = extract_portfolio_weights(pf, source=weight_source)
    portfolio_weights = portfolio_weights.loc[
        (portfolio_weights.index >= store.date_min)
        & (portfolio_weights.index <= store.date_max)
    ]
    if portfolio_weights.empty:
        raise ValueError(
            "回测区间与风险模型区间没有交集："
            f"{store.date_min.date()} ~ {store.date_max.date()}"
        )
    portfolio_agg = store.aggregate_exposure(
        portfolio_weights,
        missing_policy=missing_policy,
    )
    _validate_coverage(
        portfolio_agg.coverage,
        min_portfolio_coverage,
        "组合",
    )

    style = portfolio_agg.exposure.reindex(columns=store.style_factors)
    style_invested = portfolio_agg.invested_exposure.reindex(
        columns=store.style_factors
    )
    industry = portfolio_agg.exposure.reindex(columns=store.industry_factors)
    coverage = portfolio_agg.coverage.add_prefix("portfolio_")

    benchmark_weights = None
    benchmark_style = None
    benchmark_industry = None
    active_style = None
    active_industry = None
    benchmark_agg = None
    if benchmark_index:
        if benchmark_data_root is None:
            raise ValueError("启用基准暴露时必须提供 benchmark_data_root")
        benchmark_weights = load_benchmark_weights(
            benchmark_data_root,
            benchmark_index,
            portfolio_weights.index,
        )
        benchmark_agg = store.aggregate_exposure(
            benchmark_weights,
            missing_policy=missing_policy,
        )
        _validate_coverage(
            benchmark_agg.coverage,
            min_benchmark_coverage,
            "基准",
        )
        benchmark_style = benchmark_agg.exposure.reindex(columns=store.style_factors)
        benchmark_industry = benchmark_agg.exposure.reindex(
            columns=store.industry_factors
        )
        active_style = style - benchmark_style
        active_industry = industry - benchmark_industry
        coverage = coverage.join(
            benchmark_agg.coverage.add_prefix("benchmark_"),
            how="outer",
        )

    factor_returns = None
    factor_contribution = None
    active_factor_contribution = None
    unexplained = None
    active_unexplained = None
    if include_attribution:
        factor_returns = store.load_factor_returns(portfolio_weights.index)
        # f_t was fitted against contemporaneous B_t.  Use beginning holdings
        # w_(t-1) with B_t for an explicitly ex-post attribution.
        beginning_weights = portfolio_weights.shift(1).fillna(0.0)
        held_agg = store.aggregate_exposure(
            beginning_weights,
            missing_policy=missing_policy,
        )
        factor_contribution = held_agg.exposure * factor_returns
        portfolio_returns = _single_return_series(pf.returns(), "组合收益").reindex(
            portfolio_weights.index
        )
        unexplained = (
            portfolio_returns - factor_contribution.sum(axis=1)
        ).rename("unexplained_return")

        if benchmark_weights is not None:
            active_beginning = beginning_weights.subtract(
                benchmark_weights.shift(1).fillna(0.0),
                fill_value=0.0,
            )
            active_held_agg = store.aggregate_exposure(
                active_beginning,
                missing_policy=missing_policy,
            )
            active_factor_contribution = active_held_agg.exposure * factor_returns
            benchmark_returns = getattr(pf, "_qs_benchmark_returns", None)
            if benchmark_returns is not None:
                benchmark_returns = _single_return_series(
                    benchmark_returns,
                    "基准收益",
                ).reindex(portfolio_weights.index)
                active_return = portfolio_returns - benchmark_returns
                active_unexplained = (
                    active_return - active_factor_contribution.sum(axis=1)
                ).rename("active_unexplained_return")

    industry_names = {}
    if benchmark_data_root is not None:
        industry_names = load_industry_name_map(
            benchmark_data_root,
            portfolio_weights.index.max(),
        )
    result = ExposureAnalysisResult(
        realized_weights=portfolio_weights,
        portfolio_exposure=style,
        portfolio_exposure_invested=style_invested,
        portfolio_industry=industry,
        benchmark_exposure=benchmark_style,
        benchmark_industry=benchmark_industry,
        active_exposure=active_style,
        active_industry=active_industry,
        factor_returns=factor_returns,
        factor_contribution=factor_contribution,
        active_factor_contribution=active_factor_contribution,
        unexplained_return=unexplained,
        active_unexplained_return=active_unexplained,
        coverage=coverage,
        metadata={
            "risk_model_root": str(store.root),
            "risk_model_version": store.version,
            "weight_source": weight_source,
            "benchmark_index": benchmark_index,
            "missing_policy": missing_policy,
            "include_attribution": bool(include_attribution),
            "attribution_mode": (
                "experimental_ex_post_contemporaneous_B"
                if include_attribution
                else None
            ),
            "date_min": str(portfolio_weights.index.min().date()),
            "date_max": str(portfolio_weights.index.max().date()),
            "n_dates": len(portfolio_weights.index),
            "n_assets": len(portfolio_weights.columns),
            "style_factors": store.style_factors,
            "industry_factors": store.industry_factors,
            "industry_names": industry_names,
        },
    )
    pf._qs_exposure_analysis = result
    return result
