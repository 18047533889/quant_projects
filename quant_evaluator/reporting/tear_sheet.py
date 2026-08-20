"""
Tear sheet generation for quant_evaluator.

Generates a 22-panel tear sheet with comprehensive factor evaluation visualizations.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from quant_evaluator.reporting.chart_spec import ChartSpec


@dataclass
class EvaluationResult:
    """Container for factor evaluation results.

    This is a simplified version for tear sheet generation.
    In production, this would be the full evaluation result object.
    """
    # IC metrics
    ic_series: Optional[np.ndarray] = None
    rank_ic_series: Optional[np.ndarray] = None
    ic_mean: float = 0.0
    ic_std: float = 0.0
    icir: float = 0.0

    # Quantile metrics
    quantile_returns: Optional[np.ndarray] = None
    quantile_spread: Optional[np.ndarray] = None
    quantile_names: Optional[List[str]] = None

    # Portfolio metrics
    long_short_returns: Optional[np.ndarray] = None
    cumulative_returns: Optional[np.ndarray] = None
    drawdown_series: Optional[np.ndarray] = None
    max_drawdown: float = 0.0
    drawdown_durations: Optional[np.ndarray] = None

    # Turnover metrics
    turnover_series: Optional[np.ndarray] = None
    avg_turnover: float = 0.0

    # Coverage metrics
    coverage_series: Optional[np.ndarray] = None
    avg_coverage: float = 0.0

    # Factor correlation
    factor_correlation: Optional[np.ndarray] = None
    factor_names: Optional[List[str]] = None

    # Distribution metrics
    skewness: float = 0.0
    kurtosis: float = 0.0
    variance: float = 0.0
    cvar: float = 0.0

    # Performance metrics
    sharpe_ratio: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # HHI (Herfindahl-Hirschman Index)
    hhi: Optional[np.ndarray] = None

    # IC stability
    ic_stability: Optional[np.ndarray] = None

    # Rolling IC
    rolling_ic_20: Optional[np.ndarray] = None
    rolling_ic_60: Optional[np.ndarray] = None
    rolling_ic_120: Optional[np.ndarray] = None

    # IC decay
    ic_decay: Optional[np.ndarray] = None

    # IC autocorrelation
    ic_autocorrelation: Optional[np.ndarray] = None

    # Coverage heatmap
    coverage_heatmap: Optional[np.ndarray] = None

    # Performance attribution
    attribution: Optional[Dict[str, float]] = None


def generate_tear_sheet(
    evaluation_result: EvaluationResult,
) -> Dict[str, ChartSpec]:
    """Generate a 22-panel tear sheet from evaluation results.

    Args:
        evaluation_result: EvaluationResult containing all metrics

    Returns:
        Dictionary mapping panel names to ChartSpec objects
    """
    panels = {}

    # Panel 1: IC time series
    if evaluation_result.ic_series is not None:
        panels["ic_time_series"] = ChartSpec(
            title="IC Time Series",
            x_label="Time",
            y_label="IC",
            chart_type="line",
            data={
                "timestamps": list(range(len(evaluation_result.ic_series))),
                "series": {"IC": evaluation_result.ic_series.tolist()}
            }
        )

    # Panel 2: IC distribution
    if evaluation_result.ic_series is not None:
        panels["ic_distribution"] = ChartSpec(
            title="IC Distribution",
            x_label="IC Value",
            y_label="Frequency",
            chart_type="bar",
            data={
                "values": evaluation_result.ic_series.tolist(),
                "bins": 30
            }
        )

    # Panel 3: Rank IC time series
    if evaluation_result.rank_ic_series is not None:
        panels["rank_ic_time_series"] = ChartSpec(
            title="Rank IC Time Series",
            x_label="Time",
            y_label="Rank IC",
            chart_type="line",
            data={
                "timestamps": list(range(len(evaluation_result.rank_ic_series))),
                "series": {"Rank IC": evaluation_result.rank_ic_series.tolist()}
            }
        )

    # Panel 4: Quantile returns bar chart
    if evaluation_result.quantile_returns is not None:
        avg_quantile_returns = np.nanmean(evaluation_result.quantile_returns, axis=0)
        # If avg_quantile_returns is a scalar (single quantile), convert to array
        if np.isscalar(avg_quantile_returns):
            avg_quantile_returns = np.array([avg_quantile_returns])
        quantile_names = evaluation_result.quantile_names or [f"Q{i+1}" for i in range(len(avg_quantile_returns))]
        panels["quantile_returns_bar"] = ChartSpec(
            title="Quantile Returns",
            x_label="Quantile",
            y_label="Average Return",
            chart_type="bar",
            data={
                "categories": quantile_names,
                "values": avg_quantile_returns.tolist()
            }
        )

    # Panel 5: Quantile spread time series
    if evaluation_result.quantile_spread is not None:
        panels["quantile_spread_time_series"] = ChartSpec(
            title="Quantile Spread Time Series",
            x_label="Time",
            y_label="Spread",
            chart_type="line",
            data={
                "timestamps": list(range(len(evaluation_result.quantile_spread))),
                "series": {"Spread": evaluation_result.quantile_spread.tolist()}
            }
        )

    # Panel 6: Drawdown curve
    if evaluation_result.drawdown_series is not None:
        panels["drawdown_curve"] = ChartSpec(
            title="Drawdown Curve",
            x_label="Time",
            y_label="Drawdown",
            chart_type="line",
            data={
                "timestamps": list(range(len(evaluation_result.drawdown_series))),
                "series": {"Drawdown": evaluation_result.drawdown_series.tolist()}
            }
        )

    # Panel 7: Drawdown duration histogram
    if evaluation_result.drawdown_durations is not None and len(evaluation_result.drawdown_durations) > 0:
        panels["drawdown_duration_histogram"] = ChartSpec(
            title="Drawdown Duration Histogram",
            x_label="Duration (periods)",
            y_label="Frequency",
            chart_type="bar",
            data={
                "values": evaluation_result.drawdown_durations.tolist(),
                "bins": 20
            }
        )

    # Panel 8: Turnover time series
    if evaluation_result.turnover_series is not None:
        panels["turnover_time_series"] = ChartSpec(
            title="Turnover Time Series",
            x_label="Time",
            y_label="Turnover",
            chart_type="line",
            data={
                "timestamps": list(range(len(evaluation_result.turnover_series))),
                "series": {"Turnover": evaluation_result.turnover_series.tolist()}
            }
        )

    # Panel 9: Coverage time series
    if evaluation_result.coverage_series is not None:
        panels["coverage_time_series"] = ChartSpec(
            title="Coverage Time Series",
            x_label="Time",
            y_label="Coverage",
            chart_type="line",
            data={
                "timestamps": list(range(len(evaluation_result.coverage_series))),
                "series": {"Coverage": evaluation_result.coverage_series.tolist()}
            }
        )

    # Panel 10: Factor correlation heatmap
    if evaluation_result.factor_correlation is not None:
        panels["factor_correlation_heatmap"] = ChartSpec(
            title="Factor Correlation Heatmap",
            x_label="Factor",
            y_label="Factor",
            chart_type="heatmap",
            data={
                "matrix": evaluation_result.factor_correlation.tolist(),
                "labels": evaluation_result.factor_names or []
            }
        )

    # Panel 11: HHI bar chart
    if evaluation_result.hhi is not None:
        panels["hhi_bar_chart"] = ChartSpec(
            title="HHI (Herfindahl-Hirschman Index)",
            x_label="Factor",
            y_label="HHI",
            chart_type="bar",
            data={
                "categories": evaluation_result.factor_names or [f"F{i}" for i in range(len(evaluation_result.hhi))],
                "values": evaluation_result.hhi.tolist()
            }
        )

    # Panel 12: IC stability scatter
    if evaluation_result.ic_stability is not None:
        panels["ic_stability_scatter"] = ChartSpec(
            title="IC Stability",
            x_label="Period",
            y_label="IC",
            chart_type="scatter",
            data={
                "x": list(range(len(evaluation_result.ic_stability))),
                "y": evaluation_result.ic_stability.tolist()
            }
        )

    # Panel 13: Rolling IC (20/60/120 periods)
    rolling_ic_data = {}
    if evaluation_result.rolling_ic_20 is not None:
        rolling_ic_data["Rolling IC (20)"] = evaluation_result.rolling_ic_20.tolist()
    if evaluation_result.rolling_ic_60 is not None:
        rolling_ic_data["Rolling IC (60)"] = evaluation_result.rolling_ic_60.tolist()
    if evaluation_result.rolling_ic_120 is not None:
        rolling_ic_data["Rolling IC (120)"] = evaluation_result.rolling_ic_120.tolist()

    if rolling_ic_data:
        max_len = max(len(v) for v in rolling_ic_data.values())
        panels["rolling_ic"] = ChartSpec(
            title="Rolling IC",
            x_label="Time",
            y_label="IC",
            chart_type="line",
            data={
                "timestamps": list(range(max_len)),
                "series": rolling_ic_data
            }
        )

    # Panel 14: IC decay curve
    if evaluation_result.ic_decay is not None:
        panels["ic_decay_curve"] = ChartSpec(
            title="IC Decay Curve",
            x_label="Lag",
            y_label="IC",
            chart_type="line",
            data={
                "lags": list(range(len(evaluation_result.ic_decay))),
                "series": {"IC Decay": evaluation_result.ic_decay.tolist()}
            }
        )

    # Panel 15: Variance / CVaR bar chart
    panels["variance_cvar_bar"] = ChartSpec(
        title="Variance / CVaR",
        x_label="Metric",
        y_label="Value",
        chart_type="bar",
        data={
            "categories": ["Variance", "CVaR"],
            "values": [evaluation_result.variance, evaluation_result.cvar]
        }
    )

    # Panel 16: Skewness / Kurtosis bar chart
    panels["skewness_kurtosis_bar"] = ChartSpec(
        title="Skewness / Kurtosis",
        x_label="Metric",
        y_label="Value",
        chart_type="bar",
        data={
            "categories": ["Skewness", "Kurtosis"],
            "values": [evaluation_result.skewness, evaluation_result.kurtosis]
        }
    )

    # Panel 17: Summary statistics table
    summary_stats = {
        "IC Mean": evaluation_result.ic_mean,
        "IC Std": evaluation_result.ic_std,
        "ICIR": evaluation_result.icir,
        "Sharpe Ratio": evaluation_result.sharpe_ratio,
        "Annual Return": evaluation_result.annual_return,
        "Annual Volatility": evaluation_result.annual_volatility,
        "Max Drawdown": evaluation_result.max_drawdown,
        "Sortino Ratio": evaluation_result.sortino_ratio,
        "Calmar Ratio": evaluation_result.calmar_ratio,
        "Avg Turnover": evaluation_result.avg_turnover,
        "Avg Coverage": evaluation_result.avg_coverage,
        "Skewness": evaluation_result.skewness,
        "Kurtosis": evaluation_result.kurtosis,
    }
    panels["summary_statistics_table"] = ChartSpec(
        title="Summary Statistics",
        x_label="Metric",
        y_label="Value",
        chart_type="bar",
        data={"statistics": summary_stats}
    )

    # Panel 18: Risk-return scatter
    panels["risk_return_scatter"] = ChartSpec(
        title="Risk-Return Scatter",
        x_label="Volatility",
        y_label="Return",
        chart_type="scatter",
        data={
            "x": [evaluation_result.annual_volatility],
            "y": [evaluation_result.annual_return],
            "labels": ["Factor"]
        }
    )

    # Panel 19: Coverage heatmap
    if evaluation_result.coverage_heatmap is not None:
        panels["coverage_heatmap"] = ChartSpec(
            title="Coverage Heatmap",
            x_label="Asset",
            y_label="Time",
            chart_type="heatmap",
            data={
                "matrix": evaluation_result.coverage_heatmap.tolist()
            }
        )

    # Panel 20: Turnover histogram
    if evaluation_result.turnover_series is not None:
        panels["turnover_histogram"] = ChartSpec(
            title="Turnover Distribution",
            x_label="Turnover",
            y_label="Frequency",
            chart_type="bar",
            data={
                "values": evaluation_result.turnover_series.tolist(),
                "bins": 30
            }
        )

    # Panel 21: IC autocorrelation
    if evaluation_result.ic_autocorrelation is not None:
        panels["ic_autocorrelation"] = ChartSpec(
            title="IC Autocorrelation",
            x_label="Lag",
            y_label="Autocorrelation",
            chart_type="bar",
            data={
                "lags": list(range(len(evaluation_result.ic_autocorrelation))),
                "values": evaluation_result.ic_autocorrelation.tolist()
            }
        )

    # Panel 22: Performance attribution pie
    if evaluation_result.attribution:
        panels["performance_attribution_pie"] = ChartSpec(
            title="Performance Attribution",
            x_label="Component",
            y_label="Contribution",
            chart_type="bar",  # Using bar as pie chart approximation
            data={
                "categories": list(evaluation_result.attribution.keys()),
                "values": list(evaluation_result.attribution.values())
            }
        )

    return panels
