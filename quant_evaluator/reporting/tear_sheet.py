"""
Tear sheet generation for quant_evaluator.

Generates the full institutional 22-panel tear sheet with comprehensive factor
evaluation visualizations.

Every institutional panel consumes a pre-computed ``MetricArtifact``; the
reporting layer never recomputes metrics. When the source artifact is absent a
panel renders a ``NOT_COMPUTED`` placeholder - never a 0.0.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from quant_evaluator.reporting.chart_spec import ChartSpec


# ---------------------------------------------------------------------------
# MetricArtifact - pre-computed metric data bundle
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MetricArtifact:
    """Pre-computed metric data bundle consumed by tear-sheet panels.

    Carries only pre-aggregated values. The reporting layer renders these into
    ``ChartSpec`` objects and must never recompute the underlying metrics.

    Attributes:
        name: Canonical artifact name (matches the panel key that consumes it).
        metric_type: Chart type hint ("line", "bar", "scatter", "heatmap", ...).
        data: Pre-computed values keyed by semantic field name.
        title: Optional display title (falls back to the panel default).
        x_label / y_label: Optional axis labels.
        dimensions: Optional pre-computed dimension annotations.
    """
    name: str
    metric_type: str = "line"
    data: Dict[str, Any] = field(default_factory=dict)
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    dimensions: Dict[str, Any] = field(default_factory=dict)


# Marker for panels whose source metric artifact was not computed.
NOT_COMPUTED = "NOT_COMPUTED"
UNAVAILABLE = "UNAVAILABLE"


def metric_artifact(
    name: str,
    metric_type: str = "line",
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    **data: Any,
) -> MetricArtifact:
    """Convenience constructor for a pre-computed ``MetricArtifact``.

    ``data`` is captured as keyword arguments, e.g.::

        metric_artifact("year_month_ic_heatmap", metric_type="heatmap",
                        matrix=[[0.1, 0.2], [0.3, 0.4]], years=[2020, 2021])
    """
    return MetricArtifact(
        name=name,
        metric_type=metric_type,
        title=title,
        x_label=x_label,
        y_label=y_label,
        data=dict(data),
    )


# ---------------------------------------------------------------------------
# The full institutional 22-panel set
# ---------------------------------------------------------------------------
# (panel_key, default_title, artifact_key, default_chart_type)
INSTITUTIONAL_PANELS: Tuple[Tuple[str, str, str, str], ...] = (
    ("year_month_ic_heatmap", "Year x Month IC Heatmap",
     "year_month_ic_heatmap", "heatmap"),
    ("rolling_rank_ic", "Rolling Rank IC (60/120/252)",
     "rolling_rank_ic", "line"),
    ("ic_recent_vs_full", "Recent vs Full-Period IC",
     "ic_recent_vs_full", "bar"),
    ("ic_positive_ratio_sign_survival", "Positive IC Ratio / Sign Survival",
     "ic_positive_ratio_sign_survival", "bar"),
    ("ic_hac_confidence_band", "IC HAC Confidence Band",
     "ic_hac_confidence_band", "line"),
    ("ic_horizon_surface", "IC Horizon Surface (1/2/3/5/10/20/60d)",
     "ic_horizon_surface", "line"),
    ("quantile_monotonicity", "Quantile Monotonicity",
     "quantile_monotonicity", "bar"),
    ("quantile_curvature", "Quantile Curvature / U-Shape",
     "quantile_curvature", "scatter"),
    ("tail_asymmetry", "Tail Asymmetry",
     "tail_asymmetry", "bar"),
    ("cumulative_quantile_return", "Cumulative Quantile Return",
     "cumulative_quantile_return", "line"),
    ("cumulative_top_bottom_spread", "Cumulative Top-Bottom Spread",
     "cumulative_top_bottom_spread", "line"),
    ("top_bottom_drawdown", "Top-Bottom Drawdown",
     "top_bottom_drawdown", "line"),
    ("size_liquidity_heatmap", "Size x Liquidity Heatmap",
     "size_liquidity_heatmap", "heatmap"),
    ("regime_heatmap", "Regime Heatmap",
     "regime_heatmap", "heatmap"),
    ("cost_sensitivity_curve", "Cost Sensitivity Curve",
     "cost_sensitivity_curve", "line"),
    ("delay_sensitivity_curve", "Delay Sensitivity Curve",
     "delay_sensitivity_curve", "line"),
    ("industry_style_exposure", "Industry / Style Exposure",
     "industry_style_exposure", "bar"),
    ("exposure_drift", "Exposure Drift",
     "exposure_drift", "line"),
    ("raw_vs_neutralized_ic", "Raw vs Neutralized IC",
     "raw_vs_neutralized_ic", "bar"),
    ("missing_staleness_timeline", "Missing / Staleness Timeline",
     "missing_staleness_timeline", "line"),
    ("spec_robustness_cube", "Spec Robustness Cube",
     "spec_robustness_cube", "scatter"),
    ("data_quality_missingness", "Data Quality / Missingness",
     "data_quality_missingness", "bar"),
)


# ---------------------------------------------------------------------------
# Rendering helpers (never recompute metrics - only serialize & structure)
# ---------------------------------------------------------------------------

def _plain(value: Any) -> Any:
    """Recursively convert numpy/pandas values to JSON-safe plain Python.

    Serialization aid only - it never computes or aggregates metrics.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) or (
        hasattr(value, "tolist") and not isinstance(value, (list, tuple))
    ):
        return _plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def _placeholder_panel(panel_key: str, title: str, chart_type: str) -> ChartSpec:
    """NOT_COMPUTED placeholder rendered when the source artifact is absent."""
    return ChartSpec(
        title=title,
        chart_type="placeholder",
        x_label="",
        y_label="",
        data={
            "status": NOT_COMPUTED,
            "panel": panel_key,
            "message": (
                f"{title}: not computed - source metric artifact unavailable"
            ),
        },
    )


def _spec_from_artifact(
    artifact: MetricArtifact,
    title: str,
    chart_type: str,
    x_label: str = "",
    y_label: str = "",
    data: Optional[Dict[str, Any]] = None,
) -> ChartSpec:
    """Render a ChartSpec directly from a pre-computed MetricArtifact."""
    return ChartSpec(
        title=artifact.title or title,
        chart_type=artifact.metric_type or chart_type,
        x_label=artifact.x_label or x_label,
        y_label=artifact.y_label or y_label,
        source_artifact_refs=(artifact.name,),
        data=_plain(data if data is not None else artifact.data),
    )


# ---------------------------------------------------------------------------
# Panel renderers (one per institutional panel)
# ---------------------------------------------------------------------------

def _render_year_month_ic_heatmap(artifact, title, chart_type):
    data = dict(artifact.data)
    data.setdefault("months", list(range(1, 13)))
    return _spec_from_artifact(artifact, title, chart_type, "Month", "Year", data)


def _render_rolling_rank_ic(artifact, title, chart_type):
    data = dict(artifact.data)
    if "timestamps" not in data and "windows" in data:
        lengths = [len(v) for v in data["windows"].values() if v is not None]
        if lengths:
            data["timestamps"] = list(range(max(lengths)))
    return _spec_from_artifact(artifact, title, chart_type, "Time", "Rank IC", data)


def _render_ic_recent_vs_full(artifact, title, chart_type):
    return _spec_from_artifact(artifact, title, chart_type, "Window (days)", "IC")


def _render_ic_positive_ratio_sign_survival(artifact, title, chart_type):
    return _spec_from_artifact(artifact, title, chart_type, "Metric", "Value")


def _render_ic_hac_confidence_band(artifact, title, chart_type):
    return _spec_from_artifact(artifact, title, chart_type, "Time", "IC")


def _render_ic_horizon_surface(artifact, title, chart_type):
    return _spec_from_artifact(artifact, title, chart_type, "Horizon (days)", "IC")


def _render_quantile_monotonicity(artifact, title, chart_type):
    return _spec_from_artifact(artifact, title, chart_type, "Quantile", "Mean Return")


def _render_quantile_curvature(artifact, title, chart_type):
    return _spec_from_artifact(artifact, title, chart_type, "Quantile", "Mean Return")


def _render_tail_asymmetry(artifact, title, chart_type):
    return _spec_from_artifact(artifact, title, chart_type, "Metric", "Value")


def _render_cumulative_quantile_return(artifact, title, chart_type):
    return _spec_from_artifact(artifact, title, chart_type, "Time", "Cumulative Return")


def _render_cumulative_top_bottom_spread(artifact, title, chart_type):
    return _spec_from_artifact(artifact, title, chart_type, "Time", "Cumulative Spread")


def _render_top_bottom_drawdown(artifact, title, chart_type):
    return _spec_from_artifact(artifact, title, chart_type, "Time", "Drawdown")


def _render_size_liquidity_heatmap(artifact, title, chart_type):
    return _spec_from_artifact(artifact, title, chart_type, "Liquidity", "Size")


def _render_regime_heatmap(artifact, title, chart_type):
    return _spec_from_artifact(artifact, title, chart_type, "Regime", "Time")


def _render_cost_sensitivity_curve(artifact, title, chart_type):
    return _spec_from_artifact(artifact, title, chart_type, "Cost (bps)", "Net Return")


def _render_delay_sensitivity_curve(artifact, title, chart_type):
    return _spec_from_artifact(artifact, title, chart_type, "Delay (days)", "IC")


def _render_industry_style_exposure(artifact, title, chart_type):
    return _spec_from_artifact(artifact, title, chart_type, "Bucket", "Exposure")


def _render_exposure_drift(artifact, title, chart_type):
    return _spec_from_artifact(artifact, title, chart_type, "Time", "Exposure Drift")


def _render_raw_vs_neutralized_ic(artifact, title, chart_type):
    return _spec_from_artifact(artifact, title, chart_type, "Window (days)", "IC")


def _render_missing_staleness_timeline(artifact, title, chart_type):
    return _spec_from_artifact(artifact, title, chart_type, "Time", "Missing / Stale Ratio")


def _render_spec_robustness_cube(artifact, title, chart_type):
    return _spec_from_artifact(artifact, title, chart_type, "Spec Dim", "IC")


def _render_data_quality_missingness(artifact, title, chart_type):
    return _spec_from_artifact(artifact, title, chart_type, "Metric", "Value")


_PANEL_RENDERERS: Dict[str, Any] = {
    "year_month_ic_heatmap": _render_year_month_ic_heatmap,
    "rolling_rank_ic": _render_rolling_rank_ic,
    "ic_recent_vs_full": _render_ic_recent_vs_full,
    "ic_positive_ratio_sign_survival": _render_ic_positive_ratio_sign_survival,
    "ic_hac_confidence_band": _render_ic_hac_confidence_band,
    "ic_horizon_surface": _render_ic_horizon_surface,
    "quantile_monotonicity": _render_quantile_monotonicity,
    "quantile_curvature": _render_quantile_curvature,
    "tail_asymmetry": _render_tail_asymmetry,
    "cumulative_quantile_return": _render_cumulative_quantile_return,
    "cumulative_top_bottom_spread": _render_cumulative_top_bottom_spread,
    "top_bottom_drawdown": _render_top_bottom_drawdown,
    "size_liquidity_heatmap": _render_size_liquidity_heatmap,
    "regime_heatmap": _render_regime_heatmap,
    "cost_sensitivity_curve": _render_cost_sensitivity_curve,
    "delay_sensitivity_curve": _render_delay_sensitivity_curve,
    "industry_style_exposure": _render_industry_style_exposure,
    "exposure_drift": _render_exposure_drift,
    "raw_vs_neutralized_ic": _render_raw_vs_neutralized_ic,
    "missing_staleness_timeline": _render_missing_staleness_timeline,
    "spec_robustness_cube": _render_spec_robustness_cube,
    "data_quality_missingness": _render_data_quality_missingness,
}


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------

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

    # Pre-computed metric artifacts consumed by the institutional panels.
    artifacts: Dict[str, MetricArtifact] = field(default_factory=dict)


def generate_tear_sheet(
    evaluation_result: EvaluationResult,
) -> Dict[str, ChartSpec]:
    """Generate the institutional tear sheet from evaluation results.

    The 22 institutional panels render strictly from pre-computed
    ``MetricArtifact`` objects attached to ``evaluation_result.artifacts``.
    Panels whose source artifact is absent render a ``NOT_COMPUTED``
    placeholder (never a 0.0).

    Args:
        evaluation_result: EvaluationResult containing all metrics/artifacts

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

    # ------------------------------------------------------------------
    # Institutional 22-panel set - consumes pre-computed MetricArtifacts only.
    # Never recomputes metrics. Absent artifact -> NOT_COMPUTED placeholder.
    # ------------------------------------------------------------------
    artifacts = evaluation_result.artifacts or {}
    for panel_key, title, artifact_key, chart_type in INSTITUTIONAL_PANELS:
        artifact = artifacts.get(artifact_key)
        renderer = _PANEL_RENDERERS.get(panel_key)
        if artifact is None or renderer is None:
            panels[panel_key] = _placeholder_panel(panel_key, title, chart_type)
        else:
            panels[panel_key] = renderer(artifact, title, chart_type)

    return panels
