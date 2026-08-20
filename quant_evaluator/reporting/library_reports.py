"""
Library-level reports for quant_evaluator.

Provides functions for generating summary reports, comparison reports,
and adversarial test reports across multiple evaluation results.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from quant_evaluator.reporting.chart_spec import ChartSpec


@dataclass
class LibraryReport:
    """Summary report for a library of factor evaluations.

    Attributes:
        library_name: Name of the factor library
        n_factors: Number of factors evaluated
        summary_stats: Aggregated summary statistics
        charts: List of ChartSpec objects for visualization
    """
    library_name: str
    n_factors: int
    summary_stats: Dict[str, float]
    charts: List[ChartSpec]


@dataclass
class ComparisonReport:
    """Comparison report between two factor libraries.

    Attributes:
        library_a_name: Name of first library
        library_b_name: Name of second library
        comparison_stats: Comparative statistics
        charts: List of ChartSpec objects for visualization
    """
    library_a_name: str
    library_b_name: str
    comparison_stats: Dict[str, Any]
    charts: List[ChartSpec]


@dataclass
class AdversarialReport:
    """Report for adversarial testing results.

    Attributes:
        test_name: Name of the adversarial test
        results: Test results
        charts: List of ChartSpec objects for visualization
    """
    test_name: str
    results: Dict[str, Any]
    charts: List[ChartSpec]


def generate_library_report(
    evaluation_results: List[Dict[str, Any]],
    library_name: str = "Factor Library",
) -> LibraryReport:
    """Generate a summary report for a library of factor evaluations.

    Args:
        evaluation_results: List of evaluation result dictionaries
        library_name: Name of the factor library

    Returns:
        LibraryReport with aggregated statistics and charts
    """
    n_factors = len(evaluation_results)

    if n_factors == 0:
        return LibraryReport(
            library_name=library_name,
            n_factors=0,
            summary_stats={},
            charts=[]
        )

    # Aggregate statistics
    ic_means = [r.get("ic_mean", 0.0) for r in evaluation_results]
    ic_stds = [r.get("ic_std", 0.0) for r in evaluation_results]
    icirs = [r.get("icir", 0.0) for r in evaluation_results]
    sharpe_ratios = [r.get("sharpe_ratio", 0.0) for r in evaluation_results]
    max_drawdowns = [r.get("max_drawdown", 0.0) for r in evaluation_results]
    turnovers = [r.get("avg_turnover", 0.0) for r in evaluation_results]

    summary_stats = {
        "n_factors": n_factors,
        "mean_ic_mean": float(np.mean(ic_means)) if ic_means else 0.0,
        "std_ic_mean": float(np.std(ic_means)) if ic_means else 0.0,
        "median_ic_mean": float(np.median(ic_means)) if ic_means else 0.0,
        "mean_icir": float(np.mean(icirs)) if icirs else 0.0,
        "std_icir": float(np.std(icirs)) if icirs else 0.0,
        "mean_sharpe": float(np.mean(sharpe_ratios)) if sharpe_ratios else 0.0,
        "std_sharpe": float(np.std(sharpe_ratios)) if sharpe_ratios else 0.0,
        "mean_max_drawdown": float(np.mean(max_drawdowns)) if max_drawdowns else 0.0,
        "mean_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "positive_ic_ratio": float(np.mean([1 for ic in ic_means if ic > 0])) / n_factors if n_factors > 0 else 0.0,
    }

    # Generate charts
    charts = []

    # Chart 1: IC Mean distribution across factors
    charts.append(ChartSpec(
        title=f"{library_name} - IC Mean Distribution",
        x_label="Factor",
        y_label="IC Mean",
        chart_type="bar",
        data={
            "categories": [f"F{i+1}" for i in range(n_factors)],
            "values": ic_means
        }
    ))

    # Chart 2: ICIR distribution across factors
    charts.append(ChartSpec(
        title=f"{library_name} - ICIR Distribution",
        x_label="Factor",
        y_label="ICIR",
        chart_type="bar",
        data={
            "categories": [f"F{i+1}" for i in range(n_factors)],
            "values": icirs
        }
    ))

    # Chart 3: Risk-return scatter
    annual_returns = [r.get("annual_return", 0.0) for r in evaluation_results]
    annual_vols = [r.get("annual_volatility", 0.0) for r in evaluation_results]
    charts.append(ChartSpec(
        title=f"{library_name} - Risk-Return Scatter",
        x_label="Volatility",
        y_label="Return",
        chart_type="scatter",
        data={
            "x": annual_vols,
            "y": annual_returns,
            "labels": [f"F{i+1}" for i in range(n_factors)]
        }
    ))

    return LibraryReport(
        library_name=library_name,
        n_factors=n_factors,
        summary_stats=summary_stats,
        charts=charts
    )


def compare_libraries(
    results_a: List[Dict[str, Any]],
    results_b: List[Dict[str, Any]],
    name_a: str = "Library A",
    name_b: str = "Library B",
) -> ComparisonReport:
    """Compare two factor libraries.

    Args:
        results_a: Evaluation results from first library
        results_b: Evaluation results from second library
        name_a: Name of first library
        name_b: Name of second library

    Returns:
        ComparisonReport with comparative statistics and charts
    """
    # Generate individual reports
    report_a = generate_library_report(results_a, name_a)
    report_b = generate_library_report(results_b, name_b)

    # Compute comparison statistics
    comparison_stats = {
        "library_a": report_a.summary_stats,
        "library_b": report_b.summary_stats,
        "n_factors_a": report_a.n_factors,
        "n_factors_b": report_b.n_factors,
    }

    # Generate comparison charts
    charts = []

    # Chart 1: IC Mean comparison
    ic_means_a = [r.get("ic_mean", 0.0) for r in results_a]
    ic_means_b = [r.get("ic_mean", 0.0) for r in results_b]
    charts.append(ChartSpec(
        title="IC Mean Comparison",
        x_label="Library",
        y_label="IC Mean",
        chart_type="bar",
        data={
            "categories": [name_a, name_b],
            "values": [float(np.mean(ic_means_a)) if ic_means_a else 0.0,
                      float(np.mean(ic_means_b)) if ic_means_b else 0.0]
        }
    ))

    # Chart 2: Sharpe ratio comparison
    sharpe_a = [r.get("sharpe_ratio", 0.0) for r in results_a]
    sharpe_b = [r.get("sharpe_ratio", 0.0) for r in results_b]
    charts.append(ChartSpec(
        title="Sharpe Ratio Comparison",
        x_label="Library",
        y_label="Sharpe Ratio",
        chart_type="bar",
        data={
            "categories": [name_a, name_b],
            "values": [float(np.mean(sharpe_a)) if sharpe_a else 0.0,
                      float(np.mean(sharpe_b)) if sharpe_b else 0.0]
        }
    ))

    # Chart 3: Max drawdown comparison
    dd_a = [r.get("max_drawdown", 0.0) for r in results_a]
    dd_b = [r.get("max_drawdown", 0.0) for r in results_b]
    charts.append(ChartSpec(
        title="Max Drawdown Comparison",
        x_label="Library",
        y_label="Max Drawdown",
        chart_type="bar",
        data={
            "categories": [name_a, name_b],
            "values": [float(np.mean(dd_a)) if dd_a else 0.0,
                      float(np.mean(dd_b)) if dd_b else 0.0]
        }
    ))

    return ComparisonReport(
        library_a_name=name_a,
        library_b_name=name_b,
        comparison_stats=comparison_stats,
        charts=charts
    )


def generate_adversarial_report(
    adversarial_results: Dict[str, Any],
    test_name: str = "Adversarial Test",
) -> AdversarialReport:
    """Generate a report for adversarial testing results.

    Args:
        adversarial_results: Dictionary containing adversarial test results
        test_name: Name of the adversarial test

    Returns:
        AdversarialReport with test results and charts
    """
    charts = []

    # Extract results
    if "stability_scores" in adversarial_results:
        stability_scores = adversarial_results["stability_scores"]
        charts.append(ChartSpec(
            title=f"{test_name} - Stability Scores",
            x_label="Perturbation",
            y_label="Score",
            chart_type="bar",
            data={
                "categories": list(stability_scores.keys()),
                "values": list(stability_scores.values())
            }
        ))

    if "robustness_metrics" in adversarial_results:
        robustness = adversarial_results["robustness_metrics"]
        charts.append(ChartSpec(
            title=f"{test_name} - Robustness Metrics",
            x_label="Metric",
            y_label="Value",
            chart_type="bar",
            data={
                "categories": list(robustness.keys()),
                "values": list(robustness.values())
            }
        ))

    return AdversarialReport(
        test_name=test_name,
        results=adversarial_results,
        charts=charts
    )
