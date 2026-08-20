"""
Stress testing and scenario analysis for portfolios.

Provides historical scenario replay, hypothetical scenario application,
correlation breakdown analysis, and worst-case scenario identification.
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from scipy import stats


def apply_historical_scenario(
    returns: np.ndarray,
    scenario_returns: np.ndarray,
    scaling_method: str = "direct",
) -> np.ndarray:
    """
    Apply historical scenario to current returns distribution.

    Args:
        returns: Current return series (T,) or (T, F)
        scenario_returns: Historical scenario returns to apply (S,)
        scaling_method: "direct" (use scenario as-is) or "scaled" (scale by current vol)

    Returns:
        Scenario impact on returns, same shape as returns
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape
    scenario_impact = np.zeros((len(scenario_returns), F))

    for f in range(F):
        ret_f = returns[:, f]
        valid = np.isfinite(ret_f)

        if np.sum(valid) < 10:
            scenario_impact[:, f] = np.nan
            continue

        ret_valid = ret_f[valid]

        if scaling_method == "direct":
            # Use scenario returns directly
            scenario_impact[:, f] = scenario_returns
        elif scaling_method == "scaled":
            # Scale scenario by current volatility
            current_vol = np.std(ret_valid, ddof=1)
            scenario_vol = np.std(scenario_returns, ddof=1)

            if scenario_vol < 1e-10:
                scenario_impact[:, f] = scenario_returns
            else:
                scaling_factor = current_vol / scenario_vol
                scenario_impact[:, f] = scenario_returns * scaling_factor
        else:
            raise ValueError(f"Unknown scaling_method: {scaling_method}")

    return scenario_impact[:, 0] if squeeze else scenario_impact


def apply_hypothetical_scenario(
    returns: np.ndarray,
    shock_size: float,
    shock_type: str = "absolute",
    correlation_adjustment: Optional[float] = None,
) -> np.ndarray:
    """
    Apply hypothetical shock scenario.

    Args:
        returns: Return series (T,) or (T, F)
        shock_size: Size of shock (e.g., -0.10 for -10% shock)
        shock_type: "absolute" (fixed shock) or "volatility" (multiple of current vol)
        correlation_adjustment: Optional factor to adjust correlation (1.0 = no change)

    Returns:
        Shocked returns, same shape as returns
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape
    shocked = np.copy(returns)

    for f in range(F):
        ret_f = returns[:, f]
        valid = np.isfinite(ret_f)

        if np.sum(valid) < 2:
            # Fail closed: not enough finite data to shock this asset —
            # mark the entire column as invalid instead of silently
            # passing through an unshocked copy of the input.
            shocked[:, f] = np.nan
            continue

        ret_valid = ret_f[valid]

        if shock_type == "absolute":
            # Apply absolute shock
            shocked[:, f] = ret_f + shock_size
        elif shock_type == "volatility":
            # Apply shock as multiple of volatility
            current_vol = np.std(ret_valid, ddof=1)
            shocked[:, f] = ret_f + shock_size * current_vol
        else:
            raise ValueError(f"Unknown shock_type: {shock_type}")

    # Apply correlation adjustment if multi-factor
    if F > 1 and correlation_adjustment is not None:
        # Adjust cross-factor correlation
        mean_shocked = np.nanmean(shocked, axis=0, keepdims=True)
        centered = shocked - mean_shocked

        # Scale cross-correlation
        for f1 in range(F):
            for f2 in range(f1 + 1, F):
                # Extract valid pairs
                valid_pair = np.isfinite(centered[:, f1]) & np.isfinite(centered[:, f2])
                if np.sum(valid_pair) < 2:
                    continue

                # Compute correlation adjustment
                corr_current = np.corrcoef(centered[valid_pair, f1], centered[valid_pair, f2])[0, 1]
                if not np.isfinite(corr_current):
                    continue

                # Adjust correlation (simplified approach)
                adjustment_factor = correlation_adjustment - 1.0
                centered[valid_pair, f1] += adjustment_factor * centered[valid_pair, f2] * 0.5
                centered[valid_pair, f2] += adjustment_factor * centered[valid_pair, f1] * 0.5

        shocked = centered + mean_shocked

    return shocked[:, 0] if squeeze else shocked


def compute_scenario_impact(
    returns: np.ndarray,
    scenario_returns: np.ndarray,
    metrics: Optional[List[str]] = None,
    periods_per_year: int = 252,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute impact of scenario on portfolio metrics.

    Args:
        returns: Baseline return series (T,) or (T, F)
        scenario_returns: Scenario return series (S,) or (S, F)
        metrics: List of metrics to compute ("mean", "sharpe", "max_drawdown", "var", "cvar");
            defaults to ["mean", "sharpe", "max_drawdown"]
        periods_per_year: Periods per year for annualization

    Returns:
        Dictionary with baseline and scenario values for each metric
    """
    from quant_evaluator.metrics.portfolio_stats import (
        compute_sharpe_ratio,
        compute_maximum_drawdown,
    )
    from quant_evaluator.metrics.risk.var_cvar import compute_var, compute_cvar

    if metrics is None:
        metrics = ["mean", "sharpe", "max_drawdown"]

    # Fail closed: reject unknown metric names instead of silently ignoring them.
    known_metrics = {"mean", "sharpe", "max_drawdown", "var", "cvar"}
    unknown = [m for m in metrics if m not in known_metrics]
    if unknown:
        raise ValueError(
            f"Unknown metric(s) in scenario config: {sorted(unknown)}. "
            f"Supported metrics: {sorted(known_metrics)}"
        )

    result = {}

    # Ensure same shape
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
    if scenario_returns.ndim == 1:
        scenario_returns = scenario_returns[:, np.newaxis]

    F = returns.shape[1]

    for metric in metrics:
        if metric == "mean":
            baseline = np.nanmean(returns, axis=0) * periods_per_year
            scenario = np.nanmean(scenario_returns, axis=0) * periods_per_year
            result[f"{metric}_baseline"] = baseline[0] if F == 1 else baseline
            result[f"{metric}_scenario"] = scenario[0] if F == 1 else scenario
            result[f"{metric}_change"] = scenario - baseline if F > 1 else (scenario - baseline)[0]

        elif metric == "sharpe":
            baseline = compute_sharpe_ratio(returns, periods_per_year=periods_per_year)
            scenario = compute_sharpe_ratio(scenario_returns, periods_per_year=periods_per_year)
            result[f"{metric}_baseline"] = baseline
            result[f"{metric}_scenario"] = scenario
            change = scenario - baseline if isinstance(scenario, np.ndarray) else scenario - baseline
            result[f"{metric}_change"] = change

        elif metric == "max_drawdown":
            baseline, _, _ = compute_maximum_drawdown(returns)
            scenario, _, _ = compute_maximum_drawdown(scenario_returns)
            result[f"{metric}_baseline"] = baseline
            result[f"{metric}_scenario"] = scenario
            change = scenario - baseline if isinstance(scenario, np.ndarray) else scenario - baseline
            result[f"{metric}_change"] = change

        elif metric == "var":
            baseline = compute_var(returns, method="historical")
            scenario = compute_var(scenario_returns, method="historical")
            result[f"{metric}_baseline"] = baseline
            result[f"{metric}_scenario"] = scenario
            change = scenario - baseline if isinstance(scenario, np.ndarray) else scenario - baseline
            result[f"{metric}_change"] = change

        elif metric == "cvar":
            baseline = compute_cvar(returns, method="historical")
            scenario = compute_cvar(scenario_returns, method="historical")
            result[f"{metric}_baseline"] = baseline
            result[f"{metric}_scenario"] = scenario
            change = scenario - baseline if isinstance(scenario, np.ndarray) else scenario - baseline
            result[f"{metric}_change"] = change

    return result


def compute_correlation_breakdown(
    returns_x: np.ndarray,
    returns_y: np.ndarray,
    stress_quantile: float = 0.05,
    min_periods: int = 20,
) -> Dict[str, float]:
    """
    Analyze correlation during stress periods vs normal periods.

    Args:
        returns_x: First return series (T,)
        returns_y: Second return series (T,)
        stress_quantile: Quantile threshold to define stress (default 0.05)
        min_periods: Minimum periods required

    Returns:
        Dictionary with:
            - correlation_normal: Correlation during normal periods
            - correlation_stress: Correlation during stress periods
            - correlation_all: Overall correlation
            - correlation_breakdown: Difference (stress - normal)
    """
    if returns_x.ndim != 1 or returns_y.ndim != 1:
        raise ValueError("compute_correlation_breakdown requires 1D returns")

    if len(returns_x) != len(returns_y):
        raise ValueError("returns must have same length")

    # Filter valid pairs
    valid = np.isfinite(returns_x) & np.isfinite(returns_y)
    n_valid = np.sum(valid)

    if n_valid < min_periods:
        return {
            "correlation_normal": np.nan,
            "correlation_stress": np.nan,
            "correlation_all": np.nan,
            "correlation_breakdown": np.nan,
        }

    ret_x = returns_x[valid]
    ret_y = returns_y[valid]

    # Overall correlation
    corr_all = np.corrcoef(ret_x, ret_y)[0, 1]

    # Define stress periods (either x or y in lower tail)
    threshold_x = np.quantile(ret_x, stress_quantile)
    threshold_y = np.quantile(ret_y, stress_quantile)

    stress_mask = (ret_x <= threshold_x) | (ret_y <= threshold_y)
    normal_mask = ~stress_mask

    # Correlation during stress
    if np.sum(stress_mask) >= 5:
        corr_stress = np.corrcoef(ret_x[stress_mask], ret_y[stress_mask])[0, 1]
    else:
        corr_stress = np.nan

    # Correlation during normal periods
    if np.sum(normal_mask) >= 5:
        corr_normal = np.corrcoef(ret_x[normal_mask], ret_y[normal_mask])[0, 1]
    else:
        corr_normal = np.nan

    # Breakdown (correlation increases during stress if positive)
    if np.isfinite(corr_stress) and np.isfinite(corr_normal):
        breakdown = corr_stress - corr_normal
    else:
        breakdown = np.nan

    return {
        "correlation_normal": corr_normal,
        "correlation_stress": corr_stress,
        "correlation_all": corr_all,
        "correlation_breakdown": breakdown,
    }


def compute_worst_case_scenarios(
    returns: np.ndarray,
    n_scenarios: int = 10,
    window_size: int = 20,
    metric: str = "cumulative",
) -> List[Dict[str, Union[int, float, np.ndarray]]]:
    """
    Identify worst-case historical scenarios.

    Args:
        returns: Return series (T,) or (T, F)
        n_scenarios: Number of worst scenarios to identify
        window_size: Window size for rolling worst periods
        metric: "cumulative" (worst cumulative return) or "volatility" (highest volatility)

    Returns:
        List of worst scenarios, each with:
            - start_idx: Start index
            - end_idx: End index
            - metric_value: Metric value for this period
            - returns: Returns during this period
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape

    if T < window_size:
        return []

    # Compute rolling metric
    scenarios = []

    for start in range(T - window_size + 1):
        end = start + window_size
        window_returns = returns[start:end, :]

        if metric == "cumulative":
            # Cumulative return in window
            cum_ret = np.prod(1.0 + window_returns, axis=0) - 1.0
            metric_value = np.mean(cum_ret) if F > 1 else cum_ret[0]

        elif metric == "volatility":
            # Volatility in window
            vol = np.nanstd(window_returns, axis=0, ddof=1)
            metric_value = np.mean(vol) if F > 1 else vol[0]

        else:
            raise ValueError(f"Unknown metric: {metric}")

        scenarios.append({
            "start_idx": start,
            "end_idx": end - 1,
            "metric_value": metric_value,
            "returns": window_returns[:, 0] if squeeze else window_returns,
        })

    # Sort by metric (ascending for cumulative, descending for volatility).
    # Fail closed on NaN: a window whose metric is NaN must never be ranked
    # as a worst-case winner — exclude it from the ranking entirely.
    valid_scenarios = [s for s in scenarios if np.isfinite(s["metric_value"])]
    scenarios = valid_scenarios

    if metric == "cumulative":
        scenarios.sort(key=lambda x: x["metric_value"])
    else:
        scenarios.sort(key=lambda x: -x["metric_value"])

    # Return top n_scenarios
    return scenarios[:n_scenarios]
