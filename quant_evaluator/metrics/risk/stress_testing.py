"""
Stress testing and scenario analysis for portfolios.

Provides historical scenario replay, hypothetical scenario application,
correlation breakdown analysis, and worst-case scenario identification.
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from scipy import stats
import hashlib


def apply_historical_scenario(
    returns: np.ndarray,
    scenario_returns: np.ndarray,
    scaling_method: str = "direct",
    return_evidence: bool = False,
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

    values=scenario_impact[:, 0] if squeeze else scenario_impact
    if return_evidence:
        return {"values":values,"evidence_type":"GENERATED_RETURN_SCENARIO",
                "formal_strategy_stress_eligible":False,
                "scaling_method":scaling_method}
    return values


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
    returns = np.asarray(returns, dtype=float)
    if returns.ndim not in (1, 2):
        raise ValueError("shock returns require (T,) or (T,F) axes")
    if isinstance(shock_size, (bool, np.bool_)) or not np.isscalar(shock_size) or not np.isfinite(shock_size):
        raise ValueError("shock size must be a finite number")
    if shock_type not in ("absolute", "volatility"):
        raise ValueError("unknown shock type")
    if correlation_adjustment is not None and (isinstance(correlation_adjustment, (bool, np.bool_))
            or not np.isscalar(correlation_adjustment) or not np.isfinite(correlation_adjustment)
            or correlation_adjustment < 0):
        raise ValueError("correlation multiplier must be finite and nonnegative")
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

    # Explicit scalar semantics: multiply original off-diagonal correlations,
    # preserve means and marginal sample volatility, reject infeasible targets.
    if correlation_adjustment is not None and correlation_adjustment != 1.0:
        if not np.isfinite(correlation_adjustment) or correlation_adjustment < 0:
            raise ValueError("correlation multiplier must be finite and nonnegative")
        target = np.eye(F) + correlation_adjustment * (np.corrcoef(shocked, rowvar=False) - np.eye(F))
        shocked = apply_target_correlation(shocked, target)

    return shocked[:, 0] if squeeze else shocked


def apply_target_correlation(returns, target_correlation, *, asset_ids=None,
                             target_asset_ids=None, return_evidence=False):
    """Symmetric whiten/color transform, no in-place pair mutation.

    Columns and target rows/columns share the same caller-supplied asset order.
    A simultaneous permutation commutes with the symmetric matrix functions.
    Singular source, indefinite target and incomplete samples are unsupported.
    This generates returns, not a position/execution replay.
    """
    x = np.asarray(returns, dtype=float)
    target = np.asarray(target_correlation, dtype=float)
    if x.ndim != 2 or x.shape[0] <= x.shape[1] or not np.isfinite(x).all():
        raise ValueError("complete full-rank sample required for correlation mapping")
    f = x.shape[1]
    if target.shape != (f, f):
        raise ValueError("target correlation axes must match source assets")
    if (asset_ids is None) != (target_asset_ids is None):
        raise ValueError("both asset_ids and target_asset_ids are required for named mapping")
    if asset_ids is not None:
        asset_ids, target_asset_ids = tuple(asset_ids), tuple(target_asset_ids)
        if len(asset_ids)!=f or len(set(asset_ids))!=f or len(target_asset_ids)!=f or set(asset_ids)!=set(target_asset_ids):
            raise ValueError("asset axes must be unique and identify the same assets")
        order=[target_asset_ids.index(name) for name in asset_ids]
        target=target[np.ix_(order,order)]
    if target.shape != (f, f) or not np.isfinite(target).all() or not np.allclose(target, target.T, atol=1e-12, rtol=0) or not np.allclose(np.diag(target), 1., atol=1e-12, rtol=0):
        raise ValueError("target correlation must be symmetric with unit diagonal")
    std = np.std(x, axis=0, ddof=1)
    if np.any(std <= 0):
        raise ValueError("constant source margin")
    mean = np.mean(x, axis=0)
    z = (x-mean)/std
    source = z.T@z/(len(x)-1)
    eigen, vectors = np.linalg.eigh(source)
    target_eigen, target_vectors = np.linalg.eigh(target)
    if np.min(eigen) <= 1e-12 or np.min(target_eigen) < -1e-12:
        raise ValueError("source is singular or target is not PSD")
    whitening = (vectors * (1/np.sqrt(eigen)))@vectors.T
    coloring = (target_vectors * np.sqrt(np.maximum(target_eigen, 0)))@target_vectors.T
    mapped=(z@whitening@coloring)*std + mean
    if not return_evidence: return mapped
    achieved=np.corrcoef(mapped,rowvar=False)
    return {"values":mapped,"evidence_type":"GENERATED_RETURN_SCENARIO",
            "asset_ids":asset_ids,"target_correlation":target.copy(),
            "achieved_correlation":achieved,
            "max_abs_error":float(np.max(np.abs(achieved-target))),
            "qualification":"DIAGNOSTIC_ONLY"}


def compute_scenario_impact(
    returns: np.ndarray,
    scenario_returns: np.ndarray,
    metrics: Optional[List[str]] = None,
    periods_per_year: int = 252,
    evidence_type: str = "GENERATED_RETURN_SCENARIO",
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

    if evidence_type != "GENERATED_RETURN_SCENARIO":
        raise ValueError("naked returns cannot assert POSITION_REPLAY; use compute_position_replay_impact")
    result = {"evidence_type": evidence_type,
              "formal_strategy_stress_eligible": False}

    returns = np.asarray(returns, dtype=float)
    scenario_returns = np.asarray(scenario_returns, dtype=float)
    if returns.ndim not in (1, 2) or scenario_returns.ndim not in (1, 2):
        raise ValueError("scenario returns require (T,) or (T,F) axes")
    if (1 if returns.ndim == 1 else returns.shape[1]) != (1 if scenario_returns.ndim == 1 else scenario_returns.shape[1]):
        raise ValueError("baseline and scenario factor axes differ")

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


def compute_position_replay_impact(baseline, scenario, *, metrics=None, periods_per_year=252):
    """Consume the existing execution-domain trajectories, never a type label.

    Execution-bound eligibility is separate from numerical qualification and
    the FA policy's common-scenario/usage approval; this function grants neither.
    """
    from vectorbt_qs.contracts.trajectories import PortfolioTrajectory
    from vectorbt_qs.contracts.costs import CostScope
    from quant_evaluator.adapters.execution_trajectory import trajectory_to_probe_artifact
    if not isinstance(baseline, PortfolioTrajectory) or not isinstance(scenario, PortfolioTrajectory):
        raise TypeError("position replay requires PortfolioTrajectory artifacts")
    if (baseline.refs.factor_ids != scenario.refs.factor_ids or baseline.profile != scenario.profile
            or baseline.dates != scenario.dates or baseline.refs.portfolio_ref != scenario.refs.portfolio_ref
            or baseline.scope != scenario.scope):
        raise ValueError("position replay comparison domain mismatch")
    if baseline.scenario_id == scenario.scenario_id:
        raise ValueError("baseline and stressed scenario need distinct identities")
    profile = {CostScope.GROSS_DIAGNOSTIC:'gross', CostScope.NET_ASSUMED:'net-base',
               CostScope.NET_EXECUTABLE:'net-executable'}[baseline.scope]
    left = trajectory_to_probe_artifact(baseline, expected_portfolio_profile=baseline.profile,
                                       expected_cost_profile=profile)
    right = trajectory_to_probe_artifact(scenario, expected_portfolio_profile=scenario.profile,
                                        expected_cost_profile=profile)
    result = compute_scenario_impact(left.values, right.values, metrics, periods_per_year)
    result.update(evidence_type='POSITION_REPLAY', baseline_trajectory_ref=baseline.artifact_id,
                  scenario_trajectory_ref=scenario.artifact_id, scenario_id=scenario.scenario_id,
                  formal_strategy_stress_eligible=bool(baseline.scope == CostScope.NET_EXECUTABLE
                      and baseline.executable_certified and scenario.executable_certified),
                  numerical_qualification='REQUIRES_SEPARATE_RECEIPT')
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


def evaluate_official_scenario_set(returns_by_candidate, scenarios, *, metrics=None):
    """Evaluate every candidate against one fixed, deduplicated event set.

    This is a comparison coordinator, not a position replay engine. Scenario
    records must carry predeclared ``event_id`` and ``returns``. Equivalent
    aliases declare their canonical_event_id before equal weighting. Equal
    return arrays do not establish that two economic events are equivalent.
    Conflicting reuse of a canonical event identity fails closed.
    """
    if not isinstance(returns_by_candidate, dict) or not returns_by_candidate:
        raise ValueError("nonempty candidate mapping required")
    unique={}
    for item in scenarios:
        event_id=item.get("canonical_event_id",item.get("event_id")); values=np.asarray(item.get("returns"),dtype=float)
        if not isinstance(event_id,str) or not event_id or values.ndim not in (1,2) or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("complete finite event_id/returns scenario required")
        digest=hashlib.sha256(values.tobytes()+str(values.shape).encode()).hexdigest()
        if event_id in unique and unique[event_id][0]!=digest: raise ValueError("conflicting scenario event identity")
        unique[event_id]=(digest,values)
    if not unique: raise ValueError("official scenario set cannot be empty")
    event_ids=tuple(sorted(unique))
    import json
    identity=json.dumps([(event_id,unique[event_id][0]) for event_id in event_ids],
                        separators=(',',':')).encode()
    output={}
    for candidate_id,baseline in returns_by_candidate.items():
        output[candidate_id]={event_id:compute_scenario_impact(np.asarray(baseline,dtype=float),unique[event_id][1],metrics=metrics,evidence_type="GENERATED_RETURN_SCENARIO") for event_id in event_ids}
    return {"scenario_set_id":"official:"+hashlib.sha256(identity).hexdigest(),
            "event_ids":event_ids,"event_weight":1/len(event_ids),
            "evidence_type":"GENERATED_RETURN_SCENARIO","results":output,
            "formal_strategy_stress_eligible":False}
