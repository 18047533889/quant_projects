"""Strict bridge from the execution-domain trajectory to QE's input artifact."""

from __future__ import annotations

import numpy as np

from quant_evaluator.contracts.artifact_types import ProbePortfolioArtifact
from vectorbt_qs.contracts.costs import CostScope
from vectorbt_qs.contracts.trajectories import PortfolioTrajectory


_COST_SCOPE = {
    "gross": CostScope.GROSS_DIAGNOSTIC,
    "net-base": CostScope.NET_ASSUMED,
    "net-stress": CostScope.NET_ASSUMED,
    "net-executable": CostScope.NET_EXECUTABLE,
}


def trajectory_to_probe_artifact(
    trajectory: PortfolioTrajectory,
    *,
    expected_portfolio_profile: str,
    expected_cost_profile: str,
    factor_ids: tuple[str, ...] | None = None,
    require_borrow_evidence: bool = False,
    expected_leg: str = "unspecified",
) -> ProbePortfolioArtifact:
    """Validate scenario provenance and expose net/gross returns as ``(T,F)``.

    Research LS with an assumed 8% borrow charge remains consumable, but it is
    explicitly non-executable.  A caller requesting borrow evidence or
    ``net-executable`` fails unless an actual borrow-availability ref exists.
    """
    if not isinstance(trajectory, PortfolioTrajectory):
        raise TypeError("trajectory must be a PortfolioTrajectory")
    if trajectory.profile != expected_portfolio_profile:
        raise ValueError("portfolio_profile does not match trajectory provenance")
    try:
        expected_scope = _COST_SCOPE[expected_cost_profile]
    except KeyError as exc:
        raise ValueError(f"unknown cost_profile: {expected_cost_profile}") from exc
    if trajectory.scope != expected_scope:
        raise ValueError("cost_profile does not match trajectory scope")
    if len(trajectory.refs.factor_ids) != 1:
        raise ValueError("one PortfolioTrajectory must represent exactly one factor")
    requested_ids = trajectory.refs.factor_ids if factor_ids is None else tuple(factor_ids)
    if requested_ids != trajectory.refs.factor_ids:
        raise ValueError("factor_ids are not covered by trajectory provenance")
    if not trajectory.refs.source_fingerprint or not trajectory.refs.portfolio_ref or not trajectory.refs.cost_ref:
        raise ValueError("trajectory requires source/portfolio/cost references")
    if trajectory.profile == "LONG_ONLY_RESEARCH" and not trajectory.refs.benchmark_ref:
        raise ValueError("long-only trajectory requires benchmark provenance")
    if (require_borrow_evidence or expected_scope == CostScope.NET_EXECUTABLE) and trajectory.profile == "LONG_SHORT_RESEARCH":
        if not trajectory.refs.borrow_ref:
            raise ValueError("BORROW_AVAILABILITY_NOT_PROVIDED")
    valid_legs = {"unspecified", "total", "long", "short", "bottom_holding", "active",
                  "relative_return", "investment_fraction", "cost_drag"}
    if expected_leg not in valid_legs:
        raise ValueError(f"unknown trajectory leg: {expected_leg}")
    if expected_leg in {"unspecified", "total"}:
        values = trajectory.gross_return if expected_scope == CostScope.GROSS_DIAGNOSTIC else trajectory.net_return
    elif expected_leg == "active":
        if trajectory.active_return is None:
            raise ValueError("active leg requires benchmark-bound active returns")
        values = (tuple(trajectory.gross_return[i] - trajectory.benchmark_return[i]
                        for i in range(len(trajectory.dates)))
                  if expected_scope == CostScope.GROSS_DIAGNOSTIC else trajectory.active_return)
    elif expected_leg == "relative_return":
        if trajectory.relative_wealth is None:
            raise ValueError("relative_return leg requires benchmark-bound relative wealth")
        previous, increments = 1.0, []
        for wealth in trajectory.relative_wealth:
            if not np.isfinite(wealth) or previous == 0:
                increments.append(np.nan)
            else:
                increments.append(float(wealth) / previous - 1.0)
            previous = float(wealth)
        values = tuple(increments)
    elif expected_leg == "investment_fraction":
        if expected_leg not in trajectory.contributions:
            raise ValueError("trajectory does not contain actual investment_fraction")
        values = trajectory.contributions[expected_leg]
    elif expected_leg == "cost_drag":
        names = tuple(trajectory.cost_component_names)
        if not names:
            raise ValueError("cost_drag requires bound cost_component_names")
        if any(name not in trajectory.contributions for name in names):
            raise ValueError("cost_drag component is absent from trajectory contributions")
        components = [np.asarray(trajectory.contributions[name], dtype=np.float64) for name in names]
        if any(component.shape != (len(trajectory.dates),) for component in components):
            raise ValueError("cost_drag components must align with trajectory dates")
        matrix_cost = np.vstack(components)
        if np.any(~np.isfinite(matrix_cost)) or np.any(matrix_cost < 0):
            raise ValueError("cost_drag components must be finite nonnegative magnitudes")
        values = tuple(np.sum(matrix_cost, axis=0))
    else:
        contribution_key = (expected_leg if expected_scope == CostScope.GROSS_DIAGNOSTIC
                            else f"net_{expected_leg}")
        if contribution_key not in trajectory.contributions:
            raise ValueError(f"trajectory does not contain requested leg: {contribution_key}")
        values = trajectory.contributions[contribution_key]
    matrix = np.asarray(values, dtype=np.float64)[:, None]
    return ProbePortfolioArtifact(
        matrix, time_index=trajectory.dates, factor_ids=requested_ids,
        metric_id=f"probe_portfolio.{expected_portfolio_profile.lower()}.{expected_cost_profile}",
        provenance={
            "trajectory_artifact_id": trajectory.artifact_id,
            "scenario_id": trajectory.scenario_id,
            "portfolio_profile": trajectory.profile,
            "cost_profile": expected_cost_profile,
            "cost_scope": trajectory.scope.value,
            "source_fingerprint": trajectory.refs.source_fingerprint,
            "portfolio_ref": trajectory.refs.portfolio_ref,
            "execution_ref": trajectory.refs.execution_ref,
            "cost_ref": trajectory.refs.cost_ref,
            "benchmark_ref": trajectory.refs.benchmark_ref,
            "borrow_ref": trajectory.refs.borrow_ref,
            "statuses": trajectory.statuses,
            "leg": expected_leg,
        },
    )


def trajectories_to_probe_artifact(trajectories, *, expected_portfolio_profile,
                                   expected_cost_profile, factor_ids, expected_leg="unspecified"):
    """Explicitly concatenate independently validated per-factor trajectories."""
    requested = tuple(factor_ids)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("factor_ids must be non-empty and unique")
    if set(trajectories) != set(requested):
        raise ValueError("trajectory mapping must exactly match requested factor_ids")
    columns, artifacts = [], []
    for factor_id in requested:
        if factor_id not in trajectories:
            raise ValueError(f"missing trajectory for factor {factor_id}")
        artifact = trajectory_to_probe_artifact(
            trajectories[factor_id], expected_portfolio_profile=expected_portfolio_profile,
            expected_cost_profile=expected_cost_profile, factor_ids=(factor_id,),
            expected_leg=expected_leg)
        artifacts.append(artifact); columns.append(artifact.values[:, 0])
    if any(a.time_index != artifacts[0].time_index for a in artifacts[1:]):
        raise ValueError("factor trajectories do not share time coordinates")
    return ProbePortfolioArtifact(np.column_stack(columns), time_index=artifacts[0].time_index,
                                  factor_ids=tuple(factor_ids), metric_id=artifacts[0].metric_id,
                                  provenance={"per_factor_trajectory_refs": {
                                      fid: a.provenance["trajectory_artifact_id"] for fid, a in zip(factor_ids, artifacts)},
                                              "leg": expected_leg, "cost_profile": expected_cost_profile,
                                              "portfolio_profile": expected_portfolio_profile})


__all__ = ["trajectory_to_probe_artifact", "trajectories_to_probe_artifact"]
