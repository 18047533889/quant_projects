from dataclasses import replace

import pytest

from quant_evaluator.adapters.execution_trajectory import trajectory_to_probe_artifact, trajectories_to_probe_artifact
from quant_evaluator.contracts.artifact_types import ExecutablePortfolioArtifact, ProbePortfolioArtifact
from vectorbt_qs.contracts.costs import CostScope
from vectorbt_qs.contracts.trajectories import PortfolioTrajectory, TrajectoryRefs, build_research_trajectory


def test_strict_profile_cost_and_research_ls_borrow_gate():
    refs = TrajectoryRefs(("f",), "sha256:data", "portfolio:1", "cost:v5", None, "fills:1", None)
    trajectory = build_research_trajectory(
        scenario_id="stress", profile="LONG_SHORT_RESEARCH", dates=("2024-01-02",),
        gross_return=(.01,), benchmark_return=None, cost_contributions={"borrow_assumed": (.001,)},
        refs=refs, long_contribution=(.02,), short_contribution=(-.01,),
    )
    artifact = trajectory_to_probe_artifact(
        trajectory, expected_portfolio_profile="LONG_SHORT_RESEARCH", expected_cost_profile="net-stress"
    )
    assert artifact.values[0, 0] == pytest.approx(.009)
    assert artifact.provenance["borrow_ref"] is None
    with pytest.raises(ValueError, match="BORROW_AVAILABILITY_NOT_PROVIDED"):
        trajectory_to_probe_artifact(
            trajectory, expected_portfolio_profile="LONG_SHORT_RESEARCH",
            expected_cost_profile="net-stress", require_borrow_evidence=True,
        )
    with pytest.raises(ValueError, match="portfolio_profile"):
        trajectory_to_probe_artifact(
            trajectory, expected_portfolio_profile="LONG_ONLY_RESEARCH", expected_cost_profile="net-stress"
        )


def test_d01_each_factor_requires_its_own_trajectory_and_leg_is_consumed():
    def make(fid, values):
        return build_research_trajectory(
            scenario_id="gross", profile="LONG_SHORT_RESEARCH", dates=("2024-01-01", "2024-01-02"),
            gross_return=values, benchmark_return=None, cost_contributions={}, scope=CostScope.GROSS_DIAGNOSTIC,
            refs=TrajectoryRefs((fid,), f"source:{fid}", f"portfolio:{fid}", "gross", None),
            long_contribution=(.01, .02), short_contribution=(values[0]-.01, values[1]-.02))
    paths = {"f": make("f", (.03, .04)), "minus-f": make("minus-f", (-.03, -.04)),
             "third": make("third", (.1, -.1))}
    artifact = trajectories_to_probe_artifact(paths, expected_portfolio_profile="LONG_SHORT_RESEARCH",
                                              expected_cost_profile="gross", factor_ids=("f", "minus-f", "third"))
    assert artifact.values[:, 0].tolist() == [.03, .04]
    assert artifact.values[:, 1].tolist() == [-.03, -.04]
    assert trajectory_to_probe_artifact(paths["f"], expected_portfolio_profile="LONG_SHORT_RESEARCH",
                                        expected_cost_profile="gross", expected_leg="long").values[:, 0].tolist() == [.01, .02]


def test_net_leg_requires_an_explicit_net_contribution():
    trajectory = build_research_trajectory(
        scenario_id="base", profile="LONG_SHORT_RESEARCH", dates=("2024-01-01",),
        gross_return=(.01,), benchmark_return=None, cost_contributions={"fees": (.002,)},
        refs=TrajectoryRefs(("f",), "source:f", "portfolio:f", "cost:base", None),
        long_contribution=(.02,), short_contribution=(-.01,),
    )
    with pytest.raises(ValueError, match="net_long"):
        trajectory_to_probe_artifact(
            trajectory, expected_portfolio_profile="LONG_SHORT_RESEARCH",
            expected_cost_profile="net-base", expected_leg="long",
        )
    with pytest.raises(ValueError, match="net_short"):
        trajectory_to_probe_artifact(
            trajectory, expected_portfolio_profile="LONG_SHORT_RESEARCH",
            expected_cost_profile="net-base", expected_leg="short",
        )

    trajectory = replace(
        trajectory,
        contributions={**trajectory.contributions, "net_long": (.018,), "net_short": (-.01,)},
    )
    artifact = trajectory_to_probe_artifact(
        trajectory, expected_portfolio_profile="LONG_SHORT_RESEARCH",
        expected_cost_profile="net-base", expected_leg="long",
    )
    assert artifact.values[:, 0].tolist() == [.018]


def test_gross_active_is_gross_minus_benchmark_not_net_active():
    trajectory = build_research_trajectory(
        scenario_id="gross", profile="LONG_ONLY_RESEARCH", dates=("2024-01-01",),
        gross_return=(.02,), benchmark_return=(.005,), cost_contributions={},
        refs=TrajectoryRefs(("f",), "source:f", "portfolio:f", "gross", "benchmark:1"),
        scope=CostScope.GROSS_DIAGNOSTIC,
    )
    artifact = trajectory_to_probe_artifact(
        trajectory, expected_portfolio_profile="LONG_ONLY_RESEARCH",
        expected_cost_profile="gross", expected_leg="active",
    )
    assert artifact.values[:, 0].tolist() == [.015]


def test_collection_requires_an_exact_nonempty_unique_mapping():
    def make(fid):
        return build_research_trajectory(
            scenario_id="gross", profile="LONG_SHORT_RESEARCH", dates=("2024-01-01",),
            gross_return=(.01,), benchmark_return=None, cost_contributions={},
            refs=TrajectoryRefs((fid,), f"source:{fid}", f"portfolio:{fid}", "gross", None),
            scope=CostScope.GROSS_DIAGNOSTIC,
            long_contribution=(.02,), short_contribution=(-.01,),
        )

    paths = {"f": make("f"), "g": make("g")}
    common = dict(expected_portfolio_profile="LONG_SHORT_RESEARCH", expected_cost_profile="gross")
    with pytest.raises(ValueError, match="non-empty and unique"):
        trajectories_to_probe_artifact({}, factor_ids=(), **common)
    with pytest.raises(ValueError, match="non-empty and unique"):
        trajectories_to_probe_artifact({"f": paths["f"]}, factor_ids=("f", "f"), **common)
    with pytest.raises(ValueError, match="exactly match"):
        trajectories_to_probe_artifact({"f": paths["f"]}, factor_ids=("f", "g"), **common)
    with pytest.raises(ValueError, match="exactly match"):
        trajectories_to_probe_artifact(paths, factor_ids=("f",), **common)


def test_v13_executable_scope_retains_a_distinct_certified_artifact_identity():
    refs = TrajectoryRefs(("f",), "source:f", "portfolio:f", "cost:actual",
                          None, "execution-ledger:1", "borrow-snapshot:1")
    trajectory = PortfolioTrajectory(
        scenario_id="actual", scope=CostScope.NET_EXECUTABLE,
        profile="LONG_SHORT_RESEARCH", dates=("2024-01-02",),
        gross_return=(.01,), net_return=(.009,), nav=(1.009,),
        benchmark_return=None, active_return=None, relative_wealth=None,
        contributions={"fees": (.001,), "long": (.02,), "short": (-.01,)},
        refs=refs, statuses=(), executable_certified=True, cost_component_names=("fees",),
    )
    artifact = trajectory_to_probe_artifact(
        trajectory, expected_portfolio_profile="LONG_SHORT_RESEARCH",
        expected_cost_profile="net-executable",
    )
    assert isinstance(artifact, ExecutablePortfolioArtifact)
    assert not type(artifact) is ProbePortfolioArtifact
    assert artifact.provenance["execution_certified"] is True
    assert artifact.provenance["execution_ref"] == "execution-ledger:1"


def test_v13_assumed_cost_trajectory_remains_probe_only():
    trajectory = build_research_trajectory(
        scenario_id="assumed", profile="LONG_SHORT_RESEARCH", dates=("2024-01-02",),
        gross_return=(.01,), benchmark_return=None, cost_contributions={"fees": (.001,)},
        refs=TrajectoryRefs(("f",), "source:f", "portfolio:f", "cost:assumed", None),
        long_contribution=(.02,), short_contribution=(-.01,),
    )
    artifact = trajectory_to_probe_artifact(
        trajectory, expected_portfolio_profile="LONG_SHORT_RESEARCH",
        expected_cost_profile="net-base",
    )
    assert type(artifact) is ProbePortfolioArtifact
    assert artifact.provenance["execution_certified"] is False


def test_portfolio_artifact_axes_are_mandatory_and_cannot_alias_columns():
    import numpy as np

    values = np.zeros((2, 2))
    with pytest.raises(ValueError, match="time_index"):
        ProbePortfolioArtifact(values, factor_ids=("f", "g"))
    with pytest.raises(ValueError, match="factor_ids"):
        ProbePortfolioArtifact(values, time_index=("d1", "d2"), factor_ids=("f",))
    with pytest.raises(ValueError, match="factor_ids"):
        ProbePortfolioArtifact(values, time_index=("d1", "d2"), factor_ids=("f", "f"))


def test_executable_serialization_identity_cannot_be_downgraded_to_probe():
    import numpy as np

    provenance = {"execution_certified": True, "cost_scope": "NET_EXECUTABLE",
                  "execution_ref": "ledger:1"}
    executable = ExecutablePortfolioArtifact(
        np.array([[.01]]), time_index=("d1",), factor_ids=("f",), provenance=provenance)
    probe = ProbePortfolioArtifact(
        np.array([[.01]]), time_index=("d1",), factor_ids=("f",),
        metric_id=executable.metric_id, provenance=provenance)
    assert executable.to_dict()["artifact_type"] == "ExecutablePortfolioArtifact"
    assert probe.to_dict()["artifact_type"] == "ProbePortfolioArtifact"
    assert hash(executable) != hash(probe)
    with pytest.raises(ValueError, match="cannot be loaded"):
        ProbePortfolioArtifact.from_dict(executable.to_dict())
    assert ExecutablePortfolioArtifact.from_dict(executable.to_dict()) == executable


def test_executable_batch_retains_each_factor_execution_ledger_ref():
    def make(fid, ledger):
        return PortfolioTrajectory(
            scenario_id=f"actual-{fid}", scope=CostScope.NET_EXECUTABLE,
            profile="LONG_SHORT_RESEARCH", dates=("2024-01-02",),
            gross_return=(.01,), net_return=(.009,), nav=(1.009,),
            benchmark_return=None, active_return=None, relative_wealth=None,
            contributions={"fees": (.001,), "long": (.02,), "short": (-.01,)},
            refs=TrajectoryRefs((fid,), f"source:{fid}", f"portfolio:{fid}",
                                f"cost:{fid}", None, ledger, f"borrow:{fid}"),
            statuses=(), executable_certified=True, cost_component_names=("fees",),
        )

    artifact = trajectories_to_probe_artifact(
        {"f": make("f", "ledger:f"), "g": make("g", "ledger:g")},
        expected_portfolio_profile="LONG_SHORT_RESEARCH",
        expected_cost_profile="net-executable", factor_ids=("f", "g"))
    assert artifact.provenance["per_factor_execution_refs"] == {
        "f": "ledger:f", "g": "ledger:g"}
    assert artifact.provenance["execution_ref"] == {"f": "ledger:f", "g": "ledger:g"}

    invalid = dict(artifact.provenance)
    invalid["execution_ref"] = "ledger:f"
    with pytest.raises(ValueError, match="factor-keyed ledger mapping"):
        ExecutablePortfolioArtifact(
            artifact.values, time_index=artifact.time_index,
            factor_ids=artifact.factor_ids, provenance=invalid)
