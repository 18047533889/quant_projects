from dataclasses import replace

import pytest

from quant_evaluator.adapters.execution_trajectory import trajectory_to_probe_artifact, trajectories_to_probe_artifact
from vectorbt_qs.contracts.costs import CostScope
from vectorbt_qs.contracts.trajectories import TrajectoryRefs, build_research_trajectory


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
