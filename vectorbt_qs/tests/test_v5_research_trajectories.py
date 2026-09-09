import pytest
import numpy as np

from vectorbt_qs.contracts.costs import CostScope
from vectorbt_qs.contracts.trajectories import (
    PortfolioTrajectory, TrajectoryRefs, actual_calendar_carry_cost,
    build_research_trajectory,
)


REFS = TrajectoryRefs(("factor-a",), "sha256:data", "portfolio:1", "cost:v5-base", "benchmark:pit")


def test_t07_t08_long_only_active_relative_and_ls_signed_contributions():
    lo = build_research_trajectory(
        scenario_id="base", profile="LONG_ONLY_RESEARCH", dates=("2024-01-02", "2024-01-03"),
        gross_return=(0.01, -0.02), benchmark_return=(0.01, -0.02),
        cost_contributions={"commission": (0.001, 0.0)}, refs=REFS,
    )
    assert lo.active_return == pytest.approx((-0.001, 0.0))
    assert lo.relative_wealth[0] == pytest.approx(1.009 / 1.01)
    ls = build_research_trajectory(
        scenario_id="base", profile="LONG_SHORT_RESEARCH", dates=("2024-01-02",),
        gross_return=(0.03,), benchmark_return=None, cost_contributions={"tax": (0.001,)}, refs=REFS,
        long_contribution=(0.01,), short_contribution=(0.02,),
    )
    assert ls.contributions["long"][0] + ls.contributions["short"][0] == pytest.approx(ls.gross_return[0])


def test_t13_increasing_cost_never_increases_net_on_same_trajectory():
    kwargs = dict(scenario_id="x", profile="LONG_ONLY_RESEARCH", dates=("2024-01-02",),
                  gross_return=(0.01,), benchmark_return=(0.0,), refs=REFS)
    base = build_research_trajectory(cost_contributions={"slippage": (0.0005,)}, **kwargs)
    stress = build_research_trajectory(cost_contributions={"slippage": (0.002,)}, **kwargs)
    assert stress.gross_return == base.gross_return
    assert stress.net_return[0] < base.net_return[0]


def test_t14_assumed_borrow_is_not_executable_and_actual_calendar_financing():
    with pytest.raises(ValueError, match="NET_EXECUTABLE"):
        PortfolioTrajectory("x", CostScope.NET_EXECUTABLE, "LONG_SHORT_RESEARCH", ("2024-01-05",),
                            (0.0,), (0.0,), (1.0,), None, None, None, {}, REFS, (), False)
    # Fri->Mon is three actual calendar days; positive cash is not financed.
    assert actual_calendar_carry_cost((-365.0, -365.0), ("2024-01-05", "2024-01-08"), .04, negative_only=True)[1] == pytest.approx(.12)
    assert actual_calendar_carry_cost((365.0, 365.0), ("2024-01-05", "2024-01-08"), .04, negative_only=True)[1] == 0.0
    # Assumed stock-borrow accrues at 8% over the same actual-calendar weekend.
    assert actual_calendar_carry_cost((365.0, 365.0), ("2024-01-05", "2024-01-08"), .08, negative_only=False)[1] == pytest.approx(.24)


def test_t08_bottom_holding_and_signed_short_correlations_have_opposite_sign():
    long = np.array([.01, .02, -.01, .03])
    bottom_holding = np.array([.02, .01, -.02, .04])
    signed_short = -.5 * bottom_holding
    assert np.corrcoef(long, bottom_holding)[0, 1] == pytest.approx(-np.corrcoef(long, signed_short)[0, 1])


def test_nav_and_active_identities_are_fail_closed():
    with pytest.raises(ValueError, match="active return identity"):
        PortfolioTrajectory("x", CostScope.NET_ASSUMED, "LONG_ONLY_RESEARCH", ("2024-01-01",),
                            (0.0,), (0.0,), (1.0,), (0.0,), (1.0,), (1.0,), {}, REFS, ())
