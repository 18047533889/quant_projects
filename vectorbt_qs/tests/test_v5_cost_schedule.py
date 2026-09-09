from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
import importlib.util
from pathlib import Path
import sys

from vectorbt_qs.contracts.costs import (
    FeeSchedule, FeeScheduleEntry, FixedSlippageProfile, FillForBilling,
    ordinary_ashare_research_schedule,
)
from vectorbt_qs.contracts.trajectories import TrajectoryRefs, build_trajectory_from_execution_plan
_spec = importlib.util.spec_from_file_location(
    "v5_execution_engine", Path(__file__).parents[1] / "mvp" / "engine" / "execution.py"
)
_engine = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = _engine
_spec.loader.exec_module(_engine)
ExecutionCosts = _engine.ExecutionCosts
plan_ashare_orders_python = _engine.plan_ashare_orders_python


def fill(day, side, notional, group="o1"):
    return FillForBilling(date.fromisoformat(day), "CN_SH_SZ", "ORDINARY_A", "RESEARCH", side, Decimal(str(notional)), group)


def test_t10_buy_sell_rates_and_small_order_minimum():
    schedule = ordinary_ashare_research_schedule()
    slippage = FixedSlippageProfile().cost_cny(Decimal("100000"))
    assert schedule.bill([fill("2024-01-02", "buy", 100_000)]).total_cny + slippage == Decimal("76.00")
    assert schedule.bill([fill("2024-01-02", "sell", 100_000)]).total_cny + slippage == Decimal("126.00")
    small = schedule.bill([fill("2024-01-02", "buy", 1000)])
    assert small.commission_cny == Decimal("5.00")
    assert small.total_cny == Decimal("5.01")


def test_t11_effective_date_stamp_split_and_buy_zero_stamp():
    schedule = ordinary_ashare_research_schedule()
    assert schedule.bill([fill("2023-08-27", "sell", 100_000)]).stamp_tax_cny == Decimal("100.00")
    assert schedule.bill([fill("2023-08-28", "sell", 100_000)]).stamp_tax_cny == Decimal("50.00")
    assert schedule.bill([fill("2023-08-27", "buy", 100_000)]).stamp_tax_cny == 0
    assert schedule.bill([fill("2022-04-28", "buy", 100_000)]).transfer_fee_cny == Decimal("2.00")
    assert schedule.bill([fill("2022-04-29", "buy", 100_000)]).transfer_fee_cny == Decimal("1.00")
    with pytest.raises(ValueError, match="exactly once"):
        schedule.bill([fill("2015-07-31", "buy", 100_000)])


def test_t12_multiple_fills_share_minimum_and_unknown_group_is_explicit():
    schedule = ordinary_ashare_research_schedule()
    grouped = schedule.bill([fill("2024-01-02", "buy", 500, "same"), fill("2024-01-02", "buy", 500, "same")])
    assert grouped.commission_cny == Decimal("5.00")
    separate = schedule.bill([fill("2024-01-02", "buy", 500, "a"), fill("2024-01-02", "buy", 500, "b")])
    assert separate.commission_cny == Decimal("10.00")
    unknown = schedule.bill([fill("2024-01-02", "buy", 1000, None)])
    assert unknown.status == "MIN_FEE_NOT_MODELED"
    assert unknown.commission_cny == Decimal("0.25")
    zero = schedule.bill([fill("2024-01-02", "buy", 0, "zero")])
    assert zero.total_cny == 0
    base = schedule.resolve(fill("2024-01-02", "buy", 100_000))
    included = FeeSchedule("included", (FeeScheduleEntry(
        **{**base.__dict__, "included_components": frozenset({"transfer_fee"})}
    ),))
    assert included.bill([fill("2024-01-02", "buy", 100_000)]).transfer_fee_cny == 0


def test_actual_python_planner_uses_trade_date_schedule():
    idx = pd.DatetimeIndex(["2023-08-27", "2023-08-28"])
    cols = ["000001.SZ"]
    frame = lambda value: pd.DataFrame(value, index=idx, columns=cols)
    targets = pd.DataFrame([[1.0], [0.0]], index=idx, columns=cols)
    plan = plan_ashare_orders_python(
        frame(10.0), frame(10.0), targets, frame(False), frame(20.0), frame(1.0),
        init_cash=100_000, costs=ExecutionCosts(fee_schedule=ordinary_ashare_research_schedule()),
        lot_size=100, slippage=0.0,
    )
    sell = plan.log.loc[plan.log.side == "sell"].iloc[0]
    assert sell.date == pd.Timestamp("2023-08-28")
    assert plan.fees.loc[pd.Timestamp("2023-08-28"), "000001.SZ"] == pytest.approx(0.00076)
    trajectory = build_trajectory_from_execution_plan(
        plan=plan, order_price=frame(10.0), valuation_price=frame(10.0), gross_return=(0.0, 0.0),
        benchmark_return=(0.0, 0.0), initial_nav=100_000, slippage=0.0,
        scenario_id="base", refs=TrajectoryRefs(("f",), "sha256:data", "plan:1", "fee:v5", "bench:pit"),
    )
    assert trajectory.contributions["explicit_fees"][1] > 0
    assert trajectory.net_return[1] < trajectory.gross_return[1]
    stressed = build_trajectory_from_execution_plan(
        plan=plan, order_price=frame(10.0), valuation_price=frame(10.0), gross_return=(0.0, 0.0),
        benchmark_return=(0.0, 0.0), initial_nav=100_000, slippage=0.002,
        scenario_id="stress", refs=TrajectoryRefs(("f",), "sha256:data", "plan:1", "fee:v5-stress", "bench:pit"),
    )
    assert stressed.gross_return == trajectory.gross_return
    assert all(stressed.net_return[i] <= trajectory.net_return[i] for i in range(2))
