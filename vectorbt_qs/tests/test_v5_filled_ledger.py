from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from vectorbt_qs.contracts.costs import CostScope
from vectorbt_qs.contracts.trajectories import (
    TrajectoryRefs, build_trajectory_from_filled_ledger,
)

_spec = importlib.util.spec_from_file_location(
    "v5_filled_ledger_execution", Path(__file__).parents[1] / "mvp" / "engine" / "execution.py")
_engine = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = _engine
_spec.loader.exec_module(_engine)
ExecutionCosts = _engine.ExecutionCosts
plan_ashare_orders_python = _engine.plan_ashare_orders_python


def refs():
    return TrajectoryRefs(("factor:f",), "sha256:source", "portfolio:actual",
                          "cost:v5", "benchmark:pit", "execution:filled")


def frames(prices, columns=("A",)):
    index = pd.date_range("2024-01-02", periods=len(prices), freq="D")
    values = np.asarray(prices, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    price = pd.DataFrame(values, index=index, columns=columns)
    frame = lambda value: pd.DataFrame(value, index=index, columns=columns)
    return index, price, frame


def plan(prices, targets, *, costs=None, slippage=0.0, cash_dividend=None,
         share_multiplier=None, columns=("A",)):
    index, price, frame = frames(prices, columns)
    target = pd.DataFrame(targets, index=index[:len(targets)], columns=columns)
    result = plan_ashare_orders_python(
        price, price, target, frame(False), frame(1e9), frame(-1e9),
        init_cash=100.0, costs=costs or ExecutionCosts(0, 0, 0, 0),
        slippage=slippage, lot_size=1, cash_dividend=cash_dividend,
        share_multiplier=share_multiplier)
    return result, price


def trajectory(actual, price, *, scope=CostScope.NET_ASSUMED, slippage=0.0,
               valuation=None, benchmark=None):
    n = len(price)
    return build_trajectory_from_filled_ledger(
        plan=actual, order_price=price,
        valuation_price=price if valuation is None else valuation,
        initial_nav=100.0, slippage=slippage, scenario_id="actual",
        refs=refs(), benchmark_return=(0.0,) * n if benchmark is None else benchmark,
        scope=scope, annual_cash_borrow_rate=0.0)


def test_entry_before_jump_hold_and_exit_are_rebuilt_from_actual_fills():
    actual, price = plan([1, 2, 2, 2], [[1.0], [np.nan], [np.nan], [0.0]])
    assert actual.real_holdings["A"].tolist() == [100.0, 100.0, 100.0, 0.0]
    item = trajectory(actual, price)
    assert item.gross_return == pytest.approx((0.0, 1.0, 0.0, 0.0))
    assert item.net_return == pytest.approx(item.gross_return)
    assert item.nav == pytest.approx((1.0, 2.0, 2.0, 2.0))
    assert item.contributions["investment_fraction"] == pytest.approx((1.0, 1.0, 1.0, 0.0))


def test_partial_reduction_fees_use_each_previous_net_nav_denominator():
    costs = ExecutionCosts(.1, 0, 0, 0)
    actual, price = plan([1, 1, 1], [[.5], [.25], [0.0]], costs=costs)
    item = trajectory(actual, price)
    assert actual.order_size["A"].tolist() == pytest.approx((50, -27, -23))
    assert item.contributions["explicit_fees"] == pytest.approx((5/100, 2.7/95, 2.3/92.3))
    assert item.net_return == pytest.approx((-5/100, -2.7/95, -2.3/92.3))
    assert item.contributions["cash_cny"] == pytest.approx((45, 69.3, 90))


def test_slippage_is_charged_once_on_each_filled_side():
    actual, price = plan([1, 1], [[.5], [0.0]], slippage=.1)
    item = trajectory(actual, price, slippage=.1)
    assert actual.order_size["A"].tolist() == pytest.approx((50, -50))
    assert item.gross_return == pytest.approx((0.0, 0.0))
    assert item.contributions["implementation_slippage"] == pytest.approx((.05, 5/95))
    assert item.net_return == pytest.approx((-.05, -5/95))
    assert item.nav[-1] == pytest.approx(.90)


def test_gross_scope_investment_fraction_uses_gross_capital():
    actual, price = plan([1, 1], [[.5], [np.nan]], costs=ExecutionCosts(.1, 0, 0, 0))
    gross = trajectory(actual, price, scope=CostScope.GROSS_DIAGNOSTIC)
    net = trajectory(actual, price)
    assert gross.contributions["investment_fraction"] == pytest.approx((.5, .5))
    assert net.contributions["investment_fraction"] == pytest.approx((50/95, 50/95))
    assert gross.nav == pytest.approx((1, 1))


def test_post_signal_cash_dividend_and_share_distribution_are_not_dropped():
    index, price, frame = frames([1, 1, 1])
    cash_dividend = frame(0.0); cash_dividend.loc[index[1], "A"] = .1
    share_multiplier = frame(1.0); share_multiplier.loc[index[2], "A"] = 2.0
    actual, price = plan([1, 1, 1], [[1.0]], cash_dividend=cash_dividend,
                         share_multiplier=share_multiplier)
    assert actual.cash_deposits["A"].tolist() == pytest.approx((0, 10, 0))
    assert actual.asset_deposits["A"].tolist() == pytest.approx((0, 0, 100))
    assert actual.real_holdings["A"].tolist() == pytest.approx((100, 100, 200))
    item = trajectory(actual, price)
    assert item.gross_return == pytest.approx((0, .1, 100/110))


def test_missing_unheld_prices_are_allowed_but_held_valuation_fails():
    actual, price = plan([[1, np.nan], [1, np.nan]], [[1.0, 0.0], [np.nan, 0.0]],
                         columns=("A", "B"))
    assert trajectory(actual, price).nav[-1] == pytest.approx(1.0)
    bad = price.copy(); bad.loc[bad.index[1], "A"] = np.nan
    with pytest.raises(ValueError, match="held valuations"):
        trajectory(actual, price, valuation=bad)
    zero = price.copy(); zero.loc[zero.index[1], "A"] = 0.0
    with pytest.raises(ValueError, match="held valuations"):
        trajectory(actual, price, valuation=zero)


def test_negative_target_never_creates_a_short_fill():
    with pytest.raises(ValueError, match="不能为负"):
        plan([1], [[-0.1]])


def test_contributions_are_immutable_and_part_of_artifact_identity():
    actual, price = plan([1, 1], [[.5], [0.0]])
    item = trajectory(actual, price)
    with pytest.raises(TypeError):
        item.contributions["cash_cny"] = (0.0, 0.0)
    changed = dict(item.contributions)
    changed["cash_cny"] = (44.0, 100.0)
    other = replace(item, contributions=changed)
    assert other.artifact_id != item.artifact_id


def test_cost_component_names_are_frozen_validated_and_hashed():
    actual, price = plan([1, 1], [[.5], [0.0]], costs=ExecutionCosts(.1, 0, 0, 0))
    item = trajectory(actual, price)
    assert item.cost_component_names == (
        "explicit_fees", "implementation_slippage", "financing")
    assert isinstance(item.cost_component_names, tuple)
    reduced_policy = replace(item, cost_component_names=("explicit_fees",))
    assert reduced_policy.artifact_id != item.artifact_id
    with pytest.raises(ValueError, match="unique and present"):
        replace(item, cost_component_names=("not_a_contribution",))


def test_benchmark_active_and_relative_wealth_are_ledger_derived():
    actual, price = plan([1, 2], [[1.0], [np.nan]])
    item = trajectory(actual, price, benchmark=(0.0, .25))
    assert item.active_return == pytest.approx((0.0, .75))
    assert item.relative_wealth == pytest.approx((1.0, 1.6))
