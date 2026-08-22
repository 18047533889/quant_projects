# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _panel(values):
    return pd.DataFrame(values, index=pd.date_range("2024-01-01", periods=len(values)), columns=["A", "B"])


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    from cleaned_operators import load_all
    load_all()


def test_ashare_ratio_and_state_operators():
    from cleaned_operators.registry import OperatorRegistry

    pe = _panel([[10.0, -2.0], [20.0, 0.0]])
    out = OperatorRegistry.get("earnings_yield").calculate(pe)
    assert out.iloc[0, 0] == pytest.approx(0.1)
    assert np.isnan(out.iloc[0, 1])

    close = _panel([[10.0, 9.0], [8.0, 7.0]])
    upper = _panel([[10.0, 10.0], [9.0, 7.0]])
    state = OperatorRegistry.get("limit_up_state").calculate(close, upper)
    assert state.iloc[0].tolist() == [1.0, 0.0]
    assert state.iloc[1].tolist() == [0.0, 1.0]


def test_true_turnover_and_shareholder_change():
    from cleaned_operators.registry import OperatorRegistry

    volume = _panel([[10.0, 20.0], [20.0, 40.0]])
    shares = _panel([[100.0, 200.0], [100.0, 200.0]])
    turnover = OperatorRegistry.get("true_turnover_rate").calculate(volume, shares)
    assert np.allclose(turnover.to_numpy(), [[0.1, 0.1], [0.2, 0.2]])


def test_holder_change_is_snapshot_based():
    from storage.sources.relation import relation_snapshot_change

    rows = pd.DataFrame({
        "instrument": ["A", "A", "A"],
        "snapshot_id": [1, 2, 2],
        "available_at": ["2024-01-01", "2024-04-01", "2024-04-02"],
        "concentration": [0.4, 0.45, 0.45],
    })
    result = relation_snapshot_change(
        rows, value_column="concentration", decision_time="2024-04-15"
    )
    assert len(result) == 2
    assert result.iloc[-1]["concentration_change"] == pytest.approx(0.05)
    # R24-030..032: a future revision of an old snapshot is NOT visible at an
    # earlier decision time (a full-sample keep-last would leak it).  At
    # 2024-03-15 only snapshot 1 (available 2024-01-01) is visible — snapshot 2
    # has no revision visible yet.
    old = relation_snapshot_change(
        rows, value_column="concentration", decision_time="2024-03-15"
    )
    assert len(old) == 1
    assert old.iloc[0]["snapshot_id"] == 1


def test_ashare_lowerings_registered():
    from planner.composite_lowering import lowered_primitives

    assert lowered_primitives("earnings_yield") == ("where", "gt", "safe_div_null")
    assert lowered_primitives("benchmark_excess_return") == ("subtract",)
    assert lowered_primitives("limit_up_state") == ("ge", "subtract")
    assert lowered_primitives("limit_up_close") == ("ge", "subtract")


def test_valid_trade_is_strict_tradable_bool() -> None:
    # NEW-050: valid_trade must be a TradableBool {0, 1, NaN}.  A 0.2 / -1 / 2
    # flag was previously read as "tradeable" (finite & != 0); it is now a
    # data-quality error, never silently true/false.
    from cleaned_operators.ashare.state_machine import _tradeable

    assert _tradeable(np.array([[1.0]]), 0, 0) is True
    assert _tradeable(np.array([[0.0]]), 0, 0) is False
    assert _tradeable(np.array([[np.nan]]), 0, 0) is False
    for bad in (0.2, -1.0, 2.0):
        with pytest.raises(ValueError):
            _tradeable(np.array([[bad]]), 0, 0)


def test_limit_up_streak_rejects_non_bool_valid_trade() -> None:
    # NEW-050/051 end-to-end: a non-bool valid_trade flag (0.2) must fail the
    # state machine loudly instead of being silently treated as tradeable.
    from cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get("ashare_limit_up_streak")
    close = _panel([[10.0, 10.0], [10.0, 10.0]])
    high_limit = _panel([[10.0, 10.0], [10.0, 10.0]])
    bad_trade = _panel([[0.2, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        op.calculate(close, high_limit, bad_trade)
