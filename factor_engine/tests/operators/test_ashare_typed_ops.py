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

    concentration = _panel([[0.4, 0.5], [0.45, 0.4]])
    change = OperatorRegistry.get("holder_concentration_change").calculate(concentration, 1)
    assert change.iloc[1].tolist() == pytest.approx([0.05, -0.1])


def test_ashare_lowerings_registered():
    from planner.composite_lowering import lowered_primitives

    assert lowered_primitives("earnings_yield") == ("where", "gt", "safe_div_null")
    assert lowered_primitives("benchmark_excess_return") == ("subtract",)
    assert lowered_primitives("limit_up_state") == ("ge",)
