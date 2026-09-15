"""Independent finalized-registry tests for A-share daily limit operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


NAMES = (
    "ashare_limit_up_touch",
    "ashare_limit_down_touch",
    "ashare_limit_one_price",
    "ashare_limit_failed",
    "ashare_open_at_upper_limit",
    "ashare_limit_open_failed",
)


def _panels():
    idx = pd.date_range("2024-01-02", periods=4, freq="D")
    frame = lambda values: pd.DataFrame({"A": values}, index=idx, dtype=float)
    return {
        "open": frame([10.0, 10.0, 9.0, 8.0]),
        "high": frame([10.0, 10.0, 9.8, 8.0]),
        "low": frame([10.0, 9.5, 8.0, 8.0]),
        "close": frame([10.0, 9.8, 9.0, 8.0]),
        "upper_limit": frame([10.0] * 4),
        "lower_limit": frame([8.0] * 4),
    }


def _calls(p):
    return {
        "ashare_limit_up_touch": ((p["high"], p["upper_limit"]), {"tick_tolerance": 0.005}),
        "ashare_limit_down_touch": ((p["low"], p["lower_limit"]), {"tick_tolerance": 0.005}),
        "ashare_limit_one_price": (
            (p["open"], p["high"], p["low"], p["close"], p["upper_limit"], p["lower_limit"]),
            {"side": "up", "tick_tolerance": 0.005},
        ),
        "ashare_limit_failed": ((p["high"], p["close"], p["upper_limit"]), {"tick_tolerance": 0.005}),
        "ashare_open_at_upper_limit": ((p["open"], p["upper_limit"]), {"tick_tolerance": 0.005}),
        "ashare_limit_open_failed": ((p["open"], p["low"], p["upper_limit"]), {"tick_tolerance": 0.005}),
    }


def test_finalized_owner_topology_and_backend():
    load_all()
    expected = {
        "ashare_limit_up_touch": (("high", "upper_limit"), ("tick_tolerance",)),
        "ashare_limit_down_touch": (("low", "lower_limit"), ("tick_tolerance",)),
        "ashare_limit_one_price": (("open", "high", "low", "close", "upper_limit", "lower_limit"), ("side", "tick_tolerance")),
        "ashare_limit_failed": (("high", "close", "upper_limit"), ("tick_tolerance",)),
        "ashare_open_at_upper_limit": (("open", "upper_limit"), ("tick_tolerance",)),
        "ashare_limit_open_failed": (("open", "low", "upper_limit"), ("tick_tolerance",)),
    }
    for name, (panels, scalars) in expected.items():
        op = OperatorRegistry.get(name, "pandas_numpy")
        assert type(op).__module__.endswith("ashare.limit_ops")
        assert tuple(op.metadata.panel_params) == panels
        assert tuple(op.metadata.scalar_params) == scalars
        assert op.metadata.panel_arity == len(panels)


def test_default_keyword_positional_and_prefix_equivalence():
    load_all()
    panels = _panels()
    for name, (args, kwargs) in _calls(panels).items():
        op = OperatorRegistry.get(name)
        positional = op.calculate(*args, *kwargs.values())
        keyword = op.calculate(**{k: panels[k] for k in op.metadata.panel_params}, **kwargs)
        defaults = op.calculate(*args)
        pd.testing.assert_frame_equal(positional, keyword, obj=name)
        pd.testing.assert_frame_equal(defaults, keyword, obj=f"{name}: defaults")
        prefix = op.calculate(*(x.iloc[:2] for x in args), **kwargs)
        pd.testing.assert_frame_equal(prefix, keyword.iloc[:2], obj=f"{name}: prefix")


def test_independent_limit_state_oracle():
    load_all()
    panels = _panels()
    expected = {
        "ashare_limit_up_touch": [1, 1, 0, 0],
        "ashare_limit_down_touch": [0, 0, 1, 1],
        "ashare_limit_one_price": [1, 0, 0, 0],
        "ashare_limit_failed": [0, 1, 0, 0],
        "ashare_open_at_upper_limit": [1, 1, 0, 0],
        "ashare_limit_open_failed": [0, 1, 0, 0],
    }
    for name, (args, kwargs) in _calls(panels).items():
        actual = OperatorRegistry.get(name).calculate(*args, **kwargs)["A"].to_numpy()
        np.testing.assert_array_equal(actual, np.asarray(expected[name], dtype=float), err_msg=name)
    down = OperatorRegistry.get("ashare_limit_one_price").calculate(
        panels["open"], panels["high"], panels["low"], panels["close"],
        panels["upper_limit"], panels["lower_limit"], side="down",
    )
    np.testing.assert_array_equal(down["A"].to_numpy(), np.asarray([0, 0, 0, 1], dtype=float))


def test_invalid_policy_and_tolerance_rejected():
    load_all()
    p = _panels()
    with pytest.raises((TypeError, ValueError)):
        OperatorRegistry.get("ashare_limit_up_touch").calculate(p["high"], p["upper_limit"], -0.001)
    with pytest.raises((TypeError, ValueError)):
        OperatorRegistry.get("ashare_limit_one_price").calculate(
            p["open"], p["high"], p["low"], p["close"], p["upper_limit"], p["lower_limit"], "sideways"
        )
