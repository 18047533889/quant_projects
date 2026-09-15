from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.stateful.rule_language import (
    StateDeadband,
    StateHold,
    StateLatch,
    StateSlewLimit,
)
from factor_engine.runtime.execution_contract import execution_contract, execution_contract_overrides


def _frame(values, name="A"):
    return pd.DataFrame({name: values}, index=pd.date_range("2026-01-01", periods=len(values)))


def _pl(frame: pd.DataFrame) -> pl.DataFrame:
    return pl.from_pandas(frame.rename_axis("date").reset_index())


def _numeric(frame: pl.DataFrame) -> np.ndarray:
    return frame.select(pl.exclude("date")).to_numpy()


def test_contracts_defaults_topology_and_full_history():
    expected = {
        "state_latch": (("set_condition", "reset_condition"), ("initial_state",), 0.0),
        "state_hold": (("value", "update_condition", "reset_condition"), (), None),
        "state_slew_limit": (("x", "limit"), (), 0.01),
        "state_deadband": (("x", "band"), (), 0.0),
    }
    for name, (panels, scalars, default) in expected.items():
        op = OperatorRegistry.get(name, "pandas_numpy")
        twin = OperatorRegistry.get(name, "polars")
        assert op is not None and twin is not None
        assert tuple(op.metadata.panel_params) == panels
        assert tuple(op.metadata.scalar_params) == scalars
        optional = {
            "state_hold": "reset_condition",
            "state_latch": "initial_state",
            "state_slew_limit": "limit",
            "state_deadband": "band",
        }[name]
        assert op.metadata.param_specs[optional].default == default
        assert twin.metadata.param_names == op.metadata.param_names
        assert twin.metadata.param_specs == op.metadata.param_specs
        contract = execution_contract(name)
        assert contract.state_model == "recursive"
        assert contract.chunking == "required_full_history"
        assert execution_contract_overrides()[name]["history_kind"] == "full_history"


def test_latch_independent_oracle_reset_priority_and_break():
    set_condition = _frame([0, 1, 0, 1, np.nan, 1, 0])
    reset_condition = _frame([0, 0, 0, 1, 0, 0, 1])
    expected = np.array([0, 1, 1, 0, np.nan, 1, 0], dtype=float)
    actual = StateLatch().calculate(set_condition, reset_condition)["A"].to_numpy()
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_hold_independent_oracle_optional_reset_and_unknown_update():
    value = _frame([10, 20, 30, 40, 50, 60, 70])
    update = _frame([0, 1, 0, 0, 1, np.nan, 1])
    reset = _frame([0, 0, 0, 1, 0, 0, 0])
    expected = np.array([np.nan, 20, 20, np.nan, 50, np.nan, 70])
    actual = StateHold().calculate(value, update, reset)["A"].to_numpy()
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_slew_and_deadband_independent_oracles_and_panel_thresholds():
    x = pd.DataFrame({"A": [0.0, 1.0, 1.0], "B": [0.0, 1.0, 1.0]})
    threshold = pd.DataFrame({"A": [0.1] * 3, "B": [0.25] * 3})
    slew = StateSlewLimit().calculate(x, limit=threshold)
    dead = StateDeadband().calculate(x, band=threshold)
    np.testing.assert_allclose(slew.to_numpy(), [[0, 0], [0.1, 0.25], [0.2, 0.5]])
    np.testing.assert_allclose(dead.to_numpy(), [[0, 0], [0.9, 0.75], [0.9, 0.75]])


def test_defaults_and_polars_delegate_match_independent_expected_values():
    x = _frame([0.0, 1.0, 1.0])
    zeros = _frame([0.0, 0.0, 0.0])
    ones = _frame([0.0, 1.0, 0.0])
    cases = [
        ("state_latch", (ones, zeros), np.array([0.0, 1.0, 1.0])),
        ("state_hold", (x, ones), np.array([np.nan, 1.0, 1.0])),
        ("state_slew_limit", (x,), np.array([0.0, 0.01, 0.02])),
        ("state_deadband", (x,), np.array([0.0, 1.0, 1.0])),
    ]
    for name, args, expected in cases:
        pandas_result = OperatorRegistry.get(name, "pandas_numpy").calculate(*args)
        polars_result = OperatorRegistry.get(name, "polars").calculate(*map(_pl, args))
        np.testing.assert_allclose(pandas_result["A"], expected, equal_nan=True)
        np.testing.assert_allclose(_numeric(polars_result)[:, 0], expected, equal_nan=True)


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (StateLatch, {"initial_state": True}),
        (StateLatch, {"initial_state": 2}),
        (StateSlewLimit, {"limit": -0.1}),
        (StateSlewLimit, {"limit": np.inf}),
        (StateDeadband, {"band": -0.1}),
        (StateDeadband, {"band": np.nan}),
    ],
)
def test_scalar_domains_fail_at_call_boundary(cls, kwargs):
    x = _frame([0.0, 1.0])
    args = (x, x) if cls is StateLatch else (x,)
    with pytest.raises((TypeError, ValueError)):
        cls().calculate(*args, **kwargs)
