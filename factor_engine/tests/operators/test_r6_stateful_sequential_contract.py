from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.stateful.sequential import (
    StateEwmIf,
    TsCusumPressure,
    TsLagOfPeakCorr,
    TsRankIf,
)
from factor_engine.runtime.execution_contract import execution_contract, execution_contract_overrides


def _df(values, *, index=None, name="A"):
    if index is None:
        index = pd.date_range("2026-01-01", periods=len(values))
    return pd.DataFrame({name: values}, index=index)


def _pl(frame):
    return pl.from_pandas(frame.rename_axis("date").reset_index())


def test_complete_contracts_and_history_models():
    expected = {
        "ts_cusum_pressure": (("x",), ("reference_window", "drift", "min_periods")),
        "ts_rank_if": (("x", "condition"), ("window", "min_periods")),
        "state_ewm_if": (("x", "condition"), ("half_life",)),
        "ts_lag_of_peak_corr": (("x", "y"), ("window", "max_lag", "min_periods")),
    }
    for name, (panels, scalars) in expected.items():
        pandas_op = OperatorRegistry.get(name, "pandas_numpy")
        polars_op = OperatorRegistry.get(name, "polars")
        assert tuple(pandas_op.metadata.panel_params) == panels
        assert tuple(pandas_op.metadata.scalar_params) == scalars
        assert polars_op.metadata.param_names == pandas_op.metadata.param_names
        assert polars_op.metadata.param_specs == pandas_op.metadata.param_specs
    for name in ("ts_cusum_pressure", "state_ewm_if"):
        contract = execution_contract(name)
        assert (contract.state_model, contract.chunking) == ("recursive", "required_full_history")
        assert execution_contract_overrides()[name]["history_kind"] == "full_history"
    assert OperatorRegistry.get("ts_lag_of_peak_corr", "pandas_numpy").metadata.param_specs["window"].history_formula == "window + max_lag"


def test_cusum_independent_recurrence_and_scale_invariance():
    values = np.array([1, 2, 3, 4, 5, 8, 9], dtype=float)
    expected = np.full(values.size, np.nan)
    sp = sm = 0.0
    for row in range(values.size):
        past = values[max(0, row - 4):row]
        if past.size < 2:
            sp = sm = 0.0
            continue
        scale0 = np.max(np.abs(past))
        normalized = past / scale0
        med = np.median(normalized)
        mad = np.median(np.abs(normalized - med))
        if mad == 0:
            sp = sm = 0.0
            continue
        z = (values[row] / scale0 - med) / (1.4826 * mad)
        sp = max(0.0, sp + z - 0.25)
        sm = min(0.0, sm + z + 0.25)
        expected[row] = sp + sm
    op = TsCusumPressure()
    actual = op.calculate(_df(values), reference_window=4, drift=0.25, min_periods=2)["A"]
    tiny = op.calculate(_df(values * 1e-200), reference_window=4, drift=0.25, min_periods=2)["A"]
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    np.testing.assert_allclose(tiny, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_rank_if_midrank_oracle_and_unknown_condition_exclusion():
    x = _df([1, 2, 2, 4, 3])
    condition = _df([1, 1, 1, np.nan, 1])
    actual = TsRankIf().calculate(x, condition, window=4, min_periods=2)["A"].to_numpy()
    expected = np.array([np.nan, 0.75, 2 / 3, 1.0, 5 / 6])
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_ewm_if_independent_recurrence_break_and_carry():
    x = _df([10, 20, 30, 40, np.nan, 60, 70])
    condition = _df([0, 1, 0, 1, 0, 1, 0])
    alpha = 1.0 - np.exp(-np.log(2.0) / 2.0)
    expected = np.array([np.nan, 20, 20, alpha * 40 + (1 - alpha) * 20, np.nan, 60, 60])
    actual = StateEwmIf().calculate(x, condition, half_life=2.0)["A"]
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_lag_of_peak_corr_recovers_known_past_lag():
    rng = np.random.default_rng(7)
    yv = rng.normal(size=40)
    xv = np.r_[np.nan, np.nan, yv[:-2]]
    actual = TsLagOfPeakCorr().calculate(_df(xv), _df(yv), window=16, max_lag=4, min_periods=8)
    assert actual["A"].iloc[-1] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("cls", "args", "kwargs"),
    [
        (TsCusumPressure, (_df(range(8)),), {"reference_window": 4.5}),
        (TsCusumPressure, (_df(range(8)),), {"drift": -0.1}),
        (TsRankIf, (_df(range(8)), _df([1] * 8)), {"window": 3, "min_periods": 4}),
        (StateEwmIf, (_df(range(8)), _df([1] * 8)), {"half_life": 0.0}),
        (TsLagOfPeakCorr, (_df(range(8)), _df(range(8))), {"window": 5, "max_lag": 4}),
    ],
)
def test_domains_fail_at_boundary(cls, args, kwargs):
    with pytest.raises((TypeError, ValueError)):
        cls().calculate(*args, **kwargs)


def test_default_polars_calls_match_pandas_and_axes_are_preserved():
    rng = np.random.default_rng(11)
    x = _df(rng.normal(size=30))
    y = _df(rng.normal(size=30))
    condition = _df((np.arange(30) % 3 == 0).astype(float))
    cases = [
        ("ts_cusum_pressure", (x,)),
        ("ts_rank_if", (x, condition)),
        ("state_ewm_if", (x, condition)),
        ("ts_lag_of_peak_corr", (x, y)),
    ]
    for name, args in cases:
        expected = OperatorRegistry.get(name, "pandas_numpy").calculate(*args)
        actual = OperatorRegistry.get(name, "polars").calculate(*map(_pl, args))
        assert actual["date"].to_list() == list(x.index.to_pydatetime())
        np.testing.assert_allclose(actual["A"].to_numpy(), expected["A"], equal_nan=True)


def test_mismatched_panel_axes_rejected():
    x = _df([1, 2, 3])
    shifted = _df([1, 2, 3], index=pd.date_range("2026-01-02", periods=3))
    with pytest.raises(ValueError):
        TsRankIf().calculate(x, shifted)
