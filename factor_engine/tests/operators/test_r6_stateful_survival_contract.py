import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.stateful.survival import _survival_kernel
from factor_engine.runtime.execution_contract import execution_contract, execution_contract_overrides


def _state(values):
    return pd.DataFrame({"A": values}, index=pd.date_range("2026-01-01", periods=len(values)))


def test_contracts_defaults_backends_and_full_history():
    defaults = {"ts_state_age_percentile": 5, "ts_state_exit_hazard": 5, "ts_state_residual_life": 20}
    for name, minimum in defaults.items():
        op = OperatorRegistry.get(name, "pandas_numpy"); twin = OperatorRegistry.get(name, "polars")
        assert op.metadata.param_specs["history_window"].default == 60
        assert op.metadata.param_specs["min_completed_runs"].default == minimum
        assert twin.metadata.param_specs == op.metadata.param_specs
        assert execution_contract(name).chunking == "required_full_history"
        assert execution_contract_overrides()[name]["history_kind"] == "full_history"


def test_independent_completed_run_age_hazard_and_residual_with_gap():
    # Two observed completed runs L={2,3}; gap-censored run is never recorded.
    values = [0, 1, 1, 0, 1, 1, 1, 0, 1, np.nan, 1, 1]
    age, pct, hazard, residual = _survival_kernel(np.asarray(values, float), 10, 2, 2, 2, 1.0)
    assert np.isnan(age[9]) and np.isnan(age[10])
    assert pct[10] == pytest.approx(0.0) and pct[11] == pytest.approx(0.5)
    assert hazard[10] == pytest.approx(0.25)  # D1=0, N1=2, alpha=1
    assert residual[10] == pytest.approx(1.5) # mean({2,3}-1)


def test_default_polars_parity_and_axis_preservation():
    values = np.tile([0, 1, 1, 0], 8).astype(float)
    state = _state(values)
    pstate = pl.from_pandas(state.rename_axis("date").reset_index())
    for name in ("ts_state_age_percentile", "ts_state_exit_hazard", "ts_state_residual_life"):
        expected = OperatorRegistry.get(name, "pandas_numpy").calculate(state)
        actual = OperatorRegistry.get(name, "polars").calculate(pstate)
        assert actual["date"].to_list() == list(state.index.to_pydatetime())
        np.testing.assert_allclose(actual["A"], expected["A"], equal_nan=True)


@pytest.mark.parametrize("name,kwargs", [
    ("ts_state_age_percentile", {"history_window": 1.5}),
    ("ts_state_exit_hazard", {"alpha": -1.0}),
    ("ts_state_exit_hazard", {"alpha": np.inf}),
    ("ts_state_residual_life", {"gap_policy": "drop"}),
])
def test_domains_reject(name, kwargs):
    with pytest.raises((TypeError, ValueError)):
        OperatorRegistry.get(name, "pandas_numpy").calculate(_state([0, 1, 0]), **kwargs)
