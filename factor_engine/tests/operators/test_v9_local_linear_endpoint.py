import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.execution_contract import own_history_requirement


@pytest.fixture(scope="module")
def op():
    load_all()
    result = OperatorRegistry.get("ts_causal_local_linear_smoother", backend="pandas_numpy", mode="research")
    assert type(result).__module__ == "factor_engine.cleaned_operators.filter_smooth"
    return result


@pytest.mark.parametrize("gap", [[19], [5, 6, 7], [5, 6, 7, 19]])
def test_predicts_current_physical_endpoint_not_last_observation(op, gap):
    x = pd.DataFrame({"A": np.arange(20.)})
    x.iloc[gap, 0] = np.nan
    result = op.calculate(x, window=20, min_periods=10)
    assert result.iloc[-1, 0] == pytest.approx(19., abs=1e-12)


def test_exact_warmup_irregular_dates_and_future_poison(op):
    dates = pd.date_range("2021-01-01", periods=90, freq="2B")
    x = pd.DataFrame({"A": 2 * np.arange(90.) + 17}, index=dates)
    x.iloc[[25, 26, 55, 56, 89], 0] = np.nan
    full = op.calculate(x, window=20, min_periods=10)
    req = own_history_requirement("ts_causal_local_linear_smoother", {"window":20, "min_periods":10})
    assert req.rows == 19 and not req.is_full_history
    chunk = op.calculate(x.iloc[50-req.rows:], window=20, min_periods=10)
    pd.testing.assert_frame_equal(full.iloc[50:], chunk.loc[full.index[50:]])
    pd.testing.assert_frame_equal(full.iloc[:60], op.calculate(x.iloc[:60], window=20, min_periods=10))
    assert full.iloc[-1, 0] == pytest.approx(195.)


@pytest.mark.parametrize("level", [1e12, 3e307])
def test_large_constant_is_finite_and_translation_preserved(op, level):
    x = pd.DataFrame({"A": np.full(20, level)})
    x.iloc[-1, 0] = np.nan
    out = op.calculate(x, window=20, min_periods=10)
    assert out.iloc[-1, 0] == pytest.approx(level, rel=1e-14)
