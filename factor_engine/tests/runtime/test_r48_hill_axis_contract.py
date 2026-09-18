"""Hill's legal parameter domain and metadata axis match across CPU backends."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators.polars_native.ts_advanced_batch5 import TSHillTailIndexPolarsNative
from factor_engine.cleaned_operators.extreme_tail import TsHillTailIndex


@pytest.mark.parametrize("time_name", ["date", "timestamp", "__fe_time__", "trade_date"])
def test_small_legal_window_preserves_time_and_matches_independent_hill(time_name):
    values = np.arange(1., 21.)
    dates = pd.date_range("2024-01-01", periods=20)
    frame = pl.DataFrame({time_name: dates, "B": values, "A": -values})
    params = dict(window=10, side="upper", tail_fraction=.4, min_tail_count=3)
    actual = TSHillTailIndexPolarsNative().calculate(frame, **params)
    assert actual[time_name].equals(frame[time_name])
    chunk = values[-10:]
    threshold = np.quantile(chunk, .6)
    expected = np.mean(np.log(chunk[chunk > threshold]/threshold))
    assert actual["B"][-1] == pytest.approx(expected)
    reference = TsHillTailIndex().calculate(pd.DataFrame({"B": values, "A": -values}, index=dates), **params)
    np.testing.assert_allclose(actual.select("B","A").to_numpy(), reference.to_numpy(), equal_nan=True)


def test_hill_declares_same_parameter_contract_and_rejects_multisymbol_long():
    left = TSHillTailIndexPolarsNative()
    right = TsHillTailIndex()
    assert left.metadata.param_specs == right.metadata.param_specs
    frame = pl.DataFrame({"timestamp": [1,1,2,2], "symbol": ["A","B","A","B"], "value": [1.,2.,3.,4.]})
    with pytest.raises(ValueError, match="multi-stock long input"):
        left.calculate(frame, window=20)
