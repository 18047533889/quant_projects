import numpy as np
import polars as pl

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


def test_polars_rolling_delegate_preserves_positional_scalar_window():
    load_all()
    op = OperatorRegistry.get("ts_max_drawdown", "polars")
    panel = pl.DataFrame({"date": range(12), "a": np.arange(1.0, 13.0)})
    positional = op._calculate_series(panel, 4)
    keyword = op._calculate_series(panel, window=4)
    assert positional.equals(keyword)


def test_polars_rolling_delegate_preserves_multi_panel_order():
    load_all()
    op = OperatorRegistry.get("ts_corr", "polars")
    x = pl.DataFrame({"date": range(6), "a": np.arange(6.0)})
    y = pl.DataFrame({"date": range(6), "a": -np.arange(6.0)})
    got = op._calculate_series(x, y, 3)
    assert got["a"][-1] == -1.0
