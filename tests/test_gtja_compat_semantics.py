from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry


def _frame(values):
    return pd.DataFrame({"A": values}, index=pd.date_range("2024-01-01", periods=len(values)))


def test_arg_extreme_returns_days_since_current_and_recent_tie():
    load_all()
    argmax = OperatorRegistry.get("ts_argmax")
    argmin = OperatorRegistry.get("ts_argmin")
    x = _frame([3.0, 1.0, 3.0, 2.0])
    got_max = argmax.calculate(x, 4)["A"].to_numpy()
    got_min = argmin.calculate(x, 4)["A"].to_numpy()
    assert got_max[-1] == 1.0  # 最近的最大值在前一 bar
    assert got_min[-1] == 2.0


def test_arg_extreme_all_missing_is_null():
    load_all()
    x = _frame([np.nan, np.nan])
    assert OperatorRegistry.get("ts_argmax").calculate(x, 2).isna().all().all()
    assert OperatorRegistry.get("ts_argmin").calculate(x, 2).isna().all().all()


def test_time_slope_matches_linear_sequence():
    load_all()
    x = _frame([1.0, 3.0, 5.0, 7.0, 9.0])
    out = OperatorRegistry.get("ts_time_slope").calculate(x, 5)
    assert np.isclose(out.iloc[-1, 0], 2.0)


def test_pairwise_regression_uses_identical_sample_set():
    load_all()
    x = _frame([1.0, 2.0, 3.0, 4.0, 5.0])
    y = _frame([3.0, np.nan, 7.0, 9.0, 11.0])  # y = 2x+1 on valid pairs
    out = OperatorRegistry.get("ts_regression").calculate(
        y, x, 5, 0, "slope", min_periods=3
    )
    assert np.isclose(out.iloc[-1, 0], 2.0)


def test_product_preserves_zero_and_negative_sign():
    load_all()
    x = _frame([-2.0, 3.0, 0.0, -4.0])
    out = OperatorRegistry.get("ts_product").calculate(x, 2)["A"]
    assert np.isclose(out.iloc[1], -6.0)
    assert np.isclose(out.iloc[2], 0.0)
    assert np.isclose(out.iloc[3], 0.0)
