import numpy as np
import pandas as pd
import polars as pl

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _values(result):
    if isinstance(result, pl.DataFrame):
        return result.select("A").to_numpy().ravel()
    return result["A"].to_numpy()


def _active_pair(canonical):
    load_all()
    return (
        OperatorRegistry.get(canonical, backend="pandas_numpy"),
        OperatorRegistry.get(canonical, backend="polars"),
    )


def test_active_gap_fill_ratio_parity_covers_window_zero_gap_and_nan():
    """Protect the active four-argument canonical, not stale source classes."""
    pandas_op, polars_op = _active_pair("ts_gap_fill_ratio")
    close = pd.DataFrame({"A": [10.0, 12.0, 10.0, np.nan, 11.0, 10.0]})
    open_ = pd.DataFrame({"A": [10.0, 11.0, 11.0, 10.0, 10.0, 11.0]})
    pre_close = pd.DataFrame({"A": [10.0, 10.0, 12.0, 10.0, 9.0, 11.0]})

    pandas_w3 = pandas_op.calculate(close, open_, pre_close, 3)
    polars_w3 = polars_op.calculate(
        pl.from_pandas(close), pl.from_pandas(open_), pl.from_pandas(pre_close), 3
    )
    pandas_w2 = pandas_op.calculate(close, open_, pre_close, 2)
    polars_w2 = polars_op.calculate(
        pl.from_pandas(close), pl.from_pandas(open_), pl.from_pandas(pre_close), 2
    )

    expected_w3 = np.array(
        [np.nan, np.nan, 0.0, np.nan, np.nan, np.nan], dtype=float
    )
    expected_w2 = np.array(
        [np.nan, 0.0, 0.0, np.nan, np.nan, 0.0], dtype=float
    )
    np.testing.assert_allclose(_values(pandas_w3), expected_w3, equal_nan=True)
    np.testing.assert_allclose(_values(polars_w3), expected_w3, equal_nan=True)
    np.testing.assert_allclose(_values(pandas_w2), expected_w2, equal_nan=True)
    np.testing.assert_allclose(_values(polars_w2), expected_w2, equal_nan=True)


def test_active_transition_count_parity_covers_missing_policy_and_window():
    """Protect tri-state condition semantics on both active backends."""
    pandas_op, polars_op = _active_pair("ts_transition_count")
    condition = pd.DataFrame({"A": [0.0, 1.0, np.nan, 0.0, 1.0, 1.0]})
    polars_condition = pl.from_pandas(condition)

    expected = {
        (4, "break"): np.array([0.0, 1.0, np.nan, 1.0, 1.0, 1.0]),
        (4, "carry"): np.array([0.0, 1.0, np.nan, 2.0, 2.0, 1.0]),
        (2, "break"): np.array([0.0, 1.0, np.nan, 0.0, 1.0, 0.0]),
    }
    for (window, policy), hand_checked in expected.items():
        pandas_result = pandas_op.calculate(condition, window, policy)
        polars_result = polars_op.calculate(polars_condition, window, policy)
        np.testing.assert_allclose(
            _values(pandas_result), hand_checked, equal_nan=True
        )
        np.testing.assert_allclose(
            _values(polars_result), hand_checked, equal_nan=True
        )
