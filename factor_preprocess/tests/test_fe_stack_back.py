import numpy as np
import pandas as pd
import pytest

from factor_preprocess.adapters.fe_operator import _stack_back


def test_stack_back_preserves_row_labels_missing_keys_and_identity_types():
    panel = pd.DataFrame([[1., 2.], [np.nan, 4.]], index=[10, 20],
                         columns=pd.Index([1, "1"], dtype=object))
    template = pd.DataFrame({"when": [20, 10, 10, 30, 10, 20],
                             "asset": ["1", 1, "1", 1, "missing", 1]},
                            index=[9, 9, 3, 1, 6, 8])
    got = _stack_back(panel, template, "when", "asset", "signal")
    expected = pd.Series([4., 1., 2., np.nan, np.nan, np.nan],
                         index=template.index, name="signal")
    pd.testing.assert_series_equal(got, expected)


def test_stack_back_does_not_coerce_timestamp_strings_into_timestamp_identities():
    dates = pd.date_range("2024-01-01", periods=2, tz="Asia/Hong_Kong")
    panel = pd.DataFrame({"a": [1., 2.]}, index=dates)
    template = pd.DataFrame({"date": [dates[1], str(dates[0]), dates[0]],
                             "asset": ["a", "a", "a"]})
    got = _stack_back(panel, template, "date", "asset", "value")
    np.testing.assert_array_equal(got.to_numpy(), [2., np.nan, 1.])


@pytest.mark.parametrize("n", [0, 1, 200])
def test_stack_back_matches_hand_indexed_numeric_panel_without_aliasing(n):
    rng = np.random.default_rng(93)
    x = rng.normal(size=(20, 7))
    x[3, 4] = np.nan
    times = pd.date_range("2024-01-01", periods=20)
    assets = np.array([f"a{i}" for i in range(7)])
    rows, cols = rng.integers(0, 20, n), rng.integers(0, 7, n)
    template = pd.DataFrame({"date": times[rows], "asset": assets[cols]})
    panel = pd.DataFrame(x.copy(), index=times, columns=assets)
    got = _stack_back(panel, template, "date", "asset", "value")
    np.testing.assert_array_equal(got.to_numpy(), x[rows, cols])
    if n:
        got.iloc[0] = 999.
        np.testing.assert_array_equal(panel.to_numpy(), x)
