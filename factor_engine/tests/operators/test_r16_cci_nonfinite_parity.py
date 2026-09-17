"""CCI must preserve the canonical active-window nonfinite propagation."""
import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from factor_engine.cleaned_operators.technical.signal import CCI
from factor_engine.cleaned_operators.technical.polars_signal import CCIPolars


def _oracle(values, window):
    result = []
    for i, current in enumerate(values):
        active = np.asarray(values[max(0, i - window + 1):i + 1], dtype=float)
        if not np.isfinite(active).all():
            result.append(np.nan)
            continue
        center = active.mean()
        deviation = np.abs(active - center).mean()
        result.append((current - center) / (0.015 * deviation) if deviation else np.nan)
    return np.asarray(result)


@pytest.mark.parametrize("missing", [None, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("window", [1, 3, 20])
def test_cci_nonfinite_window_and_recovery(missing, window):
    values = [1., 3., 2., missing, 4., 9., 8., 12.]
    dates = pd.date_range("2025-01-01", periods=len(values))
    panel = pd.DataFrame({"A": values, "B": np.arange(len(values), dtype=float)},
                         index=dates)
    polars = pl.DataFrame({"date": dates, "A": values, "B": panel.B.to_list()})
    expected = np.column_stack([_oracle(panel[c].to_numpy(), window) for c in panel])
    # Identical H/L/C gives exactly that typical price, without introducing
    # a second independent rounding oracle.
    pandas_result = CCI()._calculate_series(panel, panel, panel, window=window)
    actual = CCIPolars()._calculate_series(polars, polars, polars, window=window)
    np.testing.assert_allclose(pandas_result, expected, equal_nan=True, atol=1e-12)
    np.testing.assert_allclose(actual.select("A", "B").to_numpy(), expected,
                               equal_nan=True, atol=1e-12)
    assert actual["date"].to_list() == polars["date"].to_list()


@pytest.mark.parametrize("values", [[4.] * 8, [0., 5e-13, 2e-13, 4e-13]])
def test_cci_zero_and_tiny_denominator(values):
    panel = pd.DataFrame({"A": values})
    polars = pl.DataFrame({"A": values})
    expected = _oracle(values, 3)
    actual = CCIPolars()._calculate_series(polars, polars, polars, window=3)
    np.testing.assert_allclose(actual["A"].to_numpy(), expected,
                               equal_nan=True, atol=1e-12)
