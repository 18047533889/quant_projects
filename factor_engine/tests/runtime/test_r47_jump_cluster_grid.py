"""Jump clustering preserves undefined estimates without dropping identities."""
import numpy as np
import pandas as pd
import polars as pl
from factor_engine.cleaned_operators.intraday.polars_intraday_full import _jump_clustering


def test_jump_clustering_grid_and_independent_gap_oracle():
    returns = np.zeros(19)
    returns[[2, 7, 15]] = [0.08, -0.09, 0.1]
    values = 100 * np.exp(np.r_[0., np.cumsum(returns)])
    dates = pd.date_range("2024-01-03 09:30", periods=20, freq="min")
    frame = pl.DataFrame({"QuoteTime": dates, "B": np.ones(20)*100,
                          "A": values, "missing": np.full(20, np.nan)})
    actual = _jump_clustering(frame, 1.)
    assert actual.columns == ["date", "B", "A", "missing"]
    assert actual.height == 1
    gaps = np.diff([2, 7, 15])
    np.testing.assert_allclose(actual["A"][0], np.std(gaps) / np.mean(gaps))
    assert np.isnan(actual["B"].to_numpy()).all()
    assert np.isnan(actual["missing"].to_numpy()).all()
    empty = _jump_clustering(frame.head(0), 1.)
    assert empty.columns == actual.columns
    assert empty.height == 0
