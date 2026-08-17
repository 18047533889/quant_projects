"""R2-P0-018: the repaired Polars ts_corr is the sole registered path."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry


def test_ts_corr_registry_points_to_repaired_native_operator() -> None:
    load_all()
    operator = OperatorRegistry.get("ts_corr", "polars", mode="research")
    assert operator.__class__.__module__ == "cleaned_operators.common.polars_ts_rolling"
    assert operator.__class__.__name__ == "TSCorrNative"


def test_ts_corr_matches_direct_window_oracle_with_missing_pairs() -> None:
    load_all()
    operator = OperatorRegistry.get("ts_corr", "polars", mode="research")
    x = pl.DataFrame({"asset": [1.0, 2.0, np.nan, 8.0, 5.0, 9.0]})
    y = pl.DataFrame({"asset": [2.0, 4.0, 6.0, np.nan, 3.0, 7.0]})

    actual = operator.calculate(x, y, window=4, min_periods=2)["asset"].to_numpy()
    expected = pd.Series(x["asset"].to_numpy()).rolling(4, min_periods=2).corr(
        pd.Series(y["asset"].to_numpy())
    ).to_numpy()

    np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-12, atol=1e-12)
