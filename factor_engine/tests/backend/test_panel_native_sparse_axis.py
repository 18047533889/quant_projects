from types import SimpleNamespace

import numpy as np
import pandas as pd

from factor_engine.backend.panel_native import finalize_panel_result, load_column_as_panel
from factor_engine.backend.long_frame import polars_long_to_multiindex_series


def test_panel_loading_preserves_sparse_order_and_explicit_nan_axis():
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "B"),
         (pd.Timestamp("2024-01-01"), "A"),
         (pd.Timestamp("2024-01-02"), "A")],
        names=["timestamp", "instrument"],
    )
    source_series = pd.Series([2.0, np.nan, 1.0], index=idx)

    class Source:
        def load_column(self, _name):
            return source_series

        def load_column_panel(self, _name):
            return source_series.unstack()

    ctx = SimpleNamespace(
        data_source=Source(), timestamp_col="timestamp", instrument_col="instrument",
        template_index=None, template_series=None, perf=SimpleNamespace(panel_native=True),
    )
    panel = load_column_as_panel(ctx.data_source, "x", ctx)
    result = finalize_panel_result(panel, ctx)

    assert result.index.equals(idx)
    assert len(result) == 3
    assert np.isnan(result.iloc[1])


def test_polars_long_boundary_does_not_cartesianize_regular_sparse_dates():
    import polars as pl

    frame = pl.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-01"),
                      pd.Timestamp("2024-01-02")],
        "instrument": ["B", "A", "A"],
        "value": [2.0, float("nan"), 1.0],
    })
    result = polars_long_to_multiindex_series(
        frame, timestamp_col="timestamp", instrument_col="instrument", value_col="value")

    assert list(result.index) == [
        (pd.Timestamp("2024-01-01"), "A"),
        (pd.Timestamp("2024-01-02"), "A"),
        (pd.Timestamp("2024-01-02"), "B"),
    ]
    assert len(result) == 3
    assert np.isnan(result.loc[(pd.Timestamp("2024-01-01"), "A")])
    assert result.loc[(pd.Timestamp("2024-01-02"), "A")] == 1.0
