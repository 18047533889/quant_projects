# -*- coding: utf-8 -*-
"""Cross-backend output index metadata must match the execution template."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from cleaned_operators import load_all


def test_panel_to_series_reuses_exact_string_dtype_multiindex_template():
    load_all()
    from backend.cleaned_bridge import panel_to_series

    timestamps = pd.date_range("2024-01-01", periods=2, name="timestamp")
    instruments = pd.Index(
        pd.array(["A", "B"], dtype="string"),
        name="instrument",
    )
    template_index = pd.MultiIndex.from_product(
        [timestamps, instruments],
        names=["timestamp", "instrument"],
    )
    template = pd.Series(
        np.arange(4.0),
        index=template_index,
        name="template",
    )

    # Object-typed columns reproduce the Polars -> Pandas round-trip shape.
    panel = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=timestamps,
        columns=pd.Index(["A", "B"], dtype=object),
    )
    ctx = SimpleNamespace(
        timestamp_col="timestamp",
        instrument_col="instrument",
    )

    result = panel_to_series(panel, ctx, template=template)

    assert result.index is template.index
    assert result.index.levels[1].dtype == template.index.levels[1].dtype
    assert result.index.names == template.index.names
    np.testing.assert_allclose(result.to_numpy(), [1.0, 2.0, 3.0, 4.0])


def test_panel_to_series_reorders_values_before_reusing_template_index():
    load_all()
    from backend.cleaned_bridge import panel_to_series

    timestamps = pd.DatetimeIndex(
        ["2024-01-01", "2024-01-02"], name="timestamp"
    )
    template_index = pd.MultiIndex.from_tuples(
        [
            (timestamps[0], "B"),
            (timestamps[0], "A"),
            (timestamps[1], "B"),
            (timestamps[1], "A"),
        ],
        names=["timestamp", "instrument"],
    )
    template = pd.Series(np.nan, index=template_index)
    panel = pd.DataFrame(
        [[10.0, 20.0], [30.0, 40.0]],
        index=timestamps,
        columns=["A", "B"],
    )
    ctx = SimpleNamespace(
        timestamp_col="timestamp",
        instrument_col="instrument",
    )

    result = panel_to_series(panel, ctx, template=template)

    assert result.index is template.index
    np.testing.assert_allclose(result.to_numpy(), [20.0, 10.0, 40.0, 30.0])
