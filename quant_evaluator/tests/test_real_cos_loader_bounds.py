"""Research COS factor loader bounds fail before remote reads."""

import numpy as np
import pandas as pd
import pytest

from quant_evaluator.scripts.load_real_cos_factor_batch import (
    _assemble_factor_values, _factors, SHA,
)


@pytest.mark.parametrize("count,max_mib,total_mib", [
    (0, 64, 256),
    (33, 64, 256),
    (True, 64, 256),
    (2, 0, 256),
    (2, 129, 256),
    (2, True, 256),
    (2, 64, 0),
    (2, 64, 2049),
    (2, 64, True),
])
def test_research_loader_rejects_out_of_bounds_before_io(count, max_mib, total_mib):
    with pytest.raises(ValueError):
        _factors(count, max_mib, SHA, total_mib)


def test_research_loader_rejects_unbound_manifest_before_io():
    with pytest.raises(ValueError, match="manifest_sha256"):
        _factors(32, 128, "not-a-digest", 2048)


def test_streamed_factor_values_match_panel_reindex_and_stack_exactly():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    panels = [
        pd.DataFrame([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
                     index=dates, columns=["B.SZ", "A.SZ"]),
        pd.DataFrame([[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]],
                     index=dates[1:], columns=["A.SZ", "B.SZ"]),
        pd.DataFrame([[20.0, 21.0], [22.0, 23.0], [24.0, 25.0]],
                     index=dates[:3], columns=["C.SZ", "A.SZ"]),
    ]
    sources = [{"factor_id": f"factor-{i}"} for i in range(len(panels))]
    values, common_dates, common_assets, got_sources, base_dates, base_assets = (
        _assemble_factor_values(zip(panels, sources), len(panels))
    )

    names = sorted(common_assets)
    actual = values[np.ix_(base_dates.get_indexer(common_dates),
                           base_assets.get_indexer(names), np.arange(len(panels)))]
    expected = np.stack([
        panel.reindex(index=common_dates, columns=names).to_numpy(dtype=np.float64)
        for panel in panels
    ], axis=-1)

    assert common_dates.equals(dates[1:3])
    assert names == ["A.SZ"]
    assert got_sources == sources
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(actual, expected)
