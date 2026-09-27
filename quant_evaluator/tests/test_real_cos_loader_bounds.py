"""Research COS factor loader bounds fail before remote reads."""

import numpy as np
import pandas as pd
import pytest

from quant_evaluator.scripts.load_real_cos_factor_batch import (
    _assemble_factor_values, _factors, _trim_factor_values_in_place, SHA,
)
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch


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


def test_in_place_factor_trim_preserves_axis_intersection_reorder_and_immutability(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    source_assets = pd.Index(["C.SZ", "A.SZ", "D.SZ", "B.SZ"])
    source = np.arange(5 * 4 * 3, dtype=np.float64).reshape(5, 4, 3)
    source[1, 3, 1] = np.nan
    expected = source[np.ix_([1, 3, 4], [3, 2, 1], np.arange(3))].copy()
    selected_dates = dates[[1, 3, 4]]
    selected_assets = ["B.SZ", "D.SZ", "A.SZ"]

    allocations = []
    real_empty = np.empty

    def tracking_empty(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) > 1:
            allocations.append(shape)
        return real_empty(shape, *args, **kwargs)

    monkeypatch.setattr(np, "empty", tracking_empty)
    trimmed = _trim_factor_values_in_place(
        source, dates, source_assets, selected_dates, selected_assets
    )

    assert allocations == [(len(selected_assets), source.shape[2])]
    assert source.shape not in allocations
    assert trimmed.dtype == np.float64
    assert trimmed.flags.c_contiguous
    assert np.shares_memory(trimmed, source)
    assert trimmed.shape == (3, 3, 3)
    np.testing.assert_array_equal(np.isnan(trimmed), np.isnan(expected))
    np.testing.assert_array_equal(trimmed, expected)
    np.testing.assert_array_equal(trimmed.view(np.uint64), expected.view(np.uint64))

    batch = FactorBatch(
        ("f0", "f1", "f2"),
        AxisRef("time", "datetime64[ns]", len(selected_dates), selected_dates.to_numpy()),
        AxisRef("asset", "str", len(selected_assets), np.asarray(selected_assets)),
        trimmed,
        validity=np.isfinite(trimmed),
    )
    assert not batch.values.flags.writeable
    assert not np.shares_memory(batch.values, trimmed)
    np.testing.assert_array_equal(np.isnan(batch.values), np.isnan(expected))
    np.testing.assert_array_equal(batch.values, expected)
    np.testing.assert_array_equal(batch.values.view(np.uint64), expected.view(np.uint64))


def test_in_place_factor_trim_rejects_nonmonotone_time_before_mutation():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    assets = pd.Index(["A.SZ", "B.SZ"])
    source = np.arange(4 * 2, dtype=np.float64).reshape(4, 2, 1)
    before = source.copy()

    with pytest.raises(ValueError, match="out of source order"):
        _trim_factor_values_in_place(source, dates, assets, dates[[2, 1]], assets)

    np.testing.assert_array_equal(source, before)


def test_in_place_factor_trim_randomized_bitwise_parity():
    dates = pd.date_range("2024-01-01", periods=9, freq="D")
    assets = pd.Index([f"S{i}.SZ" for i in range(7)])
    for seed in range(12):
        rng = np.random.default_rng(seed)
        values = rng.standard_normal((9, 7, 4))
        values[seed % 9, seed % 7, seed % 4] = np.nan
        selected_times = np.sort(rng.choice(9, size=1 + seed % 9, replace=False))
        selected_assets = rng.permutation(7)[:1 + seed % 7]
        expected = values[np.ix_(selected_times, selected_assets, np.arange(4))]
        actual = _trim_factor_values_in_place(
            values, dates, assets, dates[selected_times], assets[selected_assets])
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        assert actual.flags.c_contiguous and np.shares_memory(actual, values)
