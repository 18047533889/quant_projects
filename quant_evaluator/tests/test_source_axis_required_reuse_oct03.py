import numpy as np
import pandas as pd
import pytest

from quant_evaluator.adapters.source_axis_materializer import (
    write_axis_aligned_float64,
)


def _shuffled_frame(dates):
    return pd.DataFrame(
        [[30.0, 10.0, 20.0], [3.0, 1.0, 2.0], [300.0, 100.0, 200.0],
         [np.nan, 4.0, 5.0]],
        index=dates[[2, 0, 3, 1]], columns=["C.SZ", "A.SZ", "B.SZ"],
    )


def test_bounded_equal_required_axes_reuse_positions_and_preserve_nan(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=4)
    assets = ["A.SZ", "B.SZ", "C.SZ"]
    frame = _shuffled_frame(dates)
    # Compute the oracle before the spy so its internal reindex work is excluded.
    expected = frame.reindex(index=dates, columns=assets).to_numpy(dtype=np.float64)
    original = pd.Index.get_indexer
    calls = []

    def count_get_indexer(index, target, *args, **kwargs):
        calls.append((tuple(index), tuple(target)))
        return original(index, target, *args, **kwargs)

    monkeypatch.setattr(pd.Index, "get_indexer", count_get_indexer)
    output = np.empty(expected.shape, dtype=np.float64)
    write_axis_aligned_float64(
        frame, dates, assets, output,
        required_dates=dates, required_assets=assets,
        memory_bounded=True, chunk_bytes=1024,
    )

    assert len(calls) == 2
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(np.isnan(output), np.isnan(expected))


def test_bounded_subset_axes_keep_separate_target_lookups_and_shared_validation(
        monkeypatch):
    dates = pd.date_range("2024-01-01", periods=4)
    assets = ["A.SZ", "B.SZ", "C.SZ"]
    target_dates, target_assets = dates[[1, 3]], ["B.SZ", "C.SZ"]
    frame = _shuffled_frame(dates)
    expected = frame.reindex(
        index=target_dates, columns=target_assets).to_numpy(dtype=np.float64)
    original = pd.Index.get_indexer
    calls = []

    def count_get_indexer(index, target, *args, **kwargs):
        calls.append((tuple(index), tuple(target)))
        return original(index, target, *args, **kwargs)

    monkeypatch.setattr(pd.Index, "get_indexer", count_get_indexer)
    output = np.empty(expected.shape, dtype=np.float64)
    write_axis_aligned_float64(
        frame, target_dates, target_assets, output,
        required_dates=dates, required_assets=assets,
        memory_bounded=True, chunk_bytes=1024,
    )
    assert len(calls) == 4
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(np.isnan(output), np.isnan(expected))

    with pytest.raises(ValueError, match="indexed shared axes changed"):
        write_axis_aligned_float64(
            frame, target_dates, target_assets, np.empty(expected.shape),
            required_dates=dates, required_assets=["A.SZ", "B.SZ", "MISSING"],
            memory_bounded=True, chunk_bytes=1024,
        )


def test_reuse_keeps_integer_and_string_axis_labels_distinct():
    dates = pd.date_range("2024-01-01", periods=2)
    frame = pd.DataFrame([[1.0], [2.0]], index=dates, columns=[1])
    with pytest.raises(ValueError, match="factor final axes changed"):
        write_axis_aligned_float64(
            frame, dates, ["1"], np.empty((2, 1)),
            required_dates=dates, required_assets=[1],
            memory_bounded=True, chunk_bytes=16,
        )


def test_reuse_keeps_bounded_alias_rejection_without_mutation():
    dates = pd.date_range("2024-01-01", periods=3)
    assets = ["A", "B"]
    backing = np.arange(6, dtype=np.float64).reshape(3, 2)
    frame = pd.DataFrame(backing[:, ::-1], index=dates, columns=["B", "A"], copy=False)
    before = backing.copy()

    with pytest.raises(ValueError, match="must not share storage"):
        write_axis_aligned_float64(
            frame, dates, assets, backing,
            required_dates=dates, required_assets=assets,
            memory_bounded=True, chunk_bytes=48,
        )
    np.testing.assert_array_equal(backing, before)


def test_bounded_equal_required_contiguous_subaxes_keep_slice_path(monkeypatch):
    from pandas.core.indexing import _iLocIndexer

    dates = pd.date_range("2024-01-01", periods=5)
    assets = ["X.SZ", "A.SZ", "B.SZ", "C.SZ", "Y.SZ"]
    frame = pd.DataFrame(
        np.arange(25, dtype=np.float64).reshape(5, 5),
        index=dates, columns=assets,
    )
    frame.iloc[2, 2] = np.nan
    target_dates = dates[1:4]
    target_assets = assets[1:4]
    expected = frame.reindex(
        index=target_dates, columns=target_assets).to_numpy(dtype=np.float64)
    original_indexer = pd.Index.get_indexer
    original_iloc = _iLocIndexer.__getitem__
    lookup_calls = []
    iloc_selectors = []

    def count_get_indexer(index, target, *args, **kwargs):
        lookup_calls.append((tuple(index), tuple(target)))
        return original_indexer(index, target, *args, **kwargs)

    def slices_only(indexer, key):
        selectors = key if isinstance(key, tuple) else (key,)
        assert all(isinstance(selector, slice) for selector in selectors)
        iloc_selectors.append(key)
        return original_iloc(indexer, key)

    monkeypatch.setattr(pd.Index, "get_indexer", count_get_indexer)
    monkeypatch.setattr(_iLocIndexer, "__getitem__", slices_only)
    output = np.empty(expected.shape, dtype=np.float64)
    write_axis_aligned_float64(
        frame, target_dates, target_assets, output,
        required_dates=target_dates, required_assets=target_assets,
        memory_bounded=True, chunk_bytes=1024,
    )

    assert len(lookup_calls) == 2
    assert iloc_selectors
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(np.isnan(output), np.isnan(expected))


@pytest.mark.parametrize("axis_kind", ["date", "asset"])
def test_reuse_does_not_treat_boolean_axis_as_integer(axis_kind):
    if axis_kind == "date":
        frame = pd.DataFrame([[1.0], [2.0]], index=[1, 2], columns=["A"])
        dates, assets = [True], ["A"]
        required_dates, required_assets = [1], ["A"]
    else:
        dates, assets = pd.date_range("2024-01-01", periods=2), [True]
        frame = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]],
                             index=dates, columns=[1, 2])
        required_dates, required_assets = dates, [1]
    with pytest.raises(ValueError, match="factor final axes changed"):
        write_axis_aligned_float64(
            frame, dates, assets, np.empty((len(dates), len(assets))),
            required_dates=required_dates, required_assets=required_assets,
            memory_bounded=True, chunk_bytes=64,
        )


def test_reuse_accepts_python_and_numpy_integer_labels(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=2)
    frame = pd.DataFrame([[20.0, 10.0], [40.0, 30.0]],
                         index=dates, columns=np.asarray([2, 1], dtype=np.int64))
    target_assets = [1, 2]
    required_assets = np.asarray([1, 2], dtype=np.int64)
    expected = frame.reindex(
        index=dates, columns=target_assets).to_numpy(dtype=np.float64)
    original = pd.Index.get_indexer
    calls = []

    def count_get_indexer(index, target, *args, **kwargs):
        calls.append((tuple(index), tuple(target)))
        return original(index, target, *args, **kwargs)

    monkeypatch.setattr(pd.Index, "get_indexer", count_get_indexer)
    output = np.empty(expected.shape, dtype=np.float64)
    write_axis_aligned_float64(
        frame, dates, target_assets, output,
        required_dates=dates, required_assets=required_assets,
        memory_bounded=True, chunk_bytes=64,
    )
    assert len(calls) == 2
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("axis_kind", ["date", "asset"])
@pytest.mark.parametrize("memory_bounded", [False, True])
def test_exact_path_rejects_boolean_axis_alias_for_integer(axis_kind, memory_bounded):
    if axis_kind == "date":
        frame = pd.DataFrame([[7.0]], index=[1], columns=["A"])
        dates, assets = [True], ["A"]
        required_dates, required_assets = [1], ["A"]
    else:
        dates, assets = pd.date_range("2024-01-01", periods=1), [True]
        frame = pd.DataFrame([[7.0]], index=dates, columns=[1])
        required_dates, required_assets = dates, [1]

    with pytest.raises(ValueError, match="factor final axes changed"):
        write_axis_aligned_float64(
            frame, dates, assets, np.empty((1, 1)),
            required_dates=required_dates, required_assets=required_assets,
            memory_bounded=memory_bounded, chunk_bytes=8,
        )
