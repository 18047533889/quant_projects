"""Exact-axis fast-path contracts for factor-frame materialization."""
import numpy as np
import pandas as pd
import pytest

from quant_evaluator.adapters.source_axis_materializer import (
    write_axis_aligned_float64,
)


@pytest.mark.parametrize("memory_bounded", [False, True])
def test_exact_axes_copy_values_into_noncontiguous_output(memory_bounded):
    dates = pd.date_range("2024-01-01", periods=4)
    assets = ["A", "B", "C"]
    values = np.arange(12, dtype=np.float64).reshape(4, 3)
    frame = pd.DataFrame(values, index=dates, columns=assets)
    backing = np.full((4, 6), -1.0)
    output = backing[:, ::2]

    write_axis_aligned_float64(
        frame, dates, assets, output, memory_bounded=memory_bounded,
        chunk_bytes=192,
    )

    np.testing.assert_array_equal(output, values)
    np.testing.assert_array_equal(backing[:, 1::2], -1.0)



@pytest.mark.parametrize("memory_bounded", [False, True])
@pytest.mark.parametrize("asset_container", ["list", "tuple", "ndarray"])
def test_exact_axis_shortcut_skips_indexers_reindex_and_fancy_iloc(
        monkeypatch, memory_bounded, asset_container):
    from pandas.core.indexing import _iLocIndexer

    dates = pd.date_range("2024-01-01", periods=3)
    assets = ["A", "B"]
    frame = pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                         index=dates, columns=assets)
    targets = {"list": assets, "tuple": tuple(assets),
               "ndarray": np.asarray(assets)}[asset_container]

    def forbidden(*args, **kwargs):
        pytest.fail("exact axes should bypass reindex/indexer construction")
    monkeypatch.setattr(pd.Index, "get_indexer", forbidden)
    monkeypatch.setattr(pd.DataFrame, "reindex", forbidden)
    if memory_bounded:
        original_getitem = _iLocIndexer.__getitem__
        def only_slices(self, key):
            selectors = key if isinstance(key, tuple) else (key,)
            assert all(isinstance(selector, slice) for selector in selectors)
            return original_getitem(self, key)
        monkeypatch.setattr(_iLocIndexer, "__getitem__", only_slices)

    out = np.empty((len(dates), len(assets)), dtype=np.float64)
    write_axis_aligned_float64(frame, dates, targets, out,
                               memory_bounded=memory_bounded)
    np.testing.assert_array_equal(out, frame.to_numpy())


def test_exact_axis_alias_with_reversed_noncontiguous_output_is_snapshot_safe():
    dates = pd.date_range("2024-01-01", periods=4)
    frame = pd.DataFrame(np.arange(12, dtype=np.float64).reshape(4, 3),
                         index=dates, columns=["A", "B", "C"])
    source = frame.to_numpy(copy=False)
    source.flags.writeable = True
    expected = source.copy()
    output = source[:, ::-1]
    assert np.shares_memory(output, source)

    write_axis_aligned_float64(frame, dates, ["A", "B", "C"], output)

    np.testing.assert_array_equal(output, expected)


def test_bounded_exact_axis_alias_is_rejected_before_any_write():
    dates = pd.date_range("2024-01-01", periods=4)
    frame = pd.DataFrame(np.arange(12, dtype=np.float64).reshape(4, 3),
                         index=dates, columns=["A", "B", "C"])
    source = frame.to_numpy(copy=False)
    source.flags.writeable = True
    before = source.copy()
    output = source[:, ::-1]

    with pytest.raises(ValueError, match="must not share storage"):
        write_axis_aligned_float64(
            frame, dates, ["A", "B", "C"], output,
            memory_bounded=True, chunk_bytes=96)

    np.testing.assert_array_equal(source, before)


def test_nullable_extension_buffer_alias_is_detected_without_conversion():
    dates = pd.date_range("2024-01-01", periods=3)
    backing = np.arange(6, dtype=np.float64)
    values = pd.arrays.FloatingArray(backing, np.zeros(6, dtype=bool))
    frame = pd.DataFrame({"A": values[:3], "B": values[3:]},
                         index=dates, copy=False)
    output = backing.reshape(3, 2)
    before = backing.copy()

    with pytest.raises(ValueError, match="must not share storage"):
        write_axis_aligned_float64(
            frame, dates, ["A", "B"], output,
            memory_bounded=True, chunk_bytes=48)

    np.testing.assert_array_equal(backing, before)


def test_bounded_shifted_row_alias_is_rejected_without_mutation():
    dates = pd.date_range("2024-01-01", periods=4)
    backing = np.arange(8, dtype=np.float64).reshape(4, 2)
    frame = pd.DataFrame(backing[:3], index=dates[:3],
                         columns=["A", "B"], copy=False)
    source = frame.to_numpy(copy=False)
    assert np.shares_memory(source, backing)
    output = backing[1:]
    before = backing.copy()

    with pytest.raises(ValueError, match="must not share storage"):
        write_axis_aligned_float64(
            frame, dates[:3], ["A", "B"], output,
            memory_bounded=True, chunk_bytes=32)

    np.testing.assert_array_equal(backing, before)


def test_unbounded_shifted_row_alias_uses_snapshot_semantics():
    dates = pd.date_range("2024-01-01", periods=4)
    backing = np.arange(8, dtype=np.float64).reshape(4, 2)
    frame = pd.DataFrame(backing[:3], index=dates[:3],
                         columns=["A", "B"], copy=False)
    expected = frame.to_numpy(copy=True)
    output = backing[1:]

    write_axis_aligned_float64(frame, dates[:3], ["A", "B"], output)

    np.testing.assert_array_equal(output, expected)

@pytest.mark.parametrize("memory_bounded", [False, True])
def test_reordered_and_partial_axes_keep_alignment(memory_bounded):
    dates = pd.date_range("2024-01-01", periods=3)
    source_dates = dates[::-1]
    frame = pd.DataFrame([[9, 8], [5, 4], [1, 0]],
                         index=source_dates, columns=["B", "A"])
    out = np.empty((3, 2), dtype=np.float64)

    write_axis_aligned_float64(frame, dates, ["A", "B"], out,
                               memory_bounded=memory_bounded, chunk_bytes=64)

    np.testing.assert_array_equal(out, [[0, 1], [4, 5], [8, 9]])
    partial = np.empty((2, 1), dtype=np.float64)
    write_axis_aligned_float64(frame, dates[:2], ["B"], partial,
                               memory_bounded=memory_bounded, chunk_bytes=48)
    np.testing.assert_array_equal(partial[:, 0], [1, 5])


@pytest.mark.parametrize("memory_bounded", [False, True])
def test_missing_target_axis_raises(memory_bounded):
    dates = pd.date_range("2024-01-01", periods=2)
    frame = pd.DataFrame([[1.0]], index=dates[:1], columns=["A"])
    out = np.full((2, 1), -7.0)
    with pytest.raises(ValueError, match="final axes changed"):
        write_axis_aligned_float64(frame, dates, ["A"], out,
                                   memory_bounded=memory_bounded)
    np.testing.assert_array_equal(out, -7.0)


@pytest.mark.parametrize("duplicate_axis", ["index", "columns"])
def test_duplicate_exact_axes_do_not_take_fast_path(duplicate_axis):
    dates = pd.date_range("2024-01-01", periods=2)
    if duplicate_axis == "index":
        dates = pd.DatetimeIndex(["2024-01-01", "2024-01-01"])
        frame = pd.DataFrame([[1.0], [2.0]], index=dates, columns=["A"])
        targets_dates, targets_assets = dates, ["A"]
        shape = (2, 1)
    else:
        frame = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]],
                             index=dates, columns=["A", "A"])
        targets_dates, targets_assets = dates, ["A", "A"]
        shape = (2, 2)
    with pytest.raises(pd.errors.InvalidIndexError):
        write_axis_aligned_float64(frame, targets_dates, targets_assets,
                                   np.empty(shape))


@pytest.mark.parametrize("memory_bounded", [False, True])
def test_empty_exact_axes_are_supported(memory_bounded):
    dates = pd.date_range("2024-01-01", periods=0)
    frame = pd.DataFrame(index=dates, columns=["A"], dtype=np.float64)
    out = np.empty((0, 1), dtype=np.float64)
    write_axis_aligned_float64(frame, dates, ["A"], out,
                               memory_bounded=memory_bounded)
    assert out.shape == (0, 1)


@pytest.mark.parametrize("memory_bounded", [False, True])
def test_exact_axes_convert_mixed_dtypes_and_fail_on_nonnumeric_object(memory_bounded):
    dates = pd.date_range("2024-01-01", periods=2)
    frame = pd.DataFrame({"A": np.array([1, 2], dtype=np.int64),
                          "B": np.array([3.5, 4.5], dtype=np.float32)},
                         index=dates)
    out = np.empty((2, 2), dtype=np.float64)
    write_axis_aligned_float64(frame, dates, ["A", "B"], out,
                               memory_bounded=memory_bounded, chunk_bytes=64)
    np.testing.assert_array_equal(out, [[1.0, 3.5], [2.0, 4.5]])

    bad = frame.copy()
    bad["B"] = ["not-numeric", "4.5"]
    with pytest.raises((TypeError, ValueError)):
        write_axis_aligned_float64(bad, dates, ["A", "B"],
                                   np.empty((2, 2)), memory_bounded=memory_bounded,
                                   chunk_bytes=64)
