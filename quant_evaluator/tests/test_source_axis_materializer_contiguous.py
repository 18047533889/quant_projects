"""Mechanism and independent alignment checks for contiguous source sub-axes."""
import numpy as np
import pandas as pd
import pytest

from quant_evaluator.adapters.source_axis_materializer import write_axis_aligned_float64


def test_contiguous_subaxes_use_slices_for_every_bounded_chunk(monkeypatch):
    from pandas.core.indexing import _iLocIndexer

    dates = pd.date_range("2024-01-01", periods=9)
    values = np.arange(54, dtype=np.float64).reshape(9, 6)
    values[4, 2] = np.nan
    frame = pd.DataFrame(values, index=dates, columns=list("ABCDEF"))
    expected = values[1:8, 1:5].copy()
    backing = np.full((7, 4, 3), -999.0)
    output = backing[:, :, 1]
    original = _iLocIndexer.__getitem__
    selectors = []

    def only_slices(self, key):
        parts = key if isinstance(key, tuple) else (key,)
        assert all(isinstance(part, slice) for part in parts)
        selectors.append(parts)
        return original(self, key)

    monkeypatch.setattr(_iLocIndexer, "__getitem__", only_slices)
    write_axis_aligned_float64(
        frame, dates[1:8], list("BCDE"), output,
        memory_bounded=True, chunk_bytes=(6 + 4) * 8 * 2,
    )
    np.testing.assert_equal(output, expected)
    np.testing.assert_equal(backing[:, :, 0], -999.0)
    np.testing.assert_equal(backing[:, :, 2], -999.0)
    assert len(selectors) == 8  # Four row chunks plus four column slices.


@pytest.mark.parametrize("seed", [19, 41, 73])
def test_noncontiguous_and_repeated_targets_match_independent_numpy_oracle(seed):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(17, 11))
    values[rng.random(values.shape) < .2] = np.nan
    frame = pd.DataFrame(values, index=np.arange(17) + 100,
                         columns=[f"a{i}" for i in range(11)])
    rows = rng.integers(0, 17, size=9)
    columns = rng.integers(0, 11, size=7)
    expected = values[np.ix_(rows, columns)]
    output = np.empty(expected.shape)
    write_axis_aligned_float64(
        frame, frame.index[rows], frame.columns[columns], output,
        memory_bounded=True, chunk_bytes=(11 + 7) * 8 * 2,
    )
    np.testing.assert_equal(output, expected)
