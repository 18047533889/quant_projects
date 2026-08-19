"""
Guard against row-relabeling in per-asset decomposition filters.

groupby(asset).apply returns rows grouped by asset (first-appearance
order); the filters historically reassigned the input index positionally,
which silently swapped rows between assets for interleaved layouts that
still pass the per-asset monotonicity guard. These tests pin the
label-based realignment contract.
"""
import numpy as np
import pandas as pd
import pytest

from factor_preprocess.transforms.decomposition.cycle import (
    bandpass_filter,
    christiano_fitzgerald_filter,
)
from factor_preprocess.transforms.decomposition.trend import hp_filter
from factor_preprocess.transforms.decomposition.wavelet import wavelet_denoise

PER = 30  # per-asset length; sosfiltfilt padlen requires > 15 valid rows


def _interleaved():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=PER, freq="D").repeat(2),
            "asset_id": ["A", "B"] * PER,
            "value": rng.normal(size=2 * PER),
        }
    )
    df.index = [f"r{i}" for i in range(2 * PER)]
    return df


def _per_asset_truth(df, fn, **kwargs):
    parts = []
    for asset in ["A", "B"]:
        sub = df[df.asset_id == asset].sort_values("date")
        parts.append(fn(sub, **kwargs))
    return pd.concat(parts).reindex(df.index)


def _assert_matches_truth(df, out, expected):
    got = out.to_numpy()
    exp = expected.to_numpy()
    bad = [i for i in range(len(df)) if not np.isclose(got[i], exp[i], equal_nan=True)]
    assert not bad, f"rows mislabeled vs per-asset truth: {bad[:5]}"


@pytest.mark.parametrize(
    "fn,kwargs",
    [
        (bandpass_filter, dict(low_freq=0.05, high_freq=0.4, order=2)),
        (christiano_fitzgerald_filter, dict(low_period=5, high_period=15)),
        (hp_filter, dict(lambda_param=10.0)),
        (wavelet_denoise, dict(wavelet="db4", level=2)),
    ],
    ids=["bandpass", "cf", "hp", "wavelet"],
)
def test_interleaved_layout_rows_are_not_relabelled(fn, kwargs):
    """A,B,A,B rows (per-asset time monotonic) must keep per-asset results."""
    df = _interleaved()
    out = fn(df, **kwargs)
    expected = _per_asset_truth(df, fn, **kwargs)
    _assert_matches_truth(df, out, expected)
    assert out.index.tolist() == df.index.tolist()


@pytest.mark.parametrize(
    "fn,kwargs",
    [
        (bandpass_filter, dict(low_freq=0.05, high_freq=0.4, order=2)),
        (hp_filter, dict(lambda_param=10.0)),
    ],
    ids=["bandpass", "hp"],
)
def test_default_range_index_blocked_layout(fn, kwargs):
    """Blocked (sorted-by-asset) layout with default RangeIndex still works."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=PER, freq="D").append(
                pd.date_range("2020-01-01", periods=PER, freq="D")
            ),
            "asset_id": ["A"] * PER + ["B"] * PER,
            "value": rng.normal(size=2 * PER),
        }
    ).sort_values(["asset_id", "date"], kind="stable").reset_index(drop=True)
    out = fn(df, **kwargs)
    expected = _per_asset_truth(df, fn, **kwargs)
    _assert_matches_truth(df, out, expected)
