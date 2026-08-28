"""ROLLING-EDGE regression tests: pandas-vs-polars equivalence for the
ts_* rolling-family canonicals on edge inputs.

pandas is the reference.  A polars mismatch is a bug.  These tests call the
module-local native kernels directly (``_calculate_series``) — the same
module-local pattern the registry's own window-local tests use
(``test_polars_ts_stats_window_local``) — so they run even before the full
registry finishes building.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

pytest.importorskip("polars")
pytest.importorskip("pandas")

from factor_engine.cleaned_operators.common.polars_ts_basic import (
    TSArgmaxNative,
    TSArgminNative,
    TSMaxNative,
    TSMeanNative,
    TSMedianNative,
    TSMinNative,
    TSStdNative,
    TSSumNative,
)
from factor_engine.cleaned_operators._rolling_fast import rolling_days_since_extreme


def _pl_frame(a: np.ndarray) -> pl.DataFrame:
    return pl.from_pandas(
        pd.DataFrame({"c0": np.asarray(a, float), "c1": np.asarray(a, float) + 10.0}).reset_index(drop=True)
    )


def _pd_frame(a: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"c0": np.asarray(a, float), "c1": np.asarray(a, float) + 10.0})


def _assert_parity(a: np.ndarray, window: int, pd_fn, pl_fn) -> None:
    pd_r = np.asarray(pd_fn(_pd_frame(a), window), dtype=float)
    pl_r = np.asarray(pl_fn(_pl_frame(a), window), dtype=float)
    assert pd_r.shape == pl_r.shape
    np.testing.assert_array_equal(
        np.isnan(pd_r), np.isnan(pl_r),
        err_msg=f"NaN mask differs: pd={pd_r!r} vs pl={pl_r!r}",
    )
    np.testing.assert_allclose(
        np.nan_to_num(pd_r), np.nan_to_num(pl_r),
        rtol=1e-6, atol=1e-9, equal_nan=True,
        err_msg=f"finite values differ: pd={pd_r!r} vs pl={pl_r!r}",
    )


_EDGE = [
    ("plain", np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 4),
    ("nan_hole", np.array([1.0, np.nan, 3, 4, np.nan, 6, 7, 8, 9, 10]), 4),
    ("allnan_row", np.array([1.0, 2, np.nan, np.nan, 5, 6, 7, 8, 9, 10]), 4),
    ("const", np.array([5.0, 5, 5, 5, 5, 5, 5, 5, 5, 5]), 3),
    ("small", np.array([1.0, 2, 3]), 5),
    ("window_eq1", np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 1),
    ("decreasing", np.array([10.0, 9, 8, 7, 6, 5, 4, 3, 2, 1]), 4),
]


@pytest.mark.parametrize(("name", "arr", "window"), _EDGE, ids=[e[0] for e in _EDGE])
def test_ts_sum_edge(name, arr, window) -> None:
    _assert_parity(
        arr, window,
        lambda d, w: d.rolling(w, min_periods=1).sum().to_numpy(),
        lambda f, w: TSSumNative()._calculate_series(f, w).to_numpy(),
    )


@pytest.mark.parametrize(("name", "arr", "window"), _EDGE, ids=[e[0] for e in _EDGE])
def test_ts_mean_edge(name, arr, window) -> None:
    _assert_parity(
        arr, window,
        lambda d, w: d.rolling(w, min_periods=1).mean().to_numpy(),
        lambda f, w: TSMeanNative()._calculate_series(f, w).to_numpy(),
    )


@pytest.mark.parametrize(("name", "arr", "window"), [e for e in _EDGE if e[2] > 1], ids=[e[0] for e in _EDGE if e[2] > 1])
def test_ts_std_edge(name, arr, window) -> None:
    # ts_std kernel declares d>=2 (sample std on a single observation is
    # undefined), so window==1 is outside its allowed domain.
    _assert_parity(
        arr, window,
        lambda d, w: d.rolling(w, min_periods=1).std().to_numpy(),
        lambda f, w: TSStdNative()._calculate_series(f, w).to_numpy(),
    )


@pytest.mark.parametrize(("name", "arr", "window"), _EDGE, ids=[e[0] for e in _EDGE])
def test_ts_max_edge(name, arr, window) -> None:
    _assert_parity(
        arr, window,
        lambda d, w: d.rolling(w, min_periods=1).max().to_numpy(),
        lambda f, w: TSMaxNative()._calculate_series(f, w).to_numpy(),
    )


@pytest.mark.parametrize(("name", "arr", "window"), _EDGE, ids=[e[0] for e in _EDGE])
def test_ts_min_edge(name, arr, window) -> None:
    _assert_parity(
        arr, window,
        lambda d, w: d.rolling(w, min_periods=1).min().to_numpy(),
        lambda f, w: TSMinNative()._calculate_series(f, w).to_numpy(),
    )


@pytest.mark.parametrize(("name", "arr", "window"), _EDGE, ids=[e[0] for e in _EDGE])
def test_ts_median_edge(name, arr, window) -> None:
    _assert_parity(
        arr, window,
        lambda d, w: d.rolling(w, min_periods=1).median().to_numpy(),
        lambda f, w: TSMedianNative()._calculate_series(f, w).to_numpy(),
    )


@pytest.mark.parametrize(("name", "arr", "window"), _EDGE, ids=[e[0] for e in _EDGE])
def test_ts_argmax_age_edge(name, arr, window) -> None:
    """ts_argmax canonical semantics = AGE (0 = current bar, tie = newest),
    identical to the pandas rolling_days_since_extreme reference."""
    _assert_parity(
        arr, window,
        lambda d, w: rolling_days_since_extreme(d, w, maximum=True).to_numpy(),
        lambda f, w: TSArgmaxNative()._calculate_series(f, w).to_numpy(),
    )


@pytest.mark.parametrize(("name", "arr", "window"), _EDGE, ids=[e[0] for e in _EDGE])
def test_ts_argmin_age_edge(name, arr, window) -> None:
    _assert_parity(
        arr, window,
        lambda d, w: rolling_days_since_extreme(d, w, maximum=False).to_numpy(),
        lambda f, w: TSArgminNative()._calculate_series(f, w).to_numpy(),
    )


def test_ts_sum_window_eq1_identity() -> None:
    a = np.array([3.0, -1.0, 4.0, 1.0, 5.0])
    _assert_parity(
        a, 1,
        lambda d, w: d.rolling(w, min_periods=1).sum().to_numpy(),
        lambda f, w: TSSumNative()._calculate_series(f, w).to_numpy(),
    )


def test_ts_max_window_eq1_identity() -> None:
    a = np.array([3.0, -1.0, 4.0, 1.0, 5.0])
    _assert_parity(
        a, 1,
        lambda d, w: d.rolling(w, min_periods=1).max().to_numpy(),
        lambda f, w: TSMaxNative()._calculate_series(f, w).to_numpy(),
    )