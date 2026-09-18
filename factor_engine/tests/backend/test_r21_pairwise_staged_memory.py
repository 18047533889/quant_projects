"""Nontrivial-window correctness regressions for bounded expression construction.

RSS is checked by the external test watchdog; this suite does not assert a
portable fixed memory limit or claim O(1) storage in the window size.
"""
import numpy as np
import polars as pl
import pytest

from factor_engine.cleaned_operators.common.polars_ts_rolling import TSCorrNative, TSCovNative


@pytest.mark.parametrize("window", [20, 60])
@pytest.mark.parametrize("cls,kind", [(TSCorrNative, "corr"), (TSCovNative, "cov")])
def test_large_window_matches_direct_oracle_and_preserves_metadata(window, cls, kind):
    rng = np.random.default_rng(217)
    x = 1.e9 + rng.normal(size=80)
    y = 1.e9 + 0.6 * (x - 1.e9) + rng.normal(size=80)
    x[11] = np.nan
    y[26] = np.inf
    # Internal-looking instrument names must not collide with staging columns.
    left = pl.DataFrame({"date": np.arange(80), "__xw0": x, "stock_code": ["A"] * 80})
    right = pl.DataFrame({"date": np.arange(80), "__xw0": y, "stock_code": ["A"] * 80})
    actual = cls()._calculate_series(left, right, window=window, min_periods=2)
    assert actual.columns == left.columns
    assert actual["date"].equals(left["date"])
    assert actual["stock_code"].equals(left["stock_code"])
    expected = []
    for end in range(80):
        xs, ys = x[max(0, end - window + 1):end + 1], y[max(0, end - window + 1):end + 1]
        valid = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[valid], ys[valid]
        if len(xs) < 2:
            expected.append(np.nan)
            continue
        # Independent direct-window calculation; no Polars expression helper.
        dx, dy = xs - xs.mean(), ys - ys.mean()
        cross = np.dot(dx, dy)
        if kind == "cov":
            expected.append(cross / (len(xs) - 1))
        else:
            expected.append(cross / np.sqrt(np.dot(dx, dx) * np.dot(dy, dy)))
    np.testing.assert_allclose(
        actual["__xw0"].to_numpy(), expected, equal_nan=True, rtol=1.e-9, atol=1.e-10
    )
    prefix = cls()._calculate_series(left.head(55), right.head(55), window=window, min_periods=2)
    np.testing.assert_allclose(prefix["__xw0"].to_numpy(), actual["__xw0"].head(55).to_numpy(),
                               equal_nan=True, rtol=1.e-12, atol=1.e-12)


@pytest.mark.parametrize("cls", [TSCorrNative, TSCovNative])
def test_metadata_only_and_empty_inputs_preserve_shape(cls):
    frame = pl.DataFrame({"date": [1, 2], "stock_code": ["A", "A"]})
    out = cls()._calculate_series(frame, frame, window=20)
    assert out.equals(frame)
    empty = pl.DataFrame(schema={"date": pl.Int64, "asset": pl.Float64, "stock_code": pl.String})
    out = cls()._calculate_series(empty, empty, window=60)
    assert out.shape == empty.shape
    assert out.schema == empty.schema
