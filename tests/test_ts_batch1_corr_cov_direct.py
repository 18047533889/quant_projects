"""Direct centered-oracle regressions for the authoritative Polars ts_corr/ts_cov."""
from __future__ import annotations

import math

import polars as pl
import pytest

from factor_engine.cleaned_operators.common.polars_ts_rolling import TSCorrNative, TSCovNative


def _direct_centered_oracle(x, y, *, window=4, min_periods=2, ddof=1):
    """Compute trailing statistics from centered finite pairs, never raw moments."""
    corr = []
    cov = []
    for end in range(len(x)):
        start = max(0, end - window + 1)
        pairs = [
            (left, right)
            for left, right in zip(x[start : end + 1], y[start : end + 1])
            if left is not None
            and right is not None
            and math.isfinite(left)
            and math.isfinite(right)
        ]
        if len(pairs) < min_periods:
            corr.append(None)
            cov.append(None)
            continue

        mean_x = math.fsum(left for left, _ in pairs) / len(pairs)
        mean_y = math.fsum(right for _, right in pairs) / len(pairs)
        centered_x = [left - mean_x for left, _ in pairs]
        centered_y = [right - mean_y for _, right in pairs]
        cross = math.fsum(dx * dy for dx, dy in zip(centered_x, centered_y))
        ss_x = math.fsum(dx * dx for dx in centered_x)
        ss_y = math.fsum(dy * dy for dy in centered_y)

        cov.append(cross / (len(pairs) - ddof) if len(pairs) > ddof else None)
        corr.append(cross / math.sqrt(ss_x * ss_y) if ss_x > 0 and ss_y > 0 else None)
    return corr, cov


def _actual(x, y):
    left = pl.DataFrame({"value": x})
    right = pl.DataFrame({"value": y})
    corr = TSCorrNative()._calculate_series(left, right, window=4)["value"].to_list()
    cov = TSCovNative()._calculate_series(left, right, window=4)["value"].to_list()
    return corr, cov


def _assert_values(actual, expected):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        if want is None:
            assert got is None
        else:
            assert got == pytest.approx(want, rel=1e-12, abs=1e-12)


def test_window_four_large_offset_matches_direct_centered_finite_pair_oracle():
    x = [1.0e12 + 1.0, 1.0e12 + 2.0, math.nan, 1.0e12 + 4.0, 1.0e12 + 8.0, 1.0e12 + 16.0]
    y = [1.0e12 + 3.0, 1.0e12 + 7.0, 1.0e12 + 99.0, math.inf, 1.0e12 + 6.0, 1.0e12 + 11.0]

    actual_corr, actual_cov = _actual(x, y)
    expected_corr, expected_cov = _direct_centered_oracle(x, y)

    _assert_values(actual_corr, expected_corr)
    _assert_values(actual_cov, expected_cov)


def test_window_four_ordinary_scale_control_matches_same_centered_oracle():
    x = [1.0, 2.0, math.nan, 4.0, 8.0, 16.0]
    y = [3.0, 7.0, 99.0, math.inf, 6.0, 11.0]

    actual_corr, actual_cov = _actual(x, y)
    expected_corr, expected_cov = _direct_centered_oracle(x, y)

    _assert_values(actual_corr, expected_corr)
    _assert_values(actual_cov, expected_cov)
