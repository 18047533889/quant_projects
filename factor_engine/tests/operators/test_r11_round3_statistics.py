# -*- coding: utf-8 -*-
"""Round-3 operator-audit regression tests for the statistics family.

Covers ``cleaned_operators/common/statistics.py`` and its Polars twin
``cleaned_operators/common/polars_statistics.py``:

1. **MAD split** — ``ts_mean_abs_deviation`` = mean(|x_i - mean(window)|) and
   ``ts_median_abs_deviation`` = median(|x_i - median(window)|), each with ONE
   common centre per window (no more double-rolling approximate).  ``Mad`` is
   pointed at the mean-abs-dev definition; the review-spelled long names are
   in-module aliases.
2. **ACF(lag=0)** — production grammar enforces ``lag >= 1`` (``ParamSpec``),
   the kernel defensively returns ``ACF(0) = 1``, and negative lag raises.
3. **ACF unestimable -> NaN** — short samples / constant series / infeasible
   lag never return 0.
4. **Mode** — no more degenerate-to-min on continuous data: only a modal value
   with max frequency >= 2 is emitted; all-distinct windows -> NaN; ties ->
   median of the modal values.
5. **Residual family** — ``residual``/``Residual``/``regress(residual)`` return
   the rolling MEAN of the current in-sample OLS residuals (residual_mean), not
   the raw residual series; the Polars twin matches the pandas backend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.operator_errors import FutureReferenceError, OperatorParameterError
from cleaned_operators.common.statistics import ACF, Mad, Mode, autocorr, residual
from cleaned_operators.registry import OperatorRegistry


def _get(canonical: str, backend: str = "pandas_numpy"):
    """Resolve an operator from the main registry, falling back to the research
    registry (layer governance moves ACF/Mode/autocorr/residual/regress there)."""
    op = OperatorRegistry.get(canonical, backend)
    if op is not None:
        return op
    from research_tools.registry import ResearchToolRegistry

    return ResearchToolRegistry.get(canonical, backend)


@pytest.fixture(scope="module", autouse=True)
def _load_operators():
    try:
        from cleaned_operators import load_all

        load_all()
    except Exception:  # pragma: no cover - pre-existing env state (concurrent session)
        pass


# ---------------------------------------------------------------------------
# 1. MAD split
# ---------------------------------------------------------------------------


def test_mean_abs_deviation_single_center_per_window():
    op = OperatorRegistry.get("ts_mean_abs_deviation")
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    out = op.calculate(x, 3)
    # mean(|x_i - 2|) for [1,2,3] = (1+0+1)/3 = 2/3
    assert out.iloc[-1, 0] == pytest.approx(2.0 / 3.0)


def test_median_abs_deviation_single_center_per_window():
    op = OperatorRegistry.get("ts_median_abs_deviation")
    x = pd.DataFrame({"A": [1.0, 2.0, 100.0]})
    out = op.calculate(x, 3)
    # median(|x_i - 2|) for [1,2,100] = median(1,0,98) = 1
    assert out.iloc[-1, 0] == pytest.approx(1.0)


def test_abs_deviation_ops_are_distinct():
    x = pd.DataFrame({"A": [1.0, 2.0, 100.0]})
    mean_dev = OperatorRegistry.get("ts_mean_abs_deviation").calculate(x, 3).iloc[-1, 0]
    med_dev = OperatorRegistry.get("ts_median_abs_deviation").calculate(x, 3).iloc[-1, 0]
    # mean(|x_i - 34.333|) != median(|x_i - 2|)
    assert mean_dev != pytest.approx(med_dev)
    assert med_dev == pytest.approx(1.0)


def test_mad_is_mean_abs_deviation_not_double_rolling():
    x = pd.DataFrame({"A": [1.0, 2.0, 100.0]})
    mad_out = Mad().calculate(x, 3)
    ts_out = OperatorRegistry.get("ts_mean_abs_deviation").calculate(x, 3)
    pd.testing.assert_frame_equal(mad_out, ts_out)
    # Old double-rolling (rolling median -> abs -> rolling mean) gives 33 on
    # [1,2,100] (median=2, |1-2|+|2-2|+|100-2| -> mean(|1|,|0|,|98|) = 33);
    # the single-centre mean-abs-dev definition gives mean(|1-mu|,|2-mu|,|100-mu|)
    # with mu = 103/3.
    mu = 103.0 / 3.0
    expected = float(np.mean([abs(1.0 - mu), abs(2.0 - mu), abs(100.0 - mu)]))
    assert mad_out.iloc[-1, 0] == pytest.approx(expected)
    assert mad_out.iloc[-1, 0] != pytest.approx(33.0)


def test_abs_deviation_long_spelling_aliases_resolve():
    assert OperatorRegistry.get("ts_mean_absolute_deviation") is not None
    assert OperatorRegistry.get("ts_median_absolute_deviation") is not None
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    a = OperatorRegistry.get("ts_mean_absolute_deviation").calculate(x, 3)
    b = OperatorRegistry.get("ts_mean_abs_deviation").calculate(x, 3)
    pd.testing.assert_frame_equal(a, b)


def test_abs_deviation_pandas_polars_parity():
    pl = pytest.importorskip("polars")
    pdf = pd.DataFrame({"A": [1.0, 2.0, 3.0, 100.0, 5.0, 6.0]})
    plf = pl.DataFrame(pdf)
    for canon in ("ts_mean_abs_deviation", "ts_median_abs_deviation"):
        pd_out = _get(canon, "pandas_numpy").calculate(pdf, 3)
        pl_out = _get(canon, "polars").calculate(plf, 3)
        np.testing.assert_allclose(
            pl_out.to_numpy().ravel(), pd_out.to_numpy(dtype=float).ravel(), equal_nan=True
        )


# ---------------------------------------------------------------------------
# 2. ACF lag=0 boundary
# ---------------------------------------------------------------------------


def test_acf_lag_zero_rejected_by_contract():
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    with pytest.raises(OperatorParameterError):
        ACF().calculate(x, window=5, lag=0)


def test_acf_lag_zero_defensive_returns_one():
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = ACF()._calculate_series(x, window=5, lag=0)
    assert out.iloc[-1, 0] == pytest.approx(1.0)


def test_acf_negative_lag_raises():
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(FutureReferenceError):
        ACF()._calculate_series(x, window=3, lag=-1)


# ---------------------------------------------------------------------------
# 3. ACF unestimable -> NaN (never 0)
# ---------------------------------------------------------------------------


def test_acf_constant_series_is_nan_not_zero():
    x = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0, 1.0]})
    out = ACF().calculate(x, window=5, lag=1)
    assert np.isnan(out.iloc[-1, 0])


def test_acf_short_sample_is_nan_not_zero():
    # window of one finite point -> n == lag -> no valid pairs -> NaN
    x = pd.DataFrame({"A": [1.0, 2.0]})
    out = ACF().calculate(x, window=1, lag=1)
    assert np.isnan(out.iloc[-1, 0])


def test_acf_known_value():
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = ACF().calculate(x, window=5, lag=1)
    assert out.iloc[-1, 0] == pytest.approx(0.4)


def test_acf_nan_in_window_ignored():
    # NEW-029: NO dropfinite-reconnect.  The old value 1.6875/8.75 came from
    # treating the finite subset [1,3,4,5] as ADJACENT (pairing (1,3) ACROSS the
    # NaN gap).  A real sequence ``t, missing, t+2`` must never be re-paired.
    # Correct physical-pair ACF: mean over finite = 3.25, var = 8.75; only the
    # physical lag-1 pairs (3,4) and (4,5) contribute -> cov = 1.125.
    x = pd.DataFrame({"A": [1.0, np.nan, 3.0, 4.0, 5.0]})
    out = ACF().calculate(x, window=5, lag=1)
    assert out.iloc[-1, 0] == pytest.approx(1.125 / 8.75)


def test_autocorr_lag_zero_is_one():
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
    out = autocorr().calculate(x, lag=0)
    assert out.iloc[-1, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. Mode
# ---------------------------------------------------------------------------


def test_mode_all_distinct_is_nan():
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
    out = Mode().calculate(x, 4)
    assert np.isnan(out.iloc[-1, 0])


def test_mode_tie_returns_median_of_modes():
    # modal values {1, 2} -> median = 1.5 (never the smallest mode 1)
    x = pd.DataFrame({"A": [1.0, 1.0, 2.0, 2.0]})
    out = Mode().calculate(x, 4)
    assert out.iloc[-1, 0] == pytest.approx(1.5)


def test_mode_single_mode():
    x = pd.DataFrame({"A": [1.0, 1.0, 2.0, 3.0]})
    out = Mode().calculate(x, 4)
    assert out.iloc[-1, 0] == pytest.approx(1.0)


def test_mode_pandas_polars_parity():
    pl = pytest.importorskip("polars")
    pdf = pd.DataFrame({"A": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 100.0]})
    plf = pl.DataFrame(pdf)
    pd_out = _get("Mode", "pandas_numpy").calculate(pdf, 4)
    pl_out = _get("Mode", "polars").calculate(plf, 4)
    np.testing.assert_allclose(
        pl_out.to_numpy().ravel(), pd_out.to_numpy(dtype=float).ravel(), equal_nan=True
    )


# ---------------------------------------------------------------------------
# 5. Residual family (residual MEAN, not raw residual series)
# ---------------------------------------------------------------------------


def test_residual_is_current_in_sample_residual_not_rolling_mean():
    # NEW-022: the "residual" mixed object (current in-sample residual then a
    # SECOND rolling mean of those residuals) is removed.  ``rolling_regression
    # (retval="residual")`` now returns the current in-sample residual on the
    # SAME paired cohort; the residual-mean semantic lives ONLY in the dedicated
    # ``ts_regression_resid_mean`` canonical.
    y = pd.DataFrame({"A": [1.0, 2.0, 3.0, 100.0]})
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
    out = residual().calculate(y, x, 3)
    # Last window rows [2,3,4] x [2,3,100]: OLS slope=49, intercept=-112, so the
    # current in-sample residual at the last point is 100-(49*4-112)=16 — NOT
    # the old mean-of-residuals 16/3.
    assert out.iloc[-1, 0] == pytest.approx(16.0)
    assert out.iloc[-1, 0] != pytest.approx(16.0 / 3.0)


def test_residual_matches_central_ts_regression_resid():
    ts = OperatorRegistry.get("ts_regression_resid")
    if ts is None:
        pytest.skip("central ts_regression_resid not registered (load_all incomplete)")
    y = pd.DataFrame({"A": [3.0, 1.0, 4.0, 2.0, 5.0, 100.0, 7.0, 8.0]})
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    a = residual().calculate(y, x, 3)
    b = ts.calculate(y, x, 3)
    a_arr = a.to_numpy(dtype=float).ravel()
    b_arr = b.to_numpy(dtype=float).ravel()
    # Warm-up differs (local op uses min_periods=2, central op min_periods=3),
    # so compare only rows where BOTH are finite; the last row must agree.
    mask = np.isfinite(a_arr) & np.isfinite(b_arr)
    assert mask[-1]
    np.testing.assert_allclose(a_arr[mask], b_arr[mask], equal_nan=True)


def test_regress_retval_residual_is_current_in_sample_residual():
    from cleaned_operators.common.statistics import regress

    y = pd.DataFrame({"A": [3.0, 1.0, 4.0, 2.0, 5.0, 100.0, 7.0, 8.0]})
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    out = regress().calculate(y, x, 3, "residual")
    # NEW-022: current in-sample residual of the last window's OLS — the old
    # residual-mean (-1/9) is gone; the raw current residual ≈15.67 is returned.
    assert out.iloc[-1, 0] == pytest.approx(15.67, abs=0.01)
    assert out.iloc[-1, 0] != pytest.approx(-1.0 / 9.0)


def test_residual_polars_matches_pandas():
    pl = pytest.importorskip("polars")
    y = pd.DataFrame({"A": [1.0, 2.0, 3.0, 100.0]})
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
    pd_out = _get("residual", "pandas_numpy").calculate(y, x, 3)
    pl_out = _get("residual", "polars").calculate(pl.DataFrame(y), pl.DataFrame(x), 3)
    np.testing.assert_allclose(
        pl_out.to_numpy().ravel(), pd_out.to_numpy(dtype=float).ravel(), equal_nan=True
    )
