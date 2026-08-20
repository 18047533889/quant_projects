# -*- coding: utf-8 -*-
"""R19-030..059 time-series operator audit regression tests.

Covers:
- R19-030/031/032 ts_corr/ts_cov current-row policy (Numba fastpath == pandas
  slow path: a window statistic may exist at a row whose current pair is missing).
- R19-033..035 ts_beta unified rolling_beta authority (min_periods default 5,
  paired-finite, ParamSpec declared).
- R19-039..042/049 ts_argmax/argmin origin/tie convergence (index_from_oldest,
  tie=latest) + new age canonicals.
- R19-043/044 ts_product signed + zero-safe.
- R19-045..048 decay partial-window / missing / alpha domain.
- R19-050..055 hidden parameter casts -> ParamSpec (ts_quantile q in [0,1],
  ts_topk_sum, ts_moment k>=2, price_spread_deviation window alias).
- R19-056..059 ts_sharpe ann_factor > 0, ts_autocorr min_periods.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.operator_errors import OperatorParameterError
from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    ensure_cleaned_loaded()


def _frame(values, cols=None):
    cols = cols or ["A"]
    data = {c: values for c in cols}
    return pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=len(values)))


def _get(canonical, backend="pandas_numpy"):
    return OperatorRegistry.get(canonical, backend=backend)


# ---------------------------------------------------------------------------
# R19-039..042/049: argmax/argmin origin + tie
#
# Coordinated origin split (age vs index_from_oldest):
#   ts_argmax / ts_argmin           -> age (0 = current bar), tie = latest
#   ts_argmax_age / ts_argmin_age   -> age (same)
#   ts_argmax_index_from_oldest ... -> index (0 = oldest bar), tie = latest
# ---------------------------------------------------------------------------


def test_ts_argmax_age_semantics_tie_latest():
    # window [5, 1, 5]: max=5 at positions 0 and 2.  age (0=current) + tie
    # latest -> [0, 1, 0] (row 0 single-element age 0; row 2 newest max is the
    # current bar).  Partial windows with >= min_periods=1 finite values DO
    # produce an age; only all-missing windows are NaN.
    op = _get("ts_argmax")
    out = op.calculate(_frame([5.0, 1.0, 5.0]), 3)["A"].to_numpy()
    np.testing.assert_array_equal(out, [0.0, 1.0, 0.0])
    # [3,1,3,2] w4: newest max at row 2, current row 3 -> age = 1.
    out2 = op.calculate(_frame([3.0, 1.0, 3.0, 2.0]), 4)["A"].to_numpy()
    assert out2[-1] == 1.0


def test_ts_argmin_age_semantics_tie_latest():
    op = _get("ts_argmin")
    out = op.calculate(_frame([5.0, 1.0, 5.0]), 3)["A"].to_numpy()
    # min=1 at position 1 -> age 1 at the last row.
    np.testing.assert_array_equal(out, [0.0, 0.0, 1.0])


def test_ts_argmax_all_missing_is_null():
    op = _get("ts_argmax")
    x = _frame([np.nan, np.nan])
    assert op.calculate(x, 2).isna().all().all()


def test_ts_argmax_index_from_oldest_tie_latest():
    # [3,1,3,2] w4: index_from_oldest (0=oldest) + tie latest -> 2 at row 2 and 3.
    op = _get("ts_argmax_index_from_oldest")
    out = op.calculate(_frame([3.0, 1.0, 3.0, 2.0]), 4)["A"].to_numpy()
    assert out[-1] == 2.0


def test_ts_argmax_age_canonical_matches_ts_argmax():
    # ts_argmax_age (registered by safe_ops) reports 0 = current bar; it must
    # agree with ts_argmax.
    base = _get("ts_argmax")
    age = _get("ts_argmax_age")
    x = _frame([5.0, 1.0, 5.0])
    np.testing.assert_array_equal(
        base.calculate(x, 3)["A"].to_numpy(),
        age.calculate(x, 3)["A"].to_numpy(),
    )


def test_ts_argmax_rejects_non_integer_window():
    op = _get("ts_argmax")
    x = _frame([5.0, 1.0, 5.0])
    with pytest.raises(OperatorParameterError):
        op.calculate(x, window=3.5)


def test_ts_argmax_polars_matches_pandas_age():
    pl = pytest.importorskip("polars")
    x_pl = pl.DataFrame({"A": [5.0, 1.0, 5.0], "date": ["2024-01-01", "2024-01-02", "2024-01-03"]})
    op = _get("ts_argmax", backend="polars")
    out = op.calculate(x_pl, 3)
    vals = out["A"].to_list()
    np.testing.assert_array_equal(vals, [0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# R19-030/031/032: ts_corr/ts_cov current-row policy
# ---------------------------------------------------------------------------


def test_ts_corr_current_row_missing_keeps_window_statistic():
    # Current-row policy: a window statistic may exist at a row whose current
    # x/y pair is NaN, as long as >= min_periods finite pairs are in the window.
    # The pandas slow path must agree with the Numba fastpath (no .where(valid)).
    x = _frame([1.0, 2.0, 3.0, 4.0, 5.0], cols=["A"])
    y = _frame([2.0, 4.0, 6.0, np.nan, 10.0], cols=["A"])
    op = _get("ts_corr")
    out = op.calculate(x, y, 3)["A"].to_numpy()
    # window at idx3 = {[2,3,4] x [4,6,nan]} -> pairs (2,4),(3,6) -> corr=1.0,
    # even though the CURRENT pair (4, nan) is missing.
    assert np.isclose(out[3], 1.0)
    assert np.isclose(out[-1], 1.0)  # pairs (3,6),(4,nan),(5,10) -> (3,6),(5,10)


def test_ts_corr_current_row_policy_matches_numba_fastpath():
    # Compare the pandas slow path against the numba kernel directly under the
    # same env flag the operator uses.
    x = _frame([1.0, 2.0, 3.0, 4.0, 5.0], cols=["A", "B"])
    y = _frame([2.0, np.nan, 6.0, 8.0, 10.0], cols=["A", "B"])
    op = _get("ts_corr")
    slow = op.calculate(x, y, 3)["A"].to_numpy()
    try:
        from backend.numba_kernels import rolling_corr_panel
        fast = rolling_corr_panel(x.to_numpy(dtype=float), y.to_numpy(dtype=float), 3, min_count=2)
    except Exception:
        pytest.skip("numba kernel unavailable")
    if fast is None:
        pytest.skip("numba kernel unavailable")
    np.testing.assert_allclose(slow, fast[:, 0], rtol=1e-12, atol=1e-12)


def test_ts_cov_current_row_missing_keeps_window_statistic():
    x = _frame([1.0, 2.0, 3.0, 4.0, 5.0], cols=["A"])
    y = _frame([2.0, 4.0, 6.0, np.nan, 10.0], cols=["A"])
    op = _get("ts_cov")
    out = op.calculate(x, y, 3)["A"].to_numpy()
    # window at idx3 = rows 1..3 -> valid pairs (2,4),(3,6); ddof=1 cov = 1.0
    # even though the CURRENT pair (4, nan) is missing.
    assert not np.isnan(out[3])
    assert np.isclose(out[3], 1.0, atol=1e-9)


def test_ts_corr_requires_exact_aligned_panels():
    # allow_panel_broadcast waiver removed: correlation panels must be exact
    # aligned (same columns in the same order).
    x = _frame([1.0, 2.0, 3.0], cols=["A"])
    y = _frame([1.0, 2.0, 3.0], cols=["B"])
    op = _get("ts_corr")
    assert "allow_panel_broadcast" not in op.metadata.tags
    with pytest.raises(ValueError):
        op.calculate(x, y, 2)


# ---------------------------------------------------------------------------
# R19-033..035: ts_beta unified rolling_beta authority
# ---------------------------------------------------------------------------


def test_ts_beta_min_periods_default_is_five_and_declared():
    op = _get("ts_beta")
    assert "min_periods" in op.metadata.param_names
    spec = op.metadata.param_specs["min_periods"]
    assert spec.dtype is int and spec.min == 2 and spec.default == 5
    assert not spec.searchable


def test_ts_beta_uses_paired_finite_mask():
    # ±Inf must not count as a valid pair.
    x = _frame([1.0, 2.0, 3.0, 4.0], cols=["A"])
    y = _frame([np.inf, 2.0, 4.0, 6.0], cols=["A"])  # y = 2*x on rows 1..3
    op = _get("ts_beta")
    out = op.calculate(y, x, 3, min_periods=2)["A"].to_numpy()
    # window at idx3 = {rows 1,2,3}: pairs (2,2),(3,4),(4,6) -> slope 2.0
    assert np.isclose(out[-1], 2.0)


def test_ts_beta_rejects_min_periods_above_window():
    op = _get("ts_beta")
    x = _frame([1.0, 2.0, 3.0], cols=["A"])
    # The central gate (and the kernel) reject min_periods > window.  Plain
    # ValueError is raised by the binder's common-integer-relation check.
    with pytest.raises(ValueError):
        op.calculate(x, x, 3, min_periods=4)


# ---------------------------------------------------------------------------
# R19-043/044: ts_product signed + zero-safe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "window", "expected"),
    [
        ([2.0, 0.0, 3.0], 3, 0.0),
        ([-2.0, -3.0], 2, 6.0),
        ([-2.0, 3.0], 2, -6.0),
        ([2.0, 3.0], 2, 6.0),
    ],
)
def test_ts_product_signed_zero_safe(values, window, expected):
    op = _get("ts_product")
    out = op.calculate(_frame(values), window)["A"].to_numpy()
    assert np.isclose(out[-1], expected)


def test_ts_product_all_missing_window_is_nan():
    op = _get("ts_product")
    out = op.calculate(_frame([np.nan, np.nan]), 2)["A"].to_numpy()
    assert np.isnan(out[-1])


# ---------------------------------------------------------------------------
# R19-045..048: decay partial-window / missing / alpha domain
# ---------------------------------------------------------------------------


def test_ts_sum_decay_partial_window_uses_newest_age_slots():
    # A partial window of length L must anchor on the L NEWEST age slots
    # (weights[-L:]) and renormalize, exactly like WMA.  With an increasing
    # weight sequence 2^(i/window), a 1-element partial window at row 0 returns
    # that value unchanged regardless of the oldest-slot weight.
    op = _get("ts_sum_decay")
    x = _frame([10.0, 20.0, 30.0], cols=["A"])
    out = op.calculate(x, 3)["A"].to_numpy()
    assert np.isclose(out[0], 10.0)  # single valid element
    # Full window [10,20,30]: weighted mean with newest weights.
    weights = np.array([2 ** (i / 3) for i in range(3)])
    w = weights[-3:]
    expected = np.dot([10.0, 20.0, 30.0], w) / w.sum()
    assert np.isclose(out[-1], expected)


def test_ts_sum_decay_skips_missing_and_rewights():
    # Missing (NaN) values are skipped and surviving weights renormalized —
    # a single NaN never collapses the whole window.
    op = _get("ts_sum_decay")
    x = _frame([10.0, np.nan, 30.0], cols=["A"])
    out = op.calculate(x, 3)["A"].to_numpy()
    assert not np.isnan(out[-1])
    weights = np.array([2 ** (i / 3) for i in range(3)])
    w = weights[-3:][[0, 2]]  # drop the NaN slot's weight
    expected = np.dot([10.0, 30.0], w) / w.sum()
    assert np.isclose(out[-1], expected)


def test_ts_decay_exp_window_alpha_domain():
    op = _get("ts_decay_exp_window")
    x = _frame([1.0, 2.0, 3.0], cols=["A"])
    for bad in (0.0, -0.5, 1.5, float("nan"), float("inf")):
        with pytest.raises(OperatorParameterError):
            op.calculate(x, 3, alpha=bad)
    # alpha=1 -> uniform weights -> plain rolling mean of the window.
    out = op.calculate(x, 3, alpha=1.0)["A"].to_numpy()
    assert np.isclose(out[-1], 2.0)


def test_ts_decay_exp_window_partial_matches_shared_policy():
    op = _get("ts_decay_exp_window")
    x = _frame([10.0, np.nan, 30.0], cols=["A"])
    out = op.calculate(x, 3, alpha=0.5)["A"].to_numpy()
    assert not np.isnan(out[-1])
    weights = np.array([0.5 ** i for i in range(3)][::-1])
    w = weights[-3:][[0, 2]]
    expected = np.dot([10.0, 30.0], w) / w.sum()
    assert np.isclose(out[-1], expected)


# ---------------------------------------------------------------------------
# R19-050..055: hidden parameter casts -> ParamSpec
# ---------------------------------------------------------------------------


def test_ts_quantile_q_must_be_in_unit_interval():
    op = _get("ts_quantile")
    x = _frame([1.0, 2.0, 3.0], cols=["A"])
    assert "window" in op.metadata.param_aliases
    assert "p" in op.metadata.param_aliases
    for bad in (-0.1, 1.1):
        with pytest.raises(OperatorParameterError):
            op.calculate(x, 3, q=bad)
    with pytest.raises(OperatorParameterError):
        op.calculate(x, 3, p=1.5)  # alias also validated
    # window alias works and binds to d.
    out = op.calculate(x, window=3, p=0.5)["A"].to_numpy()
    assert np.isclose(out[-1], 2.0)


def test_ts_topk_sum_no_hidden_casts():
    op = _get("ts_topk_sum")
    x = _frame([3.0, 1.0, 2.0], cols=["A"])
    # top-2 sum over window of 3 -> 3+2=5.
    out = op.calculate(x, 3, 2)["A"].to_numpy()
    assert np.isclose(out[-1], 5.0)
    # k must be <= window (central gate raises ValueError).
    with pytest.raises(ValueError):
        op.calculate(x, 3, k=4)


def test_price_spread_deviation_window_alias_declared():
    op = _get("price_spread_deviation")
    x = _frame([1.0, 2.0, 4.0], cols=["A"])
    out = op.calculate(x, window=2)["A"].to_numpy()
    # x/mean(window 2) - 1 at last row: 4/3 - 1
    assert np.isclose(out[-1], 4.0 / 3.0 - 1.0)


def test_ts_moment_k_ge_two_and_no_local_cast():
    op = _get("ts_moment")
    x = _frame([1.0, 2.0, 3.0], cols=["A"])
    assert op.metadata.param_specs["k"].min == 2
    with pytest.raises(OperatorParameterError):
        op.calculate(x, 3, k=1)
    out = op.calculate(x, 3, k=2)["A"].to_numpy()
    # central second moment of [1,2,3] = 2/3
    assert np.isclose(out[-1], 2.0 / 3.0)


# ---------------------------------------------------------------------------
# R19-056..059: ts_sharpe ann_factor / min_periods
# ---------------------------------------------------------------------------


def test_ts_sharpe_ann_factor_must_be_positive():
    op = _get("ts_sharpe")
    x = _frame([1.0, 2.0, 3.0, 4.0], cols=["A"])
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(OperatorParameterError):
            op.calculate(x, 3, ann_factor=bad)
    out = op.calculate(x, 3, ann_factor=1.0)["A"].to_numpy()
    assert not np.isnan(out[-1])


def test_ts_sharpe_ann_factor_not_searchable():
    op = _get("ts_sharpe")
    spec = op.metadata.param_specs["ann_factor"]
    assert not spec.searchable


def test_ts_autocorr_min_periods_declared():
    op = _get("ts_autocorr")
    assert "min_periods" in op.metadata.param_names
    spec = op.metadata.param_specs["min_periods"]
    assert spec.dtype is int and not spec.searchable
    x = _frame([1.0, 2.0, 3.0, 4.0], cols=["A"])
    out = op.calculate(x, 4, lag=1, min_periods=3)["A"].to_numpy()
    assert not np.isnan(out[-1])


def test_ts_autocorr_lag_ge_window_rejected():
    op = _get("ts_autocorr")
    x = _frame([1.0, 2.0, 3.0], cols=["A"])
    with pytest.raises(OperatorParameterError):
        op.calculate(x, 5, lag=5)
