# -*- coding: utf-8 -*-
"""R11 long-tail audit #131-#144: group statistics / JS distribution /
hierarchical neutralization / weighted moments / conditional operators.

Every assertion is a pinned reference; undefined math fails closed to NaN.
Covers:

* #131  group_topk_mean cutoff-tie validity matches fractional participation.
* #132  group JS out-of-reference values are captured (under/overflow bins).
* #133  group JS reference sample too small relative to bins -> NaN.
* #134  hierarchical_group_neutralize uses a composite (group, subgroup) key.
* #135  group_ex_self_mean / weighted_mean output unit = same_as:x.
* #137  cs_robust_resid cross-section minimum breadth gate.
* #138  weighted_moment _align is exact (misaligned axes fail loudly).
* #139/#140 cs_multi_robust_resid is ridge regression; the ridge is a
*        versioned non-searchable constant.
* #141  cs_multi_robust_resid residual-regression DOF margin gate.
* #142  conditional_ext output units (same_as:x / unit(y)/unit(x) / same_as:y).
* #143  ts_regression_resid_if min_periods constrains TRAINING observations.
* #144  condition inputs must be ConditionBool ({0,1} / NaN missing).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()


def _op(name: str):
    op = OperatorRegistry.get(name, "pandas_numpy")
    assert op is not None, name
    return op


def _meta(name: str):
    return getattr(_op(name), "metadata", None)


# ---------------------------------------------------------------------------
# #131 group_topk_mean cutoff-tie validity = fractional participation
# ---------------------------------------------------------------------------
def test_group_topk_mean_fractional_tie_participation() -> None:
    idx = pd.date_range("2024-01-01", periods=1)
    cols = list("ABCDEF")
    target = pd.DataFrame([[10.0, 20.0, 30.0, 40.0, 50.0, 100.0]], index=idx, columns=cols)
    score = pd.DataFrame([[5.0, 5.0, 5.0, 5.0, 5.0, 1.0]], index=idx, columns=cols)
    group = pd.DataFrame([["g"] * 6], index=idx, columns=cols)
    out = _op("group_topk_mean").calculate(target, score, group, k=3, exclude_self=False)
    # tie group = the five score-5 members, participates fractionally with
    # frac = n_take_eq / n_eq = 3/5 -> mean = (3/5)*(10+20+30+40+50)/3 = 30.
    np.testing.assert_allclose(out.to_numpy()[0], np.full(6, 30.0), rtol=1e-9)


def test_group_topk_mean_tie_nan_fails_row_order_invariant() -> None:
    idx = pd.date_range("2024-01-01", periods=1)
    cols = list("ABCDEF")
    # target missing on a tie member that is NOT among the first n_take_eq
    # (n_take_eq = 3, the NaN is at index 4): the fractional mean touches the
    # WHOLE tie group, so the row must fail closed, independent of column order.
    target = pd.DataFrame([[10.0, 20.0, 30.0, 40.0, np.nan, 100.0]], index=idx, columns=cols)
    score = pd.DataFrame([[5.0, 5.0, 5.0, 5.0, 5.0, 1.0]], index=idx, columns=cols)
    group = pd.DataFrame([["g"] * 6], index=idx, columns=cols)
    out = _op("group_topk_mean").calculate(target, score, group, k=3, exclude_self=False)
    assert out.isna().all().all()
    perm = list(reversed(cols))
    out2 = _op("group_topk_mean").calculate(
        target[perm], score[perm], group[perm], k=3, exclude_self=False
    )
    assert out2.isna().all().all()


# ---------------------------------------------------------------------------
# #132 group JS out-of-reference values are captured (under/overflow bins)
# ---------------------------------------------------------------------------
def test_group_js_out_of_reference_not_dropped() -> None:
    idx = pd.date_range("2024-01-01", periods=1)
    cols = [f"C{i}" for i in range(20)]
    vals = list(range(1, 19)) + [1000.0, 1001.0]
    x = pd.DataFrame([vals], index=idx, columns=cols)
    group = pd.DataFrame([["Y"] * 18 + ["X"] * 2], index=idx, columns=cols)
    out = _op("group_distribution_js_divergence").calculate(x, group, bins=5, min_group_size=2)
    row = out.to_numpy()[0]
    # The far-outlier X group lands in the overflow bin -> real positive JS,
    # never a silently re-normalized / NaN value.
    assert np.isfinite(row).any()
    assert np.all(row[np.isfinite(row)] > 0.0)


# ---------------------------------------------------------------------------
# #133 group JS reference sample must be large enough relative to bins
# ---------------------------------------------------------------------------
def test_group_js_small_reference_fails_closed() -> None:
    idx = pd.date_range("2024-01-01", periods=1)
    x = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], index=idx, columns=list("ABCDEF"))
    group = pd.DataFrame([["X", "X", "Y", "Y", "Y", "Y"]], index=idx, columns=list("ABCDEF"))
    # ex-group reference for X = 4 values < 2 * 10 requested bins -> NaN.
    out = _op("group_distribution_js_divergence").calculate(x, group, bins=10, min_group_size=2)
    assert out.isna().all().all()


# ---------------------------------------------------------------------------
# #134 hierarchical_group_neutralize must use a composite (group, subgroup) key
# ---------------------------------------------------------------------------
def test_hierarchical_neutralize_composite_key_no_cross_parent_merge() -> None:
    idx = pd.date_range("2024-01-01", periods=1)
    cols = list("ABCDEFGH")
    x = pd.DataFrame([[10.0, 0.0, 5.0, 5.0, 0.0, 0.0, 10.0, 10.0]], index=idx, columns=cols)
    group = pd.DataFrame([["G1", "G1", "G1", "G1", "G2", "G2", "G2", "G2"]], index=idx, columns=cols)
    subgroup = pd.DataFrame([["S", "S", "T", "T", "S", "S", "T", "T"]], index=idx, columns=cols)
    out = _op("hierarchical_group_neutralize").calculate(x, group, subgroup)
    # Composite (G,S) demeaning, NOT bare-subgroup demeaning that would merge the
    # S members of G1 and G2.  A bare-S demean would give [7.5, -2.5, ...];
    # the nested composite demean gives [-5, 5, 0, ...] per (G,S) cell.
    expected = [5.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(out.to_numpy()[0], expected, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# #135 group_ex_self_mean / weighted_mean output unit = same_as:x
# ---------------------------------------------------------------------------
def test_group_ex_self_mean_units_same_as_x() -> None:
    for name in ("group_ex_self_mean", "group_ex_self_weighted_mean"):
        meta = _meta(name)
        assert meta is not None, name
        assert meta.output_unit == "same_as:x", name
        assert any(t.startswith("unit:same_as:x") for t in (meta.tags or [])), name


def test_group_ex_self_mean_unit_algebra_scaling() -> None:
    idx = pd.date_range("2024-01-01", periods=1)
    x = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=idx, columns=list("ABCD"))
    group = pd.DataFrame([["g", "g", "h", "h"]], index=idx, columns=list("ABCD"))
    out1 = _op("group_ex_self_mean").calculate(x, group)
    out10 = _op("group_ex_self_mean").calculate(x * 10.0, group)
    # unit-covariant: scaling x by 10 scales the leave-one-out mean by 10.
    np.testing.assert_allclose(out10.to_numpy(), 10.0 * out1.to_numpy(), rtol=1e-6, equal_nan=True)


# ---------------------------------------------------------------------------
# #137 cross-section minimum breadth gate (cs_robust_resid)
# ---------------------------------------------------------------------------
def test_cs_robust_resid_min_breadth() -> None:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=5)
    small = pd.DataFrame(rng.standard_normal((5, 3)), index=idx, columns=list("ABC"))
    out = _op("cs_robust_resid").calculate(small, small * 0.5)
    assert out.isna().all().all()  # 3 stocks must not drive the regression
    big = pd.DataFrame(rng.standard_normal((5, 12)), index=idx, columns=[f"C{i}" for i in range(12)])
    out_big = _op("cs_robust_resid").calculate(big, big * 0.5)
    assert not out_big.isna().all().all()  # >= min_breadth names -> finite


# ---------------------------------------------------------------------------
# #138 weighted_moment _align must be exact (fail on misaligned axes)
# ---------------------------------------------------------------------------
def test_weighted_moment_align_is_exact() -> None:
    rng = np.random.default_rng(1)
    a = pd.DataFrame(rng.standard_normal((10, 3)), index=list(range(10)), columns=list("ABC"))
    shifted = pd.DataFrame(rng.standard_normal((10, 3)), index=list(range(1, 11)), columns=list("ABC"))
    with pytest.raises(ValueError):
        _op("ts_weighted_standardized_moment").calculate(a, shifted, window=5)
    diff_cols = pd.DataFrame(rng.standard_normal((10, 3)), index=list(range(10)), columns=list("ABD"))
    with pytest.raises(ValueError):
        _op("ts_cov_if").calculate(a, diff_cols, pd.DataFrame(1.0, index=list(range(10)), columns=list("ABD")), window=5)


# ---------------------------------------------------------------------------
# #139/#140 cs_multi_robust_resid is ridge regression; ridge is versioned &
#         non-searchable
# ---------------------------------------------------------------------------
def test_ridge_knob_versioned_and_non_searchable() -> None:
    from cleaned_operators.weighted_moment_ext import _RIDGE

    assert _RIDGE == 1e-3
    meta = _meta("cs_multi_robust_resid")
    assert meta is not None
    # the ridge is NOT a declared parameter -> it cannot enter the search grammar
    assert "ridge" not in (meta.param_names or [])
    assert "ridge" not in (meta.param_specs or {})
    # ...but the fixed value IS documented in the versioned contract
    desc = (meta.description or "").lower()
    assert "ridge" in desc and "1e-3" in desc
    assert "cs_multi_ridge_resid" in (meta.description or "")


# ---------------------------------------------------------------------------
# #141 residual regression DOF margin gate (cs_multi_robust_resid)
# ---------------------------------------------------------------------------
def test_cs_multi_robust_resid_dof_margin() -> None:
    rng = np.random.default_rng(2)
    idx = pd.date_range("2024-01-01", periods=6)
    small = pd.DataFrame(rng.standard_normal((6, 4)), index=idx, columns=list("ABCD"))
    small2 = pd.DataFrame(rng.standard_normal((6, 4)), index=idx, columns=list("ABCD"))
    y = 0.5 * small - 0.3 * small2 + 2.0
    out = _op("cs_multi_robust_resid").calculate(y, small, small2)
    assert out.isna().all().all()  # 4 stocks < max(20, 5*3) -> fail closed
    idx_big = pd.date_range("2024-01-02", periods=10)
    big = pd.DataFrame(rng.standard_normal((10, 25)), index=idx_big, columns=[f"C{i}" for i in range(25)])
    big2 = pd.DataFrame(rng.standard_normal((10, 25)), index=idx_big, columns=[f"C{i}" for i in range(25)])
    ybig = 0.5 * big - 0.3 * big2 + 2.0
    out_big = _op("cs_multi_robust_resid").calculate(ybig, big, big2)
    assert not out_big.isna().all().all()  # 25 stocks >= margin -> finite


# ---------------------------------------------------------------------------
# #142 conditional_ext output units
# ---------------------------------------------------------------------------
def test_conditional_ext_output_units() -> None:
    units = {
        "ts_min_if": "same_as:x",
        "ts_max_if": "same_as:x",
        "ts_quantile_if": "same_as:x",
        "ts_corr_if": "dimensionless",
        "ts_beta_if": "unit(y)/unit(x)",
        "ts_regression_resid_if": "same_as:y",
    }
    for name, unit in units.items():
        meta = _meta(name)
        assert meta is not None, name
        assert meta.output_unit == unit, name
        assert any(t.startswith(f"unit:{unit}") for t in (meta.tags or [])), name


# ---------------------------------------------------------------------------
# #143 ts_regression_resid_if min_periods constrains TRAINING observations
# ---------------------------------------------------------------------------
def test_ts_regression_resid_if_min_periods_is_training_count() -> None:
    op = _op("ts_regression_resid_if")
    x = pd.DataFrame(np.arange(10, dtype=float), index=list(range(10)), columns=["A"])
    y = pd.DataFrame(2.0 * np.arange(10, dtype=float) + 0.5, index=list(range(10)), columns=["A"])
    # Row 9: condition-true rows in window = {0,1,9} (total 3), but the TRAINING
    # rows (strictly before the current row) = {0,1} (2).  The fit drops the
    # current row, so min_periods=3 must fail closed on the training count.
    cond = pd.DataFrame([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], index=list(range(10)), columns=["A"])
    out = op.calculate(y, x, cond, window=10, min_periods=3)
    assert np.isnan(out["A"].iloc[9])
    # With a third training row the fit is valid and the (perfect linear) residual ~ 0.
    cond2 = pd.DataFrame([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], index=list(range(10)), columns=["A"])
    out2 = op.calculate(y, x, cond2, window=10, min_periods=3)
    assert abs(out2["A"].iloc[9] - 0.0) < 1e-6


# ---------------------------------------------------------------------------
# #144 condition inputs must be ConditionBool
# ---------------------------------------------------------------------------
def test_conditional_condition_must_be_condition_bool() -> None:
    op = _op("ts_min_if")
    x = pd.DataFrame([[1.0, 2.0, 3.0]], index=[0], columns=["A", "B", "C"])
    bad = pd.DataFrame([[0.0, 1.0, 2.0]], index=[0], columns=["A", "B", "C"])
    with pytest.raises(ValueError):
        op.calculate(x, bad, window=3)
    # {0, 1} with NaN as missing is valid.
    good = pd.DataFrame([[0.0, 1.0, np.nan]], index=[0], columns=["A", "B", "C"])
    out = op.calculate(x, good, window=3, min_periods=1)
    assert np.isnan(out["A"].iloc[0])  # condition false -> excluded
    assert out["B"].iloc[0] == pytest.approx(2.0)
    assert np.isnan(out["C"].iloc[0])  # condition missing -> excluded


def test_ts_cov_if_condition_must_be_condition_bool() -> None:
    op = _op("ts_cov_if")
    x = pd.DataFrame(np.arange(20, dtype=float).reshape(10, 2), index=list(range(10)), columns=["A", "B"])
    y = pd.DataFrame(np.arange(20, dtype=float).reshape(10, 2) + 1.0, index=list(range(10)), columns=["A", "B"])
    bad = pd.DataFrame(np.full((10, 2), 5.0), index=list(range(10)), columns=["A", "B"])
    with pytest.raises(ValueError):
        op.calculate(x, y, bad, window=5)
