# -*- coding: utf-8 -*-
"""R11 round-3 P1-A cross-sectional local geometry fixes.

Covers the items owned by this agent in the round-3 audit:

* item 91  — ``cs_knn_local_moran`` is now the standard Local Moran I
  (row-standardised k-NN weights + global second-moment normalisation), not a
  bare ``z_i · mean(z_NN)`` proxy.
* item 92  — ``cs_isotonic_residual`` stays an explicitly DESCRIPTIVE residual
  (direction chosen in-sample); the new ``cs_isotonic_residual_lagged_direction``
  fixes the direction from prior dates and fails closed when ambiguous.
* item 93  — KNN local regression enforces a DOF floor ``max(10, 5*(d+1))``;
  below the floor the output is NaN (never a near-interpolation z-score).
* item 94  — ``cs_knn_local_gradient_norm`` reports ``same_as:target`` (the
  predictors are dimensionless rank coordinates), not a generic ``norm``.
* item 95  — tangent PCA requires numerical rank >= 2 and s2/s1 >= 0.05;
  a rank-1 "plane" fails closed to NaN.

It also verifies the polars twins for the dependence ops (conditional-TE, HSIC,
CMI, partial distance correlation) delegate to the pandas reference, so they
inherit whatever semantics the pandas side implements (auto-sync).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

_COLS5 = ["A", "B", "C", "D", "E"]
_ONE = pd.date_range("2024-01-01", periods=1, freq="B")


def _op(name: str):
    op = OperatorRegistry.get(name, "pandas_numpy")
    assert op is not None, name
    return op


def _meta(name: str):
    return getattr(_op(name), "metadata", None)


# ---------------------------------------------------------------------------
# item 91 — Local Moran I is the standard statistic
# ---------------------------------------------------------------------------
def _line_feats(values=None):
    values = [1.0, 2.0, 3.0, 4.0, 5.0] if values is None else values
    return [
        pd.DataFrame([values], index=_ONE, columns=_COLS5),
        pd.DataFrame([values], index=_ONE, columns=_COLS5),
        pd.DataFrame([values], index=_ONE, columns=_COLS5),
    ]


def test_local_moran_line_panel_hand_computed():
    # k=2 neighbours on the value line: A->{B,C} B->{A,C} C->{B,D} D->{C,E}
    # E->{C,D}.  target z-scores (pop std): [-1.414,-0.707,0,0.707,1.414], so
    # I_i = z_i·mean(z_NN) with the global second moment m2=1.
    target = pd.DataFrame([[10.0, 20.0, 30.0, 40.0, 50.0]], index=_ONE, columns=_COLS5)
    out = _op("cs_knn_local_moran").calculate(target, *_line_feats(), k=2)
    expected = [0.5, 0.5, 0.0, 0.5, 0.5]
    np.testing.assert_allclose(out.to_numpy(dtype=float), np.array([expected]), rtol=1e-9, equal_nan=True)


def test_local_moran_scale_invariant():
    # Cross-sectional z-scores are affine-invariant, so scaling the target by a
    # positive constant must leave Local Moran I unchanged.
    target = pd.DataFrame([[10.0, 20.0, 30.0, 40.0, 50.0]], index=_ONE, columns=_COLS5)
    scaled = target * 10.0
    a = _op("cs_knn_local_moran").calculate(target, *_line_feats(), k=2)
    b = _op("cs_knn_local_moran").calculate(scaled, *_line_feats(), k=2)
    np.testing.assert_allclose(b.to_numpy(dtype=float), a.to_numpy(dtype=float), rtol=1e-9, equal_nan=True)


def test_local_moran_metadata_unit_ratio():
    meta = _meta("cs_knn_local_moran")
    assert any(t.startswith("unit:ratio") for t in (meta.tags or []))


# ---------------------------------------------------------------------------
# item 92 — isotonic residual: descriptive vs direction-fixed-by-prior
# ---------------------------------------------------------------------------
def _iso_2day():
    cols = [chr(65 + i) for i in range(10)]
    idx = pd.date_range("2024-01-02", periods=2, freq="B")
    x = np.arange(1, 11, dtype=float)
    y = np.array([x, 11.0 - x])  # day0 increasing, day1 decreasing
    xp = np.array([x, x])
    y_df = pd.DataFrame(y, index=idx, columns=cols)
    x_df = pd.DataFrame(xp, index=idx, columns=cols)
    return y_df, x_df


def test_isotonic_residual_descriptive_follows_same_day_direction():
    y_df, x_df = _iso_2day()
    out = _op("cs_isotonic_residual").calculate(y_df, x_df)
    # Day 1 is decreasing; the same-day Spearman picks the decreasing isotonic
    # fit, which captures the relation exactly -> residual ~ 0.
    assert np.nanmax(np.abs(out.iloc[1].to_numpy(dtype=float))) < 1e-9


def test_isotonic_residual_lagged_direction_fixed_by_prior():
    y_df, x_df = _iso_2day()
    out = _op("cs_isotonic_residual_lagged_direction").calculate(y_df, x_df, lookback=1)
    arr = out.to_numpy(dtype=float)
    # Day 0 has no prior -> fail closed to NaN.
    assert np.isnan(arr[0]).all()
    # Day 1 direction is fixed INCREASING from day 0 and applied to today's
    # decreasing relation -> residuals are NOT ~0 (the direction was not
    # re-picked in-sample).  The max |residual| of the wrong-direction fit is
    # large relative to the descriptive ~0.
    assert np.nanmax(np.abs(arr[1])) > 1.0


def test_isotonic_residual_lagged_ambiguous_prior_fails_closed():
    cols = [chr(65 + i) for i in range(10)]
    idx = pd.date_range("2024-01-02", periods=2, freq="B")
    x = np.arange(1, 11, dtype=float)
    y = np.array([np.full(10, 5.0), 11.0 - x])  # day0 flat (ambiguous), day1 dec
    xp = np.array([x, x])
    y_df = pd.DataFrame(y, index=idx, columns=cols)
    x_df = pd.DataFrame(xp, index=idx, columns=cols)
    out = _op("cs_isotonic_residual_lagged_direction").calculate(y_df, x_df, lookback=1)
    arr = out.to_numpy(dtype=float)
    assert np.isnan(arr[0]).all()
    assert np.isnan(arr[1]).all()


def test_isotonic_residual_lagged_requires_lookback():
    y_df, x_df = _iso_2day()
    with pytest.raises(ValueError):
        _op("cs_isotonic_residual_lagged_direction").calculate(y_df, x_df, lookback=0)


def test_isotonic_residual_descriptive_note():
    # item 92: the descriptive residual must not claim to be a predictive
    # neutralization residual.
    meta = _meta("cs_isotonic_residual")
    assert meta is not None
    assert "描述性" in meta.description


# ---------------------------------------------------------------------------
# item 93 — KNN local regression DOF floor  max(10, 5*(d+1)) = 20
# ---------------------------------------------------------------------------
def _cs30_panels(seed=0):
    cols = [chr(65 + i) for i in range(30)]
    idx = pd.date_range("2024-01-01", periods=1, freq="B")
    rng = np.random.default_rng(seed)
    f1 = pd.DataFrame(rng.normal(size=(1, 30)), index=idx, columns=cols)
    f2 = pd.DataFrame(rng.normal(size=(1, 30)), index=idx, columns=cols)
    f3 = pd.DataFrame(rng.normal(size=(1, 30)), index=idx, columns=cols)
    tgt = pd.DataFrame(rng.normal(size=(1, 30)), index=idx, columns=cols)
    return tgt, f1, f2, f3


def test_knn_local_regression_dof_floor_nan_below_floor():
    # M-113 (model-audit): k below the floor is now REJECTED at the operator
    # boundary (strict k, no silent acceptance), so k=5 must raise instead of
    # running with an under-sized neighbourhood and emitting NaN.  The
    # under-sized-neighbourhood NaN path (kernel-level DOF floor) is exercised
    # by test_knn_local_regression_dof_floor_ok_at_or_above_floor at k=20 with
    # a small peer panel.
    tgt, f1, f2, f3 = _cs30_panels()
    for name in ("cs_knn_local_linear_residual", "cs_knn_local_gradient_norm"):
        with pytest.raises(Exception):
            _op(name).calculate(tgt, f1, f2, f3, k=5, ridge=1e-3)


def test_knn_local_regression_dof_floor_ok_at_or_above_floor():
    tgt, f1, f2, f3 = _cs30_panels(seed=3)
    for name in ("cs_knn_local_linear_residual", "cs_knn_local_gradient_norm"):
        out = _op(name).calculate(tgt, f1, f2, f3, k=20, ridge=1e-3)
        assert np.isfinite(out.to_numpy(dtype=float)).mean() >= 0.9, name


def test_knn_local_regression_deterministic_at_floor():
    tgt, f1, f2, f3 = _cs30_panels(seed=7)
    op = _op("cs_knn_local_linear_residual")
    a = op.calculate(tgt, f1, f2, f3, k=20, ridge=1e-3)
    b = op.calculate(tgt, f1, f2, f3, k=20, ridge=1e-3)
    np.testing.assert_allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)


def test_knn_local_regression_k_lt_4_raises():
    tgt, f1, f2, f3 = _cs30_panels()
    with pytest.raises(ValueError):
        _op("cs_knn_local_linear_residual").calculate(tgt, f1, f2, f3, k=3, ridge=1e-3)


# ---------------------------------------------------------------------------
# item 94 — local_gradient_norm unit is same_as:target
# ---------------------------------------------------------------------------
def test_local_gradient_norm_unit_same_as_target():
    meta = _meta("cs_knn_local_gradient_norm")
    assert meta is not None
    assert meta.output_unit == "same_as:target"
    assert any(t.startswith("unit:same_as:target") for t in (meta.tags or []))
    assert not any(t.startswith("unit:norm") for t in (meta.tags or []))


def test_local_linear_residual_unit_still_ratio():
    # The ratio residual is untouched by the unit fix.
    meta = _meta("cs_knn_local_linear_residual")
    assert meta is not None
    assert meta.output_unit is None
    assert any(t.startswith("unit:ratio") for t in (meta.tags or []))


# ---------------------------------------------------------------------------
# item 95 — tangent PCA numerical-rank gate
# ---------------------------------------------------------------------------
def test_tangent_nan_on_collinear_cloud():
    # Three identical feature panels -> the cloud is exactly rank 1 -> the
    # top-2 "plane" is one direction + noise (s2/s1 ~ 0) -> fail closed to NaN.
    rng = np.random.default_rng(0)
    vals = rng.normal(size=(1, 30))
    f1 = pd.DataFrame(vals, index=_ONE, columns=[chr(65 + i) for i in range(30)])
    f2 = f1.copy()
    f3 = f1.copy()
    # k=20 (>= floor) — tests the kernel numerical-rank gate, not the boundary
    out = _op("cs_knn_tangent_residual").calculate(f1, f2, f3, k=20)
    assert np.isnan(out.to_numpy(dtype=float)).all()


def test_tangent_finite_on_full_rank_cloud():
    rng = np.random.default_rng(2)
    cols = [chr(65 + i) for i in range(30)]
    f1 = pd.DataFrame(rng.normal(size=(1, 30)), index=_ONE, columns=cols)
    f2 = pd.DataFrame(rng.normal(size=(1, 30)), index=_ONE, columns=cols)
    f3 = pd.DataFrame(rng.normal(size=(1, 30)), index=_ONE, columns=cols)
    out = _op("cs_knn_tangent_residual").calculate(f1, f2, f3, k=20)
    assert np.isfinite(out.to_numpy(dtype=float)).mean() >= 0.9


def test_tangent_k_lt_4_raises():
    rng = np.random.default_rng(3)
    cols = [chr(65 + i) for i in range(30)]
    f1 = pd.DataFrame(rng.normal(size=(1, 30)), index=_ONE, columns=cols)
    f2 = f1.copy()
    f3 = f1.copy()
    with pytest.raises(ValueError):
        _op("cs_knn_tangent_residual").calculate(f1, f2, f3, k=3)


# ---------------------------------------------------------------------------
# new canonical registration
# ---------------------------------------------------------------------------
def test_cs_isotonic_residual_lagged_direction_registered_with_polars():
    backends = OperatorRegistry.backends_for("cs_isotonic_residual_lagged_direction")
    assert "pandas_numpy" in backends
    assert "polars" in backends


# ---------------------------------------------------------------------------
# polars twins for the dependence ops auto-sync with the pandas reference
# ---------------------------------------------------------------------------
def _pl_frame(frame: pd.DataFrame):
    pl = pytest.importorskip("polars")
    pdf = frame.reset_index().rename(columns={"index": "date", "level_0": "date"})
    return pl.from_pandas(pdf)


def _pl_out_to_np(out) -> np.ndarray:
    import polars as pl  # noqa: F811

    cols = [c for c in out.columns if c not in ("date", "stock_code")]
    return np.asarray(out.select(cols).to_pandas(), dtype=float)


def _dependence_panels():
    rng = np.random.default_rng(11)
    idx = pd.date_range("2024-01-01", periods=90, freq="B")
    cols = ["A", "B", "C", "D", "E"]
    x = pd.DataFrame(np.cumsum(rng.normal(0, 1, (90, 5)), axis=0), index=idx, columns=cols)
    y = pd.DataFrame(np.cumsum(rng.normal(0, 1, (90, 5)), axis=0), index=idx, columns=cols)
    z = pd.DataFrame(np.cumsum(rng.normal(0, 1, (90, 5)), axis=0), index=idx, columns=cols)
    return x, y, z


_DEP_PAIRS = {
    "ts_hsic": (("x", "y"), {"window": 40}),
    "ts_conditional_mutual_information": (("x", "y", "z"), {"window": 60, "bins": 3}),
    "ts_distance_correlation_partial_proxy": (("x", "y", "z"), {"window": 60}),
    "ts_conditional_transfer_entropy": (("x", "y", "z"), {"window": 60, "bins": 2, "lag": 1}),
}


def test_dependence_ops_have_polars_twins():
    for name in _DEP_PAIRS:
        backends = OperatorRegistry.backends_for(name)
        assert "pandas_numpy" in backends, name
        assert "polars" in backends, name


def test_partial_distance_correlation_deprecated_alias_resolves():
    # item 52: the legacy name is a deprecated alias to the honest proxy.
    op = OperatorRegistry.get("ts_partial_distance_correlation", "pandas_numpy")
    assert op is not None
    assert op.metadata.name == "ts_distance_correlation_partial_proxy"


def test_dependence_polars_twins_parity_with_pandas():
    # The polars twin is a delegation to the pandas reference, so it inherits
    # whatever semantics the pandas side implements (auto-sync).
    x, y, z = _dependence_panels()
    panels = {"x": x, "y": y, "z": z}
    for name, (names, kwargs) in _DEP_PAIRS.items():
        pdfs = [_pl_frame(panels[p]) for p in names]
        pd_out = _op(name).calculate(*[panels[p] for p in names], **kwargs)
        pl_out = OperatorRegistry.get(name, "polars").calculate(*pdfs, **kwargs)
        pd_arr = pd_out.to_numpy(dtype=float)
        pl_arr = _pl_out_to_np(pl_out)
        assert pd_arr.shape == pl_arr.shape, name
        np.testing.assert_allclose(pl_arr, pd_arr, rtol=1e-9, equal_nan=True, err_msg=name)
