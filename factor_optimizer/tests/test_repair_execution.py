import numpy as np
import pandas as pd
import pytest

from factor_optimizer.adapters.repair_execution import IneligibleValueRepair, compile_value_repair


def panel(a, b):
    dates = pd.date_range("2026-01-01", periods=len(a)); rows = []
    for date, av, bv in zip(dates, a, b): rows.extend((("A", date, av), ("B", date, bv)))
    return pd.DataFrame(rows, columns=["asset_id", "date", "value"])


def compile_(family, params):
    return compile_value_repair(family, params, natural_time_scale=5,
                                training_context_ref="train:fixed-v1")


def test_raw_and_sign_are_explicit_and_identity_is_stable():
    values = panel([1.0, np.nan], [2.0, 4.0])
    raw = compile_("NO_OP_RAW", {"keep_raw": True})
    flip = compile_("SIGN_ORIENTATION", {"direction": "flip"})
    keep = compile_("SIGN_ORIENTATION", {"direction": "keep"})
    pd.testing.assert_series_equal(raw.execute(values, allow_research=True), values.value)
    pd.testing.assert_series_equal(flip.execute(values, allow_research=True), -values.value)
    pd.testing.assert_series_equal(keep.execute(values, allow_research=True), values.value)
    assert raw.identity == compile_("NO_OP_RAW", {"keep_raw": True}).identity
    assert raw.identity != flip.identity


@pytest.mark.parametrize("family, sign", [("U_SHAPE_REPAIR", 1), ("INVERTED_U_REPAIR", -1)])
def test_u_repairs_use_same_date_percentile_rank_coordinates(family, sign):
    values = panel([1.0, 4.0], [3.0, 2.0])
    plan = compile_(family, {"center": 0.5, "power": 2.0, "asymmetry": False})
    np.testing.assert_allclose(plan.execute(values, allow_research=True), [sign*.25] * 4)


def test_u_shape_constant_and_missing_slice_is_explicit():
    values = panel([7.0, np.nan], [7.0, np.nan])
    out = compile_("U_SHAPE_REPAIR", {"center": .5, "power": 2., "asymmetry": False}).execute(values, allow_research=True)
    np.testing.assert_allclose(out.iloc[:2], [0., 0.]); assert out.iloc[2:].isna().all()


def test_cross_sectional_rank_preserves_row_axis_and_ties():
    values = panel([3., 1.], [3., 5.]).iloc[[2, 0, 3, 1]]
    out = compile_("REPRESENTATION_RANK", {"rank_axis": "cross_sectional", "tie_method": "average"}).execute(values, allow_research=True)
    assert out.index.equals(values.index)
    np.testing.assert_allclose(out.to_numpy(), [0., .5, 1., .5])


def test_missing_fill_is_bounded_causal_and_asset_isolated():
    values = panel([1., np.nan, np.nan, np.nan], [10., np.nan, 30., np.nan])
    p = compile_("MISSINGNESS_FRESHNESS", {"mode": "fill", "freshness_window": 2})
    out = p.execute(values, allow_research=True)
    np.testing.assert_allclose(out, [1, 10, 1, 10, 1, 30, np.nan, 30], equal_nan=True)
    changed = values.copy(); changed.loc[changed.date == changed.date.max(), "value"] = 9999.
    earlier = values.date < values.date.max()
    np.testing.assert_allclose(out[earlier], p.execute(changed, allow_research=True)[earlier], equal_nan=True)


def test_outlier_saturation_hinge_scale_and_zscore_execute_numerically():
    values = pd.DataFrame({"asset_id": list("ABCDE"), "date": [pd.Timestamp("2026-01-01")]*5,
                           "value": [0., 1., 2., 3., 100.]})
    winsor = compile_("ROBUST_OUTLIER", {"lower_quantile": .05, "upper_quantile": .95}).execute(values, allow_research=True)
    np.testing.assert_allclose(winsor, [.2, 1., 2., 3., 80.6])
    sat = compile_("TAIL_SATURATION", {"saturation_quantile": .95, "saturate": "top"}).execute(values, allow_research=True)
    np.testing.assert_allclose(sat, [0., 1., 2., 3., 80.6])
    hinge = compile_("TAIL_HINGE", {"hinge": "top", "hinge_value": 1.}).execute(values, allow_research=True)
    np.testing.assert_allclose(hinge, [0., 0., 1., 2., 99.])
    z = compile_("REPRESENTATION_ZSCORE", {"zscore_axis": "cross_sectional", "cap": 2.}).execute(values, allow_research=True)
    assert abs(z.mean()) < 1e-12; assert z.abs().max() <= 2.
    robust = compile_("ROBUST_SCALE", {"scale": "mad", "center": "median"}).execute(values, allow_research=True)
    np.testing.assert_allclose(robust, [-2., -1., 0., 1., 98.])


def test_robust_scale_tied_nonconstant_falls_back_and_inf_is_missing():
    values = pd.DataFrame({"asset_id": list("ABCDEF"), "date": [pd.Timestamp("2026-01-01")]*6,
                           "value": [1., 1., 1., 1., 100., np.inf]})
    out = compile_("ROBUST_SCALE", {"scale": "mad", "center": "median"}).execute(values, allow_research=True)
    assert out.iloc[:4].eq(0.).all()
    assert out.iloc[4] > 0
    assert np.isnan(out.iloc[5])


@pytest.mark.parametrize("func, kwargs", [
    ("rank_shape", {"center": np.nan, "power": 2.}),
    ("rank_shape", {"center": .5, "power": 0.}),
    ("capped_zscore", {"cap": 0.}),
    ("robust_scale", {"scale": "bogus", "center": "median"}),
    ("robust_scale", {"scale": "mad", "center": "bogus"}),
    ("rank_shape", {"center": True, "power": 2.}),
    ("rank_shape", {"center": .5, "power": True}),
    ("capped_zscore", {"cap": True}),
])
def test_fp_repair_primitives_validate_public_parameters(func, kwargs):
    from factor_preprocess.transforms import repair_shapes
    with pytest.raises(ValueError):
        getattr(repair_shapes, func)(panel([1.], [2.]), **kwargs)


def test_robust_std_avoids_overflow_for_large_finite_values():
    values = pd.DataFrame({"asset_id": list("ABC"), "date": [pd.Timestamp("2026-01-01")]*3,
                           "value": [-1e308, 0., 1e308]})
    out = compile_("ROBUST_SCALE", {"scale": "std", "center": "mean"}).execute(values, allow_research=True)
    np.testing.assert_allclose(out, [-1., 0., 1.])


@pytest.mark.parametrize("family, params, reason", [
    ("REPRESENTATION_RANK", {"rank_axis": "ts", "tie_method": "average"}, "time-series"),
    ("SIZE_NEUTRALIZATION", {"exposure_set": "size", "method": "ols"}, "exposure"),
])
def test_declared_but_unbound_families_are_ineligible_not_raw(family, params, reason):
    with pytest.raises(IneligibleValueRepair, match=reason): compile_(family, params)


def test_asymmetric_u_normalizes_each_side_of_frozen_center():
    values = pd.DataFrame({"asset_id": list("ABC"), "date": [pd.Timestamp("2026-01-01")]*3,
                           "value": [1., 2., 3.]})
    out = compile_("U_SHAPE_REPAIR", {"center": .25, "power": 1., "asymmetry": True}).execute(values, allow_research=True)
    np.testing.assert_allclose(out, [1., 1./3., 1.])
    edge = compile_("U_SHAPE_REPAIR", {"center": 0., "power": 1., "asymmetry": True}).execute(values, allow_research=True)
    np.testing.assert_allclose(edge, [0., .5, 1.])


def test_strict_parameter_and_execution_validation():
    with pytest.raises(ValueError, match="unexpected"): compile_("NO_OP_RAW", {"keep_raw": True, "x": 1})
    with pytest.raises(ValueError, match="finite"): compile_("U_SHAPE_REPAIR", {"center": np.nan, "power": 2., "asymmetry": False})
    with pytest.raises(ValueError, match="keep_raw"): compile_("NO_OP_RAW", {"keep_raw": False})
    plan = compile_("NO_OP_RAW", {"keep_raw": True})
    with pytest.raises(ValueError, match="research-only"): plan.execute(panel([1], [2]))
    with pytest.raises(ValueError, match="columns"): plan.execute(pd.DataFrame({"value": [1.]}), allow_research=True)
    duplicate = panel([1], [2]); duplicate = pd.concat([duplicate, duplicate.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"): plan.execute(duplicate, allow_research=True)
