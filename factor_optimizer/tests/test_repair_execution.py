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


def test_sign_flip_routes_through_fe_neg_and_is_chunk_stable(monkeypatch):
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    operator = OperatorRegistry.get("neg", backend="polars", mode="any")
    assert operator is not None
    original = operator.calculate
    calls = []

    def tracked(frame):
        calls.append(frame.height)
        return original(frame)

    monkeypatch.setattr(operator, "calculate", tracked)
    values = panel([1.0, np.nan, -3.0], [2.0, 4.0, np.nan])
    plan = compile_("SIGN_ORIENTATION", {"direction": "flip"})
    expected = plan.execute(values, allow_research=True)
    assert calls == [len(values)]
    chunked = pd.concat([
        plan.execute(values.iloc[:3], allow_research=True),
        plan.execute(values.iloc[3:], allow_research=True),
    ])
    assert calls == [len(values), 3, 3]
    pd.testing.assert_series_equal(chunked, expected)


@pytest.mark.parametrize("backend", ["polars", "pandas_numpy"])
def test_sign_flip_matches_fe_backends_and_unary_minus_edge_cases(monkeypatch, backend):
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    operator = OperatorRegistry.get("neg", backend=backend, mode="any")
    assert operator is not None
    values = pd.DataFrame({
        "asset_id": ["A"] * 8,
        "date": pd.date_range("2026-01-01", periods=8),
        "value": pd.Series([0.0, -0.0, np.nan, np.inf, -np.inf, 1.25, -2.5, 3.0], dtype="float64"),
    })
    plan = compile_("SIGN_ORIENTATION", {"direction": "flip"})
    expected = -values["value"]
    if backend == "pandas_numpy":
        original_get = OperatorRegistry.get

        def pandas_only(name, *, backend, mode):
            if name == "neg" and backend == "polars":
                return None
            return original_get(name, backend=backend, mode=mode)

        monkeypatch.setattr(OperatorRegistry, "get", pandas_only)
    result = plan.execute(values, allow_research=True)
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result.to_numpy(), expected.to_numpy())
    assert np.signbit(result.iloc[0])  # +0.0 becomes -0.0
    assert not np.signbit(result.iloc[1])  # -0.0 becomes +0.0

    empty = values.iloc[:0]
    empty_result = plan.execute(empty, allow_research=True)
    assert empty_result.empty
    assert empty_result.dtype == values["value"].dtype


def test_sign_flip_fails_closed_when_fe_is_unavailable(monkeypatch):
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    original_get = OperatorRegistry.get

    def without_neg(name, *, backend, mode):
        return None if name == "neg" else original_get(name, backend=backend, mode=mode)

    monkeypatch.setattr(OperatorRegistry, "get", without_neg)
    values = panel([1.0], [2.0])
    with pytest.raises(RuntimeError, match="no executable backend"):
        compile_("SIGN_ORIENTATION", {"direction": "flip"}).execute(
            values, allow_research=True
        )


def test_sign_orientation_declares_fe_owner():
    from factor_optimizer.policy.repair_registry import (
        ExecutionDomain, RepairFamily, RepairFamilyRegistry,
    )

    declaration = RepairFamilyRegistry.default().get(RepairFamily.SIGN_ORIENTATION)
    assert declaration.owner is ExecutionDomain.FE


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


def test_cross_sectional_average_rank_routes_to_fe_and_matches_fp_edges(monkeypatch):
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_preprocess.transforms.repair_shapes import cross_sectional_rank

    operator = OperatorRegistry.get("rank", backend="pandas_numpy", mode="any")
    assert operator is not None
    original = operator.calculate
    calls = []

    def tracked(frame):
        calls.append(frame.shape)
        return original(frame)

    monkeypatch.setattr(operator, "calculate", tracked)
    date_a, date_b = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")
    values = pd.DataFrame({
        "asset_id": ["A", "B", "C", "A", "B", "C", "D", "D", "E"],
        "date": [date_a] * 3 + [date_b] * 4 + [pd.Timestamp("2026-01-03")] * 2,
        "value": [1., 1., 3., np.nan, np.inf, 4., -np.inf, np.nan, -np.inf],
    }, index=[8, 2, 7, 5, 0, 9, 3, 11, 1])
    plan = compile_("REPRESENTATION_RANK", {
        "rank_axis": "cross_sectional", "tie_method": "average",
    })
    result = plan.execute(values, allow_research=True)
    reference = cross_sectional_rank(values)
    pd.testing.assert_series_equal(result, reference)
    assert result.index.equals(values.index)
    assert result.dtype == np.dtype("float64")
    assert calls == [(3, 5)]  # Three dates by five assets in one FE batch.
    assert result.loc[9] == 0.5  # One finite value in this date's cross-section.
    assert result.loc[[5, 0, 3, 11, 1]].isna().all()

    empty = values.iloc[:0]
    empty_result = plan.execute(empty, allow_research=True)
    assert empty_result.empty and empty_result.index.equals(empty.index)


def test_cross_sectional_average_rank_is_stable_when_chunked_by_complete_dates():
    values = panel([1., 4., 2., np.nan], [1., 3., 3., np.inf])
    plan = compile_("REPRESENTATION_RANK", {
        "rank_axis": "cross_sectional", "tie_method": "average",
    })
    whole = plan.execute(values, allow_research=True)
    chunks = pd.concat([
        plan.execute(values.loc[values.date == date], allow_research=True)
        for date in values.date.drop_duplicates()
    ])
    pd.testing.assert_series_equal(chunks.sort_index(), whole.sort_index())


def test_cross_sectional_average_rank_fails_closed_without_fe_operator(monkeypatch):
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    original_get = OperatorRegistry.get

    def without_rank(name, *, backend, mode):
        return None if name == "rank" and backend == "pandas_numpy" else original_get(
            name, backend=backend, mode=mode
        )

    monkeypatch.setattr(OperatorRegistry, "get", without_rank)
    values = panel([1.0], [2.0])
    plan = compile_("REPRESENTATION_RANK", {
        "rank_axis": "cross_sectional", "tie_method": "average",
    })
    with pytest.raises(RuntimeError, match="no pandas_numpy backend"):
        plan.execute(values, allow_research=True)


def test_cross_sectional_average_rank_bounds_wide_intermediate(monkeypatch):
    from factor_optimizer.adapters import repair_execution
    from factor_preprocess.transforms.repair_shapes import cross_sectional_rank
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    monkeypatch.setattr(repair_execution, "_MAX_FE_RANK_CELLS", 4)
    operator = OperatorRegistry.get("rank", backend="pandas_numpy", mode="any")
    original = operator.calculate
    shapes = []

    def tracked(wide):
        shapes.append(wide.shape)
        return original(wide)

    monkeypatch.setattr(operator, "calculate", tracked)
    values = panel([1., 4., 2., np.nan, 3.], [1., 3., 3., np.inf, 2.])
    values = values.iloc[[8, 1, 4, 7, 2, 5, 0, 9, 3, 6]]
    plan = compile_("REPRESENTATION_RANK", {
        "rank_axis": "cross_sectional", "tie_method": "average",
    })
    pd.testing.assert_series_equal(
        plan.execute(values, allow_research=True), cross_sectional_rank(values)
    )
    assert shapes == [(2, 2), (2, 2), (1, 2)]


def test_rank_tie_variants_have_distinct_fe_and_fp_identities():
    from factor_engine.cleaned_operators.math_certificate import _rank_rowwise_np

    date = pd.Timestamp("2026-01-01")
    values = pd.DataFrame({"asset_id": ["A", "B", "C"],
                           "date": [date] * 3, "value": [1., 1., 2.]})
    average = compile_("REPRESENTATION_RANK", {
        "rank_axis": "cross_sectional", "tie_method": "average"})
    minimum = compile_("REPRESENTATION_RANK", {
        "rank_axis": "cross_sectional", "tie_method": "min"})
    assert average.transform == "cs_rank"
    assert minimum.transform == "fp_cs_rank_min"
    assert average.identity != minimum.identity
    expected_fe = _rank_rowwise_np(np.array([[1., 1., 2.]]))[0]
    np.testing.assert_allclose(average.execute(values, allow_research=True), expected_fe)
    np.testing.assert_allclose(minimum.execute(values, allow_research=True), [0., 0., 1.])
    # Existing frozen v3 plans remain readable and preserve their old semantics.
    legacy = type(average)(minimum.family, "cs_rank", (("method", "min"),),
                           minimum.training_context_ref, minimum.natural_time_scale)
    np.testing.assert_allclose(legacy.execute(values, allow_research=True), [0., 0., 1.])


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
