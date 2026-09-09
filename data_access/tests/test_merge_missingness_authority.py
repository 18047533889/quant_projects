"""G09: missingness roles must not be promoted by filling or array mutation."""
import itertools

import numpy as np
import pytest

from data_access.core.missingness import MissingReason, MissingReasonPlane


@pytest.mark.parametrize("reason", [reason.value for reason in MissingReason])
@pytest.mark.parametrize("original,filled,usable", list(itertools.product((False, True), repeat=3)))
def test_reason_mask_truth_table(reason, original, filled, usable):
    fillable = reason in {"raw_missing", "source_outage"}
    valid = (
        (not filled or (original and fillable))
        and (not (original and usable) or filled)
        and (reason != "observed" or not original)
        and (reason == "observed" or original or not usable)
        and (reason not in {"not_listed", "mask_excluded", "budget_not_run", "label_not_yet_mature"} or not usable)
    )
    args = ([reason], [original], [filled], [usable], [0.0])
    if valid:
        plane = MissingReasonPlane(*args)
        assert plane.usable.tolist() == [usable]
        assert plane.reasons.tolist() == [reason]
    else:
        with pytest.raises(ValueError):
            MissingReasonPlane(*args)


@pytest.mark.parametrize("age", [-np.inf, np.inf, -1.0])
@pytest.mark.parametrize("filled", [False, True])
def test_age_rejects_negative_or_infinite_values(age, filled):
    with pytest.raises(ValueError, match="age"):
        MissingReasonPlane(["raw_missing"], [True], [filled], [False], [age])


@pytest.mark.parametrize("age", [0.0, 1.0, np.nan])
def test_unfilled_age_allows_nonnegative_finite_or_nan(age):
    plane = MissingReasonPlane(["raw_missing"], [True], [False], [False], [age])
    np.testing.assert_allclose(plane.age, [age], equal_nan=True)


def test_filled_age_must_be_finite():
    with pytest.raises(ValueError, match="finite"):
        MissingReasonPlane(["raw_missing"], [True], [True], [True], [np.nan])


@pytest.mark.parametrize("field", ["reasons", "original_missing", "filled", "usable", "age"])
def test_frozen_plane_cannot_reenable_writes(field):
    plane = MissingReasonPlane(["observed"], [False], [False], [True], [0.0])
    with pytest.raises(ValueError):
        getattr(plane, field).flags.writeable = True


def test_input_mutation_does_not_change_authoritative_snapshot():
    reasons = np.array(["raw_missing"])
    usable = np.array([True])
    plane = MissingReasonPlane(reasons, [True], [True], usable, [0.0])
    reasons[0] = "observed"
    usable[0] = False
    assert plane.reasons.tolist() == ["raw_missing"]
    assert plane.usable.tolist() == [True]


def test_coverage_does_not_count_nonrun_as_observed():
    plane = MissingReasonPlane(["observed", "budget_not_run"], [False, False], [False, False], [True, False], [0.0, np.nan])
    assert plane.coverage() == {"total": 2, "observed": 1, "usable": 1, "filled": 0, "original_missing": 0}


def test_arrow_parquet_roundtrip_preserves_explicit_reasons(tmp_path):
    import pyarrow.parquet as pq
    plane = MissingReasonPlane(
        np.array(["observed", "raw_missing", "label_not_yet_mature"]).reshape(3, 1),
        np.array([False, True, True]).reshape(3, 1),
        np.array([False, True, False]).reshape(3, 1),
        np.array([True, True, False]).reshape(3, 1),
        np.array([0.0, 2.0, np.nan]).reshape(3, 1),
    )
    path = tmp_path / "missingness.parquet"
    try:
        pq.write_table(plane.to_arrow(), path)
        restored = MissingReasonPlane.from_arrow(pq.read_table(path))
        for field in ("reasons", "original_missing", "filled", "usable", "age"):
            np.testing.assert_equal(getattr(restored, field), getattr(plane, field))
        assert restored.coverage() == plane.coverage()
    finally:
        path.unlink(missing_ok=True)


def test_arrow_tampered_masks_are_revalidated():
    import pyarrow as pa
    plane = MissingReasonPlane(["label_not_yet_mature"], [True], [False], [False], [0.0])
    table = plane.to_arrow()
    table = table.set_column(2, "filled", pa.array([True]))
    table = table.set_column(3, "usable", pa.array([True]))
    with pytest.raises(ValueError):
        MissingReasonPlane.from_arrow(table)


def test_arrow_requires_shape_metadata():
    plane = MissingReasonPlane(["observed"], [False], [False], [True], [0.0])
    with pytest.raises(ValueError, match="metadata"):
        MissingReasonPlane.from_arrow(plane.to_arrow().replace_schema_metadata(None))


def _feature_bundle(plane):
    from factor_preprocess.contracts.feature_bundle import AxisRef, FeatureBundle
    return FeatureBundle.from_primary_with_auxiliary(
        bundle_id="merge-g09",
        primary_values=np.ones((1, 1, 1)),
        feature_ids=("f",),
        time_axis=AxisRef("time", ["2026-09-09"], "str"),
        asset_axis=AxisRef("asset", ["A"], "str"),
        original_validity_mask=~np.asarray(plane.original_missing),
        missing_reason_plane=plane,
    )


def test_fp_consumer_keeps_canonical_reason_authority():
    plane = MissingReasonPlane([[["raw_missing"]]], [[[True]]], [[[True]]], [[[True]]], [[[1.0]]])
    restored = _feature_bundle(plane).get_missing_reason_plane()
    assert type(restored) is MissingReasonPlane
    assert restored.coverage() == plane.coverage()
    assert restored is not plane


def test_fp_consumer_rejects_forged_authority():
    from types import SimpleNamespace
    from factor_preprocess.errors import InvalidContractError
    forged = SimpleNamespace(
        reasons=np.array([[["label_not_yet_mature"]]]),
        original_missing=np.ones((1, 1, 1), dtype=bool),
        filled=np.ones((1, 1, 1), dtype=bool),
        usable=np.ones((1, 1, 1), dtype=bool),
        age=np.zeros((1, 1, 1)), coverage=lambda: {},
    )
    with pytest.raises(InvalidContractError, match="invalid authoritative"):
        _feature_bundle(forged)
