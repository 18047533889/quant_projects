"""
R61-FI-022 (plan §13.6): train-vs-validation generalization evidence.

Pins the typed comparison contract in
:mod:`quant_evaluator.metrics.generalization_evidence`:

1. TrainVsValidationArtifact is a FROZEN dataclass carrying the predictive
   dimensions, robust retention (with explicit reasons), the four deltas
   (rankic / icir / sharpe / shape), parameter generalization, and the
   versioned retention policy anchors.
2. ROBUST retention: when the train metric is near zero or changes sign,
   the ratio is NOT blindly divided (plan §13.6: "do not blindly divide by
   tiny Train IC") — the artifact combines absolute delta + relative
   retention with a stable denominator + confidence interval + sign
   consistency, and reports `NaN` retention with an explicit reason when
   the denominator is unstable.  Never a fabricated ratio.
3. NO Test/sealed exposure: the artifact has no Test fields, and the
   sealed-split boundary remains the existing
   ``contracts/sealed_split.py`` gate (which rejects overlapping
   evaluations).
4. The S+..D anchors live ONLY in the versioned ``RetentionPolicy``
   constants (plan §13.6) — never scattered hard-coded thresholds.

Run from the repo root:

    cd /home/sunhaiwei/quant_projects && .venv/bin/python -m pytest -q \
        quant_evaluator/tests/test_generalization_evidence.py --timeout=300
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from quant_evaluator.contracts.sealed_split import SealedSplitRef
from quant_evaluator.metrics.generalization_evidence import (
    RetentionGrade,
    RetentionPolicy,
    TrainVsValidationArtifact,
    build_train_validation_artifact,
    compute_generalization_deltas,
    compute_validation_retention,
    compute_validation_retention_array,
)
from quant_evaluator.registry.metrics import get_metric

# ---------------------------------------------------------------------------
# 1. Frozen artifact with the plan §13.6 fields.
# ---------------------------------------------------------------------------


def test_artifact_is_frozen_dataclass():
    art = build_train_validation_artifact(
        train_dim=np.array([0.3, 0.25]),
        validation_dim=np.array([0.27, 0.2]),
    )
    assert dataclasses.is_dataclass(art)
    with pytest.raises(Exception):
        art.validation_retention = (1.0,)  # type: ignore[misc]
    assert art.validation_retention == (0.9000000000000001, 0.8)


def test_artifact_carries_all_plan_fields():
    art = build_train_validation_artifact(
        train_dim=np.array([0.3]),
        validation_dim=np.array([0.27]),
        train_rankic=np.array([0.3]),
        validation_rankic=np.array([0.27]),
        train_icir=np.array([1.0]),
        validation_icir=np.array([0.8]),
        train_sharpe=np.array([1.5]),
        validation_sharpe=np.array([1.2]),
        train_shape=np.array([0.5]),
        validation_shape=np.array([0.4]),
    )
    assert list(art.train_predictive_dimension) == [0.3]
    assert list(art.validation_predictive_dimension) == [0.27]
    assert art.validation_retention[0] == pytest.approx(0.9)
    assert art.train_validation_rankic_delta[0] == pytest.approx(-0.03)
    assert art.train_validation_icir_delta[0] == pytest.approx(-0.2)
    assert art.train_validation_sharpe_delta[0] == pytest.approx(-0.3)
    assert art.train_validation_shape_delta[0] == pytest.approx(-0.1)
    assert art.parameter_generalization is not None
    assert art.policy_id == "QE_RETENTION_POLICY"
    assert art.policy_version == "1.0.0"


def test_artifact_serialization_roundtrip():
    art = build_train_validation_artifact(
        train_dim=np.array([0.3, 0.02]),
        validation_dim=np.array([0.27, 0.05]),
    )
    payload = art.to_dict()
    rebuilt = TrainVsValidationArtifact.from_dict(payload)
    assert rebuilt == art
    assert list(rebuilt.validation_retention) == list(art.validation_retention)


# ---------------------------------------------------------------------------
# 2. Robust retention: never a blind division by tiny / sign-flipped train.
# ---------------------------------------------------------------------------


def test_train_near_zero_retention_is_nan_not_blind_ratio():
    """train==0 must NOT produce inf/NaN via division; retention is NaN
    with an explicit reason and the delta stays available."""
    retentions, reasons = compute_validation_retention_array(
        np.array([0.0, 0.01, 0.5]),
        np.array([0.1, 0.05, 0.25]),
    )
    assert retentions[0] is None
    assert reasons[0] == "train_near_zero"
    # 0.01 < min_abs_train(0.05): still guarded.
    assert retentions[1] is None
    assert reasons[1] == "train_near_zero"
    # Stable denominator -> ratio computed.
    assert retentions[2] == pytest.approx(0.5)


def test_sign_flip_guard_blocks_blind_division():
    """Opposite-sign validation with train above min_abs_train but under the
    relative sign-flip guard is guarded, not divided."""
    retentions, reasons = compute_validation_retention_array(
        np.array([0.06]),  # above min_abs_train (0.05)
        np.array([-0.20]),  # opposite sign, relatively large
    )
    assert retentions[0] is None
    assert reasons[0] == "sign_flip_guard"
    # Below min_abs_train the guard fires as the tiny-denominator reason
    # (checked first — the denominator is unstable either way).
    retentions0, reasons0 = compute_validation_retention_array(
        np.array([0.04]), np.array([-0.08])
    )
    assert retentions0[0] is None
    assert reasons0[0] == "train_near_zero"
    # Large train with a sign flip is allowed to produce its (negative)
    # retention — the denominator is stable, no blind division.
    retentions2, reasons2 = compute_validation_retention_array(
        np.array([0.5]), np.array([-0.2])
    )
    assert retentions2[0] == pytest.approx(-0.4)
    assert reasons2[0] == "computed"


def test_insufficient_data_is_explicit():
    retentions, reasons = compute_validation_retention_array(
        np.array([np.nan, 0.3]), np.array([0.1, 0.15])
    )
    assert retentions[0] is None
    assert reasons[0] == "insufficient_data"


def test_validation_retention_mean_skips_unstable_factors():
    mean_ret = compute_validation_retention(
        np.array([0.0, 0.3]), np.array([0.1, 0.24])
    )
    assert mean_ret == pytest.approx(0.8)  # only the stable factor counts


def test_validation_retention_all_unstable_is_nan_not_zero():
    mean_ret = compute_validation_retention(
        np.array([0.0, -0.02]), np.array([0.1, 0.05])
    )
    assert np.isnan(mean_ret)


def test_deltas_are_absolute_never_ratio():
    deltas = compute_generalization_deltas(
        np.array([0.0, 0.3]), np.array([0.05, 0.24])
    )
    assert list(deltas[0]) == [pytest.approx(0.05), pytest.approx(-0.06)]


def test_deltas_absent_inputs_produce_none():
    # Missing side -> each delta element is None (missing evidence, never 0).
    deltas = compute_generalization_deltas(None, None)
    assert deltas == ((), (), (), ())
    artifact = build_train_validation_artifact(
        train_dim=np.array([0.3]), validation_dim=np.array([0.27])
    )
    assert artifact.train_validation_icir_delta == (None,)
    assert artifact.train_validation_sharpe_delta == (None,)
    assert artifact.train_validation_shape_delta == (None,)


def test_confidence_band_on_artifact():
    art = build_train_validation_artifact(
        train_dim=np.array([0.3, 0.28, 0.32]),
        validation_dim=np.array([0.27, 0.25, 0.26]),
    )
    assert art.confidence_interval_95 is not None
    lo, hi = art.confidence_interval_95
    assert lo is not None and hi is not None
    assert lo < hi


def test_reasons_all_plan_states():
    art = build_train_validation_artifact(
        train_dim=np.array([0.3, 0.0, -0.02, np.nan]),
        validation_dim=np.array([0.27, 0.12, 0.03, 0.1]),
    )
    assert art.retention_reason == (
        "computed",
        "train_near_zero",
        "train_near_zero",
        "insufficient_data",
    )


# ---------------------------------------------------------------------------
# 3. NO Test exposure (plan §13.6: never compute this via sealed Test).
# ---------------------------------------------------------------------------


def test_artifact_never_has_test_fields():
    """The artifact dataclass has NO Test/sealed attribute and no reference
    to one — exposing the sealed Test via this evidence type is impossible."""
    fields = {f.name for f in dataclasses.fields(TrainVsValidationArtifact)}
    assert "test" not in fields
    assert "sealed" not in fields
    assert "test_split" not in fields
    assert "test_predictive_dimension" not in fields
    assert "Test" not in fields
    # The constructor signature is the same field set.
    params = set(inspect.signature(TrainVsValidationArtifact).parameters)
    assert not any("test" in name or "sealed" in name for name in params)


def test_sealed_split_gate_still_guards_evaluation():
    """The QE-side sealed boundary remains contracts/sealed_split.py: a
    split_ref whose window overlaps the factor/label information boundary is
    rejected — generalization evidence never bypasses it."""
    from quant_evaluator.contracts.errors import SealedSplitOverlapError
    from quant_evaluator.contracts.sealed_split import check_sealed_split_overlap

    sealed = SealedSplitRef(
        split_id="sealed_test", start_time=5, end_time=20
    )
    # A label/decision time inside the sealed window (>= start) is an
    # overlap and must be rejected fail-closed.
    with pytest.raises(SealedSplitOverlapError):
        check_sealed_split_overlap(
            sealed,
            decision_times=(0, 2, 4, 6),
            label_start_times=(2, 4, 6),
            label_end_times=(3, 5, 8),
        )
    # A label whose window ENDS after the sealed end leaks future info.
    with pytest.raises(SealedSplitOverlapError):
        check_sealed_split_overlap(
            sealed,
            decision_times=(0, 2),
            label_start_times=(0, 2),
            label_end_times=(1, 25),
        )
    # None means no check (backward compatible).
    check_sealed_split_overlap(None, decision_times=(0, 1), label_start_times=(), label_end_times=())


# ---------------------------------------------------------------------------
# 4. Anchors versioned (plan §13.6: S+..D as policy constants only).
# ---------------------------------------------------------------------------


def test_retention_policy_anchor_constants():
    policy = RetentionPolicy()
    expected = [
        (0.90, "S+"),
        (0.80, "S"),
        (0.70, "A+"),
        (0.60, "A"),
        (0.50, "B+"),
        (0.40, "B"),
        (0.25, "C"),
    ]
    assert policy.grade_anchors == tuple(expected)
    assert policy.policy_version == "1.0.0"


def test_grade_from_anchor_thresholds():
    assert RetentionGrade.from_value(0.95) == RetentionGrade.S_PLUS
    assert RetentionGrade.from_value(0.90) == RetentionGrade.S_PLUS
    assert RetentionGrade.from_value(0.85) == RetentionGrade.S
    assert RetentionGrade.from_value(0.75) == RetentionGrade.A_PLUS
    assert RetentionGrade.from_value(0.65) == RetentionGrade.A
    assert RetentionGrade.from_value(0.55) == RetentionGrade.B_PLUS
    assert RetentionGrade.from_value(0.45) == RetentionGrade.B
    assert RetentionGrade.from_value(0.30) == RetentionGrade.C
    assert RetentionGrade.from_value(0.10) == RetentionGrade.D
    assert RetentionGrade.from_value(None) == RetentionGrade.D
    assert RetentionGrade.from_value(float("nan")) == RetentionGrade.D


def test_policy_validation_rejects_bad_anchors():
    with pytest.raises(ValueError):
        RetentionPolicy(grade_anchors=((0.8, "S"), (0.9, "S+")))  # ascending
    with pytest.raises(ValueError):
        RetentionPolicy(grade_anchors=((0.9, "ZZZ"),))  # unknown grade
    with pytest.raises(ValueError):
        RetentionPolicy(min_abs_train=-1.0)


def test_artifact_carries_policy_version():
    art = build_train_validation_artifact(
        train_dim=np.array([0.3]), validation_dim=np.array([0.27])
    )
    assert art.policy_id == "QE_RETENTION_POLICY"
    assert art.policy_version == "1.0.0"
    assert art.retention_grade[0] == "S+"


def test_no_hardcoded_anchors_outside_policy():
    """Source scan: no retention threshold literal exists in the artifact
    or builder code outside the versioned RetentionPolicy."""
    from quant_evaluator.metrics import generalization_evidence as module

    src = inspect.getsource(module)
    # The only 0.90/0.80-style literals may appear inside RetentionPolicy's
    # own declaration; the builder functions must not define thresholds.
    builder_src = inspect.getsource(build_train_validation_artifact)
    for literal in ("0.90", "0.80", "0.70", "0.25"):
        assert literal not in builder_src, literal


# ---------------------------------------------------------------------------
# 5. Registry wiring of the generalization ids.
# ---------------------------------------------------------------------------

GENERALIZATION_IDS = (
    "train_predictive_dimension",
    "validation_predictive_dimension",
    "validation_retention",
    "train_validation_rankic_delta",
    "train_validation_icir_delta",
    "train_validation_sharpe_delta",
    "train_validation_shape_delta",
    "parameter_generalization",
)


def test_generalization_ids_registered_with_bound_compute_fn():
    from quant_evaluator.registry.metrics import list_metrics

    registered = set(list_metrics())
    for metric_id in GENERALIZATION_IDS:
        assert metric_id in registered, metric_id
        assert get_metric(metric_id).compute_fn is not None, metric_id


def test_expensive_statistical_profile_untouched():
    """EXPENSIVE_STATISTICAL_CN_1D's 5 ids were not modified or extended."""
    from quant_evaluator.registry.presets import EXPENSIVE_STATISTICAL_CN_1D

    assert EXPENSIVE_STATISTICAL_CN_1D.metric_ids == (
        "hac_tstat",
        "hac_pvalue",
        "block_bootstrap_ci",
        "subsample_stability",
        "ic_autocorr_lag1",
    )


def test_validation_retention_direction_is_higher_is_better():
    spec = get_metric("validation_retention")
    assert spec.direction == "higher_is_better"
    assert get_metric("parameter_generalization").direction == "higher_is_better"