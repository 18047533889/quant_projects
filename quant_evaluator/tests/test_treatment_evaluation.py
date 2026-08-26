"""Tests for the factor auto-treatment evaluation contracts (delta-vs-raw).

Covers :class:`DeltaMetrics`, :class:`TreatmentEvaluationArtifact` and the
:mod:`quant_evaluator.metrics.delta` helpers: delta computation, fail-closed
missing-raw handling, the "RAW is candidate 0" invariant, derived content
hashing, and numeric validation.
"""

from __future__ import annotations

import math

import pytest

from quant_evaluator.contracts.treatment_evaluation import (
    DeltaMetrics,
    TreatmentEvaluationArtifact,
)
from quant_evaluator.metrics.delta import (
    assert_raw_candidate_first,
    compute_delta_vs_raw,
)


def _sample_metrics() -> dict:
    return {
        "rank_ic": 0.03,
        "icir": 0.9,
        "turnover": 0.2,
        "cost_adjusted_alpha": 0.005,
        "worst_slice": -0.02,
        "exposure": 0.6,
        "coverage": 0.95,
    }


def _raw_metrics() -> dict:
    return {
        "rank_ic": 0.02,
        "icir": 0.7,
        "turnover": 0.15,
        "cost_adjusted_alpha": 0.003,
        "worst_slice": -0.03,
        "exposure": 0.5,
        "coverage": 0.9,
    }


def _make_artifact(**overrides) -> TreatmentEvaluationArtifact:
    base = {
        "factor_id": "factor_a",
        "treatment_id": "winsorize_3sigma",
        "raw_baseline_evidence_ref": "evidence:factor_a:NO_OP",
        "treatment_evidence_ref": "evidence:factor_a:winsorize_3sigma",
        "absolute_metrics": _sample_metrics(),
        "delta_vs_raw": DeltaMetrics(delta_rank_ic=0.01),
        "snapshot_ref": "snap:2024-01-01",
        "universe_ref": "uni:top500",
        "split_ref": "split:2024-07-01",
        "created_at": "2026-08-25T00:00:00Z",
    }
    base.update(overrides)
    return TreatmentEvaluationArtifact(**base)


# ---------------------------------------------------------------------------
# compute_delta_vs_raw
# ---------------------------------------------------------------------------


def test_delta_vs_raw():
    """Given treatment and raw metric dicts, deltas compute as treatment - raw."""
    delta = compute_delta_vs_raw(_sample_metrics(), _raw_metrics())
    assert delta.delta_rank_ic == pytest.approx(0.03 - 0.02)
    assert delta.delta_icir == pytest.approx(0.9 - 0.7)
    assert delta.delta_turnover == pytest.approx(0.2 - 0.15)
    assert delta.delta_cost_adjusted_alpha == pytest.approx(0.005 - 0.003)
    assert delta.delta_worst_slice == pytest.approx(-0.02 - (-0.03))
    assert delta.delta_exposure == pytest.approx(0.6 - 0.5)
    assert delta.delta_coverage == pytest.approx(0.95 - 0.9)


def test_missing_raw_metric_yields_none():
    """A metric present in treatment but absent in raw -> delta None (NOT 0)."""
    treatment = dict(_sample_metrics())
    raw = dict(_raw_metrics())
    del raw["rank_ic"]  # raw baseline lacks rank_ic
    delta = compute_delta_vs_raw(treatment, raw)
    assert delta.delta_rank_ic is None
    # All other metrics still compute normally.
    assert delta.delta_icir == pytest.approx(0.9 - 0.7)


def test_missing_raw_metric_never_fabricated_zero():
    """A missing raw metric must not silently become delta 0.0."""
    treatment = dict(_sample_metrics())
    raw = dict(_raw_metrics())
    del raw["exposure"]
    delta = compute_delta_vs_raw(treatment, raw)
    assert delta.delta_exposure is None
    assert delta.delta_exposure != 0.0


# ---------------------------------------------------------------------------
# assert_raw_candidate_first
# ---------------------------------------------------------------------------


def test_raw_candidate_must_be_first():
    """A candidate list whose first element is not RAW raises ValueError."""
    artifacts = [
        _make_artifact(treatment_id="winsorize_3sigma"),
        _make_artifact(treatment_id="NO_OP"),
    ]
    with pytest.raises(ValueError):
        assert_raw_candidate_first(artifacts)


def test_raw_candidate_first_ok():
    """A candidate list whose first element IS RAW/NO_OP passes."""
    artifacts = [
        _make_artifact(treatment_id="NO_OP"),
        _make_artifact(treatment_id="winsorize_3sigma"),
    ]
    # Should not raise.
    assert_raw_candidate_first(artifacts)


def test_raw_candidate_first_empty_raises():
    with pytest.raises(ValueError):
        assert_raw_candidate_first([])


# ---------------------------------------------------------------------------
# TreatmentEvaluationArtifact content hash
# ---------------------------------------------------------------------------


def test_artifact_content_hash():
    """Identical fields -> identical hash; a field change -> different hash."""
    a = _make_artifact()
    b = _make_artifact()
    assert a.content_hash == b.content_hash
    assert a.content_hash  # derived, non-empty

    c = _make_artifact(treatment_id="winsorize_5sigma")
    assert c.content_hash != a.content_hash

    d = _make_artifact(snapshot_ref="snap:2024-06-01")
    assert d.content_hash != a.content_hash


def test_artifact_wrong_content_hash_raises():
    """A caller-supplied content_hash that disagrees with the derived value raises."""
    with pytest.raises(ValueError):
        _make_artifact(content_hash="deadbeefdeadbeef")


def test_artifact_correct_content_hash_ok():
    """A caller-supplied content_hash that MATCHES the derived value is fine."""
    a = _make_artifact()
    # Passing the exact derived hash should not raise.
    b = _make_artifact(content_hash=a.content_hash)
    assert b.content_hash == a.content_hash


# ---------------------------------------------------------------------------
# DeltaMetrics validation
# ---------------------------------------------------------------------------


def test_delta_validation_rejects_bool():
    """A bool delta field is rejected (would coerce to 1.0/0.0)."""
    with pytest.raises(ValueError):
        DeltaMetrics(delta_rank_ic=True)


def test_delta_validation_rejects_nan():
    """A NaN delta field is rejected fail-closed."""
    with pytest.raises(ValueError):
        DeltaMetrics(delta_rank_ic=float("nan"))


def test_delta_validation_rejects_inf():
    """An infinite delta field is rejected fail-closed."""
    with pytest.raises(ValueError):
        DeltaMetrics(delta_icir=float("inf"))


def test_delta_all_none_ok():
    """All-None deltas are valid (raw baseline entirely unavailable)."""
    d = DeltaMetrics()
    assert d.delta_rank_ic is None
    assert d.delta_coverage is None
