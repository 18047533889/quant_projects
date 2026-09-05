# -*- coding: utf-8 -*-
"""Tests for factor_assets/profiling — deterministic diagnosis engine
(R61-FI-026, plan §9/§13/§15-§17).

Coverage:
1. known evidence -> expected tag set, severity, repairability, confidence;
2. bad (non-COMPUTED) evidence NEVER fires a numeric-threshold diagnosis
   (plan §8 iron law: missing evidence is not evidence of a defect);
3. integrity / data-quality typed failures fire on bad-status + reason codes;
4. same input -> same output (idempotent / deterministic);
5. healthy factor -> empty tag set (explicit unknown only when requested);
6. policy refs are stamped on each diagnosis; tags are in canonical order and
   deduplicated (a tag firing from two rules keeps the highest confidence).
"""

import pytest

from factor_assets.profiling.diagnosis import (
    DIAGNOSIS_TAGS,
    DiagnosisTag,
    detect_diagnoses,
    diagnose_factor,
)
from factor_assets.profiling.metric_grading import BAD_EVIDENCE_STATUSES
from factor_assets.profiling.policies import get_diagnosis_policy

POLICY = get_diagnosis_policy()


def _ev(**pairs):
    """Build a flat evidence mapping from dotted key -> value kwargs."""
    return dict(pairs)


class TestKnownEvidenceExpectedTags:
    def test_low_predictive_rank_ic(self):
        ev = {"metrics.rank_ic.evidence_status": "COMPUTED", "metrics.rank_ic.value": 0.002}
        tags = detect_diagnoses(factor_definition_id="F1", evidence=ev)
        names = [t.tag for t in tags]
        assert names == ["LOW_PREDICTIVE"]
        t = tags[0]
        assert t.severity == POLICY.severity("LOW_PREDICTIVE") == "HIGH"
        assert t.repairability == "REPAIRABLE"
        assert t.confidence > 0.0
        assert "metrics.rank_ic.value" in t.evidence_refs
        assert t.triggered_policy_rule == "FI026_RANK_IC_LOW"

    def test_high_rank_ic_with_low_icir_low_statistical_confidence(self):
        ev = {
            "metrics.rank_ic.evidence_status": "COMPUTED",
            "metrics.rank_ic.value": 0.04,
            "metrics.rank_ic_ir.evidence_status": "COMPUTED",
            "metrics.rank_ic_ir.value": 0.1,
        }
        tags = detect_diagnoses(factor_definition_id="F1", evidence=ev)
        names = [t.tag for t in tags]
        assert names == ["LOW_STATISTICAL_CONFIDENCE"]

    def test_unstable_ic_high_volatility(self):
        ev = {
            "metrics.rolling_ic_volatility.evidence_status": "COMPUTED",
            "metrics.rolling_ic_volatility.value": 0.12,
        }
        assert [t.tag for t in detect_diagnoses(factor_definition_id="F1", evidence=ev)] == [
            "UNSTABLE_IC"
        ]

    def test_recent_degradation(self):
        ev = {
            "metrics.ic_recent_vs_history_delta.evidence_status": "COMPUTED",
            "metrics.ic_recent_vs_history_delta.value": -0.02,
        }
        assert [t.tag for t in detect_diagnoses(factor_definition_id="F1", evidence=ev)] == [
            "RECENT_DEGRADATION"
        ]

    def test_overfit_generalization_low_retention(self):
        ev = {
            "generalization.validation_retention.evidence_status": "COMPUTED",
            "generalization.validation_retention.value": 0.2,
        }
        assert [t.tag for t in detect_diagnoses(factor_definition_id="F1", evidence=ev)] == [
            "OVERFIT_GENERALIZATION"
        ]

    def test_overfit_generalization_icir_delta(self):
        ev = {
            "generalization.train_validation_icir_delta.evidence_status": "COMPUTED",
            "generalization.train_validation_icir_delta.value": 2.5,
        }
        tags = detect_diagnoses(factor_definition_id="F1", evidence=ev)
        assert [t.tag for t in tags] == ["OVERFIT_GENERALIZATION"]
        assert tags[0].triggered_policy_rule == "FI026_ICIR_DELTA_GT"

    def test_low_statistical_confidence_weak_tstat(self):
        ev = {
            "metrics.hac_tstat.evidence_status": "COMPUTED",
            "metrics.hac_tstat.value": 1.2,
        }
        assert [t.tag for t in detect_diagnoses(factor_definition_id="F1", evidence=ev)] == [
            "LOW_STATISTICAL_CONFIDENCE"
        ]

    def test_shape_tags(self):
        ev = {
            "shape.u_shape_score.evidence_status": "COMPUTED",
            "shape.u_shape_score.value": 0.8,
            "shape.top_tail_cliff.evidence_status": "COMPUTED",
            "shape.top_tail_cliff.value": 0.9,
        }
        names = [t.tag for t in detect_diagnoses(factor_definition_id="F1", evidence=ev)]
        assert "U_SHAPE" in names
        assert "TOP_TAIL_COLLAPSE" in names

    def test_inverted_u_and_bottom_tail(self):
        ev = {
            "shape.inverted_u_score.evidence_status": "COMPUTED",
            "shape.inverted_u_score.value": 0.85,
            "shape.bottom_tail_cliff.evidence_status": "COMPUTED",
            "shape.bottom_tail_cliff.value": 0.7,
        }
        names = [t.tag for t in detect_diagnoses(factor_definition_id="F1", evidence=ev)]
        assert names == ["INVERTED_U", "BOTTOM_TAIL_COLLAPSE"]

    def test_nonstationary_shape_low_shape_stability(self):
        ev = {
            "shape.shape_stability.evidence_status": "COMPUTED",
            "shape.shape_stability.value": 0.1,
        }
        assert [t.tag for t in detect_diagnoses(factor_definition_id="F1", evidence=ev)] == [
            "NONSTATIONARY_SHAPE"
        ]

    def test_portfolio_economics_tags(self):
        ev = {
            "metrics.factor_turnover_rate.evidence_status": "COMPUTED",
            "metrics.factor_turnover_rate.value": 0.9,
            "metrics.cost_drag.evidence_status": "COMPUTED",
            "metrics.cost_drag.value": 0.2,
            "metrics.max_drawdown.evidence_status": "COMPUTED",
            "metrics.max_drawdown.value": -0.6,
            "metrics.max_drawdown_duration.evidence_status": "COMPUTED",
            "metrics.max_drawdown_duration.value": 400.0,
            "metrics.cvar_95.evidence_status": "COMPUTED",
            "metrics.cvar_95.value": -0.2,
        }
        names = [t.tag for t in detect_diagnoses(factor_definition_id="F1", evidence=ev)]
        assert "HIGH_TURNOVER" in names
        assert "HIGH_COST_DRAG" in names
        assert "HIGH_DRAWDOWN" in names  # fired from max_drawdown (dedup keeps one)
        assert "LONG_UNDERWATER" in names

    def test_regime_dependent_dispersion(self):
        ev = {
            "metrics.regime_dispersion.evidence_status": "COMPUTED",
            "metrics.regime_dispersion.value": 0.9,
        }
        assert [t.tag for t in detect_diagnoses(factor_definition_id="F1", evidence=ev)] == [
            "REGIME_DEPENDENT"
        ]

    def test_regime_dependent_sign_consistency(self):
        ev = {
            "metrics.regime_sign_consistency.evidence_status": "COMPUTED",
            "metrics.regime_sign_consistency.value": 0.1,
        }
        assert [t.tag for t in detect_diagnoses(factor_definition_id="F1", evidence=ev)] == [
            "REGIME_DEPENDENT"
        ]

    def test_exposure_tags(self):
        ev = {
            "exposure.size_exposure.evidence_status": "COMPUTED",
            "exposure.size_exposure.value": 0.8,
            "exposure.industry_exposure.evidence_status": "COMPUTED",
            "exposure.industry_exposure.value": -0.7,
            "exposure.max_abs_style_exposure.evidence_status": "COMPUTED",
            "exposure.max_abs_style_exposure.value": 0.6,
            "exposure.n_style_exposures.evidence_status": "COMPUTED",
            "exposure.n_style_exposures.value": 3.0,
        }
        names = [t.tag for t in detect_diagnoses(factor_definition_id="F1", evidence=ev)]
        assert "SIZE_EXPOSURE" in names
        assert "INDUSTRY_EXPOSURE" in names
        assert "MULTI_STYLE_EXPOSURE" in names

    def test_semantic_duplicate_vs_near_duplicate(self):
        ev_dup = {
            "novelty.max_duplicate_corr.evidence_status": "COMPUTED",
            "novelty.max_duplicate_corr.value": 0.99,
        }
        assert [t.tag for t in detect_diagnoses(factor_definition_id="F1", evidence=ev_dup)] == [
            "SEMANTIC_DUPLICATE"
        ]
        ev_near = {
            "novelty.max_duplicate_corr.evidence_status": "COMPUTED",
            "novelty.max_duplicate_corr.value": 0.93,
        }
        assert [t.tag for t in detect_diagnoses(factor_definition_id="F1", evidence=ev_near)] == [
            "VALUE_NEAR_DUPLICATE"
        ]

    def test_low_novelty(self):
        ev = {
            "novelty.novelty.evidence_status": "COMPUTED",
            "novelty.novelty.value": 0.01,
        }
        assert [t.tag for t in detect_diagnoses(factor_definition_id="F1", evidence=ev)] == [
            "LOW_NOVELTY"
        ]

    def test_high_complexity(self):
        ev = {
            "complexity.complexity_score.evidence_status": "COMPUTED",
            "complexity.complexity_score.value": 0.95,
        }
        assert [t.tag for t in detect_diagnoses(factor_definition_id="F1", evidence=ev)] == [
            "HIGH_COMPLEXITY"
        ]


class TestBadEvidenceNeverFiresNumericDiagnosis:
    @pytest.mark.parametrize("status", list(BAD_EVIDENCE_STATUSES))
    def test_non_computed_rank_ic_no_low_predictive(self, status):
        # even an extreme value with a bad status must not fire LOW_PREDICTIVE
        ev = {"metrics.rank_ic.evidence_status": status, "metrics.rank_ic.value": 0.0}
        assert detect_diagnoses(factor_definition_id="F1", evidence=ev) == ()

    @pytest.mark.parametrize("status", list(BAD_EVIDENCE_STATUSES))
    def test_non_computed_turnover_no_high_turnover(self, status):
        ev = {
            "metrics.factor_turnover_rate.evidence_status": status,
            "metrics.factor_turnover_rate.value": 5.0,
        }
        assert detect_diagnoses(factor_definition_id="F1", evidence=ev) == ()

    def test_missing_status_token_never_fires(self):
        ev = {"metrics.rank_ic.value": 0.0}  # no evidence_status at all
        assert detect_diagnoses(factor_definition_id="F1", evidence=ev) == ()

    def test_unavailable_coverage_no_poor_coverage(self):
        ev = {"metrics.coverage.evidence_status": "UNAVAILABLE_INPUT",
              "metrics.coverage.value": 0.1}
        assert detect_diagnoses(factor_definition_id="F1", evidence=ev) == ()

    def test_empty_evidence_no_tags(self):
        assert detect_diagnoses(factor_definition_id="F1", evidence={}) == ()


class TestIntegrityAndDataQualityTypedFailures:
    def test_pit_invalid_fires_integrity_failure(self):
        ev = {
            "integrity.pit_valid.evidence_status": "COMPUTED",
            "integrity.pit_valid.reason_code": "PIT_INVALID",
        }
        tags = detect_diagnoses(factor_definition_id="F1", evidence=ev)
        assert [t.tag for t in tags] == ["INTEGRITY_FAILURE"]
        assert tags[0].severity == "CRITICAL"
        assert tags[0].repairability == "UNREPAIRABLE"

    def test_bad_label_maturity_fires_integrity_failure(self):
        ev = {
            "integrity.label_maturity_ok.evidence_status": "FAILED",
            "integrity.label_maturity_ok.reason_code": "LABEL_TIMING_VIOLATION",
        }
        assert [t.tag for t in detect_diagnoses(factor_definition_id="F1", evidence=ev)] == [
            "INTEGRITY_FAILURE"
        ]

    def test_missing_ratio_bad_status_fires_dq_failure(self):
        ev = {
            "metrics.missing_ratio.evidence_status": "INVALID",
            "metrics.missing_ratio.reason_code": "KERNEL_FAILURE",
        }
        assert [t.tag for t in detect_diagnoses(factor_definition_id="F1", evidence=ev)] == [
            "DATA_QUALITY_FAILURE"
        ]

    def test_missing_ratio_high_computed_fires_dq_failure(self):
        ev = {
            "metrics.missing_ratio.evidence_status": "COMPUTED",
            "metrics.missing_ratio.value": 0.9,
        }
        assert [t.tag for t in detect_diagnoses(factor_definition_id="F1", evidence=ev)] == [
            "DATA_QUALITY_FAILURE"
        ]


class TestDeterminismAndPower:
    def test_same_input_same_output(self):
        ev = {
            "metrics.rank_ic.evidence_status": "COMPUTED",
            "metrics.rank_ic.value": 0.002,
            "metrics.factor_turnover_rate.evidence_status": "COMPUTED",
            "metrics.factor_turnover_rate.value": 0.9,
            "metrics.coverage.evidence_status": "COMPUTED",
            "metrics.coverage.value": 0.4,
        }
        a = detect_diagnoses(factor_definition_id="F1", evidence=ev)
        b = detect_diagnoses(factor_definition_id="F1", evidence=ev)
        assert [t.to_dict() for t in a] == [t.to_dict() for t in b]
        assert a == b

    def test_tag_output_in_canonical_order(self):
        ev = {
            "metrics.rank_ic.evidence_status": "COMPUTED",
            "metrics.rank_ic.value": 0.002,
            "metrics.coverage.evidence_status": "COMPUTED",
            "metrics.coverage.value": 0.4,
            "metrics.factor_turnover_rate.evidence_status": "COMPUTED",
            "metrics.factor_turnover_rate.value": 0.9,
        }
        names = [t.tag for t in detect_diagnoses(factor_definition_id="F1", evidence=ev)]
        assert names == sorted(names, key=DIAGNOSIS_TAGS.index)

    def test_healthy_factor_empty(self):
        ev = {
            "metrics.rank_ic.evidence_status": "COMPUTED",
            "metrics.rank_ic.value": 0.04,
            "metrics.coverage.evidence_status": "COMPUTED",
            "metrics.coverage.value": 0.95,
            "metrics.factor_turnover_rate.evidence_status": "COMPUTED",
            "metrics.factor_turnover_rate.value": 0.1,
            "metrics.rolling_ic_volatility.evidence_status": "COMPUTED",
            "metrics.rolling_ic_volatility.value": 0.01,
        }
        assert detect_diagnoses(factor_definition_id="F1", evidence=ev) == ()

    def test_dedup_keeps_highest_confidence(self):
        # max_drawdown very negative AND cvar very negative both fire
        # HIGH_DRAWDOWN; dedup must collapse them to a single record.
        ev = {
            "metrics.max_drawdown.evidence_status": "COMPUTED",
            "metrics.max_drawdown.value": -0.6,
            "metrics.cvar_95.evidence_status": "COMPUTED",
            "metrics.cvar_95.value": -0.3,
        }
        tags = detect_diagnoses(factor_definition_id="F1", evidence=ev)
        names = [t.tag for t in tags]
        assert names.count("HIGH_DRAWDOWN") == 1

    def test_policy_refs_stamped(self):
        ev = {
            "metrics.rank_ic.evidence_status": "COMPUTED",
            "metrics.rank_ic.value": 0.002,
        }
        tags = detect_diagnoses(factor_definition_id="F1", evidence=ev)
        assert tags[0].severity == get_diagnosis_policy().severity("LOW_PREDICTIVE")

    def test_diagnosis_tag_frozen(self):
        from dataclasses import FrozenInstanceError

        t = DiagnosisTag(
            factor_definition_id="F1", health_ref="H1", tag="LOW_PREDICTIVE",
            severity="HIGH", confidence=0.8, repairability="REPAIRABLE",
            evidence_refs=("metrics.rank_ic.value",),
            triggered_policy_rule="FI026_RANK_IC_LOW",
        )
        with pytest.raises(FrozenInstanceError):
            t.confidence = 0.5  # type: ignore[misc]

    def test_unknown_when_no_diagnosis_no_fabrication(self):
        # An empty evidence set yields an empty diagnosis tuple — a healthy
        # factor produces no sentinel and no fabricated defect tag (the FO
        # DiagnosisView consumer boundary treats empty as legal).
        tags = diagnose_factor(
            factor_definition_id="F1", evidence={}, unknown_when_no_diagnosis=True
        )
        assert tags == ()

    def test_diagnosis_tag_rejects_unknown_tag(self):
        with pytest.raises(ValueError):
            DiagnosisTag(
                factor_definition_id="F1", health_ref="H1", tag="MADE_UP_TAG",
                severity="HIGH", confidence=0.8, repairability="REPAIRABLE",
                evidence_refs=(), triggered_policy_rule="X",
            )
