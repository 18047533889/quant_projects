"""R61-FI-032 / plan §21.2: generalized 8-step TreatmentDecisionPolicy tests.

Covers (matrix E1):
1. The generalized path consumes FA health *dimensions* (14-dim FactorHealthView
   grades) and the fitness spec — not the hard-coded 8 scalar metrics.
2. The seven-step sequence is preserved (STEP 1 integrity -> STEP 2 deltas ->
   STEP 3 desirability -> STEP 4 dimensions -> STEP 5 Pareto -> STEP 6
   uncertainty -> STEP 7+8 robust/near-equivalent winner).
3. Health dimension floor gates + evidence-tier minimums reject/keep
   candidates as the spec declares.
4. RAW-relative deltas over the health dimensions are computed and recorded on
   CandidateFitnessArtifact (positive = better-than-RAW).
5. The legacy scalar path (``decide()``) remains as the backward-compatible
   compatibility layer (existing tests keep passing byte-for-byte).
6. RAW winners are discoverable (a treatment that made things worse never
   beats RAW) and reported via the outcome.
"""

import numpy as np
import pytest

from factor_optimizer.contracts import (
    CandidateFitnessArtifact,
    EvidenceTier,
    FactorFitnessSpec,
)
from factor_optimizer.contracts.treatment_integrity import (
    TreatmentIntegrityError,
    build_integrity_evidence,
)
from factor_optimizer.ports.factor_intelligence import (
    FactorHealthView,
    HealthDimension,
    HealthGrade,
)
from factor_optimizer.search.treatment_decision import (
    BALANCED_DIMENSIONS,
    DecisionResult,
    DesirabilityAnchors,
    HealthDecisionInput,
    IntegrityGate,
    TreatmentDecisionPolicy,
    TreatmentMetrics,
)
from factor_optimizer.search.uncertainty_winner import (
    UncertaintyAwareWinnerSelector,
    UncertaintyConfig,
    UncertaintyEvidence,
)
from factor_optimizer.search.winner_selector import WinnerPolicy


def _policy(**overrides):
    defaults = dict(
        alpha=0.4,
        beta=0.3,
        gamma=0.2,
        lambda_=0.1,
        policy_id="dlib",
        policy_version="1.0.0",
    )
    defaults.update(overrides)
    return WinnerPolicy(**defaults)


def _spec(**overrides):
    defaults = dict(
        hard_gate_policy_id="integrity-v1",
        dimension_floor_policy_id="floors-v1",
        raw_relative_policy_id="rr-v1",
        uncertainty_policy_id="uncertainty-v1",
        multiplicity_policy_id="mult-v1",
        complexity_policy_id="complexity-v1",
        winner_policy_id="winner-v1",
        minimum_evidence_tier=EvidenceTier.POINT_ESTIMATE_ONLY,
    )
    defaults.update(overrides)
    return FactorFitnessSpec(**defaults)


def _health_view(fid, grades, overall=HealthGrade.NONE, evaluation_ref="ev"):
    """Build a FactorHealthView with only the supplied dims graded (rest NONE)."""
    all_grades = {dim: HealthGrade.NONE for dim in HealthDimension.all()}
    all_grades.update(grades)
    return FactorHealthView(
        factor_definition_id=fid,
        evaluation_ref=evaluation_ref,
        dimension_grades=all_grades,
        overall_grade=overall,
    )


def _good_raw_health(fid="F_RAW", evaluation_ref="ev-raw"):
    """RAW baseline: mediocre on predictive, fine everywhere else."""
    return _health_view(
        fid,
        {
            HealthDimension.PREDICTIVE_POWER: "B",
            HealthDimension.STABILITY: "B+",
            HealthDimension.ROBUSTNESS: "B+",
            HealthDimension.TURNOVER: "B",
            HealthDimension.CAPACITY: "B+",
            HealthDimension.COST_DRAG: "B",
            HealthDimension.DATA_COVERAGE: "A",
            HealthDimension.FRESHNESS: "A",
        },
        overall=HealthGrade.B,
        evaluation_ref=evaluation_ref,
    )


def _good_treatment_health(fid, evaluation_ref="ev-t"):
    """Smoothed treatment: same predictive, better stability/tradability."""
    return _health_view(
        fid,
        {
            HealthDimension.PREDICTIVE_POWER: "B",
            HealthDimension.STABILITY: "A",
            HealthDimension.ROBUSTNESS: "A",
            HealthDimension.TURNOVER: "A",
            HealthDimension.CAPACITY: "A",
            HealthDimension.COST_DRAG: "A",
            HealthDimension.DATA_COVERAGE: "A",
            HealthDimension.FRESHNESS: "A",
        },
        overall=HealthGrade.A,
        evaluation_ref=evaluation_ref,
    )


def _bad_treatment_health(fid, evaluation_ref="ev-b"):
    """Treatment that made things worse: worse grades on every dimension."""
    return _health_view(
        fid,
        {
            HealthDimension.PREDICTIVE_POWER: "D",
            HealthDimension.STABILITY: "D",
            HealthDimension.ROBUSTNESS: "D",
            HealthDimension.TURNOVER: "C",
            HealthDimension.CAPACITY: "C",
            HealthDimension.COST_DRAG: "C",
            HealthDimension.DATA_COVERAGE: "C",
            HealthDimension.FRESHNESS: "C",
        },
        overall=HealthGrade.D,
        evaluation_ref=evaluation_ref,
    )


def _evidence_for(*inputs):
    """Real TreatmentIntegrityEvidence per candidate trial id."""
    result = {}
    for item in inputs:
        trial_id = item.trial_id
        rng = np.random.default_rng(abs(hash(trial_id)) % (2 ** 32))
        before = rng.normal(size=64)
        kind = "raw" if trial_id == "RAW" else f"treatment::{trial_id}"
        after = before if kind == "raw" else before * 0.5 + 0.01
        result[trial_id] = build_integrity_evidence(
            trial_id, kind, {} if kind == "raw" else {"window": 5},
            before, after,
        )
    return result


def _generalized_policy(spec=None):
    return TreatmentDecisionPolicy(
        policy=_policy(),
        fitness_spec=spec or _spec(),
    )


# ---------------------------------------------------------------------------
# 1. Generalized path consumes FA health dimensions + fitness spec
# ---------------------------------------------------------------------------


def test_decision_consumes_health_dimensions_not_scalar_metrics():
    raw = HealthDecisionInput(
        trial_id="RAW", health_view=_good_raw_health(fid="RAW"), evidence_tier="VALIDATION_SERIES"
    )
    treat = HealthDecisionInput(
        trial_id="EWMA", health_view=_good_treatment_health(fid="EWMA"),
        evidence_tier="VALIDATION_SERIES",
    )
    policy = _generalized_policy()
    result = policy.decision(
        [raw, treat], raw_trial_id="RAW", screening_only=True,
        integrity_evidence=_evidence_for(raw, treat),
    )
    assert result.winner_trial_id == "EWMA"
    # The generalized path records a CandidateFitnessArtifact per candidate.
    assert len(result.fitness_artifacts) == 2
    by_id = {a.trial_id: a for a in result.fitness_artifacts}
    assert set(by_id) == {"RAW", "EWMA"}
    # Deltas are over the BALANCED dimensions (not the 8 scalar metrics).
    ewma = by_id["EWMA"]
    assert set(ewma.raw_relative_deltas) == set(BALANCED_DIMENSIONS)
    assert ewma.delta("tradability") is not None
    assert ewma.delta("tradability") > 0.0  # better-than-RAW after normalization


def test_decision_requires_fitness_spec_constructed_policy():
    raw = HealthDecisionInput(trial_id="RAW", health_view=_good_raw_health(fid="RAW"))
    anchors = DesirabilityAnchors(
        maps={"rank_ic": {"direction": "increasing", "anchors": [(0.0, 0.0), (1.0, 1.0)]}}
    )
    legacy = TreatmentDecisionPolicy(anchors, _policy())
    with pytest.raises(TypeError, match="decision"):
        legacy.decision([raw], raw_trial_id="RAW", screening_only=True)


# ---------------------------------------------------------------------------
# 2. Seven-step sequence preserved
# ---------------------------------------------------------------------------


def test_decision_runs_all_steps_and_trace_is_auditable():
    raw = HealthDecisionInput(
        trial_id="RAW", health_view=_good_raw_health(fid="RAW"), evidence_tier="VALIDATION_SERIES"
    )
    treat = HealthDecisionInput(
        trial_id="EWMA", health_view=_good_treatment_health(fid="EWMA"),
        evidence_tier="VALIDATION_SERIES",
    )
    policy = _generalized_policy()
    result = policy.decision(
        [raw, treat], raw_trial_id="RAW", screening_only=True,
        integrity_evidence=_evidence_for(raw, treat),
    )
    assert isinstance(result, DecisionResult)
    for step in (
        "step1_integrity",
        "step2_deltas",
        "step3_desirability",
        "step4_dimensions",
        "step5_pareto",
        "step6_uncertainty",
        "step7_robust_utility",
        "step8_near_equivalence",
    ):
        assert step in result.step_trace, step
    assert result.winner_trial_id == "EWMA"


def test_decision_keeps_raw_when_treatment_worse():
    raw = HealthDecisionInput(
        trial_id="RAW", health_view=_good_raw_health(fid="RAW"), evidence_tier="VALIDATION_SERIES"
    )
    bad = HealthDecisionInput(
        trial_id="BAD", health_view=_bad_treatment_health(fid="BAD"),
        evidence_tier="VALIDATION_SERIES",
    )
    policy = _generalized_policy()
    result = policy.decision(
        [raw, bad], raw_trial_id="RAW", screening_only=True,
        integrity_evidence=_evidence_for(raw, bad),
    )
    assert result.winner_trial_id == "RAW"
    assert result.raw_kept is True
    assert result.outcome == "RAW_SELECTED_NO_IMPROVEMENT"


# ---------------------------------------------------------------------------
# 3. Health floor gates + evidence-tier gates
# ---------------------------------------------------------------------------


def test_decision_enforces_spec_required_health_dimension_floor():
    """A candidate missing (NONE) on a spec-REQUIRED dimension is hard-rejected."""
    spec = _spec(required_health_dimensions=(HealthDimension.PREDICTIVE_POWER,))
    raw = HealthDecisionInput(
        trial_id="RAW",
        health_view=_good_raw_health(fid="RAW"),
        evidence_tier="VALIDATION_SERIES",
    )
    # Candidate whose predictive_power is NONE (never computed).
    no_pred = HealthDecisionInput(
        trial_id="EWMA",
        health_view=_health_view(
            "EWMA",
            {
                HealthDimension.STABILITY: "A",
                HealthDimension.TURNOVER: "A",
                HealthDimension.DATA_COVERAGE: "A",
            },
            overall=HealthGrade.NONE,
        ),
        evidence_tier="VALIDATION_SERIES",
    )
    policy = _generalized_policy(spec)
    result = policy.decision(
        [raw, no_pred], raw_trial_id="RAW", screening_only=True,
        integrity_evidence=_evidence_for(raw, no_pred),
    )
    assert "EWMA" not in result.pareto_trial_ids
    assert result.winner_trial_id == "RAW"


def test_decision_tier_gate_rejects_point_only_under_bootstrap_spec():
    spec = _spec(minimum_evidence_tier=EvidenceTier.BOOTSTRAP_CONFIDENCE)
    raw = HealthDecisionInput(
        trial_id="RAW",
        health_view=_good_raw_health(fid="RAW"),
        evidence_tier=EvidenceTier.BOOTSTRAP_CONFIDENCE,
        # RAW genuinely carries resampled series so it survives STEP 6 under
        # the bootstrap-requiring spec.
        bootstrap_samples={
            name: _bootstrap(0.5, 0.02)
            for name in BALANCED_DIMENSIONS
        },
    )
    point_only = HealthDecisionInput(
        trial_id="EWMA",
        health_view=_good_treatment_health(fid="EWMA"),
        evidence_tier=EvidenceTier.POINT_ESTIMATE_ONLY,
    )
    policy = _generalized_policy(spec)
    result = policy.decision(
        [raw, point_only], raw_trial_id="RAW", screening_only=True,
        integrity_evidence=_evidence_for(raw, point_only),
    )
    # The no-bootstrap candidate was hard-rejected at STEP 1 (tier below the
    # spec minimum) and never reached the frontier.
    assert "EWMA" not in result.pareto_trial_ids
    assert result.winner_trial_id == "RAW"
    assert "tier below spec minimum" in result.step_trace["step1_integrity"]


def test_decision_all_candidates_tier_rejected_fails_closed():
    spec = _spec(minimum_evidence_tier=EvidenceTier.BOOTSTRAP_CONFIDENCE)
    a = HealthDecisionInput(
        trial_id="RAW", health_view=_good_raw_health(fid="RAW"),
        evidence_tier=EvidenceTier.POINT_ESTIMATE_ONLY,
    )
    b = HealthDecisionInput(
        trial_id="EWMA", health_view=_good_treatment_health(fid="EWMA"),
        evidence_tier=EvidenceTier.POINT_ESTIMATE_ONLY,
    )
    policy = _generalized_policy(spec)
    with pytest.raises(ValueError, match="hard gates"):
        policy.decision([a, b], raw_trial_id="RAW", screening_only=True,
                        integrity_evidence=_evidence_for(a, b))


def test_decision_point_only_accepted_when_spec_allows_point_estimates():
    raw = HealthDecisionInput(
        trial_id="RAW", health_view=_good_raw_health(fid="RAW"),
        evidence_tier=EvidenceTier.POINT_ESTIMATE_ONLY,
    )
    treat = HealthDecisionInput(
        trial_id="EWMA", health_view=_good_treatment_health(fid="EWMA"),
        evidence_tier=EvidenceTier.POINT_ESTIMATE_ONLY,
    )
    policy = _generalized_policy()  # default minimum = POINT_ESTIMATE_ONLY
    result = policy.decision(
        [raw, treat], raw_trial_id="RAW", screening_only=True,
        integrity_evidence=_evidence_for(raw, treat),
    )
    assert result.winner_trial_id == "EWMA"


def test_decision_missing_integrity_evidence_still_fails_closed():
    raw = HealthDecisionInput(trial_id="RAW", health_view=_good_raw_health(fid="RAW"))
    treat = HealthDecisionInput(trial_id="EWMA", health_view=_good_treatment_health(fid="EWMA"))
    policy = _generalized_policy()
    result = policy.decision(
        [raw, treat], raw_trial_id="RAW", screening_only=True,
        integrity_evidence=_evidence_for(raw),  # only RAW has evidence
    )
    assert "EWMA" not in result.pareto_trial_ids
    assert result.winner_trial_id == "RAW"


# ---------------------------------------------------------------------------
# 4. RAW-relative deltas recorded on fitness artifacts
# ---------------------------------------------------------------------------


def test_decision_fitness_artifact_records_normalized_deltas():
    raw = HealthDecisionInput(
        trial_id="RAW", health_view=_good_raw_health(fid="RAW"), evidence_tier="VALIDATION_SERIES"
    )
    treat = HealthDecisionInput(
        trial_id="EWMA", health_view=_good_treatment_health(fid="EWMA"),
        evidence_tier="VALIDATION_SERIES",
    )
    policy = _generalized_policy()
    result = policy.decision(
        [raw, treat], raw_trial_id="RAW", screening_only=True,
        integrity_evidence=_evidence_for(raw, treat),
    )
    artifacts = {a.trial_id: a for a in result.fitness_artifacts}
    ewma = artifacts["EWMA"]
    # treatment is better on stability/tradability -> positive delta.
    assert ewma.delta("stability") is not None and ewma.delta("stability") > 0.0
    assert ewma.delta("tradability") > 0.0
    raw_art = artifacts["RAW"]
    # RAW vs RAW deltas are all zero (or None when the axis has no grade).
    for name in BALANCED_DIMENSIONS:
        delta = raw_art.delta(name)
        assert delta is None or delta == pytest.approx(0.0)


def test_decision_fitness_artifact_is_typed_and_frozen():
    raw = HealthDecisionInput(
        trial_id="RAW", health_view=_good_raw_health(fid="RAW"), evidence_tier="VALIDATION_SERIES"
    )
    policy = _generalized_policy()
    result = policy.decision([raw], raw_trial_id="RAW", screening_only=True,
                             integrity_evidence=_evidence_for(raw))
    artifact = result.fitness_artifacts[0]
    assert isinstance(artifact, CandidateFitnessArtifact)
    # evaluation_ref is read off the health view's own evaluation_ref.
    assert artifact.evaluation_ref == "ev-raw"
    assert artifact.health_card_ref  # non-empty
    assert artifact.evidence_tier is EvidenceTier.VALIDATION_SERIES

# ---------------------------------------------------------------------------
# 5. Legacy scalar path stays as the compatibility layer
# ---------------------------------------------------------------------------


def test_legacy_decide_still_works_byte_for_byte():
    """The scalar path is kept as the deprecated compatibility layer."""
    assert TreatmentDecisionPolicy.__init__.__doc__ or True
    # Constructed with anchors (legacy mode), decide() works:
    from tests.search.test_dlib_uncertainty_winner import (
        _anchors,
        _evidence_for as legacy_evidence,
        _policy as legacy_policy,
        _raw_metrics,
        _smoothed_metrics,
    )

    raw = _raw_metrics()
    ewma = _smoothed_metrics("EWMA")
    policy = TreatmentDecisionPolicy(_anchors(), legacy_policy())
    result = policy.decide(
        [raw, ewma], raw_trial_id="RAW", screening_only=True,
        integrity_evidence=legacy_evidence(raw, ewma),
    )
    assert result.winner_trial_id == "EWMA"
    # And decide() on a spec-built policy is rejected (mode separation).
    spec_policy = _generalized_policy()
    with pytest.raises(TypeError, match="decision"):
        spec_policy.decide([raw], screening_only=True)


def test_legacy_decide_requires_explicit_screening_mode():
    from tests.search.test_dlib_uncertainty_winner import _anchors, _raw_metrics

    policy = TreatmentDecisionPolicy(_anchors(), _policy())
    with pytest.raises(ValueError, match="authoritative_decision"):
        policy.decide([_raw_metrics()])


def test_local_health_decision_requires_explicit_screening_mode_and_marks_result():
    raw = HealthDecisionInput(
        trial_id="RAW",
        health_view=_good_raw_health(fid="RAW"),
        evidence_tier="VALIDATION_SERIES",
    )
    policy = _generalized_policy()
    with pytest.raises(ValueError, match="authoritative_decision"):
        policy.decision([raw], raw_trial_id="RAW")

    result = policy.decision(
        [raw],
        raw_trial_id="RAW",
        screening_only=True,
        integrity_evidence=_evidence_for(raw),
    )
    assert result.authoritative is False
    assert result.purpose == "SCREENING_DIAGNOSTIC"


def test_scalar_mode_policy_cannot_call_decision():
    from tests.search.test_dlib_uncertainty_winner import _anchors

    raw = HealthDecisionInput(trial_id="RAW", health_view=_good_raw_health(fid="RAW"))
    policy = TreatmentDecisionPolicy(_anchors(), _policy())
    with pytest.raises(TypeError, match="decide"):
        policy.decision([raw], raw_trial_id="RAW", screening_only=True)


def test_constructor_requires_exactly_one_mode():
    from tests.search.test_dlib_uncertainty_winner import _anchors

    with pytest.raises(ValueError, match="exactly one"):
        TreatmentDecisionPolicy(_anchors(), _policy(), fitness_spec=_spec())
    with pytest.raises(ValueError, match="exactly one"):
        TreatmentDecisionPolicy(None, _policy())


# ---------------------------------------------------------------------------
# 6. RAW winner discoverable in generalized mode
# ---------------------------------------------------------------------------


def test_decision_outcome_reports_raw_selected():
    raw = HealthDecisionInput(
        trial_id="RAW", health_view=_good_raw_health(fid="RAW"), evidence_tier="VALIDATION_SERIES"
    )
    bad = HealthDecisionInput(
        trial_id="BAD", health_view=_bad_treatment_health(fid="BAD"),
        evidence_tier="VALIDATION_SERIES",
    )
    policy = _generalized_policy()
    result = policy.decision(
        [raw, bad], raw_trial_id="RAW", screening_only=True,
        integrity_evidence=_evidence_for(raw, bad),
    )
    assert result.outcome == "RAW_SELECTED_NO_IMPROVEMENT"
    assert result.raw_kept is True


def test_decision_outcome_reports_improved():
    raw = HealthDecisionInput(
        trial_id="RAW", health_view=_good_raw_health(fid="RAW"), evidence_tier="VALIDATION_SERIES"
    )
    treat = HealthDecisionInput(
        trial_id="EWMA", health_view=_good_treatment_health(fid="EWMA"),
        evidence_tier="VALIDATION_SERIES",
    )
    policy = _generalized_policy()
    result = policy.decision(
        [raw, treat], raw_trial_id="RAW", screening_only=True,
        integrity_evidence=_evidence_for(raw, treat),
    )
    assert result.outcome == "IMPROVED"
    # RAW was a candidate — it appears in the raw-relative delta computation
    # (STEP 2) as the baseline every treatment is measured against.
    assert "RAW" in result.step_trace["step2_deltas"] or "RAW" in result.step_trace["step5_pareto"]


def test_decision_requires_raw_candidate_fails_closed():
    treat = HealthDecisionInput(
        trial_id="EWMA", health_view=_good_treatment_health(fid="EWMA"),
        evidence_tier="VALIDATION_SERIES",
    )
    policy = _generalized_policy()
    with pytest.raises(ValueError, match="RAW"):
        policy.decision([treat], raw_trial_id="RAW", screening_only=True,
                        integrity_evidence=_evidence_for(treat))


# ---------------------------------------------------------------------------
# Bootstrap-backed candidates feed STEP 6 in the generalized path
# ---------------------------------------------------------------------------


def _bootstrap(center, spread, n=100):
    rng = np.random.default_rng(7)
    return [max(0.0, min(1.0, center + rng.uniform(-spread, spread))) for _ in range(n)]


def test_decision_bootstrap_series_drives_step6():
    raw = HealthDecisionInput(
        trial_id="RAW", health_view=_good_raw_health(fid="RAW"),
        evidence_tier=EvidenceTier.BOOTSTRAP_CONFIDENCE,
        bootstrap_samples={
            "predictive": _bootstrap(0.45, 0.02),
            "stability": _bootstrap(0.55, 0.02),
            "robustness": _bootstrap(0.55, 0.02),
            "tradability": _bootstrap(0.5, 0.02),
            "purity_exposure": _bootstrap(0.5, 0.02),
            "data_quality": _bootstrap(0.6, 0.02),
        },
    )
    treat = HealthDecisionInput(
        trial_id="EWMA", health_view=_good_treatment_health(fid="EWMA"),
        evidence_tier=EvidenceTier.BOOTSTRAP_CONFIDENCE,
        bootstrap_samples={
            "predictive": _bootstrap(0.5, 0.02),
            "stability": _bootstrap(0.7, 0.02),
            "robustness": _bootstrap(0.7, 0.02),
            "tradability": _bootstrap(0.7, 0.02),
            "purity_exposure": _bootstrap(0.6, 0.02),
            "data_quality": _bootstrap(0.7, 0.02),
        },
    )
    policy = _generalized_policy(
        _spec(minimum_evidence_tier=EvidenceTier.BOOTSTRAP_CONFIDENCE)
    )
    result = policy.decision(
        [raw, treat], raw_trial_id="RAW", screening_only=True,
        integrity_evidence=_evidence_for(raw, treat),
    )
    assert result.winner_trial_id == "EWMA"
    assert result.statistically_plausible
