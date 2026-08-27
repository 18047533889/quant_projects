"""DLIB-FO-002..007 tests: uncertainty-aware winner + desirability policy.

Covers:
- TreatmentDecisionPolicy 8-step on a RAW vs EWMA vs KAMA vs Dual-Neutral
  trade-off (RAW higher RankIC/lower ICIR/high turnover; smoothed slightly
  lower RankIC/higher ICIR/lower turnover -> must NOT pick RAW by single
  RankIC, must NOT hard-threshold out smoothed -> winner via Pareto +
  uncertainty + robust-utility + complexity).
- UncertaintyAwareWinnerSelector equivalence case (A RankIC 0.0311 vs B 0.0309
  with heavily-overlapping bootstrap CI and B 35% lower turnover -> B may win).
- DesirabilityPolicyRegistry session-freeze.
- MultiplicityArtifact full-proposal-process traceability.
- TestAuthorityBroker sealed-test cannot leak into search worker object graph
  (adversarial: forge TestDataCapability inside search worker / pass test mask
  / reuse sealed handle -> must FAIL CLOSED).
"""

import math

import pytest

from factor_optimizer.contracts.multiplicity import MultiplicityArtifact
from factor_optimizer.contracts.treatment_result import (
    TreatmentOptimizationResultArtifact,
)
from factor_optimizer.search.desirability_registry import (
    DesirabilityContext,
    DesirabilityPolicy,
    DesirabilityPolicyRegistry,
)
from factor_optimizer.search.treatment_decision import (
    DecisionResult,
    DesirabilityAnchors,
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


def _anchors():
    """Desirability anchors for the treatment decision pipeline."""
    return DesirabilityAnchors(
        maps={
            "rank_ic": {
                "direction": "increasing",
                "anchors": [
                    (0.010, 0.05),
                    (0.020, 0.35),
                    (0.030, 0.72),
                    (0.040, 0.90),
                    (0.050, 1.00),
                ],
            },
            "icir": {
                "direction": "increasing",
                "anchors": [
                    (0.0, 0.05),
                    (0.2, 0.30),
                    (0.4, 0.55),
                    (0.6, 0.75),
                    (0.8, 0.88),
                    (1.0, 1.00),
                ],
            },
            "turnover": {
                "direction": "decreasing",
                "anchors": [
                    (0.05, 1.00),
                    (0.15, 0.72),
                    (0.30, 0.42),
                    (0.50, 0.18),
                    (0.70, 0.06),
                ],
            },
            "cost_adjusted_alpha": {
                "direction": "increasing",
                "anchors": [
                    (-0.02, 0.05),
                    (0.00, 0.30),
                    (0.01, 0.60),
                    (0.02, 0.80),
                    (0.03, 0.95),
                    (0.05, 1.00),
                ],
            },
            "worst_slice": {
                "direction": "increasing",
                "anchors": [
                    (-0.02, 0.05),
                    (0.00, 0.20),
                    (0.01, 0.45),
                    (0.02, 0.72),
                    (0.03, 0.90),
                    (0.04, 1.00),
                ],
            },
            "exposure": {
                "direction": "decreasing",
                "anchors": [
                    (0.0, 1.00),
                    (0.1, 0.60),
                    (0.2, 0.30),
                    (0.3, 0.10),
                ],
            },
            "coverage": {
                "direction": "increasing",
                "anchors": [
                    (0.50, 0.05),
                    (0.70, 0.30),
                    (0.85, 0.60),
                    (0.95, 0.85),
                    (1.00, 1.00),
                ],
            },
            "stability": {
                "direction": "increasing",
                "anchors": [
                    (0.0, 0.05),
                    (0.5, 0.40),
                    (0.8, 0.70),
                    (1.0, 1.00),
                ],
            },
        }
    )


def _raw_metrics(**overrides):
    """RAW baseline: higher RankIC, lower ICIR, high turnover."""
    defaults = dict(
        trial_id="RAW",
        rank_ic=0.0311,
        icir=0.30,
        turnover=0.45,
        cost_adjusted_alpha=0.012,
        worst_slice=0.008,
        exposure=0.12,
        coverage=0.95,
        stability=0.60,
        robustness=0.50,
        complexity_score=0.0,
        n_transforms=0,
        compute_cost=1.0,
    )
    defaults.update(overrides)
    return TreatmentMetrics(**defaults)


def _smoothed_metrics(trial_id, **overrides):
    """Smoothed treatment: slightly lower RankIC, higher ICIR, lower turnover."""
    defaults = dict(
        trial_id=trial_id,
        rank_ic=0.0309,
        icir=0.45,
        turnover=0.29,
        cost_adjusted_alpha=0.014,
        worst_slice=0.010,
        exposure=0.10,
        coverage=0.95,
        stability=0.75,
        robustness=0.60,
        complexity_score=0.3,
        n_transforms=1,
        compute_cost=2.0,
    )
    defaults.update(overrides)
    return TreatmentMetrics(**defaults)


# ---------------------------------------------------------------------------
# DLIB-FO-002: TreatmentDecisionPolicy 8-step
# ---------------------------------------------------------------------------


def test_treatment_decision_8_step_does_not_pick_raw_by_single_rank_ic():
    """RAW has higher RankIC but lower ICIR and high turnover.

    The smoothed treatment has slightly lower RankIC but higher ICIR and much
    lower turnover.  The pipeline must NOT pick RAW by single-RankIC, and must
    NOT hard-threshold out the smoothed treatment.  The winner should come via
    Pareto + uncertainty + robust-utility + complexity.
    """
    raw = _raw_metrics()
    ewma = _smoothed_metrics("EWMA")
    kama = _smoothed_metrics("KAMA", rank_ic=0.0305, icir=0.50, turnover=0.25)
    dual = _smoothed_metrics("DUAL", rank_ic=0.0300, icir=0.55, turnover=0.22)

    policy = TreatmentDecisionPolicy(_anchors(), _policy())
    result = policy.decide([raw, ewma, kama, dual], raw_trial_id="RAW")

    # RAW was a candidate (treated-got-worse discoverable).  Here RAW is
    # dominated on every dimension by the smoothed treatments, so it is
    # correctly removed at STEP 5 (Pareto) — the winner is a treatment.
    assert result.winner_trial_id != "RAW"
    # The winner must be one of the smoothed treatments.
    assert result.winner_trial_id in {"EWMA", "KAMA", "DUAL"}
    # The pipeline must have run all 8 steps.
    assert "step1_integrity" in result.step_trace
    assert "step2_deltas" in result.step_trace
    assert "step3_desirability" in result.step_trace
    assert "step4_dimensions" in result.step_trace
    assert "step5_pareto" in result.step_trace
    assert "step6_uncertainty" in result.step_trace
    assert "step7_robust_utility" in result.step_trace
    assert "step8_near_equivalence" in result.step_trace


def test_treatment_decision_keeps_raw_when_treatment_worse():
    """When every treatment is worse than RAW, RAW must win (discoverable)."""
    raw = _raw_metrics()
    bad = _smoothed_metrics(
        "BAD", rank_ic=0.020, icir=0.20, turnover=0.60, stability=0.30
    )
    policy = TreatmentDecisionPolicy(_anchors(), _policy())
    result = policy.decide([raw, bad], raw_trial_id="RAW")
    assert result.winner_trial_id == "RAW"


def test_treatment_decision_integrity_gate_hard_rejects():
    """A candidate with broken coverage (PIT violation) is hard-rejected."""
    raw = _raw_metrics()
    broken = _smoothed_metrics("BROKEN", coverage=0.0)
    policy = TreatmentDecisionPolicy(_anchors(), _policy())
    result = policy.decide([raw, broken], raw_trial_id="RAW")
    # The broken candidate is hard-rejected at STEP 1 and never reaches the
    # frontier.
    assert "BROKEN" not in result.pareto_trial_ids
    assert "BROKEN" not in result.statistically_plausible


def test_treatment_decision_requires_raw_candidate():
    """The pipeline fails closed if RAW is not among the candidates."""
    ewma = _smoothed_metrics("EWMA")
    policy = TreatmentDecisionPolicy(_anchors(), _policy())
    with pytest.raises(ValueError, match="RAW"):
        policy.decide([ewma], raw_trial_id="RAW")


def test_treatment_decision_all_hard_rejected_fails_closed():
    """If every candidate fails integrity gates, the pipeline raises."""
    broken1 = _smoothed_metrics("B1", coverage=0.0)
    broken2 = _smoothed_metrics("B2", coverage=0.0)
    policy = TreatmentDecisionPolicy(_anchors(), _policy())
    with pytest.raises(ValueError, match="integrity"):
        policy.decide([broken1, broken2], raw_trial_id="B1")


# ---------------------------------------------------------------------------
# DLIB-FO-004: UncertaintyAwareWinnerSelector equivalence case
# ---------------------------------------------------------------------------


def _bootstrap_samples(center, spread, n=200):
    """Deterministic pseudo-bootstrap samples around ``center``."""
    import random

    rng = random.Random(42)
    return [max(0.0, min(1.0, center + rng.uniform(-spread, spread))) for _ in range(n)]


def test_uncertainty_selector_prefers_lower_turnover_on_equivalence():
    """A RankIC 0.0311 vs B 0.0309 with heavily-overlapping CI and B 35%
    lower turnover -> B may win.

    A has higher point RankIC but its bootstrap CI heavily overlaps B's, and
    B has much lower turnover.  The uncertainty-aware selector should prefer
    B (the simpler / lower-turnover candidate) because the RankIC edge is
    within the noise.
    """
    # A: higher RankIC point estimate, but wide CI (noisy).
    a = UncertaintyEvidence(
        trial_id="A",
        dimension_samples={
            "predictive": _bootstrap_samples(0.72, 0.10),
            "tradability": _bootstrap_samples(0.30, 0.05),
            "stability": _bootstrap_samples(0.60, 0.05),
        },
        complexity_score=0.3,
        turnover=0.45,
        compute_cost=2.0,
        n_transforms=1,
    )
    # B: slightly lower RankIC point estimate, but much lower turnover.
    b = UncertaintyEvidence(
        trial_id="B",
        dimension_samples={
            "predictive": _bootstrap_samples(0.70, 0.10),
            "tradability": _bootstrap_samples(0.55, 0.05),
            "stability": _bootstrap_samples(0.75, 0.05),
        },
        complexity_score=0.1,
        turnover=0.29,
        compute_cost=1.0,
        n_transforms=0,
    )
    selector = UncertaintyAwareWinnerSelector(
        _policy(),
        UncertaintyConfig(
            minimum_meaningful_improvement=0.02,
            equivalence_region=0.5,
            confidence_level=0.95,
            dominance_threshold=0.5,
        ),
    )
    winner = selector.select([a, b], {"A": 0.5, "B": 0.6})
    # B's lower turnover + higher tradability should win despite A's slightly
    # higher RankIC point estimate.
    assert winner.trial_id == "B"


def test_uncertainty_selector_prefers_clear_winner():
    """When A clearly dominates on every dimension, A wins regardless of
    complexity."""
    a = UncertaintyEvidence(
        trial_id="A",
        dimension_samples={
            "predictive": _bootstrap_samples(0.9, 0.02),
            "tradability": _bootstrap_samples(0.9, 0.02),
            "stability": _bootstrap_samples(0.9, 0.02),
        },
        complexity_score=0.5,
        turnover=0.3,
        compute_cost=3.0,
        n_transforms=2,
    )
    b = UncertaintyEvidence(
        trial_id="B",
        dimension_samples={
            "predictive": _bootstrap_samples(0.4, 0.02),
            "tradability": _bootstrap_samples(0.4, 0.02),
            "stability": _bootstrap_samples(0.4, 0.02),
        },
        complexity_score=0.1,
        turnover=0.1,
        compute_cost=1.0,
        n_transforms=0,
    )
    selector = UncertaintyAwareWinnerSelector(_policy())
    winner = selector.select([a, b], {"A": 0.9, "B": 0.4})
    assert winner.trial_id == "A"


def test_uncertainty_selector_empty_fails_closed():
    selector = UncertaintyAwareWinnerSelector(_policy())
    with pytest.raises(ValueError, match="empty"):
        selector.select([], {})


def test_uncertainty_selector_missing_robustness_fails_closed():
    a = UncertaintyEvidence(
        trial_id="A",
        dimension_samples={"predictive": [0.5, 0.6]},
    )
    selector = UncertaintyAwareWinnerSelector(_policy())
    with pytest.raises(KeyError, match="robustness"):
        selector.select([a], {})


def test_uncertainty_config_validation():
    with pytest.raises(ValueError, match="equivalence_region"):
        UncertaintyConfig(equivalence_region=1.5)
    with pytest.raises(ValueError, match="confidence_level"):
        UncertaintyConfig(confidence_level=1.0)
    with pytest.raises(ValueError, match="minimum_meaningful"):
        UncertaintyConfig(minimum_meaningful_improvement=-0.1)


def test_uncertainty_evidence_validation():
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        UncertaintyEvidence(trial_id="A", dimension_samples={"x": [1.5]})
    with pytest.raises(ValueError, match="finite"):
        UncertaintyEvidence(trial_id="A", dimension_samples={"x": [float("nan")]})
    with pytest.raises(ValueError, match="non-empty"):
        UncertaintyEvidence(trial_id="A", dimension_samples={})


# ---------------------------------------------------------------------------
# DLIB-FO-003: DesirabilityPolicyRegistry session-freeze
# ---------------------------------------------------------------------------


def _context(**overrides):
    defaults = dict(
        market="A-share",
        asset_type="equity",
        frequency="daily",
        factor_family="PRICE_VOLUME",
        label_horizon="10d",
        universe_profile="all",
        liquidity_tier="liquid",
        consumer_profile="alpha",
    )
    defaults.update(overrides)
    return DesirabilityContext(**defaults)


def test_registry_freeze_session_returns_stable_policy_id():
    registry = DesirabilityPolicyRegistry()
    ctx = _context()
    pid1 = registry.freeze_session(ctx)
    pid2 = registry.freeze_session(ctx)
    # The same context always freezes to the same policy id (anchors cannot
    # drift mid-search).
    assert pid1 == pid2
    policy = registry.get(pid1)
    assert isinstance(policy, DesirabilityPolicy)
    assert policy.policy_id == pid1


def test_registry_different_contexts_different_policies():
    registry = DesirabilityPolicyRegistry()
    a_share = _context(market="A-share", factor_family="PRICE_VOLUME")
    us_fund = _context(market="US", factor_family="FUNDAMENTAL")
    pid_a = registry.freeze_session(a_share)
    # US daily FUNDAMENTAL has a different turnover desirability curve
    # (fundamental factors tolerate higher turnover than price-volume).
    registry.register(
        us_fund,
        maps={
            "turnover": {
                "direction": "decreasing",
                "anchors": [
                    (0.10, 1.00),
                    (0.30, 0.72),
                    (0.50, 0.42),
                    (0.70, 0.18),
                ],
            }
        },
        source="manual",
    )
    pid_us = registry.get_for_context(us_fund).policy_id
    assert pid_a != pid_us
    # A-share daily PRICE_VOLUME turnover desirability differs from US daily
    # FUNDAMENTAL.
    policy_a = registry.get(pid_a)
    policy_us = registry.get(pid_us)
    assert policy_a.score("turnover", 0.3) != policy_us.score("turnover", 0.3)


def test_registry_append_only_no_overwrite():
    registry = DesirabilityPolicyRegistry()
    ctx = _context()
    pid = registry.freeze_session(ctx)
    # Registering again with different maps must NOT overwrite the existing
    # policy (append-only).
    policy = registry.register(ctx, maps={"rank_ic": {"direction": "increasing", "anchors": [(0.0, 0.0), (1.0, 1.0)]}})
    assert policy.policy_id == pid
    assert registry.get(pid).score("rank_ic", 0.5) != 0.5  # original anchors kept


def test_registry_get_unknown_fails_closed():
    registry = DesirabilityPolicyRegistry()
    with pytest.raises(KeyError, match="no desirability policy"):
        registry.get("nonexistent")


def test_registry_get_for_context_unknown_fails_closed():
    registry = DesirabilityPolicyRegistry()
    with pytest.raises(KeyError, match="no desirability policy"):
        registry.get_for_context(_context())


def test_registry_context_requires_all_keys():
    with pytest.raises(ValueError, match="non-empty"):
        DesirabilityContext(
            market="", asset_type="equity", frequency="daily",
            factor_family="PRICE_VOLUME", label_horizon="10d",
            universe_profile="all", liquidity_tier="liquid",
            consumer_profile="alpha",
        )


def test_registry_policy_roundtrip():
    registry = DesirabilityPolicyRegistry()
    ctx = _context()
    pid = registry.freeze_session(ctx)
    policy = registry.get(pid)
    restored = DesirabilityPolicy.from_dict(policy.to_dict())
    assert restored.policy_id == policy.policy_id
    assert restored.context == policy.context
    assert restored.score("rank_ic", 0.03) == policy.score("rank_ic", 0.03)


# ---------------------------------------------------------------------------
# DLIB-FO-006: MultiplicityArtifact full-proposal-process traceability
# ---------------------------------------------------------------------------


def test_multiplicity_tracks_full_proposal_process():
    """1000 LLM proposals -> 300 parse fail, 100 duplicate, 400 valid
    evaluated, 200 failed eval -> the artifact tracks the FULL process, not
    just the 400 survivors."""
    artifact = MultiplicityArtifact(
        search_session_id="s1",
        total_proposals=1000,
        parse_failures=300,
        duplicates=100,
        valid_evaluated=400,
        failed_evaluations=200,
    )
    assert artifact.total_proposals == 1000
    assert artifact.parse_failures == 300
    assert artifact.duplicates == 100
    assert artifact.valid_evaluated == 400
    assert artifact.failed_evaluations == 200
    # The full hypothesis count is what a multiple-testing correction uses.
    assert artifact.total_proposals == (
        artifact.parse_failures
        + artifact.duplicates
        + artifact.valid_evaluated
        + artifact.failed_evaluations
    )


def test_multiplicity_rejects_overcount():
    """Outcome counts cannot exceed total_proposals (fail-closed)."""
    with pytest.raises(ValueError, match="exceed total_proposals"):
        MultiplicityArtifact(
            search_session_id="s1",
            total_proposals=10,
            parse_failures=5,
            duplicates=5,
            valid_evaluated=5,
            failed_evaluations=0,
        )


def test_multiplicity_roundtrip_and_tamper_detection():
    artifact = MultiplicityArtifact(
        search_session_id="s1",
        total_proposals=100,
        parse_failures=30,
        duplicates=10,
        valid_evaluated=40,
        failed_evaluations=20,
    )
    restored = MultiplicityArtifact.from_dict(artifact.to_dict())
    assert restored.content_hash == artifact.content_hash
    # Tampering with the payload must be detected.  Change a count while
    # keeping the accounted total <= total_proposals so the tamper is caught
    # by the content_hash check (not the overcount guard).
    data = artifact.to_dict()
    data["valid_evaluated"] = 39
    with pytest.raises(ValueError, match="content_hash"):
        MultiplicityArtifact.from_dict(data)


# ---------------------------------------------------------------------------
# DLIB-FO-007: TreatmentOptimizationResultArtifact
# ---------------------------------------------------------------------------


def _result_artifact(**overrides):
    defaults = dict(
        search_session_id="s1",
        source_factor_value_ref="sfv-1",
        raw_baseline_evidence_ref="raw-ev-1",
        factor_profile_ref="fp-1",
        treatment_search_space_ref="tss-1",
        transform_registry_snapshot_ref="trs-1",
        desirability_policy_ref="dp-1",
        winner_policy_ref="wp-1",
        split_plan_ref="sp-1",
        trial_ledger_ref="tl-1",
        all_trial_refs=("t1", "t2", "t3"),
        pareto_trial_refs=("t1", "t2"),
        multiplicity_ref="mult-1",
        selected_trial_ref="t2",
        uncertainty_evidence_ref="ue-1",
    )
    defaults.update(overrides)
    return TreatmentOptimizationResultArtifact(**defaults)


def test_result_artifact_binds_all_refs():
    artifact = _result_artifact()
    assert artifact.search_session_id == "s1"
    assert artifact.source_factor_value_ref == "sfv-1"
    assert artifact.raw_baseline_evidence_ref == "raw-ev-1"
    assert artifact.factor_profile_ref == "fp-1"
    assert artifact.treatment_search_space_ref == "tss-1"
    assert artifact.transform_registry_snapshot_ref == "trs-1"
    assert artifact.desirability_policy_ref == "dp-1"
    assert artifact.winner_policy_ref == "wp-1"
    assert artifact.split_plan_ref == "sp-1"
    assert artifact.trial_ledger_ref == "tl-1"
    assert artifact.all_trial_refs == ("t1", "t2", "t3")
    assert artifact.pareto_trial_refs == ("t1", "t2")
    assert artifact.multiplicity_ref == "mult-1"
    assert artifact.selected_trial_ref == "t2"
    assert artifact.uncertainty_evidence_ref == "ue-1"
    assert artifact.sealed_test_ref is None
    assert artifact.content_hash


def test_result_artifact_selected_must_be_in_all_trials():
    with pytest.raises(ValueError, match="selected_trial_ref"):
        _result_artifact(selected_trial_ref="t9")


def test_result_artifact_pareto_must_be_subset():
    with pytest.raises(ValueError, match="subset"):
        _result_artifact(pareto_trial_refs=("t1", "t9"))


def test_result_artifact_roundtrip_and_tamper():
    artifact = _result_artifact(sealed_test_ref="st-1")
    restored = TreatmentOptimizationResultArtifact.from_dict(artifact.to_dict())
    assert restored.content_hash == artifact.content_hash
    assert restored.sealed_test_ref == "st-1"
    data = artifact.to_dict()
    data["selected_trial_ref"] = "t3"
    with pytest.raises(ValueError, match="content_hash"):
        TreatmentOptimizationResultArtifact.from_dict(data)


def test_result_artifact_verify_fails_closed():
    artifact = _result_artifact()
    artifact.verify()  # no-op on a genuine artifact
    object.__setattr__(artifact, "_canonical", "tampered")
    with pytest.raises(ValueError, match="tampered"):
        artifact.verify()


# ---------------------------------------------------------------------------
# DLIB-FO-006: LLM only proposes, never judges success
# ---------------------------------------------------------------------------


def test_llm_proposal_never_judges_success():
    """The LLM produces a TreatmentRecipe proposal; it does NOT judge success.

    The proposal carries a mechanism hypothesis and expected signatures, but
    the success verdict comes from the FO Trial / Pareto / winner pipeline.
    """
    from factor_optimizer.contracts.candidate_mutation import CandidateMutation

    proposal = CandidateMutation(
        mutation_id="m1",
        mutation_spec_version="1.0.0",
        parent_factor_ids=["f1"],
        mutation_type="operator_swap",
        parameters={"op": "ewma"},
        mechanism_hypothesis="smoothing reduces turnover",
        expected_signatures=["lower_turnover", "higher_icir"],
        producer="llm_proposal_generator",
    )
    # The proposal is a hypothesis, not a verdict.
    assert proposal.mechanism_hypothesis
    assert proposal.expected_signatures
    # There is no success field on the proposal.
    assert not hasattr(proposal, "success")
    assert not hasattr(proposal, "winner")


# ---------------------------------------------------------------------------
# DLIB-FO-005: sealed-test authority cannot leak into search worker
# ---------------------------------------------------------------------------


def test_search_worker_cannot_forge_test_capability():
    """Adversarial: trying to forge a TestDataCapability inside a search
    worker must FAIL CLOSED (CapabilityForgeryError)."""
    from factor_optimizer.data_capabilities import TestDataCapability
    from factor_optimizer.errors import CapabilityForgeryError

    with pytest.raises(CapabilityForgeryError, match="forgery"):
        TestDataCapability([False, False, True])


def test_search_worker_cannot_pass_test_mask_to_train_capability():
    """Adversarial: passing a test mask to a train capability must fail
    closed (exact-subset authorization)."""
    from factor_optimizer.contracts.splits import SplitPlan
    from factor_optimizer.data_capabilities import build_search_capabilities

    plan = SplitPlan(
        "p1", [True, True, False], [False, False, True], [False, False, False], {}
    )
    caps = build_search_capabilities(
        plan,
        search_session_id="s1",
        dataset_identity="ds",
        provider_identity="prov",
    )
    train = caps[__import__("factor_optimizer.data_capabilities", fromlist=["DataScope"]).DataScope.TRAIN]
    # A plan that requests the test row (index 2) as a train row is NOT
    # authorized by the train capability.
    forged_plan = SplitPlan(
        "p1", [False, False, True], [False, False, False], [False, False, False], {}
    )
    assert train.can_evaluate(forged_plan) is False


def test_search_worker_cannot_reuse_sealed_handle():
    """Adversarial: reusing a sealed test handle must fail closed (one-shot)."""
    from factor_optimizer.contracts.search_budget import BudgetTracker, SearchBudget
    from factor_optimizer.contracts.splits import EvaluationProtocol, SplitPlan
    from factor_optimizer.contracts.trial import Trial, TrialStatus
    from factor_optimizer.search.runner import SearchConfig, SearchSession

    config = SearchConfig(
        budget=SearchBudget(max_trials=1, max_evaluations=1, max_cost_units=1.0),
        enable_multifidelity=False,
    )
    trial = Trial(trial_id="t1", mutation_id="m1", status=TrialStatus.PROPOSED)
    trial.update_status(
        TrialStatus.EVALUATED,
        evaluation_ref="eval-1",
        metadata={"score": 1.0, "fidelity": 4},
    )
    session = SearchSession(
        session_id="s1",
        config=config,
        budget_tracker=BudgetTracker(config.budget),
        trials=[trial],
        best_score=1.0,
        best_trial_id="t1",
        stop_reason="budget_exhausted",
    )
    session.finish("done")
    frozen = SplitPlan("test", [True, False], [False, False], [False, True], {})
    handle = session.freeze_for_sealed_test(frozen)
    # First consume succeeds.
    session.consume_sealed_test(
        handle, frozen, EvaluationProtocol(frozen, lambda t, f: {"rank_ic": 0.25})
    )
    # Reusing the same handle must fail closed (one-shot seal).
    with pytest.raises(ValueError, match="already been consumed"):
        session.consume_sealed_test(
            handle, frozen, EvaluationProtocol(frozen, lambda t, f: {"rank_ic": 0.5})
        )
