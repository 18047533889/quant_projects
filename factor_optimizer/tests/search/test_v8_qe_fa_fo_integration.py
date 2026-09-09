import hashlib
import json

import pytest
import numpy as np

from factor_assets.selection import (
    CandidateEvidence, DecisionProvider, DecisionRequest, DecisionStatus,
    MetricRule, RawJointMetricEvidence, Relationship, SelectionPolicySpec,
    UtilityDirection,
)
from factor_optimizer.contracts.evaluation_artifact import TrialEvaluationArtifact
from factor_optimizer.contracts.search_budget import SearchBudget
from factor_optimizer.contracts.splits import EvaluationProtocol, SplitPlan
from factor_optimizer.contracts.trial import Trial, TrialStatus
from factor_optimizer.search.runner import SearchConfig, SearchRunner
from factor_optimizer.search.tiered_evaluation import EvaluationTier, TieredEvaluationPolicy
from quant_evaluator.contracts.qualification import NumericalQualificationReceipt
from quant_evaluator.contracts.resampling import ResamplingPlan
from quant_evaluator.metrics.robustness import compute_joint_block_bootstrap
from factor_preprocess.representation.policy import decide_representation
from tests.search.test_runner import _integrity_evidence


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class _Resolver:
    def __init__(self, *receipts):
        self.receipts = {r.content_hash: r for r in receipts}

    def resolve(self, ref):
        return self.receipts[ref]


def _qualification(assertion="PASS"):
    return NumericalQualificationReceipt(
        source_tree_hash="source", implementation_hash="impl",
        route="qe.runtime", backend="cpu", parameter_domain_hash="domain",
        metric_instance_hash="metric-instance", test_run_ref="run-1",
        assertions={"T123": assertion},
    )


def _requirements():
    return dict(source_tree_hash="source", implementation_hash="impl",
                route="qe.runtime", backend="cpu", parameter_domain_hash="domain",
                metric_instance_hash="metric-instance")


def _policy():
    return SelectionPolicySpec(
        "selection-v8", "1", (MetricRule("alpha", UtilityDirection.HIGHER_IS_BETTER, 1, 0, 1),),
        {}, minimum_coverage=.8, noninferiority_margin=.01,
    )


def _raw(candidate_id, context, values, qualification):
    samples = tuple((({"value": value},),) for value in ())  # shape documented below
    samples = tuple((( (float(value),), ),) for value in values)
    semantic = {
        "context": context, "plan": "bootstrap-plan", "replicates": tuple(f"r{i}" for i in range(len(values))),
        "metrics": ("alpha",), "metric_units": ("ratio",), "window_ids": ("window",), "scenario_ids": ("scenario",),
        "samples": json.loads(json.dumps(samples)), "qualification": "scope:v8",
        "execution": ("source", "impl", "qe.runtime", "cpu", "domain", "metric-instance"),
    }
    evidence = RawJointMetricEvidence(
        f"qe:{candidate_id}", _digest(semantic), context, "bootstrap-plan",
        semantic["replicates"], ("alpha",), samples, "scope:v8",
        source_tree_hash="source", implementation_hash="impl", route="qe.runtime",
        backend="cpu", parameter_domain_hash="domain", metric_instance_hash="metric-instance",
        metric_units=("ratio",),
    )
    return CandidateEvidence(
        candidate_id, (evidence.evidence_id, qualification.content_hash), "FULL_VALIDATION",
        {}, {"alpha": sum(values) / len(values)}, 1.0, "scope:v8",
        qualification_ref=qualification.content_hash,
        qualification_requirements=_requirements(), raw_joint_metric_evidence=evidence,
    )


def _real_qe_candidates(context, qualification):
    """Execute QE's public common-plan joint bootstrap, then bind its provenance."""
    plan = ResamplingPlan(tuple(range(40)), "synthetic-test-clock", 4, 20, 917)
    raw_values = np.column_stack((
        np.linspace(.35, .65, 40),
        np.linspace(.55, .85, 40),
    ))
    qe_artifact = compute_joint_block_bootstrap(raw_values, plan, ("RAW", "VAR"))
    assert qe_artifact.provenance["resampling_plan_ref"] == plan.content_hash
    assert qe_artifact.provenance["replicate_ids"] == plan.replicate_ids
    qe_ref = _digest({"metric_id": qe_artifact.metric_id, "plan": plan.content_hash,
                      "samples": qe_artifact.samples.tolist()})
    candidates = []
    for column, candidate_id in enumerate(("RAW", "VAR")):
        values = tuple(float(value) for value in qe_artifact.samples[:, column])
        candidate = _raw(candidate_id, context, values, qualification)
        # Rebind the FA raw evidence to the exact QE-produced common plan and
        # replicate identities; no independently sampled candidate draws.
        old = candidate.raw_joint_metric_evidence
        samples = tuple((((value,),),) for value in values)
        semantic = {
            "context": context, "plan": plan.content_hash,
            "replicates": plan.replicate_ids, "metrics": ("alpha",), "metric_units": ("ratio",),
            "window_ids": ("window",), "scenario_ids": ("scenario",),
            "samples": json.loads(json.dumps(samples)), "qualification": "scope:v8",
            "execution": ("source", "impl", "qe.runtime", "cpu", "domain", "metric-instance"),
        }
        evidence = RawJointMetricEvidence(
            old.evidence_id, _digest(semantic), context, plan.content_hash,
            plan.replicate_ids, ("alpha",), samples, "scope:v8",
            source_tree_hash="source", implementation_hash="impl", route="qe.runtime",
            backend="cpu", parameter_domain_hash="domain", metric_instance_hash="metric-instance",
            metric_units=("ratio",),
        )
        candidates.append(CandidateEvidence(
            candidate_id, (qe_ref, evidence.content_hash,
                           qualification.content_hash), "FULL_VALIDATION", {},
            {"alpha": float(np.mean(values))}, 1.0, "scope:v8",
            qualification_ref=qualification.content_hash,
            qualification_requirements=_requirements(),
            raw_joint_metric_evidence=evidence,
        ))
    return qe_artifact, tuple(candidates)


def _request(
    policy, candidates, *, request_id="request", baseline=None,
    original_raw=None, fidelity="FULL_VALIDATION", context="context",
):
    adjusted = tuple(
        CandidateEvidence(
            c.candidate_id, c.evidence_refs, fidelity, c.dimensions, c.metrics, c.coverage,
            c.qualification_scope, qualification_ref=c.qualification_ref,
            qualification_requirements=c.qualification_requirements,
            raw_joint_metric_evidence=c.raw_joint_metric_evidence,
        ) for c in candidates
    )
    bound_baseline = baseline or adjusted[0].candidate_id
    return DecisionRequest(
        request_id, policy.policy_id, policy.content_hash, context,
        "variant-selection", "FINAL", bound_baseline, adjusted,
        fidelity, "family:sealed", original_raw_ref=original_raw,
    )


def test_real_qe_qualification_and_fa_receipt_drive_fo_tier_and_final_paths():
    q = _qualification(); policy = _policy(); provider = DecisionProvider(policy, _Resolver(q))
    context = "context"; candidate = _raw("VAR", context, (.7, .8, .9), q)
    tiers = TieredEvaluationPolicy((EvaluationTier("CHEAP", 1, 0), EvaluationTier("FULL_VALIDATION", 2, 4)))
    config = SearchConfig(
        budget=SearchBudget(max_trials=1, max_evaluations=2, max_cost_units=2),
        tiered_evaluation=tiers, evaluation_cost_units=1, screening_only=False,
    )
    calls = []
    def evaluate(trial, fidelity):
        calls.append(fidelity)
        return TrialEvaluationArtifact(
            trial.trial_id, [{"name": "score", "value": 999.0}], "score", "search", fidelity,
            f"qe:evaluation:{fidelity}", 1, treatment_integrity_evidence=_integrity_evidence(trial.trial_id),
        )
    runner = SearchRunner(
        config, lambda: Trial("VAR", "m", status=TrialStatus.PROPOSED),
        EvaluationProtocol(SplitPlan("search", [True], [False], [False], {}), evaluate),
        decision_provider=provider,
        decision_request_factory=lambda session: _request(policy, (candidate,), context=context),
        stage_decision_request_factory=lambda session, trial, artifact, job: _request(
            policy, (candidate,), request_id=f"stage:{job.tier_name}", fidelity=job.tier_name, context=context
        ),
    )
    session = runner.run("integration")
    assert calls == [0, 4]
    assert session.best_trial_id == "VAR"
    assert session.best_score != 999.0
    assert session.selection_decision_id.startswith("decision:")


def test_missing_qe_qualification_waits_with_no_selectable_utility():
    q = _qualification(); policy = _policy(); candidate = _raw("VAR", "context", (.8, .9), q)
    receipt = DecisionProvider(policy).decide(_request(policy, (candidate,)))
    assert receipt.status is DecisionStatus.WAIT
    assert receipt.point_utility["VAR"] is None


def test_higher_point_but_inconclusive_variant_does_not_replace_raw():
    q = _qualification(); policy = _policy(); provider = DecisionProvider(policy, _Resolver(q))
    raw = _raw("RAW", "context", (.5, .5), q)
    variant = _raw("VAR", "context", (.9, .4), q)
    receipt = provider.decide(_request(policy, (raw, variant), baseline="RAW"))
    assert receipt.point_utility["VAR"] > receipt.point_utility["RAW"]
    assert receipt.relationship["VAR"] is Relationship.INCONCLUSIVE
    assert receipt.winner_id == "RAW"


def test_receipt_context_or_candidate_rebinding_is_rejected_by_fo():
    from dataclasses import replace
    from factor_optimizer.ports.factor_intelligence import require_bound_decision_receipt
    q = _qualification(); policy = _policy(); provider = DecisionProvider(policy, _Resolver(q))
    request = _request(policy, (_raw("VAR", "context", (.8, .9), q),))
    receipt = provider.decide(request)
    with pytest.raises(ValueError, match="comparison_context_hash"):
        require_bound_decision_receipt(request, replace(receipt, comparison_context_hash="other"))
    with pytest.raises(ValueError, match="candidate_set_hash"):
        require_bound_decision_receipt(request, replace(receipt, candidate_set_hash="other"))


def test_actual_qe_joint_bootstrap_yields_identical_fa_fo_fp_decision_identity():
    # This receipt is intentionally scoped to a synthetic test execution
    # domain. It proves connector behavior, not production certification.
    q = _qualification(); policy = _policy(); provider = DecisionProvider(policy, _Resolver(q))
    qe_artifact, candidates = _real_qe_candidates("context", q)
    request = _request(policy, candidates, request_id="shared", baseline="RAW")
    direct = provider.decide(request)
    fp = decide_representation(provider, request, "VAR")

    config = SearchConfig(
        budget=SearchBudget(max_trials=1, max_evaluations=1, max_cost_units=1),
        enable_multifidelity=False, screening_only=False,
    )
    def evaluate(trial, fidelity):
        return TrialEvaluationArtifact(
            trial.trial_id, [{"name": "score", "value": -123.0}], "score", "search", fidelity,
            _digest({"metric_id": qe_artifact.metric_id,
                     "plan": qe_artifact.provenance["resampling_plan_ref"],
                     "samples": qe_artifact.samples.tolist()}), 1,
            treatment_integrity_evidence=_integrity_evidence(trial.trial_id),
        )
    session = SearchRunner(
        config, lambda: Trial(direct.winner_id, "m", status=TrialStatus.PROPOSED),
        EvaluationProtocol(SplitPlan("search", [True], [False], [False], {}), evaluate),
        decision_provider=provider, decision_request_factory=lambda session: request,
    ).run("shared-provider")
    assert fp.decision_id == direct.decision_id == session.selection_decision_id
    assert fp.content_hash == direct.content_hash == session.selection_decision_hash
