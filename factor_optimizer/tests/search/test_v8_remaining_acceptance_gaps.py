from concurrent.futures import ThreadPoolExecutor

import pytest

from factor_assets.selection import DecisionProvider
from factor_optimizer.contracts.campaign_store import CampaignStateError, SQLiteCampaignStore
from factor_optimizer.contracts.search_budget import BudgetTracker, SearchBudget
from factor_optimizer.contracts.evaluation_artifact import TrialEvaluationArtifact
from factor_optimizer.contracts.splits import EvaluationProtocol, SplitPlan
from factor_optimizer.contracts.trial import Trial, TrialStatus
from factor_optimizer.search.runner import SearchRunner
from tests.search.test_runner import _integrity_evidence
from factor_optimizer.search.plateau import MultiObjectivePlateauDetector
from factor_optimizer.search.runner import SearchConfig, SearchSession
from quant_evaluator.contracts.hypothesis_family import HypothesisFamilyArtifact
from quant_evaluator.metrics.multiple_testing import compute_family_correction
from tests.search.test_v8_qe_fa_fo_integration import (
    _Resolver, _qualification, _policy, _raw, _request,
)


def test_finish_fault_before_seal_is_not_half_committed(monkeypatch):
    config = SearchConfig(budget=SearchBudget())
    session = SearchSession("fault", config, BudgetTracker(config.budget))
    original = session.ledger.verify_chain
    monkeypatch.setattr(session.ledger, "verify_chain", lambda: (_ for _ in ()).throw(ValueError("fault")))
    with pytest.raises(ValueError, match="fault"):
        session.finish("done")
    assert session.finished_at is None and session.stop_reason is None
    assert session.multiplicity_artifact is None and not session.ledger.sealed
    monkeypatch.setattr(session.ledger, "verify_chain", original)
    session.finish("recovered")
    assert session.is_finished() and session.ledger.sealed
    assert session.multiplicity_artifact is not None


def test_finish_fault_after_seal_recovers_without_half_finished_state(monkeypatch):
    config = SearchConfig(budget=SearchBudget())
    session = SearchSession("seal-fault", config, BudgetTracker(config.budget))
    real_seal = session.ledger.seal
    def seal_then_fault():
        real_seal()
        raise RuntimeError("post-seal persistence fault")
    monkeypatch.setattr(session.ledger, "seal", seal_then_fault)
    with pytest.raises(RuntimeError, match="post-seal"):
        session.finish("done")
    assert session.ledger.sealed
    assert session.finished_at is None and session.stop_reason is None
    assert session.multiplicity_artifact is None
    monkeypatch.setattr(session.ledger, "seal", real_seal)
    session.finish("recovered")
    assert session.is_finished() and session.multiplicity_artifact is not None


def test_concurrent_commit_cancel_same_started_reservation_settles_once(tmp_path):
    store = SQLiteCampaignStore(tmp_path / "race.sqlite3")
    store.create_campaign("c", max_evaluations=1, max_cost=10)
    assert store.reserve("c", "attempt", 7)
    store.start("c", "attempt")
    def settle():
        try: store.settle("c", "attempt", 5); return "settled"
        except CampaignStateError: return "rejected"
    def cancel():
        try: store.release("c", "attempt"); return "released"
        except CampaignStateError: return "rejected"
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda fn: fn(), (settle, cancel)))
    assert sorted(outcomes) == ["rejected", "settled"]
    state = store.budget_state("c")
    assert state["evaluations_used"] == 1 and state["cost_used"] == 5
    assert state["evaluations_reserved"] == 0 and state["cost_reserved"] == 0


def test_missing_worker_restore_stays_reserved_and_cannot_enlarge_budget(tmp_path):
    path = tmp_path / "orphan.sqlite3"
    store = SQLiteCampaignStore(path)
    store.create_campaign("c", max_evaluations=1, max_cost=10)
    assert store.reserve("c", "orphan-worker", 8)
    store.start("c", "orphan-worker")
    restarted = SQLiteCampaignStore(path)
    assert not restarted.reserve("c", "replacement", 1)
    state = restarted.budget_state("c")
    assert state["evaluations_reserved"] == 1 and state["cost_reserved"] == 8


def test_t166_failure_after_ninety_percent_progress_conservatively_charges_reservation(tmp_path):
    path = tmp_path / "ninety-percent.sqlite3"
    progress = []

    def fail_late(trial, fidelity):
        for unit in range(9):
            progress.append(unit)
        raise RuntimeError("worker failed after 9 of 10 units")

    config = SearchConfig(
        budget=SearchBudget(max_trials=1, max_evaluations=1, max_cost_units=10),
        enable_multifidelity=False, evaluation_cost_units=10,
        durable_campaign_store_path=str(path), screening_only=True,
    )
    session = SearchRunner(
        config,
        lambda: Trial("late-failure", "mutation", status=TrialStatus.PROPOSED),
        EvaluationProtocol(
            SplitPlan("search", [True], [False], [False], {}), fail_late,
        ),
    ).run("late-failure-campaign")

    state = SQLiteCampaignStore(path).budget_state("late-failure-campaign")
    assert len(progress) == 9
    assert state["evaluations_used"] == 1
    assert state["cost_used"] == 10
    assert state["evaluations_reserved"] == 0
    assert session.budget_tracker.remaining_cost() == 0


def test_persisted_orphan_reservation_checkpoint_never_restores_as_free_budget():
    tracker = BudgetTracker(SearchBudget(max_evaluations=1, max_cost_units=10))
    assert tracker.reserve_evaluation(8, attempt_id="orphan-worker")
    tracker.start_evaluation("orphan-worker")

    checkpoint = tracker.to_dict()
    restored = BudgetTracker.from_dict(checkpoint)
    assert restored.has_reservation("orphan-worker")
    assert restored.remaining_evaluations() == 0
    assert restored.remaining_cost() == 2
    assert not restored.reserve_evaluation(1, attempt_id="replacement")

    # A checkpoint cannot erase the orphan identity while retaining aggregate
    # reserved counters; restore rejects the inconsistency instead of creating
    # apparently free evaluation capacity.
    corrupted = dict(checkpoint)
    corrupted["reservations"] = {}
    with pytest.raises(ValueError, match="reserved evaluation count"):
        BudgetTracker.from_dict(corrupted)


def _member(i, p):
    return dict(hypothesis_id=f"h{i}", effective_spec_hash=f"s{i}",
                evaluation_intent_hash=f"i{i}", horizon=1, label_ref="label",
                direction="higher", status="COMPUTED", pvalue=p,
                pvalue_ref=f"p:{i}")


def test_same_gpu_repeated_batches_match_one_batch_full_family_correction():
    members = tuple(_member(i, p) for i, p in enumerate((.001, .01, .03, .2, .4, .8)))
    fixed_ledger_head, fixed_ledger_count = "head", 12

    def correction_after_same_gpu_batches(batches):
        # Batch boundaries are execution details. Aggregate p-values by the
        # preregistered member identity, then correct exactly once over the
        # complete family in canonical ledger order.
        observed = {}
        for batch in batches:
            for member in batch:
                observed[member["hypothesis_id"]] = member
        complete = tuple(observed[m["hypothesis_id"]] for m in members)
        family = HypothesisFamilyArtifact(
            "f", "c", "p", "ctx", complete,
            fixed_ledger_head, fixed_ledger_count,
        )
        return family, compute_family_correction(family)

    one_batch_family, one_batch = correction_after_same_gpu_batches((members,))
    repeated_family, repeated = correction_after_same_gpu_batches(
        (members[:1], members[1:4], members[4:])
    )
    assert one_batch_family.content_hash == repeated_family.content_hash
    assert one_batch == repeated


def test_t28_parent_split_branches_and_recombined_candidate_share_one_complete_family(tmp_path):
    path = tmp_path / "parent-split-recombined.sqlite3"
    candidate_ids = ("PARENT", "SPLIT_LEFT", "SPLIT_RIGHT", "RECOMBINED")
    proposals = iter(
        Trial(candidate_id, f"mutation:{candidate_id}", status=TrialStatus.PROPOSED)
        for candidate_id in candidate_ids
    )

    def evaluate(trial, fidelity):
        return {
            "evaluation_id": f"qe:{trial.trial_id}",
            "score": float(candidate_ids.index(trial.trial_id) + 1),
            "cost": 1.0,
            "treatment_integrity_evidence": _integrity_evidence(trial.trial_id),
        }

    config = SearchConfig(
        budget=SearchBudget(max_trials=4, max_evaluations=4, max_cost_units=4),
        enable_multifidelity=False, durable_campaign_store_path=str(path),
        screening_only=True,
    )
    session = SearchRunner(
        config,
        lambda: next(proposals),
        EvaluationProtocol(SplitPlan("search", [True], [False], [False], {}), evaluate),
    ).run("lineage-campaign")
    assert tuple(trial.trial_id for trial in session.trials) == candidate_ids
    assert all(trial.status is TrialStatus.EVALUATED for trial in session.trials)

    store = SQLiteCampaignStore(path)
    members = []
    for index, candidate_id in enumerate(candidate_ids):
        store.record_hypothesis_attempt(
            campaign_id="lineage-campaign", proposal_id=candidate_id,
            effective_spec_hash=f"spec:{candidate_id}",
            evaluation_intent_hash="intent:shared-confirmation",
            horizon=1, executed=True, has_pvalue=True,
        )
        members.append(_member(index, 0.01 * (index + 1)) | {
            "hypothesis_id": candidate_id,
            "effective_spec_hash": f"spec:{candidate_id}",
            "evaluation_intent_hash": "intent:shared-confirmation",
        })
    family = HypothesisFamilyArtifact(
        "lineage-family", "lineage-campaign", "policy", "context",
        tuple(members), session.ledger.sealed_head_hash,
        session.ledger.sealed_entry_count,
    )
    assert store.bind_hypothesis_family(family, session.ledger) == family.content_hash
    summary = SQLiteCampaignStore(path).hypothesis_family_summary("lineage-campaign")
    assert summary["proposal_count"] == 4
    assert summary["executed_trial_count"] == 4
    assert summary["pvalue_count"] == 4


def test_t112_cross_worker_retry_preserves_validation_without_duplicate_hypothesis(tmp_path):
    store = SQLiteCampaignStore(tmp_path / "cross-worker.sqlite3")
    shared = dict(
        campaign_id="retry-campaign", evaluation_intent_hash="intent",
        horizon=5, effective_spec_hash="same-effective-hypothesis",
        executed=True,
    )
    store.record_hypothesis_attempt(
        proposal_id="worker-a:attempt-1", has_pvalue=True, **shared,
    )
    SQLiteCampaignStore(store.path).record_hypothesis_attempt(
        proposal_id="worker-b:attempt-2", has_pvalue=False, **shared,
    )
    summary = SQLiteCampaignStore(store.path).hypothesis_family_summary("retry-campaign")
    assert summary == {
        "proposal_count": 2,
        "executed_trial_count": 2,
        "unique_effective_spec_count": 1,
        "pvalue_count": 1,
        "horizon_hypothesis_count": 1,
    }

    from factor_optimizer.contracts.trial_ledger import TrialLedger
    ledger = TrialLedger()
    ledger.append("EVALUATED", trial_id="worker-a:attempt-1")
    ledger.append("EVALUATION_FAILED", trial_id="worker-b:attempt-2", failure_reason="worker lost")
    ledger.seal()
    common = {
        "effective_spec_hash": "same-effective-hypothesis",
        "evaluation_intent_hash": "intent", "horizon": 5,
        "label_ref": "label", "direction": "higher",
    }
    family = HypothesisFamilyArtifact(
        "retry-family", "retry-campaign", "policy", "context",
        (
            common | {"hypothesis_id": "worker-a:attempt-1", "status": "COMPUTED",
                      "pvalue": 0.03, "pvalue_ref": "qe:valid"},
            common | {"hypothesis_id": "worker-b:attempt-2", "status": "FAILED"},
        ),
        ledger.sealed_head_hash, ledger.sealed_entry_count,
    )
    assert SQLiteCampaignStore(store.path).bind_hypothesis_family(family, ledger) == family.content_hash
    restored = SQLiteCampaignStore(store.path).hypothesis_family_summary("retry-campaign")
    assert restored["unique_effective_spec_count"] == 1
    assert restored["pvalue_count"] == 1


def _hypervolume_2d(points, reference=(0.0, 0.0)):
    frontier = sorted(points)
    area = 0.0; prior_x = reference[0]
    for x, y in frontier:
        area += (x - prior_x) * max(0.0, y - reference[1])
        prior_x = x
    return area


def test_stronger_point_replacing_two_old_points_is_hypervolume_progress():
    old = _hypervolume_2d(((1.0, 3.0), (3.0, 1.0)))
    stronger = _hypervolume_2d(((4.0, 4.0),))
    assert stronger > old
    detector = MultiObjectivePlateauDetector(window_size=2)
    detector.add_hypervolume(old); detector.add_hypervolume(stronger)
    assert not detector.is_plateau()
    detector.add_hypervolume(stronger)
    assert detector.is_plateau()


def test_runner_does_not_stop_when_stronger_point_replaces_two_frontier_points():
    old = _hypervolume_2d(((1.0, 3.0), (3.0, 1.0)))
    stronger = _hypervolume_2d(((4.0, 4.0),))
    detector = MultiObjectivePlateauDetector(window_size=2)
    class FrontierTrace:
        def __init__(self): self.called = False
        def __call__(self, _scores):
            assert not self.called
            detector.add_hypervolume(old)
            detector.add_hypervolume(stronger)
            self.called = True
            return detector.is_plateau()
    counter = {"n": 0}
    def proposal():
        counter["n"] += 1
        return Trial(f"t{counter['n']}", "m", status=TrialStatus.PROPOSED)
    def evaluate(trial, fidelity):
        return TrialEvaluationArtifact(
            trial.trial_id, [{"name": "score", "value": float(counter["n"])}],
            "score", "search", fidelity, f"qe:{trial.trial_id}", 1,
            treatment_integrity_evidence=_integrity_evidence(trial.trial_id),
        )
    config = SearchConfig(
        budget=SearchBudget(max_trials=3, max_evaluations=3, max_cost_units=3),
        enable_multifidelity=False, evaluation_cost_units=1,
        plateau_window=2, screening_only=True,
    )
    session = SearchRunner(
        config, proposal,
        EvaluationProtocol(SplitPlan("search", [True], [False], [False], {}), evaluate),
        plateau_detector=FrontierTrace(),
    ).run("hypervolume-trace")
    assert len(session.trials) == 3
    assert session.stop_reason != "plateau_detected"


def test_point_dominance_cannot_delete_conservative_utility_winner():
    q = _qualification(); policy = _policy(); provider = DecisionProvider(policy, _Resolver(q))
    volatile = _raw("VOLATILE", "context", (.99, .01), q)
    stable = _raw("STABLE", "context", (.45, .45), q)
    receipt = provider.decide(_request(policy, (volatile, stable), baseline="STABLE"))
    assert receipt.point_utility["VOLATILE"] > receipt.point_utility["STABLE"]
    assert receipt.conservative_utility["VOLATILE"] < receipt.conservative_utility["STABLE"]
    assert receipt.winner_id == "STABLE"
