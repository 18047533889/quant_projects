"""Adversarial regression tests for the 2026-08-19 sealed-test hardening.

Pins the audit repairs: consume-time mask equality (not just split_id),
zip-truncation in disjointness, boolean score rejection, NaN best-score
lockout, and proposal_fn failure visibility.
"""

import pytest
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import replace
from datetime import timedelta

from factor_optimizer.contracts.search_budget import (
    BudgetTracker,
    SearchBudget,
)
from factor_optimizer.contracts.objective import ObjectiveSpec
from factor_optimizer.contracts.splits import (
    EvaluationProtocol,
    SealedTestEvaluationOutcome,
    SplitPlan,
)
from factor_optimizer.contracts.trial import Trial, TrialStatus
import numpy as np

from factor_optimizer.search.runner import (
    SearchConfig,
    SearchRunner,
    SearchSession,
    SealedTestExecutor,
    _sealed_test_evaluation_ref,
    _split_semantics_hash,
    _sealed_test_disjoint,
)
from factor_optimizer.contracts.campaign_store import SQLiteCampaignStore
from factor_optimizer.data_capabilities import TestAuthorityBroker, TestStoreRef


class _CountingOpaqueStore:
    def __init__(self, payload):
        self.payload = payload
        self.read_count = 0

    def read(self):
        self.read_count += 1
        return self.payload


def _typed_outcome(session, plan, metrics, **overrides):
    spec = session.sealed_execution_spec
    values = {
        "metrics": metrics,
        "execution_spec_hash": spec.spec_hash,
        "dataset_identity": "dataset:v1",
        "split_id": plan.split_id,
        "factor_identity": spec.metadata["execution_spec"]["model_input_ref"],
        "time_identity": spec.sealed_split_hash,
        "cost_identity": spec.metadata["execution_spec"]["cost_model_ref"],
        "test_evidence_ref": _sealed_test_evaluation_ref(session, spec, plan),
    }
    values.update(overrides)
    return SealedTestEvaluationOutcome(**values)


def _protocol(fn, split_id="split", masks=([True], [False], [False])):
    return EvaluationProtocol(SplitPlan(split_id, *masks, {}), fn)


def _finished_session():
    config = SearchConfig(
        budget=SearchBudget(max_trials=1, max_evaluations=1, max_cost_units=1.0),
        enable_multifidelity=False,
        objective_spec=ObjectiveSpec("rank_ic", "maximize"),
    )
    trial = Trial(trial_id="t1", mutation_id="m1", status=TrialStatus.PROPOSED)
    trial.update_status(
        TrialStatus.EVALUATED,
        evaluation_ref="eval-1",
        metadata={
            "score": 1.0,
            "fidelity": 4,
            "execution_spec": {
                "canonical_recipe": {"op": "identity"},
                "effective_parameters": {},
                "fit_state_ref": "fit:none",
                "data_snapshot_ref": "snapshot:selection",
                "operator_versions": {"identity": "1"},
                "code_version": "code:v1",
                "orientation": 1,
                "model_input_ref": "factor:f1@v1",
                "cost_model_ref": "cost:v1",
                "required_test_metrics": ("rank_ic",),
            },
        },
    )
    return SearchSession(
        session_id="s1",
        config=config,
        budget_tracker=BudgetTracker(config.budget),
        trials=[trial],
        best_score=1.0,
        best_trial_id="t1",
        selection_decision_id="decision:v8-fixture",
        selection_decision_hash="decision-hash:v8-fixture",
        selection_request_hash="request-hash:v8-fixture",
        stop_reason="budget_exhausted",
    )


def _finish(session):
    if not session.is_finished():
        session.finish("manual_stop")
    return session


class TestSealedMaskEquality:
    @pytest.mark.parametrize("field,value", [
        ("search_session_id", "forged-session"),
        ("split_id", "forged-split"),
    ])
    def test_forged_handle_identity_is_rejected_before_retired_path(self, field, value):
        session = _finish(_finished_session())
        frozen = SplitPlan("test", [True, False], [False, False], [False, True], {})
        handle = replace(session.freeze_for_sealed_test(frozen), **{field: value})
        with pytest.raises(ValueError, match="does not match frozen session"):
            session.consume_sealed_test(
                handle, frozen, EvaluationProtocol(frozen, lambda t, f: pytest.fail("read"))
            )

    def test_forged_handle_timestamp_is_rejected_before_any_read(self):
        session = _finish(_finished_session())
        frozen = SplitPlan("test", [True, False], [False, False], [False, True], {})
        handle = session.freeze_for_sealed_test(frozen)
        forged = replace(handle, frozen_at=handle.frozen_at + timedelta(seconds=1))
        with pytest.raises(ValueError, match="does not match frozen session"):
            session.consume_sealed_test(
                forged, frozen, EvaluationProtocol(frozen, lambda t, f: pytest.fail("read"))
            )
    def test_same_split_id_different_masks_is_rejected(self):
        # Pre-fix: only split_id was compared, so a plan that moved the
        # sealed segment but kept the ID would silently pass.
        session = _finish(_finished_session())
        frozen = SplitPlan("test", [True, False], [False, False], [False, True], {})
        handle = session.freeze_for_sealed_test(frozen)
        forged = SplitPlan("test", [False, False], [False, False], [True, False], {})
        with pytest.raises(ValueError, match="masks differ from the frozen plan"):
            session.consume_sealed_test(
                handle, forged, EvaluationProtocol(forged, lambda t, f: {})
            )

    def test_evaluator_bound_to_different_masks_is_rejected(self):
        session = _finish(_finished_session())
        frozen = SplitPlan("test", [True, False], [False, False], [False, True], {})
        handle = session.freeze_for_sealed_test(frozen)
        other = SplitPlan("test", [False, False], [False, False], [False, True], {})
        with pytest.raises(ValueError, match="evaluator protocol masks must match"):
            session.consume_sealed_test(
                handle, frozen, EvaluationProtocol(other, lambda t, f: {})
            )

    def test_masks_survive_checkpoint_roundtrip(self):
        session = _finish(_finished_session())
        frozen = SplitPlan("test", [True, False], [False, False], [False, True], {})
        handle = session.freeze_for_sealed_test(frozen)
        restored = SearchSession.from_dict(session.to_dict())
        assert dict(restored.sealed_test_masks) == {
            "train": (True, False),
            "validation": (False, False),
            "test": (False, True),
        }
        with pytest.raises(ValueError, match="direct consume_sealed_test is retired"):
            restored.consume_sealed_test(
                handle, frozen, EvaluationProtocol(frozen, lambda t, f: {"rank_ic": 0.25})
            )

    def test_frozen_checkpoint_without_masks_is_rejected(self):
        session = _finish(_finished_session())
        frozen = SplitPlan("test", [True, False], [False, False], [False, True], {})
        session.freeze_for_sealed_test(frozen)
        data = session.to_dict()
        del data["sealed_test_masks"]
        with pytest.raises(ValueError, match="sealed_test_masks"):
            SearchSession.from_dict(data)

    def test_pinned_masks_are_immutable_in_place(self):
        # __setattr__ guards assignment, but an in-place dict mutation
        # would otherwise rewrite what the seal compares against.
        session = _finish(_finished_session())
        frozen = SplitPlan("test", [True, False], [False, False], [False, True], {})
        session.freeze_for_sealed_test(frozen)
        with pytest.raises(TypeError):
            session.sealed_test_masks["test"] = (True, False)
        with pytest.raises((TypeError, AttributeError)):
            session.sealed_test_masks["test"].append(True)  # type: ignore[union-attr]
        assert session.sealed_test_masks["test"] == (False, True)

    def test_restored_session_pins_deep_copy_and_blocks_seal_reset(self):
        session = _finish(_finished_session())
        frozen = SplitPlan("test", [True, False], [False, False], [False, True], {})
        handle = session.freeze_for_sealed_test(frozen)
        data = session.to_dict()
        restored = SearchSession.from_dict(data)
        assert restored.sealed_test_masks["test"] == (False, True)
        # Mutating the caller-held checkpoint dict after restoration must
        # not reach the restored session's pin (from_dict deep-copies).
        data["sealed_test_masks"]["test"] = [True, False]
        assert restored.sealed_test_masks["test"] == (False, True)
        with pytest.raises(TypeError):
            restored.sealed_test_masks["test"] = (True, False)
        with pytest.raises(ValueError, match="direct consume_sealed_test is retired"):
            restored.consume_sealed_test(
                handle, frozen, EvaluationProtocol(frozen, lambda t, f: {"rank_ic": 0.5})
            )
        with pytest.raises(ValueError, match="immutable after freeze"):
            restored.sealed_test_consumed = True
        with pytest.raises(ValueError, match="already frozen"):
            restored.freeze_for_sealed_test(frozen)


class TestV7SealedReservationAndExecutionIdentity:
    def _frozen(self):
        session = _finish(_finished_session())
        plan = SplitPlan("test", [True, False], [False, False], [False, True], {})
        return session, plan, session.freeze_for_sealed_test(plan)

    def test_direct_consume_is_retired_even_from_preconsume_checkpoint(self):
        session, plan, handle = self._frozen()
        reads = []
        protocol = EvaluationProtocol(plan, lambda spec, fidelity: reads.append(spec) or {})
        with pytest.raises(ValueError, match="direct consume_sealed_test is retired"):
            session.consume_sealed_test(handle, plan, protocol)
        assert reads == []
        restored = SearchSession.from_dict(session.to_dict())
        with pytest.raises(ValueError, match="direct consume_sealed_test is retired"):
            restored.consume_sealed_test(handle, plan, protocol)
        assert reads == []

    def test_concurrent_direct_consumers_obtain_no_read(self):
        session, plan, handle = self._frozen()
        reads = []

        def evaluate(spec, fidelity):
            reads.append(spec.spec_hash)
            return {"rank_ic": 0.2}

        protocol = EvaluationProtocol(plan, evaluate)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(session.consume_sealed_test, handle, plan, protocol) for _ in range(2)]
            for future in futures:
                with pytest.raises(ValueError, match="direct consume_sealed_test is retired"):
                    future.result(timeout=2)
        assert reads == []

    def test_frozen_nested_recipe_is_executed_and_test_ref_is_unique(self):
        session, plan, handle = self._frozen()
        winner = session.trials[0]
        winner.metadata["params"] = {"window": 80}
        winner.mutation_id = "mutated-after-freeze"
        observed = {}

        def evaluate(spec, fidelity):
            observed["mutation_id"] = spec.mutation_id
            observed["window"] = spec.metadata.get("params", {}).get("window")
            return {"rank_ic": 0.3}

        with pytest.raises(ValueError, match="direct consume_sealed_test is retired"):
            session.consume_sealed_test(handle, plan, EvaluationProtocol(plan, evaluate))
        assert observed == {}

    def test_certification_requires_durable_authority(self):
        session, plan, handle = self._frozen()
        runner = SearchRunner(
            session.config,
            lambda: Trial("unused", "unused", TrialStatus.PROPOSED),
            EvaluationProtocol(plan, lambda spec, fidelity, data: {"rank_ic": 0.1}),
        )
        broker = TestAuthorityBroker(
            dataset_identity="dataset:v1", provider_identity="test-authority",
            search_session_id=session.session_id, split_id=plan.split_id,
        )
        with pytest.raises(ValueError, match="durable campaign-store"):
            SealedTestExecutor(runner, broker).evaluate_sealed_test(
                session, handle, plan
            )

    def test_certification_rejects_incomplete_frozen_execution_spec(self):
        session = _finish(_finished_session())
        session.trials[0].metadata.pop("execution_spec")
        plan = SplitPlan("test", [True, False], [False, False], [False, True], {})
        handle = session.freeze_for_sealed_test(plan)
        runner = SearchRunner(
            session.config,
            lambda: Trial("unused", "unused", TrialStatus.PROPOSED),
            EvaluationProtocol(plan, lambda spec, fidelity, data: pytest.fail("read")),
        )
        broker = TestAuthorityBroker(
            dataset_identity="dataset:v1", provider_identity="test-authority",
            search_session_id=session.session_id, split_id=plan.split_id,
        )
        with pytest.raises(ValueError, match="metadata.execution_spec"):
            SealedTestExecutor(runner, broker).evaluate_sealed_test(
                session, handle, plan
            )

    def test_preconsume_checkpoint_cannot_double_read_across_process_analogs(self, tmp_path):
        session, plan, handle = self._frozen()
        checkpoint = session.to_dict()
        left = SearchSession.from_dict(checkpoint)
        right = SearchSession.from_dict(checkpoint)
        left.trials[0].mutation_id = "tampered-after-freeze"
        left.trials[0].metadata["execution_spec"]["orientation"] = -1
        store = SQLiteCampaignStore(tmp_path / "sealed.sqlite3")
        store.create_campaign(session.session_id, max_evaluations=2, max_cost=2.0)
        physical = _CountingOpaqueStore({"physical_test": "holdout-v1"})
        entered = threading.Event()
        release = threading.Event()
        reads = []

        def evaluate(spec, fidelity, data):
            reads.append(spec.spec_hash)
            entered.set()
            release.wait(timeout=2)
            return _typed_outcome(session, plan, {"rank_ic": 0.4})

        runner = SearchRunner(
            session.config,
            lambda: Trial("unused", "unused", TrialStatus.PROPOSED),
            EvaluationProtocol(plan, evaluate),
        )

        def broker():
            authority = TestAuthorityBroker(
                dataset_identity="dataset:v1", provider_identity="test-authority",
                search_session_id=session.session_id, split_id=plan.split_id,
                campaign_store=SQLiteCampaignStore(store.path),
                campaign_id=session.session_id,
                candidate_set_hash=session.sealed_execution_spec.spec_hash,
                profile_hash=_split_semantics_hash(plan),
            )
            authority.attach_store_ref(
                TestStoreRef(physical, dataset_identity="dataset:v1")
            )
            return authority

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                SealedTestExecutor(runner, broker()).evaluate_sealed_test,
                left, handle, plan,
            )
            assert entered.wait(timeout=2)
            second = pool.submit(
                SealedTestExecutor(runner, broker()).evaluate_sealed_test,
                right, handle, plan,
            )
            with pytest.raises(Exception, match="already running"):
                second.result(timeout=2)
            release.set()
            result = first.result(timeout=2)
        assert reads == [session.sealed_execution_spec.spec_hash]
        assert physical.read_count == 1
        assert broker().durable_binding()["state"] == "COMPLETED"
        assert result.evaluation_ref != result.selection_evaluation_ref

    def test_durable_result_must_include_frozen_required_metrics(self, tmp_path):
        session, plan, handle = self._frozen()
        store = SQLiteCampaignStore(tmp_path / "required.sqlite3")
        store.create_campaign(session.session_id, max_evaluations=1, max_cost=1.0)
        broker = TestAuthorityBroker(
            dataset_identity="dataset:v1", provider_identity="test-authority",
            search_session_id=session.session_id, split_id=plan.split_id,
            campaign_store=store, campaign_id=session.session_id,
            candidate_set_hash=session.sealed_execution_spec.spec_hash,
            profile_hash=_split_semantics_hash(plan),
        )
        broker.attach_store_ref(TestStoreRef(
            _CountingOpaqueStore({"physical_test": "holdout-v1"}),
            dataset_identity="dataset:v1",
        ))
        runner = SearchRunner(
            session.config,
            lambda: Trial("unused", "unused", TrialStatus.PROPOSED),
            EvaluationProtocol(
                plan,
                lambda spec, fidelity, data: _typed_outcome(
                    session, plan, {"unrelated": 0.1}
                ),
            ),
        )
        with pytest.raises(ValueError, match="missing required metrics: rank_ic"):
            SealedTestExecutor(runner, broker).evaluate_sealed_test(
                session, handle, plan
            )
        assert broker.durable_binding()["state"] == "RUNNING"

    def test_certified_path_reads_attached_payload_once_and_not_placeholder(self, tmp_path):
        session, plan, handle = self._frozen()
        store = SQLiteCampaignStore(tmp_path / "physical.sqlite3")
        store.create_campaign(session.session_id, max_evaluations=1, max_cost=1.0)
        physical = _CountingOpaqueStore({"payload_identity": "real-holdout-v7"})
        broker = TestAuthorityBroker(
            dataset_identity="dataset:v1", provider_identity="test-authority",
            search_session_id=session.session_id, split_id=plan.split_id,
            campaign_store=store, campaign_id=session.session_id,
            candidate_set_hash=session.sealed_execution_spec.spec_hash,
            profile_hash=_split_semantics_hash(plan),
        )
        broker.attach_store_ref(TestStoreRef(physical, dataset_identity="dataset:v1"))
        observed = []
        runner = SearchRunner(
            session.config,
            lambda: Trial("unused", "unused", TrialStatus.PROPOSED),
            EvaluationProtocol(
                plan,
                lambda spec, fidelity, data: observed.append(data)
                or _typed_outcome(session, plan, {"rank_ic": 0.2}),
            ),
        )
        SealedTestExecutor(runner, broker).evaluate_sealed_test(session, handle, plan)
        assert observed == [{"payload_identity": "real-holdout-v7"}]
        assert physical.read_count == 1
        assert broker.durable_binding()["exposed_at"] is not None

    @pytest.mark.parametrize(
        "field,bad_value",
        [
            ("execution_spec_hash", "wrong-spec"),
            ("dataset_identity", "wrong-dataset"),
            ("split_id", "wrong-split"),
            ("factor_identity", "wrong-factor"),
            ("time_identity", "wrong-time"),
            ("cost_identity", "wrong-cost"),
            ("test_evidence_ref", "wrong-evidence"),
        ],
    )
    def test_typed_outcome_rejects_wrong_certification_identity(
        self, tmp_path, field, bad_value
    ):
        session, plan, handle = self._frozen()
        store = SQLiteCampaignStore(tmp_path / f"wrong-{field}.sqlite3")
        store.create_campaign(session.session_id, max_evaluations=1, max_cost=1.0)
        broker = TestAuthorityBroker(
            dataset_identity="dataset:v1", provider_identity="test-authority",
            search_session_id=session.session_id, split_id=plan.split_id,
            campaign_store=store, campaign_id=session.session_id,
            candidate_set_hash=session.sealed_execution_spec.spec_hash,
            profile_hash=_split_semantics_hash(plan),
        )
        broker.attach_store_ref(TestStoreRef(
            _CountingOpaqueStore({"payload_identity": "identity-check"}),
            dataset_identity="dataset:v1",
        ))
        runner = SearchRunner(
            session.config,
            lambda: Trial("unused", "unused", TrialStatus.PROPOSED),
            EvaluationProtocol(
                plan,
                lambda spec, fidelity, data: _typed_outcome(
                    session, plan, {"rank_ic": 0.1}, **{field: bad_value}
                ),
            ),
        )
        with pytest.raises(ValueError, match="outcome identity"):
            SealedTestExecutor(runner, broker).evaluate_sealed_test(
                session, handle, plan
            )

    def test_failure_after_physical_read_cannot_reopen_or_read_twice(self, tmp_path):
        session, plan, handle = self._frozen()
        checkpoint = session.to_dict()
        store = SQLiteCampaignStore(tmp_path / "burned.sqlite3")
        store.create_campaign(session.session_id, max_evaluations=1, max_cost=1.0)
        physical = _CountingOpaqueStore({"payload_identity": "burn-on-read"})

        def authority():
            broker = TestAuthorityBroker(
                dataset_identity="dataset:v1", provider_identity="test-authority",
                search_session_id=session.session_id, split_id=plan.split_id,
                campaign_store=SQLiteCampaignStore(store.path),
                campaign_id=session.session_id,
                candidate_set_hash=session.sealed_execution_spec.spec_hash,
                profile_hash=_split_semantics_hash(plan),
            )
            broker.attach_store_ref(TestStoreRef(physical, dataset_identity="dataset:v1"))
            return broker

        runner = SearchRunner(
            session.config,
            lambda: Trial("unused", "unused", TrialStatus.PROPOSED),
            EvaluationProtocol(plan, lambda spec, fidelity, data: (_ for _ in ()).throw(RuntimeError("after-read"))),
        )
        first = authority()
        with pytest.raises(RuntimeError, match="after-read"):
            SealedTestExecutor(runner, first).evaluate_sealed_test(session, handle, plan)
        assert physical.read_count == 1
        with pytest.raises(Exception, match="unexposed running"):
            first.mark_test_infrastructure_failure(first.active_attempt_token)

        restored = SearchSession.from_dict(checkpoint)
        with pytest.raises(Exception, match="already running"):
            SealedTestExecutor(runner, authority()).evaluate_sealed_test(
                restored, handle, plan
            )
        assert physical.read_count == 1

    def test_broker_identity_must_match_frozen_session_and_split(self, tmp_path):
        session, plan, handle = self._frozen()
        store = SQLiteCampaignStore(tmp_path / "identity.sqlite3")
        store.create_campaign(session.session_id, max_evaluations=1, max_cost=1.0)
        broker = TestAuthorityBroker(
            dataset_identity="dataset:v1", provider_identity="test-authority",
            search_session_id="wrong-session", split_id=plan.split_id,
            campaign_store=store, campaign_id=session.session_id,
            candidate_set_hash=session.sealed_execution_spec.spec_hash,
            profile_hash=_split_semantics_hash(plan),
        )
        broker.attach_store_ref(TestStoreRef(
            _CountingOpaqueStore({"payload_identity": "must-not-read"}),
            dataset_identity="dataset:v1",
        ))
        runner = SearchRunner(
            session.config,
            lambda: Trial("unused", "unused", TrialStatus.PROPOSED),
            EvaluationProtocol(plan, lambda spec, fidelity, data: pytest.fail("read")),
        )
        with pytest.raises(ValueError, match="durable sealed authority"):
            SealedTestExecutor(runner, broker).evaluate_sealed_test(session, handle, plan)

    def test_stale_provider_cannot_consume_a_new_attempt(self, tmp_path):
        session, plan, _ = self._frozen()
        store = SQLiteCampaignStore(tmp_path / "stale-provider.sqlite3")
        store.create_campaign(session.session_id, max_evaluations=1, max_cost=1.0)
        physical = _CountingOpaqueStore({"payload_identity": "attempt-two"})

        broker = TestAuthorityBroker(
            dataset_identity="dataset:v1", provider_identity="test-authority",
            search_session_id=session.session_id, split_id=plan.split_id,
            campaign_store=SQLiteCampaignStore(store.path),
            campaign_id=session.session_id,
            candidate_set_hash=session.sealed_execution_spec.spec_hash,
            profile_hash=_split_semantics_hash(plan),
        )
        broker.attach_store_ref(TestStoreRef(physical, dataset_identity="dataset:v1"))
        stale_capability = broker.issue_test_capability(plan.test_mask)
        stale_token = broker.active_attempt_token
        stale_provider = broker.create_test_provider(stale_token)
        broker.mark_test_infrastructure_failure(stale_token)
        current_capability = broker.issue_test_capability(plan.test_mask)
        current_token = broker.active_attempt_token
        current_provider = broker.create_test_provider(current_token)
        with pytest.raises(Exception, match="already exposed or not running"):
            stale_provider.resolve(stale_capability)
        assert physical.read_count == 0
        with pytest.raises(Exception, match="attempt_token"):
            current_provider.resolve(stale_capability)
        assert physical.read_count == 0
        assert current_provider.resolve(current_capability) == {
            "payload_identity": "attempt-two"
        }
        assert physical.read_count == 1
        with pytest.raises(Exception):
            broker.complete_test_attempt(stale_token, "stale", {"bad": True})

    def test_certified_path_rejects_bare_metric_dict(self, tmp_path):
        session, plan, handle = self._frozen()
        store = SQLiteCampaignStore(tmp_path / "bare-metrics.sqlite3")
        store.create_campaign(session.session_id, max_evaluations=1, max_cost=1.0)
        broker = TestAuthorityBroker(
            dataset_identity="dataset:v1", provider_identity="test-authority",
            search_session_id=session.session_id, split_id=plan.split_id,
            campaign_store=store, campaign_id=session.session_id,
            candidate_set_hash=session.sealed_execution_spec.spec_hash,
            profile_hash=_split_semantics_hash(plan),
        )
        broker.attach_store_ref(TestStoreRef(
            _CountingOpaqueStore({"payload_identity": "typed-only"}),
            dataset_identity="dataset:v1",
        ))
        runner = SearchRunner(
            session.config,
            lambda: Trial("unused", "unused", TrialStatus.PROPOSED),
            EvaluationProtocol(plan, lambda spec, fidelity, data: {"rank_ic": 0.1}),
        )
        with pytest.raises(TypeError, match="SealedTestEvaluationOutcome"):
            SealedTestExecutor(runner, broker).evaluate_sealed_test(
                session, handle, plan
            )

    def test_completed_result_recovers_typed_without_store_or_second_read(self, tmp_path):
        session, plan, handle = self._frozen()
        checkpoint = session.to_dict()
        store = SQLiteCampaignStore(tmp_path / "completed-recovery.sqlite3")
        store.create_campaign(session.session_id, max_evaluations=1, max_cost=1.0)
        physical = _CountingOpaqueStore({"payload_identity": "completed-once"})

        def authority(attach):
            broker = TestAuthorityBroker(
                dataset_identity="dataset:v1", provider_identity="test-authority",
                search_session_id=session.session_id, split_id=plan.split_id,
                campaign_store=SQLiteCampaignStore(store.path),
                campaign_id=session.session_id,
                candidate_set_hash=session.sealed_execution_spec.spec_hash,
                profile_hash=_split_semantics_hash(plan),
            )
            if attach:
                broker.attach_store_ref(TestStoreRef(
                    physical, dataset_identity="dataset:v1"
                ))
            return broker

        runner = SearchRunner(
            session.config,
            lambda: Trial("unused", "unused", TrialStatus.PROPOSED),
            EvaluationProtocol(
                plan,
                lambda spec, fidelity, data: _typed_outcome(
                    session, plan, {"rank_ic": 0.33}
                ),
            ),
        )
        first = SealedTestExecutor(runner, authority(True)).evaluate_sealed_test(
            session, handle, plan
        )
        assert physical.read_count == 1
        restored = SearchSession.from_dict(checkpoint)
        recovered = SealedTestExecutor(runner, authority(False)).evaluate_sealed_test(
            restored, handle, plan
        )
        assert recovered == first
        assert physical.read_count == 1
        with pytest.raises(TypeError):
            recovered.test_metrics["rank_ic"] = -99
        restored_again = SearchSession.from_dict(checkpoint)
        recovered_again = SealedTestExecutor(
            runner, authority(False)
        ).evaluate_sealed_test(restored_again, handle, plan)
        assert recovered_again.test_metrics == {"rank_ic": 0.33}
        assert physical.read_count == 1

    def test_missing_store_fails_before_attempt_and_keeps_session_available(self, tmp_path):
        session, plan, handle = self._frozen()
        store = SQLiteCampaignStore(tmp_path / "missing-store.sqlite3")
        store.create_campaign(session.session_id, max_evaluations=1, max_cost=1.0)
        broker = TestAuthorityBroker(
            dataset_identity="dataset:v1", provider_identity="test-authority",
            search_session_id=session.session_id, split_id=plan.split_id,
            campaign_store=store, campaign_id=session.session_id,
            candidate_set_hash=session.sealed_execution_spec.spec_hash,
            profile_hash=_split_semantics_hash(plan),
        )
        runner = SearchRunner(
            session.config,
            lambda: Trial("unused", "unused", TrialStatus.PROPOSED),
            EvaluationProtocol(plan, lambda spec, fidelity, data: pytest.fail("read")),
        )
        with pytest.raises(ValueError, match="attached TestStoreRef"):
            SealedTestExecutor(runner, broker).evaluate_sealed_test(
                session, handle, plan
            )
        assert session.sealed_test_state == "AVAILABLE"
        assert broker.durable_binding()["state"] == "FROZEN"
        assert broker.durable_binding()["attempt_count"] == 0


class TestSealedDisjointnessLength:
    def test_unequal_mask_lengths_fail_closed(self):
        # Pre-fix: zip() truncated, so only the shared prefix was compared.
        # SplitPlan construction rejects unequal lengths itself, so build
        # the search plan via a bare object bypassing __post_init__.
        search = SplitPlan("search", [True, False], [False, True], [False, False], {})
        longer = object.__new__(SplitPlan)
        object.__setattr__(longer, "split_id", "test")
        object.__setattr__(longer, "train_mask", [False, False])
        object.__setattr__(longer, "validation_mask", [False, False])
        object.__setattr__(longer, "test_mask", [False, True, True])
        object.__setattr__(longer, "metadata", {})
        assert _sealed_test_disjoint(search, longer) is False

    def test_equal_length_disjoint_plan_passes(self):
        # The sealed test segment [True, False] never overlaps the search
        # plan's train [False, False] or validation [False, True] masks.
        search = SplitPlan("search", [False, False], [False, True], [True, False], {})
        sealed = SplitPlan("test", [False, False], [False, False], [True, False], {})
        assert _sealed_test_disjoint(search, sealed) is True

    def test_overlap_is_detected(self):
        search = SplitPlan("search", [True, False], [False, False], [False, True], {})
        sealed = SplitPlan("test", [False, True], [False, False], [True, False], {})
        assert _sealed_test_disjoint(search, sealed) is False


class TestUpdateBestValidation:
    def test_nan_score_is_rejected_not_silent_lockout(self):
        config = SearchConfig(budget=SearchBudget(max_trials=1, max_evaluations=1))
        session = SearchSession(
            session_id="s", config=config, budget_tracker=BudgetTracker(config.budget)
        )
        session.update_best("t1", 0.5)
        with pytest.raises(ValueError, match="finite non-boolean"):
            session.update_best("t2", float("nan"))
        assert session.best_score == 0.5
        assert session.best_trial_id == "t1"

    def test_bool_score_is_rejected(self):
        config = SearchConfig(budget=SearchBudget(max_trials=1, max_evaluations=1))
        session = SearchSession(
            session_id="s", config=config, budget_tracker=BudgetTracker(config.budget)
        )
        with pytest.raises(ValueError, match="finite non-boolean"):
            session.update_best("t1", True)

    def test_inf_score_is_rejected(self):
        config = SearchConfig(budget=SearchBudget(max_trials=1, max_evaluations=1))
        session = SearchSession(
            session_id="s", config=config, budget_tracker=BudgetTracker(config.budget)
        )
        with pytest.raises(ValueError, match="finite non-boolean"):
            session.update_best("t1", float("inf"))


class TestRunnerIngestion:
    def test_boolean_evaluation_score_fails_the_trial(self):
        # Pre-fix: float(True) == 1.0 passed isfinite and produced a
        # checkpoint that SearchSession.from_dict rejects.
        session = SearchRunner(
            SearchConfig(
                budget=SearchBudget(max_trials=1, max_evaluations=1, max_cost_units=10.0),
                enable_multifidelity=False,
            ),
            lambda: Trial(trial_id="t", mutation_id="m", status=TrialStatus.PROPOSED),
            _protocol(lambda t, f: {"evaluation_id": "e", "score": True, "cost": 1.0}),
        ).run("bool-score")
        trial = session.trials[0]
        assert trial.status is TrialStatus.FAILED
        assert "must not be boolean" in trial.failure_reason
        assert session.best_trial_id is None

    def test_proposal_fn_failure_is_recorded_not_silent(self):
        def boom():
            raise RuntimeError("generator exploded")

        session = SearchRunner(
            SearchConfig(
                budget=SearchBudget(max_trials=1, max_evaluations=1, max_cost_units=10.0),
                enable_multifidelity=False,
            ),
            boom,
            _protocol(lambda t, f: {"score": 1.0, "cost": 1.0, "treatment_integrity_evidence": _integrity_evidence(t.trial_id)}),
        ).run("proposal-failure")
        # FO-P0-04: a raised proposal is a PROPOSAL_FAILED ledger entry, NOT a
        # duplicate.  The burned budget is recorded in the append-only ledger.
        counts = session.ledger.status_counts()
        assert counts.get("PROPOSAL_FAILED") == 1
        assert counts.get("DUPLICATE", 0) == 0
        proposal_failed = [
            e for e in session.ledger.entries if e.status_value == "PROPOSAL_FAILED"
        ]
        assert len(proposal_failed) == 1
        assert "generator exploded" in proposal_failed[0].failure_reason
        # Burned budget stays visible across a checkpoint roundtrip.
        restored = SearchSession.from_dict(session.to_dict())
        restored_failed = [
            e for e in restored.ledger.entries if e.status_value == "PROPOSAL_FAILED"
        ]
        assert "generator exploded" in restored_failed[0].failure_reason

    def test_nan_evaluation_score_never_reaches_update_best(self):
        session = SearchRunner(
            SearchConfig(
                budget=SearchBudget(max_trials=1, max_evaluations=1, max_cost_units=10.0),
                enable_multifidelity=False,
            ),
            lambda: Trial(trial_id="t", mutation_id="m", status=TrialStatus.PROPOSED),
            _protocol(lambda t, f: {"evaluation_id": "e", "score": float("nan"), "cost": 1.0}),
        ).run("nan-score")
        trial = session.trials[0]
        assert trial.status is TrialStatus.FAILED
        assert "must be finite" in trial.failure_reason
        assert session.best_score is None


# ---------------------------------------------------------------------------
# Checkpoint roundtrip corruption tests
# ---------------------------------------------------------------------------

def _frozen_checkpoint():
    """Return a clean frozen-session checkpoint dict."""
    session = _finish(_finished_session())
    frozen = SplitPlan("test", [True, False], [False, False], [False, True], {})
    session.freeze_for_sealed_test(frozen)
    return session.to_dict()


class TestCheckpointRoundtripCorruption:
    """Inject corrupted sealed_test_results / frozen_at / identity fields
    and verify from_dict rejects them."""

    def test_result_item_missing_required_keys(self):
        data = _frozen_checkpoint()
        data["sealed_test_results"] = [{}]
        data["sealed_test_consumed"] = True
        with pytest.raises(ValueError, match="sealed_test_results"):
            SearchSession.from_dict(data)

    def test_result_item_bool_metric_value(self):
        data = _frozen_checkpoint()
        data["sealed_test_results"] = [{
            "trial_id": "t1", "split_id": "test", "evaluation_ref": "e1",
            "frozen_at": "2026-01-01T00:00:00",
            "test_metrics": {"rank_ic": True},
        }]
        data["sealed_test_consumed"] = True
        with pytest.raises(ValueError, match="test_metrics"):
            SearchSession.from_dict(data)

    def test_result_item_nan_metric_value(self):
        data = _frozen_checkpoint()
        data["sealed_test_results"] = [{
            "trial_id": "t1", "split_id": "test", "evaluation_ref": "e1",
            "frozen_at": "2026-01-01T00:00:00",
            "test_metrics": {"rank_ic": float("nan")},
        }]
        data["sealed_test_consumed"] = True
        with pytest.raises(ValueError, match="test_metrics"):
            SearchSession.from_dict(data)

    def test_result_item_inf_metric_value(self):
        data = _frozen_checkpoint()
        data["sealed_test_results"] = [{
            "trial_id": "t1", "split_id": "test", "evaluation_ref": "e1",
            "frozen_at": "2026-01-01T00:00:00",
            "test_metrics": {"rank_ic": float("inf")},
        }]
        data["sealed_test_consumed"] = True
        with pytest.raises(ValueError, match="test_metrics"):
            SearchSession.from_dict(data)

    def test_result_item_frozen_at_is_not_a_string(self):
        data = _frozen_checkpoint()
        data["sealed_test_results"] = [{
            "trial_id": "t1", "split_id": "test", "evaluation_ref": "e1",
            "frozen_at": 42,
            "test_metrics": {"rank_ic": 0.5},
        }]
        data["sealed_test_consumed"] = True
        with pytest.raises(ValueError, match="frozen_at"):
            SearchSession.from_dict(data)

    def test_result_item_frozen_at_is_bool(self):
        data = _frozen_checkpoint()
        data["sealed_test_results"] = [{
            "trial_id": "t1", "split_id": "test", "evaluation_ref": "e1",
            "frozen_at": True,
            "test_metrics": {"rank_ic": 0.5},
        }]
        data["sealed_test_consumed"] = True
        with pytest.raises(ValueError, match="frozen_at"):
            SearchSession.from_dict(data)

    def test_result_item_empty_trial_id(self):
        data = _frozen_checkpoint()
        data["sealed_test_results"] = [{
            "trial_id": "", "split_id": "test", "evaluation_ref": "e1",
            "frozen_at": "2026-01-01T00:00:00",
            "test_metrics": {"rank_ic": 0.5},
        }]
        data["sealed_test_consumed"] = True
        with pytest.raises(ValueError, match="trial_id"):
            SearchSession.from_dict(data)

    def test_top_level_frozen_at_is_bool(self):
        data = _frozen_checkpoint()
        data["frozen_at"] = True
        with pytest.raises(ValueError, match="frozen_at"):
            SearchSession.from_dict(data)

    def test_top_level_frozen_at_is_int(self):
        data = _frozen_checkpoint()
        data["frozen_at"] = 42
        with pytest.raises(ValueError, match="frozen_at"):
            SearchSession.from_dict(data)

    def test_frozen_with_empty_sealed_trial_id(self):
        data = _frozen_checkpoint()
        data["sealed_trial_id"] = ""
        with pytest.raises(ValueError, match="sealed_trial_id"):
            SearchSession.from_dict(data)

    def test_frozen_with_empty_sealed_split_id(self):
        data = _frozen_checkpoint()
        data["sealed_split_id"] = "  "
        with pytest.raises(ValueError, match="sealed_split_id"):
            SearchSession.from_dict(data)


class TestSearchSplitPlanSerialization:
    """The overlap guard's search-time plan must survive checkpointing."""

    def _search_session(self):
        # Search ran on train/validation masks that overlap the sealed test
        # segment [False, True] via validation [False, True].
        search_plan = SplitPlan(
            "search", [True, False], [False, True], [False, False], {}
        )
        config = SearchConfig(
            budget=SearchBudget(max_trials=1, max_evaluations=1, max_cost_units=1.0),
            enable_multifidelity=False,
        )
        object.__setattr__(config, "_search_split_plan", search_plan)
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
            selection_decision_id="decision:v8-fixture",
            selection_decision_hash="decision-hash:v8-fixture",
            selection_request_hash="request-hash:v8-fixture",
            stop_reason="budget_exhausted",
        )
        return _finish(session)

    def test_search_plan_roundtrips_through_checkpoint(self):
        session = self._search_session()
        data = session.to_dict()
        assert data["search_split_masks"]["split_id"] == "search"
        restored = SearchSession.from_dict(data)
        plan = getattr(restored.config, "_search_split_plan", None)
        assert plan is not None
        assert plan.split_id == "search"
        assert list(plan.train_mask) == [True, False]
        assert list(plan.validation_mask) == [False, True]
        assert list(plan.test_mask) == [False, False]

    def test_restored_session_still_rejects_overlapping_sealed_plan(self):
        # Pre-fix: from_dict dropped _search_split_plan, so a restored frozen
        # session silently skipped the sealed/search overlap check.
        session = self._search_session()
        restored = SearchSession.from_dict(session.to_dict())
        overlapping = SplitPlan(
            "test", [True, False], [False, False], [False, True], {}
        )
        with pytest.raises(ValueError, match="overlaps the search-time"):
            restored.freeze_for_sealed_test(overlapping)

    def test_restored_session_accepts_disjoint_sealed_plan(self):
        session = self._search_session()
        restored = SearchSession.from_dict(session.to_dict())
        disjoint = SplitPlan(
            "test", [True, False], [False, True], [False, False], {}
        )
        restored.freeze_for_sealed_test(disjoint)

    def test_malformed_search_split_masks_fail_closed(self):
        session = self._search_session()
        data = session.to_dict()
        data["search_split_masks"]["train"] = [True, 1]
        with pytest.raises(ValueError, match="search_split_masks"):
            SearchSession.from_dict(data)

    def test_unequal_search_split_mask_lengths_fail_closed(self):
        session = self._search_session()
        data = session.to_dict()
        data["search_split_masks"]["test"] = [False, False, False]
        with pytest.raises(ValueError, match="equal lengths"):
            SearchSession.from_dict(data)

    def test_non_bool_disjoint_mask_elements_fail_closed(self):
        # Pre-fix: `type(a) is bool and type(b) is bool` silently skipped
        # non-bool pairs, so numpy/int masks always reported disjoint.
        search = SplitPlan("search", [True, False], [False, False], [False, False], {})
        sealed = object.__new__(SplitPlan)
        object.__setattr__(sealed, "split_id", "test")
        object.__setattr__(sealed, "train_mask", [False, False])
        object.__setattr__(sealed, "validation_mask", [False, False])
        object.__setattr__(sealed, "test_mask", [1, 0])  # ints, not bools
        object.__setattr__(sealed, "metadata", {})
        assert _sealed_test_disjoint(search, sealed) is False


def _integrity_evidence(trial_id="t1", kind=None):
    """Passing TreatmentIntegrityEvidence measured from arrays (R55 P0-9)."""
    from factor_optimizer.contracts.treatment_integrity import (
        build_integrity_evidence,
    )

    rng = np.random.default_rng(abs(hash(trial_id)) % (2 ** 32))
    before = rng.normal(size=32)
    treated_kind = kind if kind else f"treatment::{trial_id}"
    if treated_kind == "raw":
        after = before
    else:
        after = before * 0.5 + 0.01
    return build_integrity_evidence(
        trial_id,
        treated_kind,
        {} if treated_kind == "raw" else {"window": 3},
        before,
        after,
    )
