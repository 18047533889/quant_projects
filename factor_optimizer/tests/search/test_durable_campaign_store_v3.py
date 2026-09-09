from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from factor_optimizer.contracts.campaign_store import (
    CampaignStateError,
    DurableBudgetTracker,
    SQLiteCampaignStore,
)


def _store(tmp_path):
    path = tmp_path / "campaign.sqlite3"
    store = SQLiteCampaignStore(path)
    store.create_campaign("campaign", max_evaluations=1, max_cost=10.0)
    return path, store


def test_two_workers_competing_for_last_budget_get_one_authorization(tmp_path):
    path, _ = _store(tmp_path)

    def reserve(attempt):
        return SQLiteCampaignStore(path).reserve("campaign", attempt, 4.0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, ("one", "two")))
    assert sorted(results) == [False, True]


def test_restart_preserves_usage_and_started_work_is_not_lease_refunded(tmp_path):
    path, store = _store(tmp_path)
    assert store.reserve("campaign", "started", 4.0, lease_seconds=0.001)
    store.start("campaign", "started")

    restarted = SQLiteCampaignStore(path)
    # A STARTED job remains charged/reserved after restart even when its lease
    # expires. Recovery must settle actual resource, not silently refund it.
    assert not restarted.reserve("campaign", "other", 1.0)
    restarted.settle("campaign", "started", 2.5)
    assert not restarted.reserve("campaign", "other", 1.0)


def test_alias_and_new_session_cannot_replace_frozen_holdout_set(tmp_path):
    path, store = _store(tmp_path)
    scope = store.freeze_candidate_set(
        campaign_id="campaign",
        candidate_set_hash="candidates-v1",
        dataset_identity="dataset-v1",
        split_id="holdout-2026",
        purpose="production_admission",
        profile_hash="profile-v1",
    )
    # A new process/session can recover the same frozen identity.
    restarted = SQLiteCampaignStore(path)
    assert restarted.freeze_candidate_set(
        campaign_id="renamed-session",
        candidate_set_hash="candidates-v1",
        dataset_identity="dataset-v1",
        split_id="holdout-2026",
        purpose="production_admission",
        profile_hash="profile-v1",
    ) == scope
    with pytest.raises(CampaignStateError, match="different candidate set"):
        restarted.freeze_candidate_set(
            campaign_id="new-session",
            candidate_set_hash="post-test-winner",
            dataset_identity="dataset-v1",
            split_id="holdout-2026",
            purpose="production_admission",
            profile_hash="profile-v1",
        )


def test_only_infrastructure_failure_replays_same_frozen_request(tmp_path):
    _, store = _store(tmp_path)
    scope = store.freeze_candidate_set(
        campaign_id="campaign", candidate_set_hash="set", dataset_identity="data",
        split_id="test", purpose="admission", profile_hash="profile",
    )
    assert store.begin_test_attempt(scope) is None
    with pytest.raises(CampaignStateError, match="already running"):
        store.begin_test_attempt(scope)
    store.mark_infrastructure_failure(scope, 1)
    assert store.begin_test_attempt(scope) is None
    store.mark_test_exposed(scope, 2)
    store.complete_test(scope, 2, "qe:evidence:1", {"qualified": True})

    replay = SQLiteCampaignStore(store.path).begin_test_attempt(scope)
    assert replay == {
        "result_ref": "qe:evidence:1",
        "result": {"qualified": True},
    }
    assert store.sealed_state(scope)["attempt_count"] == 2


def test_stale_attempt_cannot_expose_fail_or_complete_retried_attempt(tmp_path):
    _, store = _store(tmp_path)
    scope = store.freeze_candidate_set(
        campaign_id="campaign", candidate_set_hash="set", dataset_identity="data",
        split_id="test", purpose="admission", profile_hash="profile",
    )
    assert store.begin_test_attempt(scope) is None
    store.mark_infrastructure_failure(scope, 1)
    assert store.begin_test_attempt(scope) is None
    for action in (
        lambda: store.mark_test_exposed(scope, 1),
        lambda: store.mark_infrastructure_failure(scope, 1),
        lambda: store.complete_test(scope, 1, "stale", {"value": 1}),
    ):
        with pytest.raises(CampaignStateError):
            action()
    store.mark_test_exposed(scope, 2)
    store.complete_test(scope, 2, "current", {"value": 2})


def test_purpose_cannot_namespace_a_second_holdout_for_same_dataset_split(tmp_path):
    _, store = _store(tmp_path)
    store.freeze_candidate_set(
        campaign_id="campaign", candidate_set_hash="set", dataset_identity="data",
        split_id="test", purpose="sealed_test", profile_hash="profile",
    )
    with pytest.raises(CampaignStateError, match="purpose"):
        store.freeze_candidate_set(
            campaign_id="campaign", candidate_set_hash="set", dataset_identity="data",
            split_id="test", purpose="other-purpose", profile_hash="profile",
        )


def test_old_attempted_schema_is_migrated_as_unknown_exposure_and_burned(tmp_path):
    path = tmp_path / "old-schema.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE sealed_access (
                scope_hash TEXT PRIMARY KEY, campaign_id TEXT NOT NULL,
                candidate_set_hash TEXT NOT NULL, dataset_identity TEXT NOT NULL,
                split_id TEXT NOT NULL, purpose TEXT NOT NULL,
                profile_hash TEXT NOT NULL, state TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0, result_ref TEXT,
                result_json TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL
            );
            INSERT INTO sealed_access VALUES(
                'legacy','campaign','set','data','test','sealed_test','profile',
                'RUNNING',1,NULL,NULL,1.0,2.0
            );
        """)
    class CrashingMigration(SQLiteCampaignStore):
        def _after_exposure_column_added(self, db):
            raise RuntimeError("crash-after-alter")

    with pytest.raises(RuntimeError, match="crash-after-alter"):
        CrashingMigration(path)
    with sqlite3.connect(path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(sealed_access)")}
    assert "exposed_at" not in columns

    store = SQLiteCampaignStore(path)
    state = store.sealed_state("legacy")
    assert state["exposed_at"] == 2.0
    with pytest.raises(CampaignStateError, match="already running|prior or unknown"):
        store.begin_test_attempt("legacy")
    with pytest.raises(CampaignStateError, match="unexposed running"):
        store.mark_infrastructure_failure("legacy", 1)


def test_public_broker_uses_durable_one_shot_authority_across_sessions(tmp_path):
    from factor_optimizer.data_capabilities import TestAuthorityBroker

    path, store = _store(tmp_path)
    broker = TestAuthorityBroker(
        dataset_identity="dataset", provider_identity="authority",
        search_session_id="session-one", split_id="sealed",
        campaign_store=store, campaign_id="campaign",
        candidate_set_hash="set-v1", purpose="admission", profile_hash="profile-v1",
    )
    capability = broker.issue_test_capability([False, True])
    assert capability.search_session_id == "session-one"
    # This direct broker test has no store attached, so mark exposure through
    # the durable authority to exercise the same attempt binding.
    store.mark_test_exposed(broker._sealed_scope_hash, broker._active_attempt_count)
    broker.complete_test_attempt(
        broker.active_attempt_token, "qe:immutable:1", {"qualified": False}
    )

    restarted = TestAuthorityBroker(
        dataset_identity="dataset", provider_identity="authority",
        search_session_id="aliased-new-session", split_id="sealed",
        campaign_store=SQLiteCampaignStore(path), campaign_id="campaign",
        candidate_set_hash="set-v1", purpose="admission", profile_hash="profile-v1",
    )
    with pytest.raises(ValueError, match="already complete"):
        restarted.issue_test_capability([False, True])
    assert restarted.cached_test_result == {
        "result_ref": "qe:immutable:1", "result": {"qualified": False}
    }


def test_search_runner_uses_durable_budget_factory(tmp_path):
    from factor_optimizer.contracts.search_budget import SearchBudget
    from factor_optimizer.contracts.splits import EvaluationProtocol, SplitPlan
    from factor_optimizer.contracts.trial import Trial, TrialStatus
    from factor_optimizer.search.runner import SearchConfig, SearchRunner
    from tests.search.test_runner import _integrity_evidence

    store = SQLiteCampaignStore(tmp_path / "runner.sqlite3")
    budget = SearchBudget(max_trials=1, max_evaluations=1, max_cost_units=2.0)
    runner = SearchRunner(
        SearchConfig(budget=budget, enable_multifidelity=False),
        lambda: Trial(trial_id="trial-one", mutation_id="mutation-one", status=TrialStatus.PROPOSED),
        EvaluationProtocol(
            SplitPlan("search", [True], [False], [False], {}),
            lambda trial, fidelity: {
                "evaluation_id": "qe:1", "score": 0.2, "cost": 1.0,
                "treatment_integrity_evidence": _integrity_evidence(trial.trial_id),
            },
        ),
        budget_tracker_factory=lambda campaign_id, b: DurableBudgetTracker(store, campaign_id, b),
    )
    session = runner.run("durable-campaign")
    state = SQLiteCampaignStore(store.path).budget_state("durable-campaign")
    assert session.budget_tracker.evaluations_used == 1
    assert state["evaluations_used"] == 1
    assert state["cost_used"] == 1.0


def test_search_runner_config_automatically_enables_durable_store(tmp_path):
    from factor_optimizer.contracts.search_budget import SearchBudget
    from factor_optimizer.contracts.splits import EvaluationProtocol, SplitPlan
    from factor_optimizer.contracts.trial import Trial, TrialStatus
    from factor_optimizer.search.runner import SearchConfig, SearchRunner
    from tests.search.test_runner import _integrity_evidence

    path = tmp_path / "automatic.sqlite3"
    budget = SearchBudget(max_trials=1, max_evaluations=1, max_cost_units=2.0)
    runner = SearchRunner(
        SearchConfig(
            budget=budget, enable_multifidelity=False,
            durable_campaign_store_path=str(path),
        ),
        lambda: Trial(trial_id="auto", mutation_id="m", status=TrialStatus.PROPOSED),
        EvaluationProtocol(
            SplitPlan("search", [True], [False], [False], {}),
            lambda trial, fidelity: {
                "evaluation_id": "qe:auto", "score": 0.1, "cost": 0.5,
                "treatment_integrity_evidence": _integrity_evidence(trial.trial_id),
            },
        ),
    )
    runner.run("automatic-campaign")
    assert SQLiteCampaignStore(path).budget_state("automatic-campaign")["cost_used"] == 0.5


def test_failure_knowledge_is_context_and_intent_scoped(tmp_path):
    _, store = _store(tmp_path)
    store.record_repair_failure(
        effective_spec_hash="recipe", evaluation_intent_hash="intent-v1",
        context_hash="window-v1", reason="no_improvement",
        retry_condition="new_data_or_metric_version", payload={"parameter": 20},
    )
    known = SQLiteCampaignStore(store.path).repair_failure(
        effective_spec_hash="recipe", evaluation_intent_hash="intent-v1", context_hash="window-v1"
    )
    assert known["reason"] == "no_improvement"
    assert known["payload"] == {"parameter": 20}
    assert store.repair_failure(
        effective_spec_hash="recipe", evaluation_intent_hash="intent-v1", context_hash="window-v2"
    ) is None
