from dataclasses import replace
from types import SimpleNamespace
import sqlite3
import pytest
from factor_assets.adapters.platform_outbox import PlatformLifecycleOutboxAdapter
from factor_assets.contracts.asset import AssetMetadata
from factor_assets.contracts.lineage import LineageRef
from factor_assets.contracts.lifecycle import HealthState, LifecycleState, StateEventKind
from factor_assets.errors import LifecycleConflictError
from factor_assets.registry.lifecycle import LifecycleOrchestrator, TransitionRequest
from factor_assets.registry.serialization import event_from_json, event_to_json
from factor_assets.registry.sqlite_repository import SQLiteLifecycleRepository
from quant_platform.app.db.sqlite_backend import SqliteDb

def metadata():
    return AssetMetadata(factor_id="F",canonical_repr="close",canonical_hash="h",frequency="daily",domains=("price",),timing="daily")
def test_platform_outbox_is_staged_atomically_with_transition(tmp_path):
    path=str(tmp_path/"db.sqlite"); platform=SqliteDb(path)
    repo=SQLiteLifecycleRepository(path,transactional_outbox=PlatformLifecycleOutboxAdapter())
    repo.register(metadata(),LineageRef(factor_id="F",parents=()))
    repo.commit_transition("F",LifecycleState.EVALUATED,evidence_refs=("evaluation_bundle_ref",))
    rows=platform.query("SELECT status,idempotency_key FROM outbox_events")
    assert rows==[{"status":"pending","idempotency_key":"factor-lifecycle:F:1"}]
    platform.close()

def test_outbox_stage_failure_rolls_back_asset_and_event(tmp_path):
    class Broken:
        def stage(self, **kwargs): raise RuntimeError("crash before durable intent")
    path=tmp_path/"db.sqlite"; repo=SQLiteLifecycleRepository(path,transactional_outbox=Broken())
    repo.register(metadata(),LineageRef(factor_id="F",parents=()))
    with pytest.raises(RuntimeError):
        repo.commit_transition("F",LifecycleState.EVALUATED,evidence_refs=("evaluation_bundle_ref",))
    assert repo.get("F").lifecycle_state is LifecycleState.REGISTERED
    assert repo.get_revision("F")==0

@pytest.mark.parametrize("mutation,reason",(
    ({"factor_id":"OTHER"},"factor mismatch"),
    ({"recipe_hash":"old-recipe"},"recipe mismatch"),
    ({"decision":"REJECTED"},"not APPROVED"),
    ({"expires_at":"2000-01-01T00:00:00Z"},"expired"),
    ({"policy_version":"old-policy"},"policy version mismatch"),
))
def test_t152_wrong_factor_recipe_decision_expiry_or_policy_authorization_rejects(tmp_path, mutation, reason):
    path=str(tmp_path/"auth.sqlite")
    repo=SQLiteLifecycleRepository(path)
    repo.register(metadata(),LineageRef(factor_id="F",parents=()))
    repo.commit_transition("F",LifecycleState.EVALUATED,evidence_refs=("evaluation_bundle_ref",))
    values={"factor_id":"F","content_hash":"auth:F:v1","recipe_hash":"h",
            "decision":"APPROVED","expires_at":"2099-01-01T00:00:00Z",
            "policy_version":"policy-v1"}
    values.update(mutation)
    artifact=SimpleNamespace(**values)
    resolver=SimpleNamespace(resolve=lambda ref: artifact)
    lifecycle=LifecycleOrchestrator(repository=repo,authorization_resolver=resolver)
    request=TransitionRequest("F",LifecycleState.EVALUATED,LifecycleState.APPROVED,
        evidence_refs=("gate_results",),policy_version="policy-v1",
        authorization_ref="auth:F:v1")
    with pytest.raises(LifecycleConflictError,match=reason):
        lifecycle.execute_transition(request)
    assert repo.get("F").lifecycle_state is LifecycleState.EVALUATED
    assert len(repo.get_events("F")) == 2

def _fold_health(events):
    health=HealthState.ACTIVE
    for event in events:
        if event.event_kind is StateEventKind.HEALTH_TRANSITION:
            assert event.health_from is health
            health=event.health_to
    return health

def test_t172_typed_health_event_log_alone_reconstructs_persisted_snapshot(tmp_path):
    path=str(tmp_path/"health.sqlite")
    repo=SQLiteLifecycleRepository(path)
    repo.register(metadata(),LineageRef(factor_id="F",parents=()))
    repo.update_asset_health("F",HealthState.DEPRECATED,reason="first")
    repo.update_asset_health("F",HealthState.RETIRED,reason="second")
    reopened=SQLiteLifecycleRepository(path)
    events=reopened.get_events("F")
    assert _fold_health(events) is reopened.get("F").health_state is HealthState.RETIRED
    health_events=[e for e in events if e.event_kind is StateEventKind.HEALTH_TRANSITION]
    assert [event_from_json(event_to_json(e)) for e in health_events] == health_events

def test_t173_notes_translation_cannot_change_machine_health_replay(tmp_path):
    path=str(tmp_path/"notes.sqlite")
    repo=SQLiteLifecycleRepository(path)
    repo.register(metadata(),LineageRef(factor_id="F",parents=()))
    repo.update_asset_health("F",HealthState.DEPRECATED,reason="English display text")
    events=repo.get_events("F")
    translated=tuple(replace(e,notes="完全不同的展示文案") for e in events)
    assert _fold_health(events) is HealthState.DEPRECATED
    assert _fold_health(translated) is HealthState.DEPRECATED
    assert translated[-1].health_from is HealthState.ACTIVE
    assert translated[-1].health_to is HealthState.DEPRECATED
