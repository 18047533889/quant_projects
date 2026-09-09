import pytest
from pathlib import Path
from factor_assets.contracts.asset import AssetMetadata
from factor_assets.contracts.evidence_ref import EvidenceBundleRef
from factor_assets.contracts.lineage import LineageRef
from factor_assets.contracts.lifecycle import LifecycleState
from factor_assets.errors import LifecycleConflictError, SchemaVersionError
from factor_assets.registry.sqlite_repository import SQLiteLifecycleRepository

def _repo(tmp_path):
    return SQLiteLifecycleRepository(tmp_path / "registry.db")
def _register(repo):
    repo.register(AssetMetadata("F", "f", "hash", "daily", ("price",), "daily"), LineageRef("F", ()))
def _bundle(value=1.0):
    return EvidenceBundleRef("b", "run", ("F",), "2026-01-01T00:00:00Z", "qe", primary_metric="ic", primary_value=value)

def test_immediate_typed_decision_replay(tmp_path):
    repo = _repo(tmp_path); _register(repo)
    first = repo.commit_transition("F", LifecycleState.EVALUATED, decision_id="d", evidence_bundle_ref=_bundle())
    replay = repo.commit_transition("F", LifecycleState.EVALUATED, decision_id="d", evidence_bundle_ref=_bundle())
    assert replay.revision == first.revision == 1

def test_replay_after_superseding_transition_returns_original_receipt(tmp_path):
    repo = _repo(tmp_path); _register(repo)
    first = repo.commit_transition("F", LifecycleState.EVALUATED, decision_id="d", evidence_bundle_ref=_bundle())
    repo.commit_transition("F", LifecycleState.APPROVED, evidence_refs=("gate_results",))
    replay = repo.commit_transition("F", LifecycleState.EVALUATED, decision_id="d", evidence_bundle_ref=_bundle())
    assert replay.revision == first.revision == 1
    assert replay.event == first.event

def test_conflicting_typed_payload_is_rejected(tmp_path):
    repo = _repo(tmp_path); _register(repo)
    repo.commit_transition("F", LifecycleState.EVALUATED, decision_id="d", evidence_bundle_ref=_bundle(1.0))
    with pytest.raises(LifecycleConflictError):
        repo.commit_transition("F", LifecycleState.EVALUATED, decision_id="d", evidence_bundle_ref=_bundle(2.0))

@pytest.mark.parametrize("changed", ({"actor": "other"}, {"notes": "other"}, {"policy_version": "v2"}))
def test_decision_replay_rejects_changed_transition_metadata(tmp_path, changed):
    repo = _repo(tmp_path); _register(repo)
    base = {"decision_id": "d", "actor": "actor", "notes": "note", "policy_version": "v1", "evidence_bundle_ref": _bundle()}
    repo.commit_transition("F", LifecycleState.EVALUATED, **base)
    with pytest.raises(LifecycleConflictError):
        repo.commit_transition("F", LifecycleState.EVALUATED, **(base | changed))

def test_public_factory_is_in_all_exports():
    import factor_assets
    import factor_assets.registry as registry
    assert "create_repository" in factor_assets.__all__
    assert "create_repository" in registry.__all__

def test_reopen_rejects_tampered_physical_schema(tmp_path):
    path = tmp_path / "registry.db"
    SQLiteLifecycleRepository(path)
    import sqlite3

    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE assets")

    with pytest.raises(SchemaVersionError, match="schema table assets"):
        SQLiteLifecycleRepository(path)


def test_reopen_rejects_malformed_schema_meta(tmp_path):
    path = tmp_path / "registry.db"
    SQLiteLifecycleRepository(path)
    import sqlite3

    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE schema_meta")
        conn.execute("CREATE TABLE schema_meta (key INTEGER PRIMARY KEY, value TEXT NOT NULL)")

    with pytest.raises(SchemaVersionError, match="schema table schema_meta"):
        SQLiteLifecycleRepository(path)


def test_failed_initial_migration_rolls_back_schema_and_ledger(tmp_path):
    import sqlite3
    from factor_assets.registry.migrations import migrate

    path = tmp_path / "registry.db"
    conn = sqlite3.connect(path, isolation_level=None)

    def deny_ledger_insert(action, table, column, database, trigger):
        if action == sqlite3.SQLITE_INSERT and table == "schema_migrations":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    conn.set_authorizer(deny_ledger_insert)
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        migrate(conn)
    conn.close()

    with sqlite3.connect(path) as probe:
        tables = {
            row[0]
            for row in probe.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert not tables.intersection(
        {"schema_meta", "assets", "lifecycle_events", "schema_migrations"}
    )


def test_reopen_preserves_projection_and_events(tmp_path):
    repo = _repo(tmp_path); _register(repo)
    repo.commit_transition("F", LifecycleState.EVALUATED, evidence_bundle_ref=_bundle())
    reopened = SQLiteLifecycleRepository(tmp_path / "registry.db")
    assert reopened.get("F").latest_evidence_ref == _bundle()
    assert len(reopened.get_events("F")) == 2
