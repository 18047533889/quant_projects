"""Registry replay idempotency, connection hygiene, and health write path.

Covers the three FA-REGISTRY-REPLAY defects:
- identical decision_id replay must be idempotent on BOTH backends;
- divergent decision_id replay must raise LifecycleConflictError on BOTH;
- per-call SQLite connections must be closed (no leak);
- update_asset_health persists the health dimension as an append-only event.
"""

import os

import pytest

from factor_assets.contracts.asset import AssetMetadata
from factor_assets.contracts.evidence_ref import EvidenceBundleRef
from factor_assets.contracts.lineage import LineageRef
from factor_assets.contracts.lifecycle import HealthState, LifecycleState
from factor_assets.errors import LifecycleConflictError
from factor_assets.registry.repository import AssetRepository
from factor_assets.registry.repository import AssetNotFoundError
from factor_assets.registry.sqlite_repository import SQLiteLifecycleRepository


def _metadata(factor_id: str = "F") -> AssetMetadata:
    return AssetMetadata(
        factor_id=factor_id,
        canonical_repr=f"{factor_id} repr",
        canonical_hash=f"hash-{factor_id}",
        frequency="daily",
        domains=("price",),
        timing="daily",
    )


def _lineage(factor_id: str = "F") -> LineageRef:
    return LineageRef(factor_id=factor_id, parents=())


def _bundle(value: float = 1.0) -> EvidenceBundleRef:
    return EvidenceBundleRef(
        "b", "run", ("F",), "2026-01-01T00:00:00Z", "qe",
        primary_metric="ic", primary_value=value,
    )


def _memory_repo():
    repo = AssetRepository()
    repo.register(_metadata(), _lineage())
    return repo


def _sqlite_repo(tmp_path):
    repo = SQLiteLifecycleRepository(tmp_path / "registry.db")
    repo.register(_metadata(), _lineage())
    return repo


# ---------------------------------------------------------------------------
# Identical replay is idempotent on both backends
# ---------------------------------------------------------------------------

def test_memory_identical_replay_is_idempotent():
    repo = _memory_repo()
    first = repo.commit_transition(
        "F", LifecycleState.EVALUATED, decision_id="d",
        evidence_bundle_ref=_bundle(), actor="a", notes="n", policy_version="v1",
    )
    replay = repo.commit_transition(
        "F", LifecycleState.EVALUATED, decision_id="d",
        evidence_bundle_ref=_bundle(), actor="a", notes="n", policy_version="v1",
    )
    assert replay.revision == first.revision == 1
    assert replay.event == first.event
    assert len(repo.get_events("F")) == 2  # registration + one transition only


def test_sqlite_identical_replay_is_idempotent(tmp_path):
    repo = _sqlite_repo(tmp_path)
    first = repo.commit_transition(
        "F", LifecycleState.EVALUATED, decision_id="d",
        evidence_bundle_ref=_bundle(), actor="a", notes="n", policy_version="v1",
    )
    replay = repo.commit_transition(
        "F", LifecycleState.EVALUATED, decision_id="d",
        evidence_bundle_ref=_bundle(), actor="a", notes="n", policy_version="v1",
    )
    assert replay.revision == first.revision == 1
    assert replay.event == first.event
    assert len(repo.get_events("F")) == 2


# ---------------------------------------------------------------------------
# Divergent replay raises on both backends
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "changed",
    (
        {"notes": "other"},
        {"actor": "other"},
        {"policy_version": "v2"},
        {"evidence_bundle_ref": _bundle(2.0)},
    ),
)
def test_memory_divergent_replay_raises(changed):
    repo = _memory_repo()
    base = {
        "decision_id": "d", "actor": "a", "notes": "n",
        "policy_version": "v1", "evidence_bundle_ref": _bundle(),
    }
    repo.commit_transition("F", LifecycleState.EVALUATED, **base)
    with pytest.raises(LifecycleConflictError):
        repo.commit_transition("F", LifecycleState.EVALUATED, **(base | changed))


@pytest.mark.parametrize(
    "changed",
    (
        {"notes": "other"},
        {"actor": "other"},
        {"policy_version": "v2"},
        {"evidence_bundle_ref": _bundle(2.0)},
    ),
)
def test_sqlite_divergent_replay_raises(tmp_path, changed):
    repo = _sqlite_repo(tmp_path)
    base = {
        "decision_id": "d", "actor": "a", "notes": "n",
        "policy_version": "v1", "evidence_bundle_ref": _bundle(),
    }
    repo.commit_transition("F", LifecycleState.EVALUATED, **base)
    with pytest.raises(LifecycleConflictError):
        repo.commit_transition("F", LifecycleState.EVALUATED, **(base | changed))


def test_memory_replay_to_different_state_raises():
    repo = _memory_repo()
    repo.commit_transition("F", LifecycleState.EVALUATED, decision_id="d", evidence_bundle_ref=_bundle())
    with pytest.raises(LifecycleConflictError):
        repo.commit_transition("F", LifecycleState.APPROVED, decision_id="d", evidence_refs=("gate_results",))


def test_sqlite_replay_to_different_state_raises(tmp_path):
    repo = _sqlite_repo(tmp_path)
    repo.commit_transition("F", LifecycleState.EVALUATED, decision_id="d", evidence_bundle_ref=_bundle())
    with pytest.raises(LifecycleConflictError):
        repo.commit_transition("F", LifecycleState.APPROVED, decision_id="d", evidence_refs=("gate_results",))


# ---------------------------------------------------------------------------
# Connection hygiene: repeated opens must not leak
# ---------------------------------------------------------------------------

def _open_fd_count() -> int:
    # pysqlite does not expose a live-connection counter; each unclosed
    # connection keeps a file descriptor open, so /proc/self/fd is the
    # observable proxy on Linux.
    return len(os.listdir("/proc/self/fd"))


def test_sqlite_repeated_calls_do_not_leak_connections(tmp_path):
    repo = _sqlite_repo(tmp_path)
    before = _open_fd_count()
    for i in range(50):
        # Interleave reads/writes; every repository call opens (and must
        # close) its own connection.
        if i == 0:
            repo.commit_transition(
                "F", LifecycleState.EVALUATED, decision_id=f"d{i}",
                evidence_bundle_ref=_bundle(),
            )
        else:
            repo.commit_transition(
                "F", LifecycleState.EVALUATED, decision_id=f"d{i}",
                evidence_bundle_ref=_bundle(float(i)),
                expected_revision=repo.get_revision("F"),
            )
        repo.get("F")
        repo.exists("F")
        repo.get_events("F")
    after = _open_fd_count()
    # 50 leaked connections would grow fd count by >= 50; allow a small
    # margin for unrelated fd churn.
    assert after - before < 10, f"fd count grew from {before} to {after}"


# ---------------------------------------------------------------------------
# update_asset_health on both backends
# ---------------------------------------------------------------------------

def test_memory_update_asset_health_stores_event():
    repo = _memory_repo()
    repo.update_asset_health("F", HealthState.DEPRECATED, reason="bad ic", actor="ops")
    asset = repo.get("F")
    assert asset.health_state == HealthState.DEPRECATED
    events = repo.get_events("F")
    assert len(events) == 2
    assert "ACTIVE -> DEPRECATED" in events[-1].notes
    assert "bad ic" in events[-1].notes
    assert events[-1].actor == "ops"
    assert events[-1].to_state == LifecycleState.REGISTERED  # lifecycle untouched
    # Append-only: a second health change appends another event.
    repo.update_asset_health("F", HealthState.RETIRED)
    assert len(repo.get_events("F")) == 3
    assert repo.get("F").health_state == HealthState.RETIRED


def test_sqlite_update_asset_health_stores_event(tmp_path):
    repo = _sqlite_repo(tmp_path)
    repo.update_asset_health("F", HealthState.DEPRECATED, reason="bad ic", actor="ops")
    asset = repo.get("F")
    assert asset.health_state == HealthState.DEPRECATED
    events = repo.get_events("F")
    assert len(events) == 2
    assert "ACTIVE -> DEPRECATED" in events[-1].notes
    assert "bad ic" in events[-1].notes
    assert events[-1].actor == "ops"
    # Durable across reopen.
    reopened = SQLiteLifecycleRepository(tmp_path / "registry.db")
    assert reopened.get("F").health_state == HealthState.DEPRECATED
    assert len(reopened.get_events("F")) == 2


def test_update_asset_health_unknown_factor_raises_on_both(tmp_path):
    with pytest.raises(AssetNotFoundError):
        _memory_repo().update_asset_health("missing", HealthState.RETIRED)
    with pytest.raises(AssetNotFoundError):
        _sqlite_repo(tmp_path).update_asset_health("missing", HealthState.RETIRED)
