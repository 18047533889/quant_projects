from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading
from dataclasses import replace

import pytest

from quant_platform.app.contracts import ArtifactRef, ARTIFACT_TYPE_FACTOR_CANDIDATE
from quant_platform.app.db.sqlite_backend import SqliteDb
from quant_platform.app.storage.generation_coordinator import DurableGenerationCoordinator


class Publisher:
    def __init__(self):
        self.blobs = {}
        self.calls = 0

    def publish(self, artifact, data):
        self.calls += 1
        self.blobs[artifact.content_hash] = data
        return artifact


def ref(data=b"payload", *, artifact_id=None):
    digest = hashlib.sha256(data).hexdigest()
    return ArtifactRef(
        artifact_id=artifact_id or f"a-{digest[:8]}", artifact_type=ARTIFACT_TYPE_FACTOR_CANDIDATE,
        schema_version="1.0", content_hash=digest, storage_uri=f"test://raw/{digest}",
        size_bytes=len(data), created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        producer_type="test", producer_version="1",
    )


def test_transaction_failure_rolls_back_registry_generation_and_outbox(tmp_path):
    path = tmp_path / "metadata.db"
    publisher = Publisher()
    def fail(stage, _):
        if stage == "transaction_before_commit":
            raise RuntimeError("crash")
    db = SqliteDb(str(path))
    with pytest.raises(RuntimeError, match="crash"):
        DurableGenerationCoordinator(db, publisher, failure_injector=fail).stage(ref(), b"payload")
    db.close()
    reopened = SqliteDb(str(path))
    assert reopened.query("SELECT * FROM artifacts") == []
    assert reopened.query("SELECT * FROM artifact_generations") == []
    assert reopened.query("SELECT * FROM outbox_events") == []


def test_blob_crash_is_retryable_and_reader_sees_only_complete(tmp_path, monkeypatch):
    monkeypatch.setattr("quant_platform.app.outbox.time.time", lambda: 1000.0)
    path = tmp_path / "metadata.db"
    publisher = Publisher()
    armed = {"yes": True}
    def fail(stage, _):
        if stage == "blob_written_before_complete" and armed["yes"]:
            armed["yes"] = False
            raise RuntimeError("crash after blob")
    db = SqliteDb(str(path))
    coordinator = DurableGenerationCoordinator(db, publisher, failure_injector=fail)
    artifact = ref()
    coordinator.stage(artifact, b"payload")
    assert coordinator.resolve_active(artifact.artifact_id) is None
    assert coordinator.outbox.publish_pending() == 0
    assert artifact.content_hash in publisher.blobs
    assert coordinator.resolve_active(artifact.artifact_id) is None
    db.close()

    reopened = SqliteDb(str(path))
    retry = DurableGenerationCoordinator(reopened, publisher)
    assert retry.outbox.publish_pending() == 0  # persisted retry backoff survives reopen
    monkeypatch.setattr("quant_platform.app.outbox.time.time", lambda: 1002.0)
    assert retry.outbox.publish_pending() == 1
    active = retry.resolve_active(artifact.artifact_id)
    assert active is not None and active["status"] == "COMPLETE"
    assert publisher.calls == 2


def test_stage_and_publish_retries_are_idempotent(tmp_path):
    db = SqliteDb(str(tmp_path / "metadata.db"))
    publisher = Publisher()
    coordinator = DurableGenerationCoordinator(db, publisher)
    artifact = ref()
    assert coordinator.stage(artifact, b"payload") == coordinator.stage(artifact, b"payload")
    assert len(db.query("SELECT * FROM artifacts")) == 1
    assert len(db.query("SELECT * FROM artifact_generations")) == 1
    assert len(db.query("SELECT * FROM outbox_events")) == 1
    assert coordinator.outbox.publish_pending() == 1
    assert coordinator.outbox.publish_pending() == 0
    assert publisher.calls == 1


def test_same_bytes_different_artifacts_have_distinct_generations(tmp_path):
    db = SqliteDb(str(tmp_path / "metadata.db"))
    coordinator = DurableGenerationCoordinator(db, Publisher())
    first = coordinator.stage(ref(artifact_id="first"), b"payload")
    second = coordinator.stage(ref(artifact_id="second"), b"payload")
    assert first != second
    assert coordinator.outbox.publish_pending() == 2
    assert coordinator.resolve_active("first")["generation_id"] == first
    assert coordinator.resolve_active("second")["generation_id"] == second


@pytest.mark.parametrize("corruption", ["hash", "size", "mutable"])
def test_invalid_generation_bytes_rejected_before_any_write(tmp_path, corruption):
    db = SqliteDb(str(tmp_path / "metadata.db"))
    coordinator = DurableGenerationCoordinator(db, Publisher())
    artifact, data = ref(), b"payload"
    if corruption == "hash":
        data = b"changed"
    elif corruption == "size":
        artifact = replace(artifact, size_bytes=1)
    else:
        data = bytearray(data)
    with pytest.raises((ValueError, TypeError)):
        coordinator.stage(artifact, data)
    for table in ("artifacts", "artifact_generations", "outbox_events"):
        assert db.query(f"SELECT * FROM {table}") == []


def test_generation_retry_cannot_change_semantics(tmp_path):
    db = SqliteDb(str(tmp_path / "metadata.db"))
    coordinator = DurableGenerationCoordinator(db, Publisher())
    artifact = ref()
    generation = coordinator.stage(artifact, b"payload")
    assert coordinator.stage(replace(artifact, created_at=datetime.now(timezone.utc)),
                             b"payload") == generation
    with pytest.raises(ValueError, match="immutable generation"):
        coordinator.stage(replace(artifact, producer_version="changed"), b"payload")


def test_legacy_generation_identity_is_reused_without_relabeling_history(tmp_path):
    import json
    db = SqliteDb(str(tmp_path / "metadata.db"))
    coordinator = DurableGenerationCoordinator(db, Publisher())
    artifact = ref()
    current = coordinator.stage(artifact, b"payload")
    legacy = f"gen:{artifact.content_hash}"
    with db.transaction() as conn:
        conn.execute("UPDATE artifact_generations SET generation_id=? WHERE generation_id=?",
                     (legacy, current))
        conn.execute("UPDATE outbox_events SET aggregate_id=?, idempotency_key=?, payload_json=? "
                     "WHERE aggregate_id=?", (legacy, f"publish:{legacy}",
                     json.dumps({"generation_id": legacy}), current))
    assert coordinator.stage(artifact, b"payload") == legacy
    assert len(db.query("SELECT * FROM artifact_generations")) == 1
    assert coordinator.outbox.publish_pending() == 1
    assert coordinator.resolve_active(artifact.artifact_id)["generation_id"] == legacy


def test_old_complete_delivery_cannot_roll_back_active_generation(tmp_path):
    db = SqliteDb(str(tmp_path / "metadata.db"))
    publisher = Publisher()
    coordinator = DurableGenerationCoordinator(db, publisher)
    old = coordinator.stage(ref(b"old", artifact_id="stable"), b"old")
    coordinator.outbox.publish_pending()
    new = coordinator.stage(ref(b"new", artifact_id="stable"), b"new")
    coordinator.outbox.publish_pending()
    coordinator.publish({"payload": {"generation_id": old}})
    assert coordinator.resolve_active("stable")["generation_id"] == new
    assert publisher.calls == 2


def test_out_of_order_pending_delivery_cannot_roll_back_newer_generation(tmp_path):
    db = SqliteDb(str(tmp_path / "metadata.db"))
    coordinator = DurableGenerationCoordinator(db, Publisher())
    old = coordinator.stage(ref(b"old", artifact_id="stable"), b"old")
    new = coordinator.stage(ref(b"new", artifact_id="stable"), b"new")
    coordinator.publish({"payload": {"generation_id": new}})
    coordinator.publish({"payload": {"generation_id": old}})
    assert coordinator.resolve_active("stable")["generation_id"] == new
    assert db.query("SELECT status, active FROM artifact_generations WHERE generation_id=?",
                    (old,)) == [{"status": "COMPLETE", "active": 0}]


def test_complete_transition_rollback_keeps_generation_invisible(tmp_path):
    publisher = Publisher()
    def fail(stage, _):
        if stage == "complete_before_commit":
            raise RuntimeError("commit crash")
    db = SqliteDb(str(tmp_path / "metadata.db"))
    coordinator = DurableGenerationCoordinator(db, publisher, failure_injector=fail)
    artifact = ref()
    coordinator.stage(artifact, b"payload")
    assert coordinator.outbox.publish_pending() == 0
    assert coordinator.resolve_active(artifact.artifact_id) is None
    row = db.query("SELECT status, active FROM artifact_generations")[0]
    assert row == {"status": "STAGED", "active": 0}


class StoragePublisher:
    def __init__(self, adapter):
        self.adapter = adapter

    def publish(self, artifact, data):
        self.adapter.put(artifact.storage_uri, data)
        return artifact


def _two_generations(tmp_path):
    from data_access.read.object_store import LocalObjectStore
    from quant_platform.app.adapters.data_access_storage import DataAccessStorageAdapter

    path = tmp_path / "metadata.db"
    adapter = DataAccessStorageAdapter(store=LocalObjectStore(tmp_path / "objects"))
    db = SqliteDb(str(path))
    coordinator = DurableGenerationCoordinator(db, StoragePublisher(adapter))
    old, new = ref(b"old", artifact_id="stable-artifact"), ref(b"new", artifact_id="stable-artifact")
    old_gen = coordinator.stage(old, b"old")
    coordinator.outbox.publish_pending()
    new_gen = coordinator.stage(new, b"new")
    coordinator.outbox.publish_pending()
    return path, adapter, coordinator, old, old_gen, new, new_gen


def test_real_registry_snapshot_and_epoch_root_race_blocks_gc(tmp_path):
    from quant_platform.app.adapters.data_access_storage import RootAwareGarbageCollector
    from quant_platform.app.contracts import DeletionStatus

    _, adapter, coordinator, old, old_gen, _, _ = _two_generations(tmp_path)
    collector = RootAwareGarbageCollector(adapter, coordinator)
    plan = collector.dry_run()
    assert [(o.object_id, o.object_version) for o in plan.candidates] == [(old.artifact_id, old_gen)]
    coordinator.protect_gc_root("rollback", old.artifact_id, old_gen)
    receipt = collector.sweep(plan)[0]
    assert receipt.status is DeletionStatus.SKIPPED_PROTECTED
    assert adapter.get(old.storage_uri) == b"old"
    assert coordinator.get_deletion_receipt(old.artifact_id, old_gen) is None
    coordinator.release_gc_root("rollback", old.artifact_id, old_gen)
    deleted = collector.sweep(collector.dry_run())[0]
    assert deleted.status is DeletionStatus.PHYSICAL_DELETED


def test_public_gc_reference_protects_transitive_generation(tmp_path):
    db = SqliteDb(str(tmp_path / "metadata.db"))
    coordinator = DurableGenerationCoordinator(db, Publisher())
    parent = ref(b"parent", artifact_id="parent")
    child = ref(b"child", artifact_id="child")
    parent_gen = coordinator.stage(parent, b"parent")
    child_gen = coordinator.stage(child, b"child")
    coordinator.outbox.publish_pending()
    coordinator.record_gc_reference("child", child_gen, "parent", parent_gen)
    coordinator.protect_gc_root("production", "child", child_gen)
    snapshot = coordinator.snapshot_for_gc()
    assert ("parent", parent_gen) in snapshot.references[("child", child_gen)]
    assert ("child", child_gen) in snapshot.production_roots


def test_real_registry_tombstone_rejects_publish_and_new_root(tmp_path):
    _, _, coordinator, old, old_gen, _, _ = _two_generations(tmp_path)
    epoch = coordinator.snapshot_for_gc().epoch
    claim = coordinator.claim_gc_tombstone(old.artifact_id, old_gen, expected_epoch=epoch)
    assert claim.status.value == "CLAIMED"
    with pytest.raises(ValueError, match="tombstoned"):
        coordinator.stage(old, b"old")
    with pytest.raises(ValueError, match="tombstoned"):
        coordinator.protect_gc_root("active_read", old.artifact_id, old_gen)
    with pytest.raises(ValueError, match="tombstoned"):
        coordinator.publish({"payload": {"generation_id": old_gen}})


def test_gc_crash_after_tombstone_replays_to_one_durable_receipt(tmp_path):
    from quant_platform.app.adapters.data_access_storage import RootAwareGarbageCollector
    from quant_platform.app.contracts import DeletionStatus

    path, adapter, coordinator, old, old_gen, _, _ = _two_generations(tmp_path)
    plan = RootAwareGarbageCollector(adapter, coordinator).dry_run()
    claim = coordinator.claim_gc_tombstone(old.artifact_id, old_gen, expected_epoch=plan.snapshot_epoch)
    assert claim.status.value == "CLAIMED"  # simulated crash before delete/receipt
    coordinator.db.close()
    reopened = DurableGenerationCoordinator(SqliteDb(str(path)), StoragePublisher(adapter))
    receipt = RootAwareGarbageCollector(adapter, reopened).sweep(plan)[0]
    assert receipt.status is DeletionStatus.PHYSICAL_DELETED
    assert reopened.get_deletion_receipt(old.artifact_id, old_gen) == receipt


def test_concurrent_root_and_sweep_have_one_safe_winner(tmp_path):
    from quant_platform.app.adapters.data_access_storage import RootAwareGarbageCollector
    from quant_platform.app.contracts import DeletionStatus

    path, adapter, coordinator, old, old_gen, _, _ = _two_generations(tmp_path)
    plan = RootAwareGarbageCollector(adapter, coordinator).dry_run()
    root_authority = DurableGenerationCoordinator(SqliteDb(str(path)), StoragePublisher(adapter))
    sweep_authority = DurableGenerationCoordinator(SqliteDb(str(path)), StoragePublisher(adapter))
    barrier = threading.Barrier(2)

    def add_root():
        barrier.wait()
        try:
            root_authority.protect_gc_root("rollback", old.artifact_id, old_gen)
            return "rooted"
        except ValueError as exc:
            assert "tombstoned" in str(exc)
            return "tombstoned"

    def sweep():
        barrier.wait()
        return RootAwareGarbageCollector(adapter, sweep_authority).sweep(plan)[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        root_result = pool.submit(add_root)
        receipt_result = pool.submit(sweep)
        root_outcome, receipt = root_result.result(), receipt_result.result()

    if root_outcome == "rooted":
        assert receipt.status is DeletionStatus.SKIPPED_PROTECTED
        assert adapter.get(old.storage_uri) == b"old"
    else:
        assert receipt.status is DeletionStatus.PHYSICAL_DELETED
        assert sweep_authority.get_deletion_receipt(old.artifact_id, old_gen) == receipt


def test_concurrent_publish_and_sweep_never_leave_dangling_reference_or_bytes(tmp_path):
    from data_access.core.exceptions import DataError
    from quant_platform.app.adapters.data_access_storage import RootAwareGarbageCollector
    from quant_platform.app.contracts import DeletionStatus

    path, adapter, coordinator, old, old_gen, _, _ = _two_generations(tmp_path)
    plan = RootAwareGarbageCollector(adapter, coordinator).dry_run()
    publish_authority = DurableGenerationCoordinator(SqliteDb(str(path)), StoragePublisher(adapter))
    sweep_authority = DurableGenerationCoordinator(SqliteDb(str(path)), StoragePublisher(adapter))
    barrier = threading.Barrier(2)

    def republish():
        barrier.wait()
        try:
            publish_authority.publish({"payload": {"generation_id": old_gen}})
            return "already_complete"
        except ValueError as exc:
            assert "tombstoned" in str(exc)
            return "tombstoned"

    def sweep():
        barrier.wait()
        return RootAwareGarbageCollector(adapter, sweep_authority).sweep(plan)[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        publish_result = pool.submit(republish)
        receipt_result = pool.submit(sweep)
        publish_outcome, receipt = publish_result.result(), receipt_result.result()

    active = publish_authority.resolve_active(old.artifact_id)
    assert publish_outcome in {"already_complete", "tombstoned"}
    assert active is not None and active["generation_id"] != old_gen
    assert receipt.status is DeletionStatus.PHYSICAL_DELETED
    with pytest.raises(DataError):
        adapter.head(old.storage_uri)
