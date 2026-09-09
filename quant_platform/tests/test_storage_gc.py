from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from data_access.read.object_store import LocalObjectStore
from quant_platform.app.adapters.data_access_storage import (
    DataAccessStorageAdapter, LocalArtifactCacheImpl, RootAwareGarbageCollector,
    sha256_bytes,
)
from quant_platform.app.contracts import (
    DeletionReceipt, DeletionStatus, GCObject, GCRootSnapshot, GCTombstoneClaim,
    TombstoneClaimStatus,
)


class SQLiteReferenceAuthority:
    """Fixture implementation of the platform registry's transactional seam."""

    def __init__(self, path, objects, references=(), roots=()):
        self.path = str(path)
        with self._db() as db:
            db.executescript("""
              CREATE TABLE state(epoch INTEGER NOT NULL);
              INSERT INTO state VALUES(1);
              CREATE TABLE objects(id TEXT, version TEXT, payload TEXT, PRIMARY KEY(id,version));
              CREATE TABLE refs(parent_id TEXT,parent_version TEXT,child_id TEXT,child_version TEXT);
              CREATE TABLE roots(kind TEXT,id TEXT,version TEXT,PRIMARY KEY(kind,id,version));
              CREATE TABLE tombstones(id TEXT,version TEXT,claim_id TEXT,PRIMARY KEY(id,version));
              CREATE TABLE receipts(id TEXT,version TEXT,payload TEXT,PRIMARY KEY(id,version));
            """)
            for obj in objects:
                db.execute("INSERT INTO objects VALUES(?,?,?)", (*obj.key, json.dumps(asdict(obj))))
            db.executemany("INSERT INTO refs VALUES(?,?,?,?)", [(*p, *c) for p, c in references])
            db.executemany("INSERT INTO roots VALUES(?,?,?)", [(kind, *key) for kind, key in roots])

    def _db(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        return db

    def snapshot_for_gc(self):
        with self._db() as db:
            epoch = db.execute("SELECT epoch FROM state").fetchone()[0]
            objects = tuple(GCObject(**json.loads(r[0])) for r in db.execute("SELECT payload FROM objects"))
            refs = {}
            for row in db.execute("SELECT * FROM refs"):
                refs.setdefault((row[0], row[1]), []).append((row[2], row[3]))
            by_kind = {}
            for row in db.execute("SELECT * FROM roots"):
                by_kind.setdefault(row[0], set()).add((row[1], row[2]))
        return GCRootSnapshot(
            epoch, objects, {k: tuple(v) for k, v in refs.items()},
            production_roots=frozenset(by_kind.get("production", ())),
            approved_release_roots=frozenset(by_kind.get("approved", ())),
            active_read_roots=frozenset(by_kind.get("active_read", ())),
            retryable_job_roots=frozenset(by_kind.get("retryable", ())),
            retained_research_roots=frozenset(by_kind.get("research", ())),
            rollback_roots=frozenset(by_kind.get("rollback", ())),
        )

    def add_root(self, kind, key):
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR IGNORE INTO roots VALUES(?,?,?)", (kind, *key))
            db.execute("UPDATE state SET epoch=epoch+1")
            db.commit()

    def claim_gc_tombstone(self, object_id, object_version, *, expected_epoch):
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT claim_id FROM tombstones WHERE id=? AND version=?",
                               (object_id, object_version)).fetchone()
            if prior:
                db.commit()
                return GCTombstoneClaim(TombstoneClaimStatus.ALREADY_TERMINAL,
                                        object_id, object_version, prior[0], reason="already tombstoned")
            epoch = db.execute("SELECT epoch FROM state").fetchone()[0]
            if epoch != expected_epoch:
                db.commit()
                return GCTombstoneClaim(TombstoneClaimStatus.EPOCH_CHANGED,
                                        object_id, object_version, reason="reference epoch changed")
            rooted = db.execute("SELECT kind FROM roots WHERE id=? AND version=?",
                                (object_id, object_version)).fetchone()
            if rooted:
                db.commit()
                status = TombstoneClaimStatus.LEASED if rooted[0] == "active_read" else TombstoneClaimStatus.ROOTED
                return GCTombstoneClaim(status, object_id, object_version, reason=rooted[0])
            claim = f"claim:{object_id}:{object_version}:{epoch}"
            db.execute("INSERT INTO tombstones VALUES(?,?,?)", (object_id, object_version, claim))
            db.execute("UPDATE state SET epoch=epoch+1")
            db.commit()
            return GCTombstoneClaim(TombstoneClaimStatus.CLAIMED, object_id, object_version, claim, epoch)

    def record_deletion_receipt(self, receipt):
        payload = asdict(receipt)
        payload["status"] = receipt.status.value
        payload["deleted_at"] = receipt.deleted_at.isoformat()
        encoded = json.dumps(payload, sort_keys=True)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR IGNORE INTO receipts VALUES(?,?,?)",
                       (receipt.object_id, receipt.object_version, encoded))
            row = db.execute("SELECT payload FROM receipts WHERE id=? AND version=?",
                             (receipt.object_id, receipt.object_version)).fetchone()
            db.commit()
        return self._decode_receipt(row[0])

    def get_deletion_receipt(self, object_id, object_version):
        with self._db() as db:
            row = db.execute("SELECT payload FROM receipts WHERE id=? AND version=?",
                             (object_id, object_version)).fetchone()
        return None if row is None else self._decode_receipt(row[0])

    @staticmethod
    def _decode_receipt(payload):
        from datetime import datetime
        data = json.loads(payload)
        data["status"] = DeletionStatus(data["status"])
        data["deleted_at"] = datetime.fromisoformat(data["deleted_at"])
        return DeletionReceipt(**data)


def _obj(name, *, terminal=True, size=10):
    data = name.encode()
    return GCObject(name, "v1", f"raw/{name}/v1", sha256_bytes(data),
                    "FAILED" if terminal else "RUNNING", terminal, size)


def _put(adapter, *objects):
    for obj in objects:
        adapter.put(obj.storage_uri, obj.object_id.encode())


def test_dry_run_marks_transitive_and_all_governed_root_classes(tmp_path):
    parent, child, active, retry, rollback, research, orphan, running = [
        _obj(name, terminal=(name != "running"))
        for name in ("parent", "child", "active", "retry", "rollback", "research", "orphan", "running")
    ]
    authority = SQLiteReferenceAuthority(
        tmp_path / "refs.sqlite3", (parent, child, active, retry, rollback, research, orphan, running),
        references=((child.key, parent.key),),
        roots=(("production", child.key), ("active_read", active.key),
               ("retryable", retry.key), ("rollback", rollback.key), ("research", research.key)),
    )
    plan = RootAwareGarbageCollector(
        DataAccessStorageAdapter(store=LocalObjectStore(tmp_path / "objects")), authority
    ).dry_run()
    assert tuple(obj.key for obj in plan.candidates) == (orphan.key,)
    assert parent.key in plan.live_keys
    assert plan.estimated_reclaim_bytes == orphan.size_bytes


def test_epoch_race_new_reference_prevents_sweep(tmp_path):
    orphan = _obj("orphan")
    adapter = DataAccessStorageAdapter(store=LocalObjectStore(tmp_path / "objects"))
    _put(adapter, orphan)
    authority = SQLiteReferenceAuthority(tmp_path / "refs.sqlite3", (orphan,))
    collector = RootAwareGarbageCollector(adapter, authority)
    plan = collector.dry_run()
    authority.add_root("approved", orphan.key)
    receipt = collector.sweep(plan)[0]
    assert receipt.status is DeletionStatus.SKIPPED_PROTECTED
    assert "epoch" in receipt.retained_reason
    assert adapter.get(orphan.storage_uri) == b"orphan"


def test_exact_version_delete_evicts_cache_and_is_idempotent(tmp_path):
    orphan = _obj("orphan")
    adapter = DataAccessStorageAdapter(store=LocalObjectStore(tmp_path / "objects"))
    _put(adapter, orphan)
    cache = LocalArtifactCacheImpl(tmp_path / "cache")
    cache.store(orphan.content_hash, b"orphan")
    authority = SQLiteReferenceAuthority(tmp_path / "refs.sqlite3", (orphan,))
    collector = RootAwareGarbageCollector(adapter, authority, cache=cache)
    plan = collector.dry_run()
    first = collector.sweep(plan)[0]
    second = collector.sweep(plan)[0]
    assert first == second
    assert first.status is DeletionStatus.PHYSICAL_DELETED
    assert first.cache_evicted and first.head_verified_absent
    assert cache.lookup(orphan.content_hash) is None


def test_two_sweep_workers_return_one_immutable_terminal_receipt(tmp_path):
    orphan = _obj("concurrent")
    adapter = DataAccessStorageAdapter(store=LocalObjectStore(tmp_path / "objects"))
    _put(adapter, orphan)
    authority = SQLiteReferenceAuthority(tmp_path / "refs.sqlite3", (orphan,))
    collector = RootAwareGarbageCollector(adapter, authority)
    plan = collector.dry_run()
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(lambda _: collector.sweep(plan)[0], range(2)))
    assert receipts[0] == receipts[1]
    assert receipts[0].status is DeletionStatus.PHYSICAL_DELETED


def test_provider_retention_is_pending_not_physical_success(tmp_path):
    orphan = _obj("retained")
    adapter = DataAccessStorageAdapter(store=LocalObjectStore(tmp_path / "objects"))
    _put(adapter, orphan)
    adapter.delete = lambda uri: None
    authority = SQLiteReferenceAuthority(tmp_path / "refs.sqlite3", (orphan,))
    receipt = RootAwareGarbageCollector(adapter, authority).sweep(
        RootAwareGarbageCollector(adapter, authority).dry_run()
    )[0]
    assert receipt.status is DeletionStatus.PROVIDER_RETENTION_PENDING
    assert not receipt.head_verified_absent
