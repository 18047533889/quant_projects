import hashlib
import json
import sys
import types
import uuid

import pytest

from factor_engine.runtime import worker_ownership as ownership


def _context(path, selection):
    run_id = uuid.uuid4().hex
    payload = json.dumps({"run_id": run_id, "ownership_store": selection}).encode()
    (path / "identity.json").write_bytes(payload)
    return ownership.RunOwnershipContext(run_id, hashlib.sha256(payload).hexdigest())


@pytest.mark.parametrize("selection", [None, "unknown", True, []])
def test_unknown_store_identity_never_creates_a_legacy_journal(tmp_path, selection):
    context = _context(tmp_path, selection)
    before = set(tmp_path.iterdir())
    with pytest.raises(ownership.WorkerOwnershipError, match="unknown ownership store"):
        ownership.start_worker(tmp_path, "compute", context=context)
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("selection,conflicting", [
    ("json-v1", "worker-ownership.sqlite3"),
    ("sqlite-v1", "worker-ownership.jsonl"),
])
def test_incompatible_store_even_dangling_symlink_is_not_adopted(
    tmp_path, selection, conflicting,
):
    context = _context(tmp_path, selection)
    (tmp_path / conflicting).symlink_to(tmp_path / "missing")
    before = set(tmp_path.iterdir())
    with pytest.raises(ownership.WorkerOwnershipError, match="coexist"):
        ownership.start_worker(tmp_path, "compute", context=context)
    assert set(tmp_path.iterdir()) == before


def test_all_lifecycle_calls_use_only_identity_selected_sqlite(tmp_path, monkeypatch):
    context = _context(tmp_path, "sqlite-v1")
    calls = []
    instance = ownership.WorkerInstance(uuid.uuid4().hex, uuid.uuid4().hex)
    backend = types.ModuleType("factor_engine.runtime.worker_ownership_sqlite")

    def start(path, role, *, context):
        calls.append(("start", path, role, context))
        return instance

    def bind(path, actual, pid, *, context):
        calls.append(("bind", path, actual, pid, context))

    def exited(path, actual, *, context):
        calls.append(("exit", path, actual, context))

    backend.start_worker_sqlite = start
    backend.bind_worker_sqlite = bind
    backend.mark_worker_exited_sqlite = exited
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    assert ownership.start_worker(tmp_path, "compute", context=context) is instance
    ownership.bind_worker(tmp_path, instance, 123, context=context)
    ownership.mark_worker_exited(tmp_path, instance, context=context)
    assert calls == [
        ("start", tmp_path, "compute", context),
        ("bind", tmp_path, instance, 123, context),
        ("exit", tmp_path, instance, context),
    ]
    assert not (tmp_path / "worker-ownership.jsonl").exists()


def test_changed_identity_cannot_reselect_lifecycle_store(tmp_path, monkeypatch):
    context = _context(tmp_path, "sqlite-v1")
    identity = tmp_path / "identity.json"
    payload = json.loads(identity.read_text())
    payload["ownership_store"] = "json-v1"
    identity.write_text(json.dumps(payload))
    with pytest.raises(ownership.WorkerOwnershipError, match="digest mismatch"):
        ownership.start_worker(tmp_path, "compute", context=context)
    assert not (tmp_path / "worker-ownership.jsonl").exists()


def test_conflicting_sqlite_created_after_selection_blocks_json_mutation(
    tmp_path, monkeypatch,
):
    context = _context(tmp_path, "json-v1")
    original_start = ownership._start_worker_json
    injected = []

    def inject_between_selection_and_serialized_mutation(path, role, *, context):
        (tmp_path / "worker-ownership.sqlite3").write_bytes(b"conflict")
        injected.append(True)
        return original_start(path, role, context=context)

    monkeypatch.setattr(
        ownership, "_start_worker_json", inject_between_selection_and_serialized_mutation,
    )
    with pytest.raises(ownership.WorkerOwnershipError, match="coexist"):
        ownership.start_worker(tmp_path, "compute", context=context)
    assert injected == [True]
    assert not (tmp_path / "worker-ownership.jsonl").exists()
