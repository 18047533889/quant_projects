from __future__ import annotations

import multiprocessing as mp
import json
import os
import time
import shutil
import hashlib
import uuid

import pytest

import factor_engine.runtime.worker_ownership as ownership

_CONTEXTS = {}


def _context(run_dir):
    key = str(run_dir)
    if key not in _CONTEXTS:
        run_id = uuid.uuid4().hex
        payload = json.dumps({"run_id": run_id}, separators=(",", ":")).encode()
        (run_dir / "identity.json").write_bytes(payload)
        _CONTEXTS[key] = ownership.RunOwnershipContext(
            run_id, hashlib.sha256(payload).hexdigest()
        )
    return _CONTEXTS[key]


def _start_worker(run_dir, role):
    return ownership.start_worker(run_dir, role, context=_context(run_dir))


def _bind_worker(run_dir, instance, pid):
    return ownership.bind_worker(run_dir, instance, pid, context=_context(run_dir))


def _mark_worker_exited(run_dir, instance):
    return ownership.mark_worker_exited(run_dir, instance, context=_context(run_dir))


def _validate(run_dir):
    return ownership.validate_all_workers_exited(run_dir, context=_context(run_dir))


def _complete_worker(run_dir):
    process = mp.get_context("spawn").Process(target=_wait)
    instance = _start_worker(run_dir, "compute")
    process.start()
    try:
        _bind_worker(run_dir, instance, process.pid)
        process.terminate(); process.join(2.0)
        _mark_worker_exited(run_dir, instance)
    finally:
        if process.is_alive():
            process.terminate(); process.join(2.0)
    assert _validate(run_dir).all_exited


def _wait() -> None:
    time.sleep(10)


def _hold_lock(path, ready):
    import fcntl
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    ready.set()
    time.sleep(3)


def test_missing_and_unbound_starting_fail_closed(tmp_path):
    context = _context(tmp_path)
    before = tuple(tmp_path.iterdir())
    assert not ownership.validate_all_workers_exited(tmp_path, context=context).all_exited
    assert tuple(tmp_path.iterdir()) == before
    _start_worker(tmp_path, "compute")
    result = _validate(tmp_path)
    assert not result.all_exited
    assert "incomplete" in result.reasons[0]


def test_live_pid_cannot_be_marked_exited(tmp_path):
    instance = _start_worker(tmp_path, "compute")
    _bind_worker(tmp_path, instance, os.getpid())
    with pytest.raises(ownership.WorkerOwnershipError, match="still alive"):
        _mark_worker_exited(tmp_path, instance)
    assert not _validate(tmp_path).all_exited


def test_test_owned_terminated_child_has_durable_exit_proof(tmp_path):
    process = mp.get_context("spawn").Process(target=_wait)
    instance = _start_worker(tmp_path, "compute")
    process.start()
    try:
        _bind_worker(tmp_path, instance, process.pid)
        process.terminate()
        process.join(2.0)
        _mark_worker_exited(tmp_path, instance)
        result = _validate(tmp_path)
        assert result.all_exited
        assert result.instances == (instance.instance_id,)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(2.0)


def test_complete_journal_copied_to_another_run_is_rejected(tmp_path):
    source = tmp_path / "run-a"
    target = tmp_path / "run-b"
    source.mkdir(); target.mkdir()
    process = mp.get_context("spawn").Process(target=_wait)
    instance = _start_worker(source, "compute")
    process.start()
    try:
        _bind_worker(source, instance, process.pid)
        process.terminate(); process.join(2.0)
        _mark_worker_exited(source, instance)
        _start_worker(target, "target-unfinished")
        shutil.copy2(source / "worker-ownership.jsonl", target / "worker-ownership.jsonl")
        # RED: copied evidence has no run binding, but the current validator approves it.
        assert not _validate(target).all_exited
    finally:
        if process.is_alive():
            process.terminate(); process.join(2.0)


def test_pid_reuse_is_exit_evidence_for_original_identity(tmp_path, monkeypatch):
    instance = _start_worker(tmp_path, "compute")
    _bind_worker(tmp_path, instance, os.getpid())
    monkeypatch.setattr(ownership, "_process_starttime", lambda pid: ("PRESENT", "different"))
    _mark_worker_exited(tmp_path, instance)
    assert _validate(tmp_path).all_exited


def test_boot_change_requires_same_host_and_pid_namespace(tmp_path, monkeypatch):
    instance = _start_worker(tmp_path, "compute")
    _bind_worker(tmp_path, instance, os.getpid())
    monkeypatch.setattr(ownership, "_boot_id", lambda: "another-boot")
    _mark_worker_exited(tmp_path, instance)
    path = tmp_path / "worker-ownership.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[-1]["recorded_monotonic"] = 0.0
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    assert _validate(tmp_path).all_exited


def test_unknown_process_probe_and_different_host_reboot_fail_closed(tmp_path, monkeypatch):
    first = _start_worker(tmp_path, "first")
    _bind_worker(tmp_path, first, os.getpid())
    monkeypatch.setattr(ownership, "_process_starttime", lambda pid: ("UNKNOWN", None))
    with pytest.raises(ownership.WorkerOwnershipError, match="unknown"):
        _mark_worker_exited(tmp_path, first)

    monkeypatch.setattr(ownership, "_boot_id", lambda: "another-boot")
    monkeypatch.setattr(ownership, "_host_identity", lambda: ("another-machine", "another-ns"))
    with pytest.raises(ownership.WorkerOwnershipError, match="another host"):
        _mark_worker_exited(tmp_path, first)


def test_corrupt_partial_and_duplicate_transitions_fail_closed(tmp_path):
    path = tmp_path / "worker-ownership.jsonl"
    path.write_bytes(b'{"schema":"factor_engine.worker_ownership.v1"')
    assert not _validate(tmp_path).all_exited

    path.unlink()
    instance = _start_worker(tmp_path, "compute")
    ownership._append(tmp_path, {
        "schema": ownership._SCHEMA, "event": "STARTING",
        "instance_id": instance.instance_id, "capability_hash": "duplicate",
        "role": "compute", "recorded_monotonic": time.monotonic(),
    }, context=_context(tmp_path))
    assert not _validate(tmp_path).all_exited


def test_duplicate_bound_process_identity_fails_closed(tmp_path):
    machine, namespace = ownership._host_identity()
    boot = ownership._boot_id()
    for role in ("a", "b"):
        instance = _start_worker(tmp_path, role)
        common = {
            "schema": ownership._SCHEMA, "instance_id": instance.instance_id,
            "pid": 123, "starttime": "456", "boot_id": boot,
            "machine_id": machine, "pid_namespace": namespace,
            "recorded_monotonic": time.monotonic(),
        }
        ownership._append(tmp_path, {**common, "event": "BOUND"}, context=_context(tmp_path))
        ownership._append(tmp_path, {
            **common, "event": "EXITED", "exit_evidence": "PID_ABSENT",
            "observed_boot_id": boot,
        }, context=_context(tmp_path))
    result = _validate(tmp_path)
    assert not result.all_exited
    assert "duplicate worker process identity" in result.reasons

def test_symlink_journal_and_oversized_journal_fail_closed(tmp_path):
    target = tmp_path / "target"
    target.write_text("")
    (tmp_path / "worker-ownership.jsonl").symlink_to(target)
    with pytest.raises(ownership.WorkerOwnershipError):
        _start_worker(tmp_path, "compute")

    (tmp_path / "worker-ownership.jsonl").unlink()
    (tmp_path / "worker-ownership.jsonl").write_bytes(b"x" * (ownership._MAX_BYTES + 1))
    assert not _validate(tmp_path).all_exited


@pytest.mark.parametrize("filename,operation", [
    ("worker-ownership.lock", "start"), ("worker-ownership.jsonl", "validate"),
])
def test_cross_process_lock_wait_is_finite(tmp_path, filename, operation):
    if operation == "validate":
        _start_worker(tmp_path, "seed")
    ready = mp.get_context("spawn").Event()
    holder = mp.get_context("spawn").Process(
        target=_hold_lock, args=(str(tmp_path / filename), ready)
    )
    holder.start()
    try:
        assert ready.wait(1)
        started = time.monotonic()
        if operation == "start":
            with pytest.raises(ownership.WorkerOwnershipError, match="timed out"):
                _start_worker(tmp_path, "blocked")
        else:
            assert not _validate(tmp_path).all_exited
        assert time.monotonic() - started < 1.8
    finally:
        holder.terminate(); holder.join(2)


def test_malformed_kind_and_bool_pid_fail_closed_without_typeerror(tmp_path):
    for mutation, expected in (("kind", "invalid worker instance identity"),
                               ("pid", "invalid bound identity")):
        run_dir = tmp_path / mutation; run_dir.mkdir(); _complete_worker(run_dir)
        path = run_dir / "worker-ownership.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if mutation == "kind":
            records[1]["event"] = []
        else:
            records[2]["pid"] = records[3]["pid"] = True
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        result = _validate(run_dir)
        assert not result.all_exited
        assert any(expected in reason for reason in result.reasons)


def test_context_identity_header_and_event_binding_fail_closed(tmp_path):
    for mutation, expected in (("event", "event/run binding mismatch"),
                               ("header", "header/run binding mismatch"),
                               ("identity", "identity digest mismatch")):
        run_dir = tmp_path / mutation; run_dir.mkdir()
        context = _context(run_dir); _complete_worker(run_dir)
        path = run_dir / "worker-ownership.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if mutation == "event": records[1]["run_id"] = "f" * 32
        elif mutation == "header": records[0]["run_id"] = "f" * 32
        else:
            (run_dir / "identity.json").write_text('{"run_id":"' + context.run_id + '","x":1}')
        if mutation != "identity":
            path.write_text("".join(json.dumps(record) + "\n" for record in records))
        result = ownership.validate_all_workers_exited(run_dir, context=context)
        assert not result.all_exited
        assert expected in result.reasons[0]


@pytest.mark.parametrize("run_id,digest", [("x", "a" * 64), ("a" * 32, "Z" * 64)])
def test_invalid_run_context_is_rejected(run_id, digest):
    with pytest.raises(ownership.WorkerOwnershipError):
        ownership.RunOwnershipContext(run_id, digest)


def test_fifo_journal_is_rejected_without_blocking(tmp_path):
    _context(tmp_path)
    os.mkfifo(tmp_path / "worker-ownership.jsonl")
    started = time.monotonic()
    assert not _validate(tmp_path).all_exited
    assert time.monotonic() - started < 0.5
