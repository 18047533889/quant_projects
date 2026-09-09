from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
import uuid

import factor_engine.runtime.worker_ownership as ownership


def _context(run_dir):
    run_id = uuid.uuid4().hex
    payload = json.dumps({"run_id": run_id}, separators=(",", ":")).encode()
    (run_dir / "identity.json").write_bytes(payload)
    return ownership.RunOwnershipContext(run_id, hashlib.sha256(payload).hexdigest())


def _wait():
    time.sleep(10)


def _bound_child(run_dir, context):
    instance = ownership.start_worker(run_dir, "compute", context=context)
    process = mp.get_context("spawn").Process(target=_wait)
    process.start()
    ownership.bind_worker(run_dir, instance, process.pid, context=context)
    return instance, process


def test_bound_dead_child_is_retired_now_without_completing_or_writing_journal(tmp_path):
    context = _context(tmp_path)
    instance, process = _bound_child(tmp_path, context)
    process.terminate()
    process.join(2)
    journal = tmp_path / "worker-ownership.jsonl"
    before_bytes = journal.read_bytes()
    before_stat = journal.stat()
    before_names = sorted(path.name for path in tmp_path.iterdir())

    result = ownership.inspect_worker_retirement_now(tmp_path, context=context)

    assert result.journal_complete is False
    assert result.all_process_identities_retired_now is True
    assert result.instances == (instance.instance_id,)
    assert "journal history is not closed" in result.reasons[0]
    assert "is retired" in result.reasons[0]
    assert journal.read_bytes() == before_bytes
    after_stat = journal.stat()
    assert (after_stat.st_mtime_ns, after_stat.st_size) == (
        before_stat.st_mtime_ns, before_stat.st_size,
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == before_names


def test_live_bound_child_and_starting_only_remain_unknown(tmp_path):
    live_dir = tmp_path / "live"
    starting_dir = tmp_path / "starting"
    live_dir.mkdir(); starting_dir.mkdir()
    live_context = _context(live_dir)
    _instance, process = _bound_child(live_dir, live_context)
    try:
        result = ownership.inspect_worker_retirement_now(
            live_dir, context=live_context
        )
        assert not result.journal_complete
        assert not result.all_process_identities_retired_now
        assert "cannot be proven retired" in result.reasons[0]
    finally:
        process.terminate(); process.join(2)

    starting_context = _context(starting_dir)
    ownership.start_worker(starting_dir, "ingestion", context=starting_context)
    result = ownership.inspect_worker_retirement_now(
        starting_dir, context=starting_context
    )
    assert not result.journal_complete
    assert not result.all_process_identities_retired_now
    assert "current process identity is unknown" in result.reasons[0]


def test_pid_reuse_and_same_host_boot_change_are_read_only_retirement_proof(
    tmp_path, monkeypatch,
):
    for mode in ("reuse", "boot"):
        run_dir = tmp_path / mode
        run_dir.mkdir()
        context = _context(run_dir)
        _instance, process = _bound_child(run_dir, context)
        try:
            if mode == "reuse":
                monkeypatch.setattr(
                    ownership, "_process_starttime",
                    lambda pid: ("PRESENT", "different-starttime"),
                )
            else:
                monkeypatch.setattr(ownership, "_boot_id", lambda: "different-boot")
            result = ownership.inspect_worker_retirement_now(run_dir, context=context)
            assert not result.journal_complete
            assert result.all_process_identities_retired_now
        finally:
            process.terminate(); process.join(2)
            monkeypatch.undo()


def test_unknown_probe_and_cross_host_identity_fail_closed(tmp_path, monkeypatch):
    for mode in ("unknown", "host"):
        run_dir = tmp_path / mode
        run_dir.mkdir()
        context = _context(run_dir)
        _instance, process = _bound_child(run_dir, context)
        process.terminate(); process.join(2)
        if mode == "unknown":
            monkeypatch.setattr(
                ownership, "_process_starttime", lambda pid: ("UNKNOWN", None)
            )
        else:
            monkeypatch.setattr(
                ownership, "_host_identity", lambda: ("other-machine", "other-ns")
            )
        result = ownership.inspect_worker_retirement_now(run_dir, context=context)
        assert not result.journal_complete
        assert not result.all_process_identities_retired_now
        monkeypatch.undo()


def test_complete_valid_journal_reports_both_true(tmp_path):
    context = _context(tmp_path)
    instance, process = _bound_child(tmp_path, context)
    process.terminate(); process.join(2)
    ownership.mark_worker_exited(tmp_path, instance, context=context)

    result = ownership.inspect_worker_retirement_now(tmp_path, context=context)

    assert result.journal_complete
    assert result.all_process_identities_retired_now
    assert result.reasons == ()


def test_forged_complete_exit_conflicting_with_live_os_is_never_complete(tmp_path):
    context = _context(tmp_path)
    instance = ownership.start_worker(tmp_path, "compute", context=context)
    ownership.bind_worker(tmp_path, instance, os.getpid(), context=context)
    history = ownership._history(tmp_path, instance, context)
    bound = history[1]
    ownership._append(tmp_path, {
        "schema": ownership._SCHEMA, "event": "EXITED",
        "instance_id": instance.instance_id, "pid": bound["pid"],
        "starttime": bound["starttime"], "boot_id": bound["boot_id"],
        "machine_id": bound["machine_id"],
        "pid_namespace": bound["pid_namespace"],
        "exit_evidence": "PID_ABSENT", "observed_boot_id": bound["boot_id"],
        "recorded_monotonic": time.monotonic(),
    }, context=context)

    result = ownership.inspect_worker_retirement_now(tmp_path, context=context)

    assert not result.journal_complete
    assert not result.all_process_identities_retired_now
    assert "conflicts with current OS evidence" in result.reasons[0]


def test_missing_or_corrupt_authority_never_returns_partial_true_or_creates_lock(tmp_path):
    context = _context(tmp_path)
    before = sorted(path.name for path in tmp_path.iterdir())
    missing = ownership.inspect_worker_retirement_now(tmp_path, context=context)
    assert not missing.journal_complete
    assert not missing.all_process_identities_retired_now
    assert sorted(path.name for path in tmp_path.iterdir()) == before
    assert not (tmp_path / "worker-ownership.lock").exists()

    (tmp_path / "worker-ownership.jsonl").write_bytes(
        b'{"schema":"factor_engine.worker_ownership.v1"'
    )
    corrupt = ownership.inspect_worker_retirement_now(tmp_path, context=context)
    assert not corrupt.journal_complete
    assert not corrupt.all_process_identities_retired_now
    assert not (tmp_path / "worker-ownership.lock").exists()


def test_boolean_exited_pid_is_corrupt_and_never_returns_partial_true(tmp_path):
    context = _context(tmp_path)
    instance, process = _bound_child(tmp_path, context)
    process.terminate(); process.join(2)
    ownership.mark_worker_exited(tmp_path, instance, context=context)
    journal = tmp_path / "worker-ownership.jsonl"
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert type(records[-1]["pid"]) is int
    records[-1]["pid"] = True
    journal.write_text("".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ))

    result = ownership.inspect_worker_retirement_now(tmp_path, context=context)

    assert not result.journal_complete
    assert not result.all_process_identities_retired_now
    assert "invalid EXITED evidence" in result.reasons[0]


def test_inspection_uses_one_bound_journal_snapshot(tmp_path, monkeypatch):
    context = _context(tmp_path)
    instance, process = _bound_child(tmp_path, context)
    process.terminate(); process.join(2)
    ownership.mark_worker_exited(tmp_path, instance, context=context)
    original = ownership._bound_events
    calls = []

    def once(*args, **kwargs):
        calls.append(1)
        if len(calls) > 1:
            raise AssertionError("inspection reread the mutable journal")
        return original(*args, **kwargs)

    monkeypatch.setattr(ownership, "_bound_events", once)
    result = ownership.inspect_worker_retirement_now(tmp_path, context=context)

    assert result.journal_complete
    assert result.all_process_identities_retired_now
    assert calls == [1]


def test_huge_integer_timestamp_fails_closed_without_overflow(tmp_path):
    context = _context(tmp_path)
    ownership.start_worker(tmp_path, "compute", context=context)
    journal = tmp_path / "worker-ownership.jsonl"
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    records[-1]["recorded_monotonic"] = 10 ** 1000
    journal.write_text("".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ))

    strict = ownership.validate_all_workers_exited(tmp_path, context=context)
    inspected = ownership.inspect_worker_retirement_now(tmp_path, context=context)

    assert not strict.all_exited
    assert not inspected.journal_complete
    assert not inspected.all_process_identities_retired_now
