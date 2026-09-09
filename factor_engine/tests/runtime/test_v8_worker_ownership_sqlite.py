from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sqlite3
import time
import uuid

import pytest

import factor_engine.runtime.worker_ownership_sqlite as store
from factor_engine.runtime.worker_ownership import RunOwnershipContext, WorkerOwnershipError
from factor_engine.runtime.worker_ownership_sqlite import (
    STORE_NAME,
    bind_worker_sqlite,
    initialize_worker_store,
    mark_worker_exited_sqlite,
    read_worker_event_snapshot_sqlite,
    start_worker_sqlite,
    validate_all_workers_exited_sqlite,
)


def _context(run_dir):
    run_id = uuid.uuid4().hex
    payload = json.dumps(
        {"run_id": run_id, "ownership_store": "sqlite-v1"},
        separators=(",", ":"),
    ).encode()
    (run_dir / "identity.json").write_bytes(payload)
    return RunOwnershipContext(run_id, hashlib.sha256(payload).hexdigest())


def _wait():
    time.sleep(10)


def _hold_write_lock(path, ready):
    connection = sqlite3.connect(path)
    connection.execute("BEGIN IMMEDIATE")
    ready.set()
    time.sleep(3)
    connection.rollback()
    connection.close()


def _hold_descriptor_lock(path, ready):
    import fcntl
    fd = os.open(path, os.O_RDONLY)
    fcntl.flock(fd, fcntl.LOCK_EX)
    ready.set()
    time.sleep(3)
    os.close(fd)


def test_real_lifecycle_yields_canonical_indexed_events(tmp_path):
    context = _context(tmp_path)
    initialize_worker_store(tmp_path, context=context)
    instance = start_worker_sqlite(tmp_path, "compute", context=context)
    process = mp.get_context("spawn").Process(target=_wait)
    process.start()
    try:
        bind_worker_sqlite(tmp_path, instance, process.pid, context=context)
        process.terminate(); process.join(2)
        mark_worker_exited_sqlite(tmp_path, instance, context=context)
    finally:
        if process.is_alive():
            process.terminate(); process.join(2)

    events = list(read_worker_event_snapshot_sqlite(tmp_path, context=context))
    assert [event["event"] for event in events] == ["STARTING", "BOUND", "EXITED"]
    assert all(set(event).isdisjoint({"run_id", "identity_sha256"}) for event in events)
    assert validate_all_workers_exited_sqlite(tmp_path, context=context).all_exited
    with sqlite3.connect(tmp_path / STORE_NAME) as connection:
        assert connection.execute(
            "select state from worker_instances where instance_id=?",
            (instance.instance_id,),
        ).fetchone() == ("EXITED",)
        assert connection.execute("select instance_count,event_count from store_counters").fetchone() == (1, 3)


def test_transition_capability_and_live_exit_fail_closed(tmp_path):
    context = _context(tmp_path)
    initialize_worker_store(tmp_path, context=context)
    instance = start_worker_sqlite(tmp_path, "compute", context=context)
    with pytest.raises(WorkerOwnershipError, match="state mismatch"):
        start = type(instance)(instance.instance_id, "wrong-capability")
        bind_worker_sqlite(tmp_path, start, os.getpid(), context=context)
    bind_worker_sqlite(tmp_path, instance, os.getpid(), context=context)
    with pytest.raises(WorkerOwnershipError, match="still alive"):
        mark_worker_exited_sqlite(tmp_path, instance, context=context)
    with pytest.raises(WorkerOwnershipError, match="state mismatch"):
        bind_worker_sqlite(tmp_path, instance, os.getpid(), context=context)


def test_store_is_explicit_new_run_only_and_copied_identity_is_rejected(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir(); legacy_context = _context(legacy)
    (legacy / "worker-ownership.jsonl").write_text("legacy")
    with pytest.raises(WorkerOwnershipError, match="cannot coexist"):
        initialize_worker_store(legacy, context=legacy_context)

    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir(); target.mkdir()
    source_context = _context(source)
    target_context = _context(target)
    initialize_worker_store(source, context=source_context)
    start_worker_sqlite(source, "compute", context=source_context)
    shutil.copy2(source / STORE_NAME, target / STORE_NAME)
    with pytest.raises(WorkerOwnershipError, match="run binding mismatch"):
        list(read_worker_event_snapshot_sqlite(target, context=target_context))


def test_identity_must_explicitly_select_sqlite_and_dangling_legacy_blocks_init(tmp_path):
    run_id = uuid.uuid4().hex
    payload = json.dumps({"run_id": run_id}, separators=(",", ":")).encode()
    (tmp_path / "identity.json").write_bytes(payload)
    context = RunOwnershipContext(run_id, hashlib.sha256(payload).hexdigest())
    with pytest.raises(WorkerOwnershipError, match="does not select"):
        initialize_worker_store(tmp_path, context=context)

    payload = json.dumps(
        {"run_id": run_id, "ownership_store": "sqlite-v1"},
        separators=(",", ":"),
    ).encode()
    (tmp_path / "identity.json").write_bytes(payload)
    context = RunOwnershipContext(run_id, hashlib.sha256(payload).hexdigest())
    (tmp_path / "worker-ownership.jsonl").symlink_to(tmp_path / "missing")
    with pytest.raises(WorkerOwnershipError, match="cannot coexist"):
        initialize_worker_store(tmp_path, context=context)


def test_write_rechecks_legacy_coexistence_at_transaction_boundary(tmp_path, monkeypatch):
    context = _context(tmp_path)
    initialize_worker_store(tmp_path, context=context)
    original_verify = store._verify_binding

    def inject_legacy(connection, checked_context):
        original_verify(connection, checked_context)
        (tmp_path / "worker-ownership.jsonl").symlink_to(tmp_path / "missing")

    monkeypatch.setattr(store, "_verify_binding", inject_legacy)
    with pytest.raises(WorkerOwnershipError, match="cannot coexist"):
        start_worker_sqlite(tmp_path, "compute", context=context)
    with sqlite3.connect(tmp_path / STORE_NAME) as connection:
        assert connection.execute(
            "select instance_count,event_count from store_counters"
        ).fetchone() == (0, 0)


def test_readonly_missing_and_symlink_store_create_no_authority(tmp_path):
    context = _context(tmp_path)
    before = sorted(path.name for path in tmp_path.iterdir())
    with pytest.raises(WorkerOwnershipError):
        list(read_worker_event_snapshot_sqlite(tmp_path, context=context))
    assert sorted(path.name for path in tmp_path.iterdir()) == before

    target = tmp_path / "target"
    target.write_bytes(b"")
    (tmp_path / STORE_NAME).symlink_to(target)
    with pytest.raises(WorkerOwnershipError, match="database"):
        list(read_worker_event_snapshot_sqlite(tmp_path, context=context))
    assert target.read_bytes() == b""


def test_write_lock_wait_is_finite(tmp_path):
    context = _context(tmp_path)
    initialize_worker_store(tmp_path, context=context)
    ready = mp.get_context("spawn").Event()
    holder = mp.get_context("spawn").Process(
        target=_hold_write_lock, args=(str(tmp_path / STORE_NAME), ready)
    )
    holder.start()
    try:
        assert ready.wait(1)
        started = time.monotonic()
        with pytest.raises(WorkerOwnershipError, match="lock timed out"):
            start_worker_sqlite(tmp_path, "blocked", context=context)
        assert time.monotonic() - started < 1.8
    finally:
        holder.terminate(); holder.join(2)


def test_descriptor_lock_wait_is_finite_for_readonly_validation(tmp_path):
    context = _context(tmp_path)
    initialize_worker_store(tmp_path, context=context)
    ready = mp.get_context("spawn").Event()
    holder = mp.get_context("spawn").Process(
        target=_hold_descriptor_lock, args=(str(tmp_path / STORE_NAME), ready)
    )
    holder.start()
    try:
        assert ready.wait(1)
        started = time.monotonic()
        result = validate_all_workers_exited_sqlite(tmp_path, context=context)
        assert not result.all_exited
        assert "descriptor lock timed out" in result.reasons[0]
        assert time.monotonic() - started < 1.8
    finally:
        holder.terminate(); holder.join(2)


def test_readonly_fd_open_fails_closed_if_database_path_is_replaced(tmp_path, monkeypatch):
    primary = tmp_path / "primary"
    replacement = tmp_path / "replacement"
    primary.mkdir(); replacement.mkdir()
    context = _context(primary)
    replacement_context = _context(replacement)
    initialize_worker_store(primary, context=context)
    initialize_worker_store(replacement, context=replacement_context)
    original_connect = store.sqlite3.connect
    replaced = False

    def swap_before_sqlite_open(target, *args, **kwargs):
        nonlocal replaced
        if isinstance(target, str) and target.startswith("file:/proc/self/fd/"):
            os.replace(replacement / STORE_NAME, primary / STORE_NAME)
            replaced = True
        return original_connect(target, *args, **kwargs)

    monkeypatch.setattr(store.sqlite3, "connect", swap_before_sqlite_open)
    result = validate_all_workers_exited_sqlite(primary, context=context)
    assert replaced
    assert not result.all_exited


def test_counter_or_event_tampering_fails_closed(tmp_path):
    context = _context(tmp_path)
    initialize_worker_store(tmp_path, context=context)
    start_worker_sqlite(tmp_path, "compute", context=context)
    with sqlite3.connect(tmp_path / STORE_NAME) as connection:
        connection.execute("update store_counters set event_count=2")
        connection.commit()
    with pytest.raises(WorkerOwnershipError, match="counters differ"):
        list(read_worker_event_snapshot_sqlite(tmp_path, context=context))

    with sqlite3.connect(tmp_path / STORE_NAME) as connection:
        connection.execute("update store_counters set event_count='not-an-integer'")
        connection.commit()
    result = validate_all_workers_exited_sqlite(tmp_path, context=context)
    assert not result.all_exited
    with pytest.raises(WorkerOwnershipError, match="safety bound"):
        list(read_worker_event_snapshot_sqlite(tmp_path, context=context))


def test_oversized_sqlite_cell_is_rejected_before_payload_allocation(tmp_path):
    context = _context(tmp_path)
    initialize_worker_store(tmp_path, context=context)
    start_worker_sqlite(tmp_path, "compute", context=context)
    with sqlite3.connect(tmp_path / STORE_NAME) as connection:
        connection.execute(
            "update worker_events set payload=?", ("x" * (64 * 1024 + 1),)
        )
        connection.commit()

    with pytest.raises(WorkerOwnershipError, match="database read failed"):
        list(read_worker_event_snapshot_sqlite(tmp_path, context=context))
    result = validate_all_workers_exited_sqlite(tmp_path, context=context)
    assert not result.all_exited
    assert "too big" in result.reasons[0].lower()


def test_empty_initialized_store_is_not_exit_coverage(tmp_path):
    context = _context(tmp_path)
    initialize_worker_store(tmp_path, context=context)
    result = validate_all_workers_exited_sqlite(tmp_path, context=context)
    assert not result.all_exited
    assert result.instance_count == 0
    assert "no instances" in result.reasons[0]


def test_missing_event_uniqueness_and_excess_history_fail_before_payload_read(
    tmp_path, monkeypatch,
):
    context = _context(tmp_path)
    initialize_worker_store(tmp_path, context=context)
    instance = start_worker_sqlite(tmp_path, "compute", context=context)
    with sqlite3.connect(tmp_path / STORE_NAME) as connection:
        connection.executescript("""
            ALTER TABLE worker_events RENAME TO old_worker_events;
            DROP INDEX worker_event_instance_idx;
            CREATE TABLE worker_events(
              seq INTEGER PRIMARY KEY AUTOINCREMENT,
              instance_id TEXT NOT NULL,
              event TEXT NOT NULL,
              payload TEXT NOT NULL
            );
            CREATE INDEX worker_event_instance_idx ON worker_events(instance_id,seq);
            CREATE UNIQUE INDEX fake_event_unique
              ON worker_events(instance_id,event) WHERE 0;
            INSERT INTO worker_events(instance_id,event,payload)
              SELECT instance_id,event,payload FROM old_worker_events;
            DROP TABLE old_worker_events;
        """)
        payload = connection.execute(
            "select payload from worker_events limit 1"
        ).fetchone()[0]
        connection.executemany(
            "insert into worker_events(instance_id,event,payload) values(?,?,?)",
            ((instance.instance_id, "STARTING", payload) for _ in range(20)),
        )
        connection.execute("update store_counters set event_count=21")
        connection.commit()

    def payload_must_not_be_decoded(_payload):
        raise AssertionError("schema must be rejected before payload decoding")

    monkeypatch.setattr(store, "_decode_payload", payload_must_not_be_decoded)
    result = validate_all_workers_exited_sqlite(tmp_path, context=context)
    assert not result.all_exited
    assert "event uniqueness" in result.reasons[0]


def test_process_index_with_wrong_partial_predicate_is_rejected(tmp_path):
    context = _context(tmp_path)
    initialize_worker_store(tmp_path, context=context)
    with sqlite3.connect(tmp_path / STORE_NAME) as connection:
        connection.execute("DROP INDEX worker_process_idx")
        connection.execute("""
            CREATE UNIQUE INDEX worker_process_idx ON worker_instances(
              machine_id,pid_namespace,boot_id,pid,starttime
            ) WHERE pid IS NULL
        """)
        connection.commit()

    result = validate_all_workers_exited_sqlite(tmp_path, context=context)
    assert not result.all_exited
    assert "process identity predicate" in result.reasons[0]


def test_malicious_process_partial_predicate_fails_before_duplicate_scan(
    tmp_path, monkeypatch,
):
    context = _context(tmp_path)
    initialize_worker_store(tmp_path, context=context)
    with sqlite3.connect(tmp_path / STORE_NAME) as connection:
        connection.executescript("""
            DROP INDEX worker_process_idx;
            CREATE UNIQUE INDEX worker_process_idx ON worker_instances(
              machine_id,pid_namespace,boot_id,pid,starttime
            ) WHERE pid IS NULL;
        """)

    statements = []
    original_open = store._readonly_snapshot_connection

    def traced_open(path):
        connection, fd = original_open(path)
        connection.set_trace_callback(statements.append)
        return connection, fd

    monkeypatch.setattr(store, "_readonly_snapshot_connection", traced_open)
    result = validate_all_workers_exited_sqlite(tmp_path, context=context)
    assert not result.all_exited
    assert "process identity predicate" in result.reasons[0]
    assert not any("group by machine_id" in sql.lower() for sql in statements)


def test_readonly_validation_refuses_hot_journal_without_disk_side_effects(tmp_path):
    context = _context(tmp_path)
    initialize_worker_store(tmp_path, context=context)
    start_worker_sqlite(tmp_path, "compute", context=context)
    journal = tmp_path / (STORE_NAME + "-journal")
    journal.write_bytes(b"untrusted-hot-journal")
    before = {path.name: (path.stat().st_size, path.stat().st_mtime_ns)
              for path in tmp_path.iterdir()}

    result = validate_all_workers_exited_sqlite(tmp_path, context=context)

    assert not result.all_exited
    assert "hot journal" in result.reasons[0]
    after = {path.name: (path.stat().st_size, path.stat().st_mtime_ns)
             for path in tmp_path.iterdir()}
    assert after == before


def test_snapshot_row_tampering_cannot_bypass_shared_strict_validator(tmp_path):
    context = _context(tmp_path)
    initialize_worker_store(tmp_path, context=context)
    instance = start_worker_sqlite(tmp_path, "compute", context=context)
    process = mp.get_context("spawn").Process(target=_wait)
    process.start()
    bind_worker_sqlite(tmp_path, instance, process.pid, context=context)
    process.terminate(); process.join(2)
    mark_worker_exited_sqlite(tmp_path, instance, context=context)
    with sqlite3.connect(tmp_path / STORE_NAME) as connection:
        connection.execute(
            "update worker_instances set role='tampered' where instance_id=?",
            (instance.instance_id,),
        )
        connection.commit()

    result = validate_all_workers_exited_sqlite(tmp_path, context=context)
    assert not result.all_exited
    assert "snapshot differs" in result.reasons[0]


def test_ten_thousand_durable_starting_records_use_indexed_control_queries(tmp_path):
    context = _context(tmp_path)
    initialize_worker_store(tmp_path, context=context)
    for ordinal in range(10_000):
        start_worker_sqlite(tmp_path, f"synthetic-{ordinal}", context=context)

    events = list(read_worker_event_snapshot_sqlite(tmp_path, context=context))
    assert len(events) == 10_000
    assert all(event["event"] == "STARTING" for event in events)
    with sqlite3.connect(tmp_path / STORE_NAME) as connection:
        assert connection.execute("select instance_count,event_count from store_counters").fetchone() == (10_000, 10_000)
        indexes = {row[1] for row in connection.execute("pragma index_list('worker_instances')")}
        process_plan = " ".join(str(item) for row in connection.execute(
            "explain query plan select 1 from worker_instances where machine_id=? and pid_namespace=? and boot_id=? and pid=? and starttime=?",
            ("m", "n", "b", 1, "s"),
        ) for item in row)
        event_plan = " ".join(str(item) for row in connection.execute(
            "explain query plan select payload from worker_events where instance_id in (?) order by instance_id,seq limit ?",
            ("i", 4),
        ) for item in row)
    assert "worker_state_idx" in indexes and "worker_process_idx" in indexes
    assert "worker_process_idx" in process_plan
    assert "worker_event_instance_idx" in event_plan
