import hashlib
import json
import multiprocessing as mp
import sqlite3
import time
import uuid

import pytest

import factor_engine.runtime.worker_ownership_sqlite as store
from factor_engine.runtime.worker_ownership import RunOwnershipContext, WorkerInstance


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


def _bound_child(run_dir, context):
    instance = store.start_worker_sqlite(run_dir, "compute", context=context)
    process = mp.get_context("spawn").Process(target=_wait)
    process.start()
    store.bind_worker_sqlite(run_dir, instance, process.pid, context=context)
    return instance, process


def test_bound_dead_child_is_readonly_retirement_proof_without_exited_write(tmp_path):
    context = _context(tmp_path)
    store.initialize_worker_store(tmp_path, context=context)
    instance, process = _bound_child(tmp_path, context)
    process.terminate(); process.join(2)
    path = tmp_path / store.STORE_NAME
    before = path.read_bytes()
    before_stat = path.stat()

    result = store.inspect_worker_retirement_now_sqlite(tmp_path, context=context)

    assert not result.journal_complete
    assert result.all_process_identities_retired_now
    assert result.instance_count == 1
    assert result.example_instances == (instance.instance_id,)
    assert not result.truncated
    assert "history is not closed" in result.reasons[0]
    assert "is retired" in result.reasons[0]
    assert path.read_bytes() == before
    assert (path.stat().st_mtime_ns, path.stat().st_size) == (
        before_stat.st_mtime_ns, before_stat.st_size,
    )
    with sqlite3.connect(path) as db:
        assert db.execute("select state from worker_instances").fetchone() == ("BOUND",)
        assert db.execute("select event_count from store_counters").fetchone() == (2,)


def test_starting_live_unknown_cross_host_boot_and_pid_reuse_semantics(tmp_path, monkeypatch):
    starting_dir = tmp_path / "starting"
    starting_dir.mkdir()
    starting_context = _context(starting_dir)
    store.initialize_worker_store(starting_dir, context=starting_context)
    store.start_worker_sqlite(starting_dir, "ingestion", context=starting_context)
    starting = store.inspect_worker_retirement_now_sqlite(
        starting_dir, context=starting_context
    )
    assert not starting.journal_complete
    assert not starting.all_process_identities_retired_now
    assert "current process identity is unknown" in starting.reasons[0]

    run_dir = tmp_path / "bound"
    run_dir.mkdir()
    context = _context(run_dir)
    store.initialize_worker_store(run_dir, context=context)
    _instance, process = _bound_child(run_dir, context)
    try:
        live = store.inspect_worker_retirement_now_sqlite(run_dir, context=context)
        assert not live.all_process_identities_retired_now

        monkeypatch.setattr(store, "_process_starttime", lambda _pid: ("UNKNOWN", None))
        unknown = store.inspect_worker_retirement_now_sqlite(run_dir, context=context)
        assert not unknown.all_process_identities_retired_now
        monkeypatch.undo()

        monkeypatch.setattr(store, "_host_identity", lambda: ("other", "namespace"))
        cross_host = store.inspect_worker_retirement_now_sqlite(run_dir, context=context)
        assert not cross_host.all_process_identities_retired_now
        monkeypatch.undo()

        monkeypatch.setattr(store, "_boot_id", lambda: "other-boot")
        boot = store.inspect_worker_retirement_now_sqlite(run_dir, context=context)
        assert boot.all_process_identities_retired_now
        monkeypatch.undo()

        monkeypatch.setattr(
            store, "_process_starttime", lambda _pid: ("PRESENT", "999999999")
        )
        reused = store.inspect_worker_retirement_now_sqlite(run_dir, context=context)
        assert reused.all_process_identities_retired_now
    finally:
        process.terminate(); process.join(2)


def test_complete_history_and_bounded_multi_page_starting_snapshot(
    tmp_path, monkeypatch,
):
    complete_dir = tmp_path / "complete"
    complete_dir.mkdir()
    context = _context(complete_dir)
    store.initialize_worker_store(complete_dir, context=context)
    instance, process = _bound_child(complete_dir, context)
    process.terminate(); process.join(2)
    store.mark_worker_exited_sqlite(complete_dir, instance, context=context)
    complete = store.inspect_worker_retirement_now_sqlite(
        complete_dir, context=context
    )
    assert complete.journal_complete
    assert complete.all_process_identities_retired_now
    assert complete.reasons == ()

    paged_dir = tmp_path / "paged"
    paged_dir.mkdir()
    paged_context = _context(paged_dir)
    store.initialize_worker_store(paged_dir, context=paged_context)
    rows = []
    events = []
    for ordinal in range(513):
        instance = WorkerInstance(f"{ordinal:032x}", f"{ordinal + 1000:032x}")
        now = float(ordinal)
        event = {
            "schema": "factor_engine.worker_ownership.v1", "event": "STARTING",
            "instance_id": instance.instance_id,
            "capability_hash": store._capability_hash(instance.capability),
            "role": "compute", "recorded_monotonic": now,
        }
        rows.append((instance.instance_id, event["capability_hash"], now))
        events.append((instance.instance_id, store._event_payload(event, paged_context)))
    with sqlite3.connect(paged_dir / store.STORE_NAME) as db:
        db.executemany(
            "insert into worker_instances(instance_id,capability_hash,role,state,starting_monotonic) "
            "values(?,?,'compute','STARTING',?)", rows,
        )
        db.executemany(
            "insert into worker_events(instance_id,event,payload) values(?,'STARTING',?)",
            events,
        )
        db.execute(
            "update store_counters set instance_count=513,event_count=513"
        )
    fetches = []
    original_open = store._readonly_snapshot_connection

    class CursorProxy:
        def __init__(self, cursor):
            self.cursor = cursor

        def fetchmany(self, size):
            fetches.append(size)
            return self.cursor.fetchmany(size)

        def __getattr__(self, name):
            return getattr(self.cursor, name)

    class ConnectionProxy:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, sql, *args):
            cursor = self.connection.execute(sql, *args)
            if "FROM worker_instances ORDER BY instance_id" in sql:
                return CursorProxy(cursor)
            return cursor

        def __getattr__(self, name):
            return getattr(self.connection, name)

    def tracked_open(path):
        connection, fd = original_open(path)
        return ConnectionProxy(connection), fd

    monkeypatch.setattr(store, "_readonly_snapshot_connection", tracked_open)
    paged = store.inspect_worker_retirement_now_sqlite(
        paged_dir, context=paged_context
    )
    assert not paged.journal_complete
    assert not paged.all_process_identities_retired_now
    assert paged.instance_count == 513
    assert len(paged.example_instances) == store._INSPECTION_EXAMPLE_LIMIT
    assert len(paged.reasons) == store._INSPECTION_REASON_LIMIT
    assert paged.example_instances[0] == f"{0:032x}"
    assert paged.example_instances[-1] == f"{31:032x}"
    assert paged.truncated
    assert fetches == [store._INSPECTION_PAGE_SIZE] * 3


def test_corrupt_row_event_mismatch_fails_closed(tmp_path):
    context = _context(tmp_path)
    store.initialize_worker_store(tmp_path, context=context)
    store.start_worker_sqlite(tmp_path, "compute", context=context)
    with sqlite3.connect(tmp_path / store.STORE_NAME) as db:
        db.execute("update worker_instances set role='forged'")
    result = store.inspect_worker_retirement_now_sqlite(tmp_path, context=context)
    assert not result.journal_complete
    assert not result.all_process_identities_retired_now
    assert result.instance_count == 0
    assert result.example_instances == ()


@pytest.mark.parametrize("tamper", ["schema", "boolean-exited-pid"])
def test_event_schema_and_exact_exited_types_fail_closed(tmp_path, tamper):
    context = _context(tmp_path)
    store.initialize_worker_store(tmp_path, context=context)
    instance, process = _bound_child(tmp_path, context)
    process.terminate(); process.join(2)
    store.mark_worker_exited_sqlite(tmp_path, instance, context=context)
    with sqlite3.connect(tmp_path / store.STORE_NAME) as db:
        payload = json.loads(db.execute(
            "select payload from worker_events where event='EXITED'"
        ).fetchone()[0])
        if tamper == "schema":
            payload["schema"] = "forged.schema"
        else:
            payload["pid"] = True
            db.execute("update worker_instances set pid=1")
            bound = json.loads(db.execute(
                "select payload from worker_events where event='BOUND'"
            ).fetchone()[0])
            bound["pid"] = 1
            db.execute(
                "update worker_events set payload=? where event='BOUND'",
                (json.dumps(bound, sort_keys=True, separators=(",", ":")),),
            )
        db.execute(
            "update worker_events set payload=? where event='EXITED'",
            (json.dumps(payload, sort_keys=True, separators=(",", ":")),),
        )
    result = store.inspect_worker_retirement_now_sqlite(tmp_path, context=context)
    assert not result.journal_complete
    assert not result.all_process_identities_retired_now


@pytest.mark.parametrize("observed", [None, "not-digits", True])
def test_malformed_present_probe_is_not_pid_reuse_proof(tmp_path, monkeypatch, observed):
    context = _context(tmp_path)
    store.initialize_worker_store(tmp_path, context=context)
    _instance, process = _bound_child(tmp_path, context)
    try:
        monkeypatch.setattr(
            store, "_process_starttime", lambda _pid: ("PRESENT", observed)
        )
        result = store.inspect_worker_retirement_now_sqlite(
            tmp_path, context=context
        )
        assert not result.all_process_identities_retired_now
    finally:
        process.terminate(); process.join(2)
