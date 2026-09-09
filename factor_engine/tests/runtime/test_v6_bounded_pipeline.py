from __future__ import annotations

import sqlite3
import time
import pytest
import pandas as pd
from functools import partial
from types import SimpleNamespace
from dataclasses import dataclass, replace
from pathlib import Path

from factor_engine.runtime.bounded_pipeline import execute_run_many_durable
from factor_engine.runtime.finite_manifest import FiniteFactorManifest
from factor_engine.runtime.finite_manifest import ManifestInputTimeout
from factor_engine.runtime.persistent_run_state import PersistentRunState
from factor_engine.runtime.default_execution_policy import resolve_default_policy


@dataclass
class FakeFactor:
    name: str


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.resource_broker = FakeBroker()

    def compile(self, factor):
        return factor

    def run_many_parallel(self, factors, *, result_policy, sink=None, **kwargs):
        self.calls.append(tuple(f.name for f in factors))
        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2025-01-02"), "000001.SZ")], names=["time", "instrument"]
        )
        values = {f.name: pd.Series([1.0], index=idx, name=f.name) for f in factors}
        if sink is not None:
            for name, value in values.items():
                sink(name, value)
            return {"results": {}}
        return {"results": values}


class PreflightIsolationEngine(FakeEngine):
    def run_many_parallel(self, factors, *, result_policy, sink=None, **kwargs):
        self.calls.append(tuple(f.name for f in factors))
        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2025-01-02"), "000001.SZ")], names=["time", "instrument"]
        )
        return {
            "results": {
                f.name: pd.Series([1.0], index=idx, name=f.name)
                for f in factors if f.name != "bad"
            },
            "physical_preflight_errors": {
                "bad": {"code": "UNKNOWN_FIELD", "error_type": "UnknownFieldSemanticError",
                        "message": "unknown field Missing"}
            },
        }


class ExitComputeEngine:
    run_mode = "research"
    def compile(self, factor):
        return factor
    def run_many_parallel(self, factors, **kwargs):
        import os
        os._exit(23)
    def close(self):
        pass


def _build_exit_compute_engine(config):
    return ExitComputeEngine()


class _Lease:
    def release(self):
        pass


class _TrackedLease:
    def __init__(self, broker, nbytes):
        self.broker, self.nbytes, self.done = broker, nbytes, False
        broker.active += nbytes
        broker.peak = max(broker.peak, broker.active)
    def release(self):
        if not self.done:
            self.broker.active -= self.nbytes
            self.done = True


class FakeBroker:
    def summary(self):
        return {"kind": "fake-bounded-broker"}

    def automatic_result_queue_budget(self):
        return 1024 * 1024

    def acquire_protected_egress(self, *args, **kwargs):
        return (_Lease(), _Lease())

    def acquire_memory(self, *args, **kwargs):
        return _Lease()

    def try_reserve(self, *args, **kwargs):
        return _Lease()

    def resource_decision(self, **kwargs):
        return SimpleNamespace(read_wave_bytes=16 * 1024 * 1024, result_queue_bytes=1024 * 1024,
                               target_concurrency=2, target_cpu_tokens=2)

    def cpu_budget(self): return 2
    def current_read_budget(self): return 1024 * 1024
    def current_sink_budget(self): return 1024 * 1024
    def execution_budget(self): return 16 * 1024 * 1024
    hard_cpu_slots = 2


class TinyTrackingBroker(FakeBroker):
    def __init__(self):
        self.active = self.peak = 0
        self.limit = 2_000_000
    def automatic_result_queue_budget(self): return 200_000
    def acquire_protected_egress(self, queue, writer, **kwargs):
        if self.active + queue + writer > self.limit: return None
        return (_TrackedLease(self, queue), _TrackedLease(self, writer))
    def acquire_memory(self, kind, nbytes, **kwargs):
        if self.active + nbytes > self.limit: return None
        return _TrackedLease(self, nbytes)


class TinyMetadataBroker(FakeBroker):
    def current_read_budget(self): return 80


def _slow_sink(ordinal, name, value):
    time.sleep(0.05)
    return {"committed": True, "verified": True}


def _default_write_then_exit(root, run_id, policy, ordinal, name, value, budget, generation):
    import os
    from factor_engine.runtime.bounded_pipeline import _default_worker_write
    _default_worker_write(root, run_id, policy, ordinal, name, value, budget, generation)
    os._exit(17)


def test_physical_preflight_error_is_root_local_and_valid_peers_share_one_wave(tmp_path):
    engine = PreflightIsolationEngine()
    receipt = execute_run_many_durable(
        engine, [FakeFactor("good_a"), FakeFactor("bad"), FakeFactor("good_b")],
        policy=resolve_default_policy(), artifact_root=tmp_path,
    )
    assert receipt["counts"] == {"REJECTED": 1, "SUCCEEDED": 2}
    assert receipt["execution_batches"] == 1
    assert engine.calls == []  # execution happened only in the supervised child
    db = sqlite3.connect(receipt["state_path"])
    assert db.execute(
        "select state,error_code from outcomes where name='bad'"
    ).fetchone() == ("REJECTED", "UNKNOWN_FIELD")


def test_default_writer_timeout_reconciles_known_generation_without_rewrite(tmp_path, monkeypatch):
    import factor_engine.runtime.bounded_pipeline as pipeline
    from factor_engine.runtime.supervised_worker import SupervisedReusableWorker, WorkerTimedOut

    original_execute = SupervisedReusableWorker.execute
    timed_out = {"done": False}

    def execute(self, function, *args, timeout_seconds, lease=None, **kwargs):
        inherited = getattr(self, "_function", None)
        if (not timed_out["done"] and isinstance(inherited, partial)
                and inherited.func is pipeline._default_worker_write):
            timed_out["done"] = True
            inherited(function, *args, **kwargs)
            if lease is not None:
                lease.release()
            raise WorkerTimedOut("simulated timeout after durable rename")
        return original_execute(
            self, function, *args, timeout_seconds=timeout_seconds, lease=lease, **kwargs
        )

    monkeypatch.setattr(SupervisedReusableWorker, "execute", execute)
    receipt = execute_run_many_durable(
        FakeEngine(), [FakeFactor("recover")],
        policy=resolve_default_policy(), artifact_root=tmp_path,
    )
    assert timed_out["done"] is True
    assert receipt["counts"] == {"SUCCEEDED": 1}
    db = sqlite3.connect(receipt["state_path"])
    state, commit_state, generation, artifact_json = db.execute(
        "select state,commit_state,artifact_generation,artifact_json from outcomes"
    ).fetchone()
    assert (state, commit_state) == ("SUCCEEDED", "VERIFIED")
    assert len(generation) == 32
    assert generation in artifact_json


def test_worker_quarantine_aborts_run_terminalizes_ordinals_and_writes_receipt(tmp_path, monkeypatch):
    import json
    import factor_engine.runtime.bounded_pipeline as pipeline
    from factor_engine.runtime.supervised_worker import SupervisedReusableWorker, WorkerQuarantined

    original_execute = SupervisedReusableWorker.execute
    original_close = SupervisedReusableWorker.close
    quarantine_injected = {"done": False}

    def execute(self, function, *args, timeout_seconds, lease=None, **kwargs):
        inherited = getattr(self, "_function", None)
        if (isinstance(inherited, partial)
                and inherited.func is pipeline._compute_wave_worker):
            quarantine_injected["done"] = True
            raise WorkerQuarantined("simulated compute quarantine", pid=123)
        return original_execute(
            self, function, *args, timeout_seconds=timeout_seconds, lease=lease, **kwargs
        )

    monkeypatch.setattr(SupervisedReusableWorker, "execute", execute)
    def close(self):
        inherited = getattr(self, "_function", None)
        if isinstance(inherited, partial) and inherited.func is pipeline._default_worker_write:
            raise RuntimeError("simulated cleanup failure")
        return original_close(self)
    monkeypatch.setattr(SupervisedReusableWorker, "close", close)
    with pytest.raises(WorkerQuarantined) as raised:
        execute_run_many_durable(
            FakeEngine(), [FakeFactor("a"), FakeFactor("b")],
            policy=resolve_default_policy(), artifact_root=tmp_path,
        )
    assert quarantine_injected["done"] is True
    receipt = json.loads(Path(raised.value.receipt_path).read_text())
    assert receipt["status"] == "ABORTED"
    assert receipt["counts"] == {"CANCELLED": 2}
    assert str(raised.value.cleanup_errors[0]) == "simulated cleanup failure"
    db = sqlite3.connect(receipt["state_path"])
    assert db.execute("select distinct error_code from outcomes").fetchall() == [("RUN_ABORTED",)]


def test_second_spawn_worker_start_failure_closes_first_and_aborts_all(tmp_path, monkeypatch):
    import json
    from factor_engine.runtime.supervised_worker import SupervisedReusableWorker

    original_start = SupervisedReusableWorker.start
    started = []

    def start(self):
        mode = getattr(getattr(self, "_function", None), "mode", None)
        if mode == "compute":
            raise RuntimeError("simulated second worker start failure")
        original_start(self)
        if mode == "compile":
            started.append(self)

    monkeypatch.setattr(SupervisedReusableWorker, "start", start)
    with pytest.raises(RuntimeError, match="second worker start failure") as raised:
        execute_run_many_durable(
            None, [FakeFactor("a"), FakeFactor("b")],
            policy=resolve_default_policy(), artifact_root=tmp_path,
            run_kwargs={"broker": FakeBroker()}, engine_factory=_build_spawn_engine,
            engine_factory_config={},
        )
    assert len(started) == 1
    assert not started[0]._process.is_alive()
    receipt = json.loads(Path(raised.value.receipt_path).read_text())
    assert receipt["status"] == "ABORTED"
    assert receipt["counts"] == {"CANCELLED": 2}


def test_lost_writer_ack_reconciles_existing_generation_without_rewrite(tmp_path):
    import hashlib
    from factor_engine.runtime.bounded_pipeline import _default_worker_reconcile
    from factor_engine.runtime.supervised_worker import (
        SupervisedReusableWorker, WorkerTransportFailed,
    )

    policy = resolve_default_policy()
    run_id = "a" * 32
    generation = "b" * 32
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-01-02"), "000001.SZ")], names=["time", "instrument"]
    )
    value = pd.Series([1.25], index=idx, name="factor")
    writer = SupervisedReusableWorker(
        context="spawn",
        function=partial(_default_write_then_exit, tmp_path, run_id, policy),
        cancel_grace_seconds=0.01, exit_observation_seconds=2,
    )
    with pytest.raises(WorkerTransportFailed):
        writer.execute(
            0, "factor", value, 16 * 1024 * 1024, generation,
            timeout_seconds=5,
        )
    assert not writer._process.is_alive()
    manifest = tmp_path / run_id / "values" / f"00000000-{generation}" / "manifest.json"
    before = hashlib.sha256(manifest.read_bytes()).hexdigest()
    files_before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    reconcile = SupervisedReusableWorker(
        context="spawn", function=partial(_default_worker_reconcile, tmp_path, run_id, policy),
    )
    receipt = reconcile.execute(
        0, "factor", value, 16 * 1024 * 1024, generation, timeout_seconds=5,
    ).value
    reconcile.close()
    writer.close()
    assert receipt["committed"] is True and receipt["verified"] is True
    assert receipt["generation"] == generation
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == before
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")) == files_before


def test_spawn_compute_exit_aborts_later_wave_without_reusing_proxy_binding(tmp_path):
    import json
    from factor_engine.runtime.supervised_worker import WorkerTransportFailed

    policy = replace(resolve_default_policy(), initial_lookahead_factors=1)
    with pytest.raises(WorkerTransportFailed) as raised:
        execute_run_many_durable(
            None, [FakeFactor("first"), FakeFactor("prefetched"),
                   FakeFactor("not_admitted")],
            policy=policy, artifact_root=tmp_path,
            run_kwargs={"broker": FakeBroker()},
            engine_factory=_build_exit_compute_engine, engine_factory_config={},
        )
    receipt = json.loads(Path(raised.value.receipt_path).read_text())
    assert receipt["status"] == "ABORTED"
    assert receipt["counts"] == {"CANCELLED": 2, "FAILED": 1}
    db = sqlite3.connect(receipt["state_path"])
    assert db.execute(
        "select name,attempts,state,error_code,commit_state "
        "from outcomes order by ordinal"
    ).fetchall() == [
        # Slot authority fails closed as protocol integrity, while the detail
        # retains the underlying transport failure type for diagnosis.
        ("first", 1, "FAILED", "WORKER_PROTOCOL_INTEGRITY", "UNKNOWN"),
        # The paired slot is admitted before either compute starts. Once the
        # fatal exit is observed, admission never advances past that pair.
        ("prefetched", 1, "CANCELLED", "RUN_ABORTED", "UNKNOWN"),
        ("not_admitted", 0, "CANCELLED", "RUN_ABORTED", "NOT_STARTED"),
    ]
    first_detail = db.execute(
        "select error_detail from outcomes where name='first'"
    ).fetchone()[0]
    assert first_detail.startswith("WorkerTransportFailed:")
    assert receipt["fit_failure_evidence"]["availability"] == "indexed"
    assert receipt["fit_failure_evidence"]["observed_waves"] == 0
    assert receipt["fit_failure_evidence"]["unavailable_waves"] == 1
    evidence = db.execute(
        "select availability,assignment_count from fit_failure_evidence"
    ).fetchall()
    assert evidence == [("UNAVAILABLE", 1)]
    assert db.execute(
        "select ordinal from fit_failure_evidence_assignments"
    ).fetchall() == [(0,)]
    db.close()


def _build_spawn_engine(config):
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.runtime.engine import FactorEngine
    from tests.helpers import InMemorySeriesSource
    marker = config.get("__construct_marker__")
    if marker:
        with open(marker, "a", encoding="utf-8") as stream:
            stream.write("constructed\n")
    data = {k: v for k, v in config.items() if not k.startswith("__")}
    source = InMemorySeriesSource(data=data)
    close_marker = config.get("__close_marker__")
    if close_marker:
        original_close = getattr(source, "close", lambda: None)
        def close():
            original_close()
            with open(close_marker, "a", encoding="utf-8") as stream:
                stream.write("closed\n")
        source.close = close
    return FactorEngine(PandasBackend(), source)


def test_manifest_duplicate_is_per_ordinal_rejection(tmp_path):
    manifest = FiniteFactorManifest.ingest(
        [FakeFactor("same"), FakeFactor("same"), {"name": "bad", "callback": lambda: 1}],
        tmp_path / "manifest.sqlite3",
    )
    records = list(manifest.records(limit=10))
    assert [r.error_code for r in records] == [
        "DUPLICATE_FACTOR_NAME", "DUPLICATE_FACTOR_NAME", "INVALID_FACTOR_DEFINITION"
    ]
    manifest.close()


def test_blocking_iterator_is_process_supervised(tmp_path):
    class Blocking:
        def __init__(self): self.n = 0
        def __iter__(self): return self
        def __next__(self):
            if self.n == 0:
                self.n += 1
                return FakeFactor("before_block")
            time.sleep(5)
            raise StopIteration
    manifest = FiniteFactorManifest.ingest_supervised(
        Blocking(), tmp_path / "blocked.sqlite3", deadline_seconds=0.05
    )
    assert len(manifest) == 1
    assert manifest.input_complete is False
    assert "deadline" in manifest.input_error
    manifest.close()


def test_retry_budget_is_persistent_and_shared(tmp_path):
    path = tmp_path / "state.sqlite3"
    state = PersistentRunState(path, max_attempts=3)
    state.register(0, "x")
    assert state.consume_attempt(0, "source") == 1
    assert state.consume_attempt(0, "executor") == 2
    state.close()
    state = PersistentRunState(path, max_attempts=3)
    assert state.consume_attempt(0, "sink") == 3
    assert state.consume_attempt(0, "executor") is None
    state.terminal(0, "FAILED", error_code="TRANSIENT_EXHAUSTED")
    state.close()


def test_durable_pipeline_executes_real_engine_path_and_artifact(tmp_path):
    engine = FakeEngine()
    receipt = execute_run_many_durable(
        engine, [FakeFactor("a"), FakeFactor("b")],
        policy=resolve_default_policy(),
        artifact_root=tmp_path,
    )
    assert receipt["status"] == "SUCCEEDED"
    db = sqlite3.connect(receipt["state_path"])
    assert receipt["requested_factors"] == 2
    assert receipt["execution_batches"] == 1
    assert receipt["counts"] == {"SUCCEEDED": 2}
    db = sqlite3.connect(receipt["state_path"])
    rows = db.execute("SELECT ordinal,state,attempts,artifact_json FROM outcomes ORDER BY ordinal").fetchall()
    assert [(r[0], r[1], r[2]) for r in rows] == [(0, "SUCCEEDED", 1), (1, "SUCCEEDED", 1)]
    assert all(r[3] for r in rows)


def test_invalid_definition_does_not_abort_valid_ordinal(tmp_path):
    engine = FakeEngine()
    receipt = execute_run_many_durable(
        engine, [{"name": "bad", "callback": lambda: 1}, FakeFactor("good")],
        policy=resolve_default_policy(), artifact_root=tmp_path,
    )
    assert receipt["status"] == "COMPLETED_WITH_ERRORS"
    assert receipt["counts"] == {"REJECTED": 1, "SUCCEEDED": 1}


def test_actual_factor_engine_admits_two_root_wave_and_injects_scope(tmp_path):
    from dataclasses import dataclass
    from factor_engine.api.columns import col
    from factor_engine.api.factor import Factor
    from factor_engine.backend.debug_backend import DebugBackend
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.storage.datasource import DataSource

    @dataclass
    class Source(DataSource):
        def load_column(self, name):
            raise NotImplementedError

    engine = FactorEngine(backend=DebugBackend(), data_source=Source())
    engine.resource_broker = FakeBroker()
    receipt = execute_run_many_durable(
        engine, [Factor("actual_a", col("close")), Factor("actual_b", col("open"))],
        policy=resolve_default_policy(),
        artifact_root=tmp_path,
        execution_scope={"market": "ashare", "frequency": "1d", "universe_id": "all"},
    )
    assert receipt["status"] == "COMPLETED_WITH_ERRORS"
    assert receipt["execution_batches"] == 1
    assert receipt["terminal_count"] == 2
    assert receipt["counts"] == {"FAILED": 2}


def test_actual_pandas_engine_writes_two_verified_numeric_artifacts(tmp_path):
    from factor_engine.api.columns import col
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.expr.cleaned_call import CleanedCall
    from tests.helpers import InMemorySeriesSource
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2025-01-02", "2025-01-03"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    source = InMemorySeriesSource(data={
        "close": pd.Series([1.0, 2.0, 3.0, 4.0], index=idx),
        "open": pd.Series([2.0, 3.0, 4.0, 5.0], index=idx),
    })
    engine = FactorEngine(PandasBackend(), source)
    engine.resource_broker = FakeBroker()
    receipt = execute_run_many_durable(
        engine, [
            Factor("close_factor", col("close")),
            Factor("bad_window", CleanedCall("ts_mean", (col("close"),), (("window", 0),))),
            Factor("open_factor", col("open")),
        ],
        policy=resolve_default_policy(), artifact_root=tmp_path,
        execution_scope={"market": "ashare", "frequency": "1d", "universe_id": "all"},
    )
    assert receipt["status"] == "COMPLETED_WITH_ERRORS"
    assert receipt["execution_batches"] == 1
    db = sqlite3.connect(receipt["state_path"])
    artifacts = db.execute("select artifact_json from outcomes order by ordinal").fetchall()
    assert receipt["counts"] == {"REJECTED": 1, "SUCCEEDED": 2}
    assert sum(row[0] is not None and '"verified": true' in row[0] for row in artifacts) == 2


def test_oom_wave_is_not_blindly_retried(tmp_path):
    class OomEngine(FakeEngine):
        def run_many_parallel(self, factors, **kwargs):
            raise MemoryError("same shape oom")
    receipt = execute_run_many_durable(
        OomEngine(), [FakeFactor("a"), FakeFactor("b")],
        policy=resolve_default_policy(), artifact_root=tmp_path,
    )
    assert receipt["counts"] == {"FAILED": 2}
    db = sqlite3.connect(receipt["state_path"])
    assert db.execute("select max(attempts) from outcomes").fetchone()[0] == 1
    assert receipt["error_groups"][0]["count"] == 2


def test_uncontrolled_sink_timeout_is_failed_unknown_commit(tmp_path):
    receipt = execute_run_many_durable(
        FakeEngine(), [FakeFactor("a"), FakeFactor("b")],
        policy=resolve_default_policy({"sink_flush_seconds": 0.01}),
        artifact_root=tmp_path, sink=_slow_sink,
    )
    assert receipt["counts"] == {"FAILED": 2}
    db = sqlite3.connect(receipt["state_path"])
    rows = db.execute("select state,error_code,commit_state from outcomes order by ordinal").fetchall()
    assert rows[0] == ("FAILED", "UNKNOWN_COMMIT", "UNKNOWN")


def test_one_bad_output_does_not_fail_two_good_outputs(tmp_path):
    class MixedEngine(FakeEngine):
        def run_many_parallel(self, factors, **kwargs):
            values = super().run_many_parallel(factors, **kwargs)["results"]
            values["bad"] = "not numeric"
            return {"results": values}
    receipt = execute_run_many_durable(
        MixedEngine(), [FakeFactor("good_a"), FakeFactor("bad"), FakeFactor("good_b")],
        policy=resolve_default_policy(), artifact_root=tmp_path,
    )
    assert receipt["counts"] == {"FAILED": 1, "SUCCEEDED": 2}
    assert receipt["requested_total"] == 3
    assert receipt["all_requested_terminal"] is True
    assert receipt["all_outputs_valid"] is False


def test_ingestion_timeout_returns_partial_receipt(tmp_path):
    def factors():
        yield FakeFactor("accepted")
        time.sleep(5)
    receipt = execute_run_many_durable(
        FakeEngine(), factors(),
        policy=resolve_default_policy({"input_ingestion_deadline_seconds": 0.05}),
        artifact_root=tmp_path,
    )
    assert receipt["requested_total"] == 1
    assert receipt["input_complete"] is False
    assert receipt["all_requested_terminal"] is False
    assert receipt["counts"] == {"SUCCEEDED": 1}


def test_spawn_factory_executes_without_pickling_engine(tmp_path):
    from factor_engine.api.columns import col
    from factor_engine.api.factor import Factor
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2025-01-02", "2025-01-03"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    construct_marker = tmp_path / "constructed.txt"
    close_marker = tmp_path / "closed.txt"
    broker = FakeBroker()
    # This oracle exercises spawn construction, not undersized read admission.
    # Match the broker's declared 16 MiB resource decision for the source scan.
    broker.current_read_budget = lambda: 16 * 1024 * 1024
    receipt = execute_run_many_durable(
        None, [Factor("spawn_a", col("close")), Factor("spawn_b", col("open"))],
        policy=resolve_default_policy(), artifact_root=tmp_path,
        run_kwargs={"broker": broker}, engine_factory=_build_spawn_engine,
        engine_factory_config={
            "close": pd.Series([1.0, 2.0, 3.0, 4.0], index=idx),
            "open": pd.Series([2.0, 3.0, 4.0, 5.0], index=idx),
            "__construct_marker__": str(construct_marker),
            "__close_marker__": str(close_marker),
        },
    )
    assert receipt["status"] == "SUCCEEDED"
    assert receipt["execution_batches"] == 1
    assert construct_marker.read_text().count("constructed") == 2
    assert close_marker.read_text().count("closed") == 2


def test_large_result_is_rejected_before_pipe_and_small_peer_succeeds(tmp_path):
    class LargeEngine(FakeEngine):
        def run_many_parallel(self, factors, **kwargs):
            idx = pd.MultiIndex.from_arrays(
                [pd.date_range("2020-01-01", periods=100_000, freq="min"), ["A"] * 100_000],
                names=["timestamp", "instrument"],
            )
            small_idx = pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2025-01-02"), "A"), (pd.Timestamp("2025-01-03"), "A")],
                names=["timestamp", "instrument"],
            )
            return {"results": {
                "huge": pd.Series(range(100_000), index=idx, dtype="float64"),
                "small": pd.Series([1.0, 2.0], index=small_idx),
            }}
    broker = TinyTrackingBroker()
    engine = LargeEngine()
    engine.resource_broker = broker
    receipt = execute_run_many_durable(
        engine, [FakeFactor("huge"), FakeFactor("small")],
        policy=resolve_default_policy(), artifact_root=tmp_path,
    )
    assert receipt["counts"] == {"FAILED": 1, "SUCCEEDED": 1}
    assert any(g["code"] == "RESOURCE_REQUIREMENT_EXCEEDS_BUDGET" for g in receipt["error_groups"])
    assert broker.peak <= broker.limit
    assert broker.active == 0


def test_existing_state_paths_are_preserved_fail_closed(tmp_path):
    manifest = tmp_path / "existing-manifest.sqlite3"
    state = tmp_path / "existing-state.sqlite3"
    manifest.write_bytes(b"history-manifest")
    state.write_bytes(b"history-state")
    with pytest.raises(FileExistsError, match="resume is not implemented"):
        execute_run_many_durable(
            FakeEngine(), [FakeFactor("x")], policy=resolve_default_policy(),
            artifact_root=tmp_path, manifest_path=manifest, state_path=state,
        )
    assert manifest.read_bytes() == b"history-manifest"
    assert state.read_bytes() == b"history-state"


def test_lookahead_is_definition_byte_bounded(tmp_path):
    engine = FakeEngine()
    engine.resource_broker = TinyMetadataBroker()
    receipt = execute_run_many_durable(
        engine, [FakeFactor(f"factor_{i}") for i in range(5)],
        policy=resolve_default_policy(), artifact_root=tmp_path,
    )
    assert receipt["counts"] == {"SUCCEEDED": 5}
    assert receipt["execution_batches"] > 1
