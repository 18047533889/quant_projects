import os
import json
import sqlite3
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

from factor_engine.runtime.bounded_pipeline import execute_run_many_durable
from factor_engine.runtime.default_execution_policy import resolve_default_policy
from factor_engine.runtime.batch_service import _consume_bounded_roots


class Broker:
    def summary(self):
        return {"kind": "v7-direct-test"}

    def automatic_result_queue_budget(self):
        return 4 * 1024 * 1024

    def acquire_protected_egress(self, *args, **kwargs):
        return Lease(), Lease()

    def acquire_memory(self, *args, **kwargs):
        return Lease()

    def try_reserve(self, *args, **kwargs):
        return Lease()

    def current_read_budget(self):
        return 16 * 1024 * 1024

    def current_sink_budget(self):
        return 1024 * 1024

    def execution_budget(self):
        return 16 * 1024 * 1024

    def cpu_budget(self):
        return 2

    hard_cpu_slots = 2
    hard_memory_limit = 64 * 1024 * 1024


class EgressTrackingBroker(Broker):
    def __init__(self):
        self.egress_requests = []

    def acquire_protected_egress(self, queue_bytes, writer_bytes, **kwargs):
        self.egress_requests.append((queue_bytes, writer_bytes))
        return Lease(), Lease()


class Lease:
    def release(self):
        pass


class TinyQueueBroker(Broker):
    def __init__(self, allow_extra=True):
        self.allow_extra = allow_extra
        self.memory_requests = []

    def automatic_result_queue_budget(self):
        return 64 * 1024

    def acquire_memory(self, kind, *args, **kwargs):
        from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind
        self.memory_requests.append(kind)
        # This fixture denies extra result/write workspace, not input admission.
        # Default v2 ingestion now uses the same broker before compute starts.
        return Lease() if kind == MemoryLeaseKind.MANIFEST_BUFFER or self.allow_extra else None


class Factor:
    def __init__(self, name):
        self.name = name


class AckLossEngine:
    run_mode = "research"

    def __init__(self, config):
        self.marker = Path(config["marker"])
        self.modes = Path(config["modes"])

    def compile(self, factor):
        return factor

    @staticmethod
    def _value(name):
        index = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2025-01-02"), "000001.SZ")],
            names=["timestamp", "instrument"],
        )
        return pd.Series([1.25], index=index, name=name)

    def run_many_parallel(self, factors, *, result_policy, sink=None, **kwargs):
        with self.modes.open("a") as stream:
            stream.write(f"{result_policy}:{os.getpid()}\n")
        results = {factor.name: self._value(factor.name) for factor in factors}
        if result_policy == "sink":
            for name, value in results.items():
                sink(name, value)
            if not self.marker.exists():
                self.marker.write_text("artifact-written-before-ack-loss")
                os._exit(17)
            return {"results": {}}
        return {"results": results}

    def close(self):
        pass


def build_ack_loss_engine(config):
    return AckLossEngine(config)


class LargeDirectEngine(AckLossEngine):
    @staticmethod
    def _value(name):
        index = pd.MultiIndex.from_arrays(
            [
                pd.date_range("2025-01-01", periods=20_000, freq="min"),
                ["000001.SZ"] * 20_000,
            ],
            names=["timestamp", "instrument"],
        )
        return pd.Series(range(20_000), index=index, dtype=float, name=name)

    def run_many_parallel(self, factors, *, result_policy, sink=None, **kwargs):
        with self.modes.open("a") as stream:
            stream.write(f"{result_policy}:{os.getpid()}\n")
        results = {factor.name: self._value(factor.name) for factor in factors}
        if result_policy == "sink":
            for name, value in results.items():
                sink(name, value)
            return {"results": {}}
        return {"results": results}


def build_large_direct_engine(config):
    return LargeDirectEngine(config)


class CrossWaveOverlapEngine(AckLossEngine):
    def run_many_parallel(self, factors, *, result_policy, sink=None, **kwargs):
        factor = factors[0]
        root = Path(self.marker)
        quota = self.resource_broker.cpu_budget()
        with self.modes.open("a") as stream:
            stream.write(f"{factor.name}:start:{time.monotonic()}:{os.getpid()}:{quota}\n")
        if factor.name == "second":
            (root / "second-compute-started").write_text("started")
        value = self._value(factor.name)
        if result_policy == "sink":
            if factor.name == "first":
                (root / "first-write-started").write_text("started")
                deadline = time.monotonic() + 5
                while not (root / "second-compute-started").exists():
                    if time.monotonic() >= deadline:
                        raise RuntimeError("serial counterexample: next compute did not overlap write")
                    time.sleep(0.01)
                with self.modes.open("a") as stream:
                    stream.write(f"first:unblocked:{time.monotonic()}:{os.getpid()}:{quota}\n")
            sink(factor.name, value)
            with self.modes.open("a") as stream:
                stream.write(f"{factor.name}:done:{time.monotonic()}:{os.getpid()}:{quota}\n")
            return {"results": {}}
        return {"results": {factor.name: value}}


def build_cross_wave_overlap_engine(config):
    return CrossWaveOverlapEngine(config)


class PairEpochEngine(AckLossEngine):
    def __init__(self, config):
        super().__init__(config)
        self.compiles = Path(config["compiles"])

    def compile(self, factor):
        with self.compiles.open("a") as stream:
            stream.write(f"{factor.name}\n")
        return factor


def build_pair_epoch_engine(config):
    return PairEpochEngine(config)


class PhaseHangEngine(AckLossEngine):
    def run_many_parallel(self, factors, *, result_policy, sink=None, **kwargs):
        factor = factors[0]
        if self.marker.name == "write":
            from factor_engine.runtime.supervised_worker import emit_worker_progress
            emit_worker_progress("WRITE", ordinal=0)
        self.marker.write_text(f"{factor.name}:{os.getpid()}")
        time.sleep(10)


def build_phase_hang_engine(config):
    return PhaseHangEngine(config)


class ZeroCpuBroker(Broker):
    hard_cpu_slots = 0

    def cpu_budget(self):
        return 0


class CancellationPhaseEngine(AckLossEngine):
    def run_many_parallel(self, factors, *, result_policy, sink=None, **kwargs):
        from factor_engine.runtime.supervised_worker import emit_worker_progress
        factor = factors[0]
        if factor.name == "first":
            emit_worker_progress("WRITE", ordinal=0)
        marker = self.marker / f"{factor.name}.pid"
        marker.write_text(str(os.getpid()))
        time.sleep(10)


def build_cancellation_phase_engine(config):
    return CancellationPhaseEngine(config)


class VerifiedThenHangEngine(AckLossEngine):
    def run_many_parallel(self, factors, *, result_policy, sink=None, **kwargs):
        factor = factors[0]
        if factor.name == "first":
            sink(factor.name, self._value(factor.name))
            (self.marker / "first.done").write_text("verified")
            return {"results": {}}
        (self.marker / "second.started").write_text(str(os.getpid()))
        time.sleep(10)


def build_verified_then_hang_engine(config):
    return VerifiedThenHangEngine(config)


class ContinuousRefillEngine(AckLossEngine):
    def __init__(self, config):
        super().__init__(config)
        self.hold = config.get("hold", "b")
        self.hang_compile = bool(config.get("hang_compile", False))

    def compile(self, factor):
        if self.hang_compile and factor.name == "c":
            (self.marker / "compile-c-started").write_text(str(os.getpid()))
            time.sleep(10)
        return factor

    def run_many_parallel(self, factors, *, result_policy, sink=None, **kwargs):
        factor = factors[0]
        with self.modes.open("a") as stream:
            stream.write(f"{factor.name}:start:{time.monotonic()}:{os.getpid()}\n")
        if factor.name == self.hold:
            deadline = time.monotonic() + 15
            while not (self.marker / f"release-{self.hold}").exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("C did not refill while B was active")
                time.sleep(.01)
        sink(factor.name, self._value(factor.name))
        with self.modes.open("a") as stream:
            stream.write(f"{factor.name}:done:{time.monotonic()}:{os.getpid()}\n")
        return {"results": {}}


def build_continuous_refill_engine(config):
    config = dict(config)
    config["marker"] = Path(config["marker"])
    return ContinuousRefillEngine(config)


def build_real_engine_with_pid_probe(config):
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.runtime.engine import FactorEngine
    from tests.helpers import InMemorySeriesSource

    modes = Path(config.pop("__modes__"))
    engine = FactorEngine(PandasBackend(), InMemorySeriesSource(data=config))
    original = engine.run_many_parallel

    def probed(factors, *, result_policy, sink=None, **kwargs):
        with modes.open("a") as stream:
            stream.write(f"{result_policy}:{os.getpid()}\n")
        return original(
            factors, result_policy=result_policy, sink=sink, **kwargs
        )

    engine.run_many_parallel = probed
    return engine


def test_default_spawn_sinks_values_and_lost_descriptor_ack_exactly_reconciles(tmp_path):
    marker = tmp_path / "ack-lost"
    modes = tmp_path / "modes"
    receipt = execute_run_many_durable(
        None,
        [Factor("direct")],
        policy=resolve_default_policy(),
        artifact_root=tmp_path,
        run_kwargs={"broker": Broker()},
        engine_factory=build_ack_loss_engine,
        engine_factory_config={"marker": str(marker), "modes": str(modes)},
    )

    assert receipt["status"] == "SUCCEEDED"
    assert receipt["counts"] == {"SUCCEEDED": 1}
    mode_records = modes.read_text().splitlines()
    assert [item.split(":")[0] for item in mode_records] == ["sink", "return"]
    assert mode_records[0].split(":")[1] != mode_records[1].split(":")[1]
    with sqlite3.connect(receipt["state_path"]) as db:
        attempts, commit_state = db.execute(
            "SELECT attempts,commit_state FROM outcomes WHERE ordinal=0"
        ).fetchone()
    assert attempts == 2
    assert commit_state == "VERIFIED"
    manifests = list((tmp_path / receipt["run_id"] / "values").glob("*/manifest.json"))
    assert len(manifests) == 1


def test_bounded_root_queue_overlaps_next_compute_with_current_sink():
    sink_started = threading.Event()
    second_compute_finished = threading.Event()
    consumed = []

    def execute(item):
        if item == 1:
            assert sink_started.wait(1)
            second_compute_finished.set()
        return item, f"value-{item}"

    def consume(item, value):
        consumed.append((item, value))
        if item == 0:
            sink_started.set()
            assert second_compute_finished.wait(1)
            time.sleep(0.01)

    _consume_bounded_roots([0, 1], execute, consume, max_workers=2)
    assert sorted(consumed) == [(0, "value-0"), (1, "value-1")]


def test_large_default_spawn_output_bypasses_result_pickle_queue(tmp_path):
    modes = tmp_path / "modes"
    receipt = execute_run_many_durable(
        None,
        [Factor("large")],
        policy=resolve_default_policy(),
        artifact_root=tmp_path,
        run_kwargs={"broker": TinyQueueBroker(allow_extra=True)},
        engine_factory=build_large_direct_engine,
        engine_factory_config={"marker": str(tmp_path / "unused"), "modes": str(modes)},
    )
    assert receipt["status"] == "SUCCEEDED"
    assert [item.split(":")[0] for item in modes.read_text().splitlines()] == ["sink"]


def test_atomic_large_output_is_rejected_before_write_without_extra_lease(tmp_path):
    from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind
    modes = tmp_path / "modes"
    broker = TinyQueueBroker(allow_extra=False)
    receipt = execute_run_many_durable(
        None,
        [Factor("large")],
        policy=resolve_default_policy(),
        artifact_root=tmp_path,
        run_kwargs={"broker": broker},
        engine_factory=build_large_direct_engine,
        engine_factory_config={"marker": str(tmp_path / "unused"), "modes": str(modes)},
    )
    assert receipt["status"] == "COMPLETED_WITH_ERRORS"
    assert receipt["counts"] == {"FAILED": 1}
    assert receipt["error_groups"][0]["code"] == "RESOURCE_REQUIREMENT_EXCEEDS_BUDGET"
    assert MemoryLeaseKind.MANIFEST_BUFFER in broker.memory_requests
    assert MemoryLeaseKind.WRITER_BATCH in broker.memory_requests
    assert modes.read_text().splitlines()[0].split(":")[0] == "sink"
    values = tmp_path / receipt["run_id"] / "values"
    assert not values.exists() or not any(values.iterdir())


def test_two_default_spawn_waves_retire_pid_and_bind_fresh_generation(tmp_path):
    marker = tmp_path / "already-normal"
    marker.write_text("do-not-inject-ack-loss")
    modes = tmp_path / "modes"
    receipt = execute_run_many_durable(
        None,
        [Factor("first"), Factor("second")],
        policy=resolve_default_policy({"initial_lookahead_factors": 1}),
        artifact_root=tmp_path,
        run_kwargs={"broker": Broker()},
        engine_factory=build_ack_loss_engine,
        engine_factory_config={"marker": str(marker), "modes": str(modes)},
    )
    assert receipt["status"] == "SUCCEEDED"
    records = modes.read_text().splitlines()
    assert [item.split(":")[0] for item in records] == ["sink", "sink"]
    assert records[0].split(":")[1] != records[1].split(":")[1]


def test_default_two_slot_pipeline_overlaps_previous_write_with_next_compute(tmp_path):
    receipt = execute_run_many_durable(
        None, [Factor("first"), Factor("second")],
        policy=resolve_default_policy({"initial_lookahead_factors": 1}),
        artifact_root=tmp_path / "artifacts",
        run_kwargs={"broker": Broker()},
        engine_factory=build_cross_wave_overlap_engine,
        engine_factory_config={"marker": str(tmp_path), "modes": str(tmp_path / "unused")},
    )
    assert receipt["status"] == "SUCCEEDED"
    assert (tmp_path / "first-write-started").is_file()
    assert (tmp_path / "second-compute-started").is_file()
    events = [line.split(":") for line in (tmp_path / "unused").read_text().splitlines()]
    first_start = next(item for item in events if item[:2] == ["first", "start"])
    second_start = next(item for item in events if item[:2] == ["second", "start"])
    first_unblocked = next(item for item in events if item[:2] == ["first", "unblocked"])
    assert float(first_start[2]) < float(second_start[2]) < float(first_unblocked[2])
    assert first_start[3] != second_start[3]
    assert int(first_start[4]) + int(second_start[4]) <= Broker.hard_cpu_slots


def test_real_engine_two_wave_default_uses_fresh_compute_pid(tmp_path):
    from factor_engine.api.columns import col
    from factor_engine.api.factor import Factor as RealFactor

    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2025-01-02", "2025-01-03"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    modes = tmp_path / "real-modes"
    receipt = execute_run_many_durable(
        None,
        [RealFactor("close_factor", col("close")), RealFactor("open_factor", col("open"))],
        policy=resolve_default_policy({"initial_lookahead_factors": 1}),
        artifact_root=tmp_path,
        run_kwargs={"broker": Broker()},
        engine_factory=build_real_engine_with_pid_probe,
        engine_factory_config={
            "close": pd.Series([1.0, 2.0, 3.0, 4.0], index=index),
            "open": pd.Series([2.0, 3.0, 4.0, 5.0], index=index),
            "__modes__": str(modes),
        },
    )
    assert receipt["status"] == "SUCCEEDED"
    records = modes.read_text().splitlines()
    assert [item.split(":")[0] for item in records] == ["sink", "sink"]
    assert records[0].split(":")[1] != records[1].split(":")[1]


def test_two_slot_egress_is_partitioned_and_prefetched_compile_is_not_repeated(tmp_path):
    modes = tmp_path / "modes"
    compiles = tmp_path / "compiles"
    (tmp_path / "normal").write_text("no-ack-loss")
    broker = EgressTrackingBroker()
    receipt = execute_run_many_durable(
        None, [Factor(name) for name in ("a", "b", "c", "d")],
        policy=resolve_default_policy({"initial_lookahead_factors": 1}),
        artifact_root=tmp_path / "artifacts", run_kwargs={"broker": broker},
        engine_factory=build_pair_epoch_engine,
        engine_factory_config={
            "marker": str(tmp_path / "normal"), "modes": str(modes),
            "compiles": str(compiles),
        },
    )
    assert receipt["status"] == "SUCCEEDED"
    assert compiles.read_text().splitlines() == ["a", "b", "c", "d"]
    assert len(broker.egress_requests) == 1
    total_queue, total_writer = broker.egress_requests[0]
    manifests = [json.loads(path.read_text()) for path in
                 (tmp_path / "artifacts" / receipt["run_id"] / "values").glob("*/manifest.json")]
    assert len(manifests) == 4
    slot_budgets = {item["performance"]["workspace_budget_bytes"] for item in manifests}
    assert slot_budgets == {total_writer // 2}
    assert 2 * next(iter(slot_budgets)) <= total_writer


@pytest.mark.parametrize("phase,code,commit_state", [
    ("compute", "COMPUTE_TIMEOUT", "NOT_STARTED"),
    ("write", "WRITE_TIMEOUT", "UNKNOWN"),
])
def test_direct_timeout_reports_last_worker_phase(tmp_path, phase, code, commit_state):
    marker = tmp_path / phase
    receipt = execute_run_many_durable(
        None, [Factor("timed")],
        policy=resolve_default_policy({
            "compute_unknown_seconds": 1,
            "job_unknown_seconds": 3,
            "job_min_seconds": 3,
            "job_max_seconds": 3,
            "cooperative_cancel_grace_seconds": .01,
            "worker_exit_observation_seconds": .5,
        }),
        artifact_root=tmp_path / "artifacts",
        run_kwargs={"broker": Broker()},
        engine_factory=build_phase_hang_engine,
        engine_factory_config={"marker": str(marker), "modes": str(tmp_path / "modes")},
    )
    assert marker.is_file()
    assert receipt["counts"] == {"FAILED": 1}
    assert receipt["error_groups"][0]["code"] == code
    with sqlite3.connect(receipt["state_path"]) as db:
        assert db.execute(
            "select error_code,commit_state from outcomes where ordinal=0"
        ).fetchone() == (code, commit_state)


def test_zero_cpu_authority_terminalizes_without_starting_engine_or_attempt(tmp_path):
    marker = tmp_path / "engine-entered"
    receipt = execute_run_many_durable(
        None, [Factor("zero")], policy=resolve_default_policy(),
        artifact_root=tmp_path / "artifacts",
        run_kwargs={"broker": ZeroCpuBroker()},
        engine_factory=build_phase_hang_engine,
        engine_factory_config={"marker": str(marker), "modes": str(tmp_path / "modes")},
    )
    assert not marker.exists()
    assert receipt["counts"] == {"FAILED": 1}
    assert receipt["execution_batches"] == 0
    assert receipt["error_groups"][0]["code"] == "RESOURCE_REQUIREMENT_EXCEEDS_BUDGET"
    with sqlite3.connect(receipt["state_path"]) as db:
        assert db.execute(
            "select attempts,error_code from outcomes where ordinal=0"
        ).fetchone() == (0, "RESOURCE_REQUIREMENT_EXCEEDS_BUDGET")


def test_cancellation_retires_both_overlap_slots_and_preserves_per_factor_commit_state(tmp_path):
    from factor_engine.runtime.exceptions import Cancellation, CancellationToken

    token = CancellationToken()
    captured = []

    def run():
        try:
            execute_run_many_durable(
                None, [Factor("first"), Factor("second")],
                policy=resolve_default_policy({
                    "initial_lookahead_factors": 1,
                    "cooperative_cancel_grace_seconds": .01,
                    "worker_exit_observation_seconds": .5,
                }),
                artifact_root=tmp_path / "artifacts",
                run_kwargs={"broker": Broker()},
                engine_factory=build_cancellation_phase_engine,
                engine_factory_config={
                    "marker": str(tmp_path), "modes": str(tmp_path / "modes")
                },
                cancellation_token=token,
            )
        except BaseException as exc:
            captured.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 4
    while not all((tmp_path / f"{name}.pid").is_file()
                  for name in ("first", "second")):
        assert time.monotonic() < deadline
        time.sleep(.01)
    pids = [int((tmp_path / f"{name}.pid").read_text())
            for name in ("first", "second")]
    token.cancel()
    thread.join(5)
    assert not thread.is_alive()
    assert len(captured) == 1 and isinstance(captured[0], Cancellation)
    receipt = json.loads(Path(captured[0].receipt_path).read_text())
    with sqlite3.connect(receipt["state_path"]) as db:
        assert db.execute(
            "select name,state,commit_state from outcomes order by ordinal"
        ).fetchall() == [
            ("first", "CANCELLED", "UNKNOWN"),
            ("second", "CANCELLED", "NOT_STARTED"),
        ]
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_verified_current_is_persisted_before_prefetch_quarantine(tmp_path, monkeypatch):
    from factor_engine.runtime.exceptions import CancellationToken
    from factor_engine.runtime.supervised_worker import (
        AsyncWorkerCall, WorkerQuarantined,
    )

    original_cancel = AsyncWorkerCall.cancel_and_retire
    def injected_cancel(self):
        progress = dict(self._worker.last_progress or {})
        original_cancel(self)
        if progress.get("ordinal") == 1:
            raise WorkerQuarantined("injected post-retirement cleanup uncertainty")
    monkeypatch.setattr(AsyncWorkerCall, "cancel_and_retire", injected_cancel)
    token = CancellationToken()
    captured = []
    cleanup_allowed = threading.Event()
    class TrackedLease:
        released = False
        def release(self):
            self.released = True
    egress = [TrackedLease(), TrackedLease()]
    class RetentionBroker(Broker):
        def acquire_protected_egress(self, *args, **kwargs):
            return tuple(egress)
    def run():
        try:
            execute_run_many_durable(
                None, [Factor("first"), Factor("second")],
                policy=resolve_default_policy({
                    "initial_lookahead_factors": 1,
                    "cooperative_cancel_grace_seconds": .01,
                    "worker_exit_observation_seconds": .5,
                }),
                artifact_root=tmp_path / "artifacts",
                run_kwargs={"broker": RetentionBroker()},
                engine_factory=build_verified_then_hang_engine,
                engine_factory_config={
                    "marker": str(tmp_path), "modes": str(tmp_path / "modes")
                }, cancellation_token=token,
            )
        except BaseException as exc:
            captured.append(exc)
            cleanup_allowed.wait(5)
            if hasattr(exc, "run_state"):
                exc.run_state.close()
                exc.run_manifest.close()
                exc.coordinator_lock.release()

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 5
    state_path = None
    while True:
        assert time.monotonic() < deadline
        candidates = list((tmp_path / "artifacts").glob("*/state.sqlite3"))
        if candidates and (tmp_path / "second.started").is_file():
            state_path = candidates[0]
            try:
                with sqlite3.connect(state_path) as db:
                    row = db.execute(
                        "select state,commit_state from outcomes where ordinal=0"
                    ).fetchone()
            except sqlite3.Error:
                row = None
            if row == ("SUCCEEDED", "VERIFIED"):
                break
        time.sleep(.01)
    token.cancel()
    try:
        capture_deadline = time.monotonic() + 5
        while not captured:
            assert time.monotonic() < capture_deadline
            time.sleep(.01)
        assert len(captured) == 1 and isinstance(captured[0], WorkerQuarantined)
        receipt = json.loads(Path(captured[0].receipt_path).read_text())
        with sqlite3.connect(receipt["state_path"]) as db:
            assert db.execute(
                "select name,state,commit_state from outcomes order by ordinal"
            ).fetchall() == [
                ("first", "SUCCEEDED", "VERIFIED"),
                ("second", "CANCELLED", "UNKNOWN"),
            ]
        assert captured[0].coordinator_lock.owned is True
        assert not any(lease.released for lease in egress)
    finally:
        cleanup_allowed.set()
        thread.join(5)
    assert not thread.is_alive()


def test_two_slots_refill_c_after_a_finishes_without_waiting_for_b(tmp_path):
    events = tmp_path / "events"
    receipt_box = []
    def run():
        receipt_box.append(execute_run_many_durable(
            None, [Factor(name) for name in ("a", "b", "c")],
            policy=resolve_default_policy({"initial_lookahead_factors": 1}),
            artifact_root=tmp_path / "artifacts", run_kwargs={"broker": Broker()},
            engine_factory=build_continuous_refill_engine,
            engine_factory_config={"marker": str(tmp_path), "modes": str(events)},
        ))

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 5
    while True:
        assert time.monotonic() < deadline
        lines = events.read_text().splitlines() if events.exists() else []
        if any(line.startswith("c:start:") for line in lines):
            break
        time.sleep(.01)
    assert not any(line.startswith("b:done:") for line in lines)
    (tmp_path / "release-b").write_text("release")
    thread.join(10)
    assert not thread.is_alive()
    receipt = receipt_box[0]
    assert receipt["status"] == "SUCCEEDED"
    final_events = [line.split(":") for line in events.read_text().splitlines()]
    starts = {item[0]: (float(item[2]), item[3]) for item in final_events
              if item[1] == "start"}
    dones = {item[0]: float(item[2]) for item in final_events if item[1] == "done"}
    assert dones["a"] < starts["c"][0] < dones["b"]
    assert len({starts[name][1] for name in ("a", "b", "c")}) == 3
    transitions = sorted(
        [(starts[name][0], 1) for name in starts]
        + [(dones[name], -1) for name in dones],
        key=lambda item: (item[0], item[1]),
    )
    active = peak = 0
    for _when, delta in transitions:
        active += delta
        peak = max(peak, active)
    assert peak == 2
    with sqlite3.connect(receipt["state_path"]) as db:
        assert db.execute(
            "select name,attempts,state from outcomes order by ordinal"
        ).fetchall() == [("a", 1, "SUCCEEDED"), ("b", 1, "SUCCEEDED"),
                         ("c", 1, "SUCCEEDED")]


def test_two_slots_refill_c_after_b_finishes_without_waiting_for_a(tmp_path):
    events = tmp_path / "events-reverse"
    receipt_box = []

    def run():
        receipt_box.append(execute_run_many_durable(
            None, [Factor(name) for name in ("a", "b", "c")],
            policy=resolve_default_policy({"initial_lookahead_factors": 1}),
            artifact_root=tmp_path / "artifacts-reverse", run_kwargs={"broker": Broker()},
            engine_factory=build_continuous_refill_engine,
            engine_factory_config={
                "marker": str(tmp_path), "modes": str(events), "hold": "a",
            },
        ))

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 4
    observed_refill = False
    while time.monotonic() < deadline:
        lines = events.read_text().splitlines() if events.exists() else []
        if any(line.startswith("c:start:") for line in lines):
            observed_refill = True
            break
        time.sleep(.01)
    (tmp_path / "release-a").write_text("release")
    thread.join(10)
    assert not thread.is_alive()
    assert observed_refill, "B released its slot but C was not started while A remained active"
    receipt = receipt_box[0]
    assert receipt["status"] == "SUCCEEDED"
    final_events = [line.split(":") for line in events.read_text().splitlines()]
    starts = {item[0]: (float(item[2]), item[3]) for item in final_events
              if item[1] == "start"}
    dones = {item[0]: float(item[2]) for item in final_events if item[1] == "done"}
    assert dones["b"] < starts["c"][0] < dones["a"]
    assert len({starts[name][1] for name in ("a", "b", "c")}) == 3
    transitions = sorted(
        [(starts[name][0], 1) for name in starts]
        + [(dones[name], -1) for name in dones],
        key=lambda item: (item[0], item[1]),
    )
    active = peak = 0
    for _when, delta in transitions:
        active += delta
        peak = max(peak, active)
    assert peak == 2
    with sqlite3.connect(receipt["state_path"]) as db:
        assert db.execute(
            "select name,attempts,state from outcomes order by ordinal"
        ).fetchall() == [("a", 1, "SUCCEEDED"), ("b", 1, "SUCCEEDED"),
                         ("c", 1, "SUCCEEDED")]


def test_cancellation_interrupts_refill_compile_and_active_compute_slot(tmp_path):
    from factor_engine.runtime.exceptions import Cancellation, CancellationToken

    token = CancellationToken()
    captured = []

    def run():
        try:
            execute_run_many_durable(
                None, [Factor(name) for name in ("a", "b", "c")],
                policy=resolve_default_policy({
                    "initial_lookahead_factors": 1,
                    "cooperative_cancel_grace_seconds": .1,
                    "worker_exit_observation_seconds": 2,
                }),
                artifact_root=tmp_path / "artifacts-cancel-refill",
                run_kwargs={"broker": Broker()},
                engine_factory=build_continuous_refill_engine,
                engine_factory_config={
                    "marker": str(tmp_path), "modes": str(tmp_path / "cancel-events"),
                    "hold": "b", "hang_compile": True,
                },
                cancellation_token=token,
            )
        except BaseException as exc:
            captured.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 5
    while not (tmp_path / "compile-c-started").exists():
        assert time.monotonic() < deadline
        time.sleep(.01)
    token.cancel()
    thread.join(5)
    assert not thread.is_alive()
    assert len(captured) == 1 and isinstance(captured[0], Cancellation)
    compile_pid = int((tmp_path / "compile-c-started").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(compile_pid, 0)
