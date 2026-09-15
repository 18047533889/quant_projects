"""Whole-graph admission exercises the supervised durable pipeline, not CSE math."""
from dataclasses import replace
import pytest
from factor_engine.runtime.bounded_pipeline import execute_run_many_durable
from factor_engine.runtime.default_execution_policy import resolve_default_policy
from factor_engine.tests.runtime.test_v6_bounded_pipeline import FakeBroker, FakeEngine, FakeFactor


class TrackingLease:
    def __init__(self):
        self.released = False
    def release(self):
        self.released = True


class DAGBroker(FakeBroker):
    def __init__(self, allow=True):
        self.allow = allow
        self.dag_leases = []
    def acquire_memory(self, kind, nbytes, *, lease_id=""):
        if lease_id.startswith("run-dag-metadata:"):
            if not self.allow:
                return None
            lease = TrackingLease()
            self.dag_leases.append(lease)
            return lease
        return super().acquire_memory(kind, nbytes, lease_id=lease_id)


class NativeEnvelopeDAGEngine(FakeEngine):
    def __init__(self, expected_threads):
        super().__init__()
        self.expected_threads = expected_threads

    def run_many_parallel(self, factors, **kwargs):
        import os
        for name in ("POLARS_MAX_THREADS", "DUCKDB_THREADS", "OMP_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            assert os.environ.get(name) == str(self.expected_threads), name
        return super().run_many_parallel(factors, **kwargs)


def build_spawn_dag_engine(config):
    return NativeEnvelopeDAGEngine(config["expected_threads"])


@pytest.mark.parametrize("spawn", [False, True])
@pytest.mark.parametrize("allow,batches,scope", [
    (True, 1, "whole_run_dag"), (False, 3, "bounded_waves"),
])
def test_dag_admission_controls_real_worker_wave_and_releases(tmp_path, allow, batches, scope, spawn):
    engine = FakeEngine()
    engine.resource_broker = DAGBroker(allow)
    receipt = execute_run_many_durable(
        engine, [FakeFactor(f"f{i}") for i in range(6)],
        policy=replace(resolve_default_policy(), initial_lookahead_factors=2),
        artifact_root=tmp_path, automatic_dag=True,
        engine_factory=build_spawn_dag_engine if spawn else None,
        engine_factory_config={"expected_threads": 2 if allow else 1} if spawn else None,
    )
    assert receipt["counts"] == {"SUCCEEDED": 6}
    assert receipt["execution_batches"] == batches
    assert receipt["run_dag"]["execution_scope"] == scope
    assert all(lease.released for lease in engine.resource_broker.dag_leases)
    assert len(engine.resource_broker.dag_leases) == int(allow)
    assert receipt["all_outputs_valid"]
    if spawn:
        import json
        from pathlib import Path
        directory = Path(receipt["execution_ledger"]["directory"])
        ledgers = list(directory.glob("execution-*.json"))
        assert len(ledgers) == batches
        for ledger in ledgers:
            assert ledger.stat().st_size <= 65536
            observed = json.loads(ledger.read_text())
            assert observed["run_id"] == receipt["run_id"]
            assert "results" not in observed
    else:
        assert receipt["execution_ledger"] is None


def test_admitted_lease_is_released_after_preexecution_abort(tmp_path, monkeypatch):
    import factor_engine.runtime.bounded_pipeline as pipeline
    engine = FakeEngine()
    engine.resource_broker = DAGBroker()
    original = pipeline._write_control_receipt
    def fail_seal(path, receipt):
        if path.name == "manifest_identity.json":
            raise RuntimeError("synthetic seal failure after admission")
        return original(path, receipt)
    monkeypatch.setattr(pipeline, "_write_control_receipt", fail_seal)
    with pytest.raises(RuntimeError, match="synthetic seal failure"):
        execute_run_many_durable(
            engine, [FakeFactor(f"f{i}") for i in range(6)],
            policy=replace(resolve_default_policy(), initial_lookahead_factors=2),
            artifact_root=tmp_path, automatic_dag=True,
        )
    assert len(engine.resource_broker.dag_leases) == 1
    assert engine.resource_broker.dag_leases[0].released

def test_quarantine_keeps_dag_metadata_charged_until_retirement_proven(tmp_path, monkeypatch):
    from functools import partial
    import factor_engine.runtime.bounded_pipeline as pipeline
    from factor_engine.runtime.supervised_worker import SupervisedReusableWorker, WorkerQuarantined

    engine = FakeEngine()
    engine.resource_broker = DAGBroker()
    original_execute = SupervisedReusableWorker.execute
    original_close = SupervisedReusableWorker.close

    def execute(self, function, *args, timeout_seconds, lease=None, **kwargs):
        inherited = getattr(self, "_function", None)
        if isinstance(inherited, partial) and inherited.func is pipeline._compute_wave_worker:
            raise WorkerQuarantined("synthetic quarantine before compute start", pid=123)
        return original_execute(
            self, function, *args, timeout_seconds=timeout_seconds, lease=lease, **kwargs
        )

    def close(self):
        inherited = getattr(self, "_function", None)
        if isinstance(inherited, partial) and inherited.func is pipeline._default_worker_write:
            raise RuntimeError("synthetic retirement proof unavailable")
        return original_close(self)

    monkeypatch.setattr(SupervisedReusableWorker, "execute", execute)
    monkeypatch.setattr(SupervisedReusableWorker, "close", close)
    with pytest.raises(WorkerQuarantined) as raised:
        execute_run_many_durable(
            engine, [FakeFactor(f"f{i}") for i in range(6)],
            policy=replace(resolve_default_policy(), initial_lookahead_factors=2),
            artifact_root=tmp_path, automatic_dag=True,
        )
    failure = raised.value
    try:
        assert failure.cleanup_pending is True
        assert failure.dag_metadata_lease is engine.resource_broker.dag_leases[0]
        assert not failure.dag_metadata_lease.released
    finally:
        # No compute/writer process was started in this injected scenario.
        # Release only this test's fake accounting and SQLite handles.
        failure.dag_metadata_lease.release()
        failure.run_state.close()
        failure.run_manifest.close()
        failure.coordinator_lock.release()
