from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pandas as pd
import pytest

from factor_engine.runtime import bounded_pipeline as pipeline


def test_protocol_invalid_direct_receipt_is_wave_error(tmp_path, monkeypatch):
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-01-02"), "A")], names=["time", "instrument"]
    )
    value = pd.Series([1.0], index=idx)
    monkeypatch.setattr(
        "factor_engine.runtime.durable_artifact_sink.required_writer_workspace_bytes",
        lambda value: 1,
    )
    monkeypatch.setattr(
        "factor_engine.runtime.durable_artifact_sink.write_verified_factor_artifact",
        lambda *args, **kwargs: {
            "committed": True, "verified": True, "generation": "wrong"
        },
    )
    sink = pipeline._DirectVerifiedArtifactSink(
        root=tmp_path, run_id="r", policy=object(),
        assignments={"a": (0, "a" * 32)}, writer_bytes=1,
        broker=SimpleNamespace(acquire_memory=lambda *a, **k: None),
    )
    with pytest.raises(pipeline.WorkerProtocolError):
        sink("a", value)
    assert sink.errors == {}


def test_reconciler_disables_nested_retries(monkeypatch):
    entered = []

    @contextmanager
    def disabled():
        entered.append(True)
        yield

    class Engine:
        resource_broker = None

        def run_many_parallel(self, factors, **kwargs):
            assert entered
            return {"results": {"a": object()}}

    reconciler = pipeline._SpawnFactoryArtifactReconciler(
        lambda config: Engine(), {}, SimpleNamespace(acquire_memory=lambda *a, **k: None)
    )
    monkeypatch.setattr(
        "factor_engine.runtime.adaptive_batch_scheduler.disable_inner_retries_for_v2",
        disabled,
    )
    monkeypatch.setattr(
        "factor_engine.runtime.durable_artifact_sink.required_writer_workspace_bytes",
        lambda value: 0,
    )
    monkeypatch.setattr(
        "factor_engine.runtime.durable_artifact_sink.reconcile_factor_artifact",
        lambda *args, **kwargs: {"verified": True},
    )
    assert reconciler(None, {}, "/tmp", "r", object(), 0, "a", 1, "a" * 32) == {
        "verified": True
    }


def test_compute_exit_precedes_reclaim_and_quarantine_blocks_reclaim():
    events = []
    proxy = SimpleNamespace(client_id="compute-client")
    worker = SimpleNamespace(close=lambda: events.append("exit"))
    ipc = SimpleNamespace(reclaim_client=lambda client: events.append(("reclaim", client)))
    pipeline._retire_direct_compute_authority(worker, ipc, proxy)
    assert events == ["exit", ("reclaim", "compute-client")]

    from factor_engine.runtime.supervised_worker import WorkerQuarantined

    events.clear()

    def quarantined():
        events.append("quarantine")
        raise WorkerQuarantined("still alive", pid=123)

    worker.close = quarantined
    with pytest.raises(WorkerQuarantined):
        pipeline._retire_direct_compute_authority(worker, ipc, proxy)
    assert events == ["quarantine"]
