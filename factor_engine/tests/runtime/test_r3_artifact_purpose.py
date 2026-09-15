"""Synthetic U08/U09 purpose/artifact contracts; no real-data approval implied."""
import hashlib
import json
from pathlib import Path
import gc
from contextlib import contextmanager

import pandas as pd
import pytest

from factor_engine.runtime import bounded_pipeline as pipeline
from factor_engine.runtime.default_execution_policy import (
    ExecutionPurpose, resolve_default_policy,
)
from factor_engine.runtime.durable_artifact_sink import (
    read_verified_factor_artifact, write_verified_factor_artifact,
)
from factor_engine.tests.runtime.test_v6_bounded_pipeline import FakeEngine, FakeFactor
from factor_engine.tests.runtime.test_v6_bounded_pipeline import FakeBroker


@contextmanager
def _isolated_worker_broker_globals(monkeypatch):
    """Model a fresh spawned worker without weakening broker authority checks."""
    import factor_engine.runtime.resource_broker as broker_module

    with monkeypatch.context() as isolated:
        for name in (
            "_V2_BROKER", "_V2_BROKER_PID", "_V2_BROKER_POLICY_KEY",
            "_V2_WORKER_PROXY",
        ):
            isolated.setattr(broker_module, name, None)
        yield


def _value():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    return pd.Series([1.0, 2.0, 3.0, 4.0], index=index)


def _identity(purpose):
    return {"deployment_digest": "d" * 64,
            "execution_purpose": purpose.to_dict(), "scope": {"fixture": True}}


def _digest(identity):
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("purpose_name", ["research_compute", "production_compute"])
def test_manifest_receipt_and_managed_read_preserve_unverified_purpose(tmp_path, purpose_name):
    purpose = ExecutionPurpose(purpose_name)
    identity_digest = _digest(_identity(purpose))
    receipt = write_verified_factor_artifact(
        tmp_path, "a" * 32, 0, "alpha", _value(),
        policy=resolve_default_policy(), budget_bytes=8 * 1024 * 1024,
        generation="b" * 32, execution_purpose=purpose,
        run_identity_digest=identity_digest,
    )
    manifest = json.loads(Path(receipt["path"]).read_text())
    assert manifest["execution_purpose"] == purpose.to_dict()
    assert manifest["run_identity_digest"] == identity_digest
    assert receipt["assurance"] == "UNVERIFIED"
    assert receipt["publication_authorized"] is False
    restored = read_verified_factor_artifact(
        receipt, tmp_path, policy=resolve_default_policy(),
        budget_bytes=8 * 1024 * 1024, broker=FakeBroker(), execution_purpose=purpose,
        run_identity_digest=identity_digest,
    )
    pd.testing.assert_series_equal(restored["value"], _value().rename("factor_value"))
    assert restored["artifact_id"] == f'{"a" * 32}:0:{"b" * 32}'
    assert restored["manifest_projection"]["run_identity"] == {
        "digest": identity_digest
    }
    assert restored["assurance"] == "UNVERIFIED"
    assert restored["publication_authorized"] is False


def test_managed_read_rejects_purpose_or_identity_rebinding(tmp_path):
    research = ExecutionPurpose()
    digest = _digest(_identity(research))
    receipt = write_verified_factor_artifact(
        tmp_path, "a" * 32, 0, "alpha", _value(),
        policy=resolve_default_policy(), budget_bytes=8 * 1024 * 1024,
        execution_purpose=research, run_identity_digest=digest,
    )
    with pytest.raises(ValueError, match="purpose or run identity"):
        read_verified_factor_artifact(
            receipt, tmp_path, policy=resolve_default_policy(),
            budget_bytes=8 * 1024 * 1024, broker=FakeBroker(),
            execution_purpose=ExecutionPurpose("production_compute"),
            run_identity_digest=digest,
        )
    with pytest.raises(ValueError, match="purpose or run identity"):
        read_verified_factor_artifact(
            receipt, tmp_path, policy=resolve_default_policy(),
            budget_bytes=8 * 1024 * 1024, broker=FakeBroker(), execution_purpose=research,
            run_identity_digest="c" * 64,
        )


def test_managed_read_holds_real_lease_and_checks_receipt_coverage(tmp_path):
    class Lease:
        released = False
        def release(self):
            self.released = True
    class Broker:
        def __init__(self):
            self.lease = None
            self.lease_ids = []
        def acquire_memory(self, *args, **kwargs):
            self.lease_ids.append(kwargs["lease_id"])
            self.lease = Lease()
            return self.lease

    purpose = ExecutionPurpose()
    digest = _digest(_identity(purpose))
    receipt = write_verified_factor_artifact(
        tmp_path, "a" * 32, 0, "alpha", _value(),
        policy=resolve_default_policy(), budget_bytes=8 * 1024 * 1024,
        execution_purpose=purpose, run_identity_digest=digest,
    )
    broker = Broker()
    restored = read_verified_factor_artifact(
        receipt, tmp_path, policy=resolve_default_policy(),
        budget_bytes=8 * 1024 * 1024, broker=broker,
        execution_purpose=purpose, run_identity_digest=digest,
    )
    view = restored["value"].to_numpy(copy=False)
    del restored
    gc.collect()
    assert broker.lease is not None and not broker.lease.released
    assert len(broker.lease_ids) == len(set(broker.lease_ids)) == 2
    del view
    gc.collect()
    assert broker.lease.released

    with pytest.raises(ValueError, match="row/byte coverage"):
        read_verified_factor_artifact(
            {**receipt, "rows": receipt["rows"] + 1}, tmp_path,
            policy=resolve_default_policy(), budget_bytes=8 * 1024 * 1024,
            broker=Broker(), execution_purpose=purpose,
            run_identity_digest=digest,
        )


def test_managed_read_rejects_duplicate_chunk_filename(tmp_path):
    purpose = ExecutionPurpose()
    digest = _digest(_identity(purpose))
    receipt = write_verified_factor_artifact(
        tmp_path, "a" * 32, 0, "alpha", _value(),
        policy=resolve_default_policy(), budget_bytes=8 * 1024 * 1024,
        execution_purpose=purpose, run_identity_digest=digest,
    )
    path = Path(receipt["path"])
    manifest = json.loads(path.read_text())
    manifest["chunks"].append({**manifest["chunks"][0], "offset": manifest["rows"]})
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(payload)
    forged = {**receipt, "sha256": hashlib.sha256(payload).hexdigest(),
              "rows": receipt["rows"] * 2, "bytes": receipt["bytes"] * 2}
    with pytest.raises(ValueError, match="unique contiguous"):
        read_verified_factor_artifact(
            forged, tmp_path, policy=resolve_default_policy(),
            budget_bytes=8 * 1024 * 1024, broker=FakeBroker(),
            execution_purpose=purpose, run_identity_digest=digest,
        )


def test_legacy_writer_is_explicit_and_never_upgraded(tmp_path):
    receipt = write_verified_factor_artifact(
        tmp_path, "a" * 32, 0, "alpha", _value(),
        policy=resolve_default_policy(), budget_bytes=8 * 1024 * 1024,
    )
    assert receipt["execution_purpose"]["purpose"] == "legacy_unspecified"
    assert receipt["assurance"] == "LEGACY_UNSPECIFIED"
    assert receipt["publication_authorized"] is False


def test_parent_rejects_run_identity_purpose_before_input_iteration(tmp_path):
    engine = FakeEngine()
    engine.execution_purpose = ExecutionPurpose()
    touched = []

    def factors():
        touched.append(True)
        yield FakeFactor("alpha")

    wrong = _identity(ExecutionPurpose("production_compute"))
    with pytest.raises(pipeline.WorkerProtocolError, match="purpose differs"):
        pipeline.execute_run_many_durable(
            engine, factors(), policy=resolve_default_policy(),
            artifact_root=tmp_path, run_identity=wrong,
        )
    assert touched == []


def test_durable_run_receipt_and_resume_are_purpose_bound(tmp_path):
    research = ExecutionPurpose()
    engine = FakeEngine()
    engine.execution_purpose = research
    identity = _identity(research)
    receipt = pipeline.execute_run_many_durable(
        engine, [FakeFactor("alpha")], policy=resolve_default_policy(),
        artifact_root=tmp_path, run_identity=identity,
    )
    persisted = json.loads(Path(receipt["receipt_path"]).read_text())
    assert persisted["execution_purpose"] == research.to_dict()
    assert persisted["run_identity_digest"] == _digest(identity)
    assert persisted["assurance"] == "UNVERIFIED"
    assert persisted["publication_authorized"] is False

    production = ExecutionPurpose("production_compute")
    other = FakeEngine()
    other.execution_purpose = production
    with pytest.raises(Exception, match="identity|purpose"):
        pipeline.execute_run_many_durable(
            other, [], policy=resolve_default_policy(), artifact_root=tmp_path,
            run_identity=_identity(production), resume_run_id=receipt["run_id"],
        )


def test_worker_artifact_plan_cannot_override_engine_purpose(tmp_path, monkeypatch):
    engine = FakeEngine()
    engine.execution_purpose = ExecutionPurpose()
    with _isolated_worker_broker_globals(monkeypatch):
        with pytest.raises(pipeline.WorkerProtocolError, match="plan purpose differs"):
            pipeline._compute_wave_to_artifacts(
                engine, [FakeFactor("alpha")], {}, {
                    "root": tmp_path, "run_id": "a" * 32,
                    "policy": resolve_default_policy(),
                    "assignments": {"alpha": (0, "b" * 32)}, "writer_bytes": 1,
                    "execution_purpose": ExecutionPurpose("production_compute"),
                    "run_identity_digest": "d" * 64,
                }, broker_proxy=engine.resource_broker,
            )
