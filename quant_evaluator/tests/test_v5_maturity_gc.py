from datetime import datetime, timedelta, timezone
import hashlib

import numpy as np
import pytest

from data_access.core.exceptions import DataError
from data_access.read.object_store import LocalObjectStore
from quant_evaluator.api.requests import EvaluationRequest
from quant_evaluator.contracts.evaluation_refs import FactorValueRef, LabelBundleRef
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.metric_instance import MetricInstance
from quant_evaluator.runtime.label_maturation import LabelMaturationQueue
from quant_platform.app.adapters.data_access_storage import DataAccessStorageAdapter, RootAwareGarbageCollector
from quant_platform.app.contracts import ArtifactRef, ARTIFACT_TYPE_FACTOR_CANDIDATE, DeletionStatus
from quant_platform.app.db.sqlite_backend import SqliteDb
from quant_platform.app.storage.generation_coordinator import DurableGenerationCoordinator


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


class Publisher:
    def __init__(self, adapter): self.adapter = adapter
    def publish(self, artifact, data):
        self.adapter.put(artifact.storage_uri, data)
        return artifact


def artifact(data, artifact_id):
    digest = hashlib.sha256(data).hexdigest()
    return ArtifactRef(artifact_id, ARTIFACT_TYPE_FACTOR_CANDIDATE, "1.0", digest,
                       f"test://objects/{artifact_id}/{digest}", len(data), START,
                       "test", "1")


def maturity_request():
    rng = np.random.default_rng(91)
    x = rng.normal(size=(2, 30))
    dates = (START, START + timedelta(days=1))
    batch = FactorBatch(("f",), AxisRef("time", "datetime", 2), AxisRef("asset", "int", 30), x[:, :, None])
    label = LabelBundle("h1", .2 * x, 1, decision_time=dates, label_start_time=dates,
                        label_end_time=tuple(d + timedelta(days=1) for d in dates))
    return EvaluationRequest(
        batch, label, metric_ids=(), tier="research",
        metric_instances=(MetricInstance("rank_ic_series", horizon=1),),
        factor_value_ref=FactorValueRef("fv:gc", ("f",)),
        label_bundle_ref=LabelBundleRef("label:gc", "h1", 1),
    )


def test_pending_failure_release_then_real_gc_deletes_bytes_not_trial(tmp_path):
    adapter = DataAccessStorageAdapter(store=LocalObjectStore(tmp_path / "objects"))
    db = SqliteDb(str(tmp_path / "metadata.db"))
    coordinator = DurableGenerationCoordinator(db, Publisher(adapter))
    old = artifact(b"old-factor-bytes", "factor-values")
    old_gen = coordinator.stage(old, b"old-factor-bytes")
    coordinator.outbox.publish_pending()
    new = artifact(b"new-factor-bytes", "factor-values")
    coordinator.stage(new, b"new-factor-bytes")
    coordinator.outbox.publish_pending()
    assert adapter.get(old.storage_uri) == b"old-factor-bytes"

    queue = LabelMaturationQueue(db, generation_coordinator=coordinator)
    request = maturity_request()
    semantics = dict(factor_value_semantics="raw.v1", universe="synthetic.30",
                     clock="daily.complete", window="all_history", policy_version="v5")
    queue.enqueue("pending:1", request, stream_semantics=semantics, available_at=START,
                  generation_refs=((old.artifact_id, old_gen),))
    assert (old.artifact_id, old_gen) in coordinator.snapshot_for_gc().pending_label_roots
    assert not RootAwareGarbageCollector(adapter, coordinator).dry_run().candidates

    def crash(_): raise RuntimeError("checkpoint crash")
    with pytest.raises(RuntimeError, match="checkpoint crash"):
        queue.drain(START + timedelta(days=4), lambda _: request, before_commit=crash)
    assert db.query("SELECT status FROM qe_maturity_events") == [{"status": "PENDING"}]
    assert (old.artifact_id, old_gen) in coordinator.snapshot_for_gc().pending_label_roots
    assert adapter.get(old.storage_uri) == b"old-factor-bytes"

    assert queue.drain(START + timedelta(days=4), lambda _: request) == ("pending:1",)
    assert (old.artifact_id, old_gen) not in coordinator.snapshot_for_gc().pending_label_roots
    plan = RootAwareGarbageCollector(adapter, coordinator).dry_run()
    assert [(x.object_id, x.object_version) for x in plan.candidates] == [(old.artifact_id, old_gen)]
    receipt = RootAwareGarbageCollector(adapter, coordinator).sweep(plan)[0]
    assert receipt.status is DeletionStatus.PHYSICAL_DELETED
    with pytest.raises(DataError): adapter.get(old.storage_uri)
    # GC removes object bytes only. The completed trial/event and its exact observations survive.
    assert db.query("SELECT status FROM qe_maturity_events") == [{"status": "COMPLETE"}]
    assert len(db.query("SELECT * FROM qe_maturity_observations")) == 2
