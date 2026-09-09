from datetime import datetime, timezone
import hashlib, json
import pytest
from data_access.core.exceptions import DataError
from data_access.read.object_store import LocalObjectStore
from factor_optimizer.contracts.trial_ledger import TrialLedger
from jobs.e2e_f_noise_failure_gc import run_e2e_f_campaign
from quant_platform.app.adapters.data_access_storage import DataAccessStorageAdapter, RootAwareGarbageCollector
from quant_platform.app.contracts import ArtifactRef, ARTIFACT_TYPE_FACTOR_CANDIDATE, DeletionStatus
from quant_platform.app.db.sqlite_backend import SqliteDb
from quant_platform.app.storage.generation_coordinator import DurableGenerationCoordinator

class StoragePublisher:
    def __init__(self, storage): self.storage = storage
    def publish(self, artifact, data): self.storage.put(artifact.storage_uri, data); return artifact

def artifact(artifact_id, data, suffix):
    digest = hashlib.sha256(data).hexdigest()
    return ArtifactRef(artifact_id=artifact_id, artifact_type=ARTIFACT_TYPE_FACTOR_CANDIDATE,
        schema_version="1.0", content_hash=digest, storage_uri=f"test://raw/{artifact_id}/{suffix}",
        size_bytes=len(data), created_at=datetime(2026,1,1,tzinfo=timezone.utc),
        producer_type="e2e-f", producer_version="1")

def publish(coordinator, ref, data):
    generation = coordinator.stage(ref, data)
    assert coordinator.outbox.publish_pending() == 1
    return generation

def test_bounded_noise_campaign_to_real_registry_gc(tmp_path):
    result = run_e2e_f_campaign(tmp_path / "campaign")
    assert result.attempted_proposals == result.completed_evaluations == 2
    assert result.budget_reservation_denied and result.budget_exhausted
    assert all(abs(value) < .1 for value in result.observed_rank_ics)
    assert all(ref.startswith("factor-definition:") for ref in result.formula_history)
    ledger = TrialLedger.from_dict(json.loads(open(result.ledger_path, encoding="utf-8").read()))
    ledger.verify_chain(); assert ledger.status_counts()["PRUNED"] == 2
    assert all("RANK_IC_BELOW_POLICY" in e.failure_reason for e in ledger.entries if e.status_value == "PRUNED")

    storage = DataAccessStorageAdapter(store=LocalObjectStore(tmp_path / "objects"))
    coordinator = DurableGenerationCoordinator(SqliteDb(str(tmp_path / "registry.db")), StoragePublisher(storage))
    rejected = []
    for value_id, old_bytes in result.rejected_value_bytes:
        old_ref = artifact(value_id, old_bytes, "rejected-v1")
        old_gen = publish(coordinator, old_ref, old_bytes)
        publish(coordinator, artifact(value_id, b"retained successor", "retained-v2"), b"retained successor")
        rejected.append((old_ref, old_gen))
    retry_old = artifact("retryable", b"retry-old", "v1")
    retry_gen = publish(coordinator, retry_old, b"retry-old")
    coordinator.protect_gc_root("retryable_job", retry_old.artifact_id, retry_gen)
    publish(coordinator, artifact("retryable", b"retry-new", "v2"), b"retry-new")
    shared_old = artifact("shared", b"shared-old", "v1")
    shared_gen = publish(coordinator, shared_old, b"shared-old")
    publish(coordinator, artifact("shared", b"shared-new", "v2"), b"shared-new")
    prod = artifact("production", b"prod", "v1")
    prod_gen = publish(coordinator, prod, b"prod")
    coordinator.protect_gc_root("production", prod.artifact_id, prod_gen)
    coordinator.record_gc_reference(prod.artifact_id, prod_gen, shared_old.artifact_id, shared_gen)
    snapshot = coordinator.snapshot_for_gc()
    collector = RootAwareGarbageCollector(storage, coordinator); plan = collector.dry_run()
    assert snapshot.objects and {o.key for o in plan.candidates} == {
        (ref.artifact_id, generation) for ref, generation in rejected
    }
    receipts = collector.sweep(plan)
    assert all(r.status is DeletionStatus.PHYSICAL_DELETED for r in receipts)
    for ref, _ in rejected:
        with pytest.raises(DataError): storage.get(ref.storage_uri)
    assert storage.get(retry_old.storage_uri) == b"retry-old"
    assert storage.get(shared_old.storage_uri) == b"shared-old"
    assert storage.get(prod.storage_uri) == b"prod"
    ledger.verify_chain()
