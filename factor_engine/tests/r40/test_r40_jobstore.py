"""R40 #95/#103/#104: service.jobstore remediation tests."""

from __future__ import annotations

import json

from service.jobstore import MANIFEST_SCHEMA_VERSION, JobRecord, JobStore


# ---------------------------------------------------------------------------
# #95: per-job immutable policy snapshot (schema round-trip)
# ---------------------------------------------------------------------------


def test_policy_snapshot_per_job_immutable(tmp_path):
    store = JobStore(tmp_path)
    rec = JobRecord(
        run_id="policy-snap",
        status="succeeded",
        policy_id="runtime_feature_policy",
        policy_version=7,
        policy_digest="deadbeef",
    )
    store.create(rec)
    restored = store.get("policy-snap")
    assert restored is not None
    assert restored.policy_id == "runtime_feature_policy"
    assert restored.policy_version == 7
    assert restored.policy_digest == "deadbeef"


def test_policy_snapshot_survives_reopen(tmp_path):
    store = JobStore(tmp_path)
    store.create(
        JobRecord(
            run_id="policy-snap2",
            status="succeeded",
            policy_id="runtime_feature_policy",
            policy_version=3,
            policy_digest="abc123",
        )
    )
    reopened = JobStore(tmp_path)
    restored = reopened.get("policy-snap2")
    assert restored is not None
    assert restored.policy_version == 3
    assert restored.policy_digest == "abc123"


# ---------------------------------------------------------------------------
# #103: current-schema manifest WITHOUT checksum is quarantined
# ---------------------------------------------------------------------------


def test_manifest_without_checksum_is_quarantined_current_schema(tmp_path):
    root = tmp_path / "svc"
    manifest_root = root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    raw = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": "no_checksum",
        "status": "succeeded",
        "service": "factor_engine",
        "job_type": "compute",
    }
    (manifest_root / "no_checksum.json").write_text(
        json.dumps(raw), encoding="utf-8"
    )
    store = JobStore(root)
    assert store.get("no_checksum") is None
    assert store.corruption_count >= 1


# ---------------------------------------------------------------------------
# #104: manifest WITHOUT schema_version is rejected (not silently upgraded)
# ---------------------------------------------------------------------------


def test_manifest_without_schema_version_is_rejected(tmp_path):
    root = tmp_path / "svc"
    manifest_root = root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    raw = {
        "run_id": "no_schema_version",
        "status": "succeeded",
        "service": "factor_engine",
        "job_type": "compute",
    }
    (manifest_root / "no_schema_version.json").write_text(
        json.dumps(raw), encoding="utf-8"
    )
    store = JobStore(root)
    assert store.get("no_schema_version") is None
    assert store.corruption_count >= 1


def test_manifest_with_explicit_current_schema_version_ok(tmp_path):
    root = tmp_path / "svc"
    store = JobStore(root)
    store.create(
        JobRecord(run_id="with_schema", status="succeeded", job_type="compute")
    )
    raw = json.loads((root / "manifests" / "with_schema.json").read_text(encoding="utf-8"))
    assert "schema_version" in raw
    assert "_checksum" in raw
    reopened = JobStore(root)
    assert reopened.get("with_schema") is not None
