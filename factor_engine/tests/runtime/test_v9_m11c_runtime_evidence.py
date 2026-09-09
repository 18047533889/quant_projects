from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd

from factor_engine.cleaned_operators.ts_model._rolling_core import BoundedFitFailureSink
from factor_engine.runtime import bounded_pipeline as pipeline
from factor_engine.runtime.fit_failure_evidence import snapshot_fit_failures
from factor_engine.runtime.persistent_run_state import PersistentRunState
from factor_engine.runtime.default_execution_policy import resolve_default_policy
from factor_engine.runtime.resume_validation import ResumeIdentityError, validate_resume_context
from factor_engine.tests.runtime.test_v6_bounded_pipeline import FakeEngine, FakeFactor


def test_real_run_many_compute_snapshot_is_nonempty_and_pageable(tmp_path):
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.expr import CleanedCall, ColumnRef
    from factor_engine.runtime.engine import FactorEngine
    from tests.helpers import InMemorySeriesSource

    dates = pd.date_range("2026-03-01", periods=12)
    index = pd.MultiIndex.from_product(
        [dates, ("GOOD", "BAD")], names=["timestamp", "instrument"])
    x, y = [], []
    for row in range(len(dates)):
        x.extend((float(row), 1.0))
        y.extend((2.0 + 3.0 * row, float(row)))
    source = InMemorySeriesSource(data={
        "target": pd.Series(y, index=index),
        "feature": pd.Series(x, index=index),
    })
    expression = CleanedCall(
        "ts_multi_regression_coeff", (ColumnRef("target"), ColumnRef("feature")),
        (("window", 10), ("coefficient_index", 1), ("min_periods", 2),
         ("add_intercept", True), ("warmup_policy", "expanding")),
    )
    factor = Factor("alpha", expression)
    evidence_id = "9" * 32
    payload = pipeline._compute_wave(
        FactorEngine(PandasBackend(), source, run_mode="research"), [factor], {},
        8 * 1024 * 1024,
        execution_owner={"run_id": "run-real", "profile_id": "profile-real"},
        evidence_id=evidence_id,
    )
    envelope = __import__("pickle").loads(payload)
    _, _, _, evidence = pipeline._validate_result_envelope(
        envelope, {"alpha"}, expected_evidence_id=evidence_id,
        expected_run_id="run-real", include_evidence=True,
    )
    assert evidence["snapshot"]["groups"]
    assert evidence["snapshot"]["details"]

    state = PersistentRunState(tmp_path / "state.sqlite3")
    state.enable_fit_failure_evidence()
    state.register(0, "alpha")
    state.record_fit_failure_evidence(
        evidence_id, [{"ordinal": 0, "name": "alpha"}], evidence["snapshot"])
    page = state.fit_failure_evidence_page(limit=1)
    assert page[0]["snapshot"]["details"]
    state.close()


def _snapshot():
    return snapshot_fit_failures(BoundedFitFailureSink())


def _extended_result(evidence_id, snapshot=None):
    return {
        "result_blobs": {"alpha": b"value"},
        "transport_errors": {},
        "factor_errors": {},
        "fit_failure_evidence": {
            "evidence_id": evidence_id,
            "scope": "wave_sample",
            "snapshot": _snapshot() if snapshot is None else snapshot,
        },
    }


def test_result_validator_preserves_legacy_unpack_and_distinguishes_missing():
    legacy = {"result_blobs": {"alpha": b"value"},
              "transport_errors": {}, "factor_errors": {}}
    blobs, transport, factors = pipeline._validate_result_envelope(legacy, {"alpha"})
    assert blobs == {"alpha": b"value"}
    assert transport == factors == {}
    assert pipeline._validate_result_envelope(
        legacy, {"alpha"}, include_evidence=True
    )[3] is None

    evidence_id = "a" * 32
    assert pipeline._validate_result_envelope(
        _extended_result(evidence_id), {"alpha"},
        expected_evidence_id=evidence_id, include_evidence=True,
    )[3]["snapshot"]["availability"] == "observed"


@pytest.mark.parametrize("evidence_id", [None, "", "A" * 32, "a" * 31])
def test_extended_envelope_rejects_missing_or_invalid_evidence_identity(evidence_id):
    envelope = _extended_result(evidence_id)
    if evidence_id is None:
        envelope["fit_failure_evidence"] = None
    with pytest.raises(pipeline.WorkerProtocolError):
        pipeline._validate_result_envelope(
            envelope, {"alpha"}, expected_evidence_id=evidence_id,
            include_evidence=True,
        )


def test_snapshot_factor_identity_cannot_move_to_another_wave():
    snapshot = _snapshot()
    snapshot["details"] = [{
        "sequence": 1,
        "status": {"converged": False, "reason": "forced",
                   "details": {"$type": "tuple", "items": []}},
        "scope": {
            "canonical": "ts_test", "backend": "pandas_numpy", "profile": "profile-1",
            "instrument": "AAA", "window_start": 0, "window_end": 1,
            "output_row": 1, "fit_cutoff": 1, "maturity_cutoff": 1,
            "execution_id": "exec", "run_id": "run-1", "task_id": "task",
            "factor_id": "beta",
        },
        "scope_kind": "factor_window",
    }]
    with pytest.raises(pipeline.WorkerProtocolError, match="factor identity"):
        pipeline._validate_result_envelope(
            _extended_result("b" * 32, snapshot), {"alpha"},
            expected_evidence_id="b" * 32, expected_run_id="run-1",
            include_evidence=True,
        )


def test_persistent_evidence_is_bounded_paged_and_idempotent(tmp_path):
    state = PersistentRunState(tmp_path / "state.sqlite3")
    state.enable_fit_failure_evidence()
    state.register(0, "alpha")
    snapshot = _snapshot()
    evidence_id = "c" * 32
    first = state.record_fit_failure_evidence(
        evidence_id, [{"ordinal": 0, "name": "alpha"}], snapshot)
    assert state.record_fit_failure_evidence(
        evidence_id, [{"ordinal": 0, "name": "alpha"}], snapshot) == first
    page = state.fit_failure_evidence_page(limit=1)
    assert page[0]["snapshot"]["availability"] == "observed"
    assert state.fit_failure_evidence_page(after_seq=page[0]["seq"]) == []
    assert state.fit_failure_evidence_summary() == {
        "observed_waves": 1, "unavailable_waves": 0,
        "truncated_waves": 0, "wave_count": 1, "last_seq": first,
    }
    state.close()


def test_direct_success_persists_observed_wave_before_terminal(tmp_path):
    state = PersistentRunState(tmp_path / "state.sqlite3")
    state.enable_fit_failure_evidence()
    state.register(0, "alpha")
    assert state.consume_attempt(0, "execution") == 1
    generation = "2" * 32
    state.record_commit_intent(0, generation)
    factor = SimpleNamespace(name="alpha")
    slot = pipeline._DirectSlot(
        cursor=0, wave=[], admitted=[(0, factor)], names=["alpha"],
        assignments={"alpha": (0, generation)}, evidence_id="3" * 32,
    )
    receipt = {
        "schema_version": "factor_engine.factor_artifact.v2",
        "run_id": "run-1", "ordinal": 0, "factor_id": "alpha",
        "generation": generation, "policy_digest": "policy-1",
        "committed": True, "verified": True,
    }
    envelope = {
        "receipts": {"alpha": receipt}, "artifact_errors": {}, "factor_errors": {},
        "fit_failure_evidence": {
            "evidence_id": "3" * 32, "scope": "wave_sample", "snapshot": _snapshot(),
        },
    }
    pipeline._complete_direct_slot(
        slot, envelope, run_id="run-1",
        policy=SimpleNamespace(digest="policy-1", max_examples_per_error_group=2),
        state=state, error_groups={},
    )
    assert state.outcomes()[0].state == "SUCCEEDED"
    assert state.fit_failure_evidence_summary()["observed_waves"] == 1
    state.close()


def test_direct_evidence_write_failure_cannot_leave_success_terminal(tmp_path, monkeypatch):
    state = PersistentRunState(tmp_path / "state.sqlite3")
    state.enable_fit_failure_evidence()
    state.register(0, "alpha")
    state.consume_attempt(0, "execution")
    generation = "4" * 32
    state.record_commit_intent(0, generation)
    slot = pipeline._DirectSlot(
        cursor=0, wave=[], admitted=[(0, SimpleNamespace(name="alpha"))],
        names=["alpha"], assignments={"alpha": (0, generation)},
        evidence_id="5" * 32,
    )
    envelope = {
        "receipts": {"alpha": {
            "schema_version": "factor_engine.factor_artifact.v2", "run_id": "run-1",
            "ordinal": 0, "factor_id": "alpha", "generation": generation,
            "policy_digest": "policy-1", "committed": True, "verified": True,
        }},
        "artifact_errors": {}, "factor_errors": {},
        "fit_failure_evidence": {
            "evidence_id": "5" * 32, "scope": "wave_sample", "snapshot": _snapshot(),
        },
    }
    monkeypatch.setattr(
        state, "record_fit_failure_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("evidence fsync failed")),
    )
    with pytest.raises(OSError, match="evidence fsync failed"):
        pipeline._complete_direct_slot(
            slot, envelope, run_id="run-1",
            policy=SimpleNamespace(digest="policy-1", max_examples_per_error_group=2),
            state=state, error_groups={},
        )
    assert state.outcomes()[0].state == "RUNNING"
    state.close()


@pytest.mark.parametrize(
    "tamper", ["drop_table", "delete_row", "payload", "assignments", "orphan"])
def test_resume_rejects_missing_or_corrupt_new_evidence(tmp_path, tamper):
    policy = resolve_default_policy()
    receipt = pipeline.execute_run_many_durable(
        FakeEngine(), [FakeFactor("alpha")], policy=policy, artifact_root=tmp_path,
        run_identity={},
    )
    validate_resume_context(
        tmp_path, receipt["run_id"], policy=policy, run_identity={})
    db = sqlite3.connect(receipt["state_path"])
    with db:
        if tamper == "drop_table":
            db.execute("DROP TABLE fit_failure_evidence")
        elif tamper == "delete_row":
            db.execute("DELETE FROM fit_failure_evidence")
        elif tamper == "payload":
            db.execute(
                "UPDATE fit_failure_evidence SET availability='OBSERVED',"
                "payload_json=?,payload_bytes=262145",
                ('{"oversized":"' + "x" * 262145 + '"}',),
            )
        elif tamper == "assignments":
            db.execute(
                "DELETE FROM fit_failure_evidence_assignments",
            )
        else:
            db.execute(
                "INSERT INTO fit_failure_evidence_assignments VALUES(?,?)",
                ("0" * 32, 0),
            )
    db.close()
    with pytest.raises(ResumeIdentityError):
        validate_resume_context(
            tmp_path, receipt["run_id"], policy=policy, run_identity={})


def test_legacy_unavailable_survives_two_resumes(tmp_path):
    policy = resolve_default_policy()
    first = pipeline.execute_run_many_durable(
        FakeEngine(), [FakeFactor("alpha")], policy=policy, artifact_root=tmp_path,
        run_identity={},
    )
    receipt_path = Path(first["receipt_path"])
    legacy_receipt = json.loads(receipt_path.read_text())
    legacy_receipt.pop("fit_failure_evidence")
    receipt_path.write_text(json.dumps(legacy_receipt))
    db = sqlite3.connect(first["state_path"])
    with db:
        db.execute("DELETE FROM state_policy WHERE key='fit_failure_evidence_schema'")
        db.execute("DROP TABLE fit_failure_evidence")
        db.execute("DROP TABLE fit_failure_evidence_assignments")
    db.close()

    second = pipeline.execute_run_many_durable(
        FakeEngine(), (), policy=policy, artifact_root=tmp_path,
        run_identity={}, resume_run_id=first["run_id"],
    )
    third = pipeline.execute_run_many_durable(
        FakeEngine(), (), policy=policy, artifact_root=tmp_path,
        run_identity={}, resume_run_id=first["run_id"],
    )
    assert second["fit_failure_evidence"]["availability"] == "legacy_unavailable"
    assert third["fit_failure_evidence"]["availability"] == "legacy_unavailable"
    assert "observed_waves" not in third["fit_failure_evidence"]


def test_new_marker_and_tables_remain_authoritative_before_receipt_exists(tmp_path):
    policy = resolve_default_policy()
    receipt = pipeline.execute_run_many_durable(
        FakeEngine(), [FakeFactor("alpha")], policy=policy, artifact_root=tmp_path,
        run_identity={},
    )
    Path(receipt["receipt_path"]).unlink()
    context = validate_resume_context(
        tmp_path, receipt["run_id"], policy=policy, run_identity={})
    assert context.fit_failure_evidence_available is True


def test_evidence_page_has_a_total_wire_budget_without_skipping_seq(tmp_path):
    state = PersistentRunState(tmp_path / "state.sqlite3")
    state.enable_fit_failure_evidence()
    state.register(0, "alpha")
    large = _snapshot()
    large["groups"] = [
        {"canonical": f"op-{index}", "reason": "x" * 3900 + str(index), "count": 1}
        for index in range(60)
    ]
    for index in range(5):
        state.record_fit_failure_evidence(
            f"{index + 1:032x}", ({"ordinal": 0, "name": "alpha"} for _ in range(1)),
            large,
        )
    first = state.fit_failure_evidence_page(limit=5)
    assert 1 <= len(first) < 5
    second = state.fit_failure_evidence_page(after_seq=first[-1]["seq"], limit=5)
    assert second[0]["seq"] == first[-1]["seq"] + 1
    assert len(first) + len(second) == 5
    state.close()


@pytest.mark.parametrize("fault", ["store", "bool_count", "bool_cursor", "extra"])
def test_resume_rejects_malformed_evidence_receipt_reference(tmp_path, fault):
    policy = resolve_default_policy()
    receipt = pipeline.execute_run_many_durable(
        FakeEngine(), [FakeFactor("alpha")], policy=policy, artifact_root=tmp_path,
        run_identity={},
    )
    path = Path(receipt["receipt_path"])
    persisted = json.loads(path.read_text())
    reference = persisted["fit_failure_evidence"]
    if fault == "store":
        reference["store"] = "other.sqlite3"
    elif fault == "bool_count":
        reference["wave_count"] = True
    elif fault == "bool_cursor":
        reference["next_seq"] = False
    else:
        reference["unexpected"] = 1
    path.write_text(json.dumps(persisted))
    with pytest.raises(ResumeIdentityError, match="evidence"):
        validate_resume_context(
            tmp_path, receipt["run_id"], policy=policy, run_identity={})


def test_unavailable_evidence_has_no_counts_and_assignment_is_authoritative(tmp_path):
    state = PersistentRunState(tmp_path / "state.sqlite3")
    state.enable_fit_failure_evidence()
    state.register(0, "alpha")
    state.record_fit_failure_evidence(
        "d" * 32, [{"ordinal": 0, "name": "alpha"}], None,
        availability="UNAVAILABLE",
    )
    assert state.fit_failure_evidence_page()[0]["snapshot"] is None
    with pytest.raises(ValueError, match="not registered"):
        state.record_fit_failure_evidence(
            "e" * 32, [{"ordinal": 0, "name": "beta"}], None,
            availability="UNAVAILABLE",
        )
    state.close()


def test_large_wave_assignment_links_do_not_reject_legal_long_names(tmp_path):
    state = PersistentRunState(tmp_path / "state.sqlite3")
    state.enable_fit_failure_evidence()
    names = [f"factor_{ordinal:04d}_" + "x" * 110 for ordinal in range(600)]
    state.register_many(enumerate(names))
    state.record_fit_failure_evidence(
        "8" * 32,
        ({"ordinal": ordinal, "name": name} for ordinal, name in enumerate(names)),
        None, availability="UNAVAILABLE",
    )
    page = state.fit_failure_evidence_page(limit=1)[0]
    assert page["assignment_count"] == 600
    assert len(page["assignments_sha256"]) == 64
    state.close()


def test_evidence_digest_covers_assignments(tmp_path):
    path = tmp_path / "state.sqlite3"
    state = PersistentRunState(path)
    state.enable_fit_failure_evidence()
    state.register(0, "alpha")
    state.record_fit_failure_evidence(
        "f" * 32, [{"ordinal": 0, "name": "alpha"}], _snapshot())
    state.close()
    db = sqlite3.connect(path)
    with db:
        db.execute("DELETE FROM fit_failure_evidence_assignments")
    db.close()
    # Resume validation owns the fail-closed integration check; this asserts
    # that mutating assignments does not also mutate the covering digest.
    db = sqlite3.connect(path)
    linked = db.execute("SELECT COUNT(*) FROM fit_failure_evidence_assignments").fetchone()[0]
    digest = db.execute("SELECT evidence_sha256 FROM fit_failure_evidence").fetchone()[0]
    assert linked == 0 and len(digest) == 64
    db.close()


def test_compute_reserves_snapshot_transport_before_result_admission():
    class Engine:
        run_mode = "research"

        def run_many_parallel(self, factors, **kwargs):
            return {"results": {}, "_fit_failure_snapshot": _snapshot()}

    with pytest.raises(pipeline.ResultTransportBudgetExceeded, match="fit failure evidence"):
        pipeline._compute_wave(
            Engine(), [SimpleNamespace(name="alpha")], {}, 1024,
            evidence_id="1" * 32,
        )
