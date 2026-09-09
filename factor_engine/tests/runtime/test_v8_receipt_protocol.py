import json
from pathlib import Path

import pytest
from contextlib import contextmanager

from factor_engine.runtime import bounded_pipeline as pipeline
from factor_engine.runtime.default_execution_policy import resolve_default_policy
from factor_engine.tests.runtime.test_v6_bounded_pipeline import (
    FakeEngine, FakeFactor, PreflightIsolationEngine,
)


@contextmanager
def _isolated_worker_broker_globals(monkeypatch):
    import factor_engine.runtime.resource_broker as broker_module

    with monkeypatch.context() as isolated:
        for name in (
            "_V2_BROKER", "_V2_BROKER_PID", "_V2_BROKER_POLICY_KEY",
            "_V2_WORKER_PROXY",
        ):
            isolated.setattr(broker_module, name, None)
        yield


def _identity():
    return {
        "deployment_digest": "d" * 64,
        "profile_id": "approved-profile",
        "approval_id": "approval-7",
        "execution_scope": {"market": "ashare", "adjustment": "hfq"},
        "source_identity": {
            "dataset": "ashare_stock_daily_adj",
            "content_digests": {"ashare_stock_daily_adj": "a" * 32},
        },
    }


def _receipt(name, ordinal, generation, run_id, policy_digest):
    return {
        "schema_version": "factor_engine.factor_artifact.v2",
        "run_id": run_id,
        "ordinal": ordinal,
        "factor_id": name,
        "generation": generation,
        "policy_digest": policy_digest,
        "sha256": "same-payload-hash",
        "committed": True,
        "verified": True,
    }


@pytest.mark.parametrize(
    "field,bad",
    [
        ("generation", "wrong-generation"),
        ("run_id", "wrong-run"),
        ("ordinal", 99),
        ("ordinal", True),
        ("factor_id", "other-factor"),
        ("schema_version", "factor_engine.factor_artifact.v1"),
        ("policy_digest", "wrong-policy"),
    ],
)
def test_direct_artifact_identity_fault_aborts_envelope_before_terminals(field, bad):
    assignments = {"alpha": (3, "generation-3")}
    receipt = _receipt("alpha", 3, "generation-3", "run-7", "policy-7")
    receipt[field] = bad
    envelope = {"receipts": {"alpha": receipt}, "artifact_errors": {}, "factor_errors": {}}

    with pytest.raises(pipeline.WorkerProtocolError, match="identity differs"):
        pipeline._validate_artifact_envelope(
            envelope, assignments, run_id="run-7", policy_digest="policy-7"
        )


def test_swapped_factor_receipts_fail_even_with_same_payload_hash():
    assignments = {"alpha": (1, "g1"), "beta": (2, "g2")}
    envelope = {
        "receipts": {
            "alpha": _receipt("beta", 2, "g2", "run", "policy"),
            "beta": _receipt("alpha", 1, "g1", "run", "policy"),
        },
        "artifact_errors": {},
        "factor_errors": {},
    }
    with pytest.raises(pipeline.WorkerProtocolError):
        pipeline._validate_artifact_envelope(
            envelope, assignments, run_id="run", policy_digest="policy"
        )


def test_direct_artifact_error_requires_full_assignment_identity():
    envelope = {
        "receipts": {},
        "artifact_errors": {"alpha": {"code": "OUTPUT_INVALID"}},
        "factor_errors": {},
    }
    with pytest.raises(pipeline.WorkerProtocolError, match="error identity differs"):
        pipeline._validate_artifact_envelope(
            envelope, {"alpha": (1, "g1")}, run_id="run", policy_digest="policy"
        )


@pytest.mark.parametrize("engine,expected_status", [
    (FakeEngine(), "SUCCEEDED"),
    (PreflightIsolationEngine(), "COMPLETED_WITH_ERRORS"),
])
def test_run_identity_is_identical_in_returned_and_disk_receipts(
    tmp_path, engine, expected_status
):
    factors = [FakeFactor("good")] if expected_status == "SUCCEEDED" else [
        FakeFactor("good"), FakeFactor("bad")
    ]
    receipt = pipeline.execute_run_many_durable(
        engine, factors, policy=resolve_default_policy(), artifact_root=tmp_path,
        run_identity=_identity(),
    )
    persisted = json.loads(Path(receipt["receipt_path"]).read_text())
    created = json.loads(Path(receipt["identity_path"]).read_text())
    assert receipt["status"] == expected_status
    assert persisted["run_identity"] == receipt["run_identity"] == _identity()
    assert persisted["deployment_digest"] == receipt["deployment_digest"] == "d" * 64
    assert persisted["policy_digest"] == receipt["policy_digest"]
    assert created["run_identity"] == _identity()
    assert created["run_id"] == receipt["run_id"]
    assert created["policy_digest"] == receipt["policy_digest"]


def test_abort_receipt_persists_same_run_identity_and_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "_compile_wave", lambda engine, factors: {})
    with pytest.raises(pipeline.WorkerProtocolError) as caught:
        pipeline.execute_run_many_durable(
            FakeEngine(), [FakeFactor("alpha")], policy=resolve_default_policy(),
            artifact_root=tmp_path, run_identity=_identity(),
        )
    persisted = json.loads(Path(caught.value.receipt_path).read_text())
    assert persisted["status"] == "ABORTED"
    assert persisted["run_identity"] == _identity()
    assert persisted["deployment_digest"] == "d" * 64
    assert persisted["policy_digest"] == resolve_default_policy().digest
    created = json.loads(Path(persisted["identity_path"]).read_text())
    assert created["run_identity"] == persisted["run_identity"]


@pytest.mark.parametrize("wrong_identity", [
    {"run_id": "malicious-other-run"},
    {"ordinal": True},
])
def test_direct_sink_rejects_writer_supplied_wrong_identity(
    tmp_path, monkeypatch, wrong_identity
):
    from factor_engine.runtime import durable_artifact_sink
    from factor_engine.tests.runtime.test_v6_bounded_pipeline import FakeBroker

    monkeypatch.setattr(durable_artifact_sink, "required_writer_workspace_bytes", lambda value: 1)
    monkeypatch.setattr(
        durable_artifact_sink, "write_verified_factor_artifact",
        lambda *args, **kwargs: {
            "committed": True, "verified": True, "generation": "g1",
            **wrong_identity,
        },
    )
    sink = pipeline._DirectVerifiedArtifactSink(
        root=tmp_path, run_id="run", policy=resolve_default_policy(),
        assignments={"alpha": (1, "g1")}, writer_bytes=10, broker=FakeBroker(),
    )
    with pytest.raises(pipeline.WorkerProtocolError, match="identity differs"):
        sink("alpha", object())


def test_direct_compute_rejects_boolean_ordinal_in_worker_error_before_stamping(
    tmp_path, monkeypatch,
):
    from factor_engine.tests.runtime.test_v6_bounded_pipeline import FakeBroker

    class BadErrorEngine:
        run_mode = "research"

        def run_many_parallel(self, factors, **kwargs):
            return {
                "results": {},
                "physical_preflight_errors": {
                    "alpha": {"code": "BAD_INPUT", "ordinal": True}
                },
            }

    with _isolated_worker_broker_globals(monkeypatch):
        with pytest.raises(pipeline.WorkerProtocolError, match="identity differs"):
            pipeline._compute_wave_to_artifacts(
                BadErrorEngine(), [FakeFactor("alpha")], {}, {
                    "root": tmp_path,
                    "run_id": "run",
                    "policy": resolve_default_policy(),
                    "assignments": {"alpha": (1, "g1")},
                    "writer_bytes": 10,
                }, broker_proxy=FakeBroker(),
            )


@pytest.mark.parametrize("bad", [{"x": float("nan")}, {1: "bad"}, ["not-a-map"]])
def test_run_identity_is_rejected_before_run_directory_creation(tmp_path, bad):
    with pytest.raises(TypeError, match="run_identity"):
        pipeline.execute_run_many_durable(
            FakeEngine(), [], policy=resolve_default_policy(), artifact_root=tmp_path,
            run_identity=bad,
        )
    assert not list(tmp_path.iterdir())
