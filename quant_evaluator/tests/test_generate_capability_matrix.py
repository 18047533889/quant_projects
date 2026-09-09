import csv
import hashlib
from types import SimpleNamespace
import numpy as np
import pytest

from quant_evaluator.backends.capability_registry import (
    BackendCapabilityRegistry, BackendImplementation, ParityStatus,
)
from quant_evaluator.scripts import generate_capability_matrix as matrix
from quant_evaluator.scripts.generate_capability_matrix import CapabilityRunEvidence
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy


def _data_identity(values):
    return "fixture-bytes:" + hashlib.sha256(
        values.tobytes() + values.tobytes()
    ).hexdigest()


def test_declared_nonexecutable_and_fake_gpu_receipt_cannot_turn_matrix_green(
    tmp_path, monkeypatch
):
    spec = SimpleNamespace(
        metric_version="9.7.3", tier="extended", artifact_kind="scalar",
        compute_fn=None, required_inputs={"factor_batch"}, requires=["factor_batch"],
    )
    registry = BackendCapabilityRegistry()
    registry.register(BackendImplementation(
        operation_id="declared_only", backend="cuda", implementation_version="1",
        implementation_hash="a" * 64, parity_status=ParityStatus.PARITY_PASS,
        fn=None,
    ))
    monkeypatch.setattr(matrix, "list_metrics", lambda: ["declared_only"])
    monkeypatch.setattr(matrix, "get_metric", lambda _: spec)
    monkeypatch.setattr(matrix, "get_backend_capability_registry", lambda: registry)
    destination = tmp_path / "matrix.csv"
    matrix.generate_capability_matrix(
        str(destination),
        run_evidence=[{
            "metric_id": "declared_only", "backend": "cuda_strict",
            "parity": "PASS", "source": "mock-gpu",
        }],
    )
    with destination.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["metric_version"] == "9.7.3"
    assert row["declared"] == "DONE"
    assert row["kernel"] == "UNSUPPORTED"
    assert row["public_cpu"] == "UNSUPPORTED"
    assert row["public_gpu"] == "NOT_RUN"
    assert row["gpu_declaration"] == "DONE"
    assert row["gpu_parity_declaration"] == "parity_pass"
    assert row["run_evidence_refs"] == ""
    assert all(row[layer] == "NOT_RUN" for layer in (
        "adapter", "profile", "admission", "update", "publication"
    ))


def test_actual_public_cpu_bundle_is_bounded_versioned_run_evidence(tmp_path):
    values = np.arange(400, dtype=float).reshape(20, 20)
    batch = FactorBatch(("f",), AxisRef("time", "int64", 20),
                        AxisRef("asset", "int64", 20), values[..., None])
    labels = LabelBundle("forward", values, 1, decision_time=tuple(range(20)),
                         label_start_time=tuple(range(1, 21)),
                         label_end_time=tuple(range(2, 22)))
    bundle = evaluate(batch, labels, metrics=["rank_ic"])
    destination = tmp_path / "matrix.csv"
    evidence = CapabilityRunEvidence.from_public_cpu(
        bundle, "rank_ic", source_ref="qe-public-evaluate",
        data_identity=_data_identity(values),
    )
    matrix.generate_capability_matrix(str(destination), run_evidence=[evidence])
    with destination.open(newline="", encoding="utf-8") as handle:
        rows = {row["metric_id"]: row for row in csv.DictReader(handle)}
    row = rows["rank_ic"]
    assert row["metric_version"] == bundle.metric_versions["rank_ic"]
    assert row["kernel"] == "DONE"
    assert row["public_cpu"] == "DONE"
    assert row["public_gpu"] in {"NOT_RUN", "UNSUPPORTED"}
    assert row["run_evidence_refs"] == (
        f"cpu:{bundle.request_id}@qe-public-evaluate@{_data_identity(values)}"
    )


def test_backendless_bundle_and_wrong_version_do_not_mark_done(tmp_path, monkeypatch):
    values = np.arange(400, dtype=float).reshape(20, 20)
    batch = FactorBatch(("f",), AxisRef("time", "int64", 20),
                        AxisRef("asset", "int64", 20), values[..., None])
    labels = LabelBundle("forward", values, 1, decision_time=tuple(range(20)),
                         label_start_time=tuple(range(1, 21)), label_end_time=tuple(range(2, 22)))
    bundle = evaluate(batch, labels, metrics=["rank_ic"])
    destination = tmp_path / "backendless.csv"
    matrix.generate_capability_matrix(str(destination), run_evidence=[bundle])
    with destination.open(newline="", encoding="utf-8") as handle:
        row = {r["metric_id"]: r for r in csv.DictReader(handle)}["rank_ic"]
    assert row["kernel"] == "DECLARED" and row["public_cpu"] == "NOT_RUN"

    evidence = CapabilityRunEvidence.from_public_cpu(
        bundle, "rank_ic", source_ref="qe-public-evaluate", data_identity=_data_identity(values)
    )
    original = matrix.get_metric
    monkeypatch.setattr(matrix, "get_metric", lambda metric_id:
        SimpleNamespace(**({**original(metric_id).__dict__, "metric_version": "future-version"}))
        if metric_id == "rank_ic" else original(metric_id))
    destination = tmp_path / "wrong.csv"
    matrix.generate_capability_matrix(str(destination), run_evidence=[evidence])
    with destination.open(newline="", encoding="utf-8") as handle:
        row = {r["metric_id"]: r for r in csv.DictReader(handle)}["rank_ic"]
    assert row["kernel"] == "DECLARED" and row["public_cpu"] == "NOT_RUN"


def test_callable_that_raises_is_only_declared(tmp_path, monkeypatch):
    def raises(*_args, **_kwargs): raise RuntimeError("not executable")
    spec = SimpleNamespace(metric_version="1", tier="extended", artifact_kind="scalar",
                           compute_fn=raises, required_inputs=set(), requires=[])
    monkeypatch.setattr(matrix, "list_metrics", lambda: ["raises"])
    monkeypatch.setattr(matrix, "get_metric", lambda _: spec)
    monkeypatch.setattr(matrix, "get_backend_capability_registry", BackendCapabilityRegistry)
    destination = tmp_path / "raises.csv"
    matrix.generate_capability_matrix(str(destination))
    with destination.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["kernel"] == "DECLARED" and row["public_cpu"] == "NOT_RUN"


def test_facade_calendar_gpu_route_is_not_mislabeled_unsupported(tmp_path):
    destination = tmp_path / "calendar.csv"
    matrix.generate_capability_matrix(str(destination))
    with destination.open(newline="", encoding="utf-8") as handle:
        row = {r["metric_id"]: r for r in csv.DictReader(handle)}["worst_calendar_month"]
    assert row["public_gpu"] == "NOT_RUN"


def test_invalid_empty_execution_cannot_create_run_evidence():
    values = np.zeros((20, 20))
    batch = FactorBatch(("f",), AxisRef("time", "int64", 20),
                        AxisRef("asset", "int64", 20), values[..., None])
    labels = LabelBundle("forward", values, 1, decision_time=tuple(range(20)),
                         label_start_time=tuple(range(1, 21)), label_end_time=tuple(range(2, 22)))
    bundle = evaluate(batch, labels, metrics=["rank_ic"])
    with pytest.raises(ValueError, match="empty or invalid|no observations"):
        CapabilityRunEvidence.from_public_cpu(
            bundle, "rank_ic", source_ref="qe-public-evaluate", data_identity="constant:L20"
        )


def test_actual_cuda_strict_run_marks_only_gpu_and_is_revalidated(tmp_path, monkeypatch):
    values = np.arange(400, dtype=float).reshape(20, 20)
    batch = FactorBatch(("f",), AxisRef("time", "int64", 20),
                        AxisRef("asset", "int64", 20), values[..., None])
    labels = LabelBundle("forward", values, 1, decision_time=tuple(range(20)),
                         label_start_time=tuple(range(1, 21)), label_end_time=tuple(range(2, 22)))
    bundle = evaluate(batch, labels, metrics=["rank_ic"], backend="cuda",
                      gpu_policy=GPUExecutionPolicy(strict_backend=True))
    evidence = CapabilityRunEvidence.from_public_gpu(
        bundle, "rank_ic", source_ref="qe-public-cuda-strict",
        data_identity=_data_identity(values),
    )
    with pytest.raises(ValueError, match="cannot be relabeled"):
        CapabilityRunEvidence.from_public_cpu(
            bundle, "rank_ic", source_ref="wrong-cpu", data_identity=_data_identity(values)
        )
    # The public CUDA route is authoritative even when the auxiliary declaration
    # registry is empty; declaration metadata remains a separate column.
    monkeypatch.setattr(matrix, "get_backend_capability_registry", BackendCapabilityRegistry)
    destination = tmp_path / "gpu.csv"
    matrix.generate_capability_matrix(str(destination), run_evidence=[evidence])
    with destination.open(newline="", encoding="utf-8") as handle:
        row = {r["metric_id"]: r for r in csv.DictReader(handle)}["rank_ic"]
    assert row["public_gpu"] == "DONE" and row["public_cpu"] == "NOT_RUN"
    assert row["gpu_declaration"] == "NOT_DECLARED"
    assert f"gpu:{bundle.request_id}@qe-public-cuda-strict@" in row["run_evidence_refs"]

    # Evidence is revalidated during generation, not trusted after construction.
    bundle.metadata.pop("backend_used")
    destination = tmp_path / "gpu-provenance-lost.csv"
    matrix.generate_capability_matrix(str(destination), run_evidence=[evidence])
    with destination.open(newline="", encoding="utf-8") as handle:
        row = {r["metric_id"]: r for r in csv.DictReader(handle)}["rank_ic"]
    assert row["public_gpu"] == "NOT_RUN"
