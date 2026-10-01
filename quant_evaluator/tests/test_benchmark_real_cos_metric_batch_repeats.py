from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from quant_evaluator.scripts import benchmark_real_cos_metric_batch as harness


@dataclass
class _CoverageArtifact:
    values: np.ndarray
    valid_mask: np.ndarray
    counts: np.ndarray
    provenance: dict
    artifact_kind: str = "coverage"


def _bundle(backend, reason, values, *, valid_mask=(True, False), counts=(11, 13)):
    artifact = _CoverageArtifact(
        values=np.asarray(values, dtype=np.float64),
        valid_mask=np.asarray(valid_mask),
        counts=np.asarray(counts),
        provenance={"observation_counts": [11, 13]},
    )
    return SimpleNamespace(
        artifacts={"coverage": artifact},
        grouped_metrics={"factor-a": {"coverage": None},
                         "factor-b": {"coverage": None}},
        factor_ids=("factor-a", "factor-b"),
        metadata={
            "backend_used": backend,
            "auto_backend_reason": reason,
            "auto_backend_profile": "f32-coverage",
            "metric_backends": {"coverage": backend},
            "execution_receipt": {"backend_used": backend, "reason": reason},
        },
        config_hash="f" * 64,
    )


class _Connection:
    def send(self, payload):
        self.payload = payload

    def close(self):
        pass


@pytest.mark.parametrize("compact", (False, True))
def test_auto_repeat_receipts_capture_cpu_cold_and_cuda_warm(monkeypatch, compact):
    bundles = [
        _bundle("cpu", "first_auto_call_cold_fallback", [0.75, 1.0]),
        _bundle("cuda", "bounded_f32_coverage_gpu", [0.75 + 1e-12, 1.0]),
    ]
    requested_backends = []

    def evaluate(_batch, _labels, *, metrics, backend):
        requested_backends.append(backend)
        return bundles.pop(0)

    monkeypatch.setitem(
        __import__("sys").modules,
        "quant_evaluator.runtime.evaluator",
        SimpleNamespace(evaluate=evaluate),
    )
    monkeypatch.setattr(harness, "_BATCH", object())
    monkeypatch.setattr(harness, "_LABELS", object())
    monkeypatch.setattr(harness, "METRICS", harness.F32_COVERAGE_SINGLE)

    connection = _Connection()
    harness._worker(connection, "auto", 2)
    result = connection.payload

    assert result["status"] == "ok"
    assert requested_backends == ["auto", "auto"]
    assert len(result["seconds"]) == 2
    assert len(result["repeat_receipts"]) == 2
    cold, warm = result["repeat_receipts"]
    assert (cold["backend_used"], warm["backend_used"]) == ("cpu", "cuda")
    assert cold["auto_backend_reason"] == "first_auto_call_cold_fallback"
    assert warm["auto_backend_reason"] == "bounded_f32_coverage_gpu"
    assert cold["config_hash"] == warm["config_hash"] == "f" * 64
    assert cold["metrics"]["coverage"]["observation_counts"] == [11, 13]
    for field in ("finite_mask_sha256", "valid_mask_sha256", "counts_sha256",
                  "observation_counts_sha256"):
        assert cold["metrics"]["coverage"][field] == warm["metrics"]["coverage"][field]
    assert harness._compare_repeat_receipt(cold, cold)["pass"] is True
    assert harness._compare_repeat_receipt(cold, warm)["pass"] is None
    assert (cold["metrics"]["coverage"]["artifact_sha256"] !=
            warm["metrics"]["coverage"]["artifact_sha256"])
    numeric = harness._compare_repeat_numeric(cold, warm)
    assert numeric["status"] == "compared_values_rtol_1e-8_atol_1e-10"
    assert numeric["pass"] is True
    assert cold["value_snapshots"]["coverage"] == [0.75, 1.0]
    assert warm["value_snapshots"]["coverage"] == [0.75 + 1e-12, 1.0]
    if compact:
        compact_result = harness._compact_repeat_run(result)
        assert "value_snapshots" not in compact_result["repeat_receipts"][0]
    else:
        assert result["repeat_receipts"][0]["value_snapshots"]["coverage"] == [0.75, 1.0]
    assert result["backend_used"] == "cuda"
    assert result["artifacts"]["coverage"]["values"] == [0.75 + 1e-12, 1.0]


@pytest.mark.parametrize("mutation", ("swapped_masks_counts", "altered_values",
                                       "missing_snapshots"))
def test_coverage_repeat_numeric_parity_fails_closed(monkeypatch, mutation):
    monkeypatch.setattr(harness, "METRICS", harness.F32_COVERAGE_SINGLE)
    reference = harness._repeat_receipt(
        _bundle("cpu", "cold_cpu", [0.75, 1.0]), 0.01, 1)
    if mutation == "swapped_masks_counts":
        candidate_bundle = _bundle(
            "cuda", "warm_cuda", [0.75, 1.0],
            valid_mask=(False, True), counts=(13, 11))
    elif mutation == "altered_values":
        candidate_bundle = _bundle("cuda", "warm_cuda", [0.8, 1.0])
    else:
        candidate_bundle = _bundle("cuda", "warm_cuda", [0.75, 1.0])
    candidate = harness._repeat_receipt(candidate_bundle, 0.01, 2)
    if mutation == "missing_snapshots":
        candidate.pop("value_snapshots")
    result = harness._compare_repeat_numeric(reference, candidate)
    assert result["pass"] is False
    if mutation == "swapped_masks_counts":
        ref_counts = reference["metrics"]["coverage"]
        candidate_counts = candidate["metrics"]["coverage"]
        assert ref_counts["valid_count"] == candidate_counts["valid_count"]
        assert ref_counts["count_sum"] == candidate_counts["count_sum"]
        assert ref_counts["valid_mask_sha256"] != candidate_counts["valid_mask_sha256"]
        assert ref_counts["counts_sha256"] != candidate_counts["counts_sha256"]
