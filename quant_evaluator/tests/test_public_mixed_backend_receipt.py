"""Public execution receipts identify per-metric CPU/GPU execution."""

import numpy as np
import pytest

from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate


def test_explicit_cuda_receipt_marks_cpu_adaptive_count_and_hashes_route():
    pytest.importorskip("cupy")
    t, n = 20, 2000
    values = np.broadcast_to(np.arange(n, dtype=np.float64)[None, :, None], (t, n, 1))
    labels = np.broadcast_to(np.arange(n, dtype=np.float64)[None, :], (t, n))
    batch = FactorBatch(
        ("f0",), AxisRef("time", "int64", t), AxisRef("asset", "int64", n), values
    )
    label_bundle = LabelBundle(
        "next_ret", labels, 1,
        decision_time=tuple(range(t)),
        label_start_time=tuple(range(1, t + 1)),
        label_end_time=tuple(range(2, t + 2)),
    )

    result = evaluate(
        batch, label_bundle,
        metrics=("adaptive_quantile_count", "rank_ic"),
        backend="cuda_strict",
    )

    expected_backends = {"adaptive_quantile_count": "cpu", "rank_ic": "cuda"}
    receipt = result.metadata["execution_receipt"]
    assert result.metadata["metric_backends"] == expected_backends
    assert receipt["metric_backends"] == expected_backends
    assert receipt["backend_used"] == result.metadata["backend_used"] == "cuda"
    route = {
        key: receipt[key]
        for key in (
            "backend_requested", "backend_strategy", "backend_used",
            "auto_backend_policy", "auto_backend_profile", "auto_backend_reason",
            "metric_backends",
        )
    }
    assert receipt["receipt_hash"] == stable_content_hex(
        tag="EvaluationExecutionReceipt.v1",
        fields={"config_hash": receipt["config_hash"], **route},
    )
