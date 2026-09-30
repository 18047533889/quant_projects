"""Public measured-auto dispatch, receipts, and input-preservation guards."""
import pytest

from quant_evaluator import AutoCalibrationOptions, evaluate
from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.runtime import auto_calibration as facade
from quant_evaluator.runtime import backend_calibration as c
from quant_evaluator.tests.test_backend_calibration import _inputs


def test_cpu_only_admission_returns_public_bundle_with_valid_receipt(monkeypatch):
    monkeypatch.setattr(c, "_device_admission", lambda _: ("no_cuda", None))
    batch, label = _inputs()
    result = evaluate(batch, label, metrics=("coverage",), backend="auto",
                      auto_calibration=AutoCalibrationOptions())
    assert result.metadata["backend_used"] == "cpu"
    assert result.metadata["backend_strategy"] == "calibrated_auto"
    assert result.metadata["auto_calibration"]["status"] == "cpu_only_device_admission"
    receipt = dict(result.metadata["execution_receipt"])
    actual_hash = receipt.pop("receipt_hash")
    assert actual_hash == stable_content_hex(tag="EvaluationExecutionReceipt.v1", fields=receipt)
    assert result.artifacts["coverage"].factor_axis.factor_ids == ("f",)


def test_facade_does_not_mutate_selected_bundle_or_semantic_config(monkeypatch):
    batch, label = _inputs()
    original = evaluate(batch, label, metrics=("coverage",), backend="cpu")
    old_metadata = dict(original.metadata)
    calls = []

    def calibrated(*args, **kwargs):
        calls.append((args, kwargs))
        return c.CalibratedEvaluation(original, {"winner": "cpu", "status": "cache_hit"})

    monkeypatch.setattr(facade, "evaluate_calibrated_batch", calibrated)
    options = AutoCalibrationOptions()
    result = evaluate(batch, label, auto_calibration=options)
    assert original.metadata == old_metadata
    assert result is not original and result.config_hash == original.config_hash
    assert result.artifacts is original.artifacts
    assert result.metadata["backend_requested"] == "default"
    assert calls[0][1]["cache"] is options.cache
    assert calls[0][1]["metrics"] == ("coverage",)


@pytest.mark.parametrize("backend", ["cpu", "cuda", "cuda_strict", "gpu"])
def test_manual_backend_cannot_be_silently_replaced(backend):
    batch, label = _inputs()
    with pytest.raises(InvalidContractError, match="requires backend"):
        evaluate(batch, label, backend=backend, auto_calibration=AutoCalibrationOptions())


@pytest.mark.parametrize("name", [
    "context", "where", "evaluator", "split_ref", "metric_parameters",
    "portfolio_returns", "holding_returns", "portfolio_spec", "trade_eligibility",
    "calendar_snapshot", "exposure_panel", "quantile_builder_parameters",
    "generalization_evidence", "_auto_route_override", "_gpu_result_override",
    "_diagnostics_override",
])
def test_specialized_inputs_are_never_silently_dropped(name):
    batch, label = _inputs()
    with pytest.raises(InvalidContractError, match="supports only materialized"):
        evaluate(batch, label, auto_calibration=AutoCalibrationOptions(), **{name: {}})


def test_invalid_options_fail_closed():
    batch, label = _inputs()
    with pytest.raises(InvalidContractError, match="AutoCalibrationOptions"):
        evaluate(batch, label, auto_calibration=True)
    with pytest.raises(InvalidContractError, match="policy"):
        AutoCalibrationOptions(policy={})
    with pytest.raises(InvalidContractError, match="cache"):
        AutoCalibrationOptions(cache={})


def test_real_public_cuda_calibration_and_cache_receipts(monkeypatch):
    from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
    from quant_evaluator.scripts.benchmark_backend_tournament import panel
    reason, _ = c._device_admission(GPUExecutionPolicy())
    if reason:
        pytest.skip("device admission: " + str(reason))
    monkeypatch.setattr(c, "_PROCESS_SOURCE_DIGEST", None)
    monkeypatch.setattr(c, "_PROCESS_SOURCE_DRIFTED", False)
    batch, label = panel(64, 512, 8, 20261001)
    options = AutoCalibrationOptions(policy=c.CalibrationPolicy(repetitions=1, warmups=0))
    kwargs = dict(metrics=("pearson_ic", "pearson_ic_series"), auto_calibration=options)
    first = evaluate(batch, label, **kwargs)
    second = evaluate(batch, label, **kwargs)
    assert first.metadata["auto_calibration"]["calibration_record"]["parity"] == "pass"
    assert second.metadata["auto_calibration"]["status"] == "cache_hit"
    assert second.metadata["backend_used"] == first.metadata["backend_used"]
    assert c._parity_mismatch(first, second, options.policy) is None
