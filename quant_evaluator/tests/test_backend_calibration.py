from dataclasses import dataclass, field
import time

import numpy as np

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.runtime import backend_calibration as calibration


@dataclass
class Result:
    factor_ids: tuple
    metric_values: dict
    grouped_metrics: dict
    artifacts: dict
    factor_artifacts: dict
    diagnostics: dict = field(default_factory=dict)
    metric_versions: dict = field(default_factory=dict)
    instance_results: dict = field(default_factory=dict)
    instance_specs: dict = field(default_factory=dict)
    warnings: tuple = ()


def _inputs():
    times = np.array([1, 2, 3])
    assets = np.array(["a", "b"])
    batch = FactorBatch(
        factor_ids=("f",), time_axis=AxisRef("time", "int64", 3, times),
        asset_axis=AxisRef("asset", "str", 2, assets),
        values=np.arange(6.0).reshape(3, 2, 1),
    )
    label = LabelBundle(
        target_id="y", values=np.arange(6.0).reshape(3, 2), horizon=1,
        decision_time=(1, 2, 3), label_start_time=(2, 3, 4),
        label_end_time=(3, 4, 5), asset_axis=batch.asset_axis,
    )
    return batch, label


def _result(batch, value=1.0):
    return Result(tuple(batch.factor_ids), {"ic": np.array([value])}, {}, {}, {})


def test_cache_is_lru_and_expires():
    cache = calibration.BoundedCalibrationCache(max_entries=1, ttl_seconds=0.01)
    cache.put("a", {"winner": "cpu"})
    assert cache.get("a")["winner"] == "cpu"
    cache.put("b", {"winner": "cpu"})
    assert cache.get("a") is None
    time.sleep(0.02)
    assert cache.get("b") is None


def test_request_key_includes_label_axes_and_policy(monkeypatch):
    batch, label = _inputs()
    changed_label = LabelBundle(
        target_id="y", values=np.arange(6.0).reshape(3, 2), horizon=1,
        decision_time=(1, 2, 3), label_start_time=(2, 3, 4),
        label_end_time=(3, 4, 5),
        asset_axis=AxisRef("asset", "str", 2, np.array(["b", "a"])),
    )
    assert calibration._request_fingerprint(batch, label, ("ic",)) != calibration._request_fingerprint(
        batch, changed_label, ("ic",))
    assert calibration._hash_array
    assert calibration.CalibrationPolicy(absolute_tolerance=1e-8) != calibration.CalibrationPolicy()
    integer_objects = np.array([1, 2, 3], dtype=object)
    string_objects = np.array(["1", "2", "3"], dtype=object)
    object_int_batch = FactorBatch(
        batch.factor_ids, AxisRef("time", "object", 3, integer_objects),
        batch.asset_axis, batch.values)
    object_str_batch = FactorBatch(
        batch.factor_ids, AxisRef("time", "object", 3, string_objects),
        batch.asset_axis, batch.values)
    assert calibration._request_fingerprint(object_int_batch, label, ("ic",)) != calibration._request_fingerprint(
        object_str_batch, label, ("ic",))


def test_calibrates_whole_calls_and_returns_fastest(monkeypatch):
    batch, label = _inputs()
    monkeypatch.setattr(calibration, "_source_fingerprint", lambda: "src")
    monkeypatch.setattr(calibration, "_runtime_fingerprint", lambda device: {"device": device})
    monkeypatch.setattr(calibration, "_device_admission", lambda policy: (None, {"id": 0}))
    calls = []

    now = [0.0]
    monkeypatch.setattr(calibration.time, "monotonic", lambda: now[0])

    def evaluate(_batch, _label, *, metrics, backend, gpu_policy):
        calls.append(backend)
        now[0] += 0.5 if backend == "cpu" else 0.1
        return _result(_batch)

    output = calibration.evaluate_calibrated_batch(
        batch, label, metrics=("ic",), calibration_policy=calibration.CalibrationPolicy(
            repetitions=2, warmups=0), cache=calibration.BoundedCalibrationCache(),
        gpu_policy=GPUExecutionPolicy(), evaluate_fn=evaluate,
    )
    assert calls == ["cpu", "cuda_strict", "cuda_strict", "cpu"]
    assert output.metadata["status"] == "calibrated"
    assert output.metadata["winner"] == "cuda_strict"
    assert output.bundle is not None
    assert output.metadata["calibration_record"]["selection_basis"] == "steady_state_median"
    assert "timing_winner" not in output.metadata["calibration_record"]
    assert output.metadata["calibration_record"]["cpu_first_call_seconds"] == 0.5
    assert np.isclose(output.metadata["calibration_record"]["cuda_first_call_seconds"], 0.1)


def test_parity_failure_falls_back_to_cpu_and_is_not_cached(monkeypatch):
    batch, label = _inputs()
    monkeypatch.setattr(calibration, "_source_fingerprint", lambda: "src")
    monkeypatch.setattr(calibration, "_runtime_fingerprint", lambda device: {"device": device})
    monkeypatch.setattr(calibration, "_device_admission", lambda policy: (None, {"id": 0}))
    cache = calibration.BoundedCalibrationCache()

    now = [0.0]
    monkeypatch.setattr(calibration.time, "monotonic", lambda: now[0])
    calls = []

    def evaluate(_batch, _label, *, metrics, backend, gpu_policy):
        calls.append(backend)
        now[0] += 0.5 if backend == "cpu" else 0.1
        # First pair diverges; later pairs would match. Calibration must stop
        # on the first paired failure rather than hide it with a later result.
        return _result(_batch, 1.0 if backend == "cpu" or len(calls) > 2 else 2.0)

    output = calibration.evaluate_calibrated_batch(
        batch, label, metrics=("ic",), calibration_policy=calibration.CalibrationPolicy(
            repetitions=2, warmups=0), cache=cache, gpu_policy=GPUExecutionPolicy(),
        evaluate_fn=evaluate,
    )
    assert output.metadata["status"] == "parity_failed_cpu_fallback"
    assert output.metadata["winner"] == "cpu"
    record = output.metadata["calibration_record"]
    assert record["winner"] == "cpu"
    assert record["timing_winner"] == "cuda_strict"
    assert record["timing_winner"] == (
        "cpu" if record["cpu_median_seconds"] <= record["cuda_median_seconds"]
        else "cuda_strict"
    )
    assert len(cache) == 0
    assert calls == ["cpu", "cuda_strict"]


def test_parity_compares_artifact_values_and_validity_masks():
    @dataclass
    class Artifact:
        values: np.ndarray
        validity: np.ndarray
        observation_count: int
        axis: tuple

    left = Result(("f",), {}, {}, {"a": Artifact(np.array([1.0, np.nan]),
                                                    np.array([True, False]), 1, ("x",))}, {})
    equal = Result(("f",), {}, {}, {"a": Artifact(np.array([1.0, np.nan]),
                                                    np.array([True, False]), 1, ("x",))}, {})
    changed_mask = Result(("f",), {}, {}, {"a": Artifact(np.array([1.0, np.nan]),
                                                            np.array([True, True]), 1, ("x",))}, {})
    assert calibration._parity_mismatch(left, equal, calibration.CalibrationPolicy()) is None
    assert "values differ" in calibration._parity_mismatch(
        left, changed_mask, calibration.CalibrationPolicy())
    changed_counts = Result(("f",), {}, {}, {}, {}, diagnostics={"f": {"n": 4}})
    baseline = Result(("f",), {}, {}, {}, {}, diagnostics={"f": {"n": 3}})
    assert "values differ" in calibration._parity_mismatch(
        baseline, changed_counts, calibration.CalibrationPolicy())


def test_metric_artifact_parity_ignores_execution_provenance_only():
    from quant_evaluator.contracts.metric_artifacts import ScalarMetricArtifact

    cpu = ScalarMetricArtifact(metric_id="ic", domain="cross_sectional",
                               values=np.array([0.25]),
                               provenance={"backend": "cpu", "config": {"method": "pearson"}})
    cuda = ScalarMetricArtifact(metric_id="ic", domain="cross_sectional",
                                values=np.array([0.25]),
                                provenance={"backend": "cuda", "config": {"method": "pearson"}})
    bad_config = ScalarMetricArtifact(metric_id="ic", domain="cross_sectional",
                                      values=np.array([0.25]),
                                      provenance={"backend": "cuda", "config": {"method": "rank"}})
    assert calibration._compare_metric_values(cpu, cuda, rtol=1e-10, atol=1e-12,
                                              path="artifact") is None
    assert calibration._compare_metric_values(cpu, bad_config, rtol=1e-10, atol=1e-12,
                                              path="artifact") is not None
