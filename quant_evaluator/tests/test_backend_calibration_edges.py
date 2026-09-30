"""Cache-selection branch tests with a deterministic clock, no CUDA required."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from quant_evaluator.runtime import backend_calibration as c
from quant_evaluator.runtime import evaluator as public_runtime
from quant_evaluator.tests.test_backend_calibration import _inputs, _result


@pytest.fixture
def harness(monkeypatch):
    now = [0.0]
    calls = []
    durations = {"cpu": 2.0, "cuda_strict": 1.0}
    admission = [None]
    source = ["source-v1"]
    monkeypatch.setattr(c, "_PROCESS_SOURCE_DIGEST", None)
    monkeypatch.setattr(c, "_PROCESS_SOURCE_DRIFTED", False)
    monkeypatch.setattr(c.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(c, "_source_fingerprint", lambda: source[0])
    monkeypatch.setattr(c, "_runtime_fingerprint", lambda device: {"device": device})
    monkeypatch.setattr(c, "_device_admission", lambda policy: (admission[0], {"id": 0}))

    def evaluate(batch, label, *, metrics, backend, gpu_policy):
        calls.append(backend)
        now[0] += durations[backend]
        return _result(batch)

    monkeypatch.setattr(public_runtime, "evaluate", evaluate)
    batch, label = _inputs()
    cache = c.BoundedCalibrationCache()
    policy = c.CalibrationPolicy(repetitions=2, warmups=0)

    def run(current_batch=batch, current_label=label, current_policy=policy):
        return c.evaluate_calibrated_batch(current_batch, current_label,
            metrics=("ic",), calibration_policy=current_policy, cache=cache)

    return run, calls, durations, admission, source, batch, label, policy, cache


@pytest.mark.parametrize("durations,winner", [
    ({"cpu": 1.0, "cuda_strict": 2.0}, "cpu"),
    ({"cpu": 2.0, "cuda_strict": 1.0}, "cuda_strict"),
    ({"cpu": 1.0, "cuda_strict": 1.0}, "cpu"),
])
def test_cache_reruns_selected_route_and_ties_choose_cpu(harness, durations, winner):
    run, calls, times, *_ = harness
    times.update(durations)
    first = run()
    assert first.metadata["winner"] == winner
    assert calls == ["cpu", "cuda_strict", "cuda_strict", "cpu"]
    second = run()
    assert second.metadata["status"] == "cache_hit"
    assert second.metadata["winner"] == winner
    assert calls[-1] == winner and len(calls) == 5
    assert second.bundle is not first.bundle


def test_cache_hit_cannot_override_current_device_admission(harness):
    run, calls, _, admission, *_ = harness
    assert run().metadata["winner"] == "cuda_strict"
    admission[0] = "insufficient_vram"
    result = run()
    assert result.metadata["status"] == "cpu_only_device_admission"
    assert result.metadata["winner"] == "cpu"
    assert calls == ["cpu", "cuda_strict", "cuda_strict", "cpu", "cpu"]


def test_source_drift_disables_cache_even_if_disk_version_returns(harness):
    run, calls, _, _, source, *rest = harness
    run()
    source[0] = "source-v2"
    assert run().metadata["status"] == "calibrated"
    source[0] = "source-v1"
    assert run().metadata["status"] == "calibrated"
    assert len(calls) == 12
    assert len(rest[-1]) == 1


def test_content_mask_and_tolerance_changes_are_cache_misses(harness):
    run, calls, _, _, _, batch, label, policy, cache = harness
    run()
    changed_values = batch.values.copy()
    changed_values[0, 0, 0] += 1
    assert run(replace(batch, values=changed_values)).metadata["status"] == "calibrated"
    mask = np.ones(batch.values.shape, dtype=bool)
    mask[0, 0, 0] = False
    assert run(replace(batch, validity=mask)).metadata["status"] == "calibrated"
    assert run(current_policy=replace(policy, absolute_tolerance=1e-8)).metadata["status"] == "calibrated"
    assert len(cache) == 4 and len(calls) == 16


@pytest.mark.parametrize("array", [np.empty((0, 5)), np.array(1.5),
    np.arange(120., dtype=float).reshape(10, 12)[:, ::2]])
def test_array_hash_matches_logical_c_bytes_including_empty_and_strided(array):
    actual = hashlib.sha256()
    c._hash_array(actual, array, chunk_bytes=16)
    oracle = hashlib.sha256()
    oracle.update(str(array.dtype).encode("ascii"))
    oracle.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    oracle.update(array.tobytes(order="C"))
    assert actual.digest() == oracle.digest()


def test_admitted_public_cpu_cuda_bundles_and_cached_route(monkeypatch):
    """Run real kernels where this conservative device gate admits CUDA."""
    from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
    from quant_evaluator.scripts.benchmark_backend_tournament import panel

    reason, _ = c._device_admission(GPUExecutionPolicy())
    if reason is not None:
        pytest.skip("CUDA calibration device admission: " + str(reason))
    monkeypatch.setattr(c, "_PROCESS_SOURCE_DIGEST", None)
    monkeypatch.setattr(c, "_PROCESS_SOURCE_DRIFTED", False)
    batch, label = panel(64, 512, 8, 20260930)
    cache = c.BoundedCalibrationCache()
    policy = c.CalibrationPolicy(repetitions=1, warmups=0)
    kwargs = dict(metrics=("pearson_ic", "pearson_ic_series", "pearson_ic_std",
                           "pearson_ic_ir"), calibration_policy=policy, cache=cache)
    first = c.evaluate_calibrated_batch(batch, label, **kwargs)
    second = c.evaluate_calibrated_batch(batch, label, **kwargs)
    assert first.metadata["calibration_record"]["parity"] == "pass"
    assert second.metadata["status"] == "cache_hit"
    assert second.metadata["winner"] == first.metadata["winner"]
    assert second.bundle is not first.bundle
    assert first.bundle.artifacts["pearson_ic_series"].values.shape == (64, 8)
    assert c._parity_mismatch(first.bundle, second.bundle, policy) is None


@pytest.mark.parametrize("scalar_type", [int, np.int64, np.uint64])
def test_large_integer_counts_are_compared_exactly(scalar_type):
    left = scalar_type(10**12)
    right = scalar_type(10**12 + 1)
    assert c._compare_metric_values(left, right, rtol=1e-10, atol=1e-12,
                                    path="sample_count") is not None
    assert c._compare_metric_values(left, scalar_type(10**12), rtol=1e-10,
                                    atol=1e-12, path="sample_count") is None


def test_float_measurements_still_use_configured_tolerance():
    assert c._compare_metric_values(1.0, 1.0 + 1e-12, rtol=1e-10,
                                    atol=1e-12, path="measurement") is None
    assert c._compare_metric_values(1.0, 1.1, rtol=1e-10,
                                    atol=1e-12, path="measurement") is not None
