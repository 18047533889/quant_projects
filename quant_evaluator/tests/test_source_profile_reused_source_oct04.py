"""Qualified auto validates current reads, not a reusable source's whole history."""
import numpy as np
import pytest

from test_source_profile_default_cache_oct04 import (
    api, router, _install_live_context_capture, _Source, _labels, _records,
    rebind_pair_to_cpu_outputs, GPUExecutionPolicy,
)
from quant_evaluator.contracts.factor_batch import AxisRef


@pytest.fixture(autouse=True)
def isolated_cache():
    router.profile_cache.clear_validated_records()
    yield
    router.profile_cache.clear_validated_records()


def _qualified_fixture(monkeypatch):
    policy = GPUExecutionPolicy(max_factor_tile_size=2)
    _install_live_context_capture(monkeypatch, policy)
    source = _Source()
    source.time_axis = AxisRef("time", "int64", 65, np.arange(65, dtype=np.int64))
    source.asset_axis = AxisRef("asset", "int64", 64, np.arange(64, dtype=np.int64))
    source.values = np.random.default_rng(73004).normal(size=(65, 64, 5))
    labels = _labels(source)
    metadata = api.capture_factor_tile_source(source)
    fingerprint = api._source_request_fingerprint(metadata, labels, ("rank_ic",))
    context = router.capture_source_route_profile_context(
        source=source, metadata=metadata, metrics=("rank_ic",),
        request_fingerprint=fingerprint, requested_tile_size=16, policy=policy)
    expected = api.evaluate_factor_source_batch(
        source, labels, metrics=("rank_ic",), backend="cpu", max_tile_size=16,
        gpu_policy=policy)
    assert np.isfinite(expected.scalar_metrics["rank_ic"]).all()
    pair = rebind_pair_to_cpu_outputs(_records(context, cpu_width=5, gpu_width=2), expected)
    return source, labels, policy, pair, expected


@pytest.mark.parametrize("retain_initial_history", [False, True])
def test_same_source_repeated_auto_preserves_history_and_qualification(
        monkeypatch, retain_initial_history):
    source, labels, policy, pair, expected = _qualified_fixture(monkeypatch)
    if not retain_initial_history:
        source.reads.clear()
    initial_history = tuple(source.reads)
    for run_index in range(3):
        kwargs = {"source_qualification": pair} if run_index == 0 else {}
        output = api.evaluate_factor_source_batch(
            source, labels, metrics=("rank_ic",), backend="auto", max_tile_size=16,
            gpu_policy=policy, **kwargs)
        assert output.metadata["source_qualification_status"] == "qualified_current_source"
        assert output.metadata["source_qualification_applied"] is True
        assert output.metadata["source_qualification_cache_status"] == (
            "supplied" if run_index == 0 else "cache_hit")
        assert tuple(source.reads) == initial_history + ((0, 5),) * (run_index + 1)
        np.testing.assert_array_equal(output.scalar_metrics["rank_ic"],
                                      expected.scalar_metrics["rank_ic"])


@pytest.mark.parametrize("tamper", ["modify_prefix", "erase_prefix", "replay_old_reads", "extra_read"])
def test_reused_source_rejects_changed_history_and_replayed_receipts(monkeypatch, tamper):
    source, labels, policy, pair, _ = _qualified_fixture(monkeypatch)
    # Leave one earlier CPU read in history. The current invocation must append
    # a new exact schedule without changing any earlier entry.
    original_read = source.read_tile
    def damaged_read(start, end):
        tile = original_read(start, end)
        if tamper == "modify_prefix":
            source.reads[0] = (1, 5)
        elif tamper == "erase_prefix":
            source.reads[:] = source.reads[-1:]
        elif tamper == "replay_old_reads":
            source.reads.pop()
        else:
            source.reads.append((0, 5))
        return tile
    source.read_tile = damaged_read
    output = api.evaluate_factor_source_batch(
        source, labels, metrics=("rank_ic",), backend="auto", max_tile_size=16,
        gpu_policy=policy, source_qualification=pair)
    assert output.metadata["source_qualification_status"] == "execution_configuration_deviated"
    assert output.metadata["source_qualification_applied"] is False
    assert output.metadata["source_qualification_reason"] == "qualified_profile_execution_receipt_deviated"


@pytest.mark.parametrize("bad_history", [(-1, 5), (0, 0), (False, 5), ("0", 5), (0, 5, 1), (0, 6)])
def test_malformed_preexisting_history_is_not_accepted(monkeypatch, bad_history):
    source, labels, policy, pair, _ = _qualified_fixture(monkeypatch)
    source.reads[:] = [bad_history]
    output = api.evaluate_factor_source_batch(
        source, labels, metrics=("rank_ic",), backend="auto", max_tile_size=16,
        gpu_policy=policy, source_qualification=pair)
    assert output.metadata["source_qualification_applied"] is False
    assert output.metadata["source_qualification_status"] == "execution_configuration_deviated"
    assert output.metadata["source_qualification_reason"] == "qualified_profile_execution_receipt_deviated"


def test_real_cuda_reused_source_retains_per_run_qualification(monkeypatch):
    """Real CUDA arithmetic, synthetic timing/context; NOT throughput evidence."""
    import os
    from dataclasses import replace
    from scipy.stats import spearmanr
    from source_profile_fixture_identity_oct04 import _run_receipt
    from quant_evaluator.scripts.source_profile_measurement import (
        build_backend_profile_measurement, build_metric_comparison_receipts,
    )
    if os.environ.get("QE_RUN_SOURCE_PROFILE_CUDA") != "1":
        pytest.skip("opt in to actual source-profile CUDA checks")
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")
    source, labels, policy, pair, cpu = _qualified_fixture(monkeypatch)
    context = pair[0].context
    cuda = api.evaluate_factor_source_batch(
        source, labels, metrics=("rank_ic",), backend="cuda_strict",
        max_tile_size=2, gpu_policy=policy)
    independent = np.array([
        [spearmanr(source.values[t, :, f], labels.values[t]).statistic
         for f in range(5)] for t in range(65)]).mean(axis=0)
    np.testing.assert_allclose(cuda.scalar_metrics["rank_ic"], independent, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(cuda.observation_counts["rank_ic"], np.full(5, 65))
    np.testing.assert_array_equal(cpu.observation_counts["rank_ic"], cuda.observation_counts["rank_ic"])
    rebound = []
    for record in pair:
        gpu_profile = build_backend_profile_measurement(
            cuda, _run_receipt(cuda, record.cuda, context, "cuda"), context,
            correctness_validated=True)
        # Only establish a typed routing fixture. These times are synthetic;
        # this test creates no measured report or production authorization.
        gpu_profile = replace(gpu_profile, seconds=1.0)
        cpu_out, gpu_out = record.cpu.outputs[0], gpu_profile.outputs[0]
        error = float(np.max(np.abs(cpu.scalar_metrics["rank_ic"] - cuda.scalar_metrics["rank_ic"])))
        comparison = {"pass": True, "metrics": {"rank_ic": {
            "pass": True, "compared_value_count": 5, "artifact_kind": "scalar",
            "cpu_shape": [5], "cuda_shape": [5], "shape_valid": True,
            "factor_count": 5, "observation_counts_shape_valid": True,
            "cpu_observation_counts_sha256": cpu_out.observation_counts_sha256,
            "cuda_observation_counts_sha256": gpu_out.observation_counts_sha256,
            "finite_mask_equal": True, "observation_counts_equal": True,
            "max_abs_error": error, "cpu_values_sha256": cpu_out.values_sha256,
            "cuda_values_sha256": gpu_out.values_sha256,
        }}}
        receipts = build_metric_comparison_receipts(
            cpu, cuda, context, comparison, cpu_profile=record.cpu, cuda_profile=gpu_profile)
        rebound.append(replace(record, cuda=gpu_profile, comparison_evidence=receipts))
    initial_history = tuple(source.reads)
    expected_ranges = ((0, 2), (2, 4), (4, 5))
    for run_index in range(3):
        kwargs = {"source_qualification": tuple(rebound)} if run_index == 0 else {}
        output = api.evaluate_factor_source_batch(
            source, labels, metrics=("rank_ic",), backend="auto", max_tile_size=16,
            gpu_policy=policy, **kwargs)
        assert output.metadata["backend_used"] == "cuda"
        assert output.metadata["factor_tile_size"] == 2 and output.metadata["oom_retries"] == 0
        assert output.metadata["source_qualification_status"] == "qualified_current_source"
        assert output.metadata["source_qualification_applied"] is True
        assert output.metadata["source_qualification_cache_status"] == (
            "supplied" if run_index == 0 else "cache_hit")
        assert tuple(source.reads) == initial_history + expected_ranges * (run_index + 1)
        np.testing.assert_array_equal(output.scalar_metrics["rank_ic"], cuda.scalar_metrics["rank_ic"])
