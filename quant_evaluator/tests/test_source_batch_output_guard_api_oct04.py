"""Public API regressions for malformed source batch executor results.

The defect is that evaluate_factor_source_batch can publish a malformed
executor bundle when profile execution checks are absent or do not inspect
the structural field that was corrupted.
"""

from dataclasses import replace

import numpy as np
import pytest

from quant_evaluator.api import factor_source as api
from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from test_source_route_profiles_api_oct03 import _Source, _labels
from quant_evaluator.runtime import source_profile_router as router
from test_source_profile_live_guards_oct04 import _records
from test_source_profile_default_cache_oct04 import _install_live_context_capture
from source_profile_fixture_identity_oct04 import rebind_pair_to_cpu_outputs


@pytest.mark.parametrize("corruption", [
    "factor_ids", "float_counts", "short_counts", "wrong_type", "label",
    "group", "metric_shape", "metric_dtype", "negative_counts",
    "extra_counts", "vector_metric", "metadata_type", "bool_counts",
    "uint_overflow",
])
def test_explicit_cpu_rejects_malformed_executor_bundle(monkeypatch, corruption):
    source = _Source()
    labels = _labels(source)
    original = api._cpu_source_batch

    def malformed(*args, **kwargs):
        output = original(*args, **kwargs)
        if corruption == "factor_ids":
            return replace(output, factor_ids=output.factor_ids[::-1])
        if corruption == "wrong_type":
            return object()
        if corruption == "label":
            return replace(output, label_id="wrong-label")
        if corruption == "vector_metric":
            vectors = {"rank_ic": np.ones((1, len(output.factor_ids)))}
            return replace(output, vector_metrics=vectors)
        if corruption == "metadata_type":
            return replace(output, metadata=[])
        if corruption == "group":
            values = dict(output.scalar_metrics)
            series = dict(output.series_metrics)
            series["rank_ic"] = values.pop("rank_ic")
            return replace(output, scalar_metrics=values, series_metrics=series)
        if corruption == "metric_shape":
            values = dict(output.scalar_metrics)
            values["rank_ic"] = values["rank_ic"][:-1]
            return replace(output, scalar_metrics=values)
        if corruption == "metric_dtype":
            values = dict(output.scalar_metrics)
            values["rank_ic"] = np.ones(values["rank_ic"].shape, dtype=np.int64)
            return replace(output, scalar_metrics=values)
        counts = dict(output.observation_counts)
        if corruption == "float_counts":
            counts["rank_ic"] = counts["rank_ic"].astype(float)
        elif corruption == "negative_counts":
            counts["rank_ic"] = counts["rank_ic"].copy()
            counts["rank_ic"][0] = -1
        elif corruption == "extra_counts":
            counts["unexpected"] = np.zeros_like(counts["rank_ic"])
        elif corruption == "bool_counts":
            counts["rank_ic"] = counts["rank_ic"].astype(bool)
        elif corruption == "uint_overflow":
            counts["rank_ic"] = np.full(
                counts["rank_ic"].shape, np.iinfo(np.uint64).max, dtype=np.uint64)
        else:
            counts["rank_ic"] = counts["rank_ic"][:-1]
        return replace(output, observation_counts=counts)

    monkeypatch.setattr(api, "_cpu_source_batch", malformed)
    with pytest.raises(InvalidContractError):
        api.evaluate_factor_source_batch(
            source, labels, metrics=("rank_ic",), backend="cpu", max_tile_size=16)


def test_unqualified_auto_cpu_rejects_wrong_label_and_wrong_group(monkeypatch):
    source = _Source()
    labels = _labels(source)
    original = api._cpu_source_batch

    def wrong_bundle(*args, **kwargs):
        output = original(*args, **kwargs)
        return replace(output, label_id="wrong-label")

    monkeypatch.setattr(api, "_cpu_source_batch", wrong_bundle)
    with pytest.raises(InvalidContractError):
        api.evaluate_factor_source_batch(source, labels, metrics=("rank_ic",), backend="auto")


def test_stub_gpu_output_is_structurally_checked_before_return(monkeypatch):
    source = _Source()
    labels = _labels(source)
    malformed = api._cpu_source_batch(
        source, api.capture_factor_tile_source(source), labels, ("rank_ic",),
        GPUExecutionPolicy(), 16)
    malformed.observation_counts["rank_ic"] = np.asarray([1.0] * len(malformed.factor_ids))
    from quant_evaluator.runtime import gpu_executor

    class StubExecutor:
        def __init__(self, session):
            pass

        def run_source_tiled(self, *args, **kwargs):
            return malformed

    class StubSession:
        def __init__(self, policy):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(gpu_executor, "GPUExecutor", StubExecutor)
    from quant_evaluator.runtime import device_session
    monkeypatch.setattr(device_session, "DeviceEvaluationSession", StubSession)
    with pytest.raises(InvalidContractError):
        api.evaluate_factor_source_batch(
            source, labels, metrics=("rank_ic",), backend="cuda_strict", max_tile_size=16)


def test_series_metric_requires_expected_shape_and_float_dtype(monkeypatch):
    source = _Source()
    labels = _labels(source)
    original = api._cpu_source_batch

    def malformed(*args, **kwargs):
        output = original(*args, **kwargs)
        values = dict(output.series_metrics)
        values["rank_ic_series"] = values["rank_ic_series"][:, :-1]
        return replace(output, series_metrics=values)

    monkeypatch.setattr(api, "_cpu_source_batch", malformed)
    with pytest.raises(InvalidContractError):
        api.evaluate_factor_source_batch(
            source, labels, metrics=("rank_ic_series",), backend="cpu", max_tile_size=16)


def test_malformed_output_evicts_cached_cpu_profile_before_rejection(monkeypatch):
    router.profile_cache.clear_validated_records()
    source = _Source()
    labels = _labels(source)
    policy = GPUExecutionPolicy(max_factor_tile_size=2)
    metadata = api.capture_factor_tile_source(source)
    fingerprint = api._source_request_fingerprint(metadata, labels, ("rank_ic",))
    _install_live_context_capture(monkeypatch, policy)
    context = router.capture_source_route_profile_context(
        source=source, metadata=metadata, metrics=("rank_ic",),
        request_fingerprint=fingerprint, requested_tile_size=16, policy=policy)
    pair = _records(context, cpu_width=5, gpu_width=2)
    actual_cpu = api.evaluate_factor_source_batch(
        source, labels, metrics=("rank_ic",), backend="cpu", max_tile_size=16,
        gpu_policy=policy)
    source.reads.clear()
    pair = rebind_pair_to_cpu_outputs(pair, actual_cpu)
    cache_key = router.source_profile_cache_key(
        source=source, metadata=metadata, metrics=("rank_ic",),
        request_fingerprint=fingerprint, requested_tile_size=16, policy=policy)

    cached = api.evaluate_factor_source_batch(
        source, labels, metrics=("rank_ic",), backend="auto", max_tile_size=16,
        gpu_policy=policy, source_qualification=pair)
    assert cached.metadata["source_qualification_applied"] is True
    assert router.profile_cache.get_validated_records(cache_key) is not None

    original = api._cpu_source_batch
    def reversed_ids(*args, **kwargs):
        output = original(*args, **kwargs)
        return replace(output, factor_ids=output.factor_ids[::-1])

    monkeypatch.setattr(api, "_cpu_source_batch", reversed_ids)
    with pytest.raises(InvalidContractError):
        api.evaluate_factor_source_batch(
            source, labels, metrics=("rank_ic",), backend="auto", max_tile_size=16,
            gpu_policy=policy)
    assert router.profile_cache.get_validated_records(cache_key) is None
