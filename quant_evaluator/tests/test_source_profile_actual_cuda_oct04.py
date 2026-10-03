"""Opt-in real CUDA route glue; live qualification is tested separately.

Synthetic data and stubbed qualification do not establish COS throughput or
a fastest backend. Stubbed receipts have no actual-output identity and must
not retain qualification. The source reader, GPU executor and post-execution
schedule checker remain real; SciPy supplies the numerical reference.
"""
import os

import numpy as np
import pytest
from scipy.stats import spearmanr

from quant_evaluator.api import factor_source as api
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.contracts.factor_batch import AxisRef
from quant_evaluator.runtime import source_profile_router as router
from test_source_route_profiles_api_oct03 import _Source, _decision, _labels


@pytest.mark.skipif(os.environ.get("QE_RUN_SOURCE_PROFILE_CUDA") != "1",
                    reason="opt in to actual source-profile CUDA checks")
@pytest.mark.parametrize("asset_count", [12, 48])
def test_actual_cuda_profile_width_and_independent_rank_reference(monkeypatch, asset_count):
    cp = pytest.importorskip("cupy")
    try:
        devices = cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("CUDA runtime unavailable")
    if devices < 1:
        pytest.skip("CUDA device unavailable")
    source = _Source()
    source.asset_axis = AxisRef("asset", "int64", asset_count,
                               np.arange(asset_count, dtype=np.int64))
    source.values = np.random.default_rng(61004).normal(size=(8, asset_count, 5))
    labels, decision = _labels(source), _decision("cuda")
    monkeypatch.setattr(api, "is_source_route_profile_pair", lambda records: True)
    monkeypatch.setattr(api, "qualify_source_route_profiles", lambda **kwargs: decision)
    monkeypatch.setattr(api, "cuda_route_rejection", lambda policy: None)
    monkeypatch.setattr(router, "capture_source_route_profile_context",
                        lambda **kwargs: decision.context)
    result = api.evaluate_factor_source_batch(
        source, labels, metrics=("rank_ic",), backend="auto", max_tile_size=16,
        gpu_policy=GPUExecutionPolicy(max_factor_tile_size=2),
        source_qualification=(object(), object()))
    if asset_count < 20:  # The public default min_assets gate applies first.
        expected, count = np.full(5, np.nan), 0
    else:
        expected = np.array([
            [spearmanr(source.values[t, :, f], labels.values[t]).statistic
             for f in range(5)] for t in range(8)]).mean(axis=0)
        count = 8
    np.testing.assert_allclose(result.scalar_metrics["rank_ic"], expected,
                               rtol=1e-12, atol=1e-14, equal_nan=True)
    np.testing.assert_array_equal(result.observation_counts["rank_ic"],
                                  np.full(5, count, dtype=np.int64))
    assert source.reads == [(0, 2), (2, 4), (4, 5)]
    assert result.metadata["backend_used"] == "cuda"
    assert result.metadata["factor_tile_size"] == 2
    assert result.metadata["oom_retries"] == 0
    assert result.metadata["source_qualification_applied"] is False
    assert result.metadata["source_qualification_reason"] == (
        "qualified_profile_output_identity_deviated")
