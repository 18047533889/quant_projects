"""Opt-in bounded CPU/CUDA report adapter parity, never a production workload."""
import os
import numpy as np
import pytest
from factor_engine.reporting.quant_evaluator_adapter import evaluate_report_batch

pytestmark = pytest.mark.skipif(
    os.environ.get("FE_RUN_GPU_SMOKE") != "1",
    reason="requires explicit opt-in and an available CUDA device",
)


def test_small_real_cuda_report_matches_cpu():
    rng = np.random.default_rng(3018)
    factors = rng.normal(size=(12, 40, 3))
    returns = rng.normal(scale=0.01, size=(12, 40))
    factors[3, 2, 1] = np.nan
    returns[6, 5] = np.nan
    kwargs = dict(factor_ids=("a", "b", "c"), n_quantiles=10,
                  min_assets=20, min_ic_periods=3, fixed_directions=(1, -1, 1))
    cpu = evaluate_report_batch(factors, returns, backend="cpu", **kwargs)
    gpu = evaluate_report_batch(factors, returns, backend="cuda_strict", **kwargs)
    assert gpu.backend_used == "cuda" and gpu.backend_fallback_reason is None
    for name in kwargs["factor_ids"]:
        for field in ("rank_ic_series", "quantile_returns", "long_short_returns"):
            np.testing.assert_allclose(getattr(gpu.factors[name], field),
                                       getattr(cpu.factors[name], field),
                                       equal_nan=True, rtol=1e-9, atol=1e-11)
