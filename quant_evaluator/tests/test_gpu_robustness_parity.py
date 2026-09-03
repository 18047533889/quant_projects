"""GPU Temporal / Robustness parity tests (spec §20 Wave 4, §54 #26-30).

Each GPU kernel must match its CPU reference oracle to rtol<=1e-8 / atol<=1e-10
where the oracle is deterministic.  The bootstrap / subsample kernels use a
fixed seed, so all five tests are deterministic.  All tests skip when CuPy is
not importable (CPU-only environment).
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_evaluator.kernels.gpu.robustness import (
    hac_variance,
    hac_tstat,
    block_bootstrap_ci,
    subsample_ic,
    subsample_ic_std,
    rolling_ic_stats,
)
from quant_evaluator.kernels.gpu.temporal import (
    ic_autocorrelation_series,
    half_life,
)

# CPU oracles (read-only imports)
from quant_evaluator.metrics.robustness import (
    compute_hac_variance,
    compute_hac_tstat,
    compute_block_bootstrap_ci,
    compute_subsample_ic,
    compute_subsample_ic_std,
)
from quant_evaluator.metrics.temporal import (
    compute_ic_autocorrelation,
    compute_half_life,
)
from quant_evaluator.metrics.ic_summary import compute_rolling_ic_stats

cp = pytest.importorskip("cupy")

RTOL = 1e-8
ATOL = 1e-10


def _ic_series():
    rng = np.random.default_rng(7)
    T, F = 240, 5
    ic = rng.normal(size=(T, F)) * 0.4
    # sprinkle NaNs that leave calendar gaps
    ic[10, 1] = np.nan
    ic[40:44, 2] = np.nan
    ic[3, 4] = np.nan
    return ic


def test_gpu_hac_parity():
    ic = _ic_series()
    for kernel in ("bartlett", "uniform"):
        cpu = compute_hac_variance(ic, max_lag=5, kernel=kernel)
        gpu = hac_variance(ic, max_lag=5, kernel=kernel)
        np.testing.assert_allclose(gpu, cpu, rtol=RTOL, atol=ATOL, equal_nan=True)

    ct, cs = compute_hac_tstat(ic, max_lag=5, kernel="bartlett")
    gt, gs = hac_tstat(ic, max_lag=5, kernel="bartlett")
    np.testing.assert_allclose(gt, ct, rtol=RTOL, atol=ATOL, equal_nan=True)
    np.testing.assert_allclose(gs, cs, rtol=RTOL, atol=ATOL, equal_nan=True)


def test_gpu_ic_autocorr_parity():
    ic = _ic_series()
    cpu = compute_ic_autocorrelation(ic, max_lag=20, min_obs=30)
    gpu = ic_autocorrelation_series(ic, max_lag=20, min_obs=30)
    assert cpu.shape == gpu.shape
    np.testing.assert_allclose(gpu, cpu, rtol=RTOL, atol=ATOL, equal_nan=True)


def test_gpu_half_life_parity():
    ic = _ic_series()
    cpu = compute_half_life(ic, min_periods=60)
    gpu = half_life(ic, min_periods=60)
    assert cpu.shape == gpu.shape
    np.testing.assert_allclose(gpu, cpu, rtol=RTOL, atol=ATOL, equal_nan=True)


def test_gpu_bootstrap_determinism():
    ic = _ic_series()
    seed = 123
    cl, cu = compute_block_bootstrap_ci(
        ic, block_length=10, num_bootstrap=500, confidence_level=0.95,
        random_seed=seed,
    )
    gl, gu = block_bootstrap_ci(
        ic, block_length=10, num_bootstrap=500, confidence_level=0.95,
        random_seed=seed,
    )
    np.testing.assert_allclose(gl, cl, rtol=RTOL, atol=ATOL, equal_nan=True)
    np.testing.assert_allclose(gu, cu, rtol=RTOL, atol=ATOL, equal_nan=True)

    # deterministic across repeated calls with the same seed
    gl2, gu2 = block_bootstrap_ci(
        ic, block_length=10, num_bootstrap=500, confidence_level=0.95,
        random_seed=seed,
    )
    np.testing.assert_array_equal(gl2, gl)
    np.testing.assert_array_equal(gu2, gu)

    # different seed -> different draw (CI may differ)
    gl3, _ = block_bootstrap_ci(
        ic, block_length=10, num_bootstrap=500, confidence_level=0.95,
        random_seed=seed + 1,
    )
    assert not np.allclose(gl3, gl, rtol=0, atol=0)


def test_gpu_subsample_parity():
    ic = _ic_series()
    seed = 42
    cm, csd = compute_subsample_ic(
        ic, num_subsamples=200, subsample_fraction=0.8, random_seed=seed
    )
    gm, gsd = subsample_ic(
        ic, num_subsamples=200, subsample_fraction=0.8, random_seed=seed
    )
    assert cm.shape == gm.shape
    np.testing.assert_allclose(gm, cm, rtol=RTOL, atol=ATOL, equal_nan=True)
    np.testing.assert_allclose(gsd, csd, rtol=RTOL, atol=ATOL, equal_nan=True)

    cstd = compute_subsample_ic_std(
        ic, num_subsamples=200, subsample_fraction=0.8, random_seed=seed
    )
    gstd = subsample_ic_std(
        ic, num_subsamples=200, subsample_fraction=0.8, random_seed=seed
    )
    np.testing.assert_allclose(gstd, cstd, rtol=RTOL, atol=ATOL, equal_nan=True)


def test_gpu_rolling_parity():
    ic = _ic_series()
    cpu = compute_rolling_ic_stats(ic, window=60, min_periods=20)
    gmean, gir = rolling_ic_stats(ic, window=60, min_periods=20)
    assert gmean.shape == cpu["rolling_ic_mean"].shape
    np.testing.assert_allclose(
        gmean, cpu["rolling_ic_mean"], rtol=RTOL, atol=ATOL, equal_nan=True
    )
    np.testing.assert_allclose(
        gir, cpu["rolling_ic_ir"], rtol=RTOL, atol=ATOL, equal_nan=True
    )
