"""Tests for the opt-in CUDA backend of ``compute_daily_ic(backend="gpu")``.

Contracts tested here
---------------------
gpu backend (``compute_daily_ic(..., backend="gpu")``, wired 2026-09-25):
  * IC values agree with the exact path within the house rule
    rtol<=1e-8 / atol<=1e-10 across shapes, seeds, dtypes, missingness,
    ties, validity masks and degenerate panels (constant rows, all-NaN
    columns, sub-min_assets cross-sections);
  * NaN positions (eligibility / constant-rejection gates) and
    ``valid_counts`` are IDENTICAL to the exact path — pinned by
    construction host-side, asserted here on every fixture;
  * deterministic: two calls are bitwise identical (no cross-stream
    reductions on the kernel path);
  * performance guard: faster than the exact path on a mid-size panel
    (loose 2x margin to stay CI-stable);
  * cross-over documentation pinned as data (see
    ``quant_evaluator.metrics.ic`` docstring): GPU wins pearson at every
    measured shape and spearman while F*N is small; numba wins wide-factor
    spearman.  The guard asserts the GPU win on pearson only.

Skips cleanly on CPU-only machines via ``pytest.importorskip("cupy")``.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.ic import compute_daily_ic

cp = pytest.importorskip("cupy")

METHODS = ("pearson", "spearman")
HOUSE = {"rtol": 1e-8, "atol": 1e-10}


def _make_batch(values, labels, factor_validity=None, label_validity=None):
    T, N, F = values.shape
    fb = FactorBatch(
        factor_ids=tuple(f"f{i}" for i in range(F)),
        time_axis=AxisRef("t", "int", T),
        asset_axis=AxisRef("a", "str", N),
        values=values,
        validity=factor_validity,
    )
    lb = LabelBundle(
        target_id="r",
        values=labels,
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
        validity=label_validity,
    )
    return fb, lb


def _panel(seed, T=40, N=60, F=3, miss=0.15, ties=False, float32=False,
           validity=False, label1d=False):
    rng = np.random.default_rng(seed)
    if ties:
        vals = rng.integers(0, 4, size=(T, N, F)).astype(np.float64)
        labels = rng.integers(0, 3, size=(T, N)).astype(np.float64)
    else:
        vals = rng.normal(size=(T, N, F))
        labels = rng.normal(size=(T, N))
    if miss:
        vals[rng.random(vals.shape) < miss] = np.nan
        labels[rng.random(labels.shape) < miss] = np.nan
    fval = lval = None
    if validity:
        fval = rng.random(vals.shape) > miss
        lval = rng.random(labels.shape) > miss
        vals = np.where(fval, vals, np.nan)
        labels = np.where(lval, labels, np.nan)
    if label1d:
        labels = labels[:, 0]
    if float32:
        vals = vals.astype(np.float32)
        labels = labels.astype(np.float32)
    return vals, labels, fval, lval


def _assert_gpu_matches_exact(vals, labels, fval=None, lval=None,
                              min_assets=10):
    fb, lb = _make_batch(vals, labels, fval, lval)
    for method in METHODS:
        ic_ex, cnt_ex = compute_daily_ic(fb, lb, method=method,
                                         backend="exact",
                                         min_assets=min_assets)
        ic_g, cnt_g = compute_daily_ic(fb, lb, method=method,
                                       backend="gpu",
                                       min_assets=min_assets)
        ic_ex = np.asarray(ic_ex, dtype=np.float64)
        ic_g = np.asarray(ic_g, dtype=np.float64)
        fin_ex, fin_g = np.isfinite(ic_ex), np.isfinite(ic_g)
        # NaN positions (insufficient data / constant cross-sections) must
        # be identical — the core backend-dispatch contract.
        assert np.array_equal(fin_ex, fin_g), (
            f"NaN positions diverge ({method}): "
            f"{int((fin_ex ^ fin_g).sum())} cells"
        )
        assert np.array_equal(cnt_ex, cnt_g), f"valid_counts diverge ({method})"
        if fin_ex.any():
            np.testing.assert_allclose(ic_ex[fin_ex], ic_g[fin_g],
                                       **HOUSE)


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("shape", [(40, 60, 3), (25, 30, 2), (60, 120, 4)])
def test_gpu_matches_exact_random_panels(seed, shape):
    vals, labels, _, _ = _panel(seed, T=shape[0], N=shape[1], F=shape[2])
    _assert_gpu_matches_exact(vals, labels)


def test_gpu_matches_exact_float32_input():
    vals, labels, _, _ = _panel(3, float32=True)
    _assert_gpu_matches_exact(vals, labels)


def test_gpu_matches_exact_ties():
    vals, labels, _, _ = _panel(4, ties=True)
    _assert_gpu_matches_exact(vals, labels)


def test_gpu_matches_exact_validity_masks():
    vals, labels, fval, lval = _panel(5, validity=True)
    _assert_gpu_matches_exact(vals, labels, fval, lval)


def test_gpu_matches_exact_1d_labels():
    vals, labels, _, _ = _panel(6, label1d=True)
    _assert_gpu_matches_exact(vals, labels)


def test_gpu_matches_exact_constant_rows():
    rng = np.random.default_rng(8)
    vals = rng.normal(size=(30, 40, 3))
    labels = rng.normal(size=(30, 40))
    vals[rng.random(vals.shape) < 0.1] = np.nan
    labels[rng.random(labels.shape) < 0.1] = np.nan
    vals[3, :, 1] = 2.5       # constant factor cross-section
    vals[7, :, 2] = -1.0      # another one
    labels[2] = 1.0           # constant label day
    labels[5] = 0.0
    _assert_gpu_matches_exact(vals, labels)


def test_gpu_matches_exact_degenerate_panels():
    rng = np.random.default_rng(9)
    vals = rng.normal(size=(20, 30, 3))
    labels = rng.normal(size=(20, 30))
    vals[rng.random(vals.shape) < 0.1] = np.nan
    labels[rng.random(labels.shape) < 0.1] = np.nan
    vals[:, :, 2] = np.nan                    # all-NaN column
    # sub-min_assets cross-sections (min_assets=20, N=15): every day NaN
    small_vals = rng.normal(size=(10, 15, 2))
    small_labels = rng.normal(size=(10, 15))
    fb, lb = _make_batch(small_vals, small_labels)
    for method in METHODS:
        ic_e, _ = compute_daily_ic(fb, lb, method=method, backend="exact",
                                   min_assets=20)
        ic_g, _ = compute_daily_ic(fb, lb, method=method, backend="gpu",
                                   min_assets=20)
        assert np.all(np.isnan(ic_e)) and np.all(np.isnan(ic_g))
    _assert_gpu_matches_exact(vals, labels)


def test_gpu_deterministic_bitwise():
    vals, labels, _, _ = _panel(11, T=30, N=50, F=4)
    fb, lb = _make_batch(vals, labels)
    for method in METHODS:
        a = compute_daily_ic(fb, lb, method=method, backend="gpu",
                             min_assets=10)
        b = compute_daily_ic(fb, lb, method=method, backend="gpu",
                             min_assets=10)
        assert np.array_equal(a[0], b[0], equal_nan=True)
        assert np.array_equal(a[1], b[1])


def test_gpu_unknown_backend_message():
    vals, labels, _, _ = _panel(12, T=10, N=10, F=1)
    fb, lb = _make_batch(vals, labels)
    with pytest.raises(ValueError, match="'exact', 'numba', 'polars' or 'gpu'"):
        compute_daily_ic(fb, lb, backend="cuda")


def test_gpu_performance_guard_pearson():
    """Loose CI-stable guard: the gpu path beats exact on a mid-size panel.

    Full crossover table lives in the ``compute_daily_ic`` docstring
    (2026-09-25 L20 AB benchmark); the GPU win is largest on pearson, so
    the guard pins that method only and keeps a generous 2x margin.
    """
    vals, labels, _, _ = _panel(13, T=250, N=120, F=6, miss=0.1)
    fb, lb = _make_batch(vals, labels)
    compute_daily_ic(fb, lb, method="pearson", backend="gpu",
                     min_assets=10)  # warmup: CUDA context + kernel JIT
    t0 = time.perf_counter()
    compute_daily_ic(fb, lb, method="pearson", backend="gpu", min_assets=10)
    t_gpu = time.perf_counter() - t0
    t0 = time.perf_counter()
    compute_daily_ic(fb, lb, method="pearson", backend="exact", min_assets=10)
    t_exact = time.perf_counter() - t0
    assert t_gpu < t_exact / 2.0, (
        f"gpu pearson regressed: {t_gpu:.3f}s vs exact {t_exact:.3f}s"
    )
