"""Integration tests for ``compute_daily_ic(backend="polars")``.

Pins the third opt-in IC backend (2026-09-25) after the AB benchmark showed
the optimized polars lazy-groupby path beats the exact numpy path at every
measured width while losing to the numba JIT path:

    T=1250 N=300: exact 48ms / polars 29ms / numba 12ms        (F=1)
                  exact 427ms / polars 209ms / numba 44ms      (F=10)
                  exact 2210ms / polars 1496ms / numba 143ms   (F=50)

Contract pinned here:
* finite IC values match the exact path below 1e-12 (measured ~5.6e-17);
* NaN positions are identical across backends (eligibility + constant gates);
* results are byte-deterministic across repeated calls;
* the documented ``valid_counts`` divergence: groups filtered out by the
  ``min_obs`` plan report the exact-path count on the facade boundary (the
  raw polars plan would report 0) - i.e. the caller-visible counts equal the
  exact path wherever a day/factor is eligible, and the raw divergence only
  exists for ineligible rows, which the wrapper re-stamps.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_evaluator.backends.polars_backend import POLARS_AVAILABLE
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.ic import _compute_daily_ic_reference, compute_daily_ic

pytestmark = pytest.mark.skipif(
    not POLARS_AVAILABLE, reason="polars not installed"
)


def _fixtures(T, N, F, seed=31, nan_rate=0.05):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(T, N, F))
    lab = rng.normal(size=(T, N))
    v[rng.random(v.shape) < nan_rate] = np.nan
    lab[rng.random(lab.shape) < nan_rate] = np.nan
    times = AxisRef("t", "int", T, np.arange(T))
    assets = AxisRef("a", "str", N, tuple(f"s{i}" for i in range(N)))
    fb = FactorBatch(tuple(f"f{i}" for i in range(F)), times, assets,
                     np.ascontiguousarray(v))
    lb = LabelBundle(target_id="r", values=np.ascontiguousarray(lab), horizon=1,
                     decision_time=tuple(range(T)), label_start_time=tuple(range(T)),
                     label_end_time=tuple(range(1, T + 1)), asset_axis=assets)
    return fb, lb


@pytest.mark.parametrize("T,N,F", [(250, 80, 1), (250, 80, 4)])
def test_polars_backend_matches_exact(T, N, F):
    fb, lb = _fixtures(T, N, F)
    ic_e, cnt_e = compute_daily_ic(fb, lb, method="spearman", min_assets=20,
                                   backend="exact")
    ic_p, cnt_p = compute_daily_ic(fb, lb, method="spearman", min_assets=20,
                                   backend="polars")
    assert np.array_equal(np.isnan(ic_e), np.isnan(ic_p))
    finite = np.isfinite(ic_e) & np.isfinite(ic_p)
    assert np.allclose(ic_p[finite], ic_e[finite], rtol=0.0, atol=1e-12,
                       equal_nan=False)
    # The facade boundary restores the exact-path counts for ineligible
    # rows; eligible rows carry the polars pairwise count which equals the
    # exact pairwise count by construction of the shared finite filter.
    assert np.array_equal(cnt_p[~np.isfinite(ic_e)], cnt_e[~np.isfinite(ic_e)])


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_polars_backend_matches_reference_oracle(method):
    """Against the verbatim per-day loop oracle (house AB convention)."""
    fb, lb = _fixtures(150, 60, 2, seed=77, nan_rate=0.2)
    ic_r, cnt_r = _compute_daily_ic_reference(fb, lb, method=method, min_assets=20)
    ic_p, cnt_p = compute_daily_ic(fb, lb, method=method, min_assets=20,
                                   backend="polars")
    assert np.array_equal(np.isnan(ic_r), np.isnan(ic_p))
    finite = np.isfinite(ic_r) & np.isfinite(ic_p)
    assert np.allclose(ic_p[finite], ic_r[finite], rtol=0.0, atol=1e-12)
    # counts divergence documented: eligible rows agree.
    eligible = np.isfinite(ic_r)
    assert np.array_equal(cnt_p[eligible], cnt_r[eligible])


def test_polars_backend_deterministic():
    fb, lb = _fixtures(200, 60, 3, seed=99)
    a = compute_daily_ic(fb, lb, method="spearman", min_assets=20, backend="polars")
    b = compute_daily_ic(fb, lb, method="spearman", min_assets=20, backend="polars")
    assert np.array_equal(a[0], b[0], equal_nan=True)
    assert np.array_equal(a[1], b[1])


def test_polars_backend_constant_cross_section_is_nan():
    """Constant factor columns must yield NaN on the polars path too
    (zero-variance correlation), matching the exact gate."""
    rng = np.random.default_rng(5)
    T, N = 120, 50
    v = rng.normal(size=(T, N, 1))
    lab = rng.normal(size=(T, N))
    v[:, :, 0] = 3.14  # constant factor on every day
    times = AxisRef("t", "int", T, np.arange(T))
    assets = AxisRef("a", "str", N, tuple(f"s{i}" for i in range(N)))
    fb = FactorBatch(("prof",), times, assets, np.ascontiguousarray(v))
    lb = LabelBundle(target_id="r", values=np.ascontiguousarray(lab), horizon=1,
                     decision_time=tuple(range(T)), label_start_time=tuple(range(T)),
                     label_end_time=tuple(range(1, T + 1)), asset_axis=assets)
    ic_p, _ = compute_daily_ic(fb, lb, method="spearman", min_assets=20,
                               backend="polars")
    ic_e, _ = compute_daily_ic(fb, lb, method="spearman", min_assets=20,
                               backend="exact")
    assert np.isnan(ic_p).all()
    assert np.array_equal(np.isnan(ic_p), np.isnan(ic_e))


def test_unknown_backend_rejected():
    fb, lb = _fixtures(60, 30, 1)
    with pytest.raises(ValueError, match="Unknown backend"):
        compute_daily_ic(fb, lb, backend="cuda")


def test_polars_float32_tied_panel_keeps_reference_precision():
    """The grouped Pearson moments must not accumulate in float32."""
    rng = np.random.default_rng(260926)
    t, n = 120, 300
    values = rng.standard_normal((t, n, 1), dtype=np.float32)
    labels = rng.standard_normal((t, n), dtype=np.float32)
    values += labels[:, :, None] * np.float32(0.03)
    np.multiply(np.round(values * 20), 0.05, out=values)
    values[rng.random(values.shape) < 0.04] = np.nan
    labels[rng.random(labels.shape) < 0.03] = np.nan
    time_axis = AxisRef("t", "int", t, np.arange(t))
    asset_axis = AxisRef("a", "str", n, tuple(f"s{i}" for i in range(n)))
    batch = FactorBatch(("f0",), time_axis, asset_axis, values)
    bundle = LabelBundle(
        target_id="r", values=labels, horizon=1,
        decision_time=tuple(range(t)), label_start_time=tuple(range(t)),
        label_end_time=tuple(range(1, t + 1)), asset_axis=asset_axis,
    )
    exact, exact_counts = compute_daily_ic(batch, bundle, method="pearson", backend="exact")
    polars, polars_counts = compute_daily_ic(batch, bundle, method="pearson", backend="polars")
    np.testing.assert_array_equal(polars_counts, exact_counts)
    np.testing.assert_allclose(polars, exact, rtol=1e-8, atol=1e-10, equal_nan=True)
