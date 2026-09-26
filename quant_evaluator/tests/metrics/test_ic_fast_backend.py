"""Tests for the opt-in numba fast backend of ``compute_daily_ic`` and for
the (unwired) polars IC backend's parity with the exact numpy path.

Contracts tested here
---------------------
numba backend (``compute_daily_ic(..., backend="numba")``):
  * NaN positions (insufficient data / constant cross-sections) and
    ``valid_counts`` are IDENTICAL to the exact path on every input;
  * finite IC values differ only at the ulp level (sequential JIT
    accumulation vs pairwise-summed means + BLAS gemm); the measured max
    deviation is asserted below 1e-12 and printed for the record —
    evidence content hashes from this path differ from exact;
  * deterministic: two calls are bitwise identical (no cross-thread
    reductions: parallelism is over the time axis with one writer per cell);
  * performance guard: faster than the exact path on the L tier.

exact path (default): the bit-identical contract vs the historical
reference is enforced by ``test_daily_ic_vectorized_equivalence.py``;
here we additionally pin the corrcoef-replication helpers bit-for-bit
against ``np.corrcoef`` directly.

polars backend (``backends/polars_backend.polars_ic_batch`` — NOT wired
into the runtime; see the audit note at the bottom of this file):
  * IC values agree with the exact path within 1e-8 / 1e-10 (house rule),
    NaN positions agree, counts agree wherever polars writes them;
  * documented divergence: rows whose pairwise-finite count is below
    ``min_obs`` keep ``valid_counts == 0`` in the polars output (the
    exact path records the true count for every row) — pinned by test.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from scipy import stats as sstats

from quant_evaluator.backends.polars_backend import polars_ic_batch
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.ic import (
    _compute_daily_ic_reference,
    _corrcoef_pair_1d,
    _corrcoef_rank_last,
    compute_daily_ic,
)

_METHODS = ("pearson", "spearman")


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


def _panel(seed, T=40, N=30, F=2, miss=0.15, ties=False, float32=False,
           validity=False, label1d=False):
    rng = np.random.default_rng(seed)
    vals = rng.normal(size=(T, N, F))
    labels = rng.normal(size=(T, N))
    if miss:
        vals[rng.random(vals.shape) < miss] = np.nan
        labels[rng.random(labels.shape) < miss] = np.nan
    if ties:
        vals = np.round(vals, 1)
        labels = np.round(labels, 1)
    f_valid = rng.random(vals.shape) > 0.2 if validity else None
    l_valid = rng.random(labels.shape) > 0.2 if validity else None
    if float32:
        vals = vals.astype(np.float32)
        labels = labels.astype(np.float32)
    if label1d:
        labels = labels.mean(axis=1)
    return _make_batch(vals, labels, f_valid, l_valid)


# ---------------------------------------------------------------------------
# numba fast backend vs exact
# ---------------------------------------------------------------------------

_PANEL_CASES = [
    dict(seed=201, miss=0.0, ties=False),
    dict(seed=202, miss=0.15, ties=False),
    dict(seed=203, miss=0.3, ties=True),
    dict(seed=204, miss=0.15, ties=False, float32=True),
    dict(seed=205, miss=0.15, validity=True),
    dict(seed=206, miss=0.15, label1d=True),
    dict(seed=207, miss=0.4, ties=True, float32=True),
]


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("case", _PANEL_CASES, ids=lambda c: str(c["seed"]))
def test_numba_matches_exact_nan_positions_counts_and_tolerance(method, case):
    """NaN positions and valid_counts identical; finite values within 1e-12.

    The measured ulp-level deviation is recorded per case (expected
    ~1e-16..1e-15; the numba kernel accumulates sequentially while the
    exact path uses pairwise-summed means + BLAS gemm).
    """
    fb, lb = _panel(**case)
    ex_ic, ex_n = compute_daily_ic(fb, lb, method=method, min_assets=5)
    nb_ic, nb_n = compute_daily_ic(fb, lb, method=method, min_assets=5, backend="numba")

    assert np.array_equal(np.isnan(nb_ic), np.isnan(ex_ic)), "NaN positions differ"
    assert np.array_equal(nb_n, ex_n), "valid_counts differ"

    fin = np.isfinite(ex_ic) & np.isfinite(nb_ic)
    if not fin.any():
        # e.g. the 1-D-label panel: every label cross-section is constant,
        # so the NaN-position equality above IS the full contract here.
        print(f"\n[numba vs exact] case={case} method={method}: all-NaN panel "
              "(NaN positions identical); no finite cells to compare")
        return
    dev = float(np.max(np.abs(ex_ic[fin] - nb_ic[fin])))
    print(f"\n[numba vs exact] case={case} method={method} max|dIC|={dev:.3e}")
    assert dev < 1e-12, f"numba deviates more than documented ulp bound: {dev:.3e}"


def test_numba_extreme_values_and_inf():
    """1e300-scale rows and +-inf entries: NaN positions must match exactly."""
    rng = np.random.default_rng(210)
    vals = rng.normal(size=(30, 25, 1))
    labels = rng.normal(size=(30, 25))
    vals[3, 0, 0] = np.inf
    vals[4, 1, 0] = -np.inf
    labels[5, 2] = np.inf
    vals[6] = 1e300  # constant extreme day -> NaN in both paths
    labels[7] = -1e300
    fb, lb = _make_batch(vals, labels)
    for method in _METHODS:
        ex_ic, ex_n = compute_daily_ic(fb, lb, method=method, min_assets=5)
        nb_ic, nb_n = compute_daily_ic(fb, lb, method=method, min_assets=5, backend="numba")
        assert np.array_equal(np.isnan(nb_ic), np.isnan(ex_ic)), method
        assert np.array_equal(nb_n, ex_n), method


def test_numba_deterministic_bitwise():
    """Two numba calls must be bitwise identical (parallel over t, one
    writer per cell, no cross-thread reductions)."""
    fb, lb = _panel(seed=211, miss=0.2, ties=True)
    for method in _METHODS:
        a, an = compute_daily_ic(fb, lb, method=method, min_assets=5, backend="numba")
        b, bn = compute_daily_ic(fb, lb, method=method, min_assets=5, backend="numba")
        assert np.array_equal(a, b, equal_nan=True)
        assert np.array_equal(an, bn)


def test_numba_unknown_backend_and_method_fail_closed():
    fb, lb = _panel(seed=212)
    with pytest.raises(ValueError, match="backend"):
        compute_daily_ic(fb, lb, method="pearson", backend="fast")
    with pytest.raises(ValueError, match="method"):
        compute_daily_ic(fb, lb, method="kendall", backend="numba")


def test_numba_perf_guard_L_tier():
    """Loose performance guard: on the L tier (T=1250, N=300, F=1) the
    numba spearman path must be faster than the exact path (measured
    ~3-10x; we only require a win with margin)."""
    T, N, F = 1250, 300, 1
    fb, lb = _panel(seed=213, T=T, N=N, F=F, miss=0.15)

    def _med(fn):
        fn()  # warm (JIT compile happens here on cold caches)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return float(np.median(times))

    exact_s = _med(lambda: compute_daily_ic(fb, lb, method="spearman", min_assets=20))
    numba_s = _med(lambda: compute_daily_ic(fb, lb, method="spearman", min_assets=20, backend="numba"))
    print(f"\n[perf] L tier spearman exact={exact_s*1e3:.1f}ms numba={numba_s*1e3:.1f}ms")
    assert numba_s < exact_s


# ---------------------------------------------------------------------------
# exact-path corrcoef replication helpers: bit-for-bit vs np.corrcoef
# ---------------------------------------------------------------------------

def test_corrcoef_pair_replication_bit_exact():
    """_corrcoef_pair_1d must replicate np.corrcoef(x, y)[0, 1] bitwise
    across random, tied, float32 and near-overflow inputs."""
    rng = np.random.default_rng(214)
    checked = 0
    for trial in range(500):
        n = int(rng.integers(2, 300))
        kind = trial % 4
        if kind == 0:
            x, y = rng.normal(size=n), rng.normal(size=n)
        elif kind == 1:
            x, y = np.round(rng.normal(size=n), 1), np.round(rng.normal(size=n), 1)
        elif kind == 2:
            x, y = (rng.normal(size=n)).astype(np.float32), (rng.normal(size=n)).astype(np.float32)
        else:
            x, y = rng.normal(size=n) * 1e150, rng.normal(size=n) * 1e150
        ref = np.corrcoef(x, y)[0, 1]
        got = _corrcoef_pair_1d(x, y, n)
        if np.isnan(ref):
            assert np.isnan(got)
        else:
            assert got == ref, f"trial={trial} ref={ref!r} got={got!r}"
            checked += 1
    assert checked > 400


def test_corrcoef_rank_replication_bit_exact():
    """_corrcoef_rank_last must replicate
    np.corrcoef(column_stack((rx, ry)), rowvar=False)[1, 0] bitwise for
    genuine average-tie rank vectors (its only input domain)."""
    rng = np.random.default_rng(215)
    checked = 0
    for trial in range(500):
        n = int(rng.integers(2, 300))
        rx = sstats.rankdata(rng.normal(size=n), method="average")
        ry = sstats.rankdata(rng.normal(size=n), method="average")
        ref = np.corrcoef(np.column_stack((rx, ry)), rowvar=False)[1, 0]
        got = _corrcoef_rank_last(rx, ry, n)
        assert got == ref, f"trial={trial} ref={ref!r} got={got!r}"
        checked += 1
    assert checked == 500


# ---------------------------------------------------------------------------
# polars backend parity (unwired kernel; house tolerances 1e-8 / 1e-10)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", _METHODS)
def test_polars_ic_batch_parity_with_exact(method):
    """polars_ic_batch agrees with compute_daily_ic within 1e-8/1e-10 on
    finite values; NaN positions agree; counts agree wherever polars
    writes them.  Documented divergence: rows with fewer than min_obs
    pairwise-finite pairs keep valid_counts == 0 in the polars output."""
    rng = np.random.default_rng(216)
    T, N, F = 25, 30, 2
    vals = rng.normal(size=(T, N, F))
    labels = rng.normal(size=(T, N))
    vals[rng.random(vals.shape) < 0.25] = np.nan  # some rows above and below min_obs
    labels[rng.random(labels.shape) < 0.25] = np.nan
    fb, lb = _make_batch(vals, labels)

    ex_ic, ex_n = compute_daily_ic(fb, lb, method=method, min_assets=20)
    pl_ic, pl_n = polars_ic_batch(fb.values, lb.values, fb.factor_ids,
                                  method=method, min_obs=20)

    # rows below min_obs: polars zeroes the count (documented divergence)
    below = (ex_n > 0) & (ex_n < 20)
    if below.any():
        assert np.all(pl_n[below] == 0), "polars must not report sub-min_obs counts"

    # counts agree wherever polars wrote them
    assert np.array_equal(ex_n[pl_n > 0], pl_n[pl_n > 0])

    # NaN positions for eligible rows agree
    eligible = ex_n >= 20
    assert np.array_equal(np.isnan(pl_ic[eligible]), np.isnan(ex_ic[eligible]))

    fin = np.isfinite(ex_ic) & np.isfinite(pl_ic)
    assert fin.any()
    np.testing.assert_allclose(ex_ic[fin], pl_ic[fin], rtol=1e-8, atol=1e-10)


def test_polars_ic_batch_constant_and_inf_semantics():
    """Constant cross-sections -> NaN in both; +-inf excluded pairwise in
    both; perfectly monotone cross-sections -> +-1 in both."""
    # constant label row
    vals = np.array([[[1.0], [2.0], [3.0], [4.0]]])
    labels = np.array([[5.0, 5.0, 5.0, 5.0]])
    fb, lb = _make_batch(vals, labels)
    for method in _METHODS:
        ex_ic, _ = compute_daily_ic(fb, lb, method=method, min_assets=2)
        pl_ic, _ = polars_ic_batch(fb.values, lb.values, fb.factor_ids,
                                   method=method, min_obs=2)
        assert np.isnan(ex_ic[0, 0]) and np.isnan(pl_ic[0, 0]), method

    # inf excluded pairwise; monotone day -> exactly +-1
    vals2 = np.array([[[1.0], [2.0], [np.inf], [4.0]],
                      [[1.0], [2.0], [3.0], [4.0]]])
    labels2 = np.array([[10.0, 20.0, 30.0, 40.0],
                        [40.0, 30.0, 20.0, 10.0]])
    fb2, lb2 = _make_batch(vals2, labels2)
    ex_ic, _ = compute_daily_ic(fb2, lb2, method="pearson", min_assets=3)
    pl_ic, _ = polars_ic_batch(fb2.values, lb2.values, fb2.factor_ids,
                               method="pearson", min_obs=3)
    assert np.array_equal(np.isnan(ex_ic), np.isnan(pl_ic))
    np.testing.assert_allclose(ex_ic[np.isfinite(ex_ic)],
                               pl_ic[np.isfinite(pl_ic)], rtol=1e-8, atol=1e-10)
    # [1, 2, 4] vs [10, 20, 40] is perfectly proportional, but the float64
    # centered-dot arithmetic lands one ulp below 1.0 in both backends.
    assert ex_ic[0, 0] == pytest.approx(1.0, abs=1e-12)
    assert ex_ic[1, 0] == pytest.approx(-1.0, abs=1e-12)
    assert pl_ic[0, 0] == pytest.approx(1.0, abs=1e-12)
    assert pl_ic[1, 0] == pytest.approx(-1.0, abs=1e-12)
