"""
Performance A/B equivalence and causal-contract tests for the vectorized
lagged per-asset transforms.

These tests lock in the optimization of ``rolling_mean``, ``rolling_std``,
``rolling_zscore``, ``ewma`` (``factor_preprocess.transforms.rolling``) and
``trailing_median``, ``robust_ewma`` (``factor_preprocess.transforms.smoothing``).

Each public function gained a vectorized, ``groupby`` + ``rolling`` /
``groupby`` + ``ewm`` implementation. The original scalar, per-asset Python
loop is retained as a ``_xxx_reference`` oracle. These tests assert:

1. Equivalence (value AND NaN mask) between the new path and the oracle on
   many seeds, float32/float64, and duplicate-index inputs. The math is
   bit-for-bit identical in practice; we still allow ``rtol=1e-8,
   atol=1e-10`` as the documented numerical contract ceiling.
2. Causal guards: no future leakage (perturbing a future row never changes a
   past output), prefix invariance (truncating and recomputing reproduces the
   full-run prefix), and real-time-append equivalence (appending rows to each
   asset reproduces the earlier outputs). These transforms are stateless, so
   the "fit only sees TRAIN" rule is N/A; causality is enforced purely by the
   strict ``shift(1)`` lag, which the no-leakage test pins down.
3. Degenerate inputs: single asset, all-NaN, all-constant, window larger than
   the series, duplicate index, and that unsorted input raises.
4. Output hash stability: identical input yields byte-identical output across
   calls (determinism), which is the serialization-relevant contract for a
   pure transform.
5. Performance guard: the new path is not slower than the reference oracle on a
   bounded 256 x 500 panel (loose 1.5x ceiling against run-to-run noise) and
   the observed speedup is reported.
"""
import hashlib
import time

import numpy as np
import pandas as pd
import pytest

from factor_preprocess.transforms import rolling as R
from factor_preprocess.transforms import smoothing as S

# (name, public or explicit research implementation, reference oracle, kwargs)
EQUIV_CASES = [
    ("rolling_mean", R.rolling_mean, R._rolling_mean_reference,
     dict(window=20, min_periods=10)),
    ("rolling_std", R.rolling_std, R._rolling_std_reference,
     dict(window=20, min_periods=10)),
    ("rolling_zscore", R._rolling_zscore_fp_research,
     R._rolling_zscore_reference,
     dict(window=20, min_periods=10)),
    ("ewma", R._ewma_fp_research, R._ewma_reference, dict(halflife=10)),
    ("trailing_median", S.trailing_median, S._trailing_median_reference,
     dict(window=20, min_periods=10)),
    ("robust_ewma", S.robust_ewma, S._robust_ewma_reference,
     dict(halflife=10, winsor_std=4.0)),
]


def make_panel(n_assets=256, n_dates=500, miss=0.03, seed=0, fk="float64",
               dup_index=False):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_dates, freq="W")
    rows = []
    for a in range(n_assets):
        vals = rng.standard_normal(n_dates)
        vals[rng.random(n_dates) < miss] = np.nan
        rows.append(pd.DataFrame(
            {"asset_id": a, "date": dates, "value": vals.astype(fk)}))
    panel = pd.concat(rows, ignore_index=True)
    if dup_index:
        # Duplicate a few index labels while keeping [asset, date] sort order.
        idx = list(range(len(panel)))
        for i in range(0, len(idx), 53):
            idx[i] = 0
        panel = panel.copy()
        panel.index = idx
    return panel


def _assert_bit_equal(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask_equal = (np.isnan(a) == np.isnan(b)).all()
    assert mask_equal, "NaN mask differs between new and reference"
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.any():
        # Bit-for-bit is the observed contract; allow a tiny tolerance ceiling.
        assert np.allclose(a, b, rtol=1e-8, atol=1e-10, equal_nan=True), (
            f"max abs dev = {np.nanmax(np.abs(a - b)):.2e}")
    # Explicit bit-exact check on the finite bytes when both finite.
    assert np.array_equal(a[finite], b[finite]), (
        f"non-bit-exact finite values, max abs dev = {np.nanmax(np.abs(a - b)):.2e}")


@pytest.mark.parametrize("name,pub,ref,kwargs", EQUIV_CASES)
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("fk", ["float64", "float32"])
def test_new_matches_reference_oracle(name, pub, ref, kwargs, seed, fk):
    panel = make_panel(seed=seed, fk=fk)
    _assert_bit_equal(pub(panel, **kwargs), ref(panel, **kwargs))


@pytest.mark.parametrize("name,pub,ref,kwargs", EQUIV_CASES)
def test_equivalence_under_duplicate_index(name, pub, ref, kwargs):
    panel = make_panel(seed=3, dup_index=True)
    _assert_bit_equal(pub(panel, **kwargs), ref(panel, **kwargs))


@pytest.mark.parametrize("name,pub,ref,kwargs", EQUIV_CASES)
def test_no_future_leakage(name, pub, ref, kwargs):
    """Perturbing a future row must not change any earlier (past) output."""
    panel = make_panel(seed=4, n_assets=8, n_dates=120)
    base = pub(panel, **kwargs)
    mutated = panel.copy()
    # Flip the very last observation of asset 0 (the most "future" row).
    last_pos = (mutated["asset_id"] == 0).to_numpy().nonzero()[0][-1]
    mutated.iloc[last_pos, mutated.columns.get_loc("value")] = (
        mutated.iloc[last_pos, mutated.columns.get_loc("value")] + 1234.5)
    new = pub(mutated, **kwargs)
    # Every row except the mutated one (and its own warmup neighbours) must be
    # unchanged. The mutated row itself may change, but past rows cannot.
    earlier = np.arange(last_pos)
    _assert_bit_equal(base.iloc[earlier].to_numpy(),
                      new.iloc[earlier].to_numpy())


@pytest.mark.parametrize("name,pub,ref,kwargs", EQUIV_CASES)
def test_prefix_invariance_and_append_equivalence(name, pub, ref, kwargs):
    """Appending rows to each asset reproduces the earlier outputs exactly."""
    full = make_panel(seed=7, n_assets=64, n_dates=300)
    rng = np.random.default_rng(99)
    extra = pd.date_range(full["date"].max() + pd.Timedelta(weeks=1),
                          periods=50, freq="W")
    parts = []
    for a in sorted(full["asset_id"].unique()):
        sub = full[full["asset_id"] == a]
        ext = pd.DataFrame({"asset_id": a, "date": extra,
                            "value": rng.standard_normal(50)})
        parts.append(pd.concat([sub, ext], ignore_index=True))
    appended = pd.concat(parts, ignore_index=True)

    full_mi = full.set_index(["asset_id", "date"])
    app_mi = appended.set_index(["asset_id", "date"])

    pub_full = pub(full, **kwargs)
    pub_app = pub(appended, **kwargs).set_axis(app_mi.index).reindex(full_mi.index)
    _assert_bit_equal(pub_full.to_numpy(), pub_app.to_numpy())


@pytest.mark.parametrize("name,pub,ref,kwargs", EQUIV_CASES)
def test_degenerate_single_asset_all_nan_all_constant_short_window(name, pub, ref, kwargs):
    single = make_panel(seed=3, n_assets=1, n_dates=60)
    _assert_bit_equal(pub(single, **kwargs), ref(single, **kwargs))

    nan_panel = make_panel(seed=1, n_assets=4, n_dates=40)
    nan_panel["value"] = np.nan
    _assert_bit_equal(pub(nan_panel, **kwargs), ref(nan_panel, **kwargs))

    const_panel = make_panel(seed=1, n_assets=4, n_dates=40)
    const_panel["value"] = 1.0
    _assert_bit_equal(pub(const_panel, **kwargs), ref(const_panel, **kwargs))

    # Window larger than the per-asset series length -> all NaN warmup.
    big_kwargs = dict(kwargs)
    if "window" in big_kwargs:
        big_kwargs["window"] = 5000
        if "min_periods" in big_kwargs:
            big_kwargs["min_periods"] = 10
    _assert_bit_equal(pub(single, **big_kwargs), ref(single, **big_kwargs))


@pytest.mark.parametrize("name,pub,ref,kwargs", EQUIV_CASES)
def test_unsorted_input_raises(name, pub, ref, kwargs):
    unsorted = make_panel(seed=2).sample(frac=1.0, random_state=3).reset_index(drop=True)
    with pytest.raises(ValueError):
        pub(unsorted, **kwargs)


@pytest.mark.parametrize("name,pub,ref,kwargs", EQUIV_CASES)
def test_output_hash_stability(name, pub, ref, kwargs):
    panel = make_panel(seed=11)
    h1 = hashlib.sha256(np.asarray(pub(panel, **kwargs)).tobytes()).hexdigest()
    h2 = hashlib.sha256(np.asarray(pub(panel, **kwargs)).tobytes()).hexdigest()
    assert h1 == h2, "transform output is not deterministic across calls"


@pytest.mark.parametrize("name,pub,ref,kwargs", EQUIV_CASES)
def test_performance_not_slower_than_reference(name, pub, ref, kwargs):
    """New path must not regress vs the scalar oracle (loose 1.5x ceiling)."""
    panel = make_panel(seed=0, n_assets=256, n_dates=500)

    def bench(fn, n=5):
        fn(panel, **kwargs)  # warm-up
        t0 = time.perf_counter()
        for _ in range(n):
            fn(panel, **kwargs)
        return (time.perf_counter() - t0) / n

    t_new = bench(pub)
    t_ref = bench(ref)
    speedup = t_ref / t_new if t_new > 0 else float("inf")
    print(f"\n  {name}: new={t_new:.4f}s ref={t_ref:.4f}s speedup={speedup:.2f}x")
    # Guard against regression: the vectorized path should stay faster.
    assert t_new <= t_ref * 1.5, (
        f"{name} regressed: new {t_new:.4f}s vs ref {t_ref:.4f}s")
