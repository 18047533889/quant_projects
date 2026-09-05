# -*- coding: utf-8 -*-
"""R61-P1 #57: shared sufficient-statistics vectorization tests.

Covers:
(a) the ``sufficient_stats`` accumulator correctness on a known small input
    (hand-computed sums / sumsq / max / min / first / last / argmax / argmin /
    return moments);
(b) O(1)-derivation parity for every newly vectorized sufficient-stats
    operator vs its scalar reference (rtol/atol 1e-12), incl. NaN gaps and an
    all-NaN day — plus a scan-count / source-read proof that the vec path
    consumes the SHARED BUNDLE (the bundle is built once for a frame and the
    vec kernel derives from it; a second operator on the same frame reuses the
    identical bundle object — no per-operator grid re-scan);
(c) cache reuse: two operators on the same frame share the same bundle object;
(d) fail-closed bind: count_bound() == len(bind_whitelist()) and every
    sufficient-stats kernel has a live ``__vec__``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.intraday import _core
from factor_engine.cleaned_operators.intraday import sufficient_stats as ss
from factor_engine.cleaned_operators.intraday import sufficient_stats_ops as sso
from factor_engine.cleaned_operators.intraday import perf_vec_kernels as pvk


# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------

def _session_minutes(days: int = 3, start: str = "2026-01-02") -> pd.DatetimeIndex:
    out = []
    base = pd.Timestamp(start)
    for d in range(days):
        day = base + pd.Timedelta(days=d)
        while day.dayofweek >= 5:
            day = day + pd.Timedelta(days=1)
        out += list(pd.date_range(day.replace(hour=9, minute=31), day.replace(hour=11, minute=30), freq="1min"))
        out += list(pd.date_range(day.replace(hour=13, minute=1), day.replace(hour=15, minute=0), freq="1min"))
    return pd.DatetimeIndex(out)


def _close_panel(seed: int = 0, days: int = 3, cols=None):
    cols = cols or ["C0", "C1"]
    idx = _session_minutes(days)
    rng = np.random.default_rng(seed)
    data = np.exp(np.cumsum(rng.standard_normal((len(idx), len(cols))) * 0.001, axis=0)) * 100.0
    return pd.DataFrame(data, index=idx, columns=cols)


def _volume_panel(df, seed: int = 1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.exponential(scale=1e6, size=df.shape), index=df.index, columns=df.columns)


def _amount_panel(df, seed: int = 2):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.exponential(scale=1e8, size=df.shape), index=df.index, columns=df.columns)


def _inject_nan(panel, seed: int = 7, gap_frac: float = 0.15, blank_day: int | None = None):
    p = panel.copy()
    rng = np.random.default_rng(seed)
    p = p.mask(rng.random(p.shape) < gap_frac)
    if blank_day is not None:
        days = pd.DatetimeIndex(sorted(set(p.index.normalize())))
        if blank_day < len(days):
            sel = p.index.normalize() == days[blank_day]
            p.loc[sel, :] = np.nan
    return p


def _scalar(fn, *frames, min_finite=2):
    """Run daily_agg{,_two} with the kernel's own __vec__ unbound (pure scalar)."""
    saved = getattr(fn, "__vec__", None)
    if hasattr(fn, "__vec__"):
        del fn.__vec__
    try:
        if len(frames) == 1:
            return _core.daily_agg(frames[0], fn, min_finite=min_finite)
        return _core.daily_agg_two(frames[0], frames[1], fn, min_finite=min_finite)
    finally:
        if saved is not None:
            fn.__vec__ = saved


def _vec(fn, *frames, min_finite=2):
    pvk.bind_whitelist()
    assert getattr(fn, "__vec__", None) is not None, f"{fn.__name__} has no __vec__ after bind"
    if len(frames) == 1:
        return _core.daily_agg(frames[0], fn, min_finite=min_finite)
    return _core.daily_agg_two(frames[0], frames[1], fn, min_finite=min_finite)


def _eq(a: pd.DataFrame, b: pd.DataFrame) -> str:
    idx = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(a.index)).union(set(pd.DatetimeIndex(b.index)))))
    cols = sorted(set(a.columns).union(set(b.columns)))
    a = a.reindex(idx).reindex(cols, axis=1)
    b = b.reindex(idx).reindex(cols, axis=1)
    aa = a.to_numpy(dtype=float)
    bb = b.to_numpy(dtype=float)
    both = np.isfinite(aa) & np.isfinite(bb)
    if both.any() and not np.allclose(aa[both], bb[both], rtol=1e-12, atol=1e-12):
        return f"max abs diff {np.abs(aa[both] - bb[both]).max():.3e}"
    if not np.array_equal(np.isnan(aa), np.isnan(bb)):
        return "NaN-origin mismatch"
    return ""


# ---------------------------------------------------------------------------
# (a) sufficient-statistics accumulator correctness (hand-computed)
# ---------------------------------------------------------------------------

def test_bundle_correct_small_input() -> None:
    idx = pd.DatetimeIndex(["2026-01-05 09:31", "2026-01-05 09:32", "2026-01-05 09:33"])
    fr = pd.DataFrame({
        "A": [1.0, 2.0, 3.0],       # day 0: cnt 3
        "B": [np.nan, 5.0, np.nan],  # day 0: cnt 1
    }, index=idx)
    b = ss.compute_sufficient_statistics(fr)
    cnt = b["cnt"][0]
    assert cnt.tolist() == [3, 1]
    # Σ, Σ² for column A over finite values
    assert b["sum"][0].tolist() == [6.0, 5.0]
    assert b["sumsq"][0].tolist() == [14.0, 25.0]
    assert b["cube"][0].tolist() == [36.0, 125.0]
    assert b["quart"][0].tolist() == [98.0, 625.0]
    assert b["max"][0].tolist() == [3.0, 5.0]
    assert b["min"][0].tolist() == [1.0, 5.0]
    assert b["first"][0].tolist() == [1.0, 5.0]
    assert b["last"][0].tolist() == [3.0, 5.0]
    # argmax: position of first max within the FINITE series (0-based)
    assert b["argmax_pos"][0].tolist() == [2, 0]
    assert b["argmin_pos"][0].tolist() == [0, 0]


def test_bundle_returns_known_moments() -> None:
    idx = pd.DatetimeIndex(["2026-01-05 09:31", "2026-01-05 09:32", "2026-01-05 09:33"])
    fr = pd.DataFrame({"A": [100.0, 101.0, 104.0]}, index=idx)
    b = ss.compute_sufficient_statistics(fr)
    r = b["ret"][0]  # [nan, log(101/100), log(104/101)]
    r1 = np.log(101.0 / 100.0)
    r2 = np.log(104.0 / 101.0)
    assert np.isnan(r[0])
    assert r[1] == pytest.approx(r1)
    assert r[2] == pytest.approx(r2)
    assert b["r2"][0, 0] == pytest.approx(r1 * r1 + r2 * r2)
    assert b["r3"][0, 0] == pytest.approx(r1 ** 3 + r2 ** 3)
    assert b["r4"][0, 0] == pytest.approx(r1 ** 4 + r2 ** 4)


# ---------------------------------------------------------------------------
# (b) O(1)-derivation parity: vectorized suff-stats ops vs scalar reference
# ---------------------------------------------------------------------------

_ONE_PANEL_CASES = [
    (sso._ts_sum, 2), (sso._ts_mean, 2), (sso._ts_variance, 2),
    (sso._ts_std, 2), (sso._ts_min, 2), (sso._ts_max, 2),
    (sso._ts_last, 2), (sso._ts_first, 2), (sso._ts_last_value, 2),
    (sso._ts_argmax, 1), (sso._ts_argmin, 1),
    (sso._ts_realized_variance, 2),
]

_TWO_PANEL_CASES = [
    (sso._ts_vwap, 2), (sso._ts_volume_weighted_return, 2),
    (sso._ts_realized_covariance, 2), (sso._ts_amount_weighted_mean, 2),
]


@pytest.mark.parametrize("fn,mf", _ONE_PANEL_CASES)
def test_one_panel_parity(fn, mf) -> None:
    a = _inject_nan(_close_panel(days=4, seed=42), seed=7, gap_frac=0.15, blank_day=2)
    ref = _scalar(fn, a, min_finite=mf)
    got = _vec(fn, a, min_finite=mf)
    assert not _eq(got, ref), f"{fn.__name__} vec != scalar: {_eq(got, ref)}"


@pytest.mark.parametrize("fn,mf", _TWO_PANEL_CASES)
def test_two_panel_parity(fn, mf) -> None:
    a = _inject_nan(_close_panel(days=4, seed=42), seed=7, gap_frac=0.15, blank_day=2)
    b = _inject_nan(_volume_panel(a, seed=43), seed=8, gap_frac=0.15, blank_day=2)
    ref = _scalar(fn, a, b, min_finite=mf)
    got = _vec(fn, a, b, min_finite=mf)
    assert not _eq(got, ref), f"{fn.__name__} vec != scalar: {_eq(got, ref)}"


# ---------------------------------------------------------------------------
# (b') scan-count / source-read proof: the vec path consumes the shared bundle
# ---------------------------------------------------------------------------

def test_vec_path_builds_shared_bundle_and_derives_in_process() -> None:
    """The vec kernel must read the sufficient-stats bundle, not re-scan bars.

    Proof: (1) before running the vec kernel, the bundle store has no entry for
    this frame; (2) running the vec kernel populates exactly one bundle entry;
    (3) the same bundle object is what a second operator on the same frame gets
    (cache reuse) — so the grid + stats are materialized ONCE, not per operator.
    """
    a = _close_panel(days=2, seed=5)
    # fresh store -> no entry for this frame yet
    ss._STORE._data.clear()
    before = dict(ss._STORE._data)
    assert not any("pair" in k or k[0] == id(a) for k in before)

    res_sum = _vec(sso._ts_sum, a, min_finite=2)
    key = ss._make_key(a)
    assert key in ss._STORE._data, "vec path did not materialize the bundle"
    bundle = ss._STORE._data[key]
    # bundle carries the full grid + sums + packed prefix
    assert set(bundle) >= {"g", "fin", "cnt", "sum", "sumsq", "max", "min",
                           "first", "last", "argmax_pos", "argmin_pos", "r2", "r3", "r4"}
    # the vec kernel's output IS derived from the bundle sums
    assert np.allclose(res_sum.to_numpy(), bundle["sum"], rtol=0, atol=0) or np.allclose(
        res_sum.to_numpy(), bundle["sum"], rtol=1e-12)

    # second operator on the same frame -> identical bundle object (cache reuse)
    bundle2 = ss.compute_sufficient_statistics(a)
    assert bundle2 is bundle


def test_cache_reuse_across_two_operators_same_frame() -> None:
    a = _close_panel(days=2, seed=6)
    ss._STORE._data.clear()
    _vec(sso._ts_mean, a, min_finite=2)
    key = ss._make_key(a)
    bundle = ss._STORE._data[key]
    # compute the bundle directly -> same object (no second grid materialization)
    assert ss.compute_sufficient_statistics(a) is bundle
    # two different one-panel ops on the same frame hit the same bundle
    _vec(sso._ts_max, a, min_finite=2)
    assert ss._STORE._data[key] is bundle
    # exactly one bundle entry for this frame id
    matches = [k for k in ss._STORE._data if k == key]
    assert len(matches) == 1


def test_pair_bundle_reused_for_two_panel_ops() -> None:
    a = _close_panel(days=2, seed=7)
    b = _volume_panel(a, seed=8)
    ss._STORE._data.clear()
    _vec(sso._ts_vwap, a, b, min_finite=2)
    key = ("pair", ss._make_key(a), ss._make_key(b))
    assert key in ss._STORE._data
    bundle = ss._STORE._data[key]
    assert set(bundle) >= {"g", "gb", "cnt", "vsum", "pvsum", "close_cnt"}
    # the second two-panel op reuses the same pair bundle
    _vec(sso._ts_amount_weighted_mean, a, b, min_finite=2)
    assert ss._STORE._data[key] is bundle


# ---------------------------------------------------------------------------
# (c) fail-closed bind
# ---------------------------------------------------------------------------

def test_fail_closed_bind_count() -> None:
    ids = list(pvk.bind_whitelist())
    bound = int(pvk.count_bound())
    assert bound == len(ids), f"count_bound()={bound} != len(bind_whitelist())={len(ids)}"
    assert bound >= 29, "expected >= 29 bound kernels after R61-P1 #57"
    # every sufficient-stats kernel carries a live __vec__
    for fn in [sso._ts_sum, sso._ts_mean, sso._ts_variance, sso._ts_std,
               sso._ts_min, sso._ts_max, sso._ts_last, sso._ts_first,
               sso._ts_last_value, sso._ts_argmax, sso._ts_argmin,
               sso._ts_realized_variance, sso._ts_vwap,
               sso._ts_volume_weighted_return, sso._ts_realized_covariance,
               sso._ts_amount_weighted_mean]:
        assert getattr(fn, "__vec__", None) is not None, f"{fn.__name__} lost its __vec__ bind"
    ts_ids = [i for i in ids if "intra_ts_" in i]
    assert len(ts_ids) == 16, f"expected 16 sufficient-stats whitelist ids, got {len(ts_ids)}"


def test_operator_registered_at_load_all() -> None:
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    ensure_cleaned_loaded()
    for name in ["intra_ts_sum", "intra_ts_mean", "intra_ts_variance",
                 "intra_ts_std", "intra_ts_min", "intra_ts_max",
                 "intra_ts_last", "intra_ts_first", "intra_ts_last_value",
                 "intra_ts_argmax", "intra_ts_argmin",
                 "intra_ts_realized_variance", "intra_ts_vwap",
                 "intra_ts_volume_weighted_return",
                 "intra_ts_realized_covariance",
                 "intra_ts_amount_weighted_mean"]:
        op = OperatorRegistry.get(name, mode="any")
        assert op is not None, f"{name} not registered"
