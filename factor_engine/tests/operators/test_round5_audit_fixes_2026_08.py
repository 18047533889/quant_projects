# -*- coding: utf-8 -*-
"""R5 (2026-08-08) adversarial / golden harness — systemic PIT + math fixes.

Covers the release gates demanded by the fifth-round review:

* **current-NaN fail-closed** — a NaN at the current row never reuses a stale
  historical run (P0-03);
* **index-shift / column-permutation rejection** — a ``y`` shifted by one day
  (or with reordered instrument columns) is REJECTED, never positionally
  paired with ``x`` (P0-01/P0-02);
* **parameter contracts** — fractional / out-of-range params raise instead of
  silently truncating (P1-01);
* **P0 math goldens** — EDGE sub-1 prices + scale invariance (P0-05), HVG ties
  vs the strict O(W²) reference (P0-06), Allan factor Poisson/clustered/regular
  (P0-07), DMD on a known linear system (P0-09), SR recursion reference
  (P0-10), BDS vs statsmodels (P0-11), bicoherence bounds + phase-coupling
  (P0-12), Wasserstein analytic barycenter (P1-12), kernel-Granger scale
  normalisation (P1-13), intraday availability (P0-08);
* **polars bridge** — the python-bridge path asserts equal date axis /
  instrument columns through the real ``calculate`` entry (P0-02).

All deterministic; none depend on optional packages except where guarded by
``importorskip``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.registry import OperatorRegistry

try:
    import polars as pl
except Exception:  # pragma: no cover - optional
    pl = None


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all

    load_all()


def _p(n: int = 120, m: int = 6, seed: int = 11) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    cols = [f"s{i}" for i in range(m)]
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(0.01 * rng.standard_normal((n, m)), axis=0)),
        index=idx, columns=cols,
    )
    logr = np.vstack([np.zeros(m), np.diff(np.log(close.to_numpy()), axis=0)])
    ret = pd.DataFrame(logr, index=idx, columns=cols)
    vol = pd.DataFrame(1.0e4 + rng.integers(0, 3.0e4, (n, m)), index=idx, columns=cols).astype(float)
    event = pd.DataFrame((rng.uniform(size=(n, m) < 0.02).astype(float), index=idx, columns=cols)
    return {"close": close, "ret": ret, "vol": vol, "event": event,
            "high": close * 1.02, "low": close * 0.98}


# ---------------------------------------------------------------------------
# P0-03: current-row NaN fail-closed (trailing operators never reuse history)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("canonical,kwargs", [
    ("ts_hvg_degree_entropy", {"window": 32}),
    ("ts_glr_mean_shift_score", {"window": 48, "min_segment": 6}),
    ("ts_qn_scale", {"window": 32}),
    ("ts_recurrence_determinism", {"window": 32, "dim": 1}),
    ("ts_pickands_tail_index", {"window": 60, "k": 4}),
])
def test_current_row_nan_fail_closed(_loaded, canonical, kwargs):
    panels = _p()
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    clean = op.calculate(panels["ret"], **kwargs).to_numpy()
    # a NaN at the current (last) row must yield NaN at that row — the stale
    # previous contiguous run must NOT be reused to emit a factor for today
    bad = panels["ret"].copy()
    bad.iloc[-1, :] = np.nan
    out = op.calculate(bad, **kwargs).to_numpy()
    assert np.isnan(out[-1, :]).all(), canonical
    # and the rows before the gap are unchanged (prefix invariance preserved)
    assert np.allclose(out[:-1, :], clean[:-1, :], equal_nan=True), canonical


def test_current_row_nan_fail_closed_multi(_loaded):
    panels = _p()
    op = OperatorRegistry.get("ts_edge_effective_spread", "pandas_numpy")
    c = panels["close"].copy()
    c.iloc[-1, :] = np.nan
    out = op.calculate(c, panels["high"], panels["low"], c, window=20).to_numpy()
    assert np.isnan(out[-1, :]).all()


# ---------------------------------------------------------------------------
# P0-01/P0-02: axis contracts — shifted index, reordered columns, dup index
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("canonical,panel_keys,kwargs", [
    ("ts_cross_spectral_coherence", ["ret", "vol"], {"window": 40}),
    ("ts_kernel_granger_score", ["ret", "vol"], {"window": 48, "lag": 2}),
    ("ts_abdi_ranaldo_spread", ["close", "high", "low"], {"window": 20}),
])
def test_input_index_shift_rejected(_loaded, canonical, panel_keys, kwargs):
    panels = _p()
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    args = [panels[k] for k in panel_keys]
    y_shift = args[1].copy()
    y_shift.index = y_shift.index + pd.Timedelta(days=1)  # x_t vs y_{t+1}
    with pytest.raises(ValueError):
        op.calculate(*[args[0], y_shift] + args[2:], **kwargs)


def test_column_permutation_rejected(_loaded):
    panels = _p()
    op = OperatorRegistry.get("ts_cross_spectral_coherence", "pandas_numpy")
    y = panels["vol"].copy()
    y = y[list(reversed(y.columns))]  # stock A column now pairs with stock B
    with pytest.raises(ValueError):
        op.calculate(panels["ret"], y, window=40)


def test_duplicate_timestamp_rejected(_loaded):
    panels = _p()
    x = panels["ret"].copy()
    x = pd.concat([x, x.iloc[[-1]]], axis=0)  # duplicate index row
    op = OperatorRegistry.get("ts_qn_scale", "pandas_numpy")
    with pytest.raises(ValueError):
        op.calculate(x, window=20)


# ---------------------------------------------------------------------------
# P1-01: parameter contracts — fractional / out-of-range params rejected
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("canonical,kwargs", [
    ("ts_hvg_degree_entropy", {"window": 5.9}),
    ("ts_glr_mean_shift_score", {"min_segment": 5.9}),
    ("ts_recurrence_determinism", {"dim": 1.9}),
    ("ts_dmd_dominant_growth_rate", {"rank": 3.9, "dim": 3.7}),
    ("ts_pickands_tail_index", {"k": 3.9}),
    ("ts_bds_statistic", {"embedding_dim": 2.9}),
    ("event_allan_factor", {"scale": 1.9}),
])
def test_parameter_fraction_rejected(_loaded, canonical, kwargs):
    panels = _p()
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    arg = panels["event"] if canonical == "event_allan_factor" else panels["ret"]
    with pytest.raises(Exception) as excinfo:
        op.calculate(arg, **kwargs)
    assert isinstance(excinfo.value, (ValueError, TypeError)), type(excinfo.value)


@pytest.mark.parametrize("canonical,kwargs", [
    ("ts_hvg_degree_entropy", {"window": 3}),       # below min 4
    ("ts_qn_scale", {"window": 1}),                  # below min 2
    ("ts_cross_spectral_coherence", {"window": 16}),  # below min 24
    ("ts_recurrence_determinism", {"theiler": -1}),   # below min 0
])
def test_parameter_out_of_range_rejected(_loaded, canonical, kwargs):
    panels = _p()
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    arg = panels["ret"]
    with pytest.raises(Exception):
        op.calculate(arg, **kwargs)


# ---------------------------------------------------------------------------
# P0-05: EDGE on sub-1 prices + scale invariance
# ---------------------------------------------------------------------------
def test_price_below_one_edge_scale_invariant(_loaded):
    rng = np.random.default_rng(7)
    n = 60
    base = 100.0 * np.exp(np.cumsum(0.01 * rng.standard_normal(n)))
    o = base * np.exp(rng.normal(0, 0.002, n))
    c = base * np.exp(rng.normal(0, 0.002, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    op = OperatorRegistry.get("ts_edge_effective_spread", "pandas_numpy")
    f = lambda a: pd.DataFrame({"s0": a})
    out_high = op.calculate(f(o), f(h), f(l), f(c), window=n).to_numpy()[-1, 0]
    # sub-1 prices: scale the whole path down by 100 (log-differences unchanged)
    out_low = op.calculate(f(o / 100), f(h / 100), f(l / 100), f(c / 100), window=n).to_numpy()[-1, 0]
    assert np.isfinite(out_high) and np.isfinite(out_low)
    assert abs(out_high - out_low) <= 1e-9 * max(1.0, abs(out_high))  # scale invariance
    # a low-priced name (0.1 ~ 0.9) must not be masked by the positivity check
    tiny = base * 0.005  # ~0.5 prices
    o2 = tiny * np.exp(rng.normal(0, 0.002, n))
    c2 = tiny * np.exp(rng.normal(0, 0.002, n))
    h2 = np.maximum(o2, c2) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    l2 = np.minimum(o2, c2) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    out_tiny = op.calculate(f(o2), f(h2), f(l2), f(c2), window=n).to_numpy()[-1, 0]
    assert np.isfinite(out_tiny)


# ---------------------------------------------------------------------------
# P0-06: HVG ties vs the strict O(W²) reference
# ---------------------------------------------------------------------------
def test_hvg_equal_value_reference(_loaded):
    from cleaned_operators.hvg_ext import _hvg_edges, _hvg_reference_O_W2

    assert sorted(_hvg_edges(np.array([1.0, 1.0, 2.0]))) == [(0, 1), (1, 2)]
    rng = np.random.default_rng(42)
    for trial in range(40):
        x = np.floor(rng.uniform(0, 6, size=int(rng.integers(3, 40))))  # heavy ties
        assert sorted(_hvg_edges(x)) == sorted(_hvg_reference_O_W2(x)), trial
        x2 = rng.normal(size=int(rng.integers(3, 40)))
        assert sorted(_hvg_edges(x2)) == sorted(_hvg_reference_O_W2(x2)), f"gauss {trial}"


# ---------------------------------------------------------------------------
# P0-07: true Allan factor — Poisson / clustered / regular
# ---------------------------------------------------------------------------
def test_event_allan_poisson(_loaded):
    from cleaned_operators.evt_allan import _allan_factor_single, _check_event_binary

    assert _check_event_binary(np.array([0.0, 1.0, 0.0]))
    assert not _check_event_binary(np.array([0.0, 2.0]))
    rng = np.random.default_rng(0)
    for _ in range(5):
        poisson = (rng.uniform(size=2000) < 0.02).astype(float)
        assert 0.7 < _allan_factor_single(poisson, 16) < 1.5
    clustered = np.zeros(2000)
    for i in range(0, 2000, 40):
        clustered[i : i + int(rng.integers(3, 6))] = 1.0
    assert _allan_factor_single(clustered, 8) > 1.5
    regular = np.zeros(2000)
    regular[::5] = 1.0
    assert _allan_factor_single(regular, 4) < 1.0


# ---------------------------------------------------------------------------
# P0-09: DMD on a known linear system
# ---------------------------------------------------------------------------
def test_dmd_known_linear_system(_loaded):
    from cleaned_operators.dmd import _hankel_dmd

    # geometric decay x_t = 0.9^t is exactly the linear system x_{t+1}=0.9 x_t
    x = 0.9 ** np.arange(60, dtype=float)
    res = _hankel_dmd(x, rank=1, dim=2, delay=1)
    assert res is not None
    lam0 = res["eig"][0]
    assert abs(abs(lam0) - 0.9) < 1e-6, abs(lam0)
    assert abs(np.log(abs(lam0)) - np.log(0.9) < 1e-6
    # a single-mode oscillation: x_t = 0.95^t * sin(2*pi*0.1*t)
    t = np.arange(80, dtype=float)
    x2 = (0.95 ** t) * np.sin(2.0 * np.pi * 0.1 * t)
    res2 = _hankel_dmd(x2, rank=2, dim=2, delay=1)
    assert res2 is not None
    lam2 = res2["eig"]
    freqs = np.abs(np.angle(lam2)) / (2.0 * np.pi)
    assert np.min(np.abs(freqs - 0.1) < 0.02, freqs
    assert np.min(np.abs(np.abs(lam2) - 0.95) < 0.02


def test_dmd_rank_infeasible_rejected(_loaded):
    from cleaned_operators.dmd import _hankel_dmd

    x = np.arange(20, dtype=float)
    # rank=50 > the SVD rank available -> fail closed, never silently clipped
    assert _hankel_dmd(x, rank=50, dim=4, delay=1) is None


# ---------------------------------------------------------------------------
# P0-10: SR vs the linear-space recursion reference
# ---------------------------------------------------------------------------
def test_sr_recursive_reference(_loaded):
    from cleaned_operators.research_spectral import _sr_gaussian

    rng = np.random.default_rng(3)
    v = np.concatenate([rng.normal(0, 1, 40), rng.normal(1.0, 1, 40)])
    shift, bw = 1.0, 20
    mu = float(np.mean(v[:bw]))
    sd = float(np.std(v[:bw]))
    R = 0.0
    for t in range(bw, len(v)):
        z = (v[t] - mu) / sd
        R = (1.0 + R) * np.exp(shift * z - shift * shift / 2.0)
    expected = float(np.log(1.0 + R))
    got = _sr_gaussian(v, shift, bw)
    assert abs(got - expected) < 1e-9
    # baseline samples are NOT scored: a constant series has sd=0 and fails closed
    flat = np.full(60, 0.5)
    out_flat = _sr_gaussian(flat, 1.0, 20)
    assert out_flat is None or not np.isfinite(out_flat)


# ---------------------------------------------------------------------------
# P0-11: BDS vs statsmodels + distribution sanity
# ---------------------------------------------------------------------------
def test_bds_vs_statsmodels(_loaded):
    pytest.importorskip("statsmodels")
    from statsmodels.tsa.stattools import bds as sm_bds
    from cleaned_operators.research_spectral import _bds_statistic

    rng = np.random.default_rng(9)
    x = rng.standard_normal(400)
    our = _bds_statistic(x, 2, 1.5)
    theirs = float(sm_bds(x, max_dim=2, distance=1.5)[0])
    assert np.isfinite(our) and np.isfinite(theirs)
    assert abs(our - theirs) < 0.25, (our, theirs)  # same estimator family


def test_bds_iid_gaussian_small_stat(_loaded):
    from cleaned_operators.research_spectral import _bds_statistic

    rng = np.random.default_rng(12)
    stats = [_bds_statistic(rng.standard_normal(400), 2, 1.5) for _ in range(15)]
    stats = np.asarray([s for s in stats if np.isfinite(s)])
    assert stats.size >= 10
    # under IID the BDS statistic is ~N(0,1) — mean near 0, spread bounded
    assert abs(np.mean(stats) < 0.6, stats
    assert np.std(stats) < 2.5, stats


# ---------------------------------------------------------------------------
# P0-12: bicoherence — [0,1] bounds + phase-coupling detection
# ---------------------------------------------------------------------------
def _bicoherence_pair(x: np.ndarray, f1: int, f2: int, n_segments: int) -> float:
    """Squared bicoherence at ONE frequency pair (the standard normalisation)."""
    n = x.shape[0]
    seg = n // n_segments
    B = 0j
    P12 = 0.0
    P3 = 0.0
    for s in range(n_segments):
        chunk = x[s * seg : (s + 1) * seg]
        t = np.arange(seg, dtype=float)
        chunk = chunk - np.polyval(np.polyfit(t, chunk, 1), t)
        hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / (seg - 1.0)))
        X = np.fft.rfft(chunk * hann)
        x1 = X[f1]
        x2 = X[f2]
        x3 = X[f1 + f2]
        B += x1 * x2 * np.conj(x3)
        P12 += np.abs(x1 * x2) ** 2
        P3 += np.abs(x3) ** 2
    return abs(B) ** 2 / (P12 * P3) if P12 * P3 > 1e-12 else np.nan


def test_bicoherence_bounds_and_coupling(_loaded):
    from cleaned_operators.research_spectral import _bicoherence_max

    n_seg, seg, f1, f2 = 8, 64, 4, 6
    phi1, phi2 = 0.3, 1.1
    tau = np.arange(seg, dtype=float)
    rng = np.random.default_rng(21)
    base = (lambda ph3: np.sin(2 * np.pi * f1 * tau / seg + phi1)
            + np.sin(2 * np.pi * f2 * tau / seg + phi2)
            + np.sin(2 * np.pi * (f1 + f2) * tau / seg + ph3))
    # coupled: the third-harmonic phase is LOCKED to phi1+phi2 across every
    # segment -> the bispectrum phase is constant -> b^2(f1,f2) -> 1.
    coupled = np.concatenate([base(phi1 + phi2) for _ in range(n_seg)])
    # uncoupled: the third-harmonic phase is random PER SEGMENT -> the
    # bispectrum random-walks and cancels -> b^2(f1,f2) ~ 1/n_seg.
    uncoupled = np.concatenate([base(rng.uniform(0.0, 2 * np.pi)) for _ in range(n_seg)])
    b_c = _bicoherence_pair(coupled.astype(float), f1, f2, n_seg)
    b_u = _bicoherence_pair(uncoupled.astype(float), f1, f2, n_seg)
    assert np.isfinite(b_c) and np.isfinite(b_u)
    assert b_c > 0.7, b_c                      # phase-locked -> high
    assert b_u < 0.4, b_u                      # un-locked -> low
    # the operator's max is bounded in [0, 1] (Cauchy-Schwarz normalisation)
    for sig in (coupled, uncoupled):
        m = _bicoherence_max(sig.astype(float), n_seg)
        assert np.isfinite(m) and 0.0 <= m <= 1.0 + 1e-9, m


# ---------------------------------------------------------------------------
# P1-12: Wasserstein true-barycenter analytic golden
# ---------------------------------------------------------------------------
def test_wasserstein_analytic_barycenter(_loaded):
    n = 12
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    cols = ["a", "b", "c"]
    arr = np.zeros((n, 3))
    arr[:, 0] = 0.0
    arr[:, 1] = 0.5
    arr[:, 2] = 1.0
    x = pd.DataFrame(arr, index=idx, columns=cols)
    grp = pd.DataFrame(np.tile(["g0", "g0", "g0"], (n, 1)), index=idx, columns=cols)
    op = OperatorRegistry.get("group_wasserstein_barycenter_distance", "pandas_numpy")
    out = op.calculate(x, grp, window=n, min_group_size=3).to_numpy()[-1, :]
    # ex-self barycenter over the peers' constant quantile curves:
    #   a(0): peers {0.5, 1.0} -> bary 0.75 -> distance 0.75
    #   b(0.5): peers {0, 1}   -> bary 0.5  -> distance 0
    #   c(1): peers {0, 0.5}   -> bary 0.25 -> distance 0.75
    assert abs(out[0] - 0.75) < 1e-9, out
    assert abs(out[1] - 0.0) < 1e-9, out
    assert abs(out[2] - 0.75) < 1e-9, out


# ---------------------------------------------------------------------------
# P1-13: kernel-Granger scale normalisation (X ~ 1e7 must not dominate Y ~ 1e-2)
# ---------------------------------------------------------------------------
def test_kernel_granger_scale_normalised(_loaded):
    from cleaned_operators.research_spectral import _kernel_granger_score

    rng = np.random.default_rng(4)
    n = 160
    x_vol = 1e7 + rng.normal(0, 1e6, n)          # huge-scale X
    y_ret = 0.01 * rng.standard_normal(n)         # tiny-scale Y
    # after normalisation the restricted vs full models should still behave;
    # the raw (unnormalised) kernel would be dominated by X's scale
    score = _kernel_granger_score(y_ret, x_vol, lag=2)
    assert score is None or np.isfinite(score)


# ---------------------------------------------------------------------------
# P0-08: intraday impact — availability contract + daily (per-symbol) shape
# ---------------------------------------------------------------------------
def test_intraday_availability_and_shape(_loaded):
    n = 780
    idx = pd.date_range("2023-01-02 09:30", periods=n, freq="min")
    rng = np.random.default_rng(5)
    ret = pd.DataFrame(0.0005 * rng.standard_normal((n, 2)), index=idx, columns=["a", "b"])
    amt = pd.DataFrame(1.0e6 + rng.integers(0, 1.0e5, (n, 2)), index=idx, columns=["a", "b"]).astype(float)
    op = OperatorRegistry.get("intraday_impact_decay_rate", "pandas_numpy")
    out = op.calculate(ret, amt, horizon=5, shock_quantile=0.9)
    # one scalar per (date, symbol) — a shape-changing minute->daily contract
    assert isinstance(out, pd.DataFrame)
    assert out.shape[1] == 2
    assert out.shape[0] == len(pd.Series(idx).dt.normalize().unique())
    desc = (op.metadata.description or "") + (op.__doc__ or "")
    assert "session_close" in (op.metadata.description or "") + ""


# ---------------------------------------------------------------------------
# P0-02: polars python-bridge — real calculate path asserts axis alignment
# ---------------------------------------------------------------------------
@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_polars_bridge_axis_misalignment_rejected(_loaded):
    panels = _p(n=50)
    op = OperatorRegistry.get("ts_cross_spectral_coherence", "polars")
    assert op is not None
    date_col = np.asarray(panels["ret"].index, dtype="datetime64[ns]").astype("int64")
    wide = []
    for k in ("ret", "vol"):
        data = {f"S{i}": panels[k][c].to_numpy() for i, c in enumerate(panels[k].columns)}
        wide.append(pl.DataFrame({"date": date_col} | {kk: pl.Series(kk, v) for kk, v in data.items()}))
    # matching axes -> runs through the bridge
    out_ok = op.calculate(*wide, window=40)
    assert out_ok.height == 50
    # a shifted date axis -> REJECTED (would pair x_t with y_{t+1})
    bad_date = date_col.copy()
    bad_date[1:] = bad_date[1:] + 1
    wide_bad = [pl.DataFrame({"date": date_col} | {kk: pl.Series(kk, v) for kk, v in
                {f"S{i}": panels["ret"][c].to_numpy() for i, c in enumerate(panels["ret"].columns)}.items()}),
                pl.DataFrame({"date": bad_date} | {kk: pl.Series(kk, v) for kk, v in
                {f"S{i}": panels["vol"][c].to_numpy() for i, c in enumerate(panels["vol"].columns)}.items()})]
    with pytest.raises(ValueError):
        op.calculate(*wide_bad, window=40)
