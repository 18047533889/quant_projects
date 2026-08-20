# -*- coding: utf-8 -*-
"""R20 causal intraday BVC / VPIN DirectUse rehabilitation oracles.

Slice: R20-BVC-VPIN-DIRECTUSE.

Duplicate-audit outcome (survey of cleaned_operators/ BEFORE implementing —
documented, no duplicate canonical names landed):
* ``micro_vpin`` (microstructure/ops.py, experimental legacy_proxy) is a
  volume-weighted |return| INTENSITY proxy — explicitly documented as NOT
  strict VPIN.  The landed ``vpin_pct`` is the first genuine equal-volume-
  bucket VPIN on the direct-use surface -> not a duplicate, untouched.
* ``micro_bvc_vpin`` (microstructure/, P2/research-only per
  production_hardening) is a panel-level flow_impact pipeline with a causal
  replacement mapping already maintained -> untouched, not promoted here.
* ``intraday_bvc_imbalance`` (microstructure/flow_impact.py, chip-flow pack)
  is a minute-PANEL per-(date,symbol) daily aggregate using the 2*Phi(z)-1
  CDF flow — different estimator (probabilistic, not sign) and different
  input/output shape than the series-level trailing-window canonicals here.
* ``micro_trade_imbalance`` (microstructure/ops.py, experimental) is
  session-aware with min_periods != window and no ParamSpec — a different
  contract (session resets, partial windows), documented near-neighbour.
* ``signed_volume_imbalance`` (price_volume/liquidity_v2.py) is SHARE-based
  (up-share minus down-share), not volume-magnitude-weighted BVC.

Landed canonicals (family contract: trailing windows END at t,
min_periods=window, any NaN / non-positive close / non-positive volume bar
inside a window -> NaN, fail-closed — never zero-filled, never abs()-
laundered; zero-volume window -> NaN; dimensionless outputs):
* ``bvc_sign_pct``  — sum(sign(dclose)*volume)/sum(volume) in [-1, 1];
  rolling-only bounded state (NOT in _RECURSIVE_EWM).
* ``vpin_pct``      — equal-volume-bucket VPIN (mean |bucket flow|/bucket
  volume == sum_b |OF_b|/total volume) in [0, 1]; rolling-only.
* ``bvc_imbalance_ma`` — (EMA_fast - EMA_slow) of the signed flow divided by
  EMA_slow(volume); EMA-based -> IN _RECURSIVE_EWM (stateful/full_replay).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _ensure_technical_chain() -> None:
    from cleaned_operators.registry import OperatorRegistry

    if OperatorRegistry.lifecycle() == "frozen":
        return
    if OperatorRegistry.get("bvc_sign_pct", "pandas_numpy") is not None:
        return
    from cleaned_operators.technical import signal  # noqa: F401
    from cleaned_operators.technical import polars_signal  # noqa: F401
    from cleaned_operators import composite_fastpath  # noqa: F401
    from cleaned_operators.technical import indicators_v2  # noqa: F401


@pytest.fixture(scope="module", autouse=True)
def _bootstrap():
    _ensure_technical_chain()


def _op(name: str):
    from cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get(name, "pandas_numpy") or OperatorRegistry.get(name)
    assert op is not None, f"{name} not registered"
    return op


BVC_NAMES = ("bvc_sign_pct", "vpin_pct", "bvc_imbalance_ma")


def _random_panel(n=60, seed=7, vol_seed=8):
    rng = np.random.default_rng(seed)
    vrng = np.random.default_rng(vol_seed)
    close = pd.DataFrame({"A": 100.0 + np.cumsum(rng.normal(0.05, 0.6, n))})
    vol = pd.DataFrame({"A": 1.0e6 + vrng.random(n) * 5.0e5})
    return close, vol


# ---------------------------------------------------------------------------
# manual oracles — explicit closed-form loops, fully independent of the
# operator helpers in indicators_v2
# ---------------------------------------------------------------------------
def _manual_bvc_sign(close, volume, window):
    c = close.to_numpy(float).ravel()
    v = volume.to_numpy(float).ravel()
    out = []
    for t in range(len(c)):
        if t < window - 1:
            out.append(np.nan)
            continue
        cs = c[t - window + 1 : t + 1]
        vs = v[t - window + 1 : t + 1]
        # every bar needs a FORMED price change (t-1 exists and both closes
        # are finite strict-positive) and a finite strict-positive volume
        ok = True
        flows = []
        for i in range(window):
            if i == 0:
                # first bar of the window needs close[t-window] formed
                j = t - window + 1
                if j == 0:
                    ok = False
                    break
                prev = c[j - 1]
            else:
                prev = cs[i - 1]
            cur = cs[i]
            if not (np.isfinite(cur) and np.isfinite(prev) and cur > 0.0 and prev > 0.0):
                ok = False
                break
            if not (np.isfinite(vs[i]) and vs[i] > 0.0):
                ok = False
                break
            flows.append(float(np.sign(cur - prev)) * float(vs[i]))
        if not ok:
            out.append(np.nan)
            continue
        den = float(vs.sum())
        out.append(np.nan if den <= 0.0 else float(np.sum(flows)) / den)
    return pd.DataFrame(out, index=close.index, columns=close.columns)


def _manual_vpin(close, volume, window, buckets):
    c = close.to_numpy(float).ravel()
    v = volume.to_numpy(float).ravel()
    out = []
    for t in range(len(c)):
        if t < window - 1:
            out.append(np.nan)
            continue
        cs = c[t - window + 1 : t + 1]
        vs = v[t - window + 1 : t + 1]
        if not (np.isfinite(cs).all() and np.isfinite(vs).all()):
            out.append(np.nan)
            continue
        if np.any(cs <= 0.0) or np.any(vs <= 0.0):
            out.append(np.nan)
            continue
        # per-bar BVC sign flow (first bar of the window has no formed change
        # INSIDE the window, but its level still anchors the first diff)
        flow = np.sign(np.diff(cs)) * vs[1:]
        vol = vs[1:]
        total = float(vol.sum())
        if total <= 0.0:
            out.append(np.nan)
            continue
        b = max(2, int(buckets))
        target = total / b
        capacity = target
        bucket_flow = 0.0
        abs_of = 0.0
        for m in range(vol.shape[0]):
            vm = float(vol[m])
            ofm = float(flow[m])
            if vm <= 0.0:
                continue
            v_left = vm
            while v_left > 0.0:
                take = min(v_left, capacity)
                if take <= 0.0:
                    break
                bucket_flow += ofm * (take / vm)
                v_left -= take
                capacity -= take
                if capacity <= 0.0:
                    abs_of += abs(bucket_flow)
                    bucket_flow = 0.0
                    capacity = target
        if capacity < target:
            abs_of += abs(bucket_flow)
        out.append(abs_of / total)
    return pd.DataFrame(out, index=close.index, columns=close.columns)


def _manual_ema_from(values, spans, warmup_start):
    """EMA(adjust=False, min_periods=span) with skip-if-NaN state semantics,
    started from the first finite value at/after ``warmup_start``."""
    outs = []
    for span in spans:
        alpha = 2.0 / (span + 1.0)
        y = None
        cnt = 0
        col = []
        for x in values:
            if not np.isfinite(x):
                col.append(np.nan)
                continue
            y = x if y is None else (1.0 - alpha) * y + alpha * x
            cnt += 1
            col.append(y if cnt >= span else np.nan)
        outs.append(np.asarray(col))
    return outs


def _manual_bvc_ma(close, volume, fast, slow):
    c = close.to_numpy(float).ravel()
    v = volume.to_numpy(float).ravel()
    n = len(c)
    flow = []
    valid = []
    for t in range(n):
        if t == 0:
            flow.append(np.nan)
            valid.append(False)
            continue
        cur, prev = c[t], c[t - 1]
        ok = (
            np.isfinite(cur) and np.isfinite(prev) and cur > 0.0 and prev > 0.0
            and np.isfinite(v[t]) and v[t] > 0.0
        )
        valid.append(bool(ok))
        flow.append(float(np.sign(cur - prev)) * float(v[t]) if ok else np.nan)
    flow = np.asarray(flow)
    vol_cohort = np.asarray([v[t] if valid[t] else np.nan for t in range(n)])
    ef, es = _manual_ema_from(flow, [fast, slow], 0)
    ev = _manual_ema_from(vol_cohort, [slow], 0)[0]
    out = []
    for t in range(n):
        num = ef[t] - es[t]
        den = ev[t]
        if not (np.isfinite(num) and np.isfinite(den) and den > 0.0):
            out.append(np.nan)
        else:
            out.append(num / den)
    return pd.DataFrame(out, index=close.index, columns=close.columns)


@pytest.mark.parametrize("window", [5, 10])
def test_bvc_sign_pct_manual_oracle(window):
    close, vol = _random_panel(n=60, seed=7)
    out = _op("bvc_sign_pct").calculate(close, vol, window=window)
    expected = _manual_bvc_sign(close, vol, window)
    np.testing.assert_allclose(
        out.to_numpy(), expected.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True
    )
    # warmup: bar 0's price change is unformed, so the first fully-formed
    # window is bars 1..window -> output starts at t == window (w NaNs).
    assert out.iloc[:window].isna().all().all()
    assert out.iloc[window:].notna().all().all()


@pytest.mark.parametrize("window,buckets", [(10, 4), (20, 5), (12, 8)])
def test_vpin_pct_manual_oracle(window, buckets):
    close, vol = _random_panel(n=80, seed=11)
    out = _op("vpin_pct").calculate(close, vol, window=window, buckets=buckets)
    expected = _manual_vpin(close, vol, window, buckets)
    np.testing.assert_allclose(
        out.to_numpy(), expected.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True
    )
    # warmup w-1 NaNs; the diff-based flow discards only the first bar's
    # LEVEL (still finite), so vpin emits from t == window - 1
    assert out.iloc[: window - 1].isna().all().all()
    assert out.iloc[window - 1 :].notna().all().all()
    vals = out.to_numpy(float)
    assert np.nanmin(vals) >= -1e-9 and np.nanmax(vals) <= 1.0 + 1e-9


def test_bvc_imbalance_ma_manual_oracle():
    close, vol = _random_panel(n=120, seed=13)
    fast, slow = 3, 10
    out = _op("bvc_imbalance_ma").calculate(close, vol, fast_window=fast, slow_window=slow)
    expected = _manual_bvc_ma(close, vol, fast, slow)
    np.testing.assert_allclose(
        out.to_numpy(), expected.to_numpy(), rtol=1e-9, atol=1e-9, equal_nan=True
    )


# ---------------------------------------------------------------------------
# closed-form synthetic panels with known sign structure
# ---------------------------------------------------------------------------
def test_bvc_sign_pct_pure_direction_panels():
    w = 6
    # monotone rising closes -> every formed flow positive -> +1 exactly
    close = pd.DataFrame({"A": [10.0 + 0.5 * i for i in range(20)]})
    vol = pd.DataFrame({"A": [1.0e6] * 20})
    out = _op("bvc_sign_pct").calculate(close, vol, window=w)
    np.testing.assert_allclose(out.iloc[w:].to_numpy(), 1.0, atol=1e-9)
    # monotone falling -> -1 exactly
    close_dn = pd.DataFrame({"A": [30.0 - 0.5 * i for i in range(20)]})
    out_dn = _op("bvc_sign_pct").calculate(close_dn, vol, window=w)
    np.testing.assert_allclose(out_dn.iloc[w:].to_numpy(), -1.0, atol=1e-9)
    # alternating up/down with EQUAL volumes -> perfectly balanced 0
    alt = [10.0 + (0.5 if i % 2 == 0 else -0.5) for i in range(20)]
    out_alt = _op("bvc_sign_pct").calculate(pd.DataFrame({"A": alt}), vol, window=w)
    # an even window over a period-2 alternation nets exactly zero
    np.testing.assert_allclose(out_alt.iloc[w:].to_numpy(), 0.0, atol=1e-9)


def test_vpin_pct_closed_form_uniform_buckets():
    # 4 bars, 4 buckets, equal volumes 100 each, monotone up: every bucket is
    # pure buy flow -> |OF_b| = 100 per bucket, total = 400 -> VPIN == 1.
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0, 14.0]})
    vol = pd.DataFrame({"A": [100.0] * 5})
    out = _op("vpin_pct").calculate(close, vol, window=5, buckets=4)
    np.testing.assert_allclose(out.iloc[4:].to_numpy(), 1.0, atol=1e-9)
    # alternating direction with equal volumes: with 4 buckets over 400 total
    # volume each bucket is 100 = exactly one bar -> each bucket carries that
    # bar's pure one-signed flow -> VPIN == 1 (equal-volume bucketing is
    # unaffected by sign alternation when buckets align with bar boundaries).
    alt = [10.0, 10.5, 10.0, 10.5, 10.0]
    out0 = _op("vpin_pct").calculate(pd.DataFrame({"A": alt}), vol, window=5, buckets=4)
    np.testing.assert_allclose(out0.iloc[4:].to_numpy(), 1.0, atol=1e-9)
    # exact-zero construction: a bucket must contain an offsetting +/- flow
    # PAIR.  4 flow bars (+,-,+,-) with equal volumes 100 and buckets=2 (each
    # bucket = 200 = exactly two bars): bucket1 = bars 1+2 -> 0, bucket2 =
    # bars 3+4 -> 0  =>  VPIN == 0 exactly.
    zc = [10.0, 11.0, 10.0, 11.0, 10.0]  # diffs +1,-1,+1,-1
    zv = pd.DataFrame({"A": [100.0] * 5})
    outz = _op("vpin_pct").calculate(pd.DataFrame({"A": zc}), zv, window=5, buckets=2)
    np.testing.assert_allclose(outz.iloc[4:].to_numpy(), 0.0, atol=1e-9)


def test_vpin_pct_volume_bucket_split_closed_form():
    # One huge bar straddles many buckets: closes [10, 9, 10, 11, 12] give
    # flow bars 2..5 with volumes (300, 30, 30, 30) — total 390, so 4 buckets
    # of 97.5 each.  Flows: bar2 = -300, bars 3..5 = +30 each.
    # Trace: b1 = -97.5 (bar2 97.5/300); b2 = -97.5; b3 = -97.5 (bar2's last
    # 97.5); b4 = 30+30+30 = 82.5 minus... bar2 contributes exactly 3 buckets
    # (3*97.5 = 292.5, leaving 7.5 of bar2 + the three 30-volume bars = 97.5
    # = exactly b4): b4 = -300*(7.5/300) + 30 + 30 + 30 = -7.5 + 90 = 82.5.
    # sum|OF| = 97.5*3 + 82.5 = 375.0; VPIN = 375/390 = 0.961538461538...
    close = pd.DataFrame({"A": [10.0, 9.0, 10.0, 11.0, 12.0]})
    vol = pd.DataFrame({"A": [50.0, 300.0, 30.0, 30.0, 30.0]})
    out = _op("vpin_pct").calculate(close, vol, window=5, buckets=4)
    np.testing.assert_allclose(out.iloc[4, 0], 375.0 / 390.0, rtol=1e-9)


def test_bvc_imbalance_ma_sign_structure():
    # Persistent one-direction flow with NON-constant volumes: the fast EMA of
    # the flow reacts to the rising volume trend sooner than the slow EMA ->
    # a positive spread after the volume ramp (constant volumes make the two
    # EMAs exactly equal — spread 0 — which is itself a valid boundedness
    # check).  |output| <= 1: |flow| <= volume on the cohort and EMA is a
    # convex combination.
    rng = np.random.default_rng(31)
    close = pd.DataFrame({"A": [10.0 + 0.5 * i for i in range(80)]})
    vol = pd.DataFrame({"A": 1.0e6 + 2.0e4 * np.arange(80) + rng.random(80) * 1e3})
    out = _op("bvc_imbalance_ma").calculate(close, vol, fast_window=3, slow_window=20)
    tail = out.iloc[30:]
    assert np.isfinite(tail.to_numpy()).all()
    assert (tail > 0.0).all().all()
    assert np.nanmax(np.abs(out.to_numpy())) <= 1.0 + 1e-9
    # constant volumes + constant flow -> fast EMA == slow EMA -> spread 0
    # (a genuine closed-form degenerate-value oracle, not a masked NaN)
    close_c = pd.DataFrame({"A": [10.0 + 0.5 * i for i in range(60)]})
    vol_c = pd.DataFrame({"A": [1.0e6] * 60})
    out_c = _op("bvc_imbalance_ma").calculate(close_c, vol_c, fast_window=3, slow_window=20)
    np.testing.assert_allclose(out_c.iloc[40:].to_numpy(), 0.0, atol=1e-12)
    assert np.isfinite(out_c.iloc[40:].to_numpy()).all()


# ---------------------------------------------------------------------------
# masking: bad close / bad volume kill covering windows (fail-closed)
# ---------------------------------------------------------------------------
def test_bvc_masks_non_positive_and_missing_volume():
    n = 40
    close, vol = _random_panel(n=n, seed=5)
    w = 8
    # clean baseline computed BEFORE poisoning (scoped-review P1: computing it
    # from a mutated copy made the prefix comparison vacuous)
    clean = _op("bvc_sign_pct").calculate(close, vol.copy(), window=w)
    for bad in (0.0, -5.0e3, float("nan")):
        v2 = vol.copy()
        v2.iloc[20, 0] = bad
        out = _op("bvc_sign_pct").calculate(close, v2, window=w)
        assert out.iloc[20:28].isna().all().all(), f"volume={bad}: covering windows must be NaN"
        assert out.iloc[28:].notna().all().all(), f"volume={bad}: later windows must recover"
        pd.testing.assert_frame_equal(out.iloc[:12], clean.iloc[:12])
    # non-positive CLOSE anywhere in the window -> NaN (R5-38, never abs())
    c2 = close.copy()
    c2.iloc[20, 0] = 0.0
    out_c = _op("bvc_sign_pct").calculate(c2, vol, window=w)
    assert out_c.iloc[20:28].isna().all().all()
    # vpin: same fail-closed window kill on a bad volume bar
    v3 = vol.copy()
    v3.iloc[20, 0] = 0.0
    out_v = _op("vpin_pct").calculate(close, v3, window=w, buckets=4)
    assert out_v.iloc[20:28].isna().all().all()
    assert out_v.iloc[28:].notna().all().all()
    # bvc_imbalance_ma: the bad bar drops the flow/volume cohort at t=20;
    # EMA(adjust=False) freezes the state across the NaN (skip semantics), so
    # values stay finite but are NOT contaminated by the bad bar — assert the
    # poisoned suffix differs from the clean baseline while the pre-poison
    # prefix is bit-identical (causal contamination check).
    clean_m = _op("bvc_imbalance_ma").calculate(close, vol.copy(), fast_window=3, slow_window=8)
    out_m = _op("bvc_imbalance_ma").calculate(close, v3, fast_window=3, slow_window=8)
    pd.testing.assert_frame_equal(out_m.iloc[:20], clean_m.iloc[:20], check_freq=False)
    assert not out_m.iloc[21:].equals(clean_m.iloc[21:])


def test_bvc_zero_volume_window_is_nan():
    vals = list(np.linspace(10.0, 12.0, 12))
    close = pd.DataFrame({"A": vals})
    vol = pd.DataFrame({"A": [0.0] * 12})
    for name, run in {
        "bvc_sign_pct": lambda: _op("bvc_sign_pct").calculate(close, vol, window=3),
        "vpin_pct": lambda: _op("vpin_pct").calculate(close, vol, window=3, buckets=2),
    }.items():
        out = run()
        assert out.isna().all().all(), name


def test_bvc_imbalance_ma_degenerate_denominator():
    # all volume zero -> EMA(volume) is all NaN/0 -> output all NaN
    close = pd.DataFrame({"A": [10.0 + 0.3 * i for i in range(30)]})
    vol = pd.DataFrame({"A": [0.0] * 30})
    out = _op("bvc_imbalance_ma").calculate(close, vol, fast_window=3, slow_window=8)
    assert out.isna().all().all()


def test_nan_in_window_fail_closed():
    close, vol = _random_panel(n=40, seed=5)
    w = 8
    close2 = close.copy()
    close2.iloc[20, 0] = np.nan
    out = _op("bvc_sign_pct").calculate(close2, vol, window=w)
    # windows containing bar 20 as the FIRST bar need close[19] vs close[20]:
    # the bad close kills both the t=20 window (bad close at t) and the
    # t=21 window (bad close at t-1) -> NaN for t in 20..28
    assert out.iloc[20:29].isna().all().all()
    assert out.iloc[29:].notna().all().all()
    out_v = _op("vpin_pct").calculate(close2, vol, window=w, buckets=4)
    # vpin consumes only the close LEVELS (the diff is formed inside the
    # kernel): the NaN bar kills exactly the w covering windows
    assert out_v.iloc[20:28].isna().all().all()
    assert out_v.iloc[28:].notna().all().all()


# ---------------------------------------------------------------------------
# causality: mutating bar t must not change outputs at t' < t
# ---------------------------------------------------------------------------
def test_causality_mutation_does_not_change_past():
    close, vol = _random_panel(n=50, seed=9)
    w = 10
    c2 = close.copy()
    c2.iloc[30, 0] = c2.iloc[30, 0] * 1.5
    v2 = vol.copy()
    v2.iloc[30, 0] = v2.iloc[30, 0] * 2.0
    for name, base, mut in [
        ("bvc_sign_pct",
         _op("bvc_sign_pct").calculate(close, vol, window=w),
         _op("bvc_sign_pct").calculate(c2, v2, window=w)),
        ("vpin_pct",
         _op("vpin_pct").calculate(close, vol, window=w, buckets=5),
         _op("vpin_pct").calculate(c2, v2, window=w, buckets=5)),
        ("bvc_imbalance_ma",
         _op("bvc_imbalance_ma").calculate(close, vol, fast_window=3, slow_window=10),
         _op("bvc_imbalance_ma").calculate(c2, v2, fast_window=3, slow_window=10)),
    ]:
        pd.testing.assert_frame_equal(base.iloc[:30], mut.iloc[:30], check_freq=False)


def test_prefix_invariance_rolling_only():
    # bvc_sign_pct / vpin_pct are rolling-only: a truncated panel must equal
    # the full-panel prefix EXACTLY.  bvc_imbalance_ma is EMA-based
    # (stateful): the EMA over the truncated panel equals the full-panel EMA
    # only while the state is identical — pandas ewm(adjust=False) is a pure
    # forward recursion, so the prefix property holds there too (documented:
    # the op is tagged stateful/full_replay for production governance, but
    # the pandas kernel itself is a forward recursion).
    close, vol = _random_panel(n=60, seed=13)
    cases = {
        "bvc_sign_pct": lambda c, v: _op("bvc_sign_pct").calculate(c, v, window=7),
        "vpin_pct": lambda c, v: _op("vpin_pct").calculate(c, v, window=12, buckets=4),
        "bvc_imbalance_ma": lambda c, v: _op("bvc_imbalance_ma").calculate(c, v, fast_window=3, slow_window=10),
    }
    for name, run in cases.items():
        full = run(close, vol)
        head = run(close.iloc[:40], vol.iloc[:40])
        pd.testing.assert_frame_equal(full.iloc[:40], head, check_freq=False)


# ---------------------------------------------------------------------------
# bounds on a long random panel
# ---------------------------------------------------------------------------
def test_bounds_on_random_panel():
    close, vol = _random_panel(n=400, seed=21, vol_seed=22)
    b = _op("bvc_sign_pct").calculate(close, vol, window=20).to_numpy(float)
    assert np.nanmin(b) >= -1.0 - 1e-9
    assert np.nanmax(b) <= 1.0 + 1e-9
    v = _op("vpin_pct").calculate(close, vol, window=30, buckets=10).to_numpy(float)
    assert np.nanmin(v) >= -1e-9
    assert np.nanmax(v) <= 1.0 + 1e-9
    m = _op("bvc_imbalance_ma").calculate(close, vol, fast_window=5, slow_window=40).to_numpy(float)
    # |flow| <= volume on the cohort, and EMA is a convex combination ->
    # |numerator| <= EMA_fast(volume) -> |output| <= 1 (up to the fast/slow
    # spread crossing zero, still bounded by 1 in magnitude)
    assert np.nanmax(np.abs(m)) <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# param contracts / governance
# ---------------------------------------------------------------------------
def test_param_specs_and_governance():
    from cleaned_operators.technical.indicators_v2 import _RECURSIVE_EWM

    specs = _op("bvc_sign_pct").metadata.param_specs
    assert set(specs) == {"window"}
    assert specs["window"].min == 2 and specs["window"].dtype is int
    assert specs["window"].param_role is not None

    specs = _op("vpin_pct").metadata.param_specs
    assert set(specs) == {"window", "buckets"}
    assert specs["window"].min == 2 and specs["buckets"].min == 2
    assert specs["buckets"].dtype is int
    assert specs["buckets"].param_role is not None

    specs = _op("bvc_imbalance_ma").metadata.param_specs
    assert set(specs) == {"fast_window", "slow_window"}
    assert specs["fast_window"].min == 1 and specs["slow_window"].min == 1
    rel = _op("bvc_imbalance_ma").metadata.relational_specs
    assert rel and rel[0].expression == "fast_window < slow_window"

    # rolling-only vs stateful split
    assert "bvc_sign_pct" not in _RECURSIVE_EWM
    assert "vpin_pct" not in _RECURSIVE_EWM
    assert "bvc_imbalance_ma" in _RECURSIVE_EWM
    tags_ma = set(_op("bvc_imbalance_ma").metadata.tags or [])
    assert {"stateful", "full_replay"} <= tags_ma
    for name in BVC_NAMES:
        tags = set(_op(name).metadata.tags or [])
        assert {"causal", "pit_safe"} <= tags, name
        assert "stateful" not in tags if name != "bvc_imbalance_ma" else True


def test_below_min_windows_rejected():
    close, vol = _random_panel(n=20, seed=4)
    with pytest.raises(ValueError):
        _op("bvc_sign_pct").calculate(close, vol, window=1)
    with pytest.raises(ValueError):
        _op("vpin_pct").calculate(close, vol, window=5, buckets=1)
    with pytest.raises(ValueError):
        _op("vpin_pct").calculate(close, vol, window=1, buckets=2)
    with pytest.raises(ValueError):
        # fast >= slow is a duplicate search node (relational spec mirrors
        # the runtime guard)
        _op("bvc_imbalance_ma").calculate(close, vol, fast_window=10, slow_window=10)
    with pytest.raises(ValueError):
        _op("bvc_imbalance_ma").calculate(close, vol, fast_window=11, slow_window=10)


def test_promotion_membership_and_duplicate_skip():
    from mining.direct_use import _RELATIVE_ALPHA_OPS
    from cleaned_operators.operator_surface import (
        _INTRADAY_BVC_PACK_2026_08,
        _TECHNICAL_V2_CANONICALS,
        DAILY_FACTOR_MIGRATED,
        classify_canonical,
        daily_factor_migrated,
    )

    assert _INTRADAY_BVC_PACK_2026_08 == frozenset(BVC_NAMES)
    assert _INTRADAY_BVC_PACK_2026_08 <= _TECHNICAL_V2_CANONICALS
    assert _INTRADAY_BVC_PACK_2026_08 <= DAILY_FACTOR_MIGRATED
    assert _INTRADAY_BVC_PACK_2026_08 <= daily_factor_migrated()
    assert set(BVC_NAMES) <= _RELATIVE_ALPHA_OPS
    for name in BVC_NAMES:
        assert classify_canonical(name) == "daily", name

    # duplicate-skip assertions: the audited near-neighbours stay OFF the
    # direct-use surface (legacy proxy / research-only / panel-shaped)
    assert "micro_vpin" not in _RELATIVE_ALPHA_OPS
    assert "micro_vpin" not in daily_factor_migrated()
    assert "micro_bvc_vpin" not in _RELATIVE_ALPHA_OPS
    assert "micro_bvc_vpin" not in daily_factor_migrated()
    assert "micro_trade_imbalance" not in _RELATIVE_ALPHA_OPS
    assert "intraday_bvc_imbalance" not in _RELATIVE_ALPHA_OPS
    # the panel-level BVC aggregate keeps its own (chip-flow) promotion, NOT
    # this slice's pack
    assert "intraday_bvc_imbalance" not in _INTRADAY_BVC_PACK_2026_08


def test_legacy_micro_vpin_is_not_true_vpin():
    # estimator sanity on the audited legacy proxy: it measures volume-
    # weighted |return| intensity, so on a monotone-trending panel with large
    # per-bar moves it behaves differently from the landed true VPIN (equal-
    # volume buckets, BVC sign flow).  This documents the non-duplicate
    # rationale with an executable check.
    from cleaned_operators.registry import OperatorRegistry

    op = (
        OperatorRegistry.get("micro_vpin", "pandas_numpy")
        or OperatorRegistry.get("micro_vpin")
    )
    if op is None:
        pytest.skip("micro_vpin not loadable in this session")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = op.calculate(
            pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0, 14.0]}),
            pd.DataFrame({"A": [100.0] * 5}),
            window=5,
            min_periods=1,
        )
    # the legacy proxy is a |return|-based intensity, NOT bucketed flow
    assert float(legacy.iloc[-1, 0]) > 0.0
    true_vpin = _op("vpin_pct").calculate(
        pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0, 14.0]}),
        pd.DataFrame({"A": [100.0] * 5}),
        window=5,
        buckets=4,
    )
    assert abs(float(legacy.iloc[-1, 0]) - float(true_vpin.iloc[-1, 0])) > 1e-6
