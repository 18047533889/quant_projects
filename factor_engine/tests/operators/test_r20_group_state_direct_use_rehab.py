# -*- coding: utf-8 -*-
"""R20 group-state / prior-beta DirectUse oracles.

Slice: R20-GROUPSTATE-BETA-DIRECTUSE.

Duplicate-audit outcome (survey BEFORE implementing):
* ``peer_residual_z`` and ``within_group_rank_pct`` from the original brief
  are EXACT semantic duplicates of ``ex_self_zscore`` / ``ex_self_rank_pct``
  (cleaned_operators/technical/exself_cs_v1.py, sibling R20-EXSELF-CS
  slice) -> SKIPPED, never landed here.  This module tests that the skip
  holds (the names are NOT registered under this module's source) and that
  the sibling canonicals remain registered.
* ``beta_residual_z`` / ``beta_divergence_pct`` / ``relative_strength_group_pct``
  landed — no existing canonical combines a STRICTLY PRIOR beta fit with
  stock-specific residual standardization / |market| normalization / the
  group ex-self trailing-return gap.  Audited neighbors:
  ``rolling_beta_to_market``, ``downside_beta``, ``tail_beta``,
  ``ts_regression_slope``, ``ts_regression_intercept``, ``ts_regression_r2``,
  ``ts_multi_regression_resid_z``, ``ts_huber_regression_resid_z``,
  ``ts_ridge_regression_resid_z``, ``reg_residual_zscore``,
  ``reg_forecast_error_pct``, ``benchmark_excess_return``,
  ``ts_multi_regression_forecast_error_z``, ``group_ex_self_mean``,
  ``group_peer_beta_deviation``, ``group_zscore``, ``cs_demean``,
  ``group_neutralize`` — all remain registered and untouched.

Oracle strategy: every check is an INDEPENDENT hand-rolled reimplementation
(plain python loops / pandas, sharing no code with
cleaned_operators/technical/group_state_v1.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _ensure_group_state_chain() -> None:
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    if OperatorRegistry.lifecycle() == "frozen":
        return
    if OperatorRegistry.get("beta_residual_z", "pandas_numpy") is not None:
        return
    from factor_engine.cleaned_operators.technical import group_state_v1  # noqa: F401


@pytest.fixture(scope="module", autouse=True)
def _bootstrap():
    _ensure_group_state_chain()


def _op(name: str):
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get(name, "pandas_numpy") or OperatorRegistry.get(name)
    assert op is not None, f"{name} not registered"
    return op


ALL_NAMES = (
    "beta_residual_z",
    "beta_divergence_pct",
    "relative_strength_group_pct",
)

BETA_NAMES = ("beta_residual_z", "beta_divergence_pct")

_AUDITED_NEIGHBORS = (
    "rolling_beta_to_market",
    "tail_beta",
    "ts_regression_slope",
    "ts_regression_intercept",
    "ts_regression_r2",
    "ts_multi_regression_resid_z",
    "ts_huber_regression_resid_z",
    "ts_ridge_regression_resid_z",
    "reg_residual_zscore",
    "reg_forecast_error_pct",
    "benchmark_excess_return",
    "ts_multi_regression_forecast_error_z",
    "group_ex_self_mean",
    "group_peer_beta_deviation",
    "group_zscore",
    "cs_demean",
    "group_neutralize",
    "ex_self_zscore",   # sibling slice — the peer_residual_z duplicate
    "ex_self_rank_pct",  # sibling slice — the within_group_rank_pct duplicate
)


def _beta_panel(rows=60, cols=5, seed=42):
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.001, 0.01, size=(rows, 1)) @ np.ones((1, cols))
    mkt = pd.DataFrame(mkt, columns=[f"s{i}" for i in range(cols)])
    # per-column true beta: s0.. -> 0.5, 1.0, 1.5, 2.0, -0.5
    betas = np.array([0.5, 1.0, 1.5, 2.0, -0.5])[:cols]
    idio = rng.normal(0.0, 0.02, size=(rows, cols))
    ret = pd.DataFrame(
        mkt.to_numpy() * betas[None, :] + idio,
        index=mkt.index,
        columns=mkt.columns,
    )
    return ret, mkt


def _group_panel(rows=30, cols=9, seed=42):
    rng = np.random.default_rng(seed)
    ret = pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(rows, cols)),
        columns=[f"s{i}" for i in range(cols)],
    )
    labels = np.array(["A"] * 5 + ["B"] * 3 + ["C"], dtype=object)
    g = pd.DataFrame(
        np.tile(labels, (rows, 1)), index=ret.index, columns=ret.columns
    )
    return ret, g


# ---------------------------------------------------------------------------
# independent manual oracles (numpy.polyfit / plain python, no shared code
# with the impl — the fit is an explicitly fitted OLS (polyfit deg=1), NOT a
# re-derivation of the implementation's closed-form cov/var identity)
# ---------------------------------------------------------------------------
def _oracle_prior_beta(r_col: np.ndarray, m_col: np.ndarray, w: int, t: int):
    """Strictly-prior OLS fit r ~ 1 + m via numpy.polyfit (deg=1) on rows
    [t-w, t-1] ONLY — the signal row never enters the fit.

    Returns (alpha, beta, resid_std) with resid_std at dof = n-2
    (intercept + slope consumed), or None for a fail-closed/degenerate
    window (non-finite inputs, zero market variance, dof < 1)."""
    if t < w:
        return None
    r_fit = np.asarray([float(r_col[k]) for k in range(t - w, t)])
    m_fit = np.asarray([float(m_col[k]) for k in range(t - w, t)])
    if not (np.isfinite(r_fit).all() and np.isfinite(m_fit).all()):
        return None
    if float(np.var(m_fit)) <= 0.0:  # degenerate market variance
        return None
    beta, alpha = np.polyfit(m_fit, r_fit, deg=1)  # polyfit returns high->low
    resid = r_fit - (alpha + beta * m_fit)
    n = r_fit.size
    dof = n - 2
    if dof < 1:
        return None
    ss = float(np.dot(resid, resid))
    return float(alpha), float(beta), float((ss / dof) ** 0.5)


def _oracle_beta_stat(ret: pd.DataFrame, mkt: pd.DataFrame, w: int, stat: str) -> pd.DataFrame:
    out = np.full(ret.shape, np.nan, dtype=float)
    rv = ret.to_numpy(dtype=float)
    mv = mkt.to_numpy(dtype=float)
    for c in range(rv.shape[1]):
        for t in range(rv.shape[0]):
            r_t, m_t = rv[t, c], mv[t, c]
            if not (np.isfinite(r_t) and np.isfinite(m_t)):
                continue
            fit = _oracle_prior_beta(rv[:, c], mv[:, c], w, t)
            if fit is None:
                continue
            alpha, beta, rstd = fit
            resid = float(r_t) - (alpha + beta * float(m_t))
            if stat == "z":
                if not (np.isfinite(rstd) and rstd > 0.0):
                    continue
                out[t, c] = resid / rstd
            else:  # divergence
                denom = abs(float(m_t))
                if denom <= 0.0:
                    continue
                out[t, c] = resid / denom
    return pd.DataFrame(out, index=ret.index, columns=ret.columns)


def _oracle_group_rs(ret: pd.DataFrame, g: pd.DataFrame, w: int) -> pd.DataFrame:
    out = np.full(ret.shape, np.nan, dtype=float)
    rv = ret.to_numpy(dtype=float)
    gv = g.to_numpy()
    rows, cols = rv.shape
    for t in range(rows):
        if t < w - 1:
            continue
        trail = {}
        for c in range(cols):
            chunk = [float(rv[k, c]) for k in range(t - w + 1, t + 1)]
            if all(np.isfinite(v) for v in chunk):
                prod = 1.0
                for v in chunk:
                    prod *= (1.0 + v)
                trail[c] = prod - 1.0
        for c in range(cols):
            if c not in trail:
                continue
            lab = gv[t, c]
            peers = [trail[k] for k in range(cols)
                     if k != c and gv[t, k] == lab and k in trail]
            if len(peers) < 1:
                continue
            out[t, c] = trail[c] - (sum(peers) / len(peers))
    return pd.DataFrame(out, index=ret.index, columns=ret.columns)


@pytest.mark.parametrize(
    "name,stat", [("beta_residual_z", "z"), ("beta_divergence_pct", "div")]
)
def test_matches_manual_oracle(name, stat):
    ret, mkt = _beta_panel(rows=60, cols=5, seed=101)
    out = _op(name).calculate(ret, mkt, window=15)
    exp = _oracle_beta_stat(ret, mkt, 15, "z" if stat == "z" else "div")
    np.testing.assert_allclose(
        out.to_numpy(), exp.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True
    )
    # warmup: rows t < window have no complete prior window -> all NaN
    assert np.isnan(out.to_numpy()[:15]).all()
    assert np.isfinite(out.to_numpy()[15:]).all()


# ---------------------------------------------------------------------------
# P0-02 intercept counterexample (R20-P0-BETA-INTERCEPT):
# r = alpha + beta*m with a NONZERO alpha — pre-fix the signal-row residual
# dropped the fitted intercept (resid = r_t - beta*m_t), leaving a constant
# ~alpha bias (~0.01) in every signal row.  With the intercept included the
# exact-relation residual must be ~0.
# ---------------------------------------------------------------------------
def _intercept_panel(rows=40, seed=201):
    """r = 0.01 + 1.5*m EXACTLY (zero noise) — the prior-window OLS recovers
    alpha=0.01, beta=1.5 exactly, so signal-row residuals must vanish."""
    rng = np.random.default_rng(seed)
    m = rng.normal(0.0, 0.01, size=rows)
    mkt = pd.DataFrame(np.tile(m[:, None], (1, 2)), columns=list("ab"))
    ret = pd.DataFrame(np.tile((0.01 + 1.5 * m)[:, None], (1, 2)), columns=list("ab"))
    return ret, mkt


def test_intercept_exact_fit_divergence_residual_zero():
    # regression note: PRE-FIX this left resid ~= alpha ~= 0.01 at every
    # signal row (divergence ~= 0.01/|m_t|, an O(1) spurious signal);
    # post-fix the residual must be ~0 (|resid| < 1e-10).
    ret, mkt = _intercept_panel(rows=40, seed=201)
    w = 10
    div = _op("beta_divergence_pct").calculate(ret, mkt, window=w).to_numpy()
    arr = div[w:]
    m_arr = mkt.to_numpy()[w:]
    mask = np.abs(m_arr[:, 0]) > 1e-12  # flat-market rows are NaN by guard
    assert mask.any()
    resid = arr[:, 0][mask] * np.abs(m_arr[:, 0][mask])  # recover raw residual
    assert np.isfinite(arr[:, 0][mask]).all()
    np.testing.assert_allclose(resid, 0.0, atol=1e-10)  # pre-fix: ~= 0.01
    np.testing.assert_allclose(arr[:, 0][mask], 0.0, atol=1e-6)
    # warmup rows NaN (incomplete prior window)
    assert np.isnan(div[:w]).all()


def test_intercept_exact_fit_z_zero_resid_std_still_nan():
    # zero-noise exact relation -> prior resid_std == 0 -> z NaN (fail-closed
    # guard UNCHANGED by the intercept fix); the divergence stays finite.
    ret, mkt = _intercept_panel(rows=40, seed=202)
    z = _op("beta_residual_z").calculate(ret, mkt, window=10).to_numpy()
    assert np.isnan(z).all()


def test_intercept_counterexample_z_with_tiny_burnin_noise():
    """z-counterexample: exact relation everywhere EXCEPT one burn-in row
    carrying +1e-6 noise (so the prior-window resid_std > 0 and z is
    finite).  At the signal row whose prior window contains that row, the
    fitted (alpha, beta) deviate from (0.01, 1.5) by O(noise/n); the
    residual must be O(noise) — NOT the ~0.01 constant intercept.
    Regression note: pre-fix z ~= 0.01/resid_std ~= 30; post-fix |z| < 2."""
    rng = np.random.default_rng(203)
    rows, w = 40, 10
    m = rng.normal(0.0, 0.01, size=rows)
    r = 0.01 + 1.5 * m
    r[15] += 1e-6  # single tiny noise bar (burn-in; also in [6,15]'s windows)
    mkt = pd.DataFrame(np.tile(m[:, None], (1, 2)), columns=list("ab"))
    ret = pd.DataFrame(np.tile(r[:, None], (1, 2)), columns=list("ab"))
    z = _op("beta_residual_z").calculate(ret, mkt, window=w).to_numpy()
    # warmup + pure prior windows (rows 0..15 fit on exact rows) -> resid_std
    # == 0 -> NaN (fail-closed); ONLY rows whose prior window contains the
    # noise bar 15 (rows 16..25) have resid_std ~ 3e-7 > 0 -> finite z.
    assert np.isnan(z[:16, 0]).all()
    assert np.isfinite(z[16:26, 0]).all()  # windows containing the noise bar
    assert np.isnan(z[26, 0])  # window [16,25] pure -> resid_std == 0 -> NaN
    # residual at signal rows must be ~O(1e-6), never the ~0.01 intercept
    np.testing.assert_allclose(np.abs(z[16:26, 0]), 0.0, atol=2.0)  # pre-fix: ~30


def test_intercept_sign_preserved():
    # +delta / -delta shocks at a signal row give exactly antisymmetric
    # outputs (the intercept cancels in the difference; sign preserved)
    ret, mkt = _intercept_panel(rows=40, seed=204)
    w, t, c = 10, 25, 0
    up = ret.copy()
    dn = ret.copy()
    up.iloc[t, c] = ret.iloc[t, c] + 0.005
    dn.iloc[t, c] = ret.iloc[t, c] - 0.005
    for name in BETA_NAMES:
        a = _op(name).calculate(up, mkt, window=w).to_numpy()[t, c]
        b = _op(name).calculate(dn, mkt, window=w).to_numpy()[t, c]
        base = _op(name).calculate(ret, mkt, window=w).to_numpy()[t, c]
        if name == "beta_divergence_pct":  # z variant is NaN (resid_std==0)
            np.testing.assert_allclose(a, -b, rtol=1e-9)
            np.testing.assert_allclose(base, 0.0, atol=1e-10)


def test_relative_strength_matches_manual_oracle():
    ret, g = _group_panel(rows=30, cols=9, seed=102)
    out = _op("relative_strength_group_pct").calculate(ret, g, window=12)
    exp = _oracle_group_rs(ret, g, 12)
    np.testing.assert_allclose(
        out.to_numpy(), exp.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True
    )
    # warmup rows all NaN (rows 0..w-1 have no complete trailing window)
    arr = out.to_numpy()
    assert np.isnan(arr[:11]).all()
    assert np.isnan(arr[:, 8]).all()
    assert np.isfinite(arr[11:, :5]).all()
    assert np.isfinite(arr[11:, 5:8]).all()


# ---------------------------------------------------------------------------
# fail-closed windows: exact NaN row-range asserts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", BETA_NAMES)
def test_beta_nan_in_prior_window_poisons_only_forward(name):
    # a single NaN at row k kills exactly rows [k+1, k+window] (their prior
    # window contains k) — nothing before, nothing after
    ret, mkt = _beta_panel(rows=60, cols=5, seed=103)
    w = 10
    ret2 = ret.copy()
    ret2.iloc[30, 2] = np.nan
    out = _op(name).calculate(ret2, mkt, window=w).to_numpy()
    exp = _oracle_beta_stat(ret2, mkt, w, "z" if name == "beta_residual_z" else "div")
    np.testing.assert_allclose(out, exp.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True)
    assert np.isnan(out[31:41, 2]).all()   # prior window contains row 30
    assert np.isfinite(out[w + 1 : 30, 2]).all()  # post-warmup, pre-NaN: unaffected
    assert np.isfinite(out[41:, 2]).all()  # after: window slid past
    # other columns untouched by column 2's NaN
    base = _op(name).calculate(ret, mkt, window=w).to_numpy()
    np.testing.assert_allclose(
        np.delete(out, 2, axis=1), np.delete(base, 2, axis=1),
        rtol=1e-12, atol=0.0, equal_nan=True,
    )


@pytest.mark.parametrize("name", BETA_NAMES)
def test_beta_inf_in_window_fail_closed(name):
    ret, mkt = _beta_panel(rows=40, cols=4, seed=104)
    w = 8
    for bad in (np.inf, -np.inf):
        m2 = mkt.copy()
        m2.iloc[20, 1] = bad
        out = _op(name).calculate(ret, m2, window=w).to_numpy()
        # rows 21..28 (inclusive) of col 1 have the bad bar in the prior window
        assert np.isnan(out[21:29, 1]).all(), (name, bad)
        assert np.isfinite(out[w + 1 : 20, 1]).all()  # post-warmup pre-bad finite
        assert np.isfinite(out[29:, 1]).all()


def test_beta_nan_at_signal_row_only():
    # NaN at the signal row t (inside the fit history) -> only rows whose
    # prior window covers t die; the row itself is NaN too (r_t non-finite)
    ret, mkt = _beta_panel(rows=40, cols=4, seed=105)
    w = 8
    ret2 = ret.copy()
    ret2.iloc[25, 0] = np.nan
    out = _op("beta_residual_z").calculate(ret2, mkt, window=w).to_numpy()
    assert np.isnan(out[25, 0])            # self NaN
    assert np.isnan(out[26:34, 0]).all()   # prior window still contains 25
    assert np.isfinite(out[24, 0])
    assert np.isfinite(out[34:, 0]).all()


@pytest.mark.parametrize("name", BETA_NAMES)
def test_beta_flat_market_window_nan(name):
    # constant market return over the prior window -> var(m) == 0 -> NaN
    rows, w = 30, 10
    rng = np.random.default_rng(106)
    ret = pd.DataFrame(rng.normal(0.0, 0.01, size=(rows, 3)), columns=list("abc"))
    mkt = pd.DataFrame(rng.normal(0.0, 0.01, size=(rows, 3)), columns=list("abc"))
    m2 = mkt.copy()
    m2.iloc[10:25, :] = 0.003  # constant over rows 10..24
    out = _op(name).calculate(ret, m2, window=w).to_numpy()
    # rows whose ENTIRE prior window [t-10, t-1] is inside 10..24 -> t in 20..25
    assert np.isnan(out[20:26, :]).all()
    assert np.isfinite(out[w:20, :]).all()  # post-warmup rows before the flat stretch


def test_divergence_flat_market_signal_row_nan():
    # |m_t| == 0 -> ratio undefined -> NaN, never 0
    rng = np.random.default_rng(107)
    ret = pd.DataFrame(rng.normal(0.0, 0.01, size=(30, 3)), columns=list("abc"))
    mkt = pd.DataFrame(rng.normal(0.0, 0.01, size=(30, 3)), columns=list("abc"))
    mkt.iloc[20, 1] = 0.0
    out = _op("beta_divergence_pct").calculate(ret, mkt, window=10).to_numpy()
    assert np.isnan(out[20, 1])
    assert np.isfinite(out[20, 0]) and np.isfinite(out[20, 2])


def test_beta_perfect_fit_zero_resid_std_nan():
    # r = 1.5*m exactly over the prior window -> residual std == 0 -> z NaN
    rng = np.random.default_rng(108)
    m = rng.normal(0.0, 0.01, size=(30, 1))
    mkt = pd.DataFrame(np.tile(m, (1, 2)), columns=list("ab"))
    ret = pd.DataFrame(np.tile(1.5 * m, (1, 2)), columns=list("ab"))
    out = _op("beta_residual_z").calculate(ret, mkt, window=10).to_numpy()
    assert np.isnan(out).all()  # zero residual std everywhere (fail-closed)
    # the divergence ratio is still finite (|m_t| > 0)
    div = _op("beta_divergence_pct").calculate(ret, mkt, window=10).to_numpy()
    np.testing.assert_allclose(div[10:], 0.0, atol=1e-12)
    assert np.isfinite(div[10:]).all()


def test_group_rs_nan_peer_excluded_and_single_member_nan():
    # group C (single member, zero peers) -> NaN; a NaN peer's trail is
    # excluded from the group mean, and the NaN peer itself is NaN
    ret, g = _group_panel(rows=30, cols=9, seed=109)
    w = 10
    ret2 = ret.copy()
    ret2.iloc[0, 1] = np.nan  # col 1 (group A): trail NaN for rows 0..9
    out = _op("relative_strength_group_pct").calculate(ret2, g, window=w).to_numpy()
    exp = _oracle_group_rs(ret2, g, w)
    np.testing.assert_allclose(out, exp.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True)
    assert np.isnan(out[:10, 1]).all()  # own window incomplete (NaN inside)
    assert np.isnan(out[:, 8]).all()    # group C: zero peers


# ---------------------------------------------------------------------------
# window guard: ValueError below spec min; ParamSpec/runtime equality
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL_NAMES)
def test_window_below_minimum_rejected(name):
    ret, mkt = _beta_panel(rows=20, cols=3, seed=110)
    ret_g, g = _group_panel(rows=20, cols=9, seed=111)
    # bools are rejected too (review P2: contract claims it, now pinned)
    for bad in (4, 3, 2, 1, 0, -5, 4.5, 5.5, True, False):
        with pytest.raises(ValueError):
            if name == "relative_strength_group_pct":
                _op(name).calculate(ret_g, g, window=bad)
            else:
                _op(name).calculate(ret, mkt, window=bad)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_param_specs_and_governance(name):
    specs = _op(name).metadata.param_specs
    assert "window" in specs, name
    spec = specs["window"]
    assert spec.dtype is int, name
    assert spec.min == 5, name
    assert spec.param_role is not None, name
    from factor_engine.cleaned_operators.base import ParamRole

    assert spec.param_role is ParamRole.HORIZON, name
    tags = _op(name).metadata.tags
    assert "causal" in tags and "pit_safe" in tags, name


@pytest.mark.parametrize("name", ALL_NAMES)
def test_param_spec_and_runtime_guard_equality(name):
    # runtime guard minimum EQUALS the ParamSpec min (5): window=5 passes,
    # window=4 raises — asserted on live calls, not just on paper
    ret, mkt = _beta_panel(rows=30, cols=3, seed=112)
    ret_g, g = _group_panel(rows=30, cols=9, seed=113)
    spec_min = _op(name).metadata.param_specs["window"].min
    assert spec_min == 5
    if name == "relative_strength_group_pct":
        _op(name).calculate(ret_g, g, window=5)  # no raise
        with pytest.raises(ValueError):
            _op(name).calculate(ret_g, g, window=5 - 1)
    else:
        _op(name).calculate(ret, mkt, window=5)  # no raise
        with pytest.raises(ValueError):
            _op(name).calculate(ret, mkt, window=5 - 1)


# ---------------------------------------------------------------------------
# causality: strictly prior fit — mutation cannot touch earlier rows
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL_NAMES)
def test_causality_mutation_does_not_touch_earlier_rows(name):
    """Mutating row t must not change any output row < t (prefix invariance).

    For the prior-beta family the mutated row is legitimately visible at
    t+1..t+window (their prior fit windows contain t); for the group
    relative strength the mutation is visible at t..t+window-1 (trailing
    return window includes t).  In ALL cases rows strictly before t are
    bit-identical."""
    if name == "relative_strength_group_pct":
        ret, g = _group_panel(rows=40, cols=9, seed=114)
        w = 10
        ret2 = ret.copy()
        ret2.iloc[25, 3] = ret.iloc[25, 3] + 0.05
        a = _op(name).calculate(ret, g, window=w)
        b = _op(name).calculate(ret2, g, window=w)
    else:
        ret, mkt = _beta_panel(rows=40, cols=5, seed=115)
        w = 10
        ret2 = ret.copy()
        ret2.iloc[25, 2] = ret.iloc[25, 2] + 0.05
        a = _op(name).calculate(ret, mkt, window=w)
        b = _op(name).calculate(ret2, mkt, window=w)
    pd.testing.assert_frame_equal(a.iloc[:25], b.iloc[:25])  # earlier rows frozen
    # non-vacuous: some later row DID move
    moved = (a.to_numpy()[25:] != b.to_numpy()[25:]) | (
        np.isnan(a.to_numpy()[25:]) != np.isnan(b.to_numpy()[25:])
    )
    assert moved.any(), name


@pytest.mark.parametrize("name", ALL_NAMES)
def test_prefix_invariance(name):
    # computing on a truncated panel equals the head of the full-panel run
    if name == "relative_strength_group_pct":
        ret, g = _group_panel(rows=35, cols=9, seed=116)
        full = _op(name).calculate(ret, g, window=10)
        head = _op(name).calculate(ret.iloc[:20], g.iloc[:20], window=10)
    else:
        ret, mkt = _beta_panel(rows=35, cols=5, seed=117)
        full = _op(name).calculate(ret, mkt, window=10)
        head = _op(name).calculate(ret.iloc[:20], mkt.iloc[:20], window=10)
    pd.testing.assert_frame_equal(full.iloc[:20], head)


@pytest.mark.parametrize("name", BETA_NAMES)
def test_beta_strictly_prior_fit_no_contemporaneous_leak(name):
    """The signal row NEVER enters its own beta fit: with an adversarial
    outlier at row t, beta-implied outputs at t must match the oracle fitted
    WITHOUT row t (an in-sample fit would shrink beta toward the outlier)."""
    rng = np.random.default_rng(118)
    rows, w, c = 40, 12, 2
    m = rng.normal(0.0, 0.01, size=rows)
    r = 1.2 * m + rng.normal(0.0, 0.005, size=rows)
    mkt = pd.DataFrame(np.tile(m[:, None], (1, 3)), columns=list("abc"))
    ret = pd.DataFrame(np.tile(r[:, None], (1, 3)), columns=list("abc"))
    ret.iloc[w, c] = 5.0  # huge outlier at the first signal row
    out = _op(name).calculate(ret, mkt, window=w)
    exp = _oracle_beta_stat(ret, mkt, w, "z" if name == "beta_residual_z" else "div")
    np.testing.assert_allclose(
        out.to_numpy(), exp.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True
    )
    # the oracle itself fits on [t-w, t-1] only — the assertion above pins it
    assert np.isfinite(out.to_numpy()[w, c])


# ---------------------------------------------------------------------------
# duplicate-audit skips + neighbor separation
# ---------------------------------------------------------------------------
def test_skipped_duplicates_not_landed_here():
    """peer_residual_z / within_group_rank_pct are EXACT duplicates of the
    sibling exself_cs_v1 canonicals -> they must NOT exist under this
    module's registration (no shadow registration, no second canonical)."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    for skipped in ("peer_residual_z", "within_group_rank_pct"):
        op = (
            OperatorRegistry.get(skipped, "pandas_numpy")
            or OperatorRegistry.get(skipped)
        )
        if op is not None:
            # if someone someday lands it, it must not be THIS module's
            assert op.metadata.name != skipped or "group_state_v1" not in str(
                op.__class__.__module__
            )
        # the sibling canonical covering the semantics stays registered
        assert OperatorRegistry.get("ex_self_zscore", "pandas_numpy") is not None
        assert OperatorRegistry.get("ex_self_rank_pct", "pandas_numpy") is not None


@pytest.mark.parametrize("name", ALL_NAMES)
def test_neighbor_canonicals_untouched(name):
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    for nb in _AUDITED_NEIGHBORS:
        found = (
            OperatorRegistry.get(nb, "pandas_numpy") is not None
            or OperatorRegistry.get(nb, "polars") is not None
            or OperatorRegistry.get(nb, mode="any") is not None
        )
        assert found, f"audited neighbor {nb} disappeared"
    assert name not in _AUDITED_NEIGHBORS
    # review P1: the old `[:0]` slice was empty — a tautology.  Assert the
    # duplicate-skip names are absent from the FULL audited list here and
    # remain unlanded in this module (sibling ex_self owns their semantics).
    assert "peer_residual_z" not in _AUDITED_NEIGHBORS
    assert "within_group_rank_pct" not in _AUDITED_NEIGHBORS
    for skipped in ("peer_residual_z", "within_group_rank_pct"):
        op = OperatorRegistry.get(skipped, "pandas_numpy") or OperatorRegistry.get(
            skipped, mode="any"
        )
        if op is not None:  # pragma: no cover - landed someday elsewhere: fine
            assert "group_state_v1" not in str(op.__class__.__module__)


def test_beta_level_neighbors_still_exist_as_levels():
    """rolling_beta_to_market / ts_regression_slope remain beta LEVEL
    estimators (contemporaneous windows) — this module landed residual
    SIGNALS, not beta levels; the distinction is pinned by registration."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    for lvl in ("rolling_beta_to_market", "ts_regression_slope"):
        op = (
            OperatorRegistry.get(lvl, "pandas_numpy")
            or OperatorRegistry.get(lvl, mode="any")
        )
        assert op is not None, lvl


def test_group_rs_excludes_self_and_uses_loo_mean():
    # hand-check on a tiny panel: group A of 3, window 5, row t=5
    rng = np.random.default_rng(119)
    vals = rng.normal(0.001, 0.005, size=(6, 3))
    ret = pd.DataFrame(vals, columns=["a", "b", "c"])
    g = pd.DataFrame(np.tile(np.array(["A"] * 3, dtype=object), (6, 1)),
                     index=ret.index, columns=ret.columns)
    out = _op("relative_strength_group_pct").calculate(ret, g, window=5).to_numpy()
    trails = [float(np.prod(1.0 + vals[1:6, j]) - 1.0) for j in range(3)]
    for j in range(3):
        peers = [trails[k] for k in range(3) if k != j]
        expected = trails[j] - sum(peers) / 2.0
        np.testing.assert_allclose(out[5, j], expected, rtol=1e-12)
    # zero-sum LOO gap: the three gaps sum to zero exactly
    np.testing.assert_allclose(out[5].sum(), 0.0, atol=1e-12)
    # warmup rows NaN (trailing window of 5 needs rows 0..4 -> first signal at t=4)
    assert np.isnan(out[:4]).all()
    assert np.isfinite(out[4]).all()


def test_group_rs_zero_sum_over_full_group():
    # LOO mean gaps over a group of m members sum to exactly zero per row
    ret, g = _group_panel(rows=25, cols=9, seed=120)
    out = _op("relative_strength_group_pct").calculate(ret, g, window=8).to_numpy()
    # group A = cols 0..4 (5 finite members): row sums exactly zero
    np.testing.assert_allclose(out[8:, :5].sum(axis=1), 0.0, atol=1e-10)
    # group B = cols 5..7
    np.testing.assert_allclose(out[8:, 5:8].sum(axis=1), 0.0, atol=1e-10)


def test_shape_mismatch_raises():
    ret, mkt = _beta_panel(rows=20, cols=4, seed=121)
    bad = mkt.iloc[:, :3]
    with pytest.raises(Exception):
        _op("beta_residual_z").calculate(ret, bad, window=10)
    ret_g, g = _group_panel(rows=20, cols=9, seed=122)
    bad_g = g.iloc[:, :5]
    with pytest.raises(Exception):
        _op("relative_strength_group_pct").calculate(ret_g, bad_g, window=10)
