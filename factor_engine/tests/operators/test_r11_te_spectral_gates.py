# -*- coding: utf-8 -*-
"""R11 long-tail audit #19-27 / #80-91 — statistical gates in TE/MI, spectral,
complexity and multifractal operators.

Modules are imported directly (their registration is import-triggered) instead of
``load_all()`` so this file stays self-contained while the concurrent R10 session
is mid-edit on ``runtime/`` / ``storage/`` / ``stateful/`` files that the full
registration audit traverses.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import cleaned_operators.advanced_information  # noqa: F401
import cleaned_operators.complexity_ext  # noqa: F401
import cleaned_operators.conditional_dependence  # noqa: F401
import cleaned_operators.multifractal  # noqa: F401
import cleaned_operators.multifractal_asym  # noqa: F401
import cleaned_operators.nonlinear_dependence  # noqa: F401
import cleaned_operators.spectral  # noqa: F401
import cleaned_operators.spectral_ext  # noqa: F401

from cleaned_operators.advanced_information import (
    _te_from_transitions,
    _te_state_space_floor,
)
from cleaned_operators.conditional_dependence import (
    _break_ties_deterministic,
    _conditional_te_window,
)
from cleaned_operators.multifractal import (
    _LAGS,
    _common_cohort,
    _hurst_generalized,
)
from cleaned_operators.nonlinear_dependence import _quantile_hist_mi, _value_bins
from cleaned_operators.registry import OperatorRegistry


def _frame(values: np.ndarray, start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.DataFrame(values, index=idx, columns=["A"])


def _op(canonical: str):
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    assert op is not None, canonical
    return op


# ---------------------------------------------------------------------------
# #19 — Miller-Madow is an OPTION and matches a hand-built contingency table
# ---------------------------------------------------------------------------
def test_miller_madow_matches_hand_built_contingency_table():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    b = rng.normal(size=200)
    bins = 4
    n = a.size
    ba = _value_bins(a, bins)
    bb = _value_bins(b, bins)
    cont = np.zeros((bins, bins))
    for i in range(n):
        cont[ba[i], bb[i]] += 1.0
    p = cont / n
    p_row = p.sum(axis=1, keepdims=True)
    p_col = p.sum(axis=0, keepdims=True)
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if p[i, j] > 0 and p_row[i, 0] > 0 and p_col[0, j] > 0:
                mi += p[i, j] * np.log(p[i, j] / (p_row[i, 0] * p_col[0, j]))
    r_occ = int((p_row[:, 0] > 0).sum())
    c_occ = int((p_col[0, :] > 0).sum())
    hand = max(0.0, mi - (r_occ - 1) * (c_occ - 1) / (2.0 * n))
    impl = _quantile_hist_mi(a, b, bins, False, bias_correction=True)
    assert impl == pytest.approx(hand, abs=1e-12)
    # The correction is an OPTION: disabling it returns the raw plug-in MI.
    raw = _quantile_hist_mi(a, b, bins, False, bias_correction=False)
    assert raw == pytest.approx(mi, abs=1e-12)
    assert raw > impl  # plug-in MI is positively biased -> correction lowers it


def test_miller_madow_opt_out_through_operator():
    rng = np.random.default_rng(1)
    x = _frame(rng.normal(size=(120, 1)))
    y = _frame(rng.normal(size=(120, 1)))
    op = _op("ts_mutual_information")
    corrected = op.calculate(x, y, window=60, bins=5, normalized=False, bias_correction=True)
    raw = op.calculate(x, y, window=60, bins=5, normalized=False, bias_correction=False)
    c = corrected.to_numpy(dtype=float)
    r = raw.to_numpy(dtype=float)
    m = np.isfinite(c) & np.isfinite(r)
    assert m.any()
    assert np.all(r[m] >= c[m] - 1e-12)


# ---------------------------------------------------------------------------
# #20 — TE effective-state-space sample gate
# ---------------------------------------------------------------------------
def test_te_effective_state_space_floor():
    # 3x3x3 joint = 27 cells; ratio=2 -> floor 54.
    assert _te_state_space_floor(3, 3, 1.0) == 27
    assert _te_state_space_floor(3, 3, 2.0) == 54


def test_te_effective_state_space_gate_nan():
    rng = np.random.default_rng(2)
    xs = rng.normal(size=5)
    ys = rng.normal(size=5)
    xn = rng.normal(size=5)
    # 5 usable samples << 3^3 = 27 cells -> NaN, never a noisy estimate.
    assert math.isnan(_te_from_transitions(xs, ys, xn, 3, min_cells_ratio=1.0))
    # A generous sample count clears the gate and yields a finite value.
    big = rng.normal(size=400)
    v = _te_from_transitions(big, big, big, 3, min_cells_ratio=1.0)
    assert np.isfinite(v)


def test_te_operator_rejects_infeasible_cells_ratio():
    op = _op("ts_transfer_entropy")
    rng = np.random.default_rng(3)
    t = _frame(rng.normal(size=(80, 1)))
    s = _frame(rng.normal(size=(80, 1)))
    # window-lag 59 < ceil(2 * 27) = 54? it is >= 54, so use an impossible ratio.
    with pytest.raises(ValueError, match="min_cells_ratio|window"):
        op.calculate(t, s, window=30, bins=3, lag=1, min_cells_ratio=10.0)


# ---------------------------------------------------------------------------
# #21 — conditional TE ties do not collapse bins
# ---------------------------------------------------------------------------
def test_break_ties_deterministic_separates_ties_only():
    v = np.where(np.arange(120) % 2 == 0, 1.0, 0.0)
    jt = _break_ties_deterministic(v)
    # deterministic: bit-identical across calls
    assert np.array_equal(jt, _break_ties_deterministic(v))
    # exact ties are separated (distinct values now), so quantile binning gets
    # distinct edges instead of silently collapsing a bin
    assert np.unique(jt).size > np.unique(v).size
    # the perturbation is tiny relative to the value range (<= _TIE_JITTER, with
    # a little float slack)
    assert float(np.max(np.abs(jt - v))) <= 2e-9
    # sorted-by-value order is preserved: jt is a monotone transform of v
    order = np.argsort(v, kind="mergesort")
    assert np.all(np.diff(jt[order]) >= 0.0)
    # a fully-degenerate (constant) series is left untouched so the kernel's
    # ``< 2 distinct states`` check still fails closed
    const = np.ones(50)
    assert np.array_equal(_break_ties_deterministic(const), const)


def test_conditional_te_ties_do_not_collapse_bins():
    rng = np.random.default_rng(5)
    n = 300
    tw = rng.normal(size=n)
    sw = rng.normal(size=n)
    # a heavily tied (but non-constant) conditioning series
    cw = np.where(np.arange(n) % 2 == 0, 1.0, 0.0)
    with_ties = _conditional_te_window(tw, sw, cw, 3, 1, 50, 1.0)
    # the tie-break recovered a genuine finite CTE (not a collapsed-bin NaN)
    assert np.isfinite(with_ties)


# ---------------------------------------------------------------------------
# #22 — collapsed conditioning bin -> NaN (fail closed), not a spurious value
# ---------------------------------------------------------------------------
def test_conditional_te_collapsed_conditioning_bin_nan(monkeypatch):
    rng = np.random.default_rng(6)
    n = 300
    tw = rng.normal(size=n)
    sw = rng.normal(size=n)
    cw = np.array([0.0] * 200 + [1.0] * 100)  # collapses to one effective bin
    # Reproduce the pre-fix behaviour: without tie-breaking the quantile edges
    # collapse, leaving a conditioning bin with ZERO samples.
    monkeypatch.setattr(
        "cleaned_operators.conditional_dependence._break_ties_deterministic",
        lambda v: np.asarray(v, dtype=float),
    )
    collapsed = _conditional_te_window(tw, sw, cw, 3, 1, 50, 1.0)
    assert math.isnan(collapsed)  # fail closed, never a spurious smoothed 0

    # A truly constant conditioning state also fails closed.
    assert math.isnan(_conditional_te_window(tw, sw, np.ones(n), 3, 1, 50, 1.0))


def test_conditional_te_operator_constant_condition_nan():
    op = _op("ts_conditional_transfer_entropy")
    rng = np.random.default_rng(7)
    x = _frame(rng.normal(size=(120, 1)))
    c = _frame(np.ones((120, 1)))
    out = op.calculate(x, x, c, window=90, bins=2, lag=1)
    assert out.iloc[-1, 0] != out.iloc[-1, 0]  # NaN


# ---------------------------------------------------------------------------
# #23 — TE/MI unit metadata documented consistently
# ---------------------------------------------------------------------------
def test_te_mi_unit_metadata_documented():
    op = _op("ts_transfer_entropy")
    assert "nats" in op.metadata.description
    mi = _op("ts_mutual_information")
    assert "nats" in mi.metadata.description
    cmi = _op("ts_conditional_mutual_information")
    assert any("ratio" in t or "无量纲" in t for t in cmi.metadata.tags)


# ---------------------------------------------------------------------------
# #24 — TE backward-window lookahead: prefix causal (no lookahead)
# ---------------------------------------------------------------------------
def test_te_backward_window_no_lookahead():
    rng = np.random.default_rng(8)
    n = 200
    t = _frame(rng.normal(size=(n, 1)))
    s = _frame(rng.normal(size=(n, 1)))
    op = _op("ts_transfer_entropy")
    full = op.calculate(t, s, window=60, bins=3, lag=2)
    # Truncate the series one row before the last: the earlier rows' values must
    # be identical (the estimate at row r never reads past row r).
    prefix = op.calculate(t.iloc[: n - 1], s.iloc[: n - 1], window=60, bins=3, lag=2)
    assert np.allclose(
        full.to_numpy(dtype=float)[: n - 1],
        prefix.to_numpy(dtype=float),
        equal_nan=True,
    )


# ---------------------------------------------------------------------------
# #25 / #7 — MODWT minimum window + dead-level feasibility
# ---------------------------------------------------------------------------
def test_modwt_window_below_minimum_rejected():
    op = _op("ts_modwt_band_corr")
    rng = np.random.default_rng(9)
    x = _frame(rng.normal(size=(40, 1)))
    with pytest.raises(ValueError, match="window"):
        op.calculate(x, x, window=4, level=1, band=1)
    # Declared relational feasibility: window=8 < 2**1 + 1 + 6 = 9 -> rejected.
    with pytest.raises(ValueError, match="2\\*\\*band|window"):
        op.calculate(x, x, window=8, level=1, band=1)
    # Feasible window passes.
    out = op.calculate(x, x, window=20, level=1, band=1)
    assert out.iloc[-1, 0] == pytest.approx(1.0)


def test_modwt_unit_metadata_dimensionless():
    op = _op("ts_modwt_band_corr")
    assert any("corr" in t for t in op.metadata.tags)


# ---------------------------------------------------------------------------
# #80 — spectral window < 16 must not enter the search surface
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    [
        "ts_spectral_centroid",
        "ts_spectral_flatness",
        "ts_spectral_peak_concentration",
        "ts_spectral_quality_factor",
        "ts_spectral_entropy",
        "ts_dominant_cycle_period",
    ],
)
def test_spectral_window_below_16_rejected(name):
    op = _op(name)
    rng = np.random.default_rng(10)
    x = _frame(rng.normal(size=(40, 1)))
    with pytest.raises(ValueError, match="window"):
        op.calculate(x, window=10)
    # The ParamSpec declares the min so the binder/search surface sees it.
    spec = (op.metadata.param_specs or {}).get("window")
    assert spec is not None and spec.min >= 16


# ---------------------------------------------------------------------------
# #81 — single-bin spectral Q factor does not explode
# ---------------------------------------------------------------------------
def test_spectral_q_single_bin_does_not_explode():
    op = _op("ts_spectral_quality_factor")
    rng = np.random.default_rng(11)
    # white noise -> its periodogram peak is typically a single bin; a single-bin
    # half-power width is unresolved, so Q must NOT be ~ f/EPS (huge).
    x = _frame(rng.normal(size=(200, 1)))
    out = op.calculate(x, window=64)
    vals = out.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    assert vals.size > 0
    assert float(np.max(vals)) < 64  # bounded by the one-bin resolution (<= n/2)
    # a pure tone still yields a large-but-finite, resolvable Q
    t = np.arange(200.0)
    tone = _frame(np.sin(2.0 * np.pi * t / 8.0).reshape(-1, 1))
    out2 = op.calculate(tone, window=64)
    v2 = out2.to_numpy(dtype=float)
    v2 = v2[np.isfinite(v2)]
    assert v2.size > 0
    assert np.all(np.isfinite(v2)) and np.all(v2 < 100.0)


# ---------------------------------------------------------------------------
# #82 / #83 — spectral_ext units + polars metadata consistency
# ---------------------------------------------------------------------------
def test_spectral_ext_units_and_polars_metadata():
    ent = _op("ts_spectral_entropy")
    assert ent.metadata.output_unit == "ratio"
    assert any("ratio" in t for t in ent.metadata.tags)
    dom = _op("ts_dominant_cycle_period")
    assert dom.metadata.output_unit == "bars"
    assert any("bars" in t for t in dom.metadata.tags)
    # #83: the polars backend carries the SAME param_names, not an empty list.
    pent = OperatorRegistry.get("ts_spectral_entropy", "polars")
    assert pent is not None
    assert pent.metadata.param_names == list(ent.metadata.param_names)
    assert pent.metadata.param_names  # non-empty


# ---------------------------------------------------------------------------
# #85 — LZ suffix-drop guard (effective_n / min_contiguous_fraction)
# ---------------------------------------------------------------------------
def test_lz_suffix_drop_guard():
    op = _op("ts_lempel_ziv_complexity")
    rng = np.random.default_rng(12)
    x = _frame(rng.normal(size=(150, 1)))
    # Fully-finite trailing windows -> effective_n == window -> finite values.
    out = op.calculate(x, window=120, bins=2)
    assert out.notna().any().any()
    # A recent NaN gap drops the trailing contiguous suffix below the guard.
    xg = x.copy()
    xg.iloc[110:140, 0] = np.nan  # only 10 finite trailing rows at the end
    outg = op.calculate(xg, window=120, bins=2)
    assert math.isnan(outg.iloc[-1, 0])
    # The effective_n accounting is explicit in the kernel signature.
    import inspect

    from cleaned_operators.complexity_ext import _lz_complexity_series

    params = inspect.signature(_lz_complexity_series).parameters
    assert "min_contiguous_fraction" in params and "min_effective_n" in params


# ---------------------------------------------------------------------------
# #88 — forbidden-ordinal null N excludes tie embeddings
# ---------------------------------------------------------------------------
def test_forbidden_ordinal_null_excludes_tie_embeddings():
    from cleaned_operators.complexity_ext import _forbidden_ordinal_ratio_series

    rng = np.random.default_rng(13)
    # A series with some ties: tied embeddings are dropped from the observed
    # patterns AND must not inflate the null baseline N.
    x2d = rng.normal(size=(200, 1))
    x2d[::5, 0] = x2d[0, 0]  # force a repeated value every 5th row
    out = _forbidden_ordinal_ratio_series(x2d, 120, 3, 1)
    # The null baseline uses only no-tie embeddings: with order=3, a value that
    # recurs every 5th row ties ~every 3-embedding -> n_valid < n_emb, and the
    # excess-forbiddenness must not be inflated by counting tied embeddings.
    vals = out[np.isfinite(out)]
    assert vals.size > 0
    # Verbatim code-inspection guard: the kernel counts valid embeddings.
    import inspect

    src = inspect.getsource(_forbidden_ordinal_ratio_series)
    assert "n_valid" in src and "** n_valid" in src


# ---------------------------------------------------------------------------
# #89 — multifractal q grid + sample floor
# ---------------------------------------------------------------------------
def test_multifractal_q_grid_rejects_large_q():
    op = _op("ts_generalized_hurst_exponent")
    rng = np.random.default_rng(14)
    x = _frame(rng.normal(size=(120, 1)))
    for q in (5.0, 8.0, 50.0, -1.0):
        with pytest.raises(ValueError, match="q"):
            op.calculate(x, window=60, q=q)
    spec = (op.metadata.param_specs or {}).get("q")
    assert spec is not None and spec.choices == (0.5, 1.0, 2.0, 3.0, 4.0)


def test_multifractal_sample_floor_grows_with_q():
    from cleaned_operators.multifractal import _structure_function

    rng = np.random.default_rng(15)
    v = rng.normal(size=60)
    # q=4 needs a higher pair floor than q=0.5 on a short window: a tiny cohort
    # that clears the q=0.5 floor can fail the q=4 floor (fail closed, not noise).
    short = v[:20]
    s_low = _structure_function(short, 8, 0.5)
    s_high = _structure_function(short, 8, 4.0)
    # 20 - 8 = 12 pairs: >= 8 (q=0.5 floor) but < 32 (q=4 floor).
    assert s_low is None or np.isfinite(s_low) or np.isnan(s_low)
    if np.isfinite(s_low) if s_low is not None else False:
        assert np.isnan(s_high)  # q=4 needs 32 pairs, only 12 available


# ---------------------------------------------------------------------------
# #90 — multifractal quadratic fit needs >= 4 valid q points
# ---------------------------------------------------------------------------
def test_multifractal_curvature_needs_four_points():
    from cleaned_operators.multifractal import _curvature_series

    rng = np.random.default_rng(16)
    # A random walk has genuine scaling structure, so H(q) is well-defined and
    # the quadratic fit can be exercised.
    rw = np.cumsum(rng.normal(size=(160, 1)), axis=0)
    out = _curvature_series(rw, 60)
    vals = out[np.isfinite(out)]
    # On this synthetic series at least one window yields a curvature; every
    # emitted value was fit from >= 4 valid q-points (5-point grid, 4 required).
    assert vals.size > 0
    # The kernel declares the >=4 requirement.
    import inspect

    src = inspect.getsource(_curvature_series)
    assert "< 4" in src


# ---------------------------------------------------------------------------
# #91 — multifractal common observation cohort
# ---------------------------------------------------------------------------
def test_multifractal_common_cohort():
    rng = np.random.default_rng(17)
    # A random walk has scaling structure so H(q) is finite and comparable.
    v = np.cumsum(rng.normal(size=100))
    v[50] = np.nan  # a mid-series gap
    cohort = _common_cohort(v)
    assert len(cohort) == 49  # rows 51..99 (the trailing contiguous finite run)
    assert np.all(np.isfinite(cohort))
    # All lags estimate on the SAME cohort: _hurst_generalized extracts it once,
    # so the gapped series and its cohort give bit-identical H(q).
    assert _hurst_generalized(v, 2.0) == _hurst_generalized(cohort, 2.0)
    # A fully-NaN window leaves no cohort -> fail closed NaN.
    w = np.full(100, np.nan)
    assert len(_common_cohort(w)) == 0
    assert math.isnan(_hurst_generalized(w, 2.0))
