# -*- coding: utf-8 -*-
"""R11 round-3 dependence audit — items #49..54.

* #49 — conditional TE must NOT deterministic-jitter ties (jitter removed;
  ties are treated as a categorical state).
* #50 — conditional TE joint tensor built at EFFECTIVE bins, never requested
  ``bins^4``.
* #51 — conditional TE default parameters feasible (default ``bins=2`` so
  ``window=60`` is callable; ``bins=3`` is rejected loudly unless the window is
  long enough).
* #52 — partial distance correlation is a Pearson proxy, NOT strict
  Székely/Rizzo pdCor -> renamed ``ts_distance_correlation_partial_proxy`` with
  the legacy ``ts_partial_distance_correlation`` as a deprecated alias.
* #53 — HSIC bandwidth is the median of the POSITIVE pairwise distances (a zero
  median under heavy ties must not fall back to a near-identity kernel).
* #54 — CMI normalization denominator uses log(effective_bins) after ties
  collapse the effective state space.

Modules are imported directly (registration is import-triggered) so this file
stays self-contained for the kernels it exercises.
"""
from __future__ import annotations

import inspect
import math

import numpy as np
import pandas as pd
import pytest

import factor_engine.cleaned_operators.conditional_dependence  # noqa: F401
import factor_engine.cleaned_operators.dependence_ext  # noqa: F401

from factor_engine.cleaned_operators.conditional_dependence import (
    _break_ties_deterministic,
    _conditional_te_window,
    _quantile_edges,
)
from factor_engine.cleaned_operators.dependence_ext import _cmi, _partial_dcor_proxy, _rbf_kernel
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _frame(values: np.ndarray, start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.DataFrame(values, index=idx, columns=["A"])


def _op(canonical: str, backend: str = "pandas_numpy"):
    op = OperatorRegistry.get(canonical, backend)
    assert op is not None, canonical
    return op


# ---------------------------------------------------------------------------
# #49 — conditional TE must NOT deterministic-jitter ties
# ---------------------------------------------------------------------------
def test_break_ties_deterministic_is_now_identity():
    # Audit #49: the stable-position jitter is REMOVED.  The helper survives only
    # as a documented identity for round-2 test imports.
    v = np.where(np.arange(120) % 2 == 0, 1.0, 0.0)
    assert np.array_equal(_break_ties_deterministic(v), v)
    const = np.ones(50)
    assert np.array_equal(_break_ties_deterministic(const), const)


def test_conditional_te_kernel_no_longer_calls_jitter():
    # Verbatim guard: the kernel must not perturb tied values.
    src = inspect.getsource(_conditional_te_window)
    assert "_break_ties_deterministic" not in src


def test_conditional_te_binary_condition_finite_via_effective_bins():
    # A binary conditioning series carries 2 distinct states -> 2 EFFECTIVE bins
    # (no jitter needed), so the conditional TE is finite, not a collapsed-bin
    # NaN and not a jitter artefact.
    rng = np.random.default_rng(5)
    n = 300
    tw = rng.normal(size=n)
    sw = rng.normal(size=n)
    cw = np.where(np.arange(n) % 2 == 0, 1.0, 0.0)
    assert np.unique(cw).size == 2
    assert _quantile_edges(cw, 3).size - 1 == 2  # 2 effective cells under bins=3
    val = _conditional_te_window(tw, sw, cw, 3, 1, 50, 1.0)
    assert np.isfinite(val)


def test_conditional_te_uneven_binary_condition_finite():
    # cw = 200 zeros + 100 ones: still 2 distinct states -> 2 effective cells,
    # both empirically occupied -> finite CTE (audit #49/#50 categorical state).
    rng = np.random.default_rng(6)
    n = 300
    tw = rng.normal(size=n)
    sw = rng.normal(size=n)
    cw = np.array([0.0] * 200 + [1.0] * 100)
    assert _quantile_edges(cw, 3).size - 1 == 2
    assert np.isfinite(_conditional_te_window(tw, sw, cw, 3, 1, 50, 1.0))


def test_conditional_te_constant_condition_still_fail_closed():
    rng = np.random.default_rng(7)
    n = 300
    tw = rng.normal(size=n)
    sw = rng.normal(size=n)
    # A truly constant conditioning state has a single distinct value -> the
    # categorical-state binning has no second cell -> NaN (never a smoothed 0).
    assert math.isnan(_conditional_te_window(tw, sw, np.ones(n), 3, 1, 50, 1.0))


# ---------------------------------------------------------------------------
# #50 — conditional TE joint tensor uses EFFECTIVE bins
# ---------------------------------------------------------------------------
def test_quantile_edges_binary_margin_collapses_to_two_effective_cells():
    v = np.where(np.arange(200) % 2 == 0, 1.0, 0.0)
    edges = _quantile_edges(v, 3)
    # Requested 3 bins but only 2 distinct values -> 2 effective cells
    # (categorical-state fallback), never the degenerate 1-cell collapse.
    assert edges.size - 1 == 2


def test_quantile_edges_continuous_keeps_requested_cells():
    rng = np.random.default_rng(0)
    c = rng.normal(size=300)
    assert _quantile_edges(c, 3).size - 1 == 3
    assert _quantile_edges(c, 2).size - 1 == 2


def test_conditional_te_joint_tensor_effective_dimensions():
    # Verbatim guard: the joint is sized to the EFFECTIVE cell counts
    # (nxb x nxb x nyb x ncb), never the requested ``bins``.
    src = inspect.getsource(_conditional_te_window)
    assert "np.zeros((nxb, nxb, nyb, ncb)" in src
    assert "np.zeros((bins, bins, bins, bins)" not in src


def test_conditional_te_operator_effective_bins_path():
    # Operator-level: a binary condition under bins=3 at a feasible window is
    # finite (joint built at 2x2x2x2, not 3^4).
    op = _op("ts_conditional_transfer_entropy")
    rng = np.random.default_rng(8)
    t = _frame(rng.normal(size=400))
    s = _frame(rng.normal(size=400))
    c = _frame(np.where(np.arange(400) % 2 == 0, 1.0, 0.0))
    out = op.calculate(t, s, c, window=260, bins=3, lag=1)
    assert np.isfinite(out.iloc[-1, 0])


# ---------------------------------------------------------------------------
# #51 — conditional TE default parameters must be feasible
# ---------------------------------------------------------------------------
def test_conditional_te_default_bins_is_two():
    op = _op("ts_conditional_transfer_entropy")
    assert op._calculate_series.__defaults__ is not None
    # defaults are (window=60, bins=2, lag=1, ...) — bins is the 2nd default.
    assert op._calculate_series.__defaults__[1] == 2


def test_conditional_te_default_call_is_feasible_and_finite():
    op = _op("ts_conditional_transfer_entropy")
    rng = np.random.default_rng(0)
    t = _frame(rng.normal(size=150))
    s = _frame(rng.normal(size=150))
    c = _frame(rng.normal(size=150))
    # Default window=60, bins=2 must NOT raise "infeasible" (audit #51).
    out = op.calculate(t, s, c)
    assert np.isfinite(out.iloc[-1, 0])


def test_conditional_te_default_vs_bins3_window60_raises():
    op = _op("ts_conditional_transfer_entropy")
    rng = np.random.default_rng(9)
    t = _frame(rng.normal(size=150))
    s = _frame(rng.normal(size=150))
    c = _frame(rng.normal(size=150))
    # bins=3 with the default window=60 is infeasible (3*3^4=243 > 59) and is
    # rejected loudly at the call boundary — the operator must not silently
    # return an all-NaN column.
    with pytest.raises(ValueError, match="window|bins"):
        op.calculate(t, s, c, window=60, bins=3, lag=1)


def test_conditional_te_bins3_long_window_is_callable():
    op = _op("ts_conditional_transfer_entropy")
    rng = np.random.default_rng(10)
    t = _frame(rng.normal(size=400))
    s = _frame(rng.normal(size=400))
    c = _frame(rng.normal(size=400))
    out = op.calculate(t, s, c, window=300, bins=3, lag=1)
    assert np.isfinite(out.iloc[-1, 0])


# ---------------------------------------------------------------------------
# #52 — partial distance correlation renamed honest proxy
# ---------------------------------------------------------------------------
def test_partial_distance_correlation_renamed_proxy_registered():
    # The new honest canonical is registered with both backends.
    for backend in ("pandas_numpy", "polars"):
        op = OperatorRegistry.get("ts_distance_correlation_partial_proxy", backend)
        assert op is not None, f"missing {backend} backend for proxy canonical"
    # The legacy name resolves as a deprecated alias to the new canonical.
    legacy = OperatorRegistry.get("ts_partial_distance_correlation", "pandas_numpy")
    assert legacy is not None and legacy.metadata.name == "ts_distance_correlation_partial_proxy"
    assert "ts_partial_distance_correlation" in OperatorRegistry._aliases


def test_partial_dcor_proxy_is_pearson_combination():
    # The proxy is the Pearson combination of distance correlations, documented
    # honestly as NOT the strict Székely/Rizzo pdCor.
    src = inspect.getsource(_partial_dcor_proxy)
    assert "U-centered" in src or "U-center" in src or "Hilbert" in src
    assert "proxy" in _partial_dcor_proxy.__doc__.lower()
    # It must reference the Pearson combination denominator form.
    assert "r_xz * r_xz" in src


def test_partial_dcor_proxy_control_z_reduces_dependence():
    # A constructed case: x,y both driven by z -> the proxy controls z out and is
    # closer to 0 than the raw distance correlation.
    rng = np.random.default_rng(11)
    n = 400
    z = rng.normal(size=n)
    x = 2.0 * z + rng.normal(0, 0.1, size=n)
    y = 2.0 * z + rng.normal(0, 0.1, size=n)
    v = _partial_dcor_proxy(x, y, z)
    assert np.isfinite(v)
    assert abs(v) < 0.2  # common-driver dependence is controlled out


# ---------------------------------------------------------------------------
# #53 — HSIC bandwidth with many ties: median of POSITIVE distances
# ---------------------------------------------------------------------------
def test_hsic_bandwidth_uses_positive_median():
    # [0,0,0,0,0,0,1]: median(all pairwise distances) = 0 but the series is not
    # constant.  The bandwidth must come from the positive distances so the RBF
    # kernel is NOT a near-identity matrix.
    v = np.array([0.0, 0, 0, 0, 0, 0, 1])
    K = _rbf_kernel(v)
    assert K is not None
    off = K[np.triu_indices(7, 1)]
    assert off.min() < 0.99  # off-diagonal mass is not all ~1 (identity)


def test_hsic_bandwidth_constant_series_nan():
    assert _rbf_kernel(np.ones(20)) is None


def test_hsic_tie_heavy_operator_finite():
    op = _op("ts_hsic")
    rng = np.random.default_rng(12)
    x = _frame(np.where(np.arange(200) % 10 == 0, 1.0, 0.0).astype(float))
    y = _frame(rng.normal(size=200))
    out = op.calculate(x, y, window=60)
    assert np.isfinite(out.iloc[-1, 0])


def test_hsic_bandwidth_source_uses_positive_distances():
    src = inspect.getsource(_rbf_kernel)
    assert "tri[tri > _EPS]" in src or "> _EPS" in src
    assert "median(pos)" in src or "median" in src


# ---------------------------------------------------------------------------
# #54 — CMI normalization uses effective_bins after ties collapse
# ---------------------------------------------------------------------------
def _hand_cmi_denominator_check(x, y, z, bins):
    """Recompute the plug-in CMI numerator and effective-cell counts by hand."""
    n = x.size
    from factor_engine.cleaned_operators.dependence_ext import _value_bins

    bx = _value_bins(x, bins)
    by = _value_bins(y, bins)
    bz = _value_bins(z, bins)

    def _ent(counts, total):
        p = counts[counts > 0] / float(total)
        return -float(np.sum(p * np.log(p)))

    cz = np.bincount(bz, minlength=bins).astype(float)
    hz = _ent(cz, n)
    cxz = np.zeros((bins, bins), dtype=float)
    np.add.at(cxz, (bx, bz), 1.0)
    hxz = _ent(cxz.ravel(), n)
    cyz = np.zeros((bins, bins), dtype=float)
    np.add.at(cyz, (by, bz), 1.0)
    hyz = _ent(cyz.ravel(), n)
    cxyz = np.zeros((bins, bins, bins), dtype=float)
    np.add.at(cxyz, (bx, by, bz), 1.0)
    hxyz = _ent(cxyz.ravel(), n)
    raw = hxz + hyz - hz - hxyz
    n_x_eff = int(np.unique(bx).size)
    n_y_eff = int(np.unique(by).size)
    return raw, min(n_x_eff, n_y_eff)


def test_cmi_normalization_uses_effective_bins_not_requested():
    rng = np.random.default_rng(2)
    n = 300
    z = rng.normal(size=n)
    y = rng.normal(size=n)
    # x: 150 zeros + 150 continuous values in [1,3] -> with bins=3 the tie block
    # collapses the effective state space to 2 occupied cells.
    x = np.concatenate([np.zeros(150), rng.uniform(1, 3, 150)])

    raw, eff = _hand_cmi_denominator_check(x, y, z, 3)
    assert eff == 2, f"expected 2 effective cells, got {eff}"
    impl = _cmi(x, y, z, 3)
    assert impl == pytest.approx(raw / math.log(2.0), abs=1e-12)
    assert impl != pytest.approx(raw / math.log(3.0), abs=1e-3)


def test_cmi_constant_marginal_fail_closed():
    rng = np.random.default_rng(3)
    n = 200
    z = rng.normal(size=n)
    y = rng.normal(size=n)
    x = np.zeros(n)  # constant x -> single occupied cell -> NaN
    assert math.isnan(_cmi(x, y, z, 3))


def test_cmi_operator_normalized_range():
    op = _op("ts_conditional_mutual_information")
    rng = np.random.default_rng(4)
    x = _frame(rng.normal(size=200))
    y = _frame(rng.normal(size=200))
    z = _frame(rng.normal(size=200))
    out = op.calculate(x, y, z, window=150, bins=3)
    v = out.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    assert v.size > 0
    assert np.all((v >= 0.0) & (v <= 1.0 + 1e-9))
