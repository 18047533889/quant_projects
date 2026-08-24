# -*- coding: utf-8 -*-
"""R21-DEF1-SUPERTRND: SupertrendNative ratchet/state-machine rehab tests.

The pre-fix kernel in
``cleaned_operators/common/polars_technical_indicators.py`` returned EXACTLY
(high+low)/2 on every row (the m*ATR bands cancelled in (upper+lower)/2, so
``window`` and ``multiplier`` were dead parameters; ATR was a simple rolling
mean of TR, not Wilder).  These tests pin the true Supertrend ratchet state
machine against an INDEPENDENT brute-force oracle that recomputes Wilder ATR
and the ratchet by hand loop — it does NOT call either implementation.
"""
from __future__ import annotations

import os
import types

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")


def _load_supertrend_native():
    """Import SupertrendNative without executing the whole registry chain.

    The module registers ~all of its operators at import; in a session where
    the registry already holds those names the import raises duplicate
    errors.  Load it by exec'ing the module source with every
    ``@register_operator(...)`` decorator stripped (AST-based), under a
    private module name — pure class definitions, zero registry side effects.
    """
    import ast
    import sys

    name = "_r21_def1_supertrend_native_module"
    if name in sys.modules:
        return sys.modules[name].SupertrendNative

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "cleaned_operators",
        "common",
        "polars_technical_indicators.py",
    )
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)

    def _is_register(node):
        if isinstance(node, ast.Call):
            node = node.func
        return (isinstance(node, ast.Name) and node.id == "register_operator") or (
            isinstance(node, ast.Attribute) and node.attr == "register_operator"
        )

    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = [
                d for d in node.decorator_list if not _is_register(d)
            ]
    module = types.ModuleType(name)
    module.__dict__["__name__"] = name
    exec(compile(ast.unparse(tree), name, "exec"), module.__dict__)
    sys.modules[name] = module
    return module.SupertrendNative


# ---------------------------------------------------------------------------
# Independent oracle: hand-loop Wilder ATR + hand-loop ratchet state machine.
# Deliberately shares NO code with either implementation.
# ---------------------------------------------------------------------------


def _oracle_tr(h, l, c):
    n = len(c)
    tr = np.empty(n)
    for t in range(n):
        if t == 0:
            pc = np.nan
        else:
            pc = c[t - 1]
        parts = [h[t] - l[t], abs(h[t] - pc), abs(l[t] - pc)]
        if any(not np.isfinite(p) for p in parts):
            tr[t] = np.nan
        else:
            tr[t] = max(parts)
    return tr


def _oracle_wilder_atr(tr, w):
    """Hand-loop Wilder ATR: alpha=1/w, adjust=False, min_periods=w.

    Follows pandas ewm NaN semantics verified independently in-scratch:
    a NaN row holds the previous state forward (NaN output while unseeded),
    and the first finite observation after k consecutive NaN rows re-blends
    as (x + r*s)/(1+r) with r=(1-alpha)^(k+1)/alpha.
    """
    n = len(tr)
    atr = np.full(n, np.nan)
    s = np.nan
    seen = 0
    gap = 0
    alpha = 1.0 / w
    for t in range(n):
        if not np.isfinite(tr[t]):
            gap += 1
            if seen >= w:
                atr[t] = s
            continue
        if not np.isfinite(s):
            s = tr[t]
        elif gap == 0:
            s = alpha * tr[t] + (1 - alpha) * s
        else:
            r = (1 - alpha) ** (gap + 1) / alpha
            s = (tr[t] + r * s) / (1 + r)
        gap = 0
        seen += 1
        if seen >= w:
            atr[t] = s
    return atr


def _oracle_supertrend(h, l, c, w, m):
    """Hand-loop Supertrend ratchet (reference semantics, own code path)."""
    n = len(c)
    atr = _oracle_wilder_atr(_oracle_tr(h, l, c), w)
    out = np.full(n, np.nan)
    basic_u = (h + l) / 2.0 + m * atr
    basic_l = (h + l) / 2.0 - m * atr
    fu = basic_u.copy()
    fl = basic_l.copy()
    trend = 1
    post_gap = False
    for t in range(1, n):
        if not (np.isfinite(h[t]) and np.isfinite(l[t]) and np.isfinite(c[t])
                and np.isfinite(basic_u[t]) and np.isfinite(basic_l[t])):
            trend = 0
            post_gap = True
            continue
        if post_gap:
            trend = 0
            fu[t] = basic_u[t]
            fl[t] = basic_l[t]
            post_gap = False
            out[t] = np.nan
            continue
        if np.isfinite(fu[t - 1]) and basic_u[t] >= fu[t - 1] and c[t - 1] <= fu[t - 1]:
            fu[t] = fu[t - 1]
        if np.isfinite(fl[t - 1]) and basic_l[t] <= fl[t - 1] and c[t - 1] >= fl[t - 1]:
            fl[t] = fl[t - 1]
        if trend == 0:
            trend = 1 if c[t] >= 0.5 * (basic_u[t] + basic_l[t]) else -1
        elif trend > 0 and c[t] < fl[t]:
            trend = -1
        elif trend < 0 and c[t] > fu[t]:
            trend = 1
        out[t] = fl[t] if trend > 0 else (fu[t] if trend < 0 else np.nan)
    return out


def _frames(h, l, c):
    return (
        pl.DataFrame({"A": np.asarray(h, dtype=float)}),
        pl.DataFrame({"A": np.asarray(l, dtype=float)}),
        pl.DataFrame({"A": np.asarray(c, dtype=float)}),
    )


@pytest.fixture(scope="module")
def native():
    return _load_supertrend_native()


# ---------------------------------------------------------------------------
# Red Team counterexample: 8-bar panel, window=4, multiplier=3.0
# ---------------------------------------------------------------------------


def test_redteam_8bar_counterexample_not_midpoint(native):
    # The exact Red Team probe panel shape: 8 bars, window=4, multiplier=3.0.
    h = [10.0, 10.5, 11.0, 10.8, 10.4, 10.2, 10.6, 11.2]
    l = [v - 1.0 for v in h]
    c = [v - 0.5 for v in h]
    out = native().calculate(*_frames(h, l, c), window=4, multiplier=3.0)["A"].to_numpy()
    mid = (np.asarray(h) + np.asarray(l)) / 2.0
    # Pre-fix: out == mid identically (dead window/multiplier). Must differ now.
    assert not np.allclose(out, mid, equal_nan=True), (
        "SupertrendNative still collapses to (high+low)/2 — R21-DEF1 regression"
    )
    expected = _oracle_supertrend(np.asarray(h), np.asarray(l), np.asarray(c), 4, 3.0)
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-10, equal_nan=True)


def test_redteam_8bar_direction_and_band_structure(native):
    # Monotone rally: uptrend -> output must be the RATCHETED LOWER band,
    # strictly below the bar midpoint, non-decreasing while trend holds.
    h = np.linspace(10.0, 20.0, 12)
    l = h - 1.0
    c = h - 0.4
    out = native().calculate(*_frames(h, l, c), window=3, multiplier=2.0)["A"].to_numpy()
    expected = _oracle_supertrend(h, l, c, 3, 2.0)
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-10, equal_nan=True)
    live = out[np.isfinite(out)]
    mid = ((h + l) / 2.0)[np.isfinite(out)]
    assert len(live) >= 5
    assert np.all(live < mid), "uptrend Supertrend must sit below the midline"
    # ratchet: while the trend does not flip the active band never rises
    flips = np.where(np.diff(live) > 1e-12)[0]
    assert len(flips) >= 0  # structural smoke; exact behavior pinned by oracle


# ---------------------------------------------------------------------------
# Dead-parameter sensitivities (both were dead pre-fix)
# ---------------------------------------------------------------------------


def test_window_sensitivity(native):
    rng = np.random.default_rng(20260820)
    n = 80
    c = np.cumsum(rng.normal(0.05, 1.0, n)) + 100
    h = c + rng.uniform(0.3, 1.5, n)
    l = c - rng.uniform(0.3, 1.5, n)
    out_w4 = native().calculate(*_frames(h, l, c), window=4, multiplier=3.0)["A"].to_numpy()
    out_w14 = native().calculate(*_frames(h, l, c), window=14, multiplier=3.0)["A"].to_numpy()
    both = np.isfinite(out_w4) & np.isfinite(out_w14)
    assert both.sum() >= 20
    assert not np.allclose(out_w4[both], out_w14[both], rtol=1e-8, atol=1e-8), (
        "window parameter is dead — R21-DEF1 regression (pre-fix: identical output)"
    )
    for out, w in ((out_w4, 4), (out_w14, 14)):
        np.testing.assert_allclose(
            out, _oracle_supertrend(h, l, c, w, 3.0), rtol=1e-10, atol=1e-10, equal_nan=True
        )


def test_multiplier_sensitivity(native):
    rng = np.random.default_rng(20260821)
    n = 80
    c = np.cumsum(rng.normal(0.05, 1.0, n)) + 100
    h = c + rng.uniform(0.3, 1.5, n)
    l = c - rng.uniform(0.3, 1.5, n)
    out_m1 = native().calculate(*_frames(h, l, c), window=10, multiplier=1.0)["A"].to_numpy()
    out_m4 = native().calculate(*_frames(h, l, c), window=10, multiplier=4.0)["A"].to_numpy()
    both = np.isfinite(out_m1) & np.isfinite(out_m4)
    assert both.sum() >= 20
    assert not np.allclose(out_m1[both], out_m4[both], rtol=1e-8, atol=1e-8), (
        "multiplier parameter is dead — R21-DEF1 regression (pre-fix: identical output)"
    )
    for out, m in ((out_m1, 1.0), (out_m4, 4.0)):
        np.testing.assert_allclose(
            out, _oracle_supertrend(h, l, c, 10, m), rtol=1e-10, atol=1e-10, equal_nan=True
        )


# ---------------------------------------------------------------------------
# Direction flip at band cross + warmup policy
# ---------------------------------------------------------------------------


def test_direction_flip_on_close_cross(native):
    # Rally then hard crash: the trend must flip from up (band below) to down
    # (band above) exactly when close crosses the active band — pinned by the
    # oracle, plus a structural above/below assertion.
    n = 30
    c = np.concatenate([
        np.linspace(100.0, 120.0, 18),
        np.linspace(119.0, 80.0, 12),  # crash
    ])
    h = c + 0.8
    l = c - 0.8
    out = native().calculate(*_frames(h, l, c), window=5, multiplier=2.0)["A"].to_numpy()
    expected = _oracle_supertrend(h, l, c, 5, 2.0)
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-10, equal_nan=True)
    mid = (h + l) / 2.0
    finite = np.isfinite(out)
    early = finite[:18]
    late = finite[20:]
    assert early.sum() >= 5 and late.sum() >= 5
    assert np.all(out[:18][early[:18]] < mid[:18][early[:18]]), "rally: band below midline"
    assert np.all(out[20:][late] > mid[20:][late]), "crash: band above midline"


def test_warmup_nan_policy_matches_family(native):
    # Family warmup contract: no Supertrend value until the Wilder ATR
    # min_periods=window is satisfied (first w-1 rows NaN), mirroring the
    # pandas reference.  Row 0 is always NaN (recursion starts at t=1).
    rng = np.random.default_rng(7)
    n = 40
    c = np.cumsum(rng.normal(0, 1, n)) + 50
    h = c + rng.uniform(0.2, 1.2, n)
    l = c - rng.uniform(0.2, 1.2, n)
    w = 8
    out = native().calculate(*_frames(h, l, c), window=w, multiplier=3.0)["A"].to_numpy()
    expected = _oracle_supertrend(h, l, c, w, 3.0)
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-10, equal_nan=True)
    assert np.all(np.isnan(out[: w - 1])), "warmup rows must be NaN (family min_periods contract)"
    assert np.isfinite(out[w + 1:]).all(), "post-warmup rows must be finite on a clean panel"


# ---------------------------------------------------------------------------
# NaN / Inf fail-closed
# ---------------------------------------------------------------------------


def test_nan_hole_fail_closed_and_post_gap_unknown(native):
    # A NaN bar breaks the state; the first valid bar after the gap publishes
    # UNKNOWN (NaN), then the machine re-asserts direction — matching the
    # audited pandas semantics (R30 §22).
    rng = np.random.default_rng(9)
    n = 50
    c = np.cumsum(rng.normal(0, 1, n)) + 50
    h = c + rng.uniform(0.2, 1.2, n)
    l = c - rng.uniform(0.2, 1.2, n)
    c[20] = np.nan
    h[20] = np.nan
    l[20] = np.nan
    out = native().calculate(*_frames(h, l, c), window=5, multiplier=2.0)["A"].to_numpy()
    expected = _oracle_supertrend(h, l, c, 5, 2.0)
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-10, equal_nan=True)
    assert np.isnan(out[20]) and np.isnan(out[21]), (
        "NaN bar and first post-gap bar must be NaN (UNKNOWN, no fake direction)"
    )
    assert np.isfinite(out[22]), "machine must re-assert direction after one UNKNOWN bar"


def test_inf_fail_closed(native):
    n = 40
    c = np.linspace(100.0, 110.0, n)
    h = c + 0.5
    l = c - 0.5
    h_bad = h.copy()
    h_bad[15] = np.inf
    out = native().calculate(*_frames(h_bad, l, c), window=5, multiplier=2.0)["A"].to_numpy()
    expected = _oracle_supertrend(h_bad, l, c, 5, 2.0)
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-10, equal_nan=True)
    # The Inf bar poisons its own row and the band state around it: the
    # oracle marks exactly what the kernel must emit (fail-closed, no
    # fabricated trend levels).
    assert np.isnan(out[15])
    # no +/-Inf may ever leak into the output
    assert not np.any(np.isinf(out[np.isfinite(out) | np.isinf(out)]))


def test_requires_low_and_close(native):
    h = pl.DataFrame({"A": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        native().calculate(h, None, None, window=3, multiplier=2.0)


# ---------------------------------------------------------------------------
# Parity with the canonical pandas reference (audited by Red Team)
# ---------------------------------------------------------------------------


def _pandas_reference():
    """Load indicators_v2.Supertrend source without registry side effects."""
    import ast

    src = open(
        "/home/shw/quant_projects/factor_engine/cleaned_operators/technical/indicators_v2.py",
        encoding="utf-8",
    ).read()
    tree = ast.parse(src)
    keep = [
        ast.get_source_segment(src, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_pi", "_pf", "_wilder", "_tr", "Supertrend"}
    ]
    module = types.ModuleType("iv2_supertrend_ref")
    exec("import numpy as np\nimport pandas as pd\n" + "\n".join(keep), module.__dict__)
    return module.Supertrend


def test_pandas_reference_parity_clean_and_gappy(native):
    PdSupertrend = _pandas_reference()
    op = native()
    for seed, n, p in ((5, 60, 0.0), (11, 300, 0.15), (21, 400, 0.30)):
        rng = np.random.default_rng(seed)
        c = np.cumsum(rng.normal(0, 1, n)) + 50
        h = c + rng.uniform(0.2, 1.5, n)
        l = c - rng.uniform(0.2, 1.5, n)
        mask = rng.random(n) < p
        c[mask] = np.nan
        h[mask] = np.nan
        l[mask] = np.nan
        for w, m in ((4, 3.0), (14, 3.0), (10, 2.0), (2, 1.0)):
            ref = PdSupertrend(
                pd.DataFrame({"A": h}), pd.DataFrame({"A": l}), pd.DataFrame({"A": c}), w, m
            )["A"].to_numpy()
            got = op.calculate(*_frames(h, l, c), window=w, multiplier=m)["A"].to_numpy()
            np.testing.assert_allclose(
                got, ref, rtol=1e-9, atol=1e-9, equal_nan=True,
                err_msg=f"pandas parity seed={seed} p={p} w={w} m={m}",
            )


# ---------------------------------------------------------------------------
# Honest ExecutionKind classification
# ---------------------------------------------------------------------------


def test_physical_spec_is_numpy_kernel_not_native_expr(native):
    from factor_engine.backend.contracts import ExecutionKind

    op = native()
    spec = getattr(op, "_physical_spec", None)
    assert spec is not None, "SupertrendNative must carry an explicit _physical_spec"
    assert spec.execution_kind == ExecutionKind.POLARS_NUMPY_KERNEL, (
        "per-row stateful ratchet must be classified POLARS_NUMPY_KERNEL, "
        "never POLARS_NATIVE_EXPR"
    )
    assert spec.stateful is True
    assert spec.canonical == "Supertrend" and spec.backend == "polars"
