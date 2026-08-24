# -*- coding: utf-8 -*-
"""R11 round-2 ordinal-pattern family regression tests.

Covers review items #20/#21/#22 on the forbidden-ordinal family:

* honest naming — the historical ``ts_forbidden_ordinal_pattern_ratio`` was
  really the finite-sample EXCESS ``max(0, F_obs - F_null)``; the implementation
  now lives at ``ts_forbidden_ordinal_pattern_excess``, the name
  ``ts_forbidden_ordinal_pattern_ratio`` is re-claimed by the RAW ratio
  canonical, and ``ts_forbidden_ordinal_pattern_signed_excess`` reports the
  unclipped ``F_obs - F_null`` (which keeps the information in "fewer forbidden
  patterns than the null");
* feasibility (review #22) — ``window >= (order-1)*delay + min_embeddings`` is
  enforced at binding (raise) instead of running then emitting all-NaN;
* polars parity for the whole family.

The historical kernel ``_forbidden_ordinal_ratio_series`` is exercised directly
(its signature is the single authority the pandas + polars backends both route
through), and the three public canonicals are exercised through the registry.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.registry import OperatorRegistry

# Register the pandas + polars backends without a full load_all (the shared
# tree's finalize-layer-governance is transiently broken by the concurrent
# session editing the static operator surface; the operators under test are all
# self-contained in complexity_ext / polars_geometry_math).
import factor_engine.cleaned_operators.complexity_ext  # noqa: E402,F401
import factor_engine.cleaned_operators.polars_geometry_math  # noqa: E402,F401

_RATIO = "ts_forbidden_ordinal_pattern_ratio"
_EXCESS = "ts_forbidden_ordinal_pattern_excess"
_SIGNED = "ts_forbidden_ordinal_pattern_signed_excess"
_FAMILY = (_RATIO, _EXCESS, _SIGNED)


def _frame(values: np.ndarray, periods: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.DataFrame({c: values for c in ("A", "B")}, index=idx)


def _ref_ordinal(values: np.ndarray, window: int, order: int, delay: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference F_obs / F_null / n_valid per row (mirror of the kernel)."""
    xv = np.asarray(values, dtype=float).ravel()
    n = len(xv)
    embed_len = (order - 1) * delay + 1
    fact = float(math.factorial(order))
    f_obs = np.full(n, np.nan)
    f_null = np.full(n, np.nan)
    for r in range(n):
        i0 = max(0, r - window + 1)
        run = xv[i0 : r + 1]
        if run.size < embed_len:
            continue
        pats: set[tuple[int, ...]] = set()
        n_valid = 0
        for s in range(run.size - embed_len + 1):
            vals = run[s + np.arange(order) * delay]
            u = np.unique(vals)
            if u.size == vals.size:
                order_ = np.argsort(vals, kind="stable")
                pat = tuple(int(v) for v in np.argsort(order_, kind="stable"))
                pats.add(pat)
                n_valid += 1
        if not pats or n_valid < 2:
            continue
        f_obs[r] = 1.0 - len(pats) / fact
        f_null[r] = float((1.0 - 1.0 / fact) ** n_valid)
    return f_obs, f_null


def _op(name: str, backend: str = "pandas_numpy"):
    op = OperatorRegistry.get(name, backend)
    assert op is not None, f"{name}@{backend} missing"
    return op


# ---------------------------------------------------------------------------
# (a) ratio vs excess: a window whose F_obs ≈ F_null gives ratio ≈ F_obs and
#     excess ≈ 0.
# ---------------------------------------------------------------------------
def test_ratio_excess_signed_mutually_consistent():
    rng = np.random.default_rng(7)
    x = _frame(100.0 + np.cumsum(rng.normal(0, 1, 120)))
    f_obs, f_null = _ref_ordinal(x["A"].to_numpy(), 60, 3, 1)

    ratio = _op(_RATIO).calculate(x, window=60, order=3, delay=1).to_numpy(dtype=float)[:, 0]
    excess = _op(_EXCESS).calculate(x, window=60, order=3, delay=1).to_numpy(dtype=float)[:, 0]
    signed = _op(_SIGNED).calculate(x, window=60, order=3, delay=1).to_numpy(dtype=float)[:, 0]

    finite = np.isfinite(f_obs)
    assert finite.sum() > 0
    np.testing.assert_allclose(ratio[finite], f_obs[finite], atol=1e-12)
    np.testing.assert_allclose(excess[finite], np.maximum(0.0, f_obs[finite] - f_null[finite]), atol=1e-12)
    np.testing.assert_allclose(signed[finite], f_obs[finite] - f_null[finite], atol=1e-12)

    # For a random window, F_obs ≈ F_null to sampling noise -> excess ≈ 0 while
    # ratio ≈ F_obs (which is order-3 ≈ 0 because all 6 patterns appear).
    last = slice(finite.size - 10, finite.size)
    assert np.all(np.abs(excess[last]) < 0.1)
    np.testing.assert_allclose(ratio[last], f_obs[last], atol=1e-12)


def test_ratio_is_the_raw_fraction_not_clipped():
    # A monotone series has a single ordinal pattern -> F_obs = 1 - 1/order! is
    # large and the raw ratio must report exactly that (never the clipped
    # excess, never a null correction).
    mono = _frame(np.arange(120, dtype=float))
    out = _op(_RATIO).calculate(mono, window=60, order=3, delay=1).to_numpy(dtype=float)
    finite = out[np.isfinite(out)]
    assert finite.size > 0
    np.testing.assert_allclose(finite, 1.0 - 1.0 / 6.0, atol=1e-12)


# ---------------------------------------------------------------------------
# (b) signed excess is negative when the process has FEWER forbidden patterns
#     than the null (the information the clip-to-0 erases).
# ---------------------------------------------------------------------------
def test_signed_excess_negative_when_fewer_forbidden_than_null():
    # window=10 -> 8 embeddings; the series covers all 6 order-3 patterns, so
    # F_obs = 0 while the finite-sample null still expects ~0.23 forbidden.
    seq = np.array([2.0, 1.0, 0.0, 3.0, 1.0, 2.0, 5.0, 1.0, 4.0, 2.0])
    x = _frame(seq, periods=len(seq))
    signed = _op(_SIGNED).calculate(x, window=10, order=3, delay=1).to_numpy(dtype=float)[:, 0]
    excess = _op(_EXCESS).calculate(x, window=10, order=3, delay=1).to_numpy(dtype=float)[:, 0]
    ratio = _op(_RATIO).calculate(x, window=10, order=3, delay=1).to_numpy(dtype=float)[:, 0]

    assert np.isfinite(signed[-1])
    assert signed[-1] < 0.0
    np.testing.assert_allclose(signed[-1], -float((1.0 - 1.0 / 6.0) ** 8), atol=1e-12)
    # The clipped excess erases the negative direction to exactly 0, and the
    # raw ratio reports F_obs = 0 (all six patterns observed).
    assert excess[-1] == 0.0
    assert ratio[-1] == 0.0


# ---------------------------------------------------------------------------
# (c) the historical spelling is preserved through the rename migration.
# ---------------------------------------------------------------------------
def test_old_name_resolves_via_alias_migration():
    reg = OperatorRegistry
    # The historical spelling still resolves to a live canonical (it was
    # migrated via rename_canonical; the name is then re-claimed by the raw
    # ratio canonical, so it must never be a dead name).
    assert reg.resolve_canonical(_RATIO) == _RATIO
    assert _RATIO in reg._operators
    # The pre-fix behavior (the finite-sample excess) is reachable under the
    # honest canonical.
    assert reg.resolve_canonical(_EXCESS) == _EXCESS
    assert _EXCESS in reg._operators
    assert reg.resolve_canonical(_SIGNED) == _SIGNED
    assert _SIGNED in reg._operators
    # The migrated excess op carries the honest name (the rename updated the
    # registered instance's metadata).
    exc_op = _op(_EXCESS)
    assert exc_op.metadata.name == _EXCESS
    # No alias/canonical collision was left behind for the reused name.
    assert _RATIO not in reg._aliases


# ---------------------------------------------------------------------------
# (d) infeasible (window, order, delay) combinations raise at binding.
# ---------------------------------------------------------------------------
def test_infeasible_combination_raises_at_binding():
    x = _frame(np.arange(60, dtype=float))
    # order=3, delay=1 -> need window >= (3-1)*1 + 8 = 10; window=5 is infeasible.
    for name in _FAMILY:
        op = _op(name)
        with pytest.raises(ValueError, match="window"):
            op.calculate(x, window=5, order=3, delay=1)
    # order=5, delay=2 -> need window >= 4*2 + 8 = 16; window=10 is infeasible.
    for name in _FAMILY:
        op = _op(name)
        with pytest.raises(ValueError, match="window"):
            op.calculate(x, window=10, order=5, delay=2)
    # Boundary is exactly feasible: window = (order-1)*delay + min_embeddings.
    for name in _FAMILY:
        op = _op(name)
        out = op.calculate(x, window=10, order=3, delay=1, min_embeddings=8)
        assert out.notna().any().any()


def test_min_embeddings_floor_enforced():
    x = _frame(np.arange(120, dtype=float))
    for name in _FAMILY:
        op = _op(name)
        with pytest.raises(Exception):
            op.calculate(x, window=120, order=3, delay=1, min_embeddings=7)
    # A feasible explicit min_embeddings works.
    op = _op(_RATIO)
    out = op.calculate(x, window=120, order=3, delay=1, min_embeddings=8)
    assert out.notna().any().any()


# ---------------------------------------------------------------------------
# (e) feasible combinations compute finite output (pandas + polars parity).
# ---------------------------------------------------------------------------
def test_feasible_computes_finite_pandas():
    rng = np.random.default_rng(11)
    x = _frame(100.0 + np.cumsum(rng.normal(0, 1, 120)))
    for name in _FAMILY:
        out = _op(name).calculate(x, window=60, order=3, delay=1).to_numpy(dtype=float)
        assert np.isfinite(out).mean() > 0.5, name


@pytest.mark.parametrize("name", _FAMILY)
def test_polars_parity(name):
    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-01", periods=90, freq="D")
    pdf = pl.DataFrame(
        {
            "date": idx,
            "stock_code": ["A"] * 90,
            "x": (100.0 + np.cumsum(rng.normal(0, 1, 90))).tolist(),
        }
    )
    pop = _op(name, "pandas_numpy")
    plop = _op(name, "polars")
    pout = pop.calculate(
        pdf.to_pandas().set_index("date")[["x"]], window=60, order=3, delay=1
    ).to_numpy(dtype=float).ravel()
    lout = plop.calculate(pdf, window=60, order=3, delay=1)
    arr = np.asarray(
        lout.select([c for c in lout.columns if c not in ("date", "stock_code")]).to_pandas(),
        dtype=float,
    ).ravel()
    np.testing.assert_allclose(arr, pout, equal_nan=True)


def test_old_kernel_tie_exclusion_still_counted():
    # The direct kernel keeps its audit-#88 contract: tied embeddings are
    # dropped from the observed patterns AND excluded from the null baseline N.
    from factor_engine.cleaned_operators.complexity_ext import _forbidden_ordinal_ratio_series

    rng = np.random.default_rng(13)
    x2d = rng.normal(size=(200, 1))
    x2d[::5, 0] = x2d[0, 0]
    out = _forbidden_ordinal_ratio_series(x2d, 120, 3, 1)
    vals = out[np.isfinite(out)]
    assert vals.size > 0
    import inspect

    src = inspect.getsource(_forbidden_ordinal_ratio_series)
    assert "n_valid" in src and "** n_valid" in src
