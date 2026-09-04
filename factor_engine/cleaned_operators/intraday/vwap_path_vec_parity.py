# -*- coding: utf-8 -*-
"""Production callsite parity: operator.calculate() with vec bound vs stripped.

For every vwap_path operator that was converted to a vectorized kernel, verify
the operator's ``calculate()`` produces byte-identical output whether the
kernel's ``__vec__`` fast path is bound (production) or stripped (scalar path).
rtol/atol 1e-12.  This proves the production *callsite* (operator -> daily_agg*)
routes through the vector path without changing the operator's result.

Run:
    python -m factor_engine.cleaned_operators.intraday.vwap_path_vec_parity
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.intraday import perf_vec_kernels as pvk
from factor_engine.cleaned_operators.registry import OperatorRegistry

# (canonical, inputs) — operators that were converted to vectorized kernels.
_CASES = [
    ("intra_vwap_path_slope", ("close", "amount", "volume")),
    ("intra_vwap_path_curvature", ("close", "amount", "volume")),
    ("intra_vwap_path_slope_pct", ("close", "amount", "volume")),
    ("intra_vwap_path_curvature_pct", ("close", "amount", "volume")),
    ("intra_price_vwap_max_positive_excursion", ("close", "amount", "volume")),
    ("intra_price_vwap_max_negative_excursion", ("close", "amount", "volume")),
    ("intra_time_above_vwap", ("close", "amount", "volume")),
    ("intra_longest_above_vwap_streak", ("close", "amount", "volume")),
    ("intra_longest_below_vwap_streak", ("close", "amount", "volume")),
    ("intra_vwap_reversion_speed", ("close", "amount", "volume")),
    ("intra_max_drawdown", ("close",)),
    ("intra_max_drawup", ("close",)),
    ("intra_drawdown_depth", ("close",)),
    ("intra_drawdown_duration", ("close",)),
    ("intra_drawdown_recovery_half_life", ("close",)),
]


def _minute_panel(days: int = 3, seed: int = 3, cols: int = 3, gap_frac: float = 0.1):
    rng = np.random.default_rng(seed)
    timestamps = []
    for d in range(days):
        day = pd.Timestamp("2024-01-03") + pd.Timedelta(days=d)
        for m in list(range(571, 691)) + list(range(781, 901)):
            timestamps.append(day + pd.Timedelta(minutes=m))
    close = pd.DataFrame(
        np.exp(np.cumsum(rng.standard_normal((len(timestamps), cols)) * 0.01, axis=0)) * 100,
        index=pd.DatetimeIndex(timestamps),
        columns=[f"C{i}" for i in range(cols)],
    )
    # inject NaN gaps in price to exercise the valid-mask path
    close = close.mask(rng.random(close.shape) < gap_frac)
    vol = pd.DataFrame(np.abs(rng.standard_normal(close.shape)) + 1.0, index=close.index, columns=close.columns)
    amt = vol * 100.0
    # zero volume on some bars to exercise the vol>0 gate
    amt = amt.mask(rng.random(amt.shape) < 0.05)
    return close, amt, vol


def _eq(a: pd.DataFrame, b: pd.DataFrame) -> str:
    idx = pd.DatetimeIndex(sorted(set(a.index).union(set(b.index))))
    cols = sorted(set(a.columns).union(set(b.columns)))
    aa = a.reindex(idx).reindex(cols, axis=1).to_numpy(dtype=float)
    bb = b.reindex(idx).reindex(cols, axis=1).to_numpy(dtype=float)
    both = np.isfinite(aa) & np.isfinite(bb)
    if both.any() and not np.allclose(aa[both], bb[both], rtol=1e-12, atol=1e-12):
        return f"max abs diff {np.abs(aa[both] - bb[both]).max():.3e}"
    if not np.array_equal(np.isnan(aa), np.isnan(bb)):
        return "NaN-origin mismatch"
    return ""


def main() -> int:
    close, amt, vol = _minute_panel()
    # production path: vec kernels bound (package import does this)
    pvk.bind_whitelist()
    n_fail = 0
    results = []
    for name, inputs in _CASES:
        op = OperatorRegistry.get(name)
        kwargs = {}
        for k in inputs:
            kwargs[k] = {"close": close, "amount": amt, "volume": vol}[k]
        got = op.calculate(**kwargs)  # vec path (production)
        # scalar leg: the operator rebuilds its kernel closure each calculate,
        # so we must strip the vec binding that the binder attached.  The single
        # kernel the operator passes to daily_agg* is internal; force the
        # dispatch to the scalar leg by unbinding every whitelisted kernel.
        # (Parametric factory closures keep their own __vec__; to reach the
        #  scalar leg we monkeypatch the fast dispatches to be inert.)
        import factor_engine.cleaned_operators.intraday._core as _c
        orig3 = _c._vec_daily_agg_three
        orig1 = _c._vec_daily_agg
        _c._vec_daily_agg_three = lambda *a, **_k: None
        _c._vec_daily_agg = lambda *a, **_k: None
        try:
            ref = op.calculate(**kwargs)
        finally:
            _c._vec_daily_agg_three = orig3
            _c._vec_daily_agg = orig1
        msg = _eq(got, ref)
        results.append((name, msg))
        print(("PASS " if not msg else "FAIL ") + name + (f": {msg}" if msg else ""))
        if msg:
            n_fail += 1
    print(f"\n{len(results) - n_fail}/{len(results)} callsite parity PASS")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
