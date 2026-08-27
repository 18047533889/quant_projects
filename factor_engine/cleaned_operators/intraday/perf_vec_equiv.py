# -*- coding: utf-8 -*-
"""PERF-2 equivalence harness: vectorized vs scalar daily_agg kernels.

Random minute fixtures (incl. NaN gaps and all-NaN days) are aggregated by
both the scalar per-(instrument, day) kernel path and the PERF-2 vectorized
(day, bar, inst) fast paths, then compared cellwise with rtol/atol 1e-12.

A kernel only enters the PERF-2 whitelist after this harness passes.

Run:
    python -m factor_engine.cleaned_operators.intraday.perf_vec_equiv
or with pytest:
    pytest factor_engine/tests/test_perf_intra_vec_equiv.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.intraday import _core
from factor_engine.cleaned_operators.intraday import perf_vec_kernels as pvk
from factor_engine.cleaned_operators.intraday import true_gap_batch3 as tg


def _session_minutes(days: int = 3, start: str = "2024-01-02") -> pd.DatetimeIndex:
    out = []
    base = pd.Timestamp(start)
    for d in range(days):
        day = base + pd.Timedelta(days=d)
        out += list(pd.date_range(day.replace(hour=9, minute=31), day.replace(hour=11, minute=30), freq="1min"))
        out += list(pd.date_range(day.replace(hour=13, minute=1), day.replace(hour=15, minute=0), freq="1min"))
    return pd.DatetimeIndex(out)


def _base_panel(days: int = 3, cols=None, seed: int = 0):
    cols = cols or ["C0", "C1"]
    idx = _session_minutes(days)
    rng = np.random.default_rng(seed)
    data = np.exp(np.cumsum(rng.standard_normal((len(idx), len(cols))) * 0.001, axis=0)) * 100.0
    return pd.DataFrame(data, index=idx, columns=cols)


def _volume_panel(df, seed: int = 1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.exponential(scale=1e6, size=df.shape), index=df.index, columns=df.columns)


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


def _eq(a: pd.DataFrame, b: pd.DataFrame) -> str:
    idx = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(a.index)).union(set(pd.DatetimeIndex(b.index)))))
    cols = sorted(set(a.columns).union(set(b.columns)))
    a = a.reindex(idx).reindex(cols, axis=1)
    b = b.reindex(idx).reindex(cols, axis=1)
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    aa = a.to_numpy(dtype=float)
    bb = b.to_numpy(dtype=float)
    both = np.isfinite(aa) & np.isfinite(bb)
    if both.any() and not np.allclose(aa[both], bb[both], rtol=1e-12, atol=1e-12):
        return f"max abs diff {np.abs(aa[both] - bb[both]).max():.3e}"
    if not np.array_equal(np.isnan(aa), np.isnan(bb)):
        return "NaN-origin mismatch"
    return ""


def _scalar(fn, *frames, min_finite=2):
    """Run daily_agg{,_two} with all vec kernels unbound (pure scalar)."""
    for f in [tg._session_mean_reversion_kernel, tg._price_delay_kernel, tg._volume_imbalance_kernel]:
        if hasattr(f, "__vec__"):
            del f.__vec__
    if len(frames) == 1:
        return _core.daily_agg(frames[0], lambda v, t: fn(v, t), min_finite=min_finite)
    return _core.daily_agg_two(frames[0], frames[1], lambda a, b: fn(a, b, None), min_finite=min_finite)


def _vec(fn, *frames, min_finite=2):
    pvk.bind_whitelist()
    if len(frames) == 1:
        return _core.daily_agg(frames[0], lambda v, t: fn(v, t), min_finite=min_finite)
    return _core.daily_agg_two(frames[0], frames[1], lambda a, b: fn(a, b, None), min_finite=min_finite)


# (kernel, frames-builder, min_finite, label)
_CASES = [
    (tg._session_mean_reversion_kernel, "one", 5, "session_mean_reversion"),
    (tg._price_delay_kernel, "two", 5, "price_delay"),
    (tg._volume_imbalance_kernel, "two", 3, "volume_imbalance"),
]


def run_all(verbose: bool = True) -> list[tuple[str, str]]:
    """Run all whitelisted-kernel equivalence cases; returns [(label, msg)]."""
    results = []
    for fn, kind, mf, label in _CASES:
        # build fixtures
        if kind == "one":
            a = _inject_nan(_base_panel(days=4, seed=42), seed=7, gap_frac=0.15, blank_day=2)
            frames = (a,)
        else:
            a = _inject_nan(_base_panel(days=4, seed=42), seed=7, gap_frac=0.15, blank_day=2)
            b = _inject_nan(_volume_panel(a, seed=43), seed=8, gap_frac=0.15, blank_day=2)
            frames = (a, b)
        ref = _scalar(fn, *frames, min_finite=mf)
        got = _vec(fn, *frames, min_finite=mf)
        msg = _eq(got, ref)
        results.append((label, msg))
        if verbose:
            print(("PASS " if not msg else "FAIL ") + label + (f": {msg}" if msg else ""))
    return results


def main() -> int:
    results = run_all()
    n_pass = sum(1 for _, m in results if not m)
    print(f"\n{n_pass}/{len(results)} equivalence PASS")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())