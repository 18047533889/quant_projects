# -*- coding: utf-8 -*-
"""PERF-2 benchmark: scalar vs vectorized daily_agg at production scale.

Fixture: 4000 instruments x 240 bars/day x 20 days (A-share 1-minute grid),
top-3 whitelisted kernels:
  - intra_session_mean_reversion  (1 panel, min_finite=5)
  - intra_price_delay             (2 panels, min_finite=5)
  - intra_volume_imbalance        (2 panels, min_finite=3)
Measures wall-clock of the operator's full ``calculate()``.
Run:
    python -m factor_engine.cleaned_operators.intraday.perf_vec_bench
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.intraday import _core
from factor_engine.cleaned_operators.intraday import perf_vec_kernels as pvk
from factor_engine.cleaned_operators.intraday import true_gap_batch3 as tg


def _session_minutes(days: int, start: str = "2020-01-02") -> pd.DatetimeIndex:
    out = []
    base = pd.Timestamp(start)
    for d in range(days):
        day = base + pd.Timedelta(days=d)
        # weekend-free: skip Sat/Sun
        while day.dayofweek >= 5:
            day = day + pd.Timedelta(days=1)
        out += list(pd.date_range(day.replace(hour=9, minute=31), day.replace(hour=11, minute=30), freq="1min"))
        out += list(pd.date_range(day.replace(hour=13, minute=1), day.replace(hour=15, minute=0), freq="1min"))
    return pd.DatetimeIndex(out)


def _big_fixture(n_inst: int = 4000, days: int = 20, seed: int = 0):
    idx = _session_minutes(days)
    rng = np.random.default_rng(seed)
    n = len(idx)
    close = np.exp(np.cumsum(rng.standard_normal((n, n_inst)) * 0.001, axis=0)) * 100.0
    df = pd.DataFrame(close, index=idx, columns=[f"C{i}" for i in range(n_inst)])
    vol = rng.exponential(scale=1e6, size=(n, n_inst))
    vol = pd.DataFrame(vol, index=idx, columns=df.columns)
    return df, vol


def _time_one(fn, *frames, min_finite=2, n=2):
    """Wall-clock of daily_agg{,_two} over the large panels."""
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        if len(frames) == 1:
            _core.daily_agg(frames[0], lambda v, t: fn(v, t), min_finite=min_finite)
        else:
            _core.daily_agg_two(frames[0], frames[1], lambda a, b: fn(a, b, None), min_finite=min_finite)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> int:
    df, vol = _big_fixture(n_inst=4000, days=20)
    mem = df.memory_usage(deep=True).sum() / 1e6
    n_days = len(set(df.index.normalize()))
    print(f"fixture: {df.shape[1]} inst x {n_days} days x 240 bars = {len(df):,} rows "
          f"({mem:.1f} MB panel)")

    cases = [
        ("session_mean_reversion", tg._session_mean_reversion_kernel, (df,), 5),
        ("price_delay", tg._price_delay_kernel, (df, vol), 5),
        ("volume_imbalance", tg._volume_imbalance_kernel, (df, vol), 3),
    ]

    print(f"\n{'kernel':<22}{'scalar (s)':>14}{'vec (s)':>14}{'speedup':>10}")
    print("-" * 62)
    total_scalar = total_vec = 0.0
    for name, fn, frames, mf in cases:
        # scalar: unbind
        for f in [tg._session_mean_reversion_kernel, tg._price_delay_kernel, tg._volume_imbalance_kernel]:
            if hasattr(f, "__vec__"):
                del f.__vec__
        t_scalar = _time_one(fn, *frames, min_finite=mf)
        # vec
        pvk.bind_whitelist()
        t_vec = _time_one(fn, *frames, min_finite=mf)
        speedup = t_scalar / t_vec if t_vec > 0 else float("nan")
        total_scalar += t_scalar
        total_vec += t_vec
        print(f"{name:<22}{t_scalar:>14.3f}{t_vec:>14.3f}{speedup:>9.2f}x")

    # restore whitelist bound
    pvk.bind_whitelist()
    print("-" * 62)
    print(f"{'TOTAL':<22}{total_scalar:>14.3f}{total_vec:>14.3f}{total_scalar / total_vec:>9.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())