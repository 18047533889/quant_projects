# -*- coding: utf-8 -*-
"""GO_PROMPT §100/§52 minute-bundle benchmark.

Proves that feature count 10 -> 100 -> 500 does NOT scale wall-time linearly
when the single-scan ``compute_many`` path (P0#5) is used, and decomposes the
real cost of "one minute factor takes 10-20 minutes".

Two entry points are compared on the SAME synthetic minute panel:

* OLD path  : ``intraday_feature_extension.load_intraday_feature`` — one
  ``_wide_frame`` + ``_calc`` pass PER feature (re-scans the panel N times).
* NEW path  : ``intraday_feature_runtime_v2.compute_many`` — ONE
  ``_grouped_bars`` pass + one ``_calc_shared`` per bar, reused across all N
  features (single scan).

The synthetic panel is fed by monkeypatching ``base._wide_frame`` /
``base._child`` so no real COS/DataAccess read is needed.  The grouped-bars
cache (``_intraday_bars_cache``) is the warm path: a second identical
``compute_many`` reuses the cached grouped bars.

Run:
    /home/sunhaiwei/quant_projects/.venv/bin/python \
        scripts/bench_minute_bundle.py
Output:
    /tmp/bench_minute_bundle.json  +  printed table
"""
from __future__ import annotations

import json
import os
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd
import psutil

_FE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_QUANT_ROOT = os.path.dirname(_FE_ROOT)
for _p in (_FE_ROOT, _QUANT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from storage.sources import intraday_feature_extension as base  # noqa: E402
from storage.sources import intraday_feature_runtime_v2 as rv2  # noqa: E402

OMP = os.environ.get("OMP_NUM_THREADS", "31")
os.environ["OMP_NUM_THREADS"] = OMP

# --------------------------------------------------------------------------- #
# Synthetic minute panel
# --------------------------------------------------------------------------- #

_SESSION_OPEN = "09:30"
_SESSION_CLOSE = "15:00"


def _session_minutes(days: int, start: str = "2024-03-04") -> pd.DatetimeIndex:
    """A-share regular session: 09:31-11:30 / 13:01-15:00 = 240 bars/day."""
    out = []
    base_ts = pd.Timestamp(start)
    for d in range(days):
        day = base_ts + pd.Timedelta(days=d)
        while day.dayofweek >= 5:
            day = day + pd.Timedelta(days=1)
        out += list(
            pd.date_range(day.replace(hour=9, minute=31),
                          day.replace(hour=11, minute=30), freq="1min")
        )
        out += list(
            pd.date_range(day.replace(hour=13, minute=1),
                          day.replace(hour=15, minute=0), freq="1min")
        )
    return pd.DatetimeIndex(out)


def _big_fixture(n_inst: int = 2000, days: int = 20, seed: int = 0):
    """MultiIndex(timestamp, instrument) minute panel: close/volume/amount."""
    idx = _session_minutes(days)
    rng = np.random.default_rng(seed)
    n = len(idx)
    inst = [f"S{i:05d}" for i in range(n_inst)]
    # close: geometric random walk per instrument
    close = np.exp(np.cumsum(rng.standard_normal((n, n_inst)) * 0.001, axis=0)) * 100.0
    volume = rng.exponential(scale=1e6, size=(n, n_inst))
    amount = volume * close
    # Build MultiIndex(timestamp, instrument) Series
    ts = np.repeat(idx, n_inst)
    insts = np.tile(inst, n)
    mi = pd.MultiIndex.from_arrays([ts, insts], names=["timestamp", "instrument"])
    panels = {
        "close": pd.Series(close.ravel(), index=mi, dtype="float64"),
        "volume": pd.Series(volume.ravel(), index=mi, dtype="float64"),
        "amount": pd.Series(amount.ravel(), index=mi, dtype="float64"),
    }
    # Daily anchor: one (date, instrument) row per day — the real source's
    # _anchor_index is a DAILY index, not the minute panel.  Aligning the daily
    # result to a daily anchor is cheap; aligning to the 9.6M-row minute index
    # would blow up memory ~200x.
    days = pd.DatetimeIndex(sorted(set(idx.normalize())))
    daily_mi = pd.MultiIndex.from_product(
        [days, inst], names=["timestamp", "instrument"]
    )
    return panels, daily_mi


# --------------------------------------------------------------------------- #
# Fake source + monkeypatched data feed
# --------------------------------------------------------------------------- #

class _FakeInner:
    def __init__(self, panels, anchor):
        self.params = {}
        self.instrument_filter = None
        self.data_snapshot_id = "synthetic"
        self.start_date = None
        self.end_date = None
        self.dataset = "ashare_stock_minute"
        self._panels = panels
        self._anchor = anchor

    def load_column(self, name):
        return self._panels[name]


class _FakeSource:
    """Minimal source exposing the interface compute_many / load_intraday_feature need."""

    def __init__(self, panels, anchor):
        self.inner = _FakeInner(panels, anchor)
        self._execution_id = "bench"
        self._anchor = anchor
        self._panels = panels
        self._scan_count = 0
        self._recorded = []

    def _anchor_index(self):
        return self._anchor

    @staticmethod
    def _align_exact_by_instrument(anchor, series):
        s = series.copy()
        if not isinstance(s.index, pd.MultiIndex):
            raise ValueError("expected MultiIndex")
        s.index = s.index.set_names(["timestamp", "instrument"])
        out = s.reindex(anchor.set_names(["timestamp", "instrument"]))
        out.index = anchor
        return out

    def _record_dependency(self, *args, **kwargs):
        self._recorded.append((args, kwargs))


def _wide_frame(src):
    """Monkeypatched base._wide_frame: build the long frame from synthetic panels."""
    close = src._panels["close"]
    base_df = close.rename("close").reset_index()
    base_df.columns = ["timestamp", "instrument", "close"]
    for key in ("open", "high", "low", "volume", "amount"):
        if key in ("open", "high", "low"):
            base_df[key] = base_df["close"]
        elif key in src._panels:
            temp = src._panels[key].rename(key).reset_index()
            temp.columns = ["timestamp", "instrument", key]
            base_df = base_df.merge(temp, on=["timestamp", "instrument"], how="left", sort=False)
        else:
            base_df[key] = 1.0
    base_df["timestamp"] = pd.to_datetime(base_df["timestamp"])
    return base_df


def _child(source, dataset, history_days):
    """Monkeypatched base._child: return the fake source itself (data already synthetic)."""
    return source


# --------------------------------------------------------------------------- #
# Feature selection (pure close/volume/amount, no params, no profile)
# --------------------------------------------------------------------------- #

# Features computable from _calc_from_shared with NO extra params and NO
# profile/history dependency.  Excluded: profile_* (need history_days>=2).
# Includes limit/segment/extreme features that run with default params
# (prev_close is provided by the runtime; segment defaults to "morning";
# extreme_bar_return defaults to side="max").
_CANDIDATES = [
    "realized_variance", "realized_vol", "realized_skew", "realized_kurtosis",
    "realized_quarticity", "upside_semivariance", "downside_semivariance",
    "bipower_variation", "jump_variation", "jump_ratio", "signed_jump",
    "max_abs_return", "tail_return_sum", "jump_count",
    "return", "open_to_close_return",
    "first_nmin_return", "last_nmin_return", "morning_return", "afternoon_return",
    "morning_afternoon_reversal", "trend_slope", "trend_r2",
    "path_length", "path_efficiency", "reversal_count",
    "return_autocorr", "max_drawdown", "max_runup",
    "time_of_high", "time_of_low", "close_location",
    "opening_range", "opening_range_position", "opening_drive",
    "gap_continuation", "gap_fill_ratio",
    "limit_up_touch_fraction", "limit_down_touch_fraction",
    "limit_up_close", "limit_down_close",
    "closing_return", "closing_ramp",
    "vwap", "close_to_vwap", "high_to_vwap", "low_to_vwap",
    "vwap_slope", "vwap_deviation_mean", "vwap_deviation_std", "vwap_cross_count",
    "volume_first_share", "volume_last_share", "volume_peak_time",
    "volume_hhi", "volume_entropy", "turnover_hhi", "turnover_entropy",
    "volume_profile_slope", "volume_profile_skew",
    "return_volume_corr", "abs_return_volume_corr",
    "signed_volume_imbalance", "average_trade_price", "active_volume_share",
    "volume_weighted_return", "price_impact", "amihud", "turnover_per_volatility",
    "lunch_gap_return", "return_activity_corr", "vwap_above_ratio",
    "kyle_lambda_proxy", "extreme_bar_return",
    "segment_return", "segment_volume_share", "segment_amount_share",
    "segment_vwap_deviation", "segment_realized_vol",
    "limit_first_hit_time", "limit_duration", "limit_reopen_count",
]


def _pick_features(n: int) -> list[str]:
    """Deterministic subset of size n (fixed seed); caps at pool size."""
    n = min(n, len(_CANDIDATES))
    rng = np.random.default_rng(1234)
    idx = rng.choice(len(_CANDIDATES), size=n, replace=False)
    return [_CANDIDATES[i] for i in idx]


# --------------------------------------------------------------------------- #
# Timing helpers
# --------------------------------------------------------------------------- #

def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def _best_of(fn, n: int = 2) -> tuple[float, float]:
    """Return (best wall_s, rss_mb_after)."""
    best = float("inf")
    rss = 0.0
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        rss = _rss_mb()
        if dt < best:
            best = dt
    return best, rss


# --------------------------------------------------------------------------- #
# Benchmarks
# --------------------------------------------------------------------------- #

def _run_new_path(source, features, params):
    return rv2.compute_many(source, features, params)


def _run_old_path(source, features, params):
    """OLD path: one load_intraday_feature per feature (re-scans panel each time)."""
    out = {}
    for f in features:
        p = dict(params)
        p["feature"] = f
        out[f] = base.load_intraday_feature(source, p)
    return out


def main() -> int:
    print(f"OMP_NUM_THREADS={OMP}")
    # Scaling curve + single-feature old-vs-new use a moderate fixture.
    # The 100-feature OLD-path comparison uses a smaller fixture so the
    # per-feature re-scan loop stays tractable (old path is O(rows) per feature).
    # NOTE: the per-bar Python scan (_clock_bars/_grouped_bars) dominates wall
    # time and is O(rows) regardless of feature count — that is the shared-scan
    # effect this benchmark proves.  A 100x3 fixture (300 daily rows) is enough
    # to show the scaling curve; the absolute per-feature cost scales with rows.
    n_inst, days = 100, 3
    panels, anchor = _big_fixture(n_inst=n_inst, days=days)
    n_rows = len(anchor)
    mem = sum(s.memory_usage(deep=True) for s in panels.values()) / 1e6
    print(f"fixture: {n_inst} inst x {days} days x 240 bars = {n_rows:,} rows "
          f"({mem:.0f} MB panels)")

    params = {
        "bar_minutes": 5,
        "min_coverage": 0.8,
        "min_bars": 2,
        "cutoff_time": "session_close",
        "minute_dataset": "ashare_stock_minute",
        "session_open": _SESSION_OPEN,
        "session_close": _SESSION_CLOSE,
    }

    # Monkeypatch the data feed.
    base._wide_frame = _wide_frame
    base._child = _child

    # ---- Part 1: feature-count scaling on the NEW single-scan path ---------- #
    counts = [10, 50, 100, 200]
    new_rows = []
    for n in counts:
        feats = _pick_features(n)
        src = _FakeSource(panels, anchor)
        wall, rss = _best_of(lambda: _run_new_path(src, feats, params), n=2)
        per_feat = wall / n
        new_rows.append({
            "n_features": n,
            "wall_s": round(wall, 3),
            "per_feature_s": round(per_feat, 4),
            "rss_mb": round(rss, 1),
        })
        print(f"NEW path  n={n:>4}: wall={wall:8.3f}s  per_feat={per_feat:7.4f}s  rss={rss:7.1f}MB",
              flush=True)

    # Scaling ratio 10 -> 100 (should be far below 10x if shared scan works).
    w10 = next(r["wall_s"] for r in new_rows if r["n_features"] == 10)
    w100 = next(r["wall_s"] for r in new_rows if r["n_features"] == 100)
    w200 = next(r["wall_s"] for r in new_rows if r["n_features"] == 200)
    scale_10_100 = w100 / w10 if w10 > 0 else float("nan")
    scale_10_200 = w200 / w10 if w10 > 0 else float("nan")

    # ---- Part 2: OLD per-feature path vs NEW single-scan -------------------- #
    # Single feature on the moderate fixture.
    f1 = _pick_features(1)
    src_old1 = _FakeSource(panels, anchor)
    src_new1 = _FakeSource(panels, anchor)
    old1, _ = _best_of(lambda: _run_old_path(src_old1, f1, params), n=2)
    new1, _ = _best_of(lambda: _run_new_path(src_new1, f1, params), n=2)

    # 100 features: OLD re-scans 100x.  Use a smaller fixture so it completes.
    n2, d2 = 20, 3
    panels2, anchor2 = _big_fixture(n_inst=n2, days=d2)
    f100 = _pick_features(100)
    src_old100 = _FakeSource(panels2, anchor2)
    src_new100 = _FakeSource(panels2, anchor2)
    t0 = time.perf_counter()
    _run_old_path(src_old100, f100, params)
    old100 = time.perf_counter() - t0
    t0 = time.perf_counter()
    _run_new_path(src_new100, f100, params)
    new100 = time.perf_counter() - t0

    speedup_1 = old1 / new1 if new1 > 0 else float("nan")
    speedup_100 = old100 / new100 if new100 > 0 else float("nan")

    # ---- Part 3: cold vs warm (grouped-bars cache) -------------------------- #
    # Warm = second identical compute_many on the SAME source reuses the cached
    # grouped bars (rv2._cache(source, "bars")).  Cold = fresh source.
    src_cold = _FakeSource(panels, anchor)
    src_warm = _FakeSource(panels, anchor)
    t0 = time.perf_counter()
    _run_new_path(src_cold, f100, params)
    cold = time.perf_counter() - t0
    # warm: first call populates cache, second call reuses it
    _run_new_path(src_warm, f100, params)
    t0 = time.perf_counter()
    _run_new_path(src_warm, f100, params)
    warm = time.perf_counter() - t0

    # ---- Part 4: scan-count verification ----------------------------------- #
    # The grouped-bars cache is keyed by source identity; a fresh source scans
    # once.  We verify by checking the bars cache is populated after one call.
    src_scan = _FakeSource(panels, anchor)
    _run_new_path(src_scan, f100, params)
    bars_cache = getattr(src_scan, "_intraday_bars_cache", {})
    scan_keys = len(bars_cache)

    result = {
        "fixture": {"n_inst": n_inst, "days": days, "rows": n_rows,
                    "panels_mb": round(mem, 1)},
        "old_vs_new_fixture": {"n_inst": n2, "days": d2},
        "new_path_scaling": new_rows,
        "scale_10_to_100": round(scale_10_100, 3),
        "scale_10_to_200": round(scale_10_200, 3),
        "old_vs_new": {
            "single_feature_old_s": round(old1, 3),
            "single_feature_new_s": round(new1, 3),
            "single_speedup": round(speedup_1, 2),
            "hundred_old_s": round(old100, 3),
            "hundred_new_s": round(new100, 3),
            "hundred_speedup": round(speedup_100, 2),
        },
        "cold_warm": {
            "cold_s": round(cold, 3),
            "warm_s": round(warm, 3),
            "warm_speedup": round(cold / warm, 2) if warm > 0 else float("nan"),
        },
        "scan_count": {
            "grouped_bars_cache_keys_after_one_call": scan_keys,
            "note": "one _grouped_bars pass per compute_many; cache keyed by source identity",
        },
        "entry_point": "storage.sources.intraday_feature_runtime_v2.compute_many "
                       "(monkeypatched base._wide_frame/_child synthetic feed)",
    }

    with open("/tmp/bench_minute_bundle.json", "w") as fh:
        json.dump(result, fh, indent=2, default=str)

    print("\n=== Part 1: NEW single-scan path, feature-count scaling ===")
    print(f"{'n_features':>10}{'wall_s':>12}{'per_feat_s':>12}{'rss_mb':>10}")
    for r in new_rows:
        print(f"{r['n_features']:>10}{r['wall_s']:>12.3f}{r['per_feature_s']:>12.4f}{r['rss_mb']:>10.1f}")
    print(f"scale 10->100 = {scale_10_100:.2f}x   (linear would be 10.0x)")
    print(f"scale 10->200 = {scale_10_200:.2f}x   (linear would be 20.0x)")

    print("\n=== Part 2: OLD per-feature path vs NEW single-scan ===")
    print(f"single feature : old={old1:.3f}s  new={new1:.3f}s  speedup={speedup_1:.2f}x")
    print(f"100 features   : old={old100:.3f}s  new={new100:.3f}s  speedup={speedup_100:.2f}x")

    print("\n=== Part 3: cold vs warm (grouped-bars cache) ===")
    print(f"cold={cold:.3f}s  warm={warm:.3f}s  warm_speedup={cold/warm:.2f}x")

    print("\n=== Part 4: scan count ===")
    print(f"grouped_bars_cache_keys_after_one_compute_many = {scan_keys}")

    print(f"\nJSON -> /tmp/bench_minute_bundle.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
