#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""One-click output exporter for the lightgbm_qs portfolio backtest.

Guarantees the three curve PNGs + the equity_curves.csv all land in outputs/:

  1. equity_curve.png     (strategy vs benchmark, log axis)      [already produced by
                             scripts/portfolio_and_backtest.py, re-rendered here]
  2. excess_curve.png     (excess NAV = strategy NAV / benchmark NAV, + excess Sharpe)
  3. benchmark_curve.png  (standalone benchmark NAV, + benchmark Sharpe)
  4. equity_curves.csv    (date,strategy_nav,benchmark_nav,excess_nav,
                           strategy_ret,benchmark_ret,excess_ret)

Input preference:
  * If outputs/equity_curves.csv exists (written by scripts/portfolio_and_backtest.py
    in its plot_charts step) it is used as the source of truth -- nothing is
    recomputed, the figures are only re-rendered with the shared style.
  * Otherwise (--recompute, or csv missing) the curves are rebuilt from
    data/build/weights.parquet + data/panel/vwap_trad_adj.parquet using the SAME
    timing semantics as the backtest (user-mandated 2026-08-28):
        - wmat rows are already labeled with the EXECUTION day (decision d -> day d+1);
        - strategy daily return = sum(w.ffill() * vwap.pct_change()), NO shift(1);
        - transaction cost = TC_BPS * daily one-way turnover, charged every day.
    These semantics are identical to portfolio_and_backtest.run_backtest and must
    never be changed here.

Chart style mirrors portfolio_and_backtest.plot_charts: matplotlib Agg, English
labels (no CJK glyphs), same palette constants (C_STRATEGY / C_BENCH / C_AQUA /
SURF / INK / INK2 / GRID).

Usage:
  /home/sunhaiwei/quant_projects/.venv/bin/python export_curve_outputs.py [--tag smoke]
      [--recompute] [--tc 0.0005]
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
DATA = os.path.join(ROOT, "data")
BUILD = os.path.join(DATA, "build")
PANEL = os.path.join(DATA, "panel")
OUT = os.path.join(ROOT, "outputs")

WEIGHTS_OUT = os.path.join(BUILD, "weights.parquet")
VWAP_TRAD = os.path.join(PANEL, "vwap_trad_adj.parquet")
CSV_OUT = os.path.join(OUT, "equity_curves.csv")

# ---- chart palette (identical to portfolio_and_backtest.py) -------------------
C_STRATEGY = "#2a78d6"   # blue
C_BENCH = "#eb6834"      # orange
C_AQUA = "#1baf7a"       # aqua (secondary series)
C_RED = "#e34948"        # red (drawdown)
SURF = "#fcfcfb"         # chart surface
INK = "#0b0b0b"          # primary ink
INK2 = "#52514e"         # secondary ink
GRID = "#e1e0d9"         # hairline grid

# ---- backtest constants (read-only mirrors of portfolio_and_backtest.py) ------
TRADING_DAYS = 252
OOS_START = "2019-01-01"
TC_BPS = 0.0005

# required CSV column order
CSV_COLS = [
    "strategy_nav", "benchmark_nav", "excess_nav",
    "strategy_ret", "benchmark_ret", "excess_ret",
]


def log(msg):
    print(msg, flush=True)


def _annualise(cum_ret, n_days):
    return float((1.0 + cum_ret) ** (TRADING_DAYS / max(n_days, 1)) - 1.0)


def _max_drawdown(nav):
    return float((nav / nav.cummax() - 1.0).min())


def load_csv_curves():
    """Read outputs/equity_curves.csv written by portfolio_and_backtest.py."""
    df = pd.read_csv(CSV_OUT)
    if "date" not in df.columns:
        # some writers keep the index unnamed; first unnamed column is the date
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for c in CSV_COLS:
        if c not in df.columns:
            raise ValueError("equity_curves.csv missing column %s" % c)
    return df[CSV_COLS]


def recompute_curves(tc_bps=TC_BPS):
    """Rebuild the curves from weights.parquet + vwap panel.

    Replicates portfolio_and_backtest.run_backtest timing semantics exactly.
    Returns a DataFrame with the CSV_COLS columns, index = date.
    """
    wmat = pd.read_parquet(WEIGHTS_OUT)              # zeros already stored as NaN
    vwap = pd.read_parquet(VWAP_TRAD)
    vwap.index = pd.to_datetime(vwap.index)
    vwap = vwap.sort_index()

    ret = vwap.pct_change().replace([np.inf, -np.inf], np.nan)
    r = ret.fillna(0.0)
    w = wmat.ffill().fillna(0.0)                     # carry previous rebalance
    strat = (w * r).sum(axis=1, min_count=1)         # no shift: w[t] eats ret[t]
    prev = w.shift(1).fillna(0.0)
    to = w.sub(prev).abs().sum(axis=1) / 2.0         # daily one-way turnover
    strat = strat - tc_bps * to

    bench = ret.loc[ret.index >= pd.Timestamp(OOS_START)].mean(axis=1, skipna=True)

    oos = pd.concat([strat, bench], axis=1)
    oos.columns = ["strategy", "bench"]
    oos = oos.loc[oos.index >= pd.Timestamp(OOS_START)]
    oos = oos.dropna(subset=["strategy"]).dropna()

    eq = (1.0 + oos).cumprod()
    curve = pd.DataFrame({
        "strategy_nav": eq["strategy"],
        "benchmark_nav": eq["bench"],
        "excess_nav": eq["strategy"] / eq["bench"],
        "strategy_ret": eq["strategy"].pct_change(),
        "benchmark_ret": eq["bench"].pct_change(),
    })
    curve["excess_ret"] = curve["strategy_ret"] - curve["benchmark_ret"]
    curve.index.name = "date"
    return curve[CSV_COLS]


def excess_sharpe(curve):
    ex_ret = curve["strategy_nav"].pct_change() - curve["benchmark_nav"].pct_change()
    return float(ex_ret.mean() / (ex_ret.std(ddof=1) + 1e-12) * np.sqrt(TRADING_DAYS))


def _finalize(fig, path):
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log("[export] wrote %s" % path)


def plot_equity(curve, tag):
    strat = curve["strategy_nav"]
    bench = curve["benchmark_nav"]
    strat_ret = strat.pct_change().dropna()
    bench_ret = bench.pct_change().dropna()
    s_sharpe = float(strat_ret.mean() / (strat_ret.std(ddof=1) + 1e-12)
                     * np.sqrt(TRADING_DAYS))
    b_sharpe = float(bench_ret.mean() / (bench_ret.std(ddof=1) + 1e-12)
                     * np.sqrt(TRADING_DAYS))
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(SURF)
    ax.plot(strat.index, strat.values, lw=2.0, color=C_STRATEGY,
            label="Strategy max-Sharpe (Sharpe %.2f)" % s_sharpe)
    ax.plot(bench.index, bench.values, lw=1.6, color=C_BENCH,
            label="Benchmark equal-weight (Sharpe %.2f)" % b_sharpe)
    ax.axhline(1.0, color=INK2, lw=0.8, ls="--", alpha=0.6)
    ax.set_yscale("log")
    ax.set_title("Equity Curve (log scale) - Vwap basis, OOS%s"
                 % ("  [%s]" % tag if tag else ""))
    ax.set_xlabel("Date")
    ax.set_ylabel("Net value (log)")
    ax.grid(True, color=GRID, lw=0.7, alpha=0.8)
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    _finalize(fig, os.path.join(OUT, "equity_curve.png"))


def plot_excess(curve, tag):
    excess = curve["excess_nav"]
    ex_sharpe = excess_sharpe(curve)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(SURF)
    ax.plot(excess.index, excess.values, lw=1.8, color=C_AQUA,
            label="Excess NAV (strategy / benchmark) (excess Sharpe %.2f)" % ex_sharpe)
    ax.axhline(1.0, color=INK2, lw=0.8, ls="--", alpha=0.6, label="parity = 1.0")
    ax.set_title("Excess Curve (geometric, vs equal-weight benchmark) - Vwap basis, OOS%s"
                 % ("  [%s]" % tag if tag else ""))
    ax.set_xlabel("Date")
    ax.set_ylabel("Excess NAV")
    ax.grid(True, color=GRID, lw=0.7, alpha=0.8)
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    _finalize(fig, os.path.join(OUT, "excess_curve.png"))


def plot_benchmark(curve, tag):
    bench = curve["benchmark_nav"]
    bench_ret = bench.pct_change().dropna()
    b_sharpe = float(bench_ret.mean() / (bench_ret.std(ddof=1) + 1e-12)
                     * np.sqrt(TRADING_DAYS))
    n = len(bench)
    cum = bench.iloc[-1] - 1.0
    ann = _annualise(cum, n)
    mdd = _max_drawdown(bench) * 100
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(SURF)
    ax.plot(bench.index, bench.values, lw=2.0, color=C_BENCH,
            label="Benchmark NAV (Sharpe %.2f, ann. %.2f%%, MDD %.2f%%)"
                  % (b_sharpe, ann * 100, mdd))
    ax.axhline(1.0, color=INK2, lw=0.8, ls="--", alpha=0.6)
    ax.set_yscale("log")
    ax.set_title("Benchmark Curve (equal-weight full pool, log scale) - Vwap basis, OOS%s"
                 % ("  [%s]" % tag if tag else ""))
    ax.set_xlabel("Date")
    ax.set_ylabel("Net value (log)")
    ax.grid(True, color=GRID, lw=0.7, alpha=0.8)
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    _finalize(fig, os.path.join(OUT, "benchmark_curve.png"))


def write_csv(curve):
    curve.index.name = "date"
    curve.to_csv(CSV_OUT, float_format="%.8f")
    log("[export] wrote %s (%d rows)" % (CSV_OUT, len(curve)))


def main():
    ap = argparse.ArgumentParser(description="Export curve outputs (3 PNG + csv).")
    ap.add_argument("--tag", default="", help="e.g. smoke -> appended to chart titles")
    ap.add_argument("--recompute", action="store_true",
                    help="ignore existing equity_curves.csv and rebuild from weights")
    ap.add_argument("--tc", type=float, default=TC_BPS, help="per-side tc in return units")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    tag = args.tag.strip()

    if args.recompute or not os.path.exists(CSV_OUT):
        if not os.path.exists(WEIGHTS_OUT):
            sys.exit("[export] neither %s nor %s exists -- run portfolio_and_backtest.py first"
                     % (CSV_OUT, WEIGHTS_OUT))
        curve = recompute_curves(tc_bps=args.tc)
        log("[export] recomputed curves from weights (%d rows)" % len(curve))
    else:
        curve = load_csv_curves()
        log("[export] loaded curves from %s (%d rows)" % (CSV_OUT, len(curve)))

    # normalise column order exactly as the spec requires
    curve = curve[CSV_COLS]

    write_csv(curve)
    plot_equity(curve, tag)
    plot_excess(curve, tag)
    plot_benchmark(curve, tag)

    ex_sharpe = excess_sharpe(curve)
    strat = curve["strategy_nav"]
    bench = curve["benchmark_nav"]
    s_ret = strat.pct_change().dropna()
    b_ret = bench.pct_change().dropna()
    s_sharpe = float(s_ret.mean() / (s_ret.std(ddof=1) + 1e-12) * np.sqrt(TRADING_DAYS))
    b_sharpe = float(b_ret.mean() / (b_ret.std(ddof=1) + 1e-12) * np.sqrt(TRADING_DAYS))
    log("[export] DONE  rows=%d  strategy_sharpe=%.3f  benchmark_sharpe=%.3f  "
        "excess_sharpe=%.3f  period=%s .. %s  (tag=%r)"
        % (len(curve), s_sharpe, b_sharpe, ex_sharpe,
           curve.index.min().date(), curve.index.max().date(), tag))


if __name__ == "__main__":
    main()
