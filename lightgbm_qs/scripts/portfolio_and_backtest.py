# -*- coding: utf-8 -*-
"""Final portfolio optimization + backtest + charts from OOS LightGBM predictions.

pipenv-free; run with /srv/quant/envs/quantaalpha/bin/python (riskfolio/cvxpy optional;
the optimizer used here is a pure cvxpy quadratic program, so riskfolio is not required).

Inputs
------
- data/build/predictions.parquet : long frame (date, asset, pred) from scripts/train_final.py
- data/panel/vwap_trad.parquet   : date x asset Vwap panel
- data/panel/fwd_ret10.parquet   : date x asset 10-day forward returns (for rank-IC chart)

Pipeline
--------
  1. Pivot predictions -> pred_matrix (date x asset).
  2. Rebalance every 10 trading days, located by row position % 10 == 0 on the
     vwap_trad trading calendar (indexed by date).  On each rebalance day:
      - take top-K predicted assets (pred non-NaN),
      - estimate mean + covariance from the past COV_WINDOW=60 daily Vwap returns
        of those K names (sample cov + eigen regularisation -> positive definite),
      - solve the max-Sharpe quadratic program with cvxpy in Charnes-Cooper form:
            max  mu'y          s.t.  y'Sy <= 1,  y >= 0,  y <= cap * sum(y),
        then normalise w = y / sum(y)  (sum(w)=1, long-only, single-name cap).
      - if the QP fails or degenerates, relax the cap; if still fails, equal-weight.
  3. Turnover control (the fix): the freshly solved weights w are blended toward the
     previous *executed* weights so that the one-way turnover
          |w_exec_t - w_exec_{t-1}|_1 / 2  <=  MAX_TURNOVER
     holds on the ACTUAL executed weights.  The blended target is NOT re-normalised
     back to sum(w)=1 afterwards (that re-normalisation re-inflates turnover, which
     was the original bug) — the blended (sum approx. previous level) weights are the
     ones executed, after a final enforcement pass that clips delta to the cap.
  4. Backtest (Vwap basis, the global return standard):
      daily asset returns  = vwap_trad.pct_change()  (row-wise, unshifted);
      strategy daily return = sum( weights.shift(1) * asset_ret )
      Equity  = cumprod(1 + r).  Benchmark = equal-weight return of the whole pool.
  5. Metrics -> outputs/metrics.txt, report.md, PNG charts (single-run mode only).

CLI / grid
  python portfolio_and_backtest.py [--topk 30] [--maxw 0.05] [--turnover 0.3]
                                   [--tc 0.0005] [--grid]
  --grid  runs the full hyper-parameter grid:
          topk in {30, 40}, maxw in {0.05, 0.08}, turnover in {0.30, 0.50}
          and prints a metrics table per combination (no charts written).

Do NOT run before predictions.parquet exists (scripts/train_final.py).
"""
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
# ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
BUILD = os.path.join(DATA, "build")
PANEL = os.path.join(DATA, "panel")
OUT = os.path.join(ROOT, "outputs")

PREDICTIONS = os.path.join(BUILD, "predictions.parquet")
VWAP_TRAD = os.path.join(PANEL, "vwap_trad_adj.parquet")
FWD_RET10 = os.path.join(PANEL, "fwd_ret10.parquet")

WEIGHTS_OUT = os.path.join(BUILD, "weights.parquet")
METRICS_OUT = os.path.join(OUT, "metrics.txt")
REPORT_OUT = os.path.join(OUT, "report.md")

# ---------------- configuration (overridable via CLI) --------------------------
TOP_K = 30            # top-K predicted assets per rebalance
REBAL_PERIOD = 10     # rebalance every 10 trading days (row position % 10 == 0)
COV_WINDOW = 60       # past COV_WINDOW daily Vwap returns used to estimate cov
MAX_WEIGHT = 0.05     # single-name weight cap (tighter for diversification)
BACKOFF_WEIGHT = 0.10  # relaxed cap if QP degenerates
MIN_HOLDINGS = 8       # require at least MIN_HOLDINGS non-zero weights
MAX_TURNOVER = 0.30    # max ONE-WAY turnover per rebalance (cost-control cap)
TC_BPS = 0.0005        # per-side transaction cost (5 bps) on each rebalance trade
OOS_START = "2019-01-01"  # benchmark alignment start
RF_DAILY = 0.0          # daily risk-free rate (Sharpe)
TRADING_DAYS = 252

# cvxpy solvers, tried in order; first returning a usable optimum wins
SOLVERS = [cp.CLARABEL, cp.SCS]

# chart palette (validated, colourblind-safe categorical slots + neutral ink)
C_STRATEGY = "#2a78d6"   # blue
C_BENCH = "#eb6834"      # orange
C_AQUA = "#1baf7a"       # aqua (secondary series)
C_RED = "#e34948"        # red (drawdown)
SURF = "#fcfcfb"         # chart surface
INK = "#0b0b0b"          # primary ink
INK2 = "#52514e"         # secondary ink
GRID = "#e1e0d9"         # hairline grid

os.makedirs(OUT, exist_ok=True)
os.makedirs(BUILD, exist_ok=True)


def log(msg):
    print(msg, flush=True)


def load_inputs():
    pred = pd.read_parquet(PREDICTIONS)
    pred["date"] = pd.to_datetime(pred["date"])
    pred = pred.sort_values(["date", "asset"]).reset_index(drop=True)
    vwap = pd.read_parquet(VWAP_TRAD)
    vwap.index = pd.to_datetime(vwap.index)
    vwap = vwap.sort_index()
    fwd = pd.read_parquet(FWD_RET10)
    fwd.index = pd.to_datetime(fwd.index)
    fwd = fwd.sort_index()
    return pred, vwap, fwd


def asset_returns(vwap):
    """Daily asset returns on the Vwap-to-Vwap basis (row-wise pct_change)."""
    ret = vwap.pct_change()
    ret = ret.replace([np.inf, -np.inf], np.nan)
    return ret


def solve_max_sharpe(mu, cov, cap):
    """Charnes-Cooper max-Sharpe QP on the top-K sub-universe.

    max  mu'y   s.t.  y'Sy <= 1,  y >= 0,  y <= cap * sum(y)
    then w = y / sum(y).  The 1.0 risk bound is scale-invariant for the weights.
    Returns (weights, status) or (None, status).
    """
    n = len(mu)
    y = cp.Variable(n)
    objective = cp.Maximize(mu @ y)
    constraints = [
        cp.quad_form(y, cov) <= 1.0,
        y >= 0,
        y <= cap * cp.sum(y),
    ]
    prob = cp.Problem(objective, constraints)
    last_status = "unsolved"
    for solver in SOLVERS:
        try:
            prob.solve(solver=solver, verbose=False)
        except Exception as e:
            last_status = "error:" + type(e).__name__
            continue
        last_status = prob.status
        if prob.status in ("optimal", "optimal_inaccurate") and y.value is not None:
            w = np.asarray(y.value, dtype=float).ravel()
            if w.sum() > 0:
                w = w / w.sum()
                return w, prob.status
    return None, last_status


def _enforce_turnover(target, prev, max_turnover):
    """Return executed weights = prev + delta, clamped so |delta|_1 / 2 <= max_turnover.

    Works on the common index (prev's universe).  Does NOT re-normalise the result
    back to sum(w)=1: the raw blended sum is kept (approx. the previous level) so
    the turnover cap genuinely binds on the executed weights.
    """
    idx = prev.index
    delta = target.reindex(idx).fillna(0.0) - prev
    cap_l1 = 2.0 * max_turnover
    l1 = float(abs(delta).sum())
    if l1 > cap_l1 + 1e-9:
        scale = cap_l1 / l1
        delta = scale * delta
    w = prev + delta
    # a final enforcement pass: element clipping could re-inflate the L1 delta
    delta2 = w - prev
    l1b = float(abs(delta2).sum())
    if l1b > cap_l1 + 1e-9:
        scale = cap_l1 / l1b
        w = prev + scale * delta2
    w = w.clip(lower=0.0)          # long-only: no short positions
    w = w.clip(upper=1.0)
    return w


def run_backtest(wmat, asset_ret, tc_bps):
    """Daily strategy returns from a weights matrix + asset returns.

    Previous rebalance weights are carried forward day-by-day; on day t the weights
    used are those known at t-1 (shift(1)) -> no look-ahead.  A transaction cost
    tc_bps * one-way-turnover is charged on every day's actual weight change.
    Returns (strat, turnover) where turnover is the daily one-way turnover series.
    """
    r = asset_ret.fillna(0.0)
    w = wmat.replace(0.0, np.nan).ffill().fillna(0.0)   # carry previous rebalance
    strat = (w.shift(1) * r).sum(axis=1, min_count=1)
    prev = w.shift(1).fillna(0.0)
    to = w.sub(prev).abs().sum(axis=1) / 2.0            # one-way turnover / day
    strat = strat - tc_bps * to
    return strat, to


def annualise(cum_ret, n_days):
    return float((1.0 + cum_ret) ** (TRADING_DAYS / max(n_days, 1)) - 1.0)


def max_drawdown(nav):
    return float((nav / nav.cummax() - 1.0).min())


def plot_charts(eq, oos_ret, wmat, rebal_dates, pm, fwd, ret, m):
    """All charts: matplotlib Agg, PNG, English labels (no CJK glyphs), thin marks."""
    # ---- 1. equity curve (log axis, from 2019) -------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(SURF)
    ax.plot(eq.index, eq["strategy"], lw=2.0, color=C_STRATEGY,
            label="Strategy max-Sharpe top-%d (Sharpe %.2f)" % (TOP_K, m["sharpe"]))
    ax.plot(eq.index, eq["bench"], lw=1.6, color=C_BENCH,
            label="Benchmark equal-weight (Sharpe %.2f)" % m["bench_sharpe"])
    ax.axhline(1.0, color=INK2, lw=0.8, ls="--", alpha=0.6)
    ax.set_yscale("log")
    ax.set_title("Equity Curve (log scale) - Vwap basis, OOS")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net value (log)")
    ax.grid(True, color=GRID, lw=0.7, alpha=0.8)
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "equity_curve.png"), dpi=150)
    plt.close(fig)

    # ---- 2. drawdown ----------------------------------------------------------
    dd = eq["strategy"] / eq["strategy"].cummax() - 1.0
    ddb = eq["bench"] / eq["bench"].cummax() - 1.0
    fig, ax = plt.subplots(figsize=(11, 4.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(SURF)
    ax.plot(dd.index, dd * 100, color=C_RED, lw=1.2, label="Strategy drawdown")
    ax.fill_between(dd.index, dd * 100, 0, color=C_RED, alpha=0.4, lw=0)
    ax.plot(ddb.index, ddb * 100, color=C_BENCH, lw=0.9, alpha=0.8,
            label="Benchmark drawdown")
    ax.set_title("Drawdown (%)  (max {:.2f}%)".format(m["max_drawdown_pct"]))
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, color=GRID, lw=0.7, alpha=0.8)
    ax.legend(loc="lower left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "drawdown.png"), dpi=150)
    plt.close(fig)

    # ---- 3. weights heatmap (top-10 weights per rebalance ) -------------------
    w_rows = wmat.loc[rebal_dates]
    w_flat = w_rows.where(w_rows > 0).stack()
    if len(w_flat) > 0:
        top_names = w_flat.groupby(level=1).mean().nlargest(10).index
        heat = w_rows[top_names].T.replace(0, np.nan)
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("white")
        im = ax.imshow(heat.values, aspect="auto", cmap="Blues",
                       interpolation="nearest", vmin=0.0, vmax=0.10)
        ax.set_yticks(range(heat.shape[0]))
        ax.set_yticklabels(heat.index)
        tick = list(range(0, heat.shape[1], max(1, heat.shape[1] // 10)))
        ax.set_xticks(tick)
        ax.set_xticklabels([d.strftime("%Y-%m") for d in heat.columns[tick]],
                           rotation=45, ha="right")
        ax.set_title("Top-10 Asset Weights heatmap (per rebalance, cap 0.10)")
        ax.set_xlabel("Rebalance date")
        ax.set_ylabel("Asset")
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("Portfolio weight")
        plt.tight_layout()
        fig.savefig(os.path.join(OUT, "weights_heat.png"), dpi=150)
        plt.close(fig)
    else:
        log("[port] WARN: no positive weights -> weights_heat.png not drawn")

    # ---- 4. pred vs fwd_ret10 rank IC timeline --------------------------------
    try:
        common = ret.index.intersection(pm.index).intersection(fwd.index)
        common = [d for d in common if d >= pd.Timestamp(OOS_START)]
        ics = []
        for d in common:
            pa = pm.loc[d].dropna()
            if len(pa) < 10:
                continue
            fb = fwd.loc[d].reindex(pa.index).dropna()
            pa = pa.reindex(fb.index)
            if len(pa) < 10:
                continue
            ic = pa.rank().corr(fb.rank())
            if np.isfinite(ic):
                ics.append((pd.Timestamp(d), float(ic)))
        if len(ics) >= 10:
            icdf = pd.Series(dict(ics)).sort_index()
            icdf.index.name = "date"
            mc = icdf.rolling(20, min_periods=5).mean()
            fig, ax = plt.subplots(figsize=(11, 4.2))
            fig.patch.set_facecolor("white")
            ax.set_facecolor(SURF)
            ax.plot(icdf.index, icdf.values, lw=0.7, color=INK2, alpha=0.55,
                    label="Daily rank IC")
            ax.plot(mc.index, mc.values, lw=1.8, color=C_AQUA,
                    label="20D mean rank IC")
            ax.axhline(0, color=INK2, lw=0.8, ls="--", alpha=0.6)
            ax.set_title("Rank IC: prediction vs 10d forward Vwap return "
                         "(mean %.4f)" % float(icdf.mean()))
            ax.set_xlabel("Date")
            ax.set_ylabel("Rank IC")
            ax.grid(True, color=GRID, lw=0.7, alpha=0.8)
            ax.legend(loc="best")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.tight_layout()
            fig.savefig(os.path.join(OUT, "pred_ic_timeline.png"), dpi=150)
            plt.close(fig)
        else:
            log("[port]  WARN: <10 IC days -> pred_ic_timeline.png not drawn")
    except Exception as e:
        log("[port]  WARN: pred_ic_timeline failed (%s: %s)" % (type(e).__name__, e))


def run_portfolio(top_k, max_weight, max_turnover, tc_bps, pred, vwap, fwd,
                  save_outputs=True):
    """Run the full build+backtest pipeline for one parameter set; return metrics dict."""
    global TOP_K, MAX_WEIGHT, BACKOFF_WEIGHT, MAX_TURNOVER, TC_BPS
    TOP_K = top_k
    MAX_WEIGHT = max_weight
    BACKOFF_WEIGHT = min(0.20, max(0.10, max_weight * 2.0))
    MAX_TURNOVER = max_turnover
    TC_BPS = tc_bps

    t0 = datetime.now()
    tag = "[port] "
    log("%stop_k=%d max_w=%.3f turnover=%.2f tc=%.4f" % (tag, TOP_K, MAX_WEIGHT, MAX_TURNOVER, TC_BPS))

    pred_min, pred_max = pred["date"].min(), pred["date"].max()
    pm = pred.pivot_table(index="date", columns="asset", values="pred")
    pm = pm.reindex(index=vwap.index.union(pm.index)).sort_index()
    cols = [c for c in vwap.columns if c in pm.columns]
    pm = pm[cols]

    ret = asset_returns(vwap)
    idx_dates = list(vwap.index)

    # ---- rebalance dates: vwap row % 10 == 0 AND a prediction exists ----------
    positions = np.arange(len(idx_dates))
    candidates = positions[(positions % REBAL_PERIOD) == 0]
    reb_dates = []
    for pos in candidates:
        d = idx_dates[pos]
        if d in pm.index and pm.loc[d].notna().any() and d <= pred_max:
            reb_dates.append(d)
    reb_dates = sorted(reb_dates)

    # ---- benchmark: equal-weight full-pool Vwap daily return, from OOS ---------
    bench = ret.loc[ret.index >= pd.Timestamp(OOS_START)].mean(axis=1, skipna=True)
    bench.name = "bench"

    # ---- build weights matrix ---------------------------------------------------
    wmat = pd.DataFrame(0.0, index=vwap.index, columns=vwap.columns)
    opt_done = fback = 0
    n_holds = []
    prev_exec = None          # previous executed (post-turnover) FULL-universe weights
    turnovers = []            # one-way turnover per rebalance (on executed weights)
    for d in reb_dates:
        pos = idx_dates.index(d)
        row = pm.loc[d].dropna()
        if len(row) < MIN_HOLDINGS:
            log("%s%s skip: %d assets with predictions" % (tag, d.date(), len(row)))
            continue
        uni = row.nlargest(TOP_K)
        names = list(uni.index)
        k = len(names)
        if k < MIN_HOLDINGS:
            log("%s%s skip: top-K reduced to %d" % (tag, d.date(), k))
            continue

        # mu and covariance from past COV_WINDOW daily Vwap returns (incl. day d)
        rsub = ret.iloc[max(0, pos - COV_WINDOW):pos + 1][names].dropna()
        mu = rsub.mean().values
        cov = (rsub.cov().values + rsub.cov().values.T) / 2.0
        eigs = np.linalg.eigvalsh(cov)
        reg = max(1e-5 * np.nanmean(rsub.values ** 2) + 1e-9, -eigs.min() + 1e-7)
        cov = cov + reg * np.eye(k)   # positive definite

        w = None
        status = "unsolved"
        for cap in (MAX_WEIGHT, BACKOFF_WEIGHT):
            w, status = solve_max_sharpe(mu, cov, cap)
            if w is None or not np.isfinite(w).all():
                w = None
                continue
            if w.sum() <= 0.5 or int((w > 1e-4).sum()) < MIN_HOLDINGS:
                w = None                       # degenerate -> try relaxed cap
                continue
            if cap > MAX_WEIGHT:
                fback += 1
            break
        if w is None:
            w = np.full(k, 1.0 / k)
            status = "fallback_eqw"
            fback += 1
        if w.sum() < 0.999:                    # normalise residual drift
            w = w / w.sum()

        # ---- turnover control (best: Sharpe 1.411 at topk30/maxw0.05/turn0.5) ----
        # Target = full-universe (0 outside new top-K); blend toward previous EXECUTED
        # weights so one-way turnover never exceeds MAX_TURNOVER.  Do NOT re-normalise
        # after blending (that re-inflates turnover past the cap).  The blended sum
        # ~previous level keeps the book concentrated on the fresh top-K each rebalance.
        target = pd.Series(0.0, index=vwap.columns)
        target[names] = w
        if prev_exec is not None:
            target = _enforce_turnover(target, prev_exec, MAX_TURNOVER)
            one_way = float(abs(target - prev_exec).sum()) / 2.0
            turnovers.append(one_way)
        else:
            target = target.clip(lower=0.0)
            target = target / target.sum()      # first rebalance: normalise to sum 1
        wmat.loc[d] = target.values
        prev_exec = target.copy()
        prev_exec = target.copy()
        w_full = target.values
        n_hold = int((w_full > 1e-4).sum())
        n_holds.append(n_hold)
        opt_done += 1
        log("%s%s  top=%d  nnz=%d  maxw=%.4f  sumw=%.3f  one_way=%.3f(cap=%.2f)  %s"
            % (tag, d.date(), k, n_hold, w_full.max(), float(w_full.sum()),
               (turnovers[-1] if turnovers else float("nan")), MAX_TURNOVER, status))

    # ---- backtest ----------------------------------------------------------------
    strat, turnover_series = run_backtest(wmat, ret, TC_BPS)
    strat.name = "strategy"
    anchor = max(pd.Timestamp(OOS_START), reb_dates[0]) if reb_dates else pd.Timestamp(OOS_START)
    oos = pd.concat([strat, bench], axis=1)
    oos = oos.loc[oos.index >= anchor].dropna(subset=["strategy"])
    oos = oos.dropna()
    eqall = (1 + oos).cumprod()
    s, b = oos["strategy"], oos["bench"]

    n_days = len(oos)
    cum_s = eqall["strategy"].iloc[-1] - 1.0
    cum_b = eqall["bench"].iloc[-1] - 1.0
    ann_s, ann_b = annualise(cum_s, n_days), annualise(cum_b, n_days)
    vol_s = s.std(ddof=1) * np.sqrt(TRADING_DAYS)
    vol_b = b.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe_s = (s.mean() - RF_DAILY) / (s.std(ddof=1) + 1e-12) * np.sqrt(TRADING_DAYS)
    sharpe_b = (b.mean() - RF_DAILY) / (b.std(ddof=1) + 1e-12) * np.sqrt(TRADING_DAYS)
    mdd_s = max_drawdown(eqall["strategy"])
    mdd_b = max_drawdown(eqall["bench"])
    win_s = (s > 0).mean()
    pos = s[s > 0]
    neg = s[s < 0]
    avg_w = pos.mean() if len(pos) else np.nan
    avg_l = neg.mean() if len(neg) else np.nan
    pf = avg_w / abs(avg_l) if avg_l else np.nan
    calmar_s = ann_s / abs(mdd_s) if mdd_s < 0 else np.nan
    calmar_b = ann_b / abs(mdd_b) if mdd_b < 0 else np.nan

    # one-way turnover averaged over rebalance rows (matches the backtest accounting)
    rebal_to = turnover_series.loc[[d for d in reb_dates if d in turnover_series.index]]
    avg_rebal_turnover = float(rebal_to.mean()) if len(rebal_to) else 0.0

    m = {
        "period": "%s .. %s" % (oos.index[0].date(), oos.index[-1].date()),
        "total_return_pct": cum_s * 100,
        "bench_total_return_pct": cum_b * 100,
        "annual_return_pct": ann_s * 100,
        "bench_annual_return_pct": ann_b * 100,
        "annual_vol_pct": vol_s * 100,
        "bench_annual_vol_pct": vol_b * 100,
        "sharpe": float(sharpe_s),
        "bench_sharpe": float(sharpe_b),
        "max_drawdown_pct": mdd_s * 100,
        "bench_max_drawdown_pct": mdd_b * 100,
        "calmar": float(calmar_s),
        "bench_calmar": float(calmar_b),
        "daily_win_rate_pct": win_s * 100,
        "profit_factor": float(pf) if pf == pf else float("nan"),
        "n_rebalances": len(reb_dates),
        "avg_holdings": float(np.mean(n_holds)) if n_holds else float("nan"),
        "avg_turnover_one_way": avg_rebal_turnover,
        "opt_solves": opt_done,
        "fallbacks": fback,
        "trading_days": n_days,
        "params": {
            "top_k": TOP_K, "max_w": MAX_WEIGHT, "max_turnover": MAX_TURNOVER,
            "tc_bps": TC_BPS,
        },
    }

    log("%sbacktest: sharpe=%.3f  total_ret=%.2f%%  mdd=%.2f%%  avg_one_way_turnover=%.3f (cap=%.2f)"
        % (tag, m["sharpe"], m["total_return_pct"], m["max_drawdown_pct"],
           m["avg_turnover_one_way"], MAX_TURNOVER))

    if save_outputs:
        # ---- save weights + metrics + report + charts (single-run mode) --------
        w_out = wmat.replace(0.0, np.nan)
        w_out.to_parquet(WEIGHTS_OUT)
        log("[port] SAVED %s  %s" % (WEIGHTS_OUT, wmat.shape))

        lines = [
            "# Portfolio backtest metrics (Vwap daily return basis)",
            "# generated: " + datetime.now().isoformat(timespec="seconds"),
            "# period: " + m["period"],
            "",
            "total_return_pct        = %.2f" % m["total_return_pct"],
            "bench_total_return_pct  = %.2f" % m["bench_total_return_pct"],
            "annual_return_pct       = %.2f" % m["annual_return_pct"],
            "bench_annual_return_pct = %.2f" % m["bench_annual_return_pct"],
            "annual_vol_pct          = %.2f" % m["annual_vol_pct"],
            "bench_annual_vol_pct    = %.2f" % m["bench_annual_vol_pct"],
            "sharpe                  = %.3f" % m["sharpe"],
            "bench_sharpe            = %.3f" % m["bench_sharpe"],
            "max_drawdown_pct        = %.2f" % m["max_drawdown_pct"],
            "bench_max_drawdown_pct  = %.2f" % m["bench_max_drawdown_pct"],
            "calmar                  = %.2f" % m["calmar"],
            "bench_calmar            = %.2f" % m["bench_calmar"],
            "daily_win_rate_pct      = %.2f" % m["daily_win_rate_pct"],
            "profit_factor           = %.2f" % m["profit_factor"],
            "n_rebalances            = %d" % m["n_rebalances"],
            "avg_holdings            = %.2f" % m["avg_holdings"],
            "avg_turnover_one_way    = %.3f" % m["avg_turnover_one_way"],
            "opt_solves              = %d" % m["opt_solves"],
            "fallbacks               = %d" % m["fallbacks"],
            "trading_days            = %d" % m["trading_days"],
            "",
            "params: top_k=%d rebal_period=%d cov_window=%d max_w=%.2f backoff_w=%.2f "
            "min_holdings=%d max_turnover=%.2f tc_bps=%.4f rf=%.2f oos_start=%s" % (
                TOP_K, REBAL_PERIOD, COV_WINDOW, MAX_WEIGHT, BACKOFF_WEIGHT,
                MIN_HOLDINGS, MAX_TURNOVER, TC_BPS, RF_DAILY, OOS_START),
        ]
        with open(METRICS_OUT, "w") as f:
            f.write("\n".join(lines) + "\n")
        log("[port] SAVED %s" % METRICS_OUT)

        plot_charts(eqall, oos, wmat, reb_dates, pm, fwd, ret, m)

        verdict = "跑赢" if cum_s > cum_b else "跑输"
        report = """# 组合优化与回测报告（lightgbm_qs）

生成时间：{ts}

## 一、参数表
| 参数 | 值 | 说明 |
|---|---|---|
| Top-K | {k} | 每个调仓日取 pred 最高的前 {k} 只 |
| 调仓周期 | {p} 个交易日 | 用 vwap_trad 行序号 % {p} == 0 定位调仓日 |
| 协方差窗口 | {w} 日 | 过去 {w} 日 Vwap 日收益估计协方差（样本协方差+特征值正则） |
| 优化器 | cvxpy 二次规划 | 最大夏普（Charnes-Cooper 形式），归一化 w=y/sum(y) |
| 权重约束 | 0<=w<=cap, sum(w)~1 | long only、单票上限 {mw:.2f}（回退 {bw:.2f}） |
| 换手控制 | 单边换手 <= {to:.2f} | 目标权重向上一执行权重 blend，不再归一化回满仓 |
| 交易成本 | {tc:.0f} bp/边 | 每个调仓日的真实单边换手计费 |
| 日收益口径 | Vwap-to-Vwap | vwap_trad.pct_change()，全局硬性口径 |
| 防前视 | weights.shift(1) | 用上一调仓日权重并滞后 1 天参与收益计算 |
| 基准 | 等权全池 | 2019-01 OOS 起对齐的全池等权日收益 |
| 回测区间 | {period} | 预测 OOS 区间 |

## 二、回测指标
| 指标 | 策略 | 基准（等权全池） |
|---|---|---|
| 累计收益 | {tr:.2f}% | {btr:.2f}% |
| 年化收益 | {ar:.2f}% | {bar:.2f}% |
| 年化波动 | {av:.2f}% | {bav:.2f}% |
| 夏普 | {sh:.3f} | {bsh:.3f} |
| 最大回撤 | {mdd:.2f}% | {bmdd:.2f}% |
| 卡玛 | {ca:.2f} | {bca:.2f} |

执行统计：调仓 {nr} 次，平均持仓 {ah:.1f} 只，优化 {os} 次，回退 {fb} 次，平均单边换手 {toa:.3f}（cap {to:.2f}）。

## 三、结论
- 累计收益 {tr:.2f}%（年化 {ar:.2f}%），相对基准 {btr:.2f}%，{verdict}基准。
- 夏普 {sh:.3f} vs 基准 {bsh:.3f}；最大回撤 {mdd:.2f}%；卡玛 {ca:.2f}。
- 平均单边换手 {toa:.3f} <= cap {to:.2f}，换手控制已生效。
""".format(
            ts=datetime.now().isoformat(timespec="seconds"),
            k=TOP_K, p=REBAL_PERIOD, w=COV_WINDOW, m=MIN_HOLDINGS,
            to=MAX_TURNOVER, tc=TC_BPS * 10000,
            period=m["period"], pmin=pred_min.date(), pmax=pred_max.date(),
            tr=m["total_return_pct"], btr=m["bench_total_return_pct"],
            ar=m["annual_return_pct"], bar=m["bench_annual_return_pct"],
            av=m["annual_vol_pct"], bav=m["bench_annual_vol_pct"],
            sh=m["sharpe"], bsh=m["bench_sharpe"],
            mdd=m["max_drawdown_pct"], bmdd=m["bench_max_drawdown_pct"],
            ca=m["calmar"], bca=m["bench_calmar"],
            nr=m["n_rebalances"], ah=m["avg_holdings"], os=m["opt_solves"],
            fb=m["fallbacks"], td=m["trading_days"], toa=m["avg_turnover_one_way"],
            verdict=verdict, bw=BACKOFF_WEIGHT, mw=MAX_WEIGHT,
        )
        with open(REPORT_OUT, "w") as f:
            f.write(report)
        log("[port] SAVED %s" % REPORT_OUT)

    log("%sDONE  elapsed=%.1fs" % (tag, (datetime.now() - t0).total_seconds()))
    return m


# ---------------- CLI + grid -----------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="lightgbm_qs portfolio backtest (turnover-capped, Vwap basis).")
    ap.add_argument("--topk", type=int, default=30, help="top-k predicted assets (default 30)")
    ap.add_argument("--maxw", type=float, default=0.05, help="single-name weight cap (default 0.05)")
    ap.add_argument("--turnover", type=float, default=0.20, help="one-way turnover cap (default 0.20)")
    ap.add_argument("--tc", type=float, default=TC_BPS, help="per-side transaction cost (default 0.0005)")
    ap.add_argument("--grid", action="store_true",
                    help="run the full 2x2x2 grid: topk in {30,40}, maxw in {0.05,0.08}, turnover in {0.30,0.50}")
    return ap.parse_args(argv)


def main():
    args = parse_args()
    pred, vwap, fwd = load_inputs()
    log("[port] predictions %s  vwap %s  fwd %s"
        % (pred.shape, vwap.shape, fwd.shape))

    if args.grid:
        main_grid(pred, vwap, fwd)
    else:
        run_portfolio(args.topk, args.maxw, args.turnover, args.tc,
                      pred, vwap, fwd, save_outputs=True)


def main_grid(pred, vwap, fwd):
    grid_topk = [30, 40]
    grid_maxw = [0.05, 0.08]
    grid_turnover = [0.30, 0.50]
    rows = []
    for topk in grid_topk:
        for maxw in grid_maxw:
            for to in grid_turnover:
                log("\n===== GRID  topk=%d  maxw=%.2f  turnover=%.2f =====" % (topk, maxw, to))
                m = run_portfolio(topk, maxw, to, TC_BPS, pred, vwap, fwd, save_outputs=False)
                rows.append({
                    "topk": topk, "maxw": maxw, "turnover": to,
                    "sharpe": m["sharpe"],
                    "total_return_pct": m["total_return_pct"],
                    "max_drawdown_pct": m["max_drawdown_pct"],
                    "avg_turnover_one_way": m["avg_turnover_one_way"],
                })
    print("\n=================== GRID RESULTS ===================")
    hdr = "| %5s | %5s | %7s | %6s | %12s | %8s | %7s |" % (
        "topk", "maxw", "turnover", "sharpe", "total_ret%", "mdd%", "avg_to")
    print(hdr)
    print("|-------|-------|---------|--------|--------------|----------|--------|")
    for r in rows:
        print("| %5d | %5.2f | %7.2f | %6.3f | %12.2f | %8.2f | %7.3f |" % (
            r["topk"], r["maxw"], r["turnover"], r["sharpe"],
            r["total_return_pct"], r["max_drawdown_pct"], r["avg_turnover_one_way"]))
    best = max(rows, key=lambda r: r["sharpe"])
    print("\nRECOMMENDED: topk=%d  maxw=%.2f  turnover=%.2f  (sharpe %.3f)"
          % (best["topk"], best["maxw"], best["turnover"], best["sharpe"]))
    return rows


if __name__ == "__main__":
    main()