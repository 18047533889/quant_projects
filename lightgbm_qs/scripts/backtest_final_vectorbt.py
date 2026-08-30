# -*- coding: utf-8 -*-
"""Step 5: final portfolio backtest through the vectorbt_qs library.

The user explicitly requires ("回测肯定得用我们的这个库") that the final curve run
through /home/sunhaiwei/quant_projects/vectorbt_qs (its mvp/engine/runner.run_backtest
wraps vectorbt.Portfolio.from_orders).

This script builds the SAME portfolio weights that portfolio_and_backtest.py produces
(top-K max-Sharpe QP with turnover cap, Vwap basis, T+1 execution), then feeds them to
vectorbt_qs.run_backtest('ashare', wmat, config={execution_mode:'accurate', ...}).
It also produces equity / drawdown / sharpe charts via vectorbt_qs' own portfolio_report
+ nav curves.

Timing semantics (aligned to COS TargetVwapReturnH10, and to the user's just-verified
run_backtest convention):
  - decision on rebalance day d (prediction exists) -> weights EXECUTE at day d+1.
  - wmat rows are labeled with the execution day d+1 (matching portfolio_and_backtest).
  - vectorbt_qs signal rows = decision days; its execution_lag=1 maps each signal to the
    next session — identical semantics to our d+1 convention.

Usage:
  python backtest_final_vectorbt.py [--topk 30] [--maxw 0.05] [--turnover 0.30]
      [--tc 0.0005] [--pred data/build/predictions_adj_flip.parquet] [--mode accurate]
Env:
  ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data DATA_ACCESS_COS_READ_MODE=mirror
  QUANT_SCHEMA_CHECK=warn  (run with /srv/quant/envs/quantaalpha/bin/python)
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import cvxpy as cp

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
BUILD = os.path.join(ROOT, "data/build")
PANEL = os.path.join(ROOT, "data/panel")
OUT = os.path.join(ROOT, "outputs")
VECTORBT_QS = "/home/sunhaiwei/quant_projects/vectorbt_qs"
# vectorbt_qs is a package under /home/sunhaiwei/quant_projects; its parent must be
# on sys.path (not the package dir itself) so `import vectorbt_qs` resolves.
sys.path.insert(0, "/home/sunhaiwei/quant_projects")
sys.path.insert(0, os.path.join(VECTORBT_QS, "vectorbt"))   # vendored vectorbt import
sys.path.insert(0, VECTORBT_QS)

# vectorbt_qs imports (needs /srv/quant/envs/quantaalpha/bin/python: plotly, vectorbt)
from vectorbt_qs.mvp.data.adapter import set_data_root
from vectorbt_qs.mvp.engine.runner import run_backtest, portfolio_report
from vectorbt_qs.mvp.visualization import build_nav_curves

PREDICTIONS = os.environ.get("PREDICTIONS", os.path.join(BUILD, "predictions_adj_flip.parquet"))
VWAP_TRAD = os.environ.get("VWAP_TRAD", os.path.join(PANEL, "vwap_trad_adj.parquet"))

REBAL_PERIOD = 10
COV_WINDOW = 60
MIN_HOLDINGS = 8
OOS_START = "2019-01-01"
SOLVERS = [cp.CLARABEL, cp.SCS]

set_data_root("ashare", os.environ.get("ASHARE_PARQUET_ROOT", "/home/sunhaiwei/cos_data"))


def log(m):
    print(m, flush=True)


def asset_returns(vwap):
    return vwap.pct_change().replace([np.inf, -np.inf], np.nan)


def solve_max_sharpe(mu, cov, cap):
    n = len(mu)
    y = cp.Variable(n)
    objective = cp.Maximize(mu @ y)
    constraints = [cp.quad_form(y, cov) <= 1.0, y >= 0, y <= cap * cp.sum(y)]
    prob = cp.Problem(objective, constraints)
    for solver in SOLVERS:
        try:
            prob.solve(solver=solver, verbose=False)
        except Exception:
            continue
        if prob.status in ("optimal", "optimal_inaccurate") and y.value is not None:
            w = np.asarray(y.value, dtype=float).ravel()
            if w.sum() > 0:
                return w / w.sum()
    return None


def _enforce_turnover(target, prev, max_turnover):
    idx = prev.index
    delta = target.reindex(idx).fillna(0.0) - prev
    cap_l1 = 2.0 * max_turnover
    l1 = float(abs(delta).sum())
    if l1 > cap_l1 + 1e-9:
        delta = delta * (cap_l1 / l1)
    w = prev + delta
    delta2 = w - prev
    l1b = float(abs(delta2).sum())
    if l1b > cap_l1 + 1e-9:
        w = prev + delta2 * (cap_l1 / l1b)
    return w.clip(lower=0.0).clip(upper=1.0)


def build_weights(pred, vwap, top_k, max_weight, max_turnover, backoff_weight=0.10,
                  label_by="execution"):
    """Build the same top-K max-Sharpe QP weights as portfolio_and_backtest.py.

    label_by='execution' (default): rows labeled by the EXECUTION date d+1 (decision
      day d takes effect at day d+1) — this matches portfolio_and_backtest.py's
      convention and is what we pass to vectorbt_qs with execution_lag=0.
    label_by='signal': rows labeled by the DECISION day d — pass to vectorbt_qs with
      its default execution_lag=1 (it executes at the next session = d+1).
    """
    pm = pred.pivot_table(index="date", columns="asset", values="pred")
    pm = pm.reindex(index=vwap.index.union(pm.index)).sort_index()
    pm = pm[[c for c in vwap.columns if c in pm.columns]]
    ret = asset_returns(vwap)
    idx_dates = list(vwap.index)
    positions = np.arange(len(idx_dates))
    reb_dates = [idx_dates[pos] for pos in positions[positions % REBAL_PERIOD == 0]
                 if idx_dates[pos] in pm.index and pm.loc[idx_dates[pos]].notna().any()
                 and idx_dates[pos] <= pred["date"].max()]

    wmat = pd.DataFrame(0.0, index=vwap.index, columns=vwap.columns)
    prev_exec = None
    for d in reb_dates:
        pos = idx_dates.index(d)
        row = pm.loc[d].dropna()
        if len(row) < MIN_HOLDINGS:
            continue
        names = list(row.nlargest(top_k).index)
        if len(names) < MIN_HOLDINGS:
            continue
        rsub = ret.iloc[max(0, pos - COV_WINDOW):pos + 1][names].dropna()
        mu = rsub.mean().values
        cov = (rsub.cov().values + rsub.cov().values.T) / 2.0
        eigs = np.linalg.eigvalsh(cov)
        reg = max(1e-5 * np.nanmean(rsub.values ** 2) + 1e-9, -eigs.min() + 1e-7)
        cov = cov + reg * np.eye(len(names))
        w = None
        for cap in (max_weight, backoff_weight):
            w = solve_max_sharpe(mu, cov, cap)
            if w is not None and np.isfinite(w).all() and w.sum() > 0.5 \
                    and int((w > 1e-4).sum()) >= MIN_HOLDINGS:
                break
            w = None
        if w is None:
            w = np.full(len(names), 1.0 / len(names))
        if w.sum() < 0.999:
            w = w / w.sum()
        target = pd.Series(0.0, index=vwap.columns)
        target[names] = w
        if prev_exec is not None:
            target = _enforce_turnover(target, prev_exec, max_turnover)
        else:
            target = target.clip(lower=0.0)
            target = target / target.sum()
        exec_idx = pos + 1 if pos + 1 < len(idx_dates) else pos
        exec_date = idx_dates[exec_idx]
        row_date = exec_date if label_by == "execution" else d
        wmat.loc[row_date] = target.values
        prev_exec = target.copy()
    return wmat


def drop_untradable(wmat):
    """Restrict the portfolio to continuously-listed stocks.

    vectorbt_qs accurate/fast need price + Factor for every held symbol on every
    execution day. A stock that delists/suspends mid-window breaks the signal->exec
    mapping (signal on d, order at d+1 where the stock no longer has data). The
    robust fix: drop stocks that lack valid Factor data at ANY date in the window
    (delisted stocks can't be held by a real portfolio either).
    """
    from vectorbt_qs.mvp.data.adapter import load_market_data
    symbols = list(wmat.columns)
    start = str(wmat.index.min().date())
    end = str(wmat.index.max().date())
    data = load_market_data("ashare", symbols=symbols, start=start, end=end)
    open_adj = data.get("open_adj", data.get("open"))
    factor = data.get("factor")
    if factor is None and open_adj is None:
        return wmat
    if factor is not None:
        factor = factor.reindex(wmat.index)
        valid = factor.notna().all(axis=0)
    else:
        valid = open_adj.reindex(wmat.index).notna().all(axis=0)
    keep = [c for c in wmat.columns if c in valid.index and valid[c]]
    dropped_cols = [c for c in wmat.columns if c not in keep]
    if dropped_cols:
        log(f"[bt] dropped {len(dropped_cols)} non-continuously-listed columns: {dropped_cols[:8]}...")
    return wmat[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--maxw", type=float, default=0.05)
    ap.add_argument("--turnover", type=float, default=0.30)
    ap.add_argument("--tc", type=float, default=0.0005)
    ap.add_argument("--mode", default="accurate", choices=["accurate", "fast"])
    ap.add_argument("--benchmark", default="000300.SH")
    ap.add_argument("--weights-out", default=os.path.join(BUILD, "weights_final_flip.parquet"),
                    help="output path for the weight matrix (default: "
                         "data/build/weights_final_flip.parquet). Parallel runs should pass a "
                         "run-specific path (e.g. outputs/<run>/weights_final_<tag>.parquet) to "
                         "avoid clobbering the shared default.")
    args = ap.parse_args()

    pred = pd.read_parquet(PREDICTIONS)
    pred["date"] = pd.to_datetime(pred["date"])
    vwap = pd.read_parquet(VWAP_TRAD)
    vwap.index = pd.to_datetime(vwap.index)
    log(f"[bt] pred {pred.shape} vwap {vwap.shape} topk={args.topk} maxw={args.maxw} to={args.turnover}")

    # wmat rows = DECISION days (signal rows). vectorbt_qs execution_lag=1 (default)
    # executes each signal at the NEXT session (d+1) — exactly our T+1 convention.
    wmat = build_weights(pred, vwap, args.topk, args.maxw, args.turnover, label_by="signal")
    wmat = wmat.replace(0.0, np.nan)
    wmat = wmat.loc[wmat.index >= pd.Timestamp(OOS_START)]
    wmat = wmat.loc[wmat.index <= pd.Timestamp("2026-08-18")]
    wmat = wmat.dropna(how="all")
    wmat = drop_untradable(wmat)
    log(f"[bt] weight matrix: {wmat.shape}  rows {wmat.index.min().date()}..{wmat.index.max().date()}")

    cfg = {
        "execution_mode": args.mode,
        "init_cash": 10_000_000.0,
        "slippage": 0.001 if args.mode == "accurate" else 0.0,
        "benchmark_index": args.benchmark,
        "execution_lag": 1,          # signal day d -> execute next session d+1
    }
    if args.mode == "fast":
        cfg["fees"] = 0.0
    pf = run_backtest("ashare", wmat, config=cfg)
    rep = portfolio_report(pf)
    log("========= vectorbt_qs 绩效 =========")
    for k in ["Start Value", "End Value", "Total Return [%]", "Sharpe Ratio",
              "Max Drawdown [%]", "Total Trades", "Benchmark Return [%]"]:
        v = rep.get(k)
        log(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    nav = pf.value()
    nav_curves = build_nav_curves(pf)
    os.makedirs(OUT, exist_ok=True)
    nav.to_frame("nav").to_parquet(os.path.join(OUT, "backtest_final_nav.parquet"))
    nav_curves.to_parquet(os.path.join(OUT, "backtest_final_nav_curves.parquet"))
    wmat.to_parquet(args.weights_out)
    log(f"[bt] weights -> {args.weights_out}")

    # charts (matplotlib Agg)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    nc = nav_curves
    ax1.plot(nc.index, nc["strategy_nav"], color="#1f77b4", lw=1.3, label="Strategy NAV")
    if "benchmark_nav" in nc.columns:
        ax1.plot(nc.index, nc["benchmark_nav"], color="#7f7f7f", lw=1.0, label="Benchmark NAV")
    ax1.set_ylabel("Net value")
    ax1.set_title("Final Equity Curve (vectorbt_qs) — Sharpe %.2f" % rep.get("Sharpe Ratio", float("nan")))
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    dd = nav / nav.cummax() - 1.0
    ax2.fill_between(dd.index, dd.values * 100, 0, color="#d62728", alpha=0.6)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "backtest_final_equity.png"), dpi=150)
    plt.close(fig)
    log(f"[bt] charts -> outputs/backtest_final_equity.png")
    log("[bt] DONE")


if __name__ == "__main__":
    main()
