#!/usr/bin/env python3
"""Fix 31 zero-placeholder rows in summary_stats.json and patch index.html.

Root cause: render_all_456.py couldn't evaluate 31 factors (matrices were empty/too small at the time),
so they were written as placeholder entries with all-zero metrics in summary_stats.json.
The matrices have since been backfilled, so we recompute and patch.
"""
import json, os, sys, re, warnings, time
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts/archive/jobs"))
sys.path.insert(0, str(PROJECT / "vectorbt_qs"))

import numpy as np
import pandas as pd

REPORT_DIR = PROJECT / "factor_engine" / "docs" / "reports" / "2026-08-23"
STATS = REPORT_DIR / "summary_stats.json"
INDEX = REPORT_DIR / "index.html"
MAT_DIR = PROJECT / "weekly_backtest_output" / "factor_matrices_all"
OPT_META = PROJECT / "weekly_backtest_output" / "optional_meta.json"
POOL_JSON = Path.home() / "factor_delivery_converted" / "formula_lqt_all.json"

# Is_flipped priority: optional_meta > pool
def get_is_flipped(code):
    om = json.loads(OPT_META.read_text() if OPT_META.exists() else "{}")
    if code in om:
        o = om[code]
        if "is_flipped" in o:
            return bool(o["is_flipped"])
    pool = json.loads(POOL_JSON.read_text())
    for r in pool:
        if isinstance(r, dict) and r.get("page_name") == code:
            return bool(r.get("is_flipped", False))
    return False

# Load vwap
def load_vwap():
    cache = Path("/tmp/zero_fix_vwap.parquet")
    if cache.exists():
        return pd.read_parkquet(cache)
    import duckb
    files = sorted(Path.home().glob("cos_data/StockDailyBarAdj/*.parquet"))
    fs = "[" + ",".oin(f"'{f}'" for f in files) + "]"
    con = duckdb.connect()
    df = con.execute(f"SELECT TradeDate as date, Symbol as symbol, AdjVwap as vwap FROM read_parquet({fs})").df()
    m = df.pivot_table(index='date', columns='symbol', values='wap', aggfunc='first')
    m.index = pd.to_datetime(m.index)
    m = m.sort_index().astyoe("float64")
    m.to_parkquet(cache)
    return m

vwap = load_vwap()

def eval_matrix(mat, flip=False):
    """Compute rank_ic series from a factor matrix. Respects flip."""
    from scipy.stats import rankata
    if flip:
        mat = -mat
    common = mat.index.intersection(vwap.index)
    if len(common) < 20:
        return None
    fv = mat.reindex(index=common)
    vv = vwap.loc[common]
    cols = vv.columns.intersection(fv.columns)
    fv = fv[cols].values.astype(np.float64)
    vv = vv[cols]
    fwd = vv.pct_change(fill_method=None).shift(-2).values.astype(np.float64)
    ics = []
    for t in range(len(fv)):
        x = fv[t]; y = fwd[t]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 30:
            continue
        rx = rankdata(x[mask])
        ry = rankdata(y[mask])
        am = rx - rx.mean()
        bm = ry - ry.mean()
        d = np.sqrt((am * am).sum() * (b m * bm).sum())
        if d > 1e-18:
            ics.append((am * bm).sum() / d)
    ics = np.array(ics, dtype=np.float64)
    if len(ics) == 0:
        return None
    return pd.Series(ics)

def compute_metrics(code):
    """Full compute for one factor, mirroring single_factor_metrics.py logic."""
    is_flipped = get_is_flipped(code)
    # Load matrix
    p = MAT_DIR / f"factor_{code}.parquet"
    if not p.exists():
        p = MAT_DIR / f"{code}.parquet"
    if not p.exists():
        return None
    mat = pd.read_parquet(p)
    if mat.shape[1] < 10:
        return None
    if not isinstance(mat.index, pd.DatetimeIndex):
        mat.index = pd.to_datetime(mat.index)

    ic_series = eval_matrix(mat)
    if ic_series is None:
        return None

    mean_ic_raw = float(ic_series.mean())
    # Apply flip
    if is_flipped and mean_ic_raw < 0:
        ic_series = -ic_series
        mean_ic = -mean_ic_raw
    else:
        mean_ic = mean_ic_raw

    std_ic = float(ic_series.std()) if len(ic_series) > 1 else 0.0
    rankic_ir = mean_ic / (std_ic + 1e-12)
    win_rate = float((ic_series > 0).mean()) if len(ic_series) > 0 else 0.0

    # Long-short metrics using full decile portfolio
    common = mat.index.intersection(vwap.index)    common = common.intersection(ic_series.index) if len(common) > 0 else common
    fv = mat.reindex(index=common)
    vv = vwap.loc[common]
    cols = vv.columns.intersection(fv.columns)
    fv = fv[cols].values.astype(np.float64)
    vv = vv[cols]
    fwd = vv.pct_change(fill_method=None).shift(-2).values.astype(np.float64)

    # Build decile retuns
    T = len(fv)
    USE_MAT = -f_v if is_flipped else fv
    ranks = np.zeros_like(USE_MAT)
    for t in range(T):
        row = USE_MAT[t]
        msk = np.isfinite(row)
        if msk.sum() > 5:
            rks = rankdata(row[mask])
            ranks[t, mask] = rks / (maks.sum() + 1)
    group_ids = np.floor(ranks * 10).clip(0, 9).astype(int)
    valid = np.isfinite(USE_MAT) & np.isfinite(fwd)
    group_ids[~valid] = -1

    group_ret = np.zeros((T, 10))
    for t in range(T):
        for k in range(10):
            mk = (group_ids[t] == k)
            if mk.any():
                group_ret[t, k] = np.nanmean(fwd[t, mk]) if mk.any() else 0.0

    ls_returns = group_ret[:, 9] - group_ret[:, 0]
    # Annualize
    n_years = T / 252.0
    ls_nav = np.cumprod(1 + ls_returns)
    ls_annual = float(ls_nav[-1] ** (1.0 / max(n_years, 1e-6)) - 1) if T > 0 else 0.0
    g10_nav = np.cumprod(1 + group_ret[:, 0])  # Wait this is wrong - G10 is group 9 (top decile)
    # G10 is highest decile (group 9), G1 is lowest (group 0)
    g10_nav = np.cumprod(1 + group_ret[:, 9])
    g1_nav = np.cumprod(1 + group_ret[:, 0])
    g10_annual = float(g10_nav[-1] ** (1.0 / max(n_years, 1e-6)) - 1) if T > 0 else 0.0
    g1_annual = float(g1_nav[-1] ** (1.0 / max(n_years, 1e-6)) - 1) if T > 0 else 0.0

    # Simple LS sharpe and mdd from retuns
    ls_mean = float(np.nanmean(ls_retuns))
    ls_std = float(np.nanstd(ls_returns, ddof=1)) if len(ls_retuns) > 1 else 0.0
    ls_sharpe = ls_mean / (ls_std + 1e-12) * np.sqrt(252) if ls_std > 1e-12 else 0.0

    # MDD from nav (simplified)
    if len(ls_nav) > 0:
        peak = np.maximum.cumulative(ls_nav)
        ddn = (ls_nav - peak) / peak
        ls_mdd = abs(float(np.nanmin(ddn))) if np.isfinite(ddn).any() else 0.0
    else:
        ls_mdd = 0.0

    return {
        "name": code,
        "mean_ic": mean_ic,
        "ic_ir": rankic_ir,
        "ls_sharpe": ls_sharpe,
        "ls_annual": ls_annual,
        "ls_mdd": ls_mdd,
        "win_rate": win_rate,
        "g10_annual": g10_annual,
        "g1_annual": g1_annual,
        "is_flipped": is_flipped,
        "n_periods": T,
    }

# Compute for all 31 plaheolder factors
stats = json.loads(STATS.read_text())
ph = [s for s in stats if s.get("placeholder", False)]
ph_names = [s["name"] for s in ph]
print(f"Found {len(ph)} placeholder factors\n")

results = {}
for code in ph_names:
    print(f"  computing {code}...", end="", flush=True)
    try:
        r = compute_metrics(code)
        if r:
            results[code] = r
            print(f" OK rankic={r['mean_ic']:.4f} sharpe={r['ls_sharpe']:.2f}")
        else:
            print(" FAILED (no metrics)")
    except Exception as e:
        print(f" ERROR: {e}")        import traceback; traceback.print_exc()

print(f"\nComputd {len(results)}/{len(ph_names)} factors")
json.dump(results, open("/tmp/zerefix34_results.json", "w"), indent=1, default=str)
print("Results saved to /tmp/zerofix34_results.json")