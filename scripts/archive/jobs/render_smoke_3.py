#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smoke 渲染：3 个代表因子（1 翻转 / 1 不翻 / 1 代理型），验证修复后的详情页。

样本：
  - Alpha158_Volume_Price_Diversity_Enhanced_Mutated : is_flipped=True（meta）
  - adaptive_volume_smoothness                        : is_flipped=False（meta）
  - volume_adjusted_price_range                       : is_flipped=True（meta, 61 代理型）

用法：.venv/bin/python scripts/archive/jobs/render_smoke_3.py
"""
import sys, os, json, time, re
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

PROJECT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts" / "archive" / "jobs"))
sys.path.insert(0, str(PROJECT / "jobs"))

os.environ["FACTOR_SET"] = "all"
os.environ["FACTOR_REPORT_DIR"] = str(PROJECT / "factor_engine" / "docs" / "reports" / "2026-08-23")

import rebuild_factor_detail_pages as R
from single_factor_metrics import compute_factor_metrics_single

SAMPLE = [
    "Alpha158_Volume_Price_Diversity_Enhanced_Mutated",  # flipped
    "adaptive_volume_smoothness",                         # not flipped
    "volume_adjusted_price_range",                        # proxy 61 factor (flipped in meta)
]


def render_one(name: str) -> dict:
    fi = R.extract_formula_info(name)
    is_flipped = fi["is_flipped"] or R._is_flipped_by_meta(name)

    mat_path = PROJECT / "weekly_backtest_output" / "factor_matrices_all" / f"{name}.parquet"
    if not mat_path.exists():
        return {"name": name, "ok": False, "error": "no matrix"}
    matrix = pd.read_parquet(mat_path)
    if not isinstance(matrix.index, pd.DatetimeIndex):
        matrix.index = pd.to_datetime(matrix.index)

    fm = compute_factor_metrics_single(name, matrix)
    if not fm or fm.get("perf", {}).get("n_periods", 0) < 20:
        return {"name": name, "ok": False, "error": "no metrics"}

    ic_series = fm.get("ic_series", pd.Series(dtype=float)).copy()
    ic_series = ic_series.astype(float).replace([np.inf, -np.inf], np.nan)
    ic_stats = R.compute_ic_stats(ic_series)

    # 翻转同步：与主渲染一致（is_flipped → IC/方向翻正）
    if is_flipped and ic_series.mean() < 0:
        ic_series = -ic_series
        ic_stats = R.compute_ic_stats(ic_series)
        fm = dict(fm); fm["perf"] = dict(fm["perf"])
        for k in ["ls_sharpe", "ls_annual", "g10_annual", "g1_annual", "g10_sharpe", "g1_sharpe"]:
            fm["perf"][k] = -fm["perf"].get(k, 0)
        if "ls_winrate" in fm["perf"]:
            fm["perf"]["ls_winrate"] = 1.0 - fm["perf"].get("ls_winrate", 0)
        if "win_rate" in fm:
            fm["win_rate"] = 1.0 - fm.get("win_rate", 0)
        fm["perf"]["ls_mdd"] = abs(fm["perf"].get("ls_mdd", 0))
        # mirror deciles
        dr = dict(fm.get("decile_navs", {}))
        gs = []
        for k in range(1, 11):
            key = f"G{k}"
            if key in dr and len(dr[key]) > 0:
                gs.append((k, dr[key]))
        if len(gs) == 10:
            nd = {"dates": dr.get("dates", [])}
            for nk, (_, v) in zip(range(1, 11), reversed(gs)):
                nd[f"G{nk}"] = v
            ls0 = dr.get("LS", [])
            if ls0 and len(ls0) > 0:
                nd["LS"] = [1.0 / x if x not in (0, None) else 0.0 for x in ls0]
            fm["decile_navs"] = nd
        fm["mean_ic"] = -fm.get("mean_ic", 0)
        fm["ic_ir"] = -fm.get("ic_ir", 0)

    decile_data = {"dates": fm.get("dates_out", [])}
    for k, v in fm.get("decile_navs", {}).items():
        decile_data[k] = v
    perf = fm.get("perf", {})

    monthly_chart = R.plot_ic_monthly_heatmap(ic_stats.get("monthly_ic", pd.Series(dtype=float)), name, is_flipped)
    decile_chart = R.plot_decile_nav(decile_data, name, is_flipped)
    ls_chart = R.plot_long_short_nav(decile_data, name, is_flipped)
    dist_chart = R.plot_ic_distribution(ic_series, name, is_flipped) if len(ic_series) > 0 else ""
    svg_ts = R.plot_ic_timeseries(ic_series, name, is_flipped)

    html = R.build_detail_html(
        factor_name=name, formula=fi["formula"], code=fi["code"],
        ic_stats=ic_stats, decile_data=decile_data, svg_timeseries=svg_ts,
        monthly_chart=monthly_chart, decile_chart=decile_chart,
        ls_chart=ls_chart, dist_chart=dist_chart,
        is_flipped=is_flipped, perf=perf,
        dsl=fi.get("dsl", ""), rationale=fi.get("rationale", ""),
        required_columns=fi.get("required_columns", ""),
        dsl_status=fi.get("dsl_status", "ok"), dsl_note=fi.get("dsl_note", ""),
        title=fi.get("title", ""), steps=fi.get("steps", []),
        manual_note=fi.get("manual_note", ""),
    )
    out_path = R.FACTORS_DIR / f"factor_{name}.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return {
        "name": name, "ok": True, "is_flipped": is_flipped,
        "mean_ic": ic_stats.get("mean_rankic", 0),
        "ls_sharpe": perf.get("ls_sharpe", 0),
        "ls_annual": perf.get("ls_annual", 0),
        "ls_winrate": perf.get("ls_winrate", 0),
        "turnover": perf.get("turnover", 0),
        "cost_bps": perf.get("cost_bps", R._COST_BPS),
        "n_periods": perf.get("n_periods", 0),
        "html_path": str(out_path),
    }


def main():
    print("=" * 70)
    print("smoke: 3 因子详情页重渲染（翻转同步 + 换手成本 + 胜率修复）")
    print("=" * 70)
    for name in SAMPLE:
        t0 = time.time()
        try:
            r = render_one(name)
            if r["ok"]:
                print(f"\n[{name}] is_flipped={r['is_flipped']} 耗时{time.time()-t0:.1f}s")
                print(f"  Mean RankIC     : {r['mean_ic']:+.4f}")
                print(f"  LS Sharpe       : {r['ls_sharpe']:+.2f}")
                print(f"  LS 年化         : {r['ls_annual']*100:+.1f}%")
                print(f"  LS 日胜率       : {r['ls_winrate']*100:.1f}%")
                print(f"  Top10% 换手率   : {r['turnover']*100:.1f}% (日)")
                print(f"  双边费率假设    : {r['cost_bps']:.1f} bp")
                print(f"  交易日          : {r['n_periods']}")
                print(f"  HTML            : {r['html_path']}")
            else:
                print(f"\n[{name}] FAILED: {r.get('error')}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n[{name}] EXC: {e}")


if __name__ == "__main__":
    main()
