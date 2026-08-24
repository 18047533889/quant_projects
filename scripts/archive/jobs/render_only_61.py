#!/usr/bin/env python3
"""
只重渲染 61 个因子详情页 (用已存的 summary_stats + batch_metrics cache).
不重算指标; 不重落值.
"""
import sys, os, json, re, math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import io, base64

PROJECT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(PROJECT / "jobs"))
REPORT_DIR = PROJECT / "docs" / "reports" / "2026-08-23"
FACTORS_DIR = REPORT_DIR / "factors"

# 复用 rebuild_factor_detail_pages 中的渲染函数 (只读不重算)
import rebuild_factor_detail_pages as r
import fe_dsl_formula


def _get_cached_batch_metrics():
    """从 weekly_factor_backtest.py 落地的 parquet 重新读出 perf 数据"""
    # 不重算; 只读取最新的 summary_stats 缓存
    with open(REPORT_DIR / "summary_stats.json") as f:
        return json.load(f)


def render_one(name: str) -> bool:
    """渲染单个因子详情页 (不重算 perf)"""
    try:
        fi = r.extract_formula_info(name)
        is_flipped = fi["is_flipped"]

        # 没有 ic_stats / decile_data / 图表时, 详情页仍能展示公式 + 步骤
        # 找已存的 icStats (从既有 HTML 抽)
        html_path = FACTORS_DIR / f"factor_{name}.html"
        ic_series = []
        decile_data = {"dates": []}
        svg_ts = ""
        monthly_chart = ""
        decile_chart = ""
        ls_chart = ""
        dist_chart = ""
        perf = {}

        if html_path.exists():
            content = html_path.read_text()
            m = re.search(r"const icData\s*=\s*(\[.*?\]);", content, re.DOTALL)
            if m:
                try:
                    arr = json.loads(m.group(1))
                    ic_series = pd.Series([x.get("ic", 0) for x in arr])
                except Exception:
                    ic_series = pd.Series(dtype=float)
            # 抽 metrics 字典
            metric_map = {}
            for mm in re.finditer(r"<b[^>]*>\s*([+-]?\d+\.?\d*[a-zA-Z%]*)\s*</b>\s*<span>([^<]+)</span>",
                                  content):
                val = mm.group(1).replace("%", "")
                try:
                    v = float(val)
                except Exception:
                    continue
                metric_map[mm.group(2).strip()] = v
            perf = {
                "ls_sharpe": metric_map.get("LS Sharpe", 0) or 0,
                "ls_annual": (metric_map.get("LS 年化", 0) or 0) / 100,
                "ls_mdd": (metric_map.get("LS 最大回撤", 0) or 0) / 100,
                "ls_winrate": (metric_map.get("LS 日胜率", 0) or 0) / 100,
                "g10_annual": (metric_map.get("G10 (多头) 年化", 0) or 0) / 100,
                "g1_annual": (metric_map.get("G1 (空头) 年化", 0) or 0) / 100,
            }

        # ic_stats
        ic_stats = {}
        if len(ic_series) > 0:
            ic_stats = {
                "mean_rankic": float(ic_series.mean()),
                "std_ric": float(ic_series.std()),
                "rankic_ir": float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0,
                "win_rate": float((ic_series > 0).mean()),
                "n_periods": len(ic_series),
            }

        html = r.build_detail_html(
            factor_name=name,
            formula=fi["formula"],
            code=fi["code"],
            ic_stats=ic_stats,
            decile_data=decile_data,
            svg_timeseries=svg_ts,
            monthly_chart=monthly_chart,
            decile_chart=decile_chart,
            ls_chart=ls_chart,
            dist_chart=dist_chart,
            is_flipped=is_flipped,
            perf=perf,
            dsl=fi.get("dsl", ""),
            rationale=fi.get("rationale", ""),
            required_columns=fi.get("required_columns", ""),
            dsl_status=fi.get("dsl_status", "ok"),
            dsl_note=fi.get("dsl_note", ""),
            title=fi.get("title", ""),
            steps=fi.get("steps", []),
            manual_note=fi.get("manual_note", ""),
        )

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False


def main():
    names = sorted([p.stem.replace("factor_", "")
                    for p in FACTORS_DIR.glob("factor_*.html")])
    print(f"重渲染 {len(names)} 个因子详情页...")
    ok = 0; fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(render_one, n): n for n in names}
        for fut in as_completed(futs):
            n = futs[fut]
            if fut.result():
                ok += 1
            else:
                fail += 1
            if (ok + fail) % 10 == 0:
                print(f"  进度 {ok+fail}/{len(names)}, 成功 {ok}, 失败 {fail}, 用时 {time.time()-t0:.1f}s")
    print(f"完成: {ok}/{len(names)}, 用时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
