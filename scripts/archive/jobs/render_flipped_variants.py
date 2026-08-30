#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重渲染 31 个 *_flipped 变体详情页（61 因子中带 flipped 后缀的），
用 aacd 精确版 LQTP 公式（含中文 steps + 翻转负号）。

这些页面在 456 渲染时未被覆盖（456 的 page_name 无 _flipped 后缀）。
数据源：factor_matrices/ (61 因子, 含 flipped 变体文件)。
"""
import sys, os, json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

PROJECT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts" / "archive" / "jobs"))
sys.path.insert(0, str(PROJECT / "jobs"))

os.environ["FACTOR_SET"] = "61"
os.environ["FACTOR_REPORT_DIR"] = str(PROJECT / "factor_engine" / "docs" / "reports" / "2026-08-23")

import rebuild_factor_detail_pages as R
from single_factor_metrics import compute_factor_metrics_single

# 覆盖 _LQTP_RECORDS 为 aacd 61 版（dict keyed by 因子名）
_LQTP_61 = {}
try:
    _raw = json.loads(Path("/home/sunhaiwei/factor_delivery_converted/formula_lqtp.json").read_text())
    for _k, _rec in (_raw.items() if isinstance(_raw, dict) else []):
        _rec = dict(_rec)
        _rec["dsl"] = _rec.get("lqtp_formula") or _rec.get("fe_formula") or _rec.get("custom_dsl") or ""
        _rec["can_use_factor_engine"] = (_rec.get("status") == "ok")
        _LQTP_61[_k] = _rec
except Exception:
    pass
R._LQTP_RECORDS.update(_LQTP_61)

FV61_DIR = PROJECT / "weekly_backtest_output" / "factor_matrices"


def _write_placeholder(name, fi, reason):
    R.FACTORS_DIR.mkdir(parents=True, exist_ok=True)
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"/>
<title>{name}</title><style>body{{font-family:system-ui;background:#eef2f7;color:#0f172a;margin:0}}main{{max-width:900px;margin:0 auto;padding:24px}}.card{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:20px;margin:16px 0}}h1{{color:#1e4d8c}}h2{{color:#1e4d8c;font-size:1rem}}</style></head>
<body><main><a href="../index.html" style="color:#93c5fd">&#8592; 返回汇总</a><h1>{name}</h1>
<div class="card"><h2>⚠️ 数据不可评估</h2><p style="color:#991b1b">{reason}</p></div>
<div class="card"><h2>📐 因子表达式 DSL</h2><pre style="background:#f8fafc;padding:12px;border-radius:8px">{fi.get("dsl","") or "（无公式）"}</pre></div>
</main></body></html>"""
    with open(R.FACTORS_DIR / f"factor_{name}.html", "w", encoding="utf-8") as f:
        f.write(html)


def render_flipped(name):
    try:
        fi = R.extract_formula_info(name)
        is_flipped = fi["is_flipped"] or R._is_flipped_by_meta(name)
        mat_path = FV61_DIR / f"{name}.parquet"
        if not mat_path.exists():
            _write_placeholder(name, fi, "无因子值文件")
            return name, False
        matrix = pd.read_parquet(mat_path)
        if matrix.shape[1] == 0:
            _write_placeholder(name, fi, "因子值全 NaN")
            return name, False
        fm = compute_factor_metrics_single(name, matrix)
        if not fm or fm.get("perf", {}).get("n_periods", 0) < 20:
            _write_placeholder(name, fi, "数据不足（面板常数/全NaN）")
            return name, False

        ic_series = fm.get("ic_series", pd.Series(dtype=float))
        ic_stats = R.compute_ic_stats(ic_series)
        if is_flipped and ic_series.mean() < 0:
            ic_series = -ic_series
            ic_stats = R.compute_ic_stats(ic_series)
            fm = dict(fm); fm["perf"] = dict(fm["perf"])
            for k in ["ls_sharpe","ls_annual","g10_annual","g1_annual","g10_sharpe","g1_sharpe"]:
                fm["perf"][k] = -fm["perf"].get(k, 0)
            if "ls_winrate" in fm["perf"]:
                fm["perf"]["ls_winrate"] = 1.0 - fm["perf"].get("ls_winrate", 0)
            if "win_rate" in fm:
                fm["win_rate"] = 1.0 - fm.get("win_rate", 0)
            fm["perf"]["ls_mdd"] = abs(fm["perf"].get("ls_mdd", 0))
            fm["mean_ic"] = -fm.get("mean_ic", 0)
            fm["ic_ir"] = -fm.get("ic_ir", 0)

        decile_raw = fm.get("decile_navs", {})
        dates_out = fm.get("dates_out", [])
        decile_data = {"dates": dates_out}
        for k, v in decile_raw.items():
            decile_data[k] = v
        perf = fm.get("perf", {})

        monthly_chart = R.plot_ic_monthly_heatmap(ic_stats.get("monthly_ic", pd.Series(dtype=float)), name, is_flipped) if ic_stats else ""
        decile_chart = R.plot_decile_nav(decile_data, name, is_flipped) if decile_data else ""
        ls_chart = R.plot_long_short_nav(decile_data, name, is_flipped) if decile_data else ""
        dist_chart = ""
        if len(ic_series) > 0 and np.isfinite(ic_series.dropna()).any():
            try:
                dist_chart = R.plot_ic_distribution(ic_series, name, is_flipped)
            except Exception:
                dist_chart = ""
        svg_ts = R.plot_ic_timeseries(ic_series, name, is_flipped) if len(ic_series) > 0 else ""

        html = R.build_detail_html(
            factor_name=name, formula=fi["formula"], code=fi["code"],
            ic_stats=ic_stats, decile_data=decile_data, svg_timeseries=svg_ts,
            monthly_chart=monthly_chart, decile_chart=decile_chart,
            ls_chart=ls_chart, dist_chart=dist_chart,
            is_flipped=is_flipped, perf=perf,
            dsl=fi.get("dsl",""), rationale=fi.get("rationale",""),
            required_columns=fi.get("required_columns",""),
            dsl_status=fi.get("dsl_status","ok"), dsl_note=fi.get("dsl_note",""),
            title=fi.get("title",""), steps=fi.get("steps", []),
            manual_note=fi.get("manual_note",""),
        )
        with open(R.FACTORS_DIR / f"factor_{name}.html", "w", encoding="utf-8") as f:
            f.write(html)
        return name, True
    except Exception as e:
        import traceback; traceback.print_exc()
        return name, False


def main():
    R.FACTORS_DIR.mkdir(parents=True, exist_ok=True)
    flipped = sorted([p.stem for p in FV61_DIR.glob("*_flipped.parquet")])
    print(f"[flip] 待重渲染 flipped 变体: {len(flipped)}")
    ok = 0
    for name in flipped:
        n, s = render_flipped(name)
        if s:
            ok += 1
        else:
            print(f"  [ERR] {name}")
    print(f"[flip] 完成: {ok}/{len(flipped)}")


if __name__ == "__main__":
    main()
