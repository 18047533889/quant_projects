#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
456 因子全量详情页渲染（逐因子，内存友好）。

复用 rebuild_factor_detail_pages 的绘图/HTML 函数 + single_factor_metrics 的单因子指标。
输出到 reports/2026-08-23/factors/（456 页，含 61 本周）。

用法：FACTOR_SET=all /tmp/fe2/bin/python render_all_456.py
"""
import sys, os, json, time, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

PROJECT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts" / "archive" / "jobs"))
sys.path.insert(0, str(PROJECT / "jobs"))

os.environ["FACTOR_SET"] = "all"
os.environ["FACTOR_REPORT_DIR"] = str(PROJECT / "factor_engine" / "docs" / "reports" / "2026-08-23")

import rebuild_factor_detail_pages as R
from single_factor_metrics import compute_factor_metrics_single

# 公式源合并：61 因子优先用 aacd 精确版 formula_lqtp.json，其余用 formula_lqtp_all.json
_ACCD_61 = {}
try:
    import json as _json
    _raw = _json.loads(Path("/home/sunhaiwei/factor_delivery_converted/formula_lqtp.json").read_text())
    for _k, _rec in (_raw.items() if isinstance(_raw, dict) else []):
        _rec = dict(_rec)
        _rec["dsl"] = _rec.get("lqtp_formula") or _rec.get("fe_formula") or _rec.get("custom_dsl") or ""
        _rec["can_use_factor_engine"] = (_rec.get("status") == "ok")
        _ACCD_61[_k] = _rec
except Exception:
    pass
for _k, _rec in _ACCD_61.items():
    R._LQTP_RECORDS[_k] = _rec

FV_DIR = PROJECT / "weekly_backtest_output" / "factor_matrices_all"


def _write_placeholder(name: str, fi: dict, reason: str) -> None:
    """写一个占位详情页（数据不可评估时）。"""
    R.FACTORS_DIR.mkdir(parents=True, exist_ok=True)
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{name}</title>
<style>
body {{ font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif; background:#eef2f7; color:#0f172a; margin:0 }}
main {{ max-width:900px; margin:0 auto; padding:24px }}
.card {{ background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:20px; margin:16px 0 }}
h1 {{ color:#1e4d8c }} h2 {{ color:#1e4d8c; font-size:1rem; border-bottom:1px solid #e2e8f0; padding-bottom:6px }}
.notice {{ background:#fef2f2; border:1px solid #fecaca; border-left:4px solid #dc2626; padding:12px 16px; border-radius:8px; color:#991b1b }}
code {{ background:#f1f5f9; padding:2px 6px; border-radius:4px }}
.formula {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:14px; font-family:monospace; word-break:break-all }}
</style></head><body><main>
<a href="../index.html" style="color:#93c5fd">&#8592; 返回汇总</a>
<h1>{name}</h1>
<div class="card"><h2>⚠️ 数据不可评估</h2>
<div class="notice">{reason}</div>
<p style="color:#64748b">该因子值在回测区间内无有效截面差异（面板常数或全部缺失），无法计算 RankIC / 十分层等指标。</p></div>
<div class="card"><h2>📐 因子表达式 DSL</h2>
<div class="formula">{fi.get("dsl","") or "（无公式）"}</div>
<div style="margin-top:8px;font-size:0.8rem;color:#64748b">{fi.get("dsl_note","")}</div></div>
<div class="card"><h2>🐍 原始 Python 代码</h2>
<pre style="background:#0f172a;color:#e2e8f0;padding:14px;border-radius:8px;overflow-x:auto;font-size:0.8rem">{fi.get("code","") or "（无代码）"}</pre></div>
</main></body></html>
"""
    with open(R.FACTORS_DIR / f"factor_{name}.html", "w", encoding="utf-8") as f:
        f.write(html)


def render_one(name: str) -> tuple:
    try:
        fi = R.extract_formula_info(name)
        is_flipped = fi["is_flipped"]

        # 加载因子矩阵
        mat_path = FV_DIR / f"{name}.parquet"
        if not mat_path.exists():
            # 兜底找 factor_<name>
            alt = FV_DIR / f"factor_{name}.parquet"
            if not alt.exists():
                fi = R.extract_formula_info(name)
                _write_placeholder(name, fi, "无因子值文件（该因子未落值）")
                return (name, False, {"error": "无因子值文件", "placeholder": True})
            mat_path = alt
        matrix = pd.read_parquet(mat_path)
        if matrix.shape[1] == 0:
            fi = R.extract_formula_info(name)
            _write_placeholder(name, fi, "因子值全 NaN（无法计算指标）")
            return (name, False, {"error": "因子全NaN", "placeholder": True})
        if not isinstance(matrix.index, pd.DatetimeIndex):
            matrix.index = pd.to_datetime(matrix.index)

        # 单因子指标
        fm = compute_factor_metrics_single(name, matrix)
        if not fm or fm.get("perf", {}).get("n_periods", 0) < 20:
            # 数据不可评估（面板常数 / 全NaN / 无有效IC）→ 写占位详情页
            fi = R.extract_formula_info(name)
            _write_placeholder(name, fi, "数据不足（面板常数/全NaN，无法计算 RankIC）")
            return (name, False, {"error": "数据不足", "placeholder": True})

        ic_series = fm.get("ic_series", pd.Series(dtype=float))
        ic_stats = R.compute_ic_stats(ic_series)
        # 翻转处理：_flipped 或 rankic<0
        if is_flipped and ic_series.mean() < 0:
            ic_series = -ic_series
            ic_stats = R.compute_ic_stats(ic_series)
            fm = dict(fm)
            fm["perf"] = dict(fm["perf"])
            for k in ["ls_sharpe", "ls_annual", "ls_winrate", "g10_annual", "g1_annual", "g10_sharpe", "g1_sharpe"]:
                fm["perf"][k] = -fm["perf"].get(k, 0)
            fm["perf"]["ls_mdd"] = abs(fm["perf"].get("ls_mdd", 0))
            fm["mean_ic"] = -fm.get("mean_ic", 0)
            fm["ic_ir"] = -fm.get("ic_ir", 0)

        decile_raw = fm.get("decile_navs", {})
        dates_out = fm.get("dates_out", [])
        decile_data = {"dates": dates_out}
        for k, v in decile_raw.items():
            decile_data[k] = v
        perf = fm.get("perf", {})

        # 图表（ic_series 可能为空/常量 → 各绘图函数自身容错）
        monthly_chart = R.plot_ic_monthly_heatmap(ic_stats.get("monthly_ic", pd.Series(dtype=float)), name) if ic_stats else ""
        decile_chart = R.plot_decile_nav(decile_data, name) if decile_data else ""
        ls_chart = R.plot_long_short_nav(decile_data, name) if decile_data else ""
        dist_chart = ""
        if len(ic_series) > 0 and np.isfinite(ic_series.dropna()).any():
            try:
                dist_chart = R.plot_ic_distribution(ic_series, name)
            except Exception:
                dist_chart = ""
        svg_ts = R.plot_ic_timeseries(ic_series, name) if len(ic_series) > 0 else ""

        html = R.build_detail_html(
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
        out_path = R.FACTORS_DIR / f"factor_{name}.html"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        return (name, True, {
            "name": name,
            "mean_ic": fm.get("mean_ic", 0),
            "ic_ir": fm.get("ic_ir", 0),
            "ls_sharpe": perf.get("ls_sharpe", 0),
            "ls_annual": perf.get("ls_annual", 0),
            "ls_mdd": perf.get("ls_mdd", 0),
            "win_rate": fm.get("win_rate", 0),
            "g10_annual": perf.get("g10_annual", 0),
            "g1_annual": perf.get("g1_annual", 0),
            "is_flipped": is_flipped,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (name, False, {"error": str(e)[:200]})


def main():
    R.FACTORS_DIR.mkdir(parents=True, exist_ok=True)
    names = sorted([f.stem for f in FV_DIR.glob("*.parquet")]) if FV_DIR.exists() else []
    print(f"[456render] 因子数: {len(names)}")
    if not names:
        print("[456render] factor_matrices_all 为空，先跑 extend_all_456")
        return

    n_workers = min(8, max(2, (os.cpu_count() or 4) // 4))
    results = []
    all_stats = []
    errors = []
    done = 0

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(render_one, n): n for n in names}
        for future in as_completed(futures):
            name, ok, stats = future.result()
            done += 1
            if ok:
                results.append(name)
                stats["name"] = name
                all_stats.append(stats)
            else:
                if stats.get("placeholder"):
                    # 占位页已写，仍记入统计（无指标）
                    stats["name"] = name
                    stats["mean_ic"] = 0
                    stats["ic_ir"] = 0
                    stats["ls_sharpe"] = 0
                    stats["ls_annual"] = 0
                    stats["ls_mdd"] = 0
                    stats["win_rate"] = 0
                    stats["g10_annual"] = 0
                    stats["g1_annual"] = 0
                    stats["placeholder"] = True
                    all_stats.append(stats)
                else:
                    errors.append(name)
                    print(f"  [ERR] {name}: {stats.get('error','')}")
            if done % 50 == 0 or done == len(names):
                print(f"  [{done}/{len(names)}] ok={len(results)} placeholder={sum(1 for s in all_stats if s.get('placeholder'))} err={len(errors)} 耗时{time.time()-t0:.0f}s")

    # summary_stats
    R.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(R.REPORT_DIR / "summary_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2, default=str)

    # 排名输出
    print("\n===== Top 15 (LS Sharpe) =====")
    for s in sorted(all_stats, key=lambda x: x.get("ls_sharpe", 0), reverse=True)[:15]:
        print(f"  {s['name']:<50} {s['ls_sharpe']:>+7.2f} {s['ls_annual']*100:>+6.1f}% {s['mean_ic']:>+8.4f} {s['ic_ir']:>+7.2f} {s['ls_mdd']*100:>+6.1f}%")
    print("\n===== Bottom 10 =====")
    for s in sorted(all_stats, key=lambda x: x.get("ls_sharpe", 0))[:10]:
        print(f"  {s['name']:<50} {s['ls_sharpe']:>+7.2f} {s['ls_annual']*100:>+6.1f}% {s['mean_ic']:>+8.4f} {s['ic_ir']:>+7.2f} {s['ls_mdd']*100:>+6.1f}%")

    print(f"\n[456render] 完成: ok={len(results)} err={len(errors)} 共{len(names)} 耗时{time.time()-t0:.0f}s")
    if errors:
        print(f"[456render] 失败因子: {errors[:20]}")


if __name__ == "__main__":
    main()
