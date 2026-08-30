#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量重渲 470 详情页【主体】（rebuild_factor_detail_pages 的原始 IC/十分层/多空等主图 + 指标）。

背景：
  - 优化区块（render_optimized_pages.py 注入的对比图）已由协调方重渲完成（470/470）。
  - 但详情页主体是旧渲染（未翻转原始图、无成本口径、LS 胜率 0%/负值 bug）。
  - 本脚本用修复后的 rebuild_factor_detail_pages 重新生成主体，且【只写主体、不覆盖优化区块】。

实现（关键）：
  1. 先快照 470 页当前 HTML 里的「🧬 优化因子」区块（整个 <div class="card">...优化因子...</div>），
     以及页面内其它第三方注入段（用 alt="RankIC对比/十分层对比/多空对比" 的 <img>，都包在优化区块内）。
  2. 用修复后的 build_detail_html 生成新主体 HTML。
  3. 把快照的优化区块重新追加到新主体 </main> 前（与 render_optimized_pages 的注入位置一致）。
  4. 若新主体有优化区块残留（理论上不会），以快照覆盖。

数据/口径：
  - 收益：StockDailyBarAdj.AdjVwap，shift(-2)（vwap-to-vwap 后复权）
  - 成本：双边 10bp（TRADING_COST_BPS 可配），按 Top10% 组换手率 × 费率扣减
  - 翻转：is_flipped 读 optimized_meta.json；原始图按 is_flipped 加「(已翻正)」
  - 胜率：LS 翻转后 = 1 - 原胜率（镜像）
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

FV_DIR = PROJECT / "weekly_backtest_output" / "factor_matrices_all"
OPT_META = json.loads((PROJECT / "weekly_backtest_output" / "optimized_meta.json").read_text())

# 优化区块的正则：从 <div class="card"> 到匹配的 </div>\n</div>（区块内部有嵌套 div，用深度匹配）
_OPT_BLOCK_RE = re.compile(r'<div class="card">\s*<h2>🧬 优化因子.*?(?=</main>)', re.DOTALL)


def extract_opt_block(html: str) -> str:
    """提取详情页中的优化因子区块（含 3 张对比图）。"""
    m = _OPT_BLOCK_RE.search(html)
    if not m:
        return ""
    return m.group(0).strip()


def render_one(name: str) -> dict:
    try:
        fi = R.extract_formula_info(name)
        is_flipped = fi["is_flipped"] or R._is_flipped_by_meta(name)

        # 读取旧页面，快照优化区块
        html_path = R.FACTORS_DIR / f"factor_{name}.html"
        old_html = html_path.read_text(encoding="utf-8") if html_path.exists() else ""
        opt_block = extract_opt_block(old_html)

        mat_path = FV_DIR / f"{name}.parquet"
        if not mat_path.exists():
            return {"name": name, "ok": False, "error": "no matrix"}
        matrix = pd.read_parquet(mat_path)
        if not isinstance(matrix.index, pd.DatetimeIndex):
            matrix.index = pd.to_datetime(matrix.index)
        if matrix.shape[1] == 0:
            return {"name": name, "ok": False, "error": "empty matrix"}

        # 单因子指标（AdjVwap + shift(-2) + 扣成本）
        fm = compute_factor_metrics_single(name, matrix)
        if not fm or fm.get("perf", {}).get("n_periods", 0) < 20:
            # 数据不可评估（面板常数/全NaN）：写占位页（覆盖旧口径的旧主体，避免残留错误指标）
            fi2 = R.extract_formula_info(name)
            placeholder = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>{name}</title><style>
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
<div class="notice">因子值在回测区间内无有效截面差异（面板常数或全部缺失），无法计算 RankIC / 十分层等指标。</div>
<p style="color:#64748b">该因子在 AdjVwap 后复权口径（vwap-to-vwap, shift(-2)）下逐日因子值无横截面差异（每日仅 1 个取值），RankIC 无定义。</p></div>
<div class="card"><h2>📐 因子表达式 DSL</h2>
<div class="formula">{fi2.get("dsl","") or "（无公式）"}</div>
<div style="margin-top:8px;font-size:0.8rem;color:#64748b">{fi2.get("dsl_note","")}</div></div>
<div class="card"><h2>🐍 原始 Python 代码</h2>
<pre style="background:#0f172a;color:#e2e8f0;padding:14px;border-radius:8px;overflow-x:auto;font-size:0.8rem">{fi2.get("code","") or "（无代码）"}</pre></div>
</main></body></html>
"""
            if opt_block:
                placeholder = placeholder.replace("</main>", opt_block + "\n</main>", 1)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(placeholder)
            return {"name": name, "ok": True, "is_flipped": is_flipped,
                    "placeholder": True, "opt_restored": bool(opt_block)}

        ic_series = fm.get("ic_series", pd.Series(dtype=float)).copy()
        ic_series = ic_series.astype(float).replace([np.inf, -np.inf], np.nan)
        ic_stats = R.compute_ic_stats(ic_series)

        # 翻转同步：is_flipped → IC/方向/指标镜像
        if is_flipped and ic_series.mean() < 0:
            ic_series = -ic_series
            ic_stats = R.compute_ic_stats(ic_series)
            fm = dict(fm)
            fm["perf"] = dict(fm["perf"])
            for k in ["ls_sharpe", "ls_annual", "g10_annual", "g1_annual", "g10_sharpe", "g1_sharpe"]:
                fm["perf"][k] = -fm["perf"].get(k, 0)
            if "ls_winrate" in fm["perf"]:
                fm["perf"]["ls_winrate"] = 1.0 - fm["perf"].get("ls_winrate", 0)
            if "win_rate" in fm:
                fm["win_rate"] = 1.0 - fm.get("win_rate", 0)
            fm["perf"]["ls_mdd"] = abs(fm["perf"].get("ls_mdd", 0))
            dr = dict(fm.get("decile_navs", {}))
            gs = []
            for k in range(1, 11):
                key = f"G{k}"
                if key in dr and len(dr[key]) > 0:
                    gs.append((k, dr[key]))
            if len(gs) == 10:
                # 2026-08-29 fix: 保留原始 dates（compute_factor_metrics_single 的 decile_navs 无 dates，
                # 用 fm['dates_out'] 兜底），避免 mirror 后 dates 为空导致多空图缺失。
                nd_dates = dr.get("dates") or fm.get("dates_out") or []
                nd = {"dates": nd_dates}
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
            if k == "dates":
                # mirror 后 decile_navs 自带的 dates 优先（compute_factor_metrics_single 无此 key 时用 dates_out）
                decile_data["dates"] = v or fm.get("dates_out", [])
                continue
            decile_data[k] = v
        perf = fm.get("perf", {})

        # 生成主体图（带翻转标注）
        monthly_chart = R.plot_ic_monthly_heatmap(ic_stats.get("monthly_ic", pd.Series(dtype=float)), name, is_flipped)
        decile_chart = R.plot_decile_nav(decile_data, name, is_flipped)
        ls_chart = R.plot_long_short_nav(decile_data, name, is_flipped)
        dist_chart = ""
        if len(ic_series) > 0 and np.isfinite(ic_series.dropna()).any():
            try:
                dist_chart = R.plot_ic_distribution(ic_series, name, is_flipped)
            except Exception:
                dist_chart = ""
        svg_ts = R.plot_ic_timeseries(ic_series, name, is_flipped)

        new_html = R.build_detail_html(
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

        # 把快照的优化区块追加回新主体（保留其它 agent 的注入）
        if opt_block and "🧬 优化因子" not in new_html:
            new_html = new_html.replace("</main>", opt_block + "\n</main>", 1)
        # 兜底：若新主体意外带了旧优化区块，用快照覆盖
        elif opt_block:
            new_html = _OPT_BLOCK_RE.sub(lambda m: opt_block, new_html, count=1)

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(new_html)

        return {
            "name": name, "ok": True, "is_flipped": is_flipped,
            "opt_restored": bool(opt_block),
            "mean_ic": ic_stats.get("mean_rankic", 0),
            "ls_winrate": perf.get("ls_winrate", 0),
            "turnover": perf.get("turnover", 0),
        }
    except Exception as e:
        import traceback
        return {"name": name, "ok": False, "error": str(e)[:200]}


def main():
    from concurrent.futures import ThreadPoolExecutor, as_completed

    R.FACTORS_DIR.mkdir(parents=True, exist_ok=True)
    names = sorted([f.stem for f in FV_DIR.glob("*.parquet")]) if FV_DIR.exists() else []
    print(f"[patch] 待重渲主体: {len(names)} 页", flush=True)
    if not names:
        return

    t0 = time.time()
    n_workers = min(8, max(2, (os.cpu_count() or 4) // 4))
    print(f"[patch] workers={n_workers}", flush=True)

    done = ok = 0
    errors = []
    flipped_ok = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(render_one, n): n for n in names}
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            if r["ok"]:
                ok += 1
                if r["is_flipped"]:
                    flipped_ok.append(r)
            else:
                errors.append((r["name"], r.get("error", "")))
            if done % 50 == 0 or done == len(names):
                print(f"  [{done}/{len(names)}] ok={ok} err={len(errors)} 耗时{time.time()-t0:.0f}s", flush=True)

    print(f"\n[patch] 完成: ok={ok}/{len(names)} err={len(errors)} 耗时{time.time()-t0:.0f}s", flush=True)
    if errors:
        print("[patch] 失败:", errors[:10], flush=True)

    # 抽样验证：2 个翻转因子 + 统计
    print(f"\n[patch] is_flipped 重渲成功: {len(flipped_ok)}", flush=True)
    for r in flipped_ok[:2]:
        print(f"  {r['name']}: flip={r['is_flipped']} opt_restored={r['opt_restored']} "
              f"meanIC={r['mean_ic']:+.4f} LSwin={r['ls_winrate']*100:.1f}% turn={r['turnover']*100:.1f}%", flush=True)
    # 验证图注
    import glob
    n_has = 0
    for name in [r["name"] for r in flipped_ok[:2]]:
        p = R.FACTORS_DIR / f"factor_{name}.html"
        c = p.read_text(encoding="utf-8")
        has_ann = ("(已翻正)" in c or "IC 已翻正" in c) and "🧬 优化因子" in c
        print(f"  图注验证 {name}: 已翻正标注={'yes' if has_ann else 'NO'}", flush=True)
        n_has += 1


if __name__ == "__main__":
    main()
