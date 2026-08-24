#!/usr/bin/env python3
"""
增量因子回测 — 追加新因子到现有报告
复用 weekly_factor_backtest.py 的因子计算和回测逻辑
"""

from __future__ import annotations
import sys, os, json, time, glob, tempfile, argparse, tempfile as _tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "vectorbt_qs"))

MAX_WORKERS = min(12, (os.cpu_count() or 4) + 4)

# 复用原脚本的函数
import weekly_factor_backtest as _wfb

# ============================================================
# 因子元数据加载（带排除）
# ============================================================
def load_factor_metadata(min_rank_ic=0.02, exclude=None):
    exclude = exclude or set()
    csv_path = _PROJECT_ROOT / "factor_delivery_converted" / "converted_factors.csv"
    df = pd.read_csv(csv_path)
    df["rank_ic"] = pd.to_numeric(df["rank_ic"], errors="coerce")
    df["ic"] = pd.to_numeric(df["ic"], errors="coerce")
    df = df[df["rank_ic"] > min_rank_ic].copy()

    import json as _json
    code_map = {}
    json_dir = _PROJECT_ROOT / "factor_delivery_converted" / "factors_combined"
    if json_dir.exists():
        for jf in glob.glob(str(json_dir / "factor_*.json")):
            try:
                with open(jf) as fh:
                    d = _json.load(fh)
                nm = d.get("factor_name", "")
                if nm:
                    code_map[nm] = d.get("code", "")
            except Exception:
                pass

    def classify(row):
        notes = str(row.get("lqtp_notes", ""))
        formula = row.get("lqtp_formula")
        lqtp_ok = ("Unconverted" not in notes) and (formula is not None and str(formula) != "")
        notes2 = str(row.get("fe_notes", ""))
        formula2 = row.get("fe_formula")
        fe_ok = ("Unconverted" not in notes2) and (formula2 is not None and str(formula2) != "")
        nm = row.get("factor_name", "")
        code = code_map.get(nm, "")
        has_code = bool(code) and "NotImplementedError" not in code and len(code) > 50
        if has_code:
            return "EXEC"
        elif lqtp_ok:
            return "LQTP"
        elif fe_ok:
            return "FE"
        else:
            return "NONE"

    df["status"] = df.apply(classify, axis=1)
    prio = {"EXEC": 0, "LQTP": 1, "FE": 2, "NONE": 3}
    df["_prio"] = df["status"].map(prio)
    df = df.sort_values(["_prio", "rank_ic"], ascending=[True, False])
    df = df[~df["factor_name"].isin(exclude)]

    # 注入 code
    records = df.to_dict("records")
    for f in records:
        f["code"] = code_map.get(f["factor_name"], "")

    return records


# ============================================================
# HTML 报告生成
# ============================================================
def generate_html_report(all_results, config, output_path):
    positive = {k: v for k, v in all_results.items()
                if v.get("success") and v["stats"].get("Sharpe Ratio", 0) > 0}
    negative = {k: v for k, v in all_results.items()
                if v.get("success") and v["stats"].get("Sharpe Ratio", 0) <= 0}
    failed = {k: v for k, v in all_results.items() if not v.get("success")}

    def sort_key(item):
        return -(item[1].get("stats", {}).get("Sharpe Ratio", 0) or 0)

    positive = dict(sorted(positive.items(), key=sort_key))
    negative = dict(sorted(negative.items(), key=sort_key))

    today = datetime.now().strftime("%Y-%m-%d")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>因子周报 {today}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'PingFang SC','Microsoft YaHei',Arial,sans-serif; background:#0f1419; color:#e6e8ea; padding:20px; }}
h1 {{ color:#1d9bf0; font-size:1.5em; margin-bottom:4px; }}
.subtitle {{ color:#71767b; font-size:0.85em; margin-bottom:20px; }}
.metrics {{ display:flex; gap:20px; flex-wrap:wrap; margin-bottom:24px; }}
.metric {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px 24px; text-align:center; }}
.metric b {{ display:block; font-size:2em; color:#1d9bf0; }}
.metric span {{ font-size:0.8em; color:#71767b; }}
.section {{ margin-bottom:32px; }}
.section h2 {{ color:#e6e8ea; font-size:1.1em; border-left:4px solid #1d9bf0; padding-left:12px; margin-bottom:12px; }}
table {{ width:100%; border-collapse:collapse; background:#161b22; border-radius:8px; overflow:hidden; }}
th {{ background:#21262d; color:#71767b; font-size:0.75em; text-align:left; padding:10px 12px; font-weight:400; }}
td {{ padding:10px 12px; font-size:0.85em; border-top:1px solid #21262d; }}
tr:hover td {{ background:#1c2128; }}
.pos {{ color:#3ba55c; font-weight:bold; }}
.neg {{ color:#f4212e; }}
.neu {{ color:#71767b; }}
.sharge {{ font-weight:bold; }}
</style>
</head>
<body>
<h1>因子周报</h1>
<div class="subtitle">生成时间: {today} | 区间: {config['start_date']} ~ {config['end_date']}</div>

<div class="metrics">
  <div class="metric"><b>{len(all_results)}</b><span>总因子数</span></div>
  <div class="metric"><b>{len(positive)}</b><span>正夏普</span></div>
  <div class="metric"><b>{len(failed)}</b><span>失败</span></div>
</div>

<div class="section">
<h2>正 Sharpe 因子（{len(positive)} 个）</h2>
<table>
<thead><tr>
  <th>因子名称</th><th>Sharpe</th><th>年化收益</th><th>最大回撤</th><th>IC</th><th>RankIC</th>
</tr></thead>
<tbody>
"""
    for name, res in positive.items():
        s = res["stats"]
        sh = s.get("Sharpe Ratio", 0)
        ann = s.get("Annualized Return (%)", 0)
        dd = s.get("Max Drawdown (%)", 0)
        ic = s.get("IC", 0)
        ric = s.get("Rank IC", 0)
        html += f"""<tr>
  <td>{name}</td>
  <td class="pos sharge">{sh:+.2f}</td>
  <td class="{'pos' if ann>0 else 'neg'}">{ann:+.1f}%</td>
  <td class="neg">{dd:.1f}%</td>
  <td class="{'pos' if ic>0 else 'neg'}">{ic:+.4f}</td>
  <td class="{'pos' if ric>0 else 'neg'}">{ric:+.4f}</td>
</tr>"""

    html += """</tbody></table></div>

<div class="section">
<h2>负 Sharpe 因子（{} 个）</h2>
<table>
<thead><tr>
  <th>因子名称</th><th>Sharpe</th><th>年化收益</th><th>最大回撤</th><th>IC</th><th>RankIC</th>
</tr></thead>
<tbody>
""".format(len(negative))

    for name, res in negative.items():
        s = res["stats"]
        sh = s.get("Sharpe Ratio", 0)
        ann = s.get("Annualized Return (%)", 0)
        dd = s.get("Max Drawdown (%)", 0)
        ic = s.get("IC", 0)
        ric = s.get("Rank IC", 0)
        html += f"""<tr>
  <td>{name}</td>
  <td class="neg sharge">{sh:+.2f}</td>
  <td class="{'pos' if ann>0 else 'neg'}">{ann:+.1f}%</td>
  <td class="neg">{dd:.1f}%</td>
  <td class="{'pos' if ic>0 else 'neg'}">{ic:+.4f}</td>
  <td class="{'pos' if ric>0 else 'neg'}">{ric:+.4f}</td>
</tr>"""

    if failed:
        html += """</tbody></table></div>
<div class="section">
<h2>失败因子（{} 个）</h2>
<table>
<thead><tr><th>因子名称</th><th>错误信息</th></tr></thead>
<tbody>
""".format(len(failed))
        for name, res in failed.items():
            err = res.get("error", "unknown")[:100]
            html += f"<tr><td>{name}</td><td class='neg'>{err}</td></tr>"

    html += """</tbody></table></div></body></html>"""

    Path(output_path).write_text(html)
    print(f"[报告] 已生成 {output_path}")


# ============================================================
# 主流程
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-factors", type=int, default=30)
    p.add_argument("--min-rank-ic", type=float, default=0.02)
    p.add_argument("--start", type=str, default="2024-01-02")
    p.add_argument("--end", type=str, default="2025-12-31")
    p.add_argument("--workers", type=int, default=MAX_WORKERS)
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    out_dir = Path(args.output_dir or str(_PROJECT_ROOT / "weekly_backtest_output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "backtest_results.json"
    report_path = out_dir / f"weekly_factor_report_{datetime.now().strftime('%Y%m%d')}.html"

    # 加载已有结果
    existing_results = {}
    if results_path.exists():
        with open(results_path) as f:
            existing_results = json.load(f)
    done_names = set(existing_results.keys())
    print(f"[存量] 已回测 {len(existing_results)} 个因子")

    # 选取新因子
    print(f"\n[选取] 从 rank_ic > {args.min_rank_ic} 中排除已跑的 {len(done_names)} 个...")
    all_factors = load_factor_metadata(min_rank_ic=args.min_rank_ic, exclude=done_names)

    selected = []
    for status_pref in ["EXEC", "LQTP", "FE", "NONE"]:
        candidates = [f for f in all_factors if f["status"] == status_pref and f["factor_name"] not in done_names]
        remaining = args.n_factors - len(selected)
        if remaining <= 0:
            break
        selected += candidates[:remaining]

    if not selected:
        print("没有新因子可以回测了")
        return

    print(f"[选取] 选中 {len(selected)} 个新因子:")
    for i, f in enumerate(selected):
        print(f"  {i+1:2d}. {f['rank_ic']:.4f} [{f['status']:5s}] {f['factor_name']}")

    # 股票池 + 行情（复用原脚本逻辑）
    print("\n[数据] 加载行情数据...")
    symbols = _wfb._get_stock_universe_from_sample(symbols_limit=500)
    dates = pd.date_range(args.start, args.end, freq="B")
    market_data = _wfb.load_market_data(symbols, args.start, args.end)
    if market_data.get("close") is None or market_data["close"].empty:
        print("[错误] 无法加载行情数据")
        return
    valid_dates = market_data["close"].index.intersection(dates)
    print(f"[数据] {len(valid_dates)} 个有效交易日, {len(symbols)} 只股票")

    # 计算因子值（直接复用原脚本的函数）
    print(f"\n[计算] 开始计算 {len(selected)} 个因子值...")
    factor_values = _wfb.compute_factor_values(
        selected, market_data, symbols, valid_dates, n_workers=args.workers,
    )
    print(f"[计算] 完成，形状: {factor_values.shape}")

    # 保存因子值
    fv_path = out_dir / "factor_values.parquet"
    factor_values.to_parquet(str(fv_path))
    print(f"[保存] 因子值已保存: {fv_path}")

    # 回测（复用原脚本的 worker）
    fwd_rets_path = Path(_tempfile.gettempdir()) / "fwd_rets_incremental.parquet"
    market_data["close"].to_parquet(str(fwd_rets_path))

    print(f"\n[回测] 开始 {args.workers} 并行回测...")
    new_results = {}
    fv_data_map = {}
    for name in factor_values.columns.get_level_values(0).unique():
        sub = factor_values[name]
        sub_dict = next((f for f in selected if f["factor_name"] == name), None)
        if sub_dict:
            fv_data_map[name] = (sub.to_dict(), sub_dict)

    futures = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, (fv_arr, f_dict) in fv_data_map.items():
            future = pool.submit(
                _wfb._backtest_single_factor,
                (fv_arr, f_dict),
                str(fwd_rets_path),
                10_000_000.0,
                0.2, 0.2,
            )
            futures[future] = name

        for future in as_completed(futures):
            name = futures[future]
            try:
                _, f_dict, result = future.result(timeout=300)
                new_results[name] = result
                if result.get("success"):
                    s = result["stats"]
                    print(f"  ✓ {name}: Sharpe={s.get('Sharpe Ratio', 0):+.2f}, 年化={s.get('Annualized Return (%)', 0):+.1f}%, IC={s.get('IC', 0):+.4f}")
                else:
                    print(f"  ✗ {name}: {result.get('error', 'failed')[:80]}")
            except Exception as e:
                print(f"  ✗ {name}: {e}")

    # 合并结果
    all_results = {**existing_results, **new_results}
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[合并] 结果已更新到 {results_path}（共 {len(all_results)} 个因子）")

    # 生成报告
    config = {
        "start_date": args.start,
        "end_date": args.end,
        "n_factors": len(all_results),
    }
    generate_html_report(all_results, config, str(report_path))

    # 汇总
    new_ok = sum(1 for v in new_results.values() if v.get("success"))
    new_pos = sum(1 for v in new_results.values() if v.get("success") and v["stats"].get("Sharpe Ratio", 0) > 0)
    total_pos = sum(1 for v in all_results.values() if v.get("success") and v["stats"].get("Sharpe Ratio", 0) > 0)

    print("\n" + "=" * 55)
    print("  增量回测完成")
    print(f"  新增: {len(new_results)} 个 | 成功: {new_ok} 个 | 正 Sharpe: {new_pos} 个")
    print(f"  累计正 Sharpe: {total_pos} / {len(all_results)}")
    print(f"  报告: {report_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
