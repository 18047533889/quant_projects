#!/usr/bin/env python3
"""
恢复丢失的 31 个因子 + 重算翻转 + 合并 + 生成完整报告
"""

from __future__ import annotations
import sys, os, json, tempfile as _tempfile, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "vectorbt_qs"))

MAX_WORKERS = min(12, (os.cpu_count() or 4) + 4)
import weekly_factor_backtest as _wfb


def main():
    out_dir = _PROJECT_ROOT / "weekly_backtest_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Step 1: 找出真正丢失的因子
    # ============================================================
    fv = pd.read_parquet(str(out_dir / "factor_values.parquet"))
    in_parquet = set(fv.columns.get_level_values(0).unique())

    with open(out_dir / "backtest_results.json") as f:
        results = json.load(f)

    # 真正丢失的：results里有，但没有在parquet里
    # 排除 _flipped（翻转版是动态生成的）
    lost_names = [
        k for k in results.keys()
        if not k.endswith("_flipped") and k not in in_parquet
    ]

    minute_tools_set = {
        "factor_amount_weighted_impact",
        "factor_high_volume_amount_weighted_squared_impact",
        "factor_high_volume_volume_weighted_squared_impact",
        "factor_high_volume_amount_weighted_impact",
        "factor_high_volume_squared_return_weighted_impact",
        "factor_vwap_squared_deviation_intensity",
        "factor_vwap_deviation_intensity_squared_amount_weighted",
    }
    to_compute = [n for n in lost_names if n not in minute_tools_set]

    print(f"[丢失] {len(lost_names)} 个从 parquet 丢失, 其中:")
    print(f"  - {len(minute_tools_set & set(lost_names))} 个需要分钟数据（无法恢复）")
    print(f"  - {len(to_compute)} 个有 code，可以补算:")
    for n in to_compute:
        sh = results.get(n, {}).get("stats", {}).get("Sharpe Ratio", 0)
        ic = results.get(n, {}).get("stats", {}).get("IC", 0)
        print(f"    Sharpe={sh:+.2f}  IC={ic:+.4f}  | {n}")

    # ============================================================
    # Step 2: 加载这批因子的 metadata + code
    # ============================================================
    import glob as _glob
    csv_path = _PROJECT_ROOT / "factor_delivery_converted" / "converted_factors.csv"
    df = pd.read_csv(csv_path)
    df["rank_ic"] = pd.to_numeric(df["rank_ic"], errors="coerce")
    df_indexed = df.set_index("factor_name")

    code_map = {}
    for jf in _glob.glob(str(_PROJECT_ROOT / "factor_delivery_converted" / "factors_combined" / "factor_*.json")):
        try:
            with open(jf) as fh:
                d = json.load(fh)
            nm = d.get("factor_name", "")
            if nm:
                code_map[nm] = d.get("code", "")
        except Exception:
            pass

    factors = []
    for nm in to_compute:
        if nm not in df_indexed.index:
            continue
        row = df_indexed.loc[nm]
        code = code_map.get(nm, "")
        if code and "NotImplementedError" not in code and len(code) > 50:
            factors.append({
                "factor_name": nm,
                "rank_ic": row.get("rank_ic", 0),
                "code": code,
                "status": "EXEC",
                "lqtp_formula": row.get("lqtp_formula", ""),
                "fe_formula": row.get("fe_formula", ""),
            })

    if not factors:
        print("\n[完成] 没有需要补算的因子")
        return

    print(f"\n[因子] {len(factors)} 个可补算")

    # ============================================================
    # Step 3: 加载行情
    # ============================================================
    symbols = _wfb._get_stock_universe_from_sample(symbols_limit=500)
    dates = pd.date_range("2024-01-02", "2025-12-31", freq="B")
    market_data = _wfb.load_market_data(symbols, "2024-01-02", "2025-12-31")
    valid_dates = market_data["close"].index.intersection(dates)
    print(f"[数据] {len(valid_dates)} 天, {len(symbols)} 只股票")

    # ============================================================
    # Step 4: 计算因子值
    # ============================================================
    print(f"\n[计算] {len(factors)} 个因子值...")
    new_fv = _wfb.compute_factor_values(
        factors, market_data, symbols, valid_dates, n_workers=MAX_WORKERS,
    )
    print(f"[计算] 完成: {new_fv.shape}")

    # 保存因子值
    combined_fv = pd.concat([fv, new_fv], axis=1)
    combined_fv.to_parquet(str(out_dir / "factor_values.parquet"))
    print(f"[合并] 因子值: {fv.shape} + {new_fv.shape} -> {combined_fv.shape}")

    # ============================================================
    # Step 5: 回测（并行）
    # ============================================================
    fwd_rets_path = Path(_tempfile.gettempdir()) / "fwd_rets_recover.parquet"
    market_data["close"].to_parquet(str(fwd_rets_path))

    print(f"\n[回测] {len(factors)} 个...")
    new_results = {}
    fv_data_map = {}
    for col_name in new_fv.columns.get_level_values(0).unique():
        sub = new_fv[col_name]
        f_dict = next((f for f in factors if f["factor_name"] == col_name), None)
        if f_dict is None:
            f_dict = {"factor_name": col_name, "code": "", "status": "EXEC"}
        fv_data_map[col_name] = (sub.to_dict(), f_dict)

    futures = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for name, (fv_arr, f_dict) in fv_data_map.items():
            future = pool.submit(
                _wfb._backtest_single_factor,
                (fv_arr, f_dict),
                str(fwd_rets_path),
                10_000_000.0, 0.2, 0.2,
            )
            futures[future] = name

        for future in as_completed(futures):
            name = futures[future]
            try:
                _, f_dict, result = future.result(timeout=300)
                new_results[name] = result
                if result.get("success"):
                    s = result["stats"]
                    print(f"  ✓ {name}: Sharpe={s.get('Sharpe Ratio', 0):+.2f}, IC={s.get('IC', 0):+.4f}")
                else:
                    print(f"  ✗ {name}: {result.get('error', '?')[:60]}")
            except Exception as e:
                print(f"  ✗ {name}: {e}")

    # ============================================================
    # Step 6: 合并结果 + 翻转
    # ============================================================
    all_results = {**results, **new_results}

    # 翻转 IC < 0 的因子
    flip_map = {}
    for name, res in all_results.items():
        if name.endswith("_flipped") or not res.get("success"):
            continue
        ic = res["stats"].get("IC", 0)
        if ic < -0.0001:
            flip_name = name + "_flipped"
            if flip_name not in all_results:
                flip_map[name] = flip_name

    print(f"\n[翻转] {len(flip_map)} 个 IC < 0 的因子:")
    for orig, flipped in flip_map.items():
        orig_res = all_results[orig]
        s = orig_res["stats"]
        # 翻转 IC 和 Sharpe（对称翻转）
        flip_stats = {
            **s,
            "Sharpe Ratio": -s["Sharpe Ratio"],
            "Annualized Return (%)": -s["Annualized Return (%)"],
            "Max Drawdown (%)": abs(s["Max Drawdown (%)"]) if s["Max Drawdown (%)"] < 0 else s["Max Drawdown (%)"],
            "IC": -s["IC"],
            "Rank IC": -s["Rank IC"],
        }
        all_results[flipped] = {
            "success": True,
            "stats": flip_stats,
        }
        print(f"  {orig}: Sharpe={s['Sharpe Ratio']:+.2f} -> {flipped}: Sharpe={flip_stats['Sharpe Ratio']:+.2f}")

    # 保存最终结果
    with open(out_dir / "backtest_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[保存] backtest_results.json（共 {len(all_results)} 个因子）")

    # ============================================================
    # Step 7: 生成 HTML 报告
    # ============================================================
    positive = {k: v for k, v in all_results.items()
                if v.get("success") and v["stats"].get("Sharpe Ratio", 0) > 0}
    negative = {k: v for k, v in all_results.items()
                if v.get("success") and v["stats"].get("Sharpe Ratio", 0) <= 0}
    minute_failed = {k: v for k, v in results.items()
                     if k in minute_tools_set and k not in new_results}

    def sort_key(item):
        return -(item[1].get("stats", {}).get("Sharpe Ratio", 0) or 0)

    positive = dict(sorted(positive.items(), key=sort_key))
    negative = dict(sorted(negative.items(), key=sort_key))

    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>因子周报 {today}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:"PingFang SC","Microsoft YaHei",Arial,sans-serif; background:#0f1419; color:#e6e8ea; padding:20px; }}
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
tr.flipped td {{ color:#8b949e; }}
tr.minute_fail td {{ color:#f4212e; opacity:0.6; }}
.pos {{ color:#3ba55c; font-weight:bold; }}
.neg {{ color:#f4212e; }}
.sharge {{ font-weight:bold; }}
</style>
</head>
<body>
<h1>因子周报</h1>
<div class="subtitle">生成时间: {today} | 区间: 2024-01-02 ~ 2025-12-31 | IC负自动翻转</div>

<div class="metrics">
  <div class="metric"><b>{len(all_results)}</b><span>总因子</span></div>
  <div class="metric"><b>{len(positive)}</b><span>正夏普</span></div>
  <div class="metric"><b>{len(minute_failed)}</b><span>失败(需分钟数据)</span></div>
</div>

<div class="section">
<h2>正 Sharpe（{len(positive)} 个）</h2>
<table>
<thead><tr><th>#</th><th>因子</th><th>Sharpe</th><th>年化</th><th>最大回撤</th><th>IC</th><th>RankIC</th><th>胜率</th></tr></thead>
<tbody>
"""
    for i, (name, res) in enumerate(positive.items(), 1):
        s = res["stats"]
        flipped = name.endswith("_flipped")
        html += f"""<tr class="{'flipped' if flipped else ''}">
  <td>{i}</td>
  <td>{name}{" [FLIP]" if flipped else ""}</td>
  <td class="pos sharge">{s.get("Sharpe Ratio", 0):+.2f}</td>
  <td class="pos">{s.get("Annualized Return (%)", 0):+.1f}%</td>
  <td class="neg">{s.get("Max Drawdown (%)", 0):.1f}%</td>
  <td class="pos">{s.get("IC", 0):+.4f}</td>
  <td class="pos">{s.get("Rank IC", 0):+.4f}</td>
  <td>{s.get("Win Rate (%)", 0):.1f}%</td>
</tr>"""

    html += f"""</tbody></table></div>

<div class="section">
<h2>负 Sharpe（{len(negative)} 个）</h2>
<table>
<thead><tr><th>#</th><th>因子</th><th>Sharpe</th><th>年化</th><th>最大回撤</th><th>IC</th><th>RankIC</th><th>胜率</th></tr></thead>
<tbody>
"""
    for i, (name, res) in enumerate(negative.items(), 1):
        s = res["stats"]
        flipped = name.endswith("_flipped")
        html += f"""<tr class="{'flipped' if flipped else ''}">
  <td>{i}</td>
  <td>{name}{" [FLIP]" if flipped else ""}</td>
  <td class="neg sharge">{s.get("Sharpe Ratio", 0):+.2f}</td>
  <td class="{'pos' if s.get("Annualized Return (%)", 0) > 0 else 'neg'}">{s.get("Annualized Return (%)", 0):+.1f}%</td>
  <td class="neg">{s.get("Max Drawdown (%)", 0):.1f}%</td>
  <td class="{'pos' if s.get("IC", 0) > 0 else 'neg'}">{s.get("IC", 0):+.4f}</td>
  <td class="{'pos' if s.get("Rank IC", 0) > 0 else 'neg'}">{s.get("Rank IC", 0):+.4f}</td>
  <td>{s.get("Win Rate (%)", 0):.1f}%</td>
</tr>"""

    if minute_failed:
        html += f"""</tbody></table></div>
<div class="section">
<h2>失败因子（{len(minute_failed)} 个 — 需分钟数据，日频无法计算）</h2>
<table>
<thead><tr><th>因子</th><th>说明</th></tr></thead>
<tbody>
"""
        for name in sorted(minute_failed.keys()):
            html += f"<tr class='minute_fail'><td>{name}</td><td>依赖 minute_tools，需分钟级行情数据</td></tr>"

    html += "</tbody></table></div></body></html>"

    report_path = out_dir / f"weekly_factor_report_{datetime.now().strftime('%Y%m%d')}_final.html"
    Path(report_path).write_text(html)

    # ============================================================
    # 汇总
    # ============================================================
    pos = sum(1 for v in all_results.values() if v.get("success") and v["stats"].get("Sharpe Ratio", 0) > 0)
    total_ok = sum(1 for v in all_results.values() if v.get("success"))
    new_pos = sum(1 for name, v in all_results.items() if v.get("success") and v["stats"].get("Sharpe Ratio", 0) > 0 and name.endswith("_flipped"))

    print("\n" + "=" * 60)
    print("  恢复完成")
    print(f"  新增补算: {len(new_results)} 个")
    print(f"  新增成功: {sum(1 for v in new_results.values() if v.get('success'))} 个")
    print(f"  新增翻转: {len(flip_map)} 个")
    print(f"  累计成功: {total_ok} 个 | 正 Sharpe: {pos} 个")
    print(f"  其中翻转版贡献: {new_pos} 个正 Sharpe")
    print(f"  分钟数据失败: {len(minute_failed)} 个")
    print(f"  报告: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
