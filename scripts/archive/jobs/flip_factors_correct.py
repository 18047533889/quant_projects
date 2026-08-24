#!/usr/bin/env python3
"""
正确翻转因子：
直接从 parquet 读因子值，* -1 存为 _flipped 列，重新回测。
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
import weekly_factor_backtest as _wfb

MAX_WORKERS = min(12, (os.cpu_count() or 4) + 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default="2024-01-02")
    p.add_argument("--end", type=str, default="2025-12-31")
    p.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = p.parse_args()

    out_dir = _PROJECT_ROOT / "weekly_backtest_output"
    results_path = out_dir / "backtest_results.json"

    # 加载现有结果
    with open(results_path) as f:
        existing = json.load(f)

    # 加载因子值
    fv = pd.read_parquet(str(out_dir / "factor_values.parquet"))
    in_parquet = set(fv.columns.get_level_values(0).unique())

    # 找出 IC < 0 且在 parquet 里的原版因子
    neg_ic = []
    for name, res in existing.items():
        if name.endswith("_flipped"):
            continue
        if not res.get("success"):
            continue
        ic = res["stats"].get("IC", 0)
        if ic < -0.0001 and name in in_parquet:
            neg_ic.append((name, res, ic))

    print("=" * 60)
    print("  IC 翻转（因子值 * -1 后重新回测）")
    print("=" * 60)
    print(f"共 {len(neg_ic)} 个因子需要翻转\n")
    for nm, res, ic in neg_ic:
        print(f"  IC={ic:+.4f}  Sharpe={res['stats']['Sharpe Ratio']:+.2f}  | {nm}")

    if not neg_ic:
        print("没有需要翻转的因子")
        return

    # ============================================================
    # Step 1: 创建翻转列，保存 parquet
    # ============================================================
    print(f"\n[Step 1] 创建翻转列...")
    for nm, _, ic in neg_ic:
        flipped_name = nm + "_flipped"
        if nm in fv.columns.get_level_values(0):
            sub = fv[nm].copy()
            flipped = sub * -1
            # 添加 MultiIndex 列
            for sym in sub.columns:
                fv[(flipped_name, sym)] = flipped[sym].values
    fv.to_parquet(str(out_dir / "factor_values.parquet"))
    print(f"[Step 1] parquet 已更新: {fv.shape}")

    # ============================================================
    # Step 2: 加载行情
    # ============================================================
    symbols = _wfb._get_stock_universe_from_sample(symbols_limit=500)
    dates = pd.date_range(args.start, args.end, freq="B")
    market_data = _wfb.load_market_data(symbols, args.start, args.end)
    valid_dates = market_data["close"].index.intersection(dates)
    print(f"\n[Step 2] {len(valid_dates)} 交易日, {len(symbols)} 只股票")

    fwd_path = Path(_tempfile.gettempdir()) / "fwd_rets_flip.parquet"
    market_data["close"].to_parquet(str(fwd_path))

    # ============================================================
    # Step 3: 并行回测所有翻转因子
    # ============================================================
    print(f"\n[Step 3] 回测 {len(neg_ic)} 个翻转因子...")

    # 刷新 parquet 列到每个 worker
    fv_latest = pd.read_parquet(str(out_dir / "factor_values.parquet"))

    futures_map = {}
    results = {}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for nm, _, ic in neg_ic:
            flipped_name = nm + "_flipped"
            if flipped_name not in fv_latest.columns.get_level_values(0):
                print(f"  [跳过] {flipped_name} 不在 parquet")
                continue

            sub = fv_latest[flipped_name]
            f_dict = {
                "factor_name": flipped_name,
                "code": "",
                "status": "EXEC",
            }

            future = pool.submit(
                _wfb._backtest_single_factor,
                (sub.to_dict(), f_dict),
                str(fwd_path),
                10_000_000.0,
                0.2, 0.2,
            )
            futures_map[future] = (flipped_name, nm, ic)

        for future in as_completed(futures_map):
            flipped_name, orig_name, orig_ic = futures_map[future]
            try:
                _, f_dict, res = future.result(timeout=300)
                results[flipped_name] = res
                if res.get("success"):
                    s = res["stats"]
                    new_ic = s.get("IC", 0)
                    new_sh = s.get("Sharpe Ratio", 0)
                    new_ric = s.get("Rank IC", 0)
                    print(f"  ✓ {orig_name} -> {flipped_name}: "
                          f"IC {orig_ic:+.4f} → {new_ic:+.4f}, "
                          f"Sharpe {new_sh:+.2f}, RIC {new_ric:+.4f}")
                else:
                    print(f"  ✗ {flipped_name}: {res.get('error', '?')[:80]}")
            except Exception as e:
                print(f"  ✗ {flipped_name}: {e}")

    # ============================================================
    # Step 4: 合并结果
    # ============================================================
    final = dict(existing)
    for nm, res in results.items():
        final[nm] = res

    with open(results_path, "w") as f:
        json.dump(final, f, indent=2, default=str)

    # ============================================================
    # Step 5: 统计 + 报告
    # ============================================================
    pos = [(k, v) for k, v in final.items()
           if v.get("success") and v["stats"].get("Sharpe Ratio", 0) > 0]
    neg = [(k, v) for k, v in final.items()
           if v.get("success") and v["stats"].get("Sharpe Ratio", 0) <= 0]
    pos.sort(key=lambda x: -x[1]["stats"].get("Sharpe Ratio", 0))
    neg.sort(key=lambda x: -x[1]["stats"].get("Sharpe Ratio", 0))

    flipped_pos = sum(1 for k, v in final.items()
                     if k.endswith("_flipped") and v.get("success") and v["stats"].get("IC", 0) > 0)
    still_neg_ic = [(k, v) for k, v in final.items()
                   if v.get("success") and v["stats"].get("IC", 0) < -0.0001 and not k.endswith("_flipped")]

    print(f"\n{'=' * 60}")
    print(f"  完成")
    print(f"  正 Sharpe: {len(pos)} 个 | 翻转版 IC 变正: {flipped_pos} 个")
    if still_neg_ic:
        print(f"  ⚠ 原版 IC 仍负: {len(still_neg_ic)} 个")
    print(f"{'=' * 60}")

    # TOP 15
    print(f"\nTOP 15 正 Sharpe:")
    for i, (name, res) in enumerate(pos[:15], 1):
        s = res["stats"]
        tag = " [FLIP]" if name.endswith("_flipped") else ""
        ic = s.get("IC", 0)
        ric = s.get("Rank IC", 0)
        ic_tag = "pos" if ic > 0 else "neg"
        ric_tag = "pos" if ric > 0 else "neg"
        print(f"  {i:2d}. {s['Sharpe Ratio']:+.2f}  IC={ic:+.4f}  RIC={ric:+.4f}  | {name}{tag}")

    # 生成 HTML
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
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
.pos {{ color:#3ba55c; font-weight:bold; }}
.neg {{ color:#f4212e; }}
.sharge {{ font-weight:bold; }}
</style>
</head>
<body>
<h1>因子周报</h1>
<div class="subtitle">生成: {today} | 区间: 2024-01-02~2025-12-31 | IC负因子已翻转（因子值*-1）</div>

<div class="metrics">
  <div class="metric"><b>{len(final)}</b><span>总因子</span></div>
  <div class="metric"><b>{len(pos)}</b><span>正夏普</span></div>
  <div class="metric"><b>{flipped_pos}</b><span>翻转版</span></div>
</div>

<div class="section">
<h2>正 Sharpe（{len(pos)} 个）</h2>
<table>
<thead><tr><th>#</th><th>因子</th><th>Sharpe</th><th>年化</th><th>最大回撤</th><th>IC</th><th>RankIC</th></tr></thead>
<tbody>
"""
    for i, (name, res) in enumerate(pos, 1):
        s = res["stats"]
        fl = name.endswith("_flipped")
        ic, ric = s.get("IC", 0), s.get("Rank IC", 0)
        html += f"""<tr class="{'flipped' if fl else ''}">
  <td>{i}</td>
  <td>{name}{" [FLIP]" if fl else ""}</td>
  <td class="pos sharge">{s.get('Sharpe Ratio', 0):+.2f}</td>
  <td class="pos">{s.get('Annualized Return (%)', 0):+.1f}%</td>
  <td class="neg">{s.get('Max Drawdown (%)', 0):.1f}%</td>
  <td class="{'pos' if ic > 0 else 'neg'}">{ic:+.4f}</td>
  <td class="{'pos' if ric > 0 else 'neg'}">{ric:+.4f}</td>
</tr>"""

    html += f"""</tbody></table></div>

<div class="section">
<h2>负 Sharpe（{len(neg)} 个）</h2>
<table>
<thead><tr><th>#</th><th>因子</th><th>Sharpe</th><th>年化</th><th>IC</th><th>RankIC</th></tr></thead>
<tbody>
"""
    for i, (name, res) in enumerate(neg, 1):
        s = res["stats"]
        fl = name.endswith("_flipped")
        ic, ric = s.get("IC", 0), s.get("Rank IC", 0)
        html += f"""<tr class="{'flipped' if fl else ''}">
  <td>{i}</td>
  <td>{name}{" [FLIP]" if fl else ""}</td>
  <td class="neg sharge">{s.get('Sharpe Ratio', 0):+.2f}</td>
  <td class="{'pos' if s.get('Annualized Return (%)',0) > 0 else 'neg'}">{s.get('Annualized Return (%)', 0):+.1f}%</td>
  <td class="{'pos' if ic > 0 else 'neg'}">{ic:+.4f}</td>
  <td class="{'pos' if ric > 0 else 'neg'}">{ric:+.4f}</td>
</tr>"""

    html += "</tbody></table></div></body></html>"

    report_path = out_dir / f"weekly_factor_report_{pd.Timestamp.today().strftime('%Y%m%d')}_final.html"
    Path(report_path).write_text(html, encoding="utf-8")
    print(f"\n报告: {report_path}")


if __name__ == "__main__":
    main()
