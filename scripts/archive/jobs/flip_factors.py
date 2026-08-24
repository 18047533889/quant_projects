#!/usr/bin/env python3
"""
对所有 IC < 0 的因子：改写公式 return * -1，重新计算，合并结果。
"""

from __future__ import annotations
import sys, os, json, re, tempfile as _tempfile, argparse, glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "vectorbt_qs"))
import weekly_factor_backtest as _wfb

MAX_WORKERS = min(12, (os.cpu_count() or 4) + 4)


# ============================================================
# 改写代码：return 值 * -1
# ============================================================
def flip_code(code: str) -> str:
    """在每个 return 语句末尾追加 * -1"""
    def _flip(m):
        full = m.group(0)
        if "#" in full:
            expr = full.split("#")[0].rstrip()
            comment = " " + full[full.index("#"):]
        else:
            expr = full.rstrip()
            comment = ""
        expr = expr.rstrip()
        if expr.endswith("-1"):
            return full  # 已翻转，跳过
        return expr + " * -1" + comment
    result = re.sub(r"\breturn\s+[^\n]+", _flip, code, flags=re.DOTALL)
    return result


# ============================================================
# Worker: 单因子翻转 + 重算
# ============================================================
def flip_single_factor(args) -> tuple:
    """
    接收：(因子名, 原始IC, 原始Sharpe, code, mkt_close_path, symbols, dates_str)
    返回：(flipped_name, result_dict)
    """
    import weekly_factor_backtest as _wfb
    import pandas as pd, numpy as np, tempfile as _tmp

    orig_name, orig_ic, orig_sharpe, code, mkt_close_path, symbols, dates_str = args
    flipped_name = orig_name + "_flipped"

    try:
        # 重建 market_data
        close = pd.read_parquet(mkt_close_path)
        close.index = pd.to_datetime(close.index)
        market_data = {
            "close": close,
            "open": close,   # 日频用 close 近似
            "high": close,
            "low": close,
            "volume": close * 0,
            "amount": close * 0,
        }
        dates = pd.to_datetime(dates_str)
        valid_dates = close.index.intersection(dates)

        # 翻转代码
        flipped_code = flip_code(code)

        # 计算翻转后的因子值
        fv_mat = _wfb._execute_factor_code(
            flipped_code, market_data, symbols, valid_dates,
        )

        valid_pct = fv_mat.notna().sum().sum() / max(fv_mat.size, 1) * 100

        # 回测
        result_dict = _wfb._backtest_single_factor(
            (fv_mat.to_dict(), {"factor_name": flipped_name, "code": flipped_code, "status": "EXEC"}),
            mkt_close_path,
            10_000_000.0, 0.2, 0.2,
        )
        _, _, result = result_dict

        new_ic = result.get("stats", {}).get("IC", 0)
        new_sh = result.get("stats", {}).get("Sharpe Ratio", 0)
        ok = result.get("success", False)

        status = "✓" if ok else "✗"
        print(f"  {status} {orig_name} -> {flipped_name}: "
              f"IC {orig_ic:+.4f}→{new_ic:+.4f}, "
              f"Sharpe {orig_sharpe:+.2f}→{new_sh:+.2f}, "
              f"有效率{valid_pct:.0f}%")

        return flipped_name, result

    except Exception as e:
        import traceback
        traceback.print_exc()
        return flipped_name, {"success": False, "error": str(e)}


# ============================================================
# 主流程
# ============================================================
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

    # ============================================================
    # Step 1: 找出 IC < 0 的原版因子
    # ============================================================
    neg_ic = []
    for name, res in existing.items():
        if name.endswith("_flipped"):
            continue
        if not res.get("success"):
            continue
        ic = res["stats"].get("IC", 0)
        if ic < -0.0001:
            neg_ic.append((name, res))

    print("=" * 60)
    print("  IC 翻转（改写因子公式 return * -1）")
    print("=" * 60)
    print(f"共 {len(neg_ic)} 个因子 IC < 0，需要翻转")
    for nm, res in neg_ic:
        print(f"  IC={res['stats']['IC']:+.4f}  Sharpe={res['stats']['Sharpe Ratio']:+.2f}  | {nm}")
    print()

    if not neg_ic:
        print("没有需要翻转的因子")
        return

    # ============================================================
    # Step 2: 加载行情，保存为 parquet
    # ============================================================
    symbols = _wfb._get_stock_universe_from_sample(symbols_limit=500)
    dates = pd.date_range(args.start, args.end, freq="B")
    market_data = _wfb.load_market_data(symbols, args.start, args.end)
    valid_dates = market_data["close"].index.intersection(dates)
    print(f"[数据] {len(valid_dates)} 交易日, {len(symbols)} 只股票")

    mkt_path = str(_tempfile.gettempdir()) + "/mkt_close_flip.parquet"
    market_data["close"].to_parquet(mkt_path)
    dates_str = valid_dates.strftime("%Y-%m-%d").tolist()

    # ============================================================
    # Step 3: 加载每个因子的 code
    # ============================================================
    tasks = []
    for nm, res in neg_ic:
        # 找代码文件
        pattern = _PROJECT_ROOT / "factor_delivery_converted" / "factors_combined" / f"{nm}.json"
        if not pattern.exists():
            cands = glob.glob(str(_PROJECT_ROOT / "factor_delivery_converted" / "factors_combined" / f"{nm}_*.json"))
            pattern = Path(cands[0]) if cands else None
        if not pattern or not pattern.exists():
            print(f"  [跳过] 无代码文件: {nm}")
            continue
        with open(pattern) as f:
            d = json.load(f)
        code = d.get("code", "")
        if not code or "NotImplementedError" in code or len(code) < 30:
            print(f"  [跳过] 代码无效: {nm}")
            continue
        tasks.append((nm, res["stats"]["IC"], res["stats"]["Sharpe Ratio"],
                       code, mkt_path, symbols, dates_str))

    print(f"[任务] {len(tasks)} 个因子待翻转\n")

    # ============================================================
    # Step 4: 并行翻转重算
    # ============================================================
    flipped_results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(flip_single_factor, t): t[0]
            for t in tasks
        }
        for future in as_completed(futures):
            nm = futures[future]
            try:
                flipped_name, result = future.result(timeout=600)
                flipped_results[flipped_name] = result
            except Exception as e:
                print(f"  ✗ {nm}: {e}")

    # ============================================================
    # Step 5: 合并结果
    # ============================================================
    final = dict(existing)
    for nm, res in flipped_results.items():
        final[nm] = res

    with open(results_path, "w") as f:
        json.dump(final, f, indent=2, default=str)

    # ============================================================
    # Step 6: 统计 + 报告
    # ============================================================
    pos = [(k, v) for k, v in final.items()
           if v.get("success") and v["stats"].get("Sharpe Ratio", 0) > 0]
    neg = [(k, v) for k, v in final.items()
           if v.get("success") and v["stats"].get("Sharpe Ratio", 0) <= 0]
    pos.sort(key=lambda x: -x[1]["stats"].get("Sharpe Ratio", 0))
    neg.sort(key=lambda x: -x[1]["stats"].get("Sharpe Ratio", 0))

    flipped_pos = sum(1 for k, v in final.items()
                     if k.endswith("_flipped") and v.get("success") and v["stats"].get("IC", 0) > 0)
    flipped_still_neg_ic = [(k, v) for k, v in final.items()
                             if k.endswith("_flipped") and v.get("success") and v["stats"].get("IC", 0) < -0.0001]

    ok_flipped = sum(1 for v in flipped_results.values() if v.get("success"))
    print(f"\n{'=' * 60}")
    print(f"  翻转完成")
    print(f"  新增成功: {ok_flipped}/{len(flipped_results)} 个")
    print(f"  正 Sharpe: {len(pos)} 个 | 翻转版贡献: {flipped_pos} 个")
    if flipped_still_neg_ic:
        print(f"  ⚠ 翻转版 IC 仍负: {len(flipped_still_neg_ic)}")
        for nm, res in flipped_still_neg_ic:
            print(f"    IC={res['stats']['IC']:+.4f} | {nm}")
    print(f"{'=' * 60}")

    # TOP 15
    print(f"\nTOP 15 正 Sharpe:")
    for i, (name, res) in enumerate(pos[:15], 1):
        s = res["stats"]
        tag = " [FLIP]" if name.endswith("_flipped") else ""
        print(f"  {i:2d}. {s['Sharpe Ratio']:+.2f}  IC={s['IC']:+.4f}  RIC={s['Rank IC']:+.4f}  | {name}{tag}")

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
<h1>因子周报 (IC翻转版)</h1>
<div class="subtitle">生成: {today} | 区间: 2024-01-02~2025-12-31 | IC负因子已改写公式翻转</div>

<div class="metrics">
  <div class="metric"><b>{len(final)}</b><span>总因子</span></div>
  <div class="metric"><b>{len(pos)}</b><span>正夏普</span></div>
  <div class="metric"><b>{flipped_pos}</b><span>翻转贡献</span></div>
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
        html += f"""<tr class="{'flipped' if fl else ''}">
  <td>{i}</td>
  <td>{name}{" [FLIP]" if fl else ""}</td>
  <td class="pos sharge">{s.get('Sharpe Ratio', 0):+.2f}</td>
  <td class="pos">{s.get('Annualized Return (%)', 0):+.1f}%</td>
  <td class="neg">{s.get('Max Drawdown (%)', 0):.1f}%</td>
  <td class="pos">{s.get('IC', 0):+.4f}</td>
  <td class="pos">{s.get('Rank IC', 0):+.4f}</td>
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

    report_path = out_dir / f"weekly_factor_report_{pd.Timestamp.today().strftime('%Y%m%d')}_flipped.html"
    Path(report_path).write_text(html, encoding="utf-8")
    print(f"\n报告: {report_path}")


if __name__ == "__main__":
    main()
