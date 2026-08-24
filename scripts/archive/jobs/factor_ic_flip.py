#!/usr/bin/env python3
"""
因子 IC 自动翻转 + 全量重算
流程：
1. 读取现有 factor_values.parquet 和 backtest_results.json
2. 对 IC < 0 的因子，取其因子值列 * -1，存入新列（如 factor_xxx_flipped）
3. 对所有因子（含翻转版）重新运行回测
4. 合并结果，生成报告
"""

from __future__ import annotations
import sys, os, json, time, tempfile as _tempfile, argparse, glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "vectorbt_qs"))

MAX_WORKERS = min(12, (os.cpu_count() or 4) + 4)

import weekly_factor_backtest as _wfb


# ============================================================
# Step 1: 读取现有数据
# ============================================================
def load_existing_data(out_dir: Path):
    results_path = out_dir / "backtest_results.json"
    fv_path = out_dir / "factor_values.parquet"

    results = {}
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

    fv = None
    if fv_path.exists():
        try:
            fv = pd.read_parquet(str(fv_path))
            print(f"[数据] 因子值已加载: {fv.shape}")
        except Exception as e:
            print(f"[数据] 因子值读取失败: {e}")

    return results, fv


# ============================================================
# Step 2: 确定需要翻转的因子
# ============================================================
def find_factors_to_flip(results: dict) -> dict:
    """返回 {orig_name: flipped_name} 对"""
    flip_map = {}
    for name, res in results.items():
        if not res.get("success"):
            continue
        ic = res["stats"].get("IC", 0)
        if ic < -0.0001:  # IC 明显为负
            flip_map[name] = name + "_flipped"
    return flip_map


# ============================================================
# Step 3: 创建翻转列
# ============================================================
def create_flipped_columns(fv: pd.DataFrame, flip_map: dict) -> pd.DataFrame:
    """对 factor_values 中的列取负，存入新列"""
    if fv is None or fv.empty:
        return fv

    new_cols = {}
    for orig, flipped in flip_map.items():
        if orig in fv.columns.get_level_values(0):
            print(f"  翻转: {orig} -> {flipped}")
            orig_vals = fv[orig].values
            new_cols[flipped] = pd.DataFrame(
                -orig_vals, index=fv.index, columns=fv.columns.get_level_values(1)[fv.columns.get_level_values(0) == orig].tolist()
            )

    if not new_cols:
        return fv

    # 构建新 MultiIndex 列
    new_data = {}
    for col_name, col_df in new_cols.items():
        for sym in col_df.columns:
            new_data[(col_name, sym)] = col_df[sym]

    flipped_df = pd.DataFrame(new_data)
    if isinstance(fv.columns, pd.MultiIndex):
        flipped_df.columns = pd.MultiIndex.from_tuples(flipped_df.columns)
    else:
        flipped_df.columns = pd.MultiIndex.from_tuples(
            [(c, "") for c in flipped_df.columns]
        )

    result = pd.concat([fv, flipped_df], axis=1)
    print(f"[翻转] 因子值矩阵扩展: {fv.shape} -> {result.shape}")
    return result


# ============================================================
# Step 4: 回测单个因子（支持翻转版）
# ============================================================
def _backtest_factor_with_sign(fv_data, fwd_rets_path, init_cash, top_q, bottom_q, sign: float = 1.0):
    """sign=-1.0 表示翻转版"""
    import numpy as np
    import pandas as pd

    fv_arr, f_dict = fv_data
    name = f_dict["factor_name"]
    flipped = f_dict.get("_flipped", False)

    try:
        fv_df = pd.DataFrame(fv_arr)
        fv_df.index = pd.DatetimeIndex(fv_df.index)
        dates = fv_df.index
        symbols = list(fv_df.columns)
        fv_df = fv_df.astype(float) * sign  # 翻转核心

        # 目标权重
        weights = pd.DataFrame(0.0, index=dates, columns=symbols)
        top_n = max(5, int(len(symbols) * top_q))
        bot_n = max(5, int(len(symbols) * bottom_q))
        for dt in dates:
            if dt not in fv_df.index:
                continue
            row = fv_df.loc[dt].dropna()
            if len(row) < 10:
                continue
            try:
                top_s = row.nlargest(top_n).index.tolist()
                bot_s = row.nsmallest(bot_n).index.tolist()
                weights.loc[dt, top_s] = 1.0 / top_n
                weights.loc[dt, bot_s] = -1.0 / bot_n
            except Exception:
                pass
        weights = weights.ffill().fillna(0.0)

        close_prices = pd.read_parquet(fwd_rets_path)
        close = close_prices.loc[dates]
        common = close.index.intersection(weights.index)
        close = close.loc[common]
        weights = weights.loc[common]
        rets = close.pct_change().fillna(0)
        strat_rets = (weights.shift(1) * rets).sum(axis=1)
        nav = (1 + strat_rets).cumprod() * init_cash

        rets_d = strat_rets.dropna()
        n = len(rets_d)
        if n == 0:
            return name, f_dict, {"success": False, "error": "no data"}

        total_ret = float((nav.iloc[-1] / init_cash - 1) * 100) if not nav.empty else 0
        ann_ret = float((1 + rets_d.mean()) ** 252 - 1) * 100
        ann_vol = float(rets_d.std() * np.sqrt(252) * 100)
        sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
        dd = (nav - nav.cummax()) / nav.cummax() * 100
        max_dd = float(dd.min())
        calmar = float(ann_ret / abs(max_dd)) if max_dd < 0 else 0.0
        win_rate = float((rets_d > 0).sum() / n * 100)
        wins = rets_d[rets_d > 0]
        loss = rets_d[rets_d < 0]
        wl_ratio = float(abs(wins.mean() / loss.mean())) if len(loss) > 0 and loss.mean() != 0 else 0.0

        # IC / RankIC（使用原始 sign=1 的因子值来计算）
        ic_vals, ric_vals = [], []
        try:
            fwd = close_prices.pct_change().shift(-1)
            # 用 sign=1 的因子值（原始）计算 IC
            fv_raw = pd.DataFrame(fv_arr)
            fv_raw.index = pd.DatetimeIndex(fv_raw.index)
            common_idx = fv_raw.index.intersection(fwd.index)
            for dt in common_idx:
                fv_row = fv_raw.loc[dt].dropna()
                ret_row = fwd.loc[dt].reindex(fv_row.index).dropna()
                if len(fv_row) > 10 and len(ret_row) > 10:
                    cs = fv_row.index.intersection(ret_row.index)
                    if len(cs) > 10:
                        ic = fv_row[cs].corr(ret_row[cs])
                        ric = fv_row[cs].rank().corr(ret_row[cs].rank())
                        if np.isfinite(ic):
                            ic_vals.append(ic)
                        if np.isfinite(ric):
                            ric_vals.append(ric)
            ic_mean = float(np.mean(ic_vals)) if ic_vals else 0.0
            ric_mean = float(np.mean(ric_vals)) if ric_vals else 0.0
        except Exception:
            ic_mean = ric_mean = 0.0

        stats = {
            "Total Return (%)": total_ret,
            "Annualized Return (%)": ann_ret,
            "Annualized Vol (%)": ann_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown (%)": max_dd,
            "Calmar Ratio": calmar,
            "IC": ic_mean,
            "Rank IC": ric_mean,
            "Win Rate (%)": win_rate,
            "Win/Loss Ratio": wl_ratio,
            "Best Day (%)": float(rets_d.max() * 100),
            "Worst Day (%)": float(rets_d.min() * 100),
            "Trading Days": n,
        }

        return name, f_dict, {
            "success": True,
            "stats": stats,
            "_flipped": flipped,
        }
    except Exception as e:
        return name, f_dict, {"success": False, "error": str(e)}


# ============================================================
# Step 5: 全量回测（原始 + 翻转版）
# ============================================================
def run_full_backtest(fv, results, flip_map, fwd_rets_path, init_cash, top_q, bottom_q, workers):
    """对所有因子（原始 + 翻转）重新回测"""

    # 构建因子列表（含翻转版）
    factors = []
    done_names = set(results.keys())

    # 原始因子
    for col_name in fv.columns.get_level_values(0).unique():
        if col_name in done_names and not col_name.endswith("_flipped"):
            continue  # 已回测且非翻转，跳过（后面统一重跑）
        factors.append({
            "factor_name": col_name,
            "code": "",
            "status": "EXEC",
            "_flipped": col_name.endswith("_flipped"),
        })

    # 确保翻转版也在列表中
    for orig, flipped in flip_map.items():
        if flipped not in [f["factor_name"] for f in factors]:
            factors.append({
                "factor_name": flipped,
                "code": "",
                "status": "EXEC",
                "_flipped": True,
            })

    # 也把已有的原始因子（没被翻转的）加进来重跑
    for name, res in results.items():
        if name.endswith("_flipped"):
            continue
        if name not in fv.columns.get_level_values(0):
            continue
        if name not in [f["factor_name"] for f in factors]:
            factors.append({
                "factor_name": name,
                "code": "",
                "status": "EXEC",
                "_flipped": name in flip_map,
            })

    print(f"\n[回测] 共 {len(factors)} 个因子（含翻转版）...")
    new_results = {}
    fv_data_map = {}

    for col_name in fv.columns.get_level_values(0).unique():
        sub = fv[col_name]
        f_dict = next((f for f in factors if f["factor_name"] == col_name), None)
        if f_dict is None:
            f_dict = {"factor_name": col_name, "code": "", "status": "EXEC", "_flipped": col_name.endswith("_flipped")}
        fv_data_map[col_name] = (sub.to_dict(), f_dict)

    futures = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name, (fv_arr, f_dict) in fv_data_map.items():
            sign = -1.0 if f_dict.get("_flipped", False) else 1.0
            future = pool.submit(
                _backtest_factor_with_sign,
                (fv_arr, f_dict),
                fwd_rets_path,
                init_cash, top_q, bottom_q,
                sign,
            )
            futures[future] = name

        for future in as_completed(futures):
            name = futures[future]
            try:
                _, f_dict, result = future.result(timeout=300)
                new_results[name] = result
                if result.get("success"):
                    s = result["stats"]
                    tag = " [FLIP]" if f_dict.get("_flipped") else ""
                    print(f"  ✓ {name}{tag}: Sharpe={s.get('Sharpe Ratio', 0):+.2f}, IC={s.get('IC', 0):+.4f}")
                else:
                    print(f"  ✗ {name}: {result.get('error', 'failed')[:60]}")
            except Exception as e:
                print(f"  ✗ {name}: {e}")

    return new_results


# ============================================================
# Step 6: HTML 报告
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

    def row_color(name, s):
        if s.get("Sharpe Ratio", 0) > 0:
            return "pos"
        return "neg"

    def ic_color(ic):
        return "pos" if ic > 0 else "neg"

    def val_color(v, positive=True):
        return "pos" if v > 0 else "neg"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>因子周报 {today} (IC翻转版)</title>
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
tr.flipped td {{ color:#8b949e; font-style:italic; }}
tr.flipped:hover td {{ background:#1c2128; }}
.pos {{ color:#3ba55c; font-weight:bold; }}
.neg {{ color:#f4212e; }}
.sharge {{ font-weight:bold; }}
</style>
</head>
<body>
<h1>因子周报 (含IC翻转版)</h1>
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
  <th>因子名称</th><th>Sharpe</th><th>年化收益</th><th>最大回撤</th><th>IC</th><th>RankIC</th><th>胜率</th>
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
        wr = s.get("Win Rate (%)", 0)
        flipped = name.endswith("_flipped")
        row_class = "flipped" if flipped else ""
        html += f"""<tr class="{row_class}">
  <td>{name}{" [FLIP]" if flipped else ""}</td>
  <td class="pos sharge">{sh:+.2f}</td>
  <td class="{'pos' if ann>0 else 'neg'}">{ann:+.1f}%</td>
  <td class="neg">{dd:.1f}%</td>
  <td class="{'pos' if ic>0 else 'neg'}">{ic:+.4f}</td>
  <td class="{'pos' if ric>0 else 'neg'}">{ric:+.4f}</td>
  <td>{wr:.1f}%</td>
</tr>"""

    html += """</tbody></table></div>

<div class="section">
<h2>负 Sharpe 因子（{} 个）</h2>
<table>
<thead><tr>
  <th>因子名称</th><th>Sharpe</th><th>年化收益</th><th>最大回撤</th><th>IC</th><th>RankIC</th><th>胜率</th>
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
        wr = s.get("Win Rate (%)", 0)
        flipped = name.endswith("_flipped")
        row_class = "flipped" if flipped else ""
        html += f"""<tr class="{row_class}">
  <td>{name}{" [FLIP]" if flipped else ""}</td>
  <td class="neg sharge">{sh:+.2f}</td>
  <td class="{'pos' if ann>0 else 'neg'}">{ann:+.1f}%</td>
  <td class="neg">{dd:.1f}%</td>
  <td class="{'pos' if ic>0 else 'neg'}">{ic:+.4f}</td>
  <td class="{'pos' if ric>0 else 'neg'}">{ric:+.4f}</td>
  <td>{wr:.1f}%</td>
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
    p.add_argument("--start", type=str, default="2024-01-02")
    p.add_argument("--end", type=str, default="2025-12-31")
    p.add_argument("--workers", type=int, default=MAX_WORKERS)
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    out_dir = Path(args.output_dir or str(_PROJECT_ROOT / "weekly_backtest_output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "backtest_results.json"
    fv_path = out_dir / "factor_values.parquet"
    report_path = out_dir / f"weekly_factor_report_{datetime.now().strftime('%Y%m%d')}_flipped.html"

    init_cash = 10_000_000.0
    top_q = bottom_q = 0.2

    # Step 1: 加载现有数据
    print("=" * 60)
    print("  因子 IC 翻转 + 全量重算")
    print("=" * 60)
    existing_results, fv = load_existing_data(out_dir)
    print(f"[存量] {len(existing_results)} 个因子, 因子值 {fv.shape if fv is not None else 'None'}")

    # Step 2: 确定翻转映射
    flip_map = find_factors_to_flip(existing_results)
    print(f"\n[翻转] {len(flip_map)} 个因子需要翻转:")
    for orig, flipped in flip_map.items():
        ic = existing_results[orig]["stats"].get("IC", 0)
        sh = existing_results[orig]["stats"].get("Sharpe Ratio", 0)
        print(f"  {orig}: IC={ic:+.4f} Sharpe={sh:+.2f} -> {flipped}")

    if not flip_map:
        print("没有因子需要翻转")
        return

    # Step 3: 创建翻转列
    print()
    fv_enhanced = create_flipped_columns(fv, flip_map)

    # 保存增强版因子值
    fv_enhanced.to_parquet(str(fv_path))
    print(f"[保存] 因子值已更新: {fv_path}")

    # Step 4: 准备回测数据
    fwd_rets_path = Path(_tempfile.gettempdir()) / "fwd_rets_flip.parquet"
    if fv is not None:
        # 重建 close prices
        close = fv_enhanced.iloc[:, fv_enhanced.columns.get_level_values(0) == fv_enhanced.columns.get_level_values(0)[0]]
        # 用已有行情数据
        symbols = _wfb._get_stock_universe_from_sample(symbols_limit=500)
        dates = pd.date_range(args.start, args.end, freq="B")
        market_data = _wfb.load_market_data(symbols, args.start, args.end)
        if market_data.get("close") is not None:
            market_data["close"].to_parquet(str(fwd_rets_path))
            print(f"[数据] 行情已保存: {fwd_rets_path}")
        else:
            print("[错误] 无法获取行情数据")
            return
    else:
        print("[错误] 缺少因子值数据")
        return

    # Step 5: 全量重算
    new_results = run_full_backtest(
        fv_enhanced, existing_results, flip_map,
        str(fwd_rets_path), init_cash, top_q, bottom_q, args.workers,
    )

    # Step 6: 合并结果
    # 原始失败 + 翻转成功的 → 保留翻转版
    # 翻转版替代原始版
    final_results = {}
    flipped_names = set(flip_map.values())

    # 先放所有新结果
    final_results.update(new_results)

    # 把没有新结果的原始因子（非翻转）也保留
    for name, res in existing_results.items():
        if name not in final_results and not name.endswith("_flipped"):
            final_results[name] = res

    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"\n[合并] 结果已更新: {results_path}（共 {len(final_results)} 个因子）")

    # Step 7: 生成报告
    config = {
        "start_date": args.start,
        "end_date": args.end,
        "n_factors": len(final_results),
    }
    generate_html_report(final_results, config, str(report_path))

    # 汇总
    pos = sum(1 for v in final_results.values() if v.get("success") and v["stats"].get("Sharpe Ratio", 0) > 0)
    total = sum(1 for v in final_results.values() if v.get("success"))
    flipped_pos = sum(1 for name, v in final_results.items()
                      if v.get("success") and v["stats"].get("Sharpe Ratio", 0) > 0 and name.endswith("_flipped"))

    print("\n" + "=" * 60)
    print("  完成")
    print(f"  累计因子: {len(final_results)} 个")
    print(f"  正 Sharpe: {pos} 个（其中翻转版贡献 {flipped_pos} 个）")
    print(f"  报告: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
