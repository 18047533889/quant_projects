#!/usr/bin/env python3
"""生成完整的因子分析 HTML 报告（嵌入所有图表）"""

import json
from pathlib import Path
import pandas as pd

OUT = Path("/home/sunhaiwei/quant_projects/weekly_backtest_output")

with open(OUT / "backtest_results.json") as f:
    results = json.load(f)

# 构建最终因子列表
factor_list = []
for name, res in results.items():
    if not res.get("success"):
        continue
    if name.endswith("_flipped"):
        factor_list.append((name, res, True))
    else:
        if name + "_flipped" not in results:
            factor_list.append((name, res, False))

factor_list.sort(key=lambda x: -x[1]["stats"].get("Sharpe Ratio", 0))
pos = [(n, v, f) for n, v, f in factor_list if v["stats"].get("Sharpe Ratio", 0) > 0]
neg = [(n, v, f) for n, v, f in factor_list if v["stats"].get("Sharpe Ratio", 0) <= 0]
flipped_contrib = sum(1 for _, _, f in pos if f)

today = pd.Timestamp.today().strftime("%Y-%m-%d")

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>因子周报 {today}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:"PingFang SC","Microsoft YaHei",Arial,sans-serif; background:#0f1419; color:#e6e8ea; padding:24px; }}
h1 {{ color:#1d9bf0; font-size:1.8em; margin-bottom:6px; }}
h2 {{ color:#e6e8ea; font-size:1.1em; border-left:4px solid #1d9bf0; padding-left:12px; margin:28px 0 14px; }}
h3 {{ color:#8b949e; font-size:0.9em; margin:20px 0 8px; }}
.subtitle {{ color:#71767b; font-size:0.85em; margin-bottom:24px; }}
.metrics {{ display:flex; gap:20px; flex-wrap:wrap; margin-bottom:28px; }}
.metric {{ background:#161b22; border:1px solid #30363d; border-radius:10px; padding:18px 28px; text-align:center; flex:1; min-width:120px; }}
.metric b {{ display:block; font-size:2.4em; color:#1d9bf0; font-weight:700; }}
.metric span {{ display:block; font-size:0.78em; color:#71767b; margin-top:4px; }}
.chart-block {{ background:#161b22; border:1px solid #30363d; border-radius:10px; padding:16px; margin-bottom:20px; }}
.chart-block img {{ width:100%; border-radius:6px; }}
table {{ width:100%; border-collapse:collapse; background:#161b22; border-radius:10px; overflow:hidden; }}
th {{ background:#21262d; color:#71767b; font-size:0.75em; text-align:left; padding:12px 14px; font-weight:400; letter-spacing:0.05em; }}
td {{ padding:11px 14px; font-size:0.84em; border-top:1px solid #21262d; }}
tr:hover td {{ background:#1c2128; }}
tr.flipped td {{ color:#8b949e; }}
.pos {{ color:#3ba55c; font-weight:bold; }}
.neg {{ color:#f4212e; }}
.sharge {{ font-weight:bold; }}
.tag {{
    display:inline-block;
    background:#21262d;
    color:#71767b;
    font-size:0.7em;
    padding:2px 6px;
    border-radius:4px;
    margin-left:4px;
    vertical-align:middle;
}}
.tag-flip {{ background:#1d3a5f; color:#1d9bf0; }}
</style>
</head>
<body>
<h1>因子周报</h1>
<div class="subtitle">生成: {today} &nbsp;|&nbsp; 区间: 2024-01-02 ~ 2025-12-31 &nbsp;|&nbsp; IC 负因子已通过因子值*-1 翻转</div>

<div class="metrics">
  <div class="metric"><b>{len(factor_list)}</b><span>总因子</span></div>
  <div class="metric"><b>{len(pos)}</b><span>正夏普</span></div>
  <div class="metric"><b>{len(neg)}</b><span>负夏普</span></div>
  <div class="metric"><b>{flipped_contrib}</b><span>翻转版贡献</span></div>
</div>

<h2>净值曲线（Top 10 因子）</h2>
<div class="chart-block">
  <img src="fig_nav_curves.png" alt="净值曲线">
</div>

<h2>IC / RankIC 时序（Top 8 因子，20日均线）</h2>
<div class="chart-block">
  <img src="fig_ic_timeseries.png" alt="IC时序">
</div>

<h2>月度 IC 热力图（Top 20 因子）</h2>
<div class="chart-block">
  <img src="fig_ic_heatmap.png" alt="IC热力图">
</div>

<h2>IC_IR 滚动（60日，Top 6 因子）</h2>
<div class="chart-block">
  <img src="fig_icir.png" alt="IC_IR">
</div>

<h2>因子相关性矩阵（Top 20）</h2>
<div class="chart-block">
  <img src="fig_correlation.png" alt="相关性">
</div>

<h2>统计分布</h2>
<div class="chart-block">
  <img src="fig_statistics.png" alt="统计分布">
</div>

<h2>正 Sharpe 因子（{len(pos)} 个）</h2>
<table>
<thead><tr>
  <th>#</th><th>因子</th><th>Sharpe</th><th>年化收益</th><th>最大回撤</th><th>IC</th><th>RankIC</th><th>胜率</th>
</tr></thead>
<tbody>
"""
for i, (name, res, is_flipped) in enumerate(pos, 1):
    s = res["stats"]
    ic, ric = s.get("IC", 0), s.get("Rank IC", 0)
    tag = '<span class="tag tag-flip">FLIP</span>' if is_flipped else ""
    row_cls = 'class="flipped"' if is_flipped else ""
    html += f"""<tr {row_cls}>
  <td>{i}</td>
  <td>{name}{tag}</td>
  <td class="pos sharge">{s.get("Sharpe Ratio", 0):+.2f}</td>
  <td class="pos">{s.get("Annualized Return (%)", 0):+.1f}%</td>
  <td class="neg">{s.get("Max Drawdown (%)", 0):.1f}%</td>
  <td class="{'pos' if ic > 0 else 'neg'}">{ic:+.4f}</td>
  <td class="{'pos' if ric > 0 else 'neg'}">{ric:+.4f}</td>
  <td>{s.get("Win Rate (%)", 0):.1f}%</td>
</tr>"""

html += f"""</tbody></table>

<h2>负 Sharpe 因子（{len(neg)} 个）</h2>
<table>
<thead><tr>
  <th>#</th><th>因子</th><th>Sharpe</th><th>年化收益</th><th>IC</th><th>RankIC</th>
</tr></thead>
<tbody>
"""
for i, (name, res, is_flipped) in enumerate(neg, 1):
    s = res["stats"]
    ic, ric = s.get("IC", 0), s.get("Rank IC", 0)
    tag = '<span class="tag tag-flip">FLIP</span>' if is_flipped else ""
    row_cls = 'class="flipped"' if is_flipped else ""
    ann = s.get("Annualized Return (%)", 0)
    html += f"""<tr {row_cls}>
  <td>{i}</td>
  <td>{name}{tag}</td>
  <td class="neg sharge">{s.get("Sharpe Ratio", 0):+.2f}</td>
  <td class="{'pos' if ann > 0 else 'neg'}">{ann:+.1f}%</td>
  <td class="{'pos' if ic > 0 else 'neg'}">{ic:+.4f}</td>
  <td class="{'pos' if ric > 0 else 'neg'}">{ric:+.4f}</td>
</tr>"""

html += """</tbody></table>
</body></html>"""

report_path = OUT / f"weekly_factor_report_{pd.Timestamp.today().strftime('%Y%m%d')}_full.html"
report_path.write_text(html, encoding="utf-8")
print(f"完整报告: {report_path}")
print(f"正Sharpe: {len(pos)} | 负Sharpe: {len(neg)} | 翻转贡献: {flipped_contrib}")
