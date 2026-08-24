import json
from pathlib import Path
import pandas as pd

with open('weekly_backtest_output/backtest_results.json') as f:
    r = json.load(f)

# 构建最终列表
final_list = []
for name, res in r.items():
    if not res.get('success'):
        continue
    if name.endswith('_flipped'):
        final_list.append((name, res, True))
    else:
        flipped = name + '_flipped'
        if flipped not in r:
            final_list.append((name, res, False))

final_list.sort(key=lambda x: -x[1]['stats'].get('Sharpe Ratio', 0))

pos = [(n, v, f) for n, v, f in final_list if v['stats'].get('Sharpe Ratio', 0) > 0]
neg = [(n, v, f) for n, v, f in final_list if v['stats'].get('Sharpe Ratio', 0) <= 0]
flipped_contrib = sum(1 for _, _, f in pos if f)

today = pd.Timestamp.today().strftime('%Y-%m-%d')

html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>因子周报 """ + today + """</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:"PingFang SC","Microsoft YaHei",Arial,sans-serif; background:#0f1419; color:#e6e8ea; padding:20px; }
h1 { color:#1d9bf0; font-size:1.5em; margin-bottom:4px; }
.subtitle { color:#71767b; font-size:0.85em; margin-bottom:20px; }
.metrics { display:flex; gap:20px; flex-wrap:wrap; margin-bottom:24px; }
.metric { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px 24px; text-align:center; }
.metric b { display:block; font-size:2em; color:#1d9bf0; }
.metric span { font-size:0.8em; color:#71767b; }
.section { margin-bottom:32px; }
.section h2 { color:#e6e8ea; font-size:1.1em; border-left:4px solid #1d9bf0; padding-left:12px; margin-bottom:12px; }
table { width:100%; border-collapse:collapse; background:#161b22; border-radius:8px; overflow:hidden; }
th { background:#21262d; color:#71767b; font-size:0.75em; text-align:left; padding:10px 12px; font-weight:400; }
td { padding:10px 12px; font-size:0.85em; border-top:1px solid #21262d; }
tr:hover td { background:#1c2128; }
tr.flipped td { color:#8b949e; }
.pos { color:#3ba55c; font-weight:bold; }
.neg { color:#f4212e; }
.sharge { font-weight:bold; }
</style>
</head>
<body>
<h1>因子周报</h1>
<div class="subtitle">生成: """ + today + """ | 区间: 2024-01-02~2025-12-31 | IC负因子已翻转</div>

<div class="metrics">
  <div class="metric"><b>""" + str(len(final_list)) + """</b><span>总因子</span></div>
  <div class="metric"><b>""" + str(len(pos)) + """</b><span>正夏普</span></div>
  <div class="metric"><b>""" + str(flipped_contrib) + """</b><span>翻转贡献</span></div>
</div>

<div class="section">
<h2>正 Sharpe（""" + str(len(pos)) + """ 个）</h2>
<table>
<thead><tr><th>#</th><th>因子</th><th>Sharpe</th><th>年化</th><th>最大回撤</th><th>IC</th><th>RankIC</th></tr></thead>
<tbody>
"""

for i, (name, res, is_flipped) in enumerate(pos, 1):
    s = res['stats']
    ic = s.get('IC', 0)
    ric = s.get('Rank IC', 0)
    ic_cls = 'pos' if ic > 0 else 'neg'
    ric_cls = 'pos' if ric > 0 else 'neg'
    tag = ' [FLIP]' if is_flipped else ''
    row_cls = 'flipped' if is_flipped else ''
    html += f"""<tr class="{row_cls}">
  <td>{i}</td>
  <td>{name}{tag}</td>
  <td class="pos sharge">{s.get('Sharpe Ratio', 0):+.2f}</td>
  <td class="pos">{s.get('Annualized Return (%)', 0):+.1f}%</td>
  <td class="neg">{s.get('Max Drawdown (%)', 0):.1f}%</td>
  <td class="{ic_cls}">{ic:+.4f}</td>
  <td class="{ric_cls}">{ric:+.4f}</td>
</tr>"""

html += """</tbody></table></div>

<div class="section">
<h2>负 Sharpe（""" + str(len(neg)) + """ 个）</h2>
<table>
<thead><tr><th>#</th><th>因子</th><th>Sharpe</th><th>年化</th><th>IC</th><th>RankIC</th></tr></thead>
<tbody>
"""

for i, (name, res, is_flipped) in enumerate(neg, 1):
    s = res['stats']
    ic = s.get('IC', 0)
    ric = s.get('Rank IC', 0)
    ic_cls = 'pos' if ic > 0 else 'neg'
    ric_cls = 'pos' if ric > 0 else 'neg'
    ann = s.get('Annualized Return (%)', 0)
    ann_cls = 'pos' if ann > 0 else 'neg'
    tag = ' [FLIP]' if is_flipped else ''
    row_cls = 'flipped' if is_flipped else ''
    html += f"""<tr class="{row_cls}">
  <td>{i}</td>
  <td>{name}{tag}</td>
  <td class="neg sharge">{s.get('Sharpe Ratio', 0):+.2f}</td>
  <td class="{ann_cls}">{ann:+.1f}%</td>
  <td class="{ic_cls}">{ic:+.4f}</td>
  <td class="{ric_cls}">{ric:+.4f}</td>
</tr>"""

html += """</tbody></table></div></body></html>"""

out_path = Path('weekly_backtest_output/weekly_factor_report_20260823_final.html')
out_path.write_text(html, encoding='utf-8')
print(f'报告已生成: {out_path}')
print(f'总因子: {len(final_list)} | 正Sharpe: {len(pos)} | 翻转贡献: {flipped_contrib}')
