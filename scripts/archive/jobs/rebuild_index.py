#!/usr/bin/env python3
"""重生成本周 index.html 汇总页
- 61 因子表格 (按 RankIC IR 排序)
- 顶部含总览: 平均 RankIC, 平均 LS Sharpe, 翻正因子数
- 一张汇总图 (e.g. LS Sharpe 分布)
"""
import sys, json, os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64

PROJECT = Path("/home/sunhaiwei/quant_projects")
REPORT_DIR = PROJECT / "factor_engine" / "docs" / "reports" / "2026-08-23"
FACTORS_DIR = REPORT_DIR / "factors"
ALL_EVAL = REPORT_DIR / "all_eval_full.json"
ALL_EVAL_SLIM = REPORT_DIR / "all_eval.json"


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor='#fff')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def build_summary_charts(all_eval):
    # 1) Sharpe 排序条形图
    ranked = sorted(all_eval.items(), key=lambda x: x[1].get('ls_sharpe', 0), reverse=True)
    names = [x[0].replace('_flipped', '*flp') for x in ranked]
    sharpes = [x[1].get('ls_sharpe', 0) for x in ranked]
    colors = ['#16a34a' if s > 0 else '#dc2626' for s in sharpes]

    fig, ax = plt.subplots(figsize=(11, max(6, len(names) * 0.18)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, sharpes, color=colors, alpha=0.85, edgecolor='white', linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=6, family='monospace')
    ax.invert_yaxis()
    ax.axvline(0, color='#94a3b8', linewidth=0.6, linestyle='--')
    ax.set_xlabel('Long-Short Sharpe')
    ax.set_title(f'61 因子 LS Sharpe 排序 (按 2019-{max([r[1].get("dates", [""])[-1][:4] if r[1].get("dates") else "2025" for r in ranked], default="2025")} 全样本)',
                 fontsize=10, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    sharpe_b64 = fig_to_b64(fig)
    plt.close(fig)

    # 2) RankIC vs LS Sharpe 散点
    ics = [x[1].get('mean_rankic', 0) for x in ranked]
    shps = [x[1].get('ls_sharpe', 0) for x in ranked]
    rank_irs = [x[1].get('rankic_ir', 0) for x in ranked]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sc = ax.scatter(ics, shps, c=rank_irs, cmap='RdYlGn', s=50,
                    edgecolors='white', linewidths=0.5, alpha=0.9)
    ax.axhline(0, color='#94a3b8', linewidth=0.6, linestyle='--')
    ax.axvline(0, color='#94a3b8', linewidth=0.6, linestyle='--')
    ax.set_xlabel('Mean RankIC')
    ax.set_ylabel('LS Sharpe')
    ax.set_title('RankIC vs LS Sharpe 散点 (颜色=IR)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, label='RankIC IR', shrink=0.85)
    plt.tight_layout()
    scatter_b64 = fig_to_b64(fig)
    plt.close(fig)

    return sharpe_b64, scatter_b64


def main():
    print("生成汇总 index.html ...")

    with open(ALL_EVAL) as f:
        all_eval = json.load(f)

    factor_names = sorted([
        f.stem.replace('factor_', '')
        for f in FACTORS_DIR.glob("factor_*.html")
    ])

    # Sort by rankic_ir desc
    rows = []
    for name in factor_names:
        ed = all_eval.get(name, {})
        if not ed:
            continue
        is_flipped = '_flipped' in name
        rows.append({
            'name': name + ('*' if is_flipped else ''),
            'display': name.replace('_flipped', '_flipped') + (' <span class="badge badge-yellow">flipped</span>' if is_flipped else ''),
            'is_flipped': is_flipped,
            'mean_rankic': ed.get('mean_rankic', 0),
            'rankic_ir': ed.get('rankic_ir', 0),
            'ic_winrate': ed.get('rankic_winrate', 0),
            'ls_sharpe': ed.get('ls_sharpe', 0),
            'ls_annual': ed.get('ls_annual', 0),
            'ls_mdd': ed.get('ls_mdd', 0),
            'g10': ed.get('g10_annual', 0),
            'g1': ed.get('g1_annual', 0),
        })
    rows.sort(key=lambda r: r['rankic_ir'], reverse=True)

    # Summary stats
    sharpes = [r['ls_sharpe'] for r in rows]
    rankics = [r['mean_rankic'] for r in rows]
    iris = [r['rankic_ir'] for r in rows]
    n_pos = sum(1 for s in sharpes if s > 0)
    n_neg = sum(1 for s in sharpes if s < 0)

    # charts
    sharpe_b64, scatter_b64 = build_summary_charts(all_eval)

    # Top 20 table
    def fmt(v, pct=False, signed=False):
        if pct:
            return f"{v*100:+.1f}%" if signed else f"{v*100:.1f}%"
        return f"{v:+.4f}" if abs(v) < 1 else f"{v:+.2f}"

    pcls = lambda v: "pos" if v >= 0 else "neg"

    table_rows = ""
    for i, r in enumerate(rows, 1):
        n = r['name']
        table_rows += f'''<tr>
  <td class="rank">{i}</td>
  <td><a href="factors/factor_{n}.html"><code>{r['display']}</code></a></td>
  <td class="{pcls(r['mean_rankic'])}">{fmt(r['mean_rankic'], signed=True)}</td>
  <td class="{pcls(r['rankic_ir'])}">{fmt(r['rankic_ir'], signed=True)}</td>
  <td class="{pcls(r['ic_winrate'])}">{fmt(r['ic_winrate'], pct=True, signed=True)}</td>
  <td class="{pcls(r['ls_sharpe'])}">{fmt(r['ls_sharpe'], signed=True)}</td>
  <td class="{pcls(r['ls_annual'])}">{fmt(r['ls_annual'], pct=True, signed=True)}</td>
  <td class="{pcls(r['ls_mdd'])}">{fmt(r['ls_mdd'], pct=True, signed=True)}</td>
  <td class="{pcls(r['g10'])}">{fmt(r['g10'], pct=True, signed=True)}</td>
  <td class="{pcls(r['g1'])}">{fmt(r['g1'], pct=True, signed=True)}</td>
</tr>'''

    # Date range from eval
    all_dates = []
    for n in factor_names:
        d = all_eval.get(n, {}).get('dates') if False else None
    # Get date from full eval
    with open(ALL_EVAL_SLIM) as f:
        pass
    try:
        with open(REPORT_DIR / 'all_eval_full.json') as f:
            full_eval = json.load(f)
        date_first, date_last = '2019-01-02', '2025-12-31'
        for n, d in full_eval.items():
            if d.get('dates'):
                date_first = d['dates'][0]
                date_last = d['dates'][-1]
                break
    except Exception:
        date_first, date_last = '2019-01-02', '2025-12-31'

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>因子周报 · {date_first} ~ {date_last}</title>
<style>
:root {{
  --bg:#eef2f7; --panel:#fff; --fg:#0f172a; --muted:#64748b;
  --line:#e2e8f0; --primary:#1e4d8c; --pos:#16a34a; --neg:#dc2626;
}}
* {{ box-sizing:border-box }}
body {{ margin:0; font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif; color:var(--fg); background:var(--bg) }}
header {{ background:linear-gradient(135deg,#0f2744,#1e4d8c 60%,#0d9488); color:#fff; padding:28px 48px 24px }}
header h1 {{ margin:0; font-size:1.7rem }}
header .meta {{ opacity:0.85; font-size:0.85rem; margin-top:6px }}
.back {{ display:inline-block; margin:14px 48px 0; color:#93c5fd; text-decoration:none }}
.back:hover {{ text-decoration:underline }}
main {{ max-width:1280px; margin:0 auto; padding:24px }}
.grid-4 {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:18px }}
.metric {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px; text-align:center; box-shadow:0 4px 24px rgba(15,23,42,0.06) }}
.metric b {{ display:block; font-size:1.6rem }}
.metric span {{ color:var(--muted); font-size:0.78rem }}
.pos {{ color:var(--pos) }}
.neg {{ color:var(--neg) }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px 20px; box-shadow:0 4px 24px rgba(15,23,42,0.06); margin-bottom:16px }}
h2 {{ font-size:1rem; color:var(--primary); margin:0 0 12px; border-bottom:1px solid var(--line); padding-bottom:8px }}
table {{ width:100%; border-collapse:collapse; font-size:0.84rem }}
th {{ background:#f8fafc; color:#475569; padding:8px 10px; text-align:left; font-weight:600; border-bottom:2px solid #cbd5e1; position:sticky; top:0 }}
td {{ padding:8px 10px; border-bottom:1px solid var(--line) }}
tr:hover {{ background:#f8fafc }}
.rank {{ color:#94a3b8; font-weight:600; width:36px }}
code {{ background:#f1f5f9; padding:2px 6px; border-radius:4px; font-size:0.8rem; color:#1e293b }}
.badge {{ display:inline-block; padding:1px 6px; border-radius:10px; font-size:0.68rem; margin-left:4px; vertical-align:middle }}
.badge-yellow {{ background:#fef3c7; color:#92400e }}
.chart-row {{ display:grid; grid-template-columns:1fr 1fr; gap:14px }}
@media (max-width: 900px) {{ .chart-row {{ grid-template-columns:1fr }} }}
img {{ width:100%; border-radius:8px }}
</style>
</head>
<body>
<header>
  <h1>本周因子周报 · {date_first} ~ {date_last}</h1>
  <div class="meta">
    61 个因子 · 满分 5400 标的 · 1699 个交易日 ·
    factor_engine 落值 + quant_evaluator 评估 ·
    生成 {datetime.now().strftime('%Y-%m-%d %H:%M')}
  </div>
</header>
<a class="back" href="../index.html">&#8592; 返回首页</a>

<main>

<div class="grid-4">
  <div class="metric"><b>{len(rows)}</b><span>本周因子数</span></div>
  <div class="metric"><b class="pos">{n_pos}</b><span>正 Sharpe 因子</span></div>
  <div class="metric"><b class="neg">{n_neg}</b><span>负 Sharpe 因子</span></div>
  <div class="metric"><b class="{pcls(np.mean(sharpes))}">{np.mean(sharpes):+.2f}</b><span>平均 LS Sharpe</span></div>
</div>

<div class="chart-row">
  <div class="card">
    <h2>61 因子 LS Sharpe 排序</h2>
    <img src="data:image/png;base64,{sharpe_b64}"/>
  </div>
  <div class="card">
    <h2>RankIC vs LS Sharpe 散点</h2>
    <img src="data:image/png;base64,{scatter_b64}"/>
  </div>
</div>

<div class="card">
  <h2>因子列表 (按 RankIC IR 排序)</h2>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>因子</th>
        <th>Mean RankIC</th>
        <th>RankIC IR</th>
        <th>IC 胜率</th>
        <th>LS Sharpe</th>
        <th>LS 年化</th>
        <th>LS 回撤</th>
        <th>G10 年化</th>
        <th>G1 年化</th>
      </tr>
    </thead>
    <tbody>
{table_rows}
    </tbody>
  </table>
</div>

</main>
</body>
</html>
"""

    out = REPORT_DIR / "index.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  → {out}")


if __name__ == "__main__":
    main()