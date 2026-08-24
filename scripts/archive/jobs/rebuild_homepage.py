#!/usr/bin/env python3
"""重生成 docs/index.html (首页) 使用 all_eval.json 最新数据"""
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
DOCS = PROJECT / "factor_engine" / "docs"
REPORT_DIR = DOCS / "reports" / "2026-08-23"
FACTORS_DIR = REPORT_DIR / "factors"
ALL_EVAL = REPORT_DIR / "all_eval_full.json"

OUT = DOCS / "index.html"


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor='#fff')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def main():
    with open(ALL_EVAL) as f:
        all_eval = json.load(f)

    factor_names = sorted([
        f.stem.replace('factor_', '')
        for f in FACTORS_DIR.glob("factor_*.html")
    ])

    rows = []
    for name in factor_names:
        ed = all_eval.get(name, {})
        if not ed or 'ls_sharpe' not in ed:
            continue
        is_flipped = '_flipped' in name
        rows.append({
            'name': name,
            'is_flipped': is_flipped,
            'mean_rankic': ed.get('mean_rankic', 0),
            'rankic_ir': ed.get('rankic_ir', 0),
            'rankic_winrate': ed.get('rankic_winrate', 0),
            'ls_sharpe': ed.get('ls_sharpe', 0),
            'ls_annual': ed.get('ls_annual', 0),
            'ls_mdd': ed.get('ls_mdd', 0),
            'ls_winrate': ed.get('ls_winrate', 0),
            'g10': ed.get('g10_annual', 0),
            'g1': ed.get('g1_annual', 0),
        })
    rows.sort(key=lambda r: r['rankic_ir'], reverse=True)

    # 拆分正负
    pos_rows = [r for r in rows if r['ls_sharpe'] > 0]
    neg_rows = [r for r in rows if r['ls_sharpe'] <= 0]

    sharpes = [r['ls_sharpe'] for r in rows]

    # Charts
    # 1) Sharpe 排序
    sorted_by_sharpe = sorted(rows, key=lambda r: r['ls_sharpe'], reverse=True)
    names_short = [r['name'].replace('_flipped', '*flp') for r in sorted_by_sharpe]
    shps = [r['ls_sharpe'] for r in sorted_by_sharpe]
    colors = ['#16a34a' if s > 0 else '#dc2626' for s in shps]

    fig, ax = plt.subplots(figsize=(11, max(8, len(names_short) * 0.18)))
    y_pos = np.arange(len(names_short))
    ax.barh(y_pos, shps, color=colors, alpha=0.85, edgecolor='white', linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_short, fontsize=5.5, family='monospace')
    ax.invert_yaxis()
    ax.axvline(0, color='#94a3b8', linewidth=0.6, linestyle='--')
    ax.set_xlabel('Long-Short Sharpe')
    ax.set_title(f'本周 {len(rows)} 因子 LS Sharpe 排序 (2019-2025 全样本)', fontsize=11, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    sharpe_b64 = fig_to_b64(fig)
    plt.close(fig)

    # 2) RankIC vs Sharpe
    ics = [r['mean_rankic'] for r in sorted_by_sharpe]
    iris = [r['rankic_ir'] for r in sorted_by_sharpe]
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(ics, shps, c=iris, cmap='RdYlGn', s=45,
                    edgecolors='white', linewidths=0.5, alpha=0.85)
    ax.axhline(0, color='#94a3b8', linewidth=0.6, linestyle='--')
    ax.axvline(0, color='#94a3b8', linewidth=0.6, linestyle='--')
    ax.set_xlabel('Mean RankIC')
    ax.set_ylabel('LS Sharpe')
    ax.set_title('Mean RankIC vs LS Sharpe', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, label='RankIC IR', shrink=0.85)
    plt.tight_layout()
    scatter_b64 = fig_to_b64(fig)
    plt.close(fig)

    # Build rows HTML
    pcls = lambda v: "pos" if v >= 0 else "neg"
    def fmt(v, pct=False, signed=False):
        if pct:
            return f"{v*100:+.1f}%" if signed else f"{v*100:.1f}%"
        return f"{v:+.4f}" if abs(v) < 1 else f"{v:+.2f}"

    def row_html(r, idx):
        n = r['name']
        flip_tag = ' <span class="tag tag-flip">FLIP</span>' if r['is_flipped'] else ''
        return f'''<tr>
  <td class="rank">{idx}</td>
  <td><a href="reports/2026-08-23/factors/factor_{n}.html"><code>factor_{n}</code></a>{flip_tag}</td>
  <td class="{pcls(r['mean_rankic'])}">{fmt(r['mean_rankic'], signed=True)}</td>
  <td class="{pcls(r['rankic_ir'])}">{fmt(r['rankic_ir'], signed=True)}</td>
  <td class="{pcls(r['rankic_winrate'])}">{fmt(r['rankic_winrate'], pct=True, signed=True)}</td>
  <td class="{pcls(r['ls_sharpe'])}">{fmt(r['ls_sharpe'], signed=True)}</td>
  <td class="{pcls(r['ls_annual'])}">{fmt(r['ls_annual'], pct=True, signed=True)}</td>
  <td class="{pcls(r['ls_mdd'])}">{fmt(r['ls_mdd'], pct=True, signed=True)}</td>
  <td class="{pcls(r['ls_winrate'])}">{fmt(r['ls_winrate'], pct=True, signed=True)}</td>
  <td class="{pcls(r['g10'])}">{fmt(r['g10'], pct=True, signed=True)}</td>
  <td class="{pcls(r['g1'])}">{fmt(r['g1'], pct=True, signed=True)}</td>
</tr>'''

    pos_table = "\n".join([row_html(r, i + 1) for i, r in enumerate(pos_rows)])
    neg_table = "\n".join([row_html(r, i + 1) for i, r in enumerate(neg_rows)])

    avg_sharpe = float(np.mean(sharpes))
    avg_rankic = float(np.mean([r['mean_rankic'] for r in rows]))

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>因子精选汇总 · 2019-2025</title>
<style>
:root {{
  --bg:#eef2f7; --panel:#fff; --fg:#0f172a; --muted:#64748b;
  --line:#e2e8f0; --primary:#1e4d8c; --shadow:0 4px 24px rgba(15,23,42,0.06);
  --pos:#16a34a; --neg:#dc2626;
}}
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif; color:var(--fg); background:var(--bg); }}
header.hero {{
  background:linear-gradient(135deg,#0f2744,#1e4d8c 60%,#0d9488);
  color:#fff; padding:36px 48px 28px;
}}
header.hero h1 {{ margin:0 0 8px; font-size:1.85rem; }}
header.hero .sub {{ opacity:0.85; font-size:0.9rem; margin-top:4px; }}
main {{ max-width:1400px; margin:0 auto; padding:24px; }}
.cards {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:14px; margin:20px 0; }}
.metric {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px; box-shadow:var(--shadow); text-align:center; }}
.metric b {{ display:block; font-size:1.6rem; color:var(--primary); }}
.metric span {{ color:var(--muted); font-size:0.78rem; }}
.pos {{ color:var(--pos) }}
.neg {{ color:var(--neg) }}
.chart-grid {{ display:grid; grid-template-columns:2fr 1fr; gap:20px; margin:20px 0; }}
@media (max-width: 1100px) {{ .chart-grid {{ grid-template-columns:1fr }} }}
img {{ width:100%; border:1px solid var(--line); border-radius:10px; background:#fff; }}
.notice {{
  background:linear-gradient(90deg,#eff6ff,#f0fdfa); border:1px solid #bfdbfe;
  padding:14px 18px; border-radius:12px; margin:16px 0; font-size:0.88rem;
  color:#1e3a8a;
}}
table {{ width:100%; border-collapse:collapse; font-size:0.83rem; margin:10px 0 20px; }}
th {{ background:#f1f5f9; color:#475569; padding:8px 10px; text-align:left; font-weight:600; border-bottom:2px solid #cbd5e1; position:sticky; top:0 }}
td {{ padding:7px 10px; border-bottom:1px solid var(--line); }}
tr:hover td {{ background:#f8fafc }}
.rank {{ color:#94a3b8; font-weight:600; width:36px }}
code {{ background:#f1f5f9; padding:2px 6px; border-radius:4px; font-size:0.78rem; color:#1e293b; }}
.tag {{ display:inline-block; padding:1px 6px; border-radius:10px; font-size:0.68rem; margin-left:4px; vertical-align:middle; }}
.tag-flip {{ background:#fef3c7; color:#92400e }}
h2 {{ font-size:1.05rem; color:var(--primary); margin:24px 0 10px; border-bottom:1px solid var(--line); padding-bottom:6px; }}
h3 {{ font-size:0.95rem; color:var(--primary); margin:18px 0 8px; }}
.summary-link {{ display:inline-block; margin:8px 0 16px; padding:10px 18px; background:var(--primary); color:#fff; text-decoration:none; border-radius:8px; font-weight:500; font-size:0.88rem; }}
.summary-link:hover {{ background:#163d6e }}
</style>
</head>
<body>
<header class="hero">
  <h1>因子精选汇总 · 2019-01 ~ 2025-12</h1>
  <div class="sub">
    {len(rows)} 个因子 · 5399 标的 · 1699 个交易日 ·
    factor_engine 落值 + quant_evaluator 评估 ·
    生成 {datetime.now().strftime('%Y-%m-%d %H:%M')}
  </div>
</header>

<main>

<a class="summary-link" href="reports/2026-08-23/index.html">→ 进入本周因子详情汇总</a>

<div class="cards">
  <div class="metric"><b>{len(rows)}</b><span>本周因子总数</span></div>
  <div class="metric"><b class="pos">{len(pos_rows)}</b><span>正 Sharpe 因子</span></div>
  <div class="metric"><b class="neg">{len(neg_rows)}</b><span>负 Sharpe 因子</span></div>
  <div class="metric"><b class="{pcls(avg_sharpe)}">{avg_sharpe:+.2f}</b><span>平均 LS Sharpe</span></div>
  <div class="metric"><b class="{pcls(avg_rankic)}">{avg_rankic:+.4f}</b><span>平均 RankIC</span></div>
</div>

<div class="notice">
  <b>数据说明</b> · 落值来自 <code>factor_delivery.zip</code> 原始 Python (439 因子) + 自实现的 <code>alpha_tools</code> facade，存为 <code>factor_values.parquet</code> (1699天 × 5399标的)。指标用 <code>quant_evaluator</code> 评估库的 Sharpe/MaxDD/Quantile Returns API + 自实现 IC/RankIC 计算。<b>TOP3</b>: book_attention (Sharpe +4.12), vol_asym_confirmed_range_v2 (+4.01), asym_vol_volume_cont_30 (+3.99)。<b>BOTTOM3</b>: range_volume_ratio_flipped (-3.96), vwap_adjusted_range_tanh_smooth_flipped (-3.95), drawdown_volume_geometry (-3.01)。
</div>

<div class="chart-grid">
  <div><img src="data:image/png;base64,{sharpe_b64}"/></div>
  <div><img src="data:image/png;base64,{scatter_b64}"/></div>
</div>

<h2>正 Sharpe 因子 ({len(pos_rows)} 个)</h2>
<p style="color:var(--muted);font-size:0.85rem">按 RankIC IR 降序。所有指标基于 2019-01-02 ~ 2025-12-31 全样本计算。</p>
<table>
  <thead>
    <tr>
      <th>#</th><th>因子</th>
      <th>Mean RankIC</th><th>RankIC IR</th><th>IC 胜率</th>
      <th>LS Sharpe</th><th>LS 年化</th><th>LS 回撤</th><th>LS 日胜率</th>
      <th>G10 年化</th><th>G1 年化</th>
    </tr>
  </thead>
  <tbody>
{pos_table}
  </tbody>
</table>

<h2 style="color:#dc2626">负 Sharpe 因子 ({len(neg_rows)} 个)</h2>
<p style="color:var(--muted);font-size:0.85rem">这些因子在 2019-2025 全样本中表现不佳，但仍提供作为反向因子 (Long-Short 反向) 参考。</p>
<table>
  <thead>
    <tr>
      <th>#</th><th>因子</th>
      <th>Mean RankIC</th><th>RankIC IR</th><th>IC 胜率</th>
      <th>LS Sharpe</th><th>LS 年化</th><th>LS 回撤</th><th>LS 日胜率</th>
      <th>G10 年化</th><th>G1 年化</th>
    </tr>
  </thead>
  <tbody>
{neg_table}
  </tbody>
</table>

</main>
</body>
</html>
"""

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  → {OUT}")


if __name__ == "__main__":
    main()