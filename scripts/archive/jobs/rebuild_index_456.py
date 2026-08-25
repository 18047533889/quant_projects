#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""456 因子全量首页（index.html）生成器。

读 summary_stats.json（456 条 stats），生成：
  - 顶部总览卡片（因子数 / 平均 RankIC / 平均 LS Sharpe / 翻正因子数 / 回测区间）
  - LS Sharpe 排序条形图（base64，中文）
  - 456 因子表格（名称 / RankIC / IR / LS Sharpe / 年化 / 回撤 / 翻转标记 → 详情页链接）
输出覆盖 reports/2026-08-23/index.html。
"""
import json, os, sys, io, base64
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体
_CN_FONT = "/home/sunhaiwei/.fonts/NotoSansSC-Regular.otf"
if os.path.exists(_CN_FONT):
    try:
        import matplotlib.font_manager as _fm
        _fm.fontManager.addfont(_CN_FONT)
        _CN_NAME = _fm.FontProperties(fname=_CN_FONT).get_name()
        plt.rcParams["font.sans-serif"] = [_CN_NAME, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

PROJECT = Path("/home/sunhaiwei/quant_projects")
REPORT_DIR = PROJECT / "factor_engine" / "docs" / "reports" / "2026-08-23"
STATS = REPORT_DIR / "summary_stats.json"
OUT = REPORT_DIR / "index.html"

BG = "#eef2f7"
PANEL = "#fff"
FG = "#0f172a"
MUTED = "#64748b"
LINE = "#e2e8f0"
PRIMARY = "#1e4d8c"
POS = "#16a34a"
NEG = "#dc2626"


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=PANEL)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def build_sharpe_chart(stats):
    ranked = sorted(stats, key=lambda s: s.get("ls_sharpe", 0), reverse=True)
    names = [s["name"].replace("_flipped", "·翻") for s in ranked]
    sharpes = [s.get("ls_sharpe", 0) for s in ranked]
    colors = [POS if x > 0 else NEG for x in sharpes]

    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.10)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, sharpes, color=colors, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=6)
    ax.axvline(0, color="#94a3b8", lw=0.8)
    ax.set_xlabel("LS Sharpe（多空年化夏普）")
    ax.set_title("全部因子多空 Sharpe 排序")
    ax.invert_yaxis()
    ax.grid(axis="x", ls="--", alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


def main():
    if not STATS.exists():
        print(f"[456index] 无 {STATS}，先跑 render_all_456")
        return
    stats = json.loads(STATS.read_text())
    print(f"[456index] 因子数: {len(stats)}")

    n_flipped = sum(1 for s in stats if s.get("is_flipped"))
    mean_ic = float(np.mean([s.get("mean_ic", 0) for s in stats])) if stats else 0
    mean_sharpe = float(np.mean([s.get("ls_sharpe", 0) for s in stats])) if stats else 0
    pos_sharpe = sum(1 for s in stats if s.get("ls_sharpe", 0) > 0)
    neg_ic = sum(1 for s in stats if s.get("mean_ic", 0) < 0)

    # 回测区间
    start_dates = [s.get("start_date") for s in stats if s.get("start_date")]
    end_dates = [s.get("end_date") for s in stats if s.get("end_date")]
    start = min(start_dates) if start_dates else "—"
    end = max(end_dates) if end_dates else "—"
    n_days = max((s.get("n_periods", 0) for s in stats), default=0)

    chart_b64 = build_sharpe_chart(stats)

    rows = []
    for s in sorted(stats, key=lambda x: x.get("ls_sharpe", 0), reverse=True):
        name = s["name"]
        flip_tag = '<span class="tag tag-flip">翻正</span>' if s.get("is_flipped") else ""
        cls_p = "pos" if s.get("mean_ic", 0) > 0 else "neg"
        cls_s = "pos" if s.get("ls_sharpe", 0) > 0 else "neg"
        rows.append(
            f'<tr><td class="rank">{len(rows)+1}</td>'
            f'<td><a href="factors/factor_{name}.html"><code>{name}</code></a>{flip_tag}</td>'
            f'<td class="{cls_p}">{s.get("mean_ic", 0):.4f}</td>'
            f'<td>{s.get("ic_ir", 0):.2f}</td>'
            f'<td class="{cls_s}">{s.get("ls_sharpe", 0):.2f}</td>'
            f'<td class="{cls_s}">{s.get("ls_annual", 0)*100:.1f}%</td>'
            f'<td>{s.get("ls_mdd", 0)*100:.1f}%</td>'
            f'<td>{s.get("win_rate", 0)*100:.0f}%</td>'
            f'<td>{s.get("g10_annual", 0)*100:.1f}%</td></tr>'
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>因子总览 · {start} ~ {end}（全部 {len(stats)} 因子）</title>
<style>
:root {{ --bg:#eef2f7; --panel:#fff; --fg:#0f172a; --muted:#64748b; --line:#e2e8f0; --primary:#1e4d8c; --pos:#16a34a; --neg:#dc2626; }}
* {{ box-sizing:border-box }}
body {{ margin:0; font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif; color:var(--fg); background:var(--bg) }}
header {{ background:linear-gradient(135deg,#0f2744,#1e4d8c 60%,#0d9488); color:#fff; padding:36px 48px 28px }}
header h1 {{ margin:0 0 8px; font-size:1.85rem }}
header .sub {{ opacity:0.85; font-size:0.9rem; margin-top:4px }}
main {{ max-width:1400px; margin:0 auto; padding:24px }}
.cards {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:14px; margin:20px 0 }}
.metric {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px; box-shadow:0 4px 24px rgba(15,23,42,0.06); text-align:center }}
.metric b {{ display:block; font-size:1.6rem; color:var(--primary) }}
.metric span {{ color:var(--muted); font-size:0.78rem }}
.pos {{ color:var(--pos) }} .neg {{ color:var(--neg) }}
.chart-grid {{ display:grid; grid-template-columns:1fr; gap:20px; margin:20px 0 }}
img {{ width:100%; border:1px solid var(--line); border-radius:10px; background:#fff }}
table {{ width:100%; border-collapse:collapse; font-size:0.83rem; margin:10px 0 20px }}
th {{ background:#f1f5f9; color:#475569; padding:8px 10px; text-align:left; font-weight:600; border-bottom:2px solid #cbd5e1; position:sticky; top:0 }}
td {{ padding:7px 10px; border-bottom:1px solid var(--line) }}
tr:hover td {{ background:#f8fafc }}
.rank {{ color:#94a3b8; font-weight:600; width:36px }}
code {{ background:#f1f5f9; padding:2px 6px; border-radius:4px; font-size:0.78rem; color:#1e293b }}
.tag {{ display:inline-block; padding:1px 6px; border-radius:10px; font-size:0.68rem; margin-left:4px; vertical-align:middle }}
.tag-flip {{ background:#fef3c7; color:#92400e }}
h2 {{ font-size:1.05rem; color:var(--primary); margin:24px 0 10px; border-bottom:1px solid var(--line); padding-bottom:6px }}
.badge {{ display:inline-block; background:linear-gradient(90deg,#ede9fe,#dbeafe); color:#5b21b6; padding:2px 10px; border-radius:12px; font-size:0.72rem; font-weight:600; margin-left:8px }}
</style>
</head>
<body>
<header>
  <h1>因子总览 · {start} ~ {end} <span class="badge">⚡ quant_evaluator</span></h1>
  <div class="sub">
    {len(stats)} 个因子 · 5450 标的 · {n_days} 个交易日 ·
    factor_engine 落值 + quant_evaluator 评估 ·
    生成 {datetime.now().strftime("%Y-%m-%d %H:%M")}
  </div>
</header>
<main>
  <div class="cards">
    <div class="metric"><b>{len(stats)}</b><span>因子总数</span></div>
    <div class="metric"><b>{mean_ic:.4f}</b><span>平均 RankIC</span></div>
    <div class="metric"><b>{mean_sharpe:.2f}</b><span>平均 LS Sharpe</span></div>
    <div class="metric"><b>{pos_sharpe}</b><span>Sharpe&gt;0 因子</span></div>
    <div class="metric"><b>{n_flipped}</b><span>IC&lt;0 已翻正</span></div>
  </div>

  <div class="chart-grid"><img src="data:image/png;base64,{chart_b64}" alt="LS Sharpe 排序"/></div>

  <h2>全部 {len(stats)} 个因子</h2>
  <table>
    <thead><tr><th>#</th><th>因子</th><th>RankIC</th><th>IR</th><th>LS Sharpe</th><th>LS 年化</th><th>LS 回撤</th><th>胜率</th><th>G10 年化</th></tr></thead>
    <tbody>
    {''.join(rows)}
    </tbody>
  </table>
</main>
</body>
</html>
"""
    OUT.write_text(html, encoding="utf-8")
    print(f"[456index] 首页已写入 {OUT} ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
