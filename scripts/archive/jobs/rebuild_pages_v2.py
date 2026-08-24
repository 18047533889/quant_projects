#!/usr/bin/env python3
"""用 all_eval.json + zip_orig_lookup.json 重生成 61 详情页
- 嵌入原始 Python code (来自 zip)
- 嵌入 DSL formula (来自 zip)
- 十分层 NAV 曲线 (10 条曲线) + 多空曲线
- 月度 IC 热力图
- IC 时序 SVG
- IC 分布直方图 (红线=mean)
- 摘要表 (含2019-2025 全区间指标)
"""
import sys, os, json, math, io, base64
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROJECT = Path("/home/sunhaiwei/quant_projects")
REPORT_DIR = PROJECT / "factor_engine" / "docs" / "reports" / "2026-08-23"
FACTORS_DIR = REPORT_DIR / "factors"
ZIP_LOOKUP = PROJECT / "factor_delivery_converted" / "zip_orig_lookup.json"
ALL_EVAL = REPORT_DIR / "all_eval_full.json"

# =============== Style ===============
BG = "#eef2f7"
PANEL = "#fff"
FG = "#0f172a"
MUTED = "#64748b"
LINE = "#e2e8f0"
PRIMARY = "#1e4d8c"
POS = "#16a34a"
NEG = "#dc2626"
PLOT_BG = "#ffffff"


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=PLOT_BG, edgecolor="none")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def plot_ic_monthly_heatmap(daily_ic, dates, factor_name):
    if daily_ic is None or len(daily_ic) < 30:
        return ""
    s = pd.Series(daily_ic, index=pd.to_datetime(dates[:len(daily_ic)]))
    s = s.dropna()
    monthly = s.resample("ME").mean().dropna()
    if len(monthly) < 2:
        return ""

    months = monthly.index.to_period("M")
    years = sorted(set(monthly.index.year))
    grid = np.full((len(years), 12), np.nan)
    for dt, v in monthly.items():
        yi = years.index(dt.year)
        mi = dt.month - 1
        grid[yi, mi] = v

    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(grid, aspect="auto", cmap="RdBu", vmin=-0.1, vmax=0.1)
    ax.set_xticks(range(12))
    ax.set_xticklabels([str(m) for m in range(1, 13)])
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)
    ax.set_xlabel("Month")
    ax.set_title(f"{factor_name} — Monthly RankIC", fontsize=10, color=FG, fontweight='bold')
    plt.colorbar(im, ax=ax, label="RankIC", shrink=0.8)
    for yi in range(len(years)):
        for mi in range(12):
            v = grid[yi, mi]
            if not np.isnan(v):
                ax.text(mi, yi, f"{v:.3f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(v) > 0.05 else FG)
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_decile_nav(g_nav, dates, factor_name):
    """10 曲线分层 NAV"""
    if not g_nav or not dates:
        return ""
    dates_dt = pd.to_datetime(dates[:len(g_nav[0]) if g_nav else 0])
    fig, ax = plt.subplots(figsize=(9.5, 4.5))

    colors = plt.cm.RdYlGn_r(np.linspace(0.05, 0.95, 10))
    for k in range(10):
        if k >= len(g_nav):
            continue
        g = g_nav[k]
        if len(g) != len(dates_dt):
            continue
        lw = 1.4 if k == 9 else 0.9
        ls = '-' if k in (0, 9) else '-'
        alpha = 0.95 if k in (0, 9) else 0.7
        ax.plot(dates_dt, g, color=colors[k], linewidth=lw, linestyle=ls,
                label=f"G{k+1}", alpha=alpha)

    ax.axhline(1.0, color="gray", linewidth=0.7, linestyle="--", alpha=0.7)
    ax.set_title(f"{factor_name} — Decile NAV (10 Quantiles)",
                 fontsize=11, color=FG, fontweight='bold')
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.legend(fontsize=8, loc="upper left", ncol=5, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_long_short_nav(ls_nav, dates, factor_name):
    """Long-Short NAV"""
    if not ls_nav or not dates:
        return ""
    dates_dt = pd.to_datetime(dates[:len(ls_nav)])
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.plot(dates_dt, ls_nav, color="#7c3aed", linewidth=2, label="Long-Short (G10-G1)")
    ax.axhline(1.0, color="gray", linewidth=0.7, linestyle="--", alpha=0.7)
    # 标记年化
    if len(ls_nav) > 252:
        n_years = len(ls_nav) / 252
        ann = ls_nav[-1] ** (1 / n_years) - 1
        ax.text(0.02, 0.95, f"年化: {ann*100:+.1f}%",
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_title(f"{factor_name} — Long-Short NAV",
                 fontsize=11, color=FG, fontweight='bold')
    ax.set_ylabel("NAV")
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_ic_distribution(daily_ic, factor_name):
    """IC 分布直方图，红线 = mean"""
    if daily_ic is None or len(daily_ic) < 5:
        return ""
    s = pd.Series(daily_ic)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 5:
        return ""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(s.values, bins=50, color=PRIMARY, alpha=0.75, edgecolor="white", linewidth=0.5)
    # 红线 = mean (不是 0)
    mean_v = float(s.mean())
    ax.axvline(mean_v, color="#dc2626", linewidth=2, linestyle="-", label=f"Mean = {mean_v:+.4f}")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6, label="Zero")
    ax.set_title(f"{factor_name} — RankIC Distribution",
                 fontsize=10, color=FG, fontweight='bold')
    ax.set_xlabel("RankIC")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    # 在图上标 mean 和 std
    std_v = float(s.std())
    text = f"μ={mean_v:+.4f}\nσ={std_v:.4f}\nN={len(s)}"
    ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_ic_timeseries_svg(daily_ic, dates, factor_name):
    """IC 时序 SVG（不丢 0）"""
    if not daily_ic or not dates:
        return ""
    s = pd.Series(daily_ic, index=pd.to_datetime(dates[:len(daily_ic)]))
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0)
    if len(s) < 2:
        return ""

    W, H = 720, 200
    x_pad, y_pad = 40, 20
    plot_w = W - x_pad * 2
    plot_h = H - y_pad * 2

    v_min, v_max = -0.15, 0.15
    v_range = v_max - v_min
    n = len(s)
    x_vals = np.arange(n)
    y_norm = np.clip((s.values - v_min) / v_range, 0, 1)
    xs = x_pad + (x_vals / max(n - 1, 1)) * plot_w
    ys = y_pad + (1 - y_norm) * plot_h
    zero_y = y_pad + (1 - (0 - v_min) / v_range) * plot_h

    bar_colors = ["#16a34a" if v > 0.001 else ("#dc2626" if v < -0.001 else "#cbd5e1") for v in s.values]
    bars = ""
    for i in range(n):
        cx = xs[i]
        cy = ys[i]
        bar_h = abs(cy - zero_y)
        y_top = min(cy, zero_y)
        bars += '<rect x="{:.1f}" y="{:.1f}" width="2.4" height="{:.1f}" fill="{}" opacity="0.85"/>\n'.format(
            cx - 1.2, y_top, bar_h, bar_colors[i]
        )

    # 年份标签
    x_labels = []
    seen_years = set()
    for i in range(n):
        yr = s.index[i].year
        if yr not in seen_years and (i % max(1, n // 10) == 0):
            seen_years.add(yr)
            x_labels.append(
                '<text x="{:.1f}" y="{}" text-anchor="middle" font-size="8" fill="#64748b">{}</text>\n'.format(
                    xs[i], H - 4, str(yr)
                )
            )

    # mean line
    mean_y_norm = (s.mean() - v_min) / v_range
    if 0 <= mean_y_norm <= 1:
        mean_y = y_pad + (1 - mean_y_norm) * plot_h
        bars += f'<line x1="{x_pad}" y1="{mean_y:.1f}" x2="{W-x_pad}" y2="{mean_y:.1f}" stroke="#7c3aed" stroke-width="1.2" stroke-dasharray="4,3"/>\n'

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'style="width:100%;max-height:220px;font-family:Segoe UI,Microsoft YaHei,system-ui,sans-serif">\n'
        + bars +
        f'<line x1="{x_pad}" y1="{zero_y:.1f}" x2="{W-x_pad}" y2="{zero_y:.1f}" stroke="#94a3b8" stroke-width="0.5" stroke-dasharray="3,3"/>\n'
        + ''.join(x_labels) +
        '</svg>'
    )
    return svg


def fmt_num(v, pct=False, signed=False):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    if pct:
        return f"{v*100:+.2f}%" if signed else f"{v*100:.2f}%"
    if isinstance(v, (int, np.integer)):
        return f"{v}"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.2f}"
    return f"{v:.4f}"


def build_detail_html(name: str, eval_data: dict, zip_data: dict) -> str:
    """Build HTML for one factor"""
    is_flipped = "_flipped" in name

    metrics = eval_data if eval_data else {}
    flip_note = ""
    if is_flipped:
        flip_note = '<span class="badge badge-yellow">⚠ 已翻转 (历史 IC&lt;0)</span>'

    # DSL formula
    formula_dsl = (zip_data.get('formula_dsl') or '—') if zip_data else '—'
    if formula_dsl is None:
        formula_dsl = '—'
    formula_dsl_esc = str(formula_dsl).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    rationale = (zip_data.get('rationale', '') or '') if zip_data else ''
    if rationale:
        rationale = rationale[:500] + ('...' if len(rationale) > 500 else '')
    rationale_esc = rationale.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    category = (zip_data.get('category', '') or '') if zip_data else ''

    # Original code (from zip)
    orig_code = (zip_data.get('code', '') or '# (无 zip 原始 code)') if zip_data else '# (无 zip 数据)'
    orig_code_esc = orig_code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    # Metric extraction
    mean_ic = metrics.get('mean_ic', 0)
    mean_rankic = metrics.get('mean_rankic', 0)
    rankic_ir = metrics.get('rankic_ir', 0)
    ic_ir = metrics.get('ic_ir', 0)
    ic_winrate = metrics.get('ic_winrate', 0)
    rankic_winrate = metrics.get('rankic_winrate', 0)
    ls_sharpe = metrics.get('ls_sharpe', 0)
    ls_annual = metrics.get('ls_annual', 0)
    ls_cum = metrics.get('ls_cum', 0)
    ls_mdd = metrics.get('ls_mdd', 0)
    ls_winrate = metrics.get('ls_winrate', 0)
    g_annual = metrics.get('g_annual', [0] * 10)
    g1_annual = g_annual[0] if len(g_annual) > 0 else 0
    g10_annual = g_annual[-1] if len(g_annual) > 0 else 0
    n_periods = metrics.get('n_periods', 0)

    daily_ic = metrics.get('daily_rankic', [])
    daily_ic_full = metrics.get('daily_ic', [])
    dates = metrics.get('dates', [])
    ls_nav = metrics.get('ls_nav', [])
    g_nav = metrics.get('g_nav', [])

    # Charts
    monthly_chart = plot_ic_monthly_heatmap(daily_ic, dates, name)
    decile_chart = plot_decile_nav(g_nav, dates, name)
    ls_chart = plot_long_short_nav(ls_nav, dates, name)
    dist_chart = plot_ic_distribution(daily_ic, name)
    svg_ts = plot_ic_timeseries_svg(daily_ic, dates, name)

    # HTML pieces
    pcls = lambda v: "pos" if v >= 0 else "neg"

    monthly_img = f'<img src="data:image/png;base64,{monthly_chart}" style="width:100%;border-radius:8px;"/>' if monthly_chart else '<div class="zero-notice">月度 IC 数据不足</div>'
    decile_img = f'<img src="data:image/png;base64,{decile_chart}" style="width:100%;border-radius:8px;"/>' if decile_chart else '<div class="zero-notice">十分层数据不足</div>'
    ls_img = f'<img src="data:image/png;base64,{ls_chart}" style="width:100%;border-radius:8px;"/>' if ls_chart else '<div class="zero-notice">多空数据不足</div>'
    dist_img = f'<img src="data:image/png;base64,{dist_chart}" style="width:100%;border-radius:8px;"/>' if dist_chart else '<div class="zero-notice">分布数据不足</div>'

    ic_ts_section = ''
    if svg_ts:
        ic_ts_section = f'''
<div class="card">
<h2>RankIC 时序 <span class="hint">(紫线 = mean，绿/红 = 正/负日 IC)</span></h2>
<div class="chart-wrap">{svg_ts}</div>
</div>'''

    # G 表
    g_table_rows = ''
    for k in range(10):
        v = g_annual[k] if k < len(g_annual) else 0
        g_table_rows += f'<tr><td>G{k+1}</td><td class="{pcls(v)}">{fmt_num(v, pct=True, signed=True)}</td></tr>'

    date_range = "—"
    if dates:
        date_range = f"{dates[0]} ~ {dates[-1]}"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{name}</title>
<style>
:root {{
  --bg:{BG}; --panel:{PANEL}; --fg:{FG}; --muted:{MUTED};
  --line:{LINE}; --primary:{PRIMARY}; --pos:{POS}; --neg:{NEG};
}}
* {{ box-sizing:border-box }}
body {{ margin:0; font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif; color:var(--fg); background:var(--bg) }}
header {{ background:linear-gradient(135deg,#0f2744,#1e4d8c 60%,#0d9488); color:#fff; padding:28px 48px 22px }}
header h1 {{ margin:0 0 6px; font-size:1.5rem; word-break:break-all }}
header .meta {{ opacity:0.85; font-size:0.85rem; margin-top:4px }}
main {{ max-width:1180px; margin:0 auto; padding:24px }}
.back {{ display:inline-block; margin-bottom:16px; color:#93c5fd; font-weight:500; text-decoration:none; font-size:0.88rem }}
.back:hover {{ text-decoration:underline }}
.grid-4 {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:12px }}
.grid-3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:12px }}
.metric {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px; text-align:center; box-shadow:0 4px 24px rgba(15,23,42,0.06) }}
.metric b {{ display:block; font-size:1.35rem }}
.metric span {{ color:var(--muted); font-size:0.73rem }}
.pos {{ color:var(--pos) }}
.neg {{ color:var(--neg) }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px 20px; box-shadow:0 4px 24px rgba(15,23,42,0.06); margin-bottom:16px }}
h2 {{ font-size:0.95rem; color:var(--primary); margin:0 0 12px; border-bottom:1px solid var(--line); padding-bottom:8px; display:flex; align-items:center; justify-content:space-between }}
h2 .hint {{ color:var(--muted); font-weight:400; font-size:0.78rem }}
.formula-wrap {{ background:#f8fafc; border:1px solid var(--line); border-radius:8px; padding:16px; font-family:"Courier New",monospace; font-size:0.85rem; word-break:break-all; line-height:1.7; white-space:pre-wrap; color:#0f172a }}
.code-wrap {{ background:#0f172a; color:#e2e8f0; border-radius:8px; padding:16px; font-family:"Courier New",monospace; font-size:0.78rem; word-break:break-all; line-height:1.55; white-space:pre-wrap; overflow-x:auto; max-height:480px; overflow-y:auto }}
.code-wrap .kw {{ color:#f472b6 }}
.code-wrap .str {{ color:#a3e635 }}
.code-wrap .num {{ color:#fbbf24 }}
.meta-table {{ width:100%; border-collapse:collapse; font-size:0.85rem }}
.meta-table td {{ padding:8px 12px; border-bottom:1px solid var(--line) }}
.meta-table td:first-child {{ color:var(--muted); width:160px }}
.meta-table tr:last-child td {{ border-bottom:none }}
.chart-wrap {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; margin-bottom:16px }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:12px; font-size:0.73rem; margin-left:6px }}
.badge-green {{ background:#dcfce7; color:#166534 }}
.badge-blue {{ background:#dbeafe; color:#1e40af }}
.badge-yellow {{ background:#fef3c7; color:#92400e }}
.badge-red {{ background:#fee2e2; color:#991b1b }}
.zero-notice {{ background:#fef9c3; border:1px solid #fde047; padding:12px 16px; border-radius:8px; font-size:0.85rem; color:#854d0e; margin-bottom:16px }}
img {{ border-radius:8px }}
.two-col {{ display:grid; grid-template-columns:1fr 320px; gap:16px }}
@media (max-width: 900px) {{ .two-col {{ grid-template-columns:1fr }} }}
.g-table {{ width:100%; font-size:0.8rem; border-collapse:collapse }}
.g-table td {{ padding:4px 8px; border-bottom:1px solid var(--line) }}
.g-table tr:last-child td {{ border-bottom:none }}
.g-table .glabel {{ color:var(--muted); width:32px }}
</style>
</head>
<body>
<header>
<a class="back" href="../index.html">&#8592; 返回汇总</a>
<h1><code>{name}</code> {flip_note}</h1>
<div class="meta">
  {category or '本周因子'} · 回测区间 {date_range} · 共 {n_periods} 个交易日 ·
  生成 {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>
</header>
<main>

<div class="grid-4">
  <div class="metric"><b class="{pcls(mean_rankic)}">{fmt_num(mean_rankic, signed=True)}</b><span>Mean RankIC</span></div>
  <div class="metric"><b class="{pcls(rankic_ir)}">{fmt_num(rankic_ir, signed=True)}</b><span>RankIC IR</span></div>
  <div class="metric"><b class="{pcls(ls_sharpe)}">{fmt_num(ls_sharpe, signed=True)}</b><span>LS Sharpe</span></div>
  <div class="metric"><b class="{pcls(ls_annual)}">{fmt_num(ls_annual, pct=True, signed=True)}</b><span>LS 年化</span></div>
</div>
<div class="grid-4">
  <div class="metric"><b class="{pcls(ls_cum)}">{fmt_num(ls_cum, pct=True, signed=True)}</b><span>LS 累计</span></div>
  <div class="metric"><b class="{pcls(ls_mdd)}">{fmt_num(ls_mdd, pct=True, signed=True)}</b><span>LS 最大回撤</span></div>
  <div class="metric"><b class="{pcls(ls_winrate)}">{fmt_num(ls_winrate, pct=True, signed=True)}</b><span>LS 日胜率</span></div>
  <div class="metric"><b class="{pcls(ic_winrate)}">{fmt_num(ic_winrate, pct=True, signed=True)}</b><span>RankIC 胜率</span></div>
</div>
<div class="grid-3">
  <div class="metric"><b class="{pcls(g10_annual)}">{fmt_num(g10_annual, pct=True, signed=True)}</b><span>G10 (多头) 年化</span></div>
  <div class="metric"><b class="{pcls(g1_annual)}">{fmt_num(g1_annual, pct=True, signed=True)}</b><span>G1 (空头) 年化</span></div>
  <div class="metric"><b>{n_periods}</b><span>回测交易日</span></div>
</div>

<div class="card">
<h2>因子 DSL 公式</h2>
<div class="formula-wrap">{formula_dsl_esc}</div>
</div>

<div class="two-col">
<div class="card">
<h2>原始 Python 代码 (来自 factor_delivery.zip) <span class="hint">factor_engine 落值</span></h2>
<div class="code-wrap">{orig_code_esc}</div>
{f'<div class="hint" style="margin-top:8px;color:var(--muted);font-size:0.82rem"><b>机制说明:</b> {rationale_esc}</div>' if rationale_esc else ''}
</div>

<div class="card">
<h2>十分层年化收益</h2>
<table class="g-table">
{g_table_rows}
</table>
<div class="hint" style="margin-top:10px;color:var(--muted);font-size:0.78rem">每组年化收益 (累计 2019~{dates[-1][:4] if dates else "2025"})，G10-G1 即多空</div>
</div>
</div>

{ic_ts_section}

<div class="card">
<h2>月度 RankIC 热力图</h2>
{monthly_img}
</div>

<div class="card">
<h2>十分层净值曲线 (10 条 G1~G10)</h2>
{decile_img}
</div>

<div class="card">
<h2>多空净值曲线 (G10 - G1)</h2>
{ls_img}
</div>

<div class="card">
<h2>RankIC 分布</h2>
{dist_img}
</div>

<div class="card">
<h2>因子统计摘要</h2>
<table class="meta-table">
<tr><td>因子名称</td><td><code>{name}</code></td></tr>
<tr><td>所属分类</td><td>{category or '—'}</td></tr>
<tr><td>回测区间</td><td>{date_range}</td></tr>
<tr><td>已翻转</td><td>{'是' if is_flipped else '否'}</td></tr>
<tr><td>Mean RankIC</td><td>{fmt_num(mean_rankic, signed=True)}</td></tr>
<tr><td>Mean IC (Pearson)</td><td>{fmt_num(mean_ic, signed=True)}</td></tr>
<tr><td>RankIC IR</td><td>{fmt_num(rankic_ir, signed=True)}</td></tr>
<tr><td>RankIC 胜率</td><td>{fmt_num(rankic_winrate, pct=True, signed=True)}</td></tr>
<tr><td>LS Sharpe</td><td>{fmt_num(ls_sharpe, signed=True)}</td></tr>
<tr><td>LS 年化收益</td><td>{fmt_num(ls_annual, pct=True, signed=True)}</td></tr>
<tr><td>LS 累计收益</td><td>{fmt_num(ls_cum, pct=True, signed=True)}</td></tr>
<tr><td>LS 最大回撤</td><td>{fmt_num(ls_mdd, pct=True, signed=True)}</td></tr>
<tr><td>LS 日胜率</td><td>{fmt_num(ls_winrate, pct=True, signed=True)}</td></tr>
<tr><td>G10 (多头) 年化</td><td>{fmt_num(g10_annual, pct=True, signed=True)}</td></tr>
<tr><td>G1 (空头) 年化</td><td>{fmt_num(g1_annual, pct=True, signed=True)}</td></tr>
<tr><td>回测交易日</td><td>{n_periods}</td></tr>
</table>
</div>

</main>
</body>
</html>
"""
    return html


def main():
    print("=" * 60)
    print(f"重生成 61 详情页 (含原始 Python + DSL)")
    print("=" * 60)

    with open(ZIP_LOOKUP) as f:
        zip_idx = json.load(f)
    with open(ALL_EVAL) as f:
        all_eval = json.load(f)

    factor_names = sorted([
        f.stem.replace('factor_', '')
        for f in FACTORS_DIR.glob("factor_*.html")
    ])
    print(f"因子数: {len(factor_names)}")

    ok, miss_zip, miss_eval = 0, 0, 0
    for name in factor_names:
        # ZIP lookup
        zd = None
        for c in [f'factor_{name}', f'factor_{name.replace("_flipped", "")}']:
            if c in zip_idx:
                zd = zip_idx[c]
                break
        if not zd:
            miss_zip += 1
            zd = {}

        # Eval data
        ed = all_eval.get(name, {})
        if not ed:
            miss_eval += 1

        html = build_detail_html(name, ed, zd)
        out_path = FACTORS_DIR / f"factor_{name}.html"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        ok += 1

    print(f"完成 {ok}, 缺 zip {miss_zip}, 缺 eval {miss_eval}")


if __name__ == "__main__":
    main()