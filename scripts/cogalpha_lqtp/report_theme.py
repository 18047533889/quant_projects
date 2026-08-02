#!/usr/bin/env python3
"""Shared HTML/CSS theme and matplotlib styling for factor evaluation reports."""
from __future__ import annotations

import html as html_lib
from typing import Any

# ── Matplotlib palette ──────────────────────────────────────────────────────
PALETTE = {
    "primary": "#1e4d8c",
    "secondary": "#0d9488",
    "accent": "#7c3aed",
    "positive": "#059669",
    "negative": "#dc2626",
    "neutral": "#64748b",
    "grid": "#e2e8f0",
    "bg": "#f8fafc",
}


def setup_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#fafbfc",
            "axes.edgecolor": PALETTE["grid"],
            "axes.labelcolor": "#334155",
            "axes.titleweight": "600",
            "axes.titlesize": 12,
            "axes.titlepad": 10,
            "xtick.color": "#64748b",
            "ytick.color": "#64748b",
            "grid.color": PALETTE["grid"],
            "grid.alpha": 0.7,
            "grid.linestyle": "-",
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Segoe UI",
                "Microsoft YaHei",
                "PingFang SC",
                "Helvetica Neue",
                "DejaVu Sans",
                "Arial",
            ],
        }
    )


def _metric_tone(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "neutral"
    if v != v:
        return "neutral"
    if v > 0.001:
        return "good"
    if v < -0.001:
        return "bad"
    return "neutral"


def metric_card(label: str, value: Any, *, fmt_fn: Any = None) -> str:
    display = fmt_fn(value) if fmt_fn else str(value)
    tone = _metric_tone(value)
    return (
        f'<div class="metric metric-{tone}">'
        f'<div class="metric-value">{html_lib.escape(str(display))}</div>'
        f'<div class="metric-label">{html_lib.escape(label)}</div>'
        f"</div>"
    )


def metrics_grid(cards: list[str], *, cols: int = 4) -> str:
    if not cards:
        return ""
    return f'<div class="metrics metrics-c{cols}">{"".join(cards)}</div>'


def section_block(
    section_id: str,
    number: str,
    title: str,
    body: str,
    *,
    subtitle: str = "",
) -> str:
    sub = f'<p class="section-sub">{html_lib.escape(subtitle)}</p>' if subtitle else ""
    return f"""
<section class="card" id="{html_lib.escape(section_id)}">
  <div class="section-head">
    <span class="section-num">{html_lib.escape(number)}</span>
    <div>
      <h2>{html_lib.escape(title)}</h2>
      {sub}
    </div>
  </div>
  <div class="section-body">{body}</div>
</section>"""


def chart_grid(*charts: str) -> str:
    items = [c for c in charts if c.strip()]
    if not items:
        return ""
    cls = "chart-grid chart-grid-2" if len(items) >= 2 else "chart-grid"
    figures = "".join(f'<figure class="chart-fig">{c}</figure>' for c in items)
    return f'<div class="{cls}">{figures}</div>'


def img_tag(b64: str, alt: str) -> str:
    if not b64:
        return ""
    return f'<img class="chart-img" alt="{html_lib.escape(alt)}" src="data:image/png;base64,{b64}"/>'


REPORT_CSS = """
:root {
  --bg: #eef2f7;
  --panel: #ffffff;
  --fg: #0f172a;
  --muted: #64748b;
  --line: #e2e8f0;
  --primary: #1e4d8c;
  --primary-light: #dbeafe;
  --teal: #0d9488;
  --good: #059669;
  --bad: #dc2626;
  --shadow: 0 4px 24px rgba(15, 23, 42, 0.06);
  --radius: 14px;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
  color: var(--fg);
  background: var(--bg);
  line-height: 1.6;
}
.page { max-width: 1180px; margin: 0 auto; padding: 0 20px 48px; }
.hero {
  background: linear-gradient(135deg, #0f2744 0%, #1e4d8c 55%, #0d9488 100%);
  color: #fff;
  padding: 36px 40px 28px;
  border-radius: 0 0 var(--radius) var(--radius);
  margin-bottom: 24px;
  box-shadow: var(--shadow);
}
.hero h1 { margin: 0 0 8px; font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; }
.hero-meta { opacity: 0.88; font-size: 0.9rem; }
.hero-meta code { background: rgba(255,255,255,0.15); padding: 2px 6px; border-radius: 4px; }
.card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 22px 24px;
  margin-bottom: 20px;
  box-shadow: var(--shadow);
}
.section-head { display: flex; gap: 14px; align-items: flex-start; margin-bottom: 16px; }
.section-num {
  flex: 0 0 36px; height: 36px; line-height: 36px; text-align: center;
  background: linear-gradient(135deg, var(--primary), var(--teal));
  color: #fff; font-weight: 700; font-size: 0.85rem; border-radius: 10px;
}
.section-head h2 { margin: 0; font-size: 1.15rem; font-weight: 650; color: var(--fg); }
.section-sub { margin: 4px 0 0; color: var(--muted); font-size: 0.88rem; }
.metrics { display: grid; gap: 12px; margin: 12px 0 16px; }
.metrics-c3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.metrics-c4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.metrics-c5 { grid-template-columns: repeat(5, minmax(0, 1fr)); }
.metric {
  background: linear-gradient(180deg, #fafbfc 0%, #f1f5f9 100%);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 14px 16px;
  border-left: 4px solid var(--neutral, #94a3b8);
}
.metric-good { border-left-color: var(--good); }
.metric-bad { border-left-color: var(--bad); }
.metric-neutral { border-left-color: #94a3b8; }
.metric-value { font-size: 1.25rem; font-weight: 700; color: var(--primary); line-height: 1.2; }
.metric-label { color: var(--muted); font-size: 0.78rem; margin-top: 4px; }
.chart-grid { display: grid; gap: 16px; margin-top: 8px; }
.chart-grid-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.chart-fig { margin: 0; }
.chart-img {
  width: 100%; height: auto; border-radius: 10px;
  border: 1px solid var(--line); background: #fff;
}
.note { color: var(--muted); font-size: 0.88rem; line-height: 1.55; }
.note b { color: var(--fg); }
.hl {
  background: linear-gradient(90deg, #eff6ff, #f0fdfa);
  border: 1px solid #bfdbfe;
  padding: 14px 18px;
  border-radius: 10px;
  margin-bottom: 20px;
  font-size: 0.9rem;
}
.path-badge {
  display: flex; gap: 12px; align-items: flex-start;
  border: 1px solid var(--line); border-radius: 10px;
  padding: 12px 14px; margin: 16px 0; background: #fff;
}
.path-tag { color: #fff; font-weight: 700; font-size: 0.78rem; padding: 4px 10px; border-radius: 6px; }
.path-detail { color: #475569; font-size: 0.85rem; }
pre {
  background: #0f172a; color: #e2e8f0; padding: 14px 16px;
  border-radius: 10px; overflow-x: auto; font-size: 0.78rem; line-height: 1.5;
}
table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
th, td { border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; }
table.kv th { width: 240px; background: #f8fafc; font-weight: 600; color: #475569; }
details { margin-top: 12px; }
details summary { cursor: pointer; color: var(--primary); font-weight: 600; }
.tag-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }
.tag {
  display: inline-block; padding: 4px 10px; border-radius: 999px;
  font-size: 0.75rem; font-weight: 600; background: var(--primary-light); color: var(--primary);
}
@media (max-width: 900px) {
  .metrics-c3, .metrics-c4, .metrics-c5 { grid-template-columns: 1fr 1fr; }
  .chart-grid-2 { grid-template-columns: 1fr; }
  .hero { padding: 24px 20px; }
}
"""


SUMMARY_CSS = """
:root {
  --bg: #eef2f7; --panel: #fff; --fg: #0f172a; --muted: #64748b;
  --line: #e2e8f0; --primary: #1e4d8c; --shadow: 0 4px 24px rgba(15,23,42,0.06);
}
* { box-sizing: border-box; }
body { margin: 0; font-family: "Segoe UI","Microsoft YaHei",system-ui,sans-serif; color: var(--fg); background: var(--bg); }
header.hero {
  background: linear-gradient(135deg, #0f2744, #1e4d8c 60%, #0d9488);
  color: #fff; padding: 40px 48px 32px; box-shadow: var(--shadow);
}
header.hero h1 { margin: 0 0 10px; font-size: 1.85rem; }
header.hero .muted { opacity: 0.85; }
main { max-width: 1400px; margin: 0 auto; padding: 24px; }
.cards { display: grid; grid-template-columns: repeat(5,minmax(0,1fr)); gap: 14px; margin: 20px 0; }
.metric { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 18px; box-shadow: var(--shadow); }
.metric b { display: block; font-size: 1.6rem; color: var(--primary); }
.metric span { color: var(--muted); font-size: 0.82rem; }
.notice {
  background: linear-gradient(90deg,#eff6ff,#f0fdfa); border: 1px solid #bfdbfe;
  padding: 16px 18px; border-radius: 12px; margin: 16px 0; font-size: 0.9rem;
}
table { width: 100%; border-collapse: collapse; background: var(--panel); border-radius: 12px; overflow: hidden; box-shadow: var(--shadow); font-size: 0.82rem; }
th, td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: right; }
th { background: #f1f5f9; font-weight: 650; position: sticky; top: 0; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
tr:hover td { background: #f8fafc; }
a { color: #1d4ed8; text-decoration: none; font-weight: 500; }
a:hover { text-decoration: underline; }
h2 { font-size: 1.15rem; margin: 28px 0 12px; }
@media (max-width:900px) { .cards { grid-template-columns: 1fr 1fr; } header.hero { padding: 24px; } }
"""
