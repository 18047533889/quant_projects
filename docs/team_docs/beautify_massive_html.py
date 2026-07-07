#!/usr/bin/env python3
"""Beautify HTML: left sidebar nav, schema tables, collapsible head/samples."""
import re
from pathlib import Path

HTML_PATH = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"

NEW_STYLE = r"""
    :root {
      --bg: #eef2f6;
      --card: #ffffff;
      --border: #cbd5e1;
      --text: #0f172a;
      --muted: #64748b;
      --blue: #dbeafe;
      --green: #dcfce7;
      --amber: #fef3c7;
      --red: #fee2e2;
      --purple: #ede9fe;
      --gray: #f8fafc;
      --blue-deep: #1d4ed8;
      --green-deep: #166534;
      --amber-deep: #b45309;
      --red-deep: #b91c1c;
      --purple-deep: #7c3aed;
      --sidebar-w: 280px;
      --header-h: 0px;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.65;
      font-size: 15px;
    }

    .app-layout { display: flex; min-height: 100vh; }
    .sidebar {
      position: fixed;
      left: 0; top: 0; bottom: 0;
      width: var(--sidebar-w);
      background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
      color: #e2e8f0;
      overflow-y: auto;
      z-index: 100;
      border-right: 1px solid #334155;
      scrollbar-width: thin;
    }
    .sidebar-brand {
      padding: 18px 16px 12px;
      border-bottom: 1px solid #334155;
      position: sticky; top: 0;
      background: #0f172a;
      z-index: 2;
    }
    .sidebar-brand h2 {
      margin: 0;
      font-size: 14px;
      font-weight: 700;
      color: #f8fafc;
      line-height: 1.35;
    }
    .sidebar-brand p { margin: 6px 0 0; font-size: 11px; color: #94a3b8; }
    .sidebar-nav { list-style: none; margin: 0; padding: 8px 0 24px; }
    .sidebar-nav li { margin: 0; }
    .sidebar-nav a {
      display: flex;
      align-items: flex-start;
      gap: 8px;
      padding: 7px 14px 7px 12px;
      color: #cbd5e1;
      text-decoration: none;
      font-size: 12.5px;
      line-height: 1.4;
      border-left: 3px solid transparent;
      transition: background .15s, color .15s;
    }
    .sidebar-nav a:hover { background: rgba(59,130,246,.15); color: #fff; }
    .sidebar-nav a.active { background: rgba(59,130,246,.25); color: #fff; border-left-color: #60a5fa; }
    .sidebar-nav .nav-num {
      flex-shrink: 0;
      min-width: 28px;
      font-weight: 700;
      font-size: 11px;
      color: #60a5fa;
      font-family: ui-monospace, monospace;
    }
    .sidebar-nav .nav-label { flex: 1; }
    .sidebar-nav .nav-group {
      padding: 10px 14px 4px;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .06em;
      color: #64748b;
    }

    .main-content {
      margin-left: var(--sidebar-w);
      flex: 1;
      min-width: 0;
      padding: 20px 24px 40px;
    }
    .wrap { max-width: 1100px; margin: 0 auto; }

    .header {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px 28px;
      margin-bottom: 20px;
      box-shadow: 0 4px 20px rgba(15,23,42,.06);
    }
    .header h1 { margin: 0 0 8px; font-size: 26px; letter-spacing: -.02em; }
    .header .meta { color: var(--muted); font-size: 13px; margin: 0 0 14px; }
    .header .purpose {
      background: var(--blue);
      border-left: 4px solid var(--blue-deep);
      padding: 12px 16px;
      border-radius: 0 8px 8px 0;
      margin: 0;
      font-size: 14px;
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }
    .summary-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
    }
    .summary-card h3 { margin: 0 0 8px; font-size: 13px; color: var(--muted); font-weight: 600; }
    .summary-card .num { font-size: 26px; font-weight: 700; }

    section {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 24px 28px;
      margin-bottom: 20px;
      box-shadow: 0 1px 4px rgba(15,23,42,.05);
      scroll-margin-top: 16px;
    }
    section h2 {
      margin: 0 0 8px;
      font-size: 21px;
      padding-bottom: 10px;
      border-bottom: 2px solid var(--border);
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    section h2 .sec-badge {
      font-size: 12px;
      font-weight: 700;
      font-family: ui-monospace, monospace;
      background: var(--blue);
      color: var(--blue-deep);
      padding: 3px 10px;
      border-radius: 6px;
      flex-shrink: 0;
    }
    section .section-desc { color: var(--muted); font-size: 14px; margin: 0 0 18px; }
    h3 { font-size: 17px; margin: 22px 0 10px; color: #334155; scroll-margin-top: 16px; }
    h4 { font-size: 15px; margin: 14px 0 8px; color: #475569; }

    table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 13px;
      margin: 12px 0 18px;
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
    }
    th, td {
      border-bottom: 1px solid var(--border);
      border-right: 1px solid var(--border);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }
    th:last-child, td:last-child { border-right: none; }
    tr:last-child td { border-bottom: none; }
    th {
      background: linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 100%);
      font-weight: 700;
      font-size: 12px;
      text-transform: none;
      color: #334155;
    }
    tr:nth-child(even) td { background: #fafbfc; }
    tr:hover td { background: #f0f9ff; }

    table.table-schema th:first-child,
    table.table-schema td:first-child { min-width: 160px; }
    table.table-schema code {
      font-size: 12.5px;
      background: #e0e7ff;
      color: #1e3a8a;
      padding: 3px 8px;
      border-radius: 5px;
      font-weight: 600;
      border: 1px solid #c7d2fe;
    }
    table.table-cols th:nth-child(2),
    table.table-cols td:nth-child(2) {
      font-family: ui-monospace, monospace;
      font-size: 12px;
      line-height: 1.7;
      word-break: break-word;
    }
    table.table-cols code {
      display: inline-block;
      margin: 2px 4px 2px 0;
      background: #f0fdf4;
      border: 1px solid #bbf7d0;
      color: #166534;
    }

    code, .path {
      font-family: ui-monospace, 'Cascadia Code', monospace;
      font-size: 12px;
      background: #f1f5f9;
      padding: 2px 6px;
      border-radius: 4px;
      word-break: break-word;
    }

    details.fold, details.code-fold, details.sample-fold {
      border: 1px solid var(--border);
      border-radius: 10px;
      margin: 12px 0 16px;
      background: #f8fafc;
      overflow: hidden;
    }
    details.fold > summary,
    details.code-fold > summary,
    details.sample-fold > summary {
      cursor: pointer;
      padding: 12px 16px;
      font-weight: 600;
      font-size: 14px;
      list-style: none;
      background: linear-gradient(90deg, #f1f5f9, #fff);
      user-select: none;
    }
    details.fold > summary::before,
    details.sample-fold > summary::before { content: '▸ '; color: var(--blue-deep); }
    details.fold[open] > summary::before,
    details.sample-fold[open] > summary::before { content: '▾ '; }
    details.fold .fold-body,
    details.sample-fold .fold-body { padding: 0 16px 14px; }
    details.code-fold pre {
      margin: 0;
      border-radius: 0 0 10px 10px;
    }

    .json-block {
      background: #0f172a;
      color: #a5f3fc;
      border: none;
      border-radius: 8px;
      padding: 14px 16px;
      font-family: ui-monospace, monospace;
      font-size: 12px;
      overflow-x: auto;
      margin: 8px 0;
      white-space: pre-wrap;
      line-height: 1.5;
    }

    .badge {
      display: inline-block;
      padding: 2px 10px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
    }
    .badge-p0 { background: var(--red); color: var(--red-deep); }
    .badge-p1 { background: #ffedd5; color: var(--amber-deep); }
    .badge-p2 { background: var(--amber); color: var(--amber-deep); }

    pre {
      background: #0f172a;
      color: #e2e8f0;
      padding: 14px 16px;
      border-radius: 8px;
      overflow-x: auto;
      font-size: 12px;
      line-height: 1.5;
      margin: 12px 0;
    }

    .box-problem { background: var(--red); border-left: 4px solid var(--red-deep); padding: 12px 14px; border-radius: 0 8px 8px 0; margin: 12px 0; font-size: 14px; }
    .box-fix { background: var(--green); border-left: 4px solid var(--green-deep); padding: 12px 14px; border-radius: 0 8px 8px 0; margin: 12px 0; font-size: 14px; }
    .box-warn { background: var(--amber); border-left: 4px solid var(--amber-deep); padding: 12px 14px; border-radius: 0 8px 8px 0; margin: 12px 0; font-size: 14px; }
    .box-info { background: var(--blue); border-left: 4px solid var(--blue-deep); padding: 12px 14px; border-radius: 0 8px 8px 0; margin: 12px 0; font-size: 14px; }
    .box-danger { background: var(--red); border-left: 4px solid var(--red-deep); padding: 12px 14px; border-radius: 0 8px 8px 0; margin: 12px 0; }

    .task-card { border: 1px solid var(--border); border-radius: 10px; margin-bottom: 14px; overflow: hidden; }
    .task-card-header { padding: 12px 16px; background: var(--gray); border-bottom: 1px solid var(--border); }
    .checklist { list-style: none; padding: 0; }
    .checklist li { padding: 8px 0 8px 28px; position: relative; border-bottom: 1px dashed var(--border); }
    .checklist li::before { content: '☐'; position: absolute; left: 4px; color: var(--muted); }
    .flow-steps { counter-reset: step; list-style: none; padding: 0; }
    .flow-steps li { counter-increment: step; padding: 10px 10px 10px 44px; position: relative; border-bottom: 1px dashed var(--border); }
    .flow-steps li::before {
      content: counter(step);
      position: absolute; left: 8px; top: 8px;
      background: var(--blue-deep); color: #fff;
      width: 24px; height: 24px; border-radius: 50%;
      text-align: center; font-size: 12px; line-height: 24px;
    }
    footer { text-align: center; color: var(--muted); font-size: 13px; padding: 32px 0 16px; }
    a.doc-link { font-weight: 600; color: var(--blue-deep); }

    nav.toc.legacy-hide { display: none !important; }

    @media (max-width: 960px) {
      .sidebar { width: 100%; position: relative; height: auto; max-height: 40vh; }
      .main-content { margin-left: 0; }
      .app-layout { flex-direction: column; }
    }
"""

HEAD_SAMPLES = r'''
      <details class="fold" id="sec-head-samples">
        <summary>📋 核心数据集 head(3) 与列清单（点击展开，摘自数据报告 §4）</summary>
        <div class="fold-body">
          <p class="section-desc">raw 层真实 parquet 前三行；列名字段表见 <a href="#sec-fields">§7</a>。</p>

          <details class="sample-fold">
            <summary>SIP 日 K · <code>day_aggs_v1</code>（8 列 · 不复权）</summary>
            <div class="fold-body">
              <table class="table-schema">
                <thead><tr><th>字段</th><th>类型</th><th>说明</th></tr></thead>
                <tbody>
                  <tr><td><code>ticker</code></td><td>string</td><td>证券代码</td></tr>
                  <tr><td><code>open</code></td><td>float</td><td>开盘价</td></tr>
                  <tr><td><code>high</code></td><td>float</td><td>最高价</td></tr>
                  <tr><td><code>low</code></td><td>float</td><td>最低价</td></tr>
                  <tr><td><code>close</code></td><td>float</td><td>收盘价</td></tr>
                  <tr><td><code>volume</code></td><td>float</td><td>成交量</td></tr>
                  <tr><td><code>window_start</code></td><td>float</td><td>纳秒时间戳；日 K 锚点 04:00 UTC</td></tr>
                  <tr><td><code>transactions</code></td><td>float</td><td>成交笔数</td></tr>
                </tbody>
              </table>
              <div class="json-block">[
  {"ticker":"A","volume":2869700,"open":25.4,"close":24.49,"high":25.58,"low":24.41,"window_start":1063166400000000000,"transactions":2301},
  {"ticker":"AA","volume":3543400,"open":28.2,"close":27.92,"high":28.7,"low":27.85,"window_start":1063166400000000000,"transactions":3011},
  {"ticker":"AAp","volume":550,"open":75.0,"close":73.44,"high":75.5,"low":72.65,"window_start":1063166400000000000,"transactions":6}
]</div>
            </div>
          </details>

          <details class="sample-fold">
            <summary>REST 日 K · <code>daily_market_summary</code>（已拆股复权）</summary>
            <div class="fold-body">
              <table class="table-schema">
                <thead><tr><th>字段</th><th>类型</th><th>说明</th></tr></thead>
                <tbody>
                  <tr><td><code>T</code></td><td>string</td><td>ticker</td></tr>
                  <tr><td><code>o,h,l,c,v</code></td><td>float</td><td>OHLCV</td></tr>
                  <tr><td><code>t</code></td><td>int64</td><td>窗口起始（毫秒）</td></tr>
                  <tr><td><code>trade_date</code></td><td>string</td><td>交易日</td></tr>
                </tbody>
              </table>
              <div class="json-block">[
  {"T":"HRVE","v":175,"o":3.56,"c":3.64,"h":3.64,"l":3.56,"t":1073077200000,"trade_date":"2004-01-02"},
  {"T":"PAS","v":166400,"o":17.1,"c":17.12,"h":17.2,"l":17.01,"t":1073077200000,"trade_date":"2004-01-02"},
  {"T":"SNBC","v":3734.2,"o":131.75,"c":130.75,"h":136.8,"l":130,"t":1073077200000,"trade_date":"2004-01-02"}
]</div>
            </div>
          </details>

          <details class="sample-fold">
            <summary>利润表 raw · <code>income_statement</code>（PiT 用 filing_date）</summary>
            <div class="fold-body">
              <div class="json-block">{
  "cik": "0000104169",
  "tickers": ["WMT"],
  "filing_date": "2010-06-04",
  "period_end": "2009-04-30",
  "timeframe": "quarterly",
  "revenue": 94242000000.0,
  "basic_earnings_per_share": 0.26
}</div>
            </div>
          </details>

          <details class="sample-fold">
            <summary>新闻 raw · <code>news</code>（秒级 published_utc）</summary>
            <div class="fold-body">
              <table class="table-schema">
                <thead><tr><th>字段</th><th>类型</th><th>说明</th></tr></thead>
                <tbody>
                  <tr><td><code>id</code></td><td>string</td><td>新闻 UUID</td></tr>
                  <tr><td><code>title</code></td><td>string</td><td>标题（去重键）</td></tr>
                  <tr><td><code>published_utc</code></td><td>string</td><td>ISO 秒级 UTC</td></tr>
                  <tr><td><code>tickers</code></td><td>string[]</td><td>cleaned explode</td></tr>
                </tbody>
              </table>
            </div>
          </details>

          <p><a href="Massive原始数据报告.md">完整 26 源 head(3)</a> 见姊妹数据报告 §4。</p>
        </div>
      </details>
'''


def extract_nav_items(html: str) -> list[tuple[str, str, str]]:
    """Return [(id, num, label), ...]"""
    items = []
    for m in re.finditer(
        r'<section id="([^"]+)"[^>]*>.*?<h2[^>]*>(.*?)</h2>',
        html,
        flags=re.DOTALL,
    ):
        sid, h2raw = m.group(1), m.group(2)
        h2 = re.sub(r"<[^>]+>", "", h2raw).strip()
        num_m = re.match(r"^([\d]+[a-z]?|[★][^·]*|核心)", h2)
        if num_m:
            parts = h2.split(" ", 1)
            if h2[0] in "★" or h2.startswith("核心"):
                num, label = "★", h2
            elif re.match(r"^\d", h2):
                sp = h2.split(" ", 1)
                num, label = sp[0], sp[1] if len(sp) > 1 else h2
            else:
                num, label = "·", h2
        else:
            num, label = "·", h2
        items.append((sid, num, label))
    return items


def build_sidebar(items: list[tuple[str, str, str]], *, has_head_samples: bool = False) -> str:
    groups = [
        ("总览", ["sec-arch", "sec-acl-judgment", "sec-readmap", "sec-ok"]),
        ("数据与 P0", ["sec-datasets", "sec-empirical", "sec-p0", "sec-quality", "sec-sre", "sec-p1", "sec-operator-adj", "sec-edge-cases", "sec-acl", "sec-acl-mm", "sec-acl-deep", "sec-acl-deep2", "sec-scenarios"]),
        ("工程", [
            "sec-download",
            "sec-features",
            "sec-atr",
            "sec-fatal-errors",
            "sec-fields",
            "sec-pit-algorithm",
            "sec-time",
            "sec-align",
        ]),
        ("入模前", [
            "sec-before-factor",
            "sec-engine-checklist",
            "sec-returns-policy",
            "sec-stage6-ltm",
            "sec-preproc",
            "sec-pipeline",
            "sec-engineering",
            "sec-clickhouse",
        ]),
        ("补盲点", ["sec-gemini", "sec-cleaned", "sec-blind-spots", "sec-blind-spots-2", "sec-blind-spots-3", "sec-blind-spots-4", "sec-blind-spots-registry", "sec-deep-research"]),
        ("交付", ["sec-tasks", "sec-roadmap", "sec-accept", "sec-cheatsheet"]),
    ]
    id_set = {i[0] for i in items}
    by_id = {i[0]: i for i in items}
    skip_extra = {"sec-data-catalog", "sec-head-samples", "sec-deep-research"}
    extra = [
        i for i in items
        if i[0] not in {x for g in groups for x in g[1]} and i[0] not in skip_extra
    ]
    # code sections
    for sid in ["sec-code-assets", "sec-code-extra", "sec-code-round2", "sec-code-round3"]:
        if sid in id_set:
            extra.append(by_id[sid])

    lines = ['<aside class="sidebar" id="sidebar">', '<div class="sidebar-brand">',
             '<h2>Massive 数据治理手册</h2><p>左侧导航 · 点击跳转</p></div>', '<ul class="sidebar-nav">']
    for gname, gids in groups:
        present = [by_id[g] for g in gids if g in by_id]
        if not present:
            continue
        lines.append(f'<li class="nav-group">{gname}</li>')
        for sid, num, label in present:
            short = label[:42] + ("…" if len(label) > 42 else "")
            lines.append(
                f'<li><a href="#{sid}" data-sec="{sid}">'
                f'<span class="nav-num">{num}</span><span class="nav-label">{short}</span></a></li>'
            )
        if gname == "总览":
            lines.append(
                '<li><a href="#sec-data-catalog" data-sec="sec-data-catalog">'
                '<span class="nav-num">数据</span><span class="nav-label">26 源形态手册（路径·列·样例）</span></a></li>'
            )
    if extra:
        lines.append('<li class="nav-group">附录 / 代码</li>')
        for sid, num, label in extra:
            short = label[:42] + ("…" if len(label) > 42 else "")
            lines.append(
                f'<li><a href="#{sid}" data-sec="{sid}">'
                f'<span class="nav-num">{num}</span><span class="nav-label">{short}</span></a></li>'
            )
    lines.append("</ul></aside>")
    return "\n".join(lines)


def add_sec_badges(html: str) -> str:
    def repl(m):
        sid, h2attrs, h2inner = m.group(1), m.group(2), m.group(3)
        if "sec-badge" in h2inner:
            return m.group(0)
        text = re.sub(r"<[^>]+>", "", h2inner).strip()
        num_m = re.match(r"^(\d+[a-z]?|★|核心)", text)
        badge = num_m.group(1) if num_m else sid.replace("sec-", "")[:8]
        inner = h2inner.strip()
        if inner.startswith("<span"):
            return m.group(0)
        # drop leading section number if duplicated in badge
        inner = re.sub(rf"^{re.escape(badge)}\s*\.?\s*", "", inner)
        inner = re.sub(r"^(\d+[a-z]?)\s*\.?\s*", "", inner, count=1)
        return f'<section id="{sid}">\n      <h2{h2attrs}><span class="sec-badge">{badge}</span> {inner}</h2>'
    return re.sub(
        r'<section id="([^"]+)">\s*<h2([^>]*)>(.*?)</h2>',
        repl,
        html,
        flags=re.DOTALL,
    )


def classify_tables(html: str) -> str:
    def table_repl(m):
        block = m.group(0)
        if "table-schema" in block or "table-cols" in block:
            return block
        if re.search(r"<th[^>]*>[^<]*(字段|列名|字段名|Layer 2 列)", block):
            return block.replace("<table>", '<table class="table-schema">', 1)
        if re.search(r"<th[^>]*>[^<]*(类别|字段名)</th>", block):
            return block.replace("<table>", '<table class="table-cols">', 1)
        return block
    return re.sub(r"<table>.*?</table>", table_repl, html, flags=re.DOTALL)


def fix_empty_sample_folds(html: str) -> str:
    """Repair folds where regex left only placeholder text."""
    return re.sub(
        r'<details class="sample-fold"><summary>([^<]+)</summary>'
        r'<div class="fold-body">JSON 形态</div></details>\s*'
        r'(<div class="json-block">.*?</div>)',
        r'<details class="sample-fold"><summary>\1</summary>'
        r'<div class="fold-body">\2</div></details>',
        html,
        flags=re.DOTALL,
    )


def wrap_h4_json_pairs(html: str) -> str:
    def repl(m):
        h4, jb = m.group(1), m.group(2)
        if "sample-fold" in m.group(0):
            return m.group(0)
        title = re.sub(r"<[^>]+>", "", h4).strip()
        return (
            f'<details class="sample-fold"><summary>{title}</summary>'
            f'<div class="fold-body">{jb}</div></details>'
        )

    return re.sub(
        r"(?<!</details>)\s*(<h4>[^<]+</h4>)\s*(<div class=\"json-block\">.*?</div>)",
        repl,
        html,
        flags=re.DOTALL,
    )


def strip_h2_duplicate_numbers(html: str) -> str:
    html = re.sub(
        r'(<span class="sec-badge">[^<]+</span>)\s*(?:\d+[a-z]?|★|核心)\s*\.\s*',
        r"\1 ",
        html,
    )
    return html


C13_JSON = r'''<details class="sample-fold"><summary>C.1.3 样例（raw JSON 形态）</summary><div class="fold-body"><div class="json-block">{
  "cik": "0000104169",
  "tickers": ["WMT"],
  "filing_date": "2010-06-04",
  "period_end": "2009-04-30",
  "fiscal_year": 2010,
  "fiscal_quarter": 1,
  "timeframe": "quarterly",
  "revenue": 94242000000.0,
  "net_income_loss_attributable_common_shareholders": 3022000000.0,
  "basic_earnings_per_share": 0.26
}</div></div></details>'''


def fix_c13_sample(html: str) -> str:
    broken = (
        '<details class="sample-fold"><summary>C.1.3 样例（raw JSON 形态）</summary>'
        '<div class="fold-body">JSON 形态</div></details>'
    )
    if broken in html:
        return html.replace(broken, C13_JSON, 1)
    return html


def wrap_loose_json_in_fields(html: str) -> str:
    m = re.search(r'(<section id="sec-fields">)(.*?)(</section>)', html, flags=re.DOTALL)
    if not m:
        return html

    def inner_repl(block: str) -> str:
        def wrap_pair(m2):
            if "sample-fold" in m2.group(0):
                return m2.group(0)
            prefix, jb = m2.group(1), m2.group(2)
            title = "JSON 样例"
            if "<h3>" in prefix or "<h4>" in prefix:
                ht = re.findall(r"<h[34]>([^<]+)</h[34]>", prefix)
                if ht:
                    title = ht[-1].strip()
            return (
                f'{prefix}<details class="sample-fold"><summary>{title}</summary>'
                f'<div class="fold-body">{jb}</div></details>'
            )

        return re.sub(
            r"((?:</table>|</p>)\s*)(<div class=\"json-block\">.*?</div>)",
            wrap_pair,
            block,
            flags=re.DOTALL,
        )

    inner = inner_repl(m.group(2))
    return html[: m.start()] + m.group(1) + inner + m.group(3) + html[m.end() :]


def split_blind_spots_2(html: str) -> str:
    marker = '<h2><span class="sec-badge">12b</span>'
    if '<section id="sec-blind-spots-2">' in html:
        return html
    idx = html.find(marker)
    if idx < 0:
        return html
    return (
        html[:idx]
        + "    </section>\n\n    <section id=\"sec-blind-spots-2\">\n      "
        + html[idx:].lstrip()
    )


def replace_sidebar(html: str, sidebar: str) -> str:
    if re.search(r'<aside class="sidebar" id="sidebar">', html):
        return re.sub(
            r'<aside class="sidebar" id="sidebar">.*?</aside>',
            sidebar,
            html,
            count=1,
            flags=re.DOTALL,
        )
    return html


def patch(html: str) -> str:
    html = re.sub(r"<style>.*?</style>", f"<style>{NEW_STYLE}</style>", html, flags=re.DOTALL)

    html = re.sub(
        r'<nav class="toc[^"]*">.*?</nav>\s*',
        "",
        html,
        count=1,
        flags=re.DOTALL,
    )

    items = extract_nav_items(html)
    sidebar = build_sidebar(items, has_head_samples='id="sec-head-samples"' in html)
    html = replace_sidebar(html, sidebar)

    if '<div class="app-layout">' not in html:
        html = html.replace("<body>", '<body>\n  <div class="app-layout">', 1)
        html = html.replace(
            '<div class="wrap">',
            sidebar + '\n  <main class="main-content">\n  <div class="wrap">',
            1,
        )
        if '<main class="main-content">' not in html:
            html = html.replace(
                "<body>\n  <div class=\"app-layout\">",
                "<body>\n  <div class=\"app-layout\">\n" + sidebar,
                1,
            )
            html = html.replace(
                '<div class="app-layout">\n\n    <header',
                '<div class="app-layout">\n' + sidebar + '\n  <main class="main-content">\n\n    <header',
                1,
            )
        html = html.replace("</body>", "  </main>\n  </div>\n</body>", 1)

    html = add_sec_badges(html)
    html = strip_h2_duplicate_numbers(html)
    html = split_blind_spots_2(html)
    html = classify_tables(html)
    html = fix_empty_sample_folds(html)
    html = fix_c13_sample(html)
    html = wrap_h4_json_pairs(html)
    html = wrap_loose_json_in_fields(html)

    if "sec-head-samples" not in html:
        html = html.replace(
            '<section id="sec-ok">',
            HEAD_SAMPLES + "\n    <section id=\"sec-ok\">",
            1,
        )
        # fix: head samples should be inside wrap - already before sec-ok

    scroll_js = """
<script>
(function(){
  const links = document.querySelectorAll('.sidebar-nav a[data-sec]');
  const secs = [...links].map(a => document.getElementById(a.dataset.sec)).filter(Boolean);
  const onScroll = () => {
    let cur = links[0];
    for (const s of secs) {
      if (s && s.getBoundingClientRect().top <= 120) cur = links[[...secs].indexOf(s)];
    }
    links.forEach(l => l.classList.toggle('active', l === cur));
  };
  window.addEventListener('scroll', onScroll, {passive:true});
  onScroll();
})();
</script>
"""
    if scroll_js.strip() not in html:
        html = html.replace("</body>", scroll_js + "\n</body>")

    for old in (
        "版本 v2.0 HTML 综合版",
        "版本 v2.3 HTML 综合版",
        "v2.0 综合版",
    ):
        html = html.replace(old, "版本 v3.0 HTML 导航美化版" if "版本" in old else "v3.0 导航美化版")

    return html


def main():
    html = HTML_PATH.read_text(encoding="utf-8")
    html = patch(html)
    HTML_PATH.write_text(html, encoding="utf-8")
    print("beautified, lines:", html.count("\n") + 1)


if __name__ == "__main__":
    main()
