#!/usr/bin/env python3
"""Dedupe §11.3 DDL; add 10c.12 folds + 16.5 Layer2 DoD + cheatsheet paths."""
import re
from pathlib import Path

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"

NEW_11_3 = r'''
      <h3>11.3 表结构（与 §10c 对齐，DDL 不重复粘贴）</h3>
      <div class="box-info">
        <strong>唯一 DDL 来源：</strong><a href="#sec-code-assets">§10c.11 折叠 · massive_ddl.sql</a>（含 <code>permanent_id</code>、<code>fact_shares_out_daily</code>、Projection CH-006）。
        本节仅保留<strong>表清单 + 分区键</strong>，避免与折叠块双份维护。
      </div>
      <table class="stage-table">
        <thead>
          <tr><th>CH 表</th><th>Parquet 目录</th><th>PARTITION BY</th><th>ORDER BY</th><th>行粒度</th></tr>
        </thead>
        <tbody>
          <tr><td><code>dim_calendar</code></td><td><code>dim_calendar/</code></td><td>—</td><td>(trade_date)</td><td>1 交易日</td></tr>
          <tr><td><code>dim_security_master</code></td><td><code>dim_security_master/</code></td><td>—</td><td>(ticker)</td><td>1 ticker 快照</td></tr>
          <tr><td><code>dim_ticker_map</code></td><td><code>dim_ticker_map/</code></td><td>—</td><td>(permanent_id, valid_from, ticker)</td><td>代码有效期</td></tr>
          <tr><td><code>dim_universe_daily</code></td><td><code>dim_universe_daily/year=…/month=…</code></td><td>toYYYYMM(trade_date)</td><td>(trade_date, ticker)</td><td>日×ticker 稀疏</td></tr>
          <tr><td><code>fact_bars_adjusted_daily</code></td><td><code>fact_bars_adjusted_daily/…</code></td><td>toYYYYMM(trade_date)</td><td>(trade_date, ticker)</td><td>复权 OHLCV</td></tr>
          <tr><td><code>fact_returns_daily</code></td><td><code>fact_returns_daily/…</code></td><td>toYYYYMM(trade_date)</td><td>(trade_date, ticker)</td><td>日收益</td></tr>
          <tr><td><code>fact_shares_out_daily</code></td><td><code>fact_shares_out_daily/…</code></td><td>toYYYYMM(trade_date)</td><td>(trade_date, ticker)</td><td>动态股本</td></tr>
          <tr><td><code>fact_delisting_events</code></td><td><code>fact_delisting_events/</code></td><td>—</td><td>(permanent_id, last_trading_day)</td><td>退市事件</td></tr>
          <tr><td><code>pit_fundamentals</code></td><td><code>pit_fundamentals/…</code></td><td>toYYYYMM(knowledge_date)</td><td>(ticker, knowledge_date)</td><td>PiT 长表</td></tr>
          <tr><td><code>panel_daily</code></td><td><code>layer2_factor_ready_panel/…</code></td><td>toYYYYMM(trade_date)</td><td>(trade_date, ticker)</td><td><strong>宽表</strong></td></tr>
          <tr><td><code>meta_load_batch</code></td><td>—（仅 CH）</td><td>—</td><td>(started_at, table_name)</td><td>导入审计</td></tr>
        </tbody>
      </table>
      <p><strong>Track B：</strong>不另建 11TB 副表；使用 <code>panel_daily</code> 的 Projection <code>p_ticker_timeline</code>（见 §10c.11 DDL 折叠）。</p>
      <div class="box-warn">
        <strong>P0 前：</strong>勿导入 2024 截断财报分区；或 <code>panel_daily</code> 上 2024 的 <code>pit_*</code> 强制 NULL。训练过滤见 <a href="#sec-clickhouse">§11.5.5</a>。
      </div>
'''

SUPPLEMENT_10C12 = r'''
      <h3 id="sec-code-extra">10c.12 补充折叠（运维 · 校验 · SQL 模板）</h3>
      <p class="section-desc">P0 门禁与 Layer 2 验收用；与 §16 验收标准配套。</p>

      <details class="code-fold">
        <summary>Stage 0 · REST 截断扫描（<code>scan_truncation.py</code> 单行版见 §16.4）</summary>
        <pre><code>#!/usr/bin/env python3
"""Exit 1 if any raw parquet has exactly 2000 or 10000 rows (pagination cap)."""
import sys
from pathlib import Path
import pyarrow.parquet as pq

RAW = Path("/home/yluel/share/projects/massive_parquet/raw_massive_data")
# Whitelist snapshot sources if team accepts 2k cap permanently:
WHITELIST_SUFFIXES = ()  # e.g. ("all_tickers_all.parquet",)

bad = []
for f in sorted(RAW.rglob("*.parquet")):
    if f.name.endswith(WHITELIST_SUFFIXES):
        continue
    n = pq.read_metadata(f).num_rows
    if n in (2000, 10000):
        bad.append((n, str(f.relative_to(RAW))))
print("SUSPICIOUS:", len(bad))
for n, p in bad[:50]:
    print(n, p)
if len(bad) > 50:
    print("...")
sys.exit(1 if bad else 0)</code></pre>
      </details>

      <details class="code-fold">
        <summary>物化后 · 单月分区校验（<code>validate_panel_month.py</code>）</summary>
        <pre><code>#!/usr/bin/env python3
"""PREPROC-010: validate one hive month partition before CH load."""
import sys
from pathlib import Path
import polars as pl

ROOT = Path("/home/yluel/share/projects/massive_parquet/materialized_panel/layer2_factor_ready_panel")

def main(year: int, month: int):
    glob = str(ROOT / f"year={year}" / f"month={month:02d}" / "*.parquet")
    df = pl.scan_parquet(glob).collect()
    dup = df.select(["trade_date", "ticker"]).is_duplicated().sum()
    if dup:
        raise SystemExit(f"duplicate keys: {dup}")
    if df["adj_close"].null_count() / max(len(df), 1) > 0.05:
        raise SystemExit("adj_close null rate &gt; 5%")
    if df["price_source"].n_unique() != 1:
        raise SystemExit("mixed price_source in partition")
    print("PASS", year, month, "rows", len(df))

if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))</code></pre>
      </details>

      <details class="code-fold">
        <summary>CH · PiT ASOF 模板（宽表未预 join 时，§11.5.3 展开版）</summary>
        <pre><code>SELECT p.trade_date, p.ticker, p.adj_close, f.pit_net_income, f.knowledge_date
FROM qs_massive.panel_daily AS p
ASOF LEFT JOIN qs_massive.pit_fundamentals AS f
    ON p.ticker = f.ticker AND p.trade_date &gt;= f.knowledge_date
WHERE p.trade_date = '2023-06-01'
  AND p.in_universe_base = 1
ORDER BY p.ticker;</code></pre>
      </details>

      <details class="code-fold">
        <summary>P0 · 删除坏分区 <code>.ok</code> 并触发重下（Shell 模板）</summary>
        <pre><code># TASK-DATA-001 — 示例：2024 资产负债表坏分区
BAD="/home/yluel/share/projects/massive_parquet/raw_massive_data/fundamentals/balance_sheet"
find "$BAD" -name '*.ok' -path '*2024*' -print -delete
# 然后按 §17 I.2 重下 REST，勿带 --max-pages</code></pre>
      </details>

      <h3>10c.13 落地组一日通读顺序</h3>
      <ol class="flow-steps">
        <li><a href="#sec-p0">§2 P0</a> → 跑 <a href="#sec-code-extra">截断扫描折叠</a>，未 PASS 禁止 Layer 2</li>
        <li><a href="#sec-before-factor">★入模前</a> → 理解 Stage 0–10 边界</li>
        <li><a href="#sec-engineering">§10c</a> → 目录/表/列/分区</li>
        <li>展开 <a href="#sec-code-assets">§10c.11</a> 复制 DDL 建 CH</li>
        <li>实现物化 → <a href="#sec-code-extra">校验折叠</a> → <a href="#sec-clickhouse">§11.4–11.5</a> 导入与查询</li>
        <li><a href="#sec-accept">§16.5</a> Layer 2 DoD 打勾</li>
      </ol>
'''

SEC_16_5 = r'''
      <h3>16.5 Layer 2 / ClickHouse 验收（物化完成后）</h3>
      <ul class="checklist">
        <li><code>materialized_panel/layer2_factor_ready_panel/year=2023/month=12/</code> 存在且行数与 day_aggs 锚点同量级</li>
        <li>单月分区 <code>(trade_date, ticker)</code> 重复键 = 0（见 <a href="#sec-code-extra">validate_panel_month 折叠</a>）</li>
        <li>全分区单一 <code>price_source</code>；NVDA 拆股日前后 <code>adj_close</code> 无假断崖</li>
        <li>CH <code>panel_daily</code> 同分区行数与 Parquet ±0.1%</li>
        <li>抽查 PiT：<code>knowledge_date &lt;= trade_date</code>；2024 <code>pit_*</code> 未误用截断财报（P0 前为 NULL 或禁分区）</li>
        <li>factor_engine 从 CH 或 Parquet 宽表跑通 1 条表达式</li>
        <li><code>meta_load_batch</code> 有成功记录；<code>_staging/</code> 无残留 batch 目录</li>
      </ul>
'''


def patch(html: str) -> str:
    # Replace §11.3 duplicate DDL block
    pat = r"<h3>11\.3 表结构 DDL.*?</div>\s*\n\s*</div>\s*\n\s*<h3>11\.4"
    if not re.search(pat, html, flags=re.DOTALL):
        pat = r"<h3>11\.3 表结构.*?</pre>\s*\n\s*<h3>11\.4"
    m = re.search(pat, html, flags=re.DOTALL)
    if m:
        html = html[: m.start()] + NEW_11_3.strip() + "\n\n      <h3>11.4" + html[m.end() :]
    else:
        print("warn: 11.3 block not found")

    # Also try older pattern ending at 11.3.6
    if "11.3.1 维表" in html:
        pat2 = r"<h3>11\.3 表结构[^<]*</h3>.*?<h3>11\.4 写入"
        html2 = re.sub(pat2, NEW_11_3.strip() + "\n\n      <h3>11.4 写入", html, count=1, flags=re.DOTALL)
        if html2 != html:
            html = html2
            print("replaced 11.3 via pat2")

    # Insert 10c.12 before sec-engineering box-info closing OR after last code-fold in sec-code-assets
    if "sec-code-extra" not in html:
        anchor = '<div class="box-info">\n        <strong>相关章节：</strong>\n        <a href="#sec-pipeline">§10b 管道</a>'
        if anchor in html:
            html = html.replace(anchor, SUPPLEMENT_10C12 + "\n\n      " + anchor)
        else:
            print("warn: 10c.12 anchor not found")

    # 16.5
    if "16.5 Layer 2" not in html:
        html = html.replace(
            "<h3>16.4 一键截断检测脚本</h3>",
            SEC_16_5 + "\n      <h3>16.4 一键截断检测脚本</h3>",
        )

    # Nav
    nav = html.split("</nav>")[0]
    if "sec-code-extra" not in nav:
        html = html.replace(
            '<li><a href="#sec-code-assets">10c.11 折叠代码（DDL/脚本）</a></li>',
            '<li><a href="#sec-code-assets">10c.11 折叠代码（DDL/脚本）</a></li>\n        <li><a href="#sec-code-extra">10c.12 运维/校验折叠</a></li>',
        )

    # Cheatsheet materialized_panel
    if "materialized_panel" not in html.split("I.1 路径")[1].split("I.2")[0] if "I.1 路径" in html else "":
        html = html.replace(
            "<tr><td><strong>主库 cleaned</strong></td><td><code>/home/yluel/share/projects/massive_parquet/cleaned_massive_data/</code></td></tr>",
            "<tr><td><strong>主库 cleaned</strong></td><td><code>/home/yluel/share/projects/massive_parquet/cleaned_massive_data/</code></td></tr>\n"
            "<tr><td><strong>Layer 2 物化（真相源）</strong></td><td><code>/home/yluel/share/projects/massive_parquet/materialized_panel/</code></td></tr>\n"
            "<tr><td><strong>宽表 Parquet</strong></td><td><code>.../layer2_factor_ready_panel/year=YYYY/month=MM/</code></td></tr>",
        )

    # Fix stale refs
    html = html.replace(
        "<code>data_access/clickhouse_panel.py</code>（待建）",
        '<a href="#sec-code-assets">§10c.11 clickhouse_panel.py 折叠</a>',
    )
    html = html.replace(
        "<td><code>build_panel_daily.py</code></td><td>Stage 7 物化</td><td>❌ 待建 FEAT-001</td>",
        '<td><code>build_panel_daily.py</code></td><td>Stage 7 物化</td><td>见 <a href="#sec-code-assets">§10c.11 折叠</a></td>',
    )

    # purpose trailing
    html = html.replace(
        "从 HTML 复制即可。</p> \n",
        "从 HTML 复制即可。</p>\n",
    )

    # Duplicate paragraph before 11.3 if still there
    html = re.sub(
        r"<p>与 <a href=\"#sec-engineering\">§10c</a> 字段字典一致；DDL 见 <a href=\"#sec-code-assets\">§10c\.11 折叠</a>[^<]*</p>\s*\n\s*<h3>11\.3",
        "<h3>11.3",
        html,
        count=1,
    )

    return html


def main():
    html = HTML.read_text(encoding="utf-8")
    html = patch(html)
    HTML.write_text(html, encoding="utf-8")
    print("done, lines:", html.count("\n") + 1, "11.3.1 gone:", "11.3.1 维表" not in html)


if __name__ == "__main__":
    main()
