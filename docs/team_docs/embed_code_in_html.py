#!/usr/bin/env python3
"""Embed DDL/scripts into HTML as <details> folds; remove external file references."""
import re
from pathlib import Path

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"

DDL = r"""-- Massive Layer 2 / qs_massive — paste from HTML §10c.11
CREATE DATABASE IF NOT EXISTS qs_massive;

CREATE TABLE IF NOT EXISTS qs_massive.dim_calendar
(
    trade_date Date,
    is_trading_day UInt8 DEFAULT 1,
    source LowCardinality(String) DEFAULT 'day_aggs_union'
)
ENGINE = MergeTree()
ORDER BY (trade_date);

CREATE TABLE IF NOT EXISTS qs_massive.dim_security_master
(
    ticker String,
    permanent_id Nullable(String),
    composite_figi Nullable(String),
    cik Nullable(String),
    type LowCardinality(String),
    market LowCardinality(String),
    locale LowCardinality(String),
    primary_exchange LowCardinality(String),
    active UInt8,
    list_date Nullable(Date),
    delisted_date Nullable(Date),
    updated_at DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (ticker);

CREATE TABLE IF NOT EXISTS qs_massive.dim_ticker_map
(
    ticker String,
    permanent_id String,
    valid_from Date,
    valid_to Nullable(Date),
    composite_figi Nullable(String),
    cik Nullable(String),
    map_source LowCardinality(String) DEFAULT 'all_tickers',
    batch_id String
)
ENGINE = MergeTree()
ORDER BY (permanent_id, valid_from, ticker);

CREATE TABLE IF NOT EXISTS qs_massive.dim_universe_daily
(
    trade_date Date,
    ticker String,
    permanent_id Nullable(String),
    in_universe_base UInt8,
    universe_mask UInt8 DEFAULT 1,
    has_bar UInt8 DEFAULT 1,
    price_source LowCardinality(String),
    batch_id String
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(trade_date)
ORDER BY (trade_date, ticker);

CREATE TABLE IF NOT EXISTS qs_massive.fact_bars_adjusted_daily
(
    trade_date Date,
    ticker String,
    permanent_id Nullable(String),
    align_time DateTime64(3, 'UTC'),
    adj_open Float64,
    adj_high Float64,
    adj_low Float64,
    adj_close Float64,
    adj_volume Float64,
    price_source LowCardinality(String),
    adj_method LowCardinality(String),
    batch_id String
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(trade_date)
ORDER BY (trade_date, ticker);

CREATE TABLE IF NOT EXISTS qs_massive.fact_returns_daily
(
    trade_date Date,
    ticker String,
    permanent_id Nullable(String),
    ret_price Float64,
    ret_total Nullable(Float64),
    is_delisting_day UInt8 DEFAULT 0,
    delisting_return Nullable(Float64),
    delisting_policy LowCardinality(String) DEFAULT '',
    reinvest_assumption LowCardinality(String) DEFAULT 'dividend_reinvest',
    batch_id String
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(trade_date)
ORDER BY (trade_date, ticker);

CREATE TABLE IF NOT EXISTS qs_massive.fact_shares_out_daily
(
    trade_date Date,
    ticker String,
    permanent_id Nullable(String),
    shares_out Float64,
    shares_source LowCardinality(String),
    pit_market_cap Nullable(Float64),
    batch_id String
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(trade_date)
ORDER BY (trade_date, ticker);

CREATE TABLE IF NOT EXISTS qs_massive.fact_delisting_events
(
    ticker String,
    permanent_id Nullable(String),
    last_trading_day Date,
    delisting_type LowCardinality(String),
    delisting_return Float64,
    delisting_policy LowCardinality(String),
    batch_id String
)
ENGINE = MergeTree()
ORDER BY (permanent_id, last_trading_day);

CREATE TABLE IF NOT EXISTS qs_massive.pit_fundamentals
(
    ticker String,
    permanent_id Nullable(String),
    knowledge_date Date,
    knowledge_ts DateTime64(3, 'UTC'),
    period_end Date,
    timeframe LowCardinality(String),
    pit_revenue Nullable(Float64),
    pit_net_income Nullable(Float64),
    pit_total_assets Nullable(Float64),
    pit_total_equity Nullable(Float64),
    pit_operating_revenue Nullable(Float64),
    pit_eps_basic Nullable(Float64),
    filing_date Date,
    batch_id String
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(knowledge_date)
ORDER BY (ticker, knowledge_date);

CREATE TABLE IF NOT EXISTS qs_massive.panel_daily
(
    trade_date Date,
    ticker String,
    permanent_id Nullable(String),
    align_time DateTime64(3, 'UTC'),
    batch_id String,
    price_source LowCardinality(String),
    adj_method LowCardinality(String),
    in_universe_base UInt8,
    universe_mask UInt8,
    adj_open Float64,
    adj_high Float64,
    adj_low Float64,
    adj_close Float64,
    adj_volume Float64,
    ret_price Float64,
    ret_total Nullable(Float64),
    shares_out Nullable(Float64),
    pit_market_cap Nullable(Float64),
    knowledge_ts Nullable(DateTime64(3, 'UTC')),
    pit_total_assets Nullable(Float64),
    pit_total_equity Nullable(Float64),
    pit_net_income Nullable(Float64),
    pit_operating_revenue Nullable(Float64),
    pit_short_interest Nullable(Int64),
    pit_short_volume_ratio Nullable(Float64),
    is_delisting_day UInt8 DEFAULT 0,
    delisting_return Nullable(Float64),
    news_count_1d Nullable(UInt32),
    news_sentiment_last Nullable(Float32),
    signal_date Nullable(Date)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(trade_date)
ORDER BY (trade_date, ticker);

ALTER TABLE qs_massive.panel_daily
    ADD PROJECTION IF NOT EXISTS p_ticker_timeline
    ( SELECT * ORDER BY (ticker, trade_date) );

CREATE TABLE IF NOT EXISTS qs_massive.meta_load_batch
(
    batch_id String,
    table_name String,
    partition_id String,
    row_count UInt64,
    parquet_path Nullable(String),
    started_at DateTime,
    finished_at Nullable(DateTime),
    status LowCardinality(String),
    error_msg Nullable(String)
)
ENGINE = MergeTree()
ORDER BY (started_at, table_name);

CREATE DATABASE IF NOT EXISTS qs_factor;

CREATE TABLE IF NOT EXISTS qs_factor.factor_exposures_daily
(
    trade_date Date,
    ticker String,
    factor_id String,
    exposure Float64,
    batch_id String
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(trade_date)
ORDER BY (factor_id, trade_date, ticker);
"""

BUILD_PANEL = r'''#!/usr/bin/env python3
"""Layer 2 panel builder — copy from HTML §10c.11, implement per PREPROC-002~007."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import polars as pl  # or pandas

MASSIVE_ROOT = Path("/home/yluel/share/projects/massive_parquet")
CLEANED = MASSIVE_ROOT / "cleaned_massive_data"
RAW = MASSIVE_ROOT / "raw_massive_data"
OUT = MASSIVE_ROOT / "materialized_panel"
STAGING = MASSIVE_ROOT / "_staging/build_panel"


def make_batch_id() -> str:
    from datetime import datetime
    return datetime.utcnow().strftime("%Y%m%d_01")


def write_hive(df: pl.DataFrame, table: str, year: int, month: int, batch_id: str) -> Path:
    dest = OUT / table / f"year={year}" / f"month={month:02d}"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"part-{batch_id}-000.parquet"
    df.write_parquet(path, compression="zstd")
    return path


def build_calendar(batch_id: str) -> pl.DataFrame:
    # union trade_date from cleaned day_aggs filenames
    raise NotImplementedError


def build_bars_adjusted(trade_dates: list[date], price_source: str, batch_id: str) -> pl.DataFrame:
    # SIP * splits OR REST daily — single price_source for whole run
    raise NotImplementedError


def asof_join_pit(panel: pl.DataFrame, pit: pl.DataFrame) -> pl.DataFrame:
    # knowledge_date <= trade_date, latest per (ticker, trade_date)
    raise NotImplementedError


def build_panel_for_month(year: int, month: int, price_source: str = "sip_adj") -> pl.DataFrame:
    batch_id = make_batch_id()
    staging = STAGING / batch_id
    staging.mkdir(parents=True, exist_ok=True)
    try:
        # Order: §10c.7
        bars = build_bars_adjusted([], price_source, batch_id)
        uni = pl.read_parquet(OUT / "dim_universe_daily" / f"year={year}" / f"month={month:02d}" / "*.parquet")
        ret = pl.read_parquet(OUT / "fact_returns_daily" / f"year={year}" / f"month={month:02d}" / "*.parquet")
        pit = pl.scan_parquet(OUT / "pit_fundamentals" / "**/*.parquet").collect()
        panel = bars.join(uni, on=["trade_date", "ticker"], how="left")
        panel = panel.join(ret, on=["trade_date", "ticker"], how="left")
        panel = asof_join_pit(panel, pit)
        panel = panel.with_columns(
            pl.lit(price_source).alias("price_source"),
            pl.lit(batch_id).alias("batch_id"),
        )
        assert panel.select(["trade_date", "ticker"]).is_duplicated().sum() == 0
        write_hive(panel, "layer2_factor_ready_panel", year, month, batch_id)
        return panel
    finally:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--price-source", default="sip_adj")
    args = p.parse_args()
    build_panel_for_month(args.year, args.month, args.price_source)


if __name__ == "__main__":
    main()
'''

LOAD_CH = r'''#!/usr/bin/env python3
"""CH-002: load materialized_panel Parquet partitions into qs_massive (copy from HTML §10c.11)."""
import os
from pathlib import Path

import clickhouse_connect

ROOT = Path("/home/yluel/share/projects/massive_parquet/materialized_panel/layer2_factor_ready_panel")


def load_month(year: int, month: int) -> None:
    client = clickhouse_connect.get_client(
        host=os.environ["CLICKHOUSE_HOST"],
        port=int(os.environ.get("CLICKHOUSE_PORT", "8123")),
        username=os.environ["CLICKHOUSE_USER"],
        password=os.environ["CLICKHOUSE_PASSWORD"],
        database=os.environ.get("CLICKHOUSE_DATABASE", "qs_massive"),
    )
    part = f"{year}{month:02d}"
    glob = str(ROOT / f"year={year}" / f"month={month:02d}" / "*.parquet")
    # Prefer server-side file() if CH can read NFS; else insert_df in chunks
    client.command(f"ALTER TABLE panel_daily DROP PARTITION '{part}'")
    # client.insert_df("panel_daily", df)  # implement row count → meta_load_batch
    raise NotImplementedError(f"Load {glob}")


if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--year", type=int, required=True)
    a.add_argument("--month", type=int, required=True)
    load_month(a.year, a.month)
'''

CH_PANEL = r'''"""CH-004: read-only panel API for factor_engine (copy from HTML §10c.11)."""
from __future__ import annotations

import os
from typing import Optional

import clickhouse_connect
import polars as pl


class ClickHousePanel:
    def __init__(self, database: str = "qs_massive"):
        self._client = clickhouse_connect.get_client(
            host=os.environ["CLICKHOUSE_HOST"],
            port=int(os.environ.get("CLICKHOUSE_PORT", "8123")),
            username=os.environ["CLICKHOUSE_USER"],
            password=os.environ["CLICKHOUSE_PASSWORD"],
            database=database,
        )

    def load_panel(
        self,
        start: str,
        end: str,
        tickers: Optional[list[str]] = None,
        in_universe_only: bool = True,
    ) -> pl.DataFrame:
        where = [f"trade_date BETWEEN '{start}' AND '{end}'"]
        if in_universe_only:
            where.append("in_universe_base = 1")
        if tickers:
            quoted = ",".join(f"'{t}'" for t in tickers)
            where.append(f"ticker IN ({quoted})")
        sql = f"SELECT * FROM panel_daily WHERE {' AND '.join(where)}"
        return pl.from_arrow(self._client.query_arrow(sql))

    def load_parquet_fallback(self, path: str) -> pl.DataFrame:
        return pl.scan_parquet(path).collect()
'''

TAPE_YAML = r"""# TASK-DC-005 — Consolidated Tape Filtering Protocol (draft)
version: "2026-06-03"
# Map condition id from market_operations/condition_codes — do NOT hardcode letters only.

hl_close_whitelist_ids: []   # fill from dim table: regular sale
hl_exclude_ids: []           # late, avg price, derived, etc.

rules:
  include_in_hl_close:
    description: Regular consolidated prints only
  volume_alt_bucket:
    description: Trades excluded from H/L still sum into vol_alt
  null_conditions:
    treat_as: whitelist_default
"""

CODE_SECTION = """
      <h3 id="sec-code-assets">10c.11 可复制代码（折叠，交付物均在此）</h3>
      <p class="section-desc">
        <strong>不另建仓库脚本文件。</strong>落地组从本节复制到本地路径即可；HTML 为唯一工程交付载体（另附 MD 姊妹文档）。
      </p>

      <details class="code-fold">
        <summary>CH-001 · <code>massive_ddl.sql</code> 全量建表（qs_massive + qs_factor + Projection）</summary>
        <p>执行：<code>clickhouse-client --multiquery</code> 粘贴下方 SQL。</p>
        <pre><code>__DDL__</code></pre>
      </details>

      <details class="code-fold">
        <summary>FEAT-001 / ENG-010 · <code>build_panel_daily.py</code> 骨架</summary>
        <pre><code>__BUILD__</code></pre>
      </details>

      <details class="code-fold">
        <summary>CH-002 · <code>load_panel_to_clickhouse.py</code> 按月导入</summary>
        <pre><code>__LOAD__</code></pre>
      </details>

      <details class="code-fold">
        <summary>CH-004 · <code>clickhouse_panel.py</code> 只读 API</summary>
        <pre><code>__PANEL__</code></pre>
      </details>

      <details class="code-fold">
        <summary>TASK-DC-005 · <code>tape_condition_filter.yaml</code> 草案</summary>
        <pre><code>__YAML__</code></pre>
      </details>

      <details class="code-fold">
        <summary>CH-002 · clickhouse-client 按月导入（Shell）</summary>
        <pre><code>clickhouse-client --host "$CLICKHOUSE_HOST" --user "$CLICKHOUSE_USER" --password "$CLICKHOUSE_PASSWORD" --query "
INSERT INTO qs_massive.panel_daily
SELECT * FROM file(
  '/home/yluel/share/projects/massive_parquet/materialized_panel/layer2_factor_ready_panel/year=2024/month=06/*.parquet',
  Parquet
)"
# 修复单月: ALTER TABLE qs_massive.panel_daily DROP PARTITION '202406';</code></pre>
      </details>
"""


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def patch(html: str) -> str:
    block = (
        CODE_SECTION.replace("__DDL__", esc(DDL.strip()))
        .replace("__BUILD__", esc(BUILD_PANEL.strip()))
        .replace("__LOAD__", esc(LOAD_CH.strip()))
        .replace("__PANEL__", esc(CH_PANEL.strip()))
        .replace("__YAML__", esc(TAPE_YAML.strip()))
    )

    # CSS for folds
    if ".code-fold" not in html:
        html = html.replace(
            "    .badge-todo { background: var(--purple); color: var(--purple-deep); }",
            """    .badge-todo { background: var(--purple); color: var(--purple-deep); }

    details.code-fold {
      border: 1px solid var(--border);
      border-radius: 10px;
      margin: 12px 0;
      background: #f8fafc;
    }
    details.code-fold > summary {
      cursor: pointer;
      padding: 12px 16px;
      font-weight: 600;
      font-size: 14px;
      list-style: disclosure-closed;
    }
    details.code-fold[open] > summary { border-bottom: 1px solid var(--border); }
    details.code-fold pre {
      margin: 0;
      padding: 14px 16px;
      overflow-x: auto;
      font-size: 12px;
      line-height: 1.45;
      background: #0f172a;
      color: #e2e8f0;
      border-radius: 0 0 10px 10px;
    }
    details.code-fold pre code { white-space: pre; }""",
        )

    # Refresh 10c.11 code folds (replace existing block)
    fold_pat = r'<h3 id="sec-code-assets">10c\.11.*?</details>\s*\n\s*</details>\s*\n\s*</details>\s*\n\s*</details>\s*\n\s*</details>\s*\n\s*</details>'
    mfold = re.search(
        r'<h3 id="sec-code-assets">10c\.11.*?(?=\n\s*<div class="box-info">\s*\n\s*<strong>相关章节)',
        html,
        flags=re.DOTALL,
    )
    if mfold:
        html = html[: mfold.start()] + block.strip() + "\n\n      " + html[mfold.end() :]

    # Replace 10c.10 + inject 10c.11 (first time)
    pat = r"<h3>10c\.10 工程脚本交付清单.*?</table>\s*\n\s*<div class=\"box-info\">"
    if re.search(pat, html, flags=re.DOTALL):
        html = re.sub(
            pat,
            """<h3>10c.10 工程交付物索引（代码见下方折叠）</h3>
      <table>
        <thead><tr><th>ID</th><th>折叠块</th><th>作用</th></tr></thead>
        <tbody>
          <tr><td>CH-001</td><td><a href="#sec-code-assets">massive_ddl.sql</a></td><td>建库建表 + Projection</td></tr>
          <tr><td>ENG-010 / FEAT-001</td><td><a href="#sec-code-assets">build_panel_daily.py</a></td><td>Layer 2 物化 1–9 步</td></tr>
          <tr><td>CH-002</td><td><a href="#sec-code-assets">load_panel + Shell</a></td><td>Parquet → CH</td></tr>
          <tr><td>CH-004</td><td><a href="#sec-code-assets">clickhouse_panel.py</a></td><td>因子引擎只读</td></tr>
          <tr><td>TASK-DC-005</td><td><a href="#sec-code-assets">tape_condition_filter.yaml</a></td><td>tick 条件码协议</td></tr>
        </tbody>
      </table>

"""
            + block
            + "\n      <div class=\"box-info\">",
            html,
            count=1,
            flags=re.DOTALL,
        )
    elif "sec-code-assets" not in html:
        html = html.replace(
            '        <a href="#sec-blind-spots">§12a 补盲点</a>\n      </div>\n    </section>\n\n<!-- 11 ClickHouse -->',
            '        <a href="#sec-blind-spots">§12a 补盲点</a>\n      </div>\n' + block + "\n    </section>\n\n<!-- 11 ClickHouse -->",
        )

    # Global reference fixes
    reps = [
        ("<code>deploy/clickhouse/massive_ddl.sql</code>", '<a href="#sec-code-assets">§10c.11 折叠 · massive_ddl.sql</a>'),
        ("<code>quantsociety_backend_project/deploy/clickhouse/massive_ddl.sql</code>", '<a href="#sec-code-assets">§10c.11 折叠 · DDL</a>'),
        ("生产以仓库文件为准：<code>deploy/clickhouse/massive_ddl.sql</code>。", 'DDL 见 <a href="#sec-code-assets">§10c.11 折叠</a>。'),
        ("<td><code>deploy/clickhouse/massive_ddl.sql</code></td>", '<td><a href="#sec-code-assets">§10c.11 折叠 DDL</a></td>'),
        ("<td>§12a；<code>massive_ddl.sql</code> 内嵌 PROJECTION</td>", '<td>§10c.11 折叠 DDL + §12a</td>'),
        (
            "可执行 DDL：\n        <code>deploy/clickhouse/massive_ddl.sql</code>。",
            '可执行 DDL：见 <a href="#sec-code-assets"><strong>§10c.11 折叠代码</strong></a>。',
        ),
        ("DDL 全文：<code>quantsociety_backend_project/deploy/clickhouse/massive_ddl.sql</code>（CH-001 交付物）。", 'DDL 全文：<a href="#sec-code-assets">§10c.11</a> 折叠块（CH-001）。'),
    ]
    for a, b in reps:
        html = html.replace(a, b)

    # Dedupe duplicate CH pointer paragraph
    html = re.sub(
        r"(<p>与 <a href=\"#sec-engineering\">§10c</a> 字段字典一致；DDL 见 <a href=\"#sec-code-assets\">§10c\.11 折叠</a>。</p>\s*)+",
        r"\1",
        html,
        count=1,
    )
    dup = '<p>与 <a href="#sec-engineering">§10c</a> 字段字典一致；DDL 见 <a href="#sec-code-assets">§10c.11 折叠</a>。</p>\n      <p>与 <a href="#sec-engineering">§10c</a> 字段字典一致；DDL 见 <a href="#sec-code-assets">§10c.11 折叠</a>。</p>'
    html = html.replace(dup, '<p>与 <a href="#sec-engineering">§10c</a> 字段字典一致；DDL 见 <a href="#sec-code-assets">§10c.11 折叠</a>。</p>')

    # Nav link to code
    if "sec-code-assets" not in html.split("</nav>")[0]:
        html = html.replace(
            '<li><a href="#sec-engineering"><strong>★ 10c 工程落地手册</strong></a></li>',
            '<li><a href="#sec-engineering"><strong>★ 10c 工程落地手册</strong></a></li>\n        <li><a href="#sec-code-assets">10c.11 折叠代码（DDL/脚本）</a></li>',
        )

    # Header delivery note
    if "交付载体" not in html[:4000]:
        html = html.replace(
            "<p class=\"purpose\">",
            "<p class=\"purpose\"><strong>交付：</strong>本 HTML（含 §10c.11 折叠代码）+ 姊妹 MD；<strong>不</strong>另附独立脚本仓库文件。 ",
            1,
        )

    return html


def main():
    html = HTML.read_text(encoding="utf-8")
    html = patch(html)
    HTML.write_text(html, encoding="utf-8")
    print("embedded code folds, lines:", html.count("\n") + 1)


if __name__ == "__main__":
    main()
