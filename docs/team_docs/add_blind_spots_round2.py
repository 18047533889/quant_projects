#!/usr/bin/env python3
"""Round-2 blind spots: ENG-011, PREPROC-014~016, CH-007 into HTML folds."""
import re
from pathlib import Path

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"

ROUND2 = r'''
      <h2 id="sec-blind-spots-2">12b. 二轮补盲点（分钟 · 代码 · 股本 · 新闻 · CH 分钟表）</h2>
      <p class="section-desc">
        国际 Feature Store / 资产定价实证视角下的<strong>入模前最后一公里</strong>补充。
        五条均建议在物化落盘 / 导入 CH 前封板；代码见 <a href="#sec-code-round2">§10c.14 折叠</a>。
      </p>

      <h3>盲点六：SIFMA 提前收盘与分钟骨架（ENG-011）</h3>
      <p><strong>采纳。</strong> 机械 390 根 RTH 骨架在感恩节翌日、圣诞前夕、独立日前夕等<strong>13:00 提前收盘</strong>日，会在 13:00–16:00 误 ffill 价、vol=0，污染日内波动率/成交量衰减因子。</p>
      <table>
        <thead><tr><th>项</th><th>规范</th></tr></thead>
        <tbody>
          <tr><td><code>dim_calendar</code></td><td>增 <code>market_close_time_et</code>（默认 <code>16:00</code>，提前日 <code>13:00</code>）、<code>expected_rth_bars</code>（390 / 210）</td></tr>
          <tr><td>分钟 R 轨</td><td>骨架长度 = <code>expected_rth_bars</code>；<strong>禁止</strong>对收盘后时段 ffill / 填 0</td></tr>
          <tr><td>数据源</td><td>SIFMA 日历 + <code>market_holidays</code>；提前收盘日可维护静态 YAML 直至 API 完备</td></tr>
        </tbody>
      </table>
      <p><strong>勘误：</strong> 文档已强调 minute sum ≠ day volume；提前收盘是<strong>另一独立陷阱</strong>，与 ETH 过滤并列处理。</p>
      <p><strong>任务：</strong> <code>ENG-011</code>（P1）— 数据工程组。</p>

      <h3>盲点七：跨源代码后缀归一（PREPROC-014）</h3>
      <p><strong>采纳（强化 Layer 1 T 轨）。</strong> <code>BRK/B</code>、<code>BRK.B</code>、<code>BRK-B</code> 若仅大写去空格，多源 join 仍断裂。</p>
      <ul>
        <li><strong>规范：</strong> CMS/CQS 后缀剥离：<code>/ . -</code> 及空格 → 紧凑码（如 <code>BRKB</code>），写入 <code>ticker_normalized</code></li>
        <li><strong>禁止：</strong> 将 <code>GOOG</code> 与 <code>GOOGL</code> 强行合并（不同股份类别，走 <code>permanent_id</code> / share-class 表）</li>
        <li><strong>落地：</strong> Layer 1.5 <code>dim_ticker_alias</code> 或扩展 <code>security_master</code>；Exact join 用 <code>ticker_normalized</code></li>
      </ul>
      <p><strong>任务：</strong> <code>PREPROC-014</code>（P1）— 基础架构组。</p>

      <h3>盲点八：有机 vs 非有机股本双轨（PREPROC-015）</h3>
      <p><strong>采纳（扩展 PREPROC-013）。</strong> 拆股 = 离散非有机事件 → 历史股本<strong>逆向除法</strong>与复权价同步；回购/摊薄在季报 <code>filing_date</code> 披露 = 阶梯函数。</p>
      <table>
        <thead><tr><th>事件</th><th>股本时序</th><th>禁止</th></tr></thead>
        <tbody>
          <tr><td><code>splits</code></td><td>执行日前历史股本同比例调整</td><td>—</td></tr>
          <tr><td>季报 <code>basic_shares_outstanding</code></td><td><code>filing_date</code> 起阶跃至新值；此前区间保持上一披露值</td><td><strong>禁止</strong>向历史线性插值/平滑（前视）</td></tr>
          <tr><td>日频供应商股本</td><td>若可用，优先作 <code>shares_out</code>；仍须 PiT（≤ trade_date）</td><td>—</td></tr>
        </tbody>
      </table>
      <p>市值因子（P/E、B/M、Size）须在元数据声明：<code>shares_step_at_filing=true</code>。</p>
      <p><strong>任务：</strong> <code>PREPROC-015</code>（P1）— 数据工程组。</p>

      <h3>盲点九：新闻重发去重（PREPROC-016）</h3>
      <p><strong>采纳（强化 Stage 9）。</strong> 同一事件多次 <code>published_utc</code> 会放大 <code>news_count_1d</code>；<code>signal_trade_date</code>  alone 不够。</p>
      <ul>
        <li>对 <code>title</code>（或摘要）做 SimHash / 3-gram Jaccard；24h（可按 ticker）窗口相似度 &gt; 0.85 → <strong>重发</strong></li>
        <li>情感列可保留最新；<code>news_count_*</code> <strong>不重复计数</strong></li>
        <li>P0：<code>news</code> 仍 2k 快照，规则先文档化，全量重下后启用</li>
      </ul>
      <p><strong>任务：</strong> <code>PREPROC-016</code>（P2）— 量化研究组。</p>

      <h3>盲点十：分钟表 CH 分区（CH-007）</h3>
      <p><strong>采纳，与日频分离。</strong> <code>panel_daily</code> 的 CH-006 Projection <strong>不适用于</strong> ~95 亿行分钟表；分钟横截面需专属 DDL。</p>
      <ul>
        <li>表：<code>fact_bars_adjusted_minute</code>（物化后入库，raw tick 仍不进 CH）</li>
        <li><code>PARTITION BY (toYYYYMM(timestamp), toMonday(timestamp))</code></li>
        <li><code>ORDER BY (trade_date, timestamp, ticker)</code> — 利于「某日某分钟全市场」截面扫描</li>
        <li>列：<code>is_rth</code>、<code>is_early_close</code> 与 ENG-011 联动</li>
      </ul>
      <p><strong>任务：</strong> <code>CH-007</code>（P2）— 平台 SRE；DDL 见 <a href="#sec-code-round2">§10c.14</a>。</p>

      <h3>二轮任务与阶段</h3>
      <table>
        <thead><tr><th>ID</th><th>阶段</th><th>依赖</th></tr></thead>
        <tbody>
          <tr><td>ENG-011</td><td>dim_calendar + Stage 8 分钟</td><td>market_holidays / SIFMA 表</td></tr>
          <tr><td>PREPROC-014</td><td>Layer 1.5 清洗入口</td><td>PREPROC-001</td></tr>
          <tr><td>PREPROC-015</td><td>fact_shares_out + panel</td><td>PREPROC-013、splits</td></tr>
          <tr><td>PREPROC-016</td><td>news → panel 列</td><td>PREPROC-009、P0 news</td></tr>
          <tr><td>CH-007</td><td>分钟 CH（独立于 panel_daily）</td><td>PREPROC-008、ENG-011</td></tr>
        </tbody>
      </table>
      <div class="box-info">
        五条已追加 <a href="#sec-tasks">§14 任务总表</a>；与 <a href="#sec-blind-spots">§12a</a> 正交，一并构成入模前封板清单。
      </div>
'''

CODE_R2 = r'''
      <h3 id="sec-code-round2">10c.14 二轮补盲点 · 折叠代码</h3>

      <details class="code-fold">
        <summary>ENG-011 · <code>dim_calendar</code> 提前收盘列 + SIFMA 种子（YAML）</summary>
        <pre><code># early_close_days.yaml — 按年维护，合并 market_holidays 后写入 dim_calendar
early_close_half_days:  # market_close_time_et: "13:00", expected_rth_bars: 210
  - { date: "2024-11-29", note: "Black Friday" }
  - { date: "2024-12-24", note: "Christmas Eve" }
  - { date: "2024-07-03", note: "Independence Day Eve" }
# 常规日: market_close_time_et: "16:00", expected_rth_bars: 390

# dim_calendar 追加列（合并进 §10c.11 DDL）:
#   market_close_time_et  LowCardinality(String)  -- "16:00" | "13:00"
#   expected_rth_bars     UInt16                   -- 390 | 210
#   is_early_close        UInt8</code></pre>
      </details>

      <details class="code-fold">
        <summary>PREPROC-014 · <code>ticker_normalize.py</code>（后缀剥离，不合并 GOOG/GOOGL）</summary>
        <pre><code>import re

_SUFFIX_RE = re.compile(r"[\s./\-]+")

def normalize_ticker(raw: str) -> str:
    if not raw or raw.upper() in ("NA", "NAN", "NONE"):
        return ""
    t = raw.upper().strip()
    t = _SUFFIX_RE.sub("", t)  # BRK.B / BRK/B / BRK-B -&gt; BRKB
    return t

# Join 键: ticker_normalized；展示/CH 可保留 vendor_ticker
# share-class 白名单禁止合并: {("GOOG","GOOGL"), ("BRK.A","BRKB"), ...} 走 permanent_id</code></pre>
      </details>

      <details class="code-fold">
        <summary>PREPROC-015 · 股本双轨 asof（阶梯，禁止 filing 向前插值）</summary>
        <pre><code># shares_out(trade_date) = asof_backward(filing_events, trade_date).shares
# filing_events: step at filing_date only
# splits: multiply historical shares by split factor (same as price adj)

def apply_split_adjust_shares(shares_series, split_table):
    ...

def pit_shares_no_lookahead(trade_date, filing_steps):
    # last filing with filing_date &lt;= trade_date; never interpolate between filings
    ...</code></pre>
      </details>

      <details class="code-fold">
        <summary>PREPROC-016 · 新闻 24h 滑动去重（SimHash 草图）</summary>
        <pre><code>def simhash64(text: str) -> int: ...

def is_resend(title_a: str, title_b: str, threshold: float = 0.85) -> bool:
    # 3-gram Jaccard or SimHash hamming distance
    ...

# 聚合: news_count_1d 仅对 dedup_key 首次 +1; sentiment 可取最新</code></pre>
      </details>

      <details class="code-fold">
        <summary>CH-007 · <code>fact_bars_adjusted_minute</code> DDL（勿套用 panel_daily 分区）</summary>
        <pre><code>CREATE TABLE IF NOT EXISTS qs_massive.fact_bars_adjusted_minute
(
    timestamp DateTime64(3, 'UTC'),
    trade_date Date,
    ticker LowCardinality(String),
    permanent_id Nullable(String),
    adj_open Float64,
    adj_high Float64,
    adj_low Float64,
    adj_close Float64,
    adj_volume Float64,
    is_rth UInt8,
    is_early_close UInt8,
    batch_id String
)
ENGINE = MergeTree()
PARTITION BY (toYYYYMM(timestamp), toMonday(timestamp))
ORDER BY (trade_date, timestamp, ticker)
SETTINGS index_granularity = 8192;

-- 查询示例: 某日 09:35 全市场截面
-- SELECT * FROM fact_bars_adjusted_minute
-- WHERE trade_date = '2024-06-03' AND toHour(timestamp)=9 AND toMinute(timestamp)=35;</code></pre>
      </details>
'''

TASK_ROWS = """
          <tr><td>ENG-011</td><td><span class=\"badge badge-p1\">P1</span></td><td>提前收盘日 dim_calendar + 分钟 210/390 骨架</td><td class=\"owner-cell\">数据工程组</td><td class=\"status-cell\">待办</td><td></td></tr>
          <tr><td>PREPROC-014</td><td><span class=\"badge badge-p1\">P1</span></td><td>跨源代码后缀归一（PREPROC-014 CMS）</td><td class=\"owner-cell\">基础架构组</td><td class=\"status-cell\">待办</td><td></td></tr>
          <tr><td>PREPROC-015</td><td><span class=\"badge badge-p1\">P1</span></td><td>拆股股本调整 vs 季报阶梯锁定（防前视）</td><td class=\"owner-cell\">数据工程组</td><td class=\"status-cell\">待办</td><td></td></tr>
          <tr><td>PREPROC-016</td><td><span class=\"badge badge-p2\">P2</span></td><td>新闻 24h SimHash/3-gram 去重，count 不重复</td><td class=\"owner-cell\">量化研究组</td><td class=\"status-cell\">待办</td><td></td></tr>
          <tr><td>CH-007</td><td><span class=\"badge badge-p2\">P2</span></td><td>分钟表周级复合分区 + ORDER BY (trade_date,timestamp,ticker)</td><td class=\"owner-cell\">平台 SRE</td><td class=\"status-cell\">待办</td><td></td></tr>
"""

DDL_CAL_PATCH = """CREATE TABLE IF NOT EXISTS qs_massive.dim_calendar
(
    trade_date Date,
    is_trading_day UInt8 DEFAULT 1,
    market_close_time_et LowCardinality(String) DEFAULT '16:00',
    expected_rth_bars UInt16 DEFAULT 390,
    is_early_close UInt8 DEFAULT 0,
    source LowCardinality(String) DEFAULT 'day_aggs_union'
)"""

DDL_CAL_OLD = """CREATE TABLE IF NOT EXISTS qs_massive.dim_calendar
(
    trade_date Date,
    is_trading_day UInt8 DEFAULT 1,
    source LowCardinality(String) DEFAULT 'day_aggs_union'
)"""


def patch(html: str) -> str:
    if "sec-blind-spots-2" not in html:
        html = html.replace(
            '      <div class="box-warn">\n        <strong>评审会追加行：</strong> 五条已写入 <a href="#sec-tasks">§14 任务总表</a>；Phase 3 路线图在 P0 通过后并行启动 P1 项。\n      </div>\n    </section>\n\n    <!-- 7. Task table -->',
            '      <div class="box-warn">\n        <strong>评审会追加行（§12a）：</strong> 五条已写入 <a href="#sec-tasks">§14 任务总表</a>。\n      </div>\n' + ROUND2 + "\n    </section>\n\n    <!-- 7. Task table -->",
        )

    if "sec-code-round2" not in html:
        anchor = '<h3 id="sec-code-extra">10c.12 补充折叠'
        if anchor in html:
            html = html.replace(anchor, CODE_R2 + "\n\n      " + anchor)

    if "ENG-011" not in html.split("sec-tasks")[1][:8000]:
        html = html.replace(
            "          <tr><td>PREPROC-011</td>",
            TASK_ROWS + "          <tr><td>PREPROC-011</td>",
        )

    if DDL_CAL_OLD in html:
        html = html.replace(DDL_CAL_OLD, DDL_CAL_PATCH)

    # 10c.5.1 dim_calendar table rows
    if "expected_rth_bars" not in html.split("10c.5.1")[1].split("10c.5.2")[0]:
        html = html.replace(
            "<tr><td><code>source</code></td><td>String</td><td>如 <code>day_aggs_union</code></td><td>常量</td></tr>\n        </tbody>\n      </table>\n\n      <h4>10c.5.2 dim_security_master</h4>",
            "<tr><td><code>source</code></td><td>String</td><td>如 <code>day_aggs_union</code></td><td>常量</td></tr>\n"
            "<tr><td><code>market_close_time_et</code></td><td>String</td><td><code>16:00</code> / 提前 <code>13:00</code></td><td>ENG-011</td></tr>\n"
            "<tr><td><code>expected_rth_bars</code></td><td>UInt16</td><td>390 或 210</td><td>ENG-011</td></tr>\n"
            "<tr><td><code>is_early_close</code></td><td>UInt8</td><td>SIFMA 半日市</td><td>ENG-011</td></tr>\n"
            "        </tbody>\n      </table>\n\n      <h4>10c.5.2 dim_security_master</h4>",
        )

    # 11.3 table row for minute
    if "fact_bars_adjusted_minute" not in html:
        html = html.replace(
            "<tr><td><code>meta_load_batch</code></td><td>—（仅 CH）</td>",
            "<tr><td><code>fact_bars_adjusted_minute</code></td><td><code>fact_bars_adjusted_minute/…</code></td><td>(月,周)</td><td>(trade_date, timestamp, ticker)</td><td>分钟 RTH（CH-007）</td></tr>\n"
            "          <tr><td><code>meta_load_batch</code></td><td>—（仅 CH）</td>",
        )

    # 11.1 what goes to CH
    if "fact_bars_adjusted_minute" not in html.split("11.1")[1].split("11.2")[0]:
        html = html.replace(
            "<tr><td>tick quotes/trades（~10 TB）</td><td><strong>否（首期）</strong></td>",
            "<tr><td>分钟物化 <code>fact_bars_adjusted_minute</code></td><td><strong>P2 可选</strong></td><td><code>qs_massive</code></td><td>CH-007 专属分区；raw tick 仍不进</td></tr>\n"
            "          <tr><td>tick quotes/trades（~10 TB）</td><td><strong>否（首期）</strong></td>",
        )

    # Nav
    nav = html.split("</nav>")[0]
    if "sec-blind-spots-2" not in nav:
        html = html.replace(
            '<li><a href="#sec-blind-spots">12a. 工业级补盲点（采纳）</a></li>',
            '<li><a href="#sec-blind-spots">12a. 工业级补盲点</a></li>\n        <li><a href="#sec-blind-spots-2">12b. 二轮补盲点（分钟/新闻/CH）</a></li>',
        )
    if "sec-code-round2" not in nav:
        html = html.replace(
            '<li><a href="#sec-code-extra">10c.12 运维/校验折叠</a></li>',
            '<li><a href="#sec-code-extra">10c.12 运维/校验折叠</a></li>\n        <li><a href="#sec-code-round2">10c.14 二轮折叠代码</a></li>',
        )

    # Strengthen blind spot 3 in 12a
    if "PREPROC-015" not in html.split("盲点三")[1].split("盲点四")[0] if "盲点三" in html else "":
        html = html.replace(
            "<p><strong>任务：</strong> <code>PREPROC-013</code>（P1）— 数据工程组；依赖 P0 后 REST 日频字段盘点。</p>",
            "<p><strong>任务：</strong> <code>PREPROC-013</code>（P1）+ <code>PREPROC-015</code> 双轨股本（见 <a href=\"#sec-blind-spots-2\">§12b 盲点八</a>）。</p>",
        )

    # before-factor checklist
    if "PREPROC-014" not in html.split("sec-before-factor")[1].split("sec-preproc")[0]:
        html = html.replace(
            "<li>确认未在 Layer 2 做 MAD/Barra（那是 factor_evaluation）</li>",
            "<li><code>ENG-011</code>：提前收盘日 <code>expected_rth_bars</code> 非一律 390</li>\n"
            "        <li><code>PREPROC-014</code>：<code>ticker_normalized</code> 多源 join 一致</li>\n"
            "        <li><code>PREPROC-015</code>：股本阶梯无 filing 向前插值</li>\n"
            "        <li><code>PREPROC-016</code>：新闻 count 已去重（若启用 news 列）</li>\n"
            "        <li>确认未在 Layer 2 做 MAD/Barra（那是 factor_evaluation）</li>",
        )

    # Phase 3 roadmap
    html = html.replace(
        "PREPROC-011~013 · FEAT-001~004 · CH-001~004 · CH-006 · TASK-DC-005",
        "PREPROC-011~016 · ENG-011 · FEAT-001~004 · CH-001~007 · TASK-DC-005",
    )

    # 10c.10 index
    if "CH-007" not in html.split("10c.10")[1].split("10c.11")[0]:
        html = html.replace(
            "<tr><td>TASK-DC-005</td><td><a href=\"#sec-code-assets\">tape_condition_filter.yaml</a></td>",
            "<tr><td>ENG-011 / 014~016</td><td><a href=\"#sec-code-round2\">§10c.14</a></td><td>二轮补盲点</td></tr>\n"
            "          <tr><td>CH-007</td><td><a href=\"#sec-code-round2\">分钟 DDL</a></td><td>分钟 CH 分区</td></tr>\n"
            "          <tr><td>TASK-DC-005</td><td><a href=\"#sec-code-assets\">tape_condition_filter.yaml</a></td>",
        )

    return html


def main():
    html = HTML.read_text(encoding="utf-8")
    html = patch(html)
    HTML.write_text(html, encoding="utf-8")
    print("round2 added, lines:", html.count("\n") + 1)


if __name__ == "__main__":
    main()
