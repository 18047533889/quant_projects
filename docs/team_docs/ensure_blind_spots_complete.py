#!/usr/bin/env python3
"""Ensure all round-2/3/edge blind spots are complete in HTML + MD."""
import re
from pathlib import Path

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"
MD = Path(__file__).resolve().parent / "美股原始数据预处理标准方案_因子入模前.md"

CODE_R2 = r'''
      <h3 id="sec-code-round2">10c.14 二轮补盲点 · 折叠代码</h3>

      <details class="code-fold">
        <summary>ENG-011 · <code>dim_calendar</code> 提前收盘列 + SIFMA 种子（YAML）</summary>
        <pre><code># early_close_days.yaml — Black Friday / Christmas Eve / July 3 Eve → 210 bars
early_close_half_days:
  - { date: "2024-11-29", note: "Black Friday" }
  - { date: "2024-12-24", note: "Christmas Eve" }
  - { date: "2024-07-03", note: "Independence Day Eve" }
# 常规: market_close_time_et "16:00", expected_rth_bars 390
# 提前: "13:00", 210 — 13:00-16:00 禁止 ffill/填 0</code></pre>
      </details>

      <details class="code-fold">
        <summary>PREPROC-014 · <code>ticker_normalize.py</code></summary>
        <pre><code>import re
_SUFFIX_RE = re.compile(r"[\s./\-]+")
def normalize_ticker(raw: str) -> str:
    t = raw.upper().strip()
    return _SUFFIX_RE.sub("", t)  # BRK.B / BRK/B / BRK-B -&gt; BRKB</code></pre>
      </details>

      <details class="code-fold">
        <summary>PREPROC-015 · 股本双轨 asof</summary>
        <pre><code># splits: historical shares /= factor; filing_date: step only, no interpolate</code></pre>
      </details>

      <details class="code-fold">
        <summary>PREPROC-016 · 新闻 SimHash 去重</summary>
        <pre><code># 24h window similarity &gt; 0.85 → resend; news_count_* deduped</code></pre>
      </details>

      <details class="code-fold">
        <summary>CH-007 · <code>fact_bars_adjusted_minute</code> DDL</summary>
        <pre><code>CREATE TABLE IF NOT EXISTS qs_massive.fact_bars_adjusted_minute (
    timestamp DateTime64(3,'UTC'), trade_date Date, ticker LowCardinality(String),
    adj_open Float64, adj_high Float64, adj_low Float64, adj_close Float64, adj_volume Float64,
    is_rth UInt8, is_early_close UInt8, is_market_halt UInt8, is_ticker_halt UInt8,
    is_ipo_pre_open UInt8, price_clamped_flag UInt8, batch_id String
) ENGINE = MergeTree()
PARTITION BY (toYYYYMM(timestamp), toMonday(timestamp))
ORDER BY (trade_date, timestamp, ticker);</code></pre>
      </details>
'''

# §12a 关系表（仅三列，不含 §14 任务行）
STAGE_TABLE = """
      <h3>与现有 Stage / 任务的关系</h3>
      <table>
        <thead><tr><th>新 ID</th><th>插入阶段</th><th>依赖</th></tr></thead>
        <tbody>
          <tr><td>PREPROC-011</td><td>Stage 1 之后、Panel 7 之前</td><td>all_tickers 全量；security_master</td></tr>
          <tr><td>PREPROC-012</td><td>Stage 4 returns + Stage 2 universe</td><td>delisting 元数据</td></tr>
          <tr><td>PREPROC-013</td><td>Stage 6 LTM/估值</td><td>stocks_floats、splits、REST 日频字段</td></tr>
          <tr><td>CH-006</td><td>§11 CH-002 首次导入后</td><td>panel_daily 主表</td></tr>
          <tr><td>TASK-DC-005</td><td>Stage 8 tick/分钟</td><td>condition_codes 维表</td></tr>
        </tbody>
      </table>
"""

EXPAND_12B_ENG011 = """
      <p class="box-warn"><strong>补充一（SIFMA Early Closes）：</strong>感恩节翌日 Black Friday、圣诞节前夕 Christmas Eve、独立日前夕（7月3日）等日，
        美东 <strong>13:00</strong> 提前休市；RTH 骨架须由 390 根<strong>坍缩为 210 根</strong>。
        13:00–16:00 共 180 根若被 ffill/填 0，将污染日内波动率与成交量衰减因子（伪结构异动）。</p>
"""

EXPAND_12D = r'''
      <h3>极端一补充：MWCB 分级（ENG-012）</h3>
      <table class="table-schema">
        <thead><tr><th>级别</th><th>触发（标普跌幅示意）</th><th>暂停</th><th>Layer 2</th></tr></thead>
        <tbody>
          <tr><td>Level 1</td><td>≥7%（13:00 前）等</td><td>15 分钟</td><td><code>is_market_halt=1</code>；滚动算子跳过</td></tr>
          <tr><td>Level 2</td><td>更深跌幅</td><td>15 分钟</td><td>同上；禁止误判为零波动横盘</td></tr>
          <tr><td>Level 3</td><td>极端</td><td>可全日提早收市</td><td>联动 <code>expected_rth_bars</code> / ENG-011</td></tr>
        </tbody>
      </table>
      <p>参考日：<strong>2020-03</strong> 新冠疫情连续触发熔断 — 验收金样例。</p>

      <h3>极端二补充：Ghost Trades 场景（ENG-013）</h3>
      <ul>
        <li>周末、耶稣受难日 Good Friday、全国哀悼日等非 \(\mathcal{T}\) 日期上的测试成交</li>
        <li>美东凌晨 <strong>04:00 前</strong> 网关重置残余行 — 一律 Hard Delete，不进 CH 分区</li>
        <li><code>market_holidays</code> 仅 2026–2027 时，须由 SIP 日 K 反推全历史 \(\mathcal{T}\)</li>
      </ul>

      <h3>极端三补充：LULD 与监管停牌（PREPROC-018）</h3>
      <table class="table-schema">
        <thead><tr><th>类型</th><th>行为</th><th>填充</th></tr></thead>
        <tbody>
          <tr><td>LULD 5min Pause</td><td>5 分钟内波动超阈 → 暂停 5 分钟</td><td>OHLC ffill；<code>volume=0</code>；<code>is_ticker_halt=1</code></td></tr>
          <tr><td>Regulatory Halt T1/T2</td><td>重大新闻监管停牌</td><td>reindex 补 bar；累计 RTH 停牌 &gt;60min → <code>universe_mask=False</code></td></tr>
        </tbody>
      </table>
      <p><strong>红线：</strong>禁止仅 drop 停牌日行（生存者偏见）；组合优化器不得对无法平仓股虚拟建仓。</p>

      <h3>极端四补充：仙股合股与除零（PREPROC-019）</h3>
      <p>多次 <strong>Reverse Split（如 1:50）</strong> + 特派红利可使复权 \(P_{adj}\le 0\) 或 \(\lt 10^{-4}\)，引发
        <code>FloatingPointError</code>、<code>Inf</code> 与 CH 索引扭曲。Floor clamp（如 <code>1e-5</code>）+ <code>price_clamped_flag</code>。</p>

      <h3>极端五补充：IPO Opening Cross（PREPROC-020）</h3>
      <p>09:30 敲钟常<strong>无真实成交</strong>；首笔可能在 10:30 / 11:30 / 13:00 Opening Cross。
        <strong>禁止</strong>用不存在之「昨收」ffill；<code>is_ipo_pre_open=1</code> 至首笔成交；时序算子冷启动。</p>
'''

REGISTRY = r'''
    <section id="sec-blind-spots-registry">
      <h2><span class="sec-badge">12e</span> 终审任务登记总表（评审稿对照 · 可直接认领）</h2>
      <p class="section-desc">汇总两轮评审共 <strong>15 项</strong>入模前任务（§12b–§12d 正文详述）。负责人/状态请在 <a href="#sec-tasks">§14</a> 同步填写。</p>

      <h3>表 A · Feature Store 防腐层（对应评审「补充一～五」）</h3>
      <table class="table-schema">
        <thead><tr><th>阶段</th><th>任务 ID</th><th>增补处理项</th><th>负责人</th><th>验收标准</th></tr></thead>
        <tbody>
          <tr><td>Layer 1.5</td><td><code>ENG-011</code></td><td>SIFMA 提前收盘（13:00）分钟骨架 390→210 自适应</td><td>数据工程组</td><td>7/3、Black Friday 等 Bar=210，13:00 后无 ffill</td></tr>
          <tr><td>Layer 1.5</td><td><code>PREPROC-014</code></td><td>代码后缀 <code>. / -</code> 剥离归一（<code>BRKB</code>）</td><td>基础架构组</td><td>跨源 <code>BRK.B</code>/<code>BRK/B</code> 合并无断链</td></tr>
          <tr><td>Layer 2</td><td><code>PREPROC-017</code></td><td>拆股对偶调价量；分红只调价不调量</td><td>数据工程组</td><td>派息后历史换手率/vol 无乘除</td></tr>
          <tr><td>Layer 2</td><td><code>PREPROC-021</code></td><td>Mask-First 掩码前置（防上游污染）</td><td>量化研究组</td><td>停牌脏价不进入 rolling buffer</td></tr>
          <tr><td>Layer 2 多频</td><td><code>PREPROC-022</code></td><td>SEC <code>acceptance_time</code> 秒级日内 asof</td><td>基础架构组</td><td>无 11:15 披露在 11:14 bar 的前视</td></tr>
          <tr><td>Layer 1.5→2</td><td><code>PREPROC-011</code></td><td>实体 ID + 代码回收（<code>APP</code> 僵尸污染）</td><td>基础架构组</td><td>长窗不串线不同法人实体</td></tr>
          <tr><td>Layer 2</td><td><code>PREPROC-015</code></td><td>股本双轨阶梯（§12b，≠017）</td><td>数据工程组</td><td>filing 不向前插值股数</td></tr>
          <tr><td>Layer 2</td><td><code>PREPROC-016</code></td><td>新闻 24h 去重（§12b）</td><td>量化研究组</td><td><code>news_count</code> 不重复</td></tr>
          <tr><td>CH</td><td><code>CH-007</code></td><td>分钟表专属分区（§12b）</td><td>平台 SRE</td><td>勿套用 panel_daily 分区</td></tr>
        </tbody>
      </table>

      <h3>表 B · 极端异常清洗（对应评审「极端一～五」）</h3>
      <table class="table-schema">
        <thead><tr><th>阶段</th><th>任务 ID</th><th>处理项</th><th>负责人</th><th>验收红线</th></tr></thead>
        <tbody>
          <tr><td>Layer 1.5</td><td><code>ENG-012</code></td><td>MWCB 15min 断层 + <code>is_market_halt</code></td><td>数据工程组</td><td>2020-03 无伪零波动</td></tr>
          <tr><td>Layer 1.5</td><td><code>ENG-013</code></td><td>𝒯 硬滤 + Ghost Trades 物理删除</td><td>平台 SRE</td><td>非交易日零行入库</td></tr>
          <tr><td>Layer 2</td><td><code>PREPROC-018</code></td><td>LULD/监管停牌 + universe 联动</td><td>量化研究组</td><td>长停牌日 mask=False</td></tr>
          <tr><td>Layer 2</td><td><code>PREPROC-019</code></td><td>仙股复权价 floor clamp</td><td>基础架构组</td><td>无 \(P_{adj}\le 0\)</td></tr>
          <tr><td>Layer 2</td><td><code>PREPROC-020</code></td><td>IPO 延迟开盘冷启动</td><td>基础架构组</td><td>Opening Cross 前无假 ffill</td></tr>
        </tbody>
      </table>
      <div class="box-info">
        <strong>评审稿 ID 对照：</strong>「补充三 Mask-First」→ <code>PREPROC-021</code>；「补充四 成交量非对称」→ <code>PREPROC-017</code>（非 §12b 的 <code>PREPROC-015</code> 股本双轨）；「补充五 日内错位」→ <code>PREPROC-022</code>。
      </div>
    </section>
'''

TASK_SEC14 = """
          <tr><td>PREPROC-011</td><td><span class=\"badge badge-p1\">P1</span></td><td>dim_ticker_map / permanent_id 实体血缘</td><td class=\"owner-cell\">基础架构组</td><td class=\"status-cell\">待办</td><td></td></tr>
          <tr><td>PREPROC-012</td><td><span class=\"badge badge-p1\">P1</span></td><td>退市收益惩罚 delisting_policy</td><td class=\"owner-cell\">算法策略组</td><td class=\"status-cell\">待办</td><td></td></tr>
          <tr><td>PREPROC-013</td><td><span class=\"badge badge-p1\">P1</span></td><td>PIT 动态股本 / pit_market_cap</td><td class=\"owner-cell\">数据工程组</td><td class=\"status-cell\">待办</td><td></td></tr>
          <tr><td>CH-006</td><td><span class=\"badge badge-p1\">P1</span></td><td>panel_daily Projection p_ticker_timeline</td><td class=\"owner-cell\">平台 SRE</td><td class=\"status-cell\">待办</td><td></td></tr>
          <tr><td>TASK-DC-005</td><td><span class=\"badge badge-p1\">P1</span></td><td>tape_condition_filter.yaml 条件码协议</td><td class=\"owner-cell\">量化+数据</td><td class=\"status-cell\">待办</td><td></td></tr>
          <tr><td>ENG-011</td><td><span class=\"badge badge-p1\">P1</span></td><td>SIFMA 提前收盘 210/390 分钟骨架</td><td class=\"owner-cell\">数据工程组</td><td class=\"status-cell\">待办</td><td></td></tr>
          <tr><td>PREPROC-014</td><td><span class=\"badge badge-p1\">P1</span></td><td>ticker_normalized 后缀剥离归一</td><td class=\"owner-cell\">基础架构组</td><td class=\"status-cell\">待办</td><td></td></tr>
          <tr><td>PREPROC-015</td><td><span class=\"badge badge-p1\">P1</span></td><td>股本双轨阶梯（有机/非有机）</td><td class=\"owner-cell\">数据工程组</td><td class=\"status-cell\">待办</td><td></td></tr>
          <tr><td>PREPROC-016</td><td><span class=\"badge badge-p2\">P2</span></td><td>新闻 SimHash/3-gram 去重</td><td class=\"owner-cell\">量化研究组</td><td class=\"status-cell\">待办</td><td></td></tr>
          <tr><td>CH-007</td><td><span class=\"badge badge-p2\">P2</span></td><td>fact_bars_adjusted_minute CH 分区</td><td class=\"owner-cell\">平台 SRE</td><td class=\"status-cell\">待办</td><td></td></tr>
"""

MD_FULL = r'''

## 21. 三轮补盲点与极端异常（完整登记）

### 21.1 Feature Store 防腐（§12c）

| ID | 内容 |
|----|------|
| ENG-011 | SIFMA 提前收盘 13:00 → 210 bars |
| PREPROC-014 | ticker 后缀归一 BRKB |
| PREPROC-011 | 实体 ID / 代码回收 APP 等 |
| PREPROC-017 | split 调价量；dividend 不调 vol |
| PREPROC-021 | Mask-First / Upstream Contamination |
| PREPROC-022 | acceptance_time 分钟 asof |
| PREPROC-015 | 股本双轨（§12b） |
| PREPROC-016 | 新闻去重（§12b） |
| CH-007 | 分钟 CH 分区 |

### 21.2 极端异常（§12d）

| ID | 内容 |
|----|------|
| ENG-012 | MWCB Level1/2/3, is_market_halt |
| ENG-013 | Ghost Trades, 真值日历 𝒯 |
| PREPROC-018 | LULD 5min, T1/T2, universe>60min |
| PREPROC-019 | Penny stock floor clamp |
| PREPROC-020 | IPO Opening Cross 冷启动 |

HTML 总表：<a href="Massive数据治理与改进行动清单.html#sec-blind-spots-registry">§12e 终审登记</a>。
'''


def fix_12a_stage_table(html: str) -> str:
    pat = re.compile(
        r"<h3>与现有 Stage / 任务的关系</h3>\s*<table>.*?</table>",
        re.DOTALL,
    )
    if pat.search(html):
        html = pat.sub(STAGE_TABLE.strip(), html, count=1)
    return html


def patch(html: str) -> str:
    html = fix_12a_stage_table(html)

    if 'id="sec-code-round2"' not in html:
        html = html.replace(
            '<h3 id="sec-code-round3">10c.15',
            CODE_R2.strip() + '\n\n      <h3 id="sec-code-round3">10c.15',
            1,
        )

    if "补充一（SIFMA Early Closes）" not in html:
        html = html.replace(
            "<h3>盲点六：SIFMA 提前收盘与分钟骨架（ENG-011）</h3>",
            "<h3>盲点六：SIFMA 提前收盘与分钟骨架（ENG-011）</h3>" + EXPAND_12B_ENG011,
            1,
        )

    if "MWCB 分级" not in html:
        html = html.replace(
            "<h3>极端异常任务登记</h3>",
            EXPAND_12D.strip() + "\n\n      <h3>极端异常任务登记</h3>",
            1,
        )

    if "sec-blind-spots-registry" not in html:
        marker = (
            '        五条已追加 <a href="#sec-tasks">§14</a>；与 §12c 一并构成<strong>入模前封板终审清单</strong>。'
            '代码 <a href="#sec-code-round3">§10c.15</a>。\n'
            "      </div>\n    </section>\n\n\n    <!-- 7. Task table -->"
        )
        html = html.replace(
            marker,
            '        五条已追加 <a href="#sec-tasks">§14</a>；与 §12c 一并构成<strong>入模前封板终审清单</strong>。'
            '代码 <a href="#sec-code-round3">§10c.15</a>。\n'
            "      </div>\n    </section>\n\n" + REGISTRY + "\n\n\n    <!-- 7. Task table -->",
            1,
        )

    m = re.search(r'(<section id="sec-tasks">.*?<tbody>)(.*?)(</tbody>)', html, re.DOTALL)
    if m and "<td>PREPROC-011</td>" not in m.group(2):
        html = html[: m.start(2)] + m.group(2).rstrip() + TASK_SEC14 + m.group(3) + html[m.end(3) :]

    html = html.replace(
        "版本 v3.1 HTML 三轮补盲点+极端异常版",
        "版本 v3.2 HTML 终审完整版（12a–12e）",
    )
    html = html.replace("v3.1 三轮+极端异常版", "v3.2 终审完整版")

    return html


def patch_md(md: str) -> str:
    if "### 21.2 极端异常" not in md:
        if "## 21. 三轮补盲点" in md:
            md = re.sub(r"## 21\. 三轮补盲点.*", MD_FULL.strip(), md, flags=re.DOTALL)
        else:
            md = md.rstrip() + MD_FULL
    return md


def main():
    html = HTML.read_text(encoding="utf-8")
    html = patch(html)
    HTML.write_text(html, encoding="utf-8")
    if MD.exists():
        MD.write_text(patch_md(MD.read_text(encoding="utf-8")), encoding="utf-8")
    # audit
    checks = [
        "sec-blind-spots-2", "sec-blind-spots-3", "sec-blind-spots-4", "sec-blind-spots-registry",
        "sec-code-round2", "sec-code-round3", "ENG-011", "PREPROC-017", "PREPROC-021", "PREPROC-022",
        "ENG-012", "ENG-013", "PREPROC-018", "PREPROC-019", "PREPROC-020", "Mask-First", "Ghost",
        "Black Friday", "MWCB 分级", "LULD", "Opening Cross",
    ]
    missing = [c for c in checks if c not in html]
    print("lines:", html.count("\n") + 1)
    print("missing:", missing or "none")


if __name__ == "__main__":
    main()
