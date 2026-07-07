#!/usr/bin/env python3
"""Add industrial-grade blind-spot section (PREPROC-011~013, CH-006, TASK-DC-005)."""
import re
from pathlib import Path

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"
MD = Path(__file__).resolve().parent / "美股原始数据预处理标准方案_因子入模前.md"

SECTION = r'''
    <section id="sec-blind-spots">
      <h2>12a. 工业级补盲点（评审采纳 · 2026-06）</h2>
      <p class="section-desc">
        外部评审指出 5 项「隐藏盲区」。经对照现有方案：<strong>五条均成立或部分已提及但未工程化</strong>，现纳入 Layer 2 / ClickHouse 规范。
        下列每条含：<strong>采纳结论</strong>、与现状关系、落地动作、任务 ID。
      </p>

      <div class="box-info">
        <strong>总评：</strong>与既有红线（PiT、双价源隔离、REST 截断）正交，补齐美股微观结构、标的连续性与 CH 性能；
        不替代 P0 补数，应在 <code>PREPROC-007</code> 物化前/同期设计进 schema。
      </div>

      <h3>盲点一：代码变更与标的血缘（Permanent ID）</h3>
      <p><strong>采纳。</strong> 当前 <code>panel_daily</code> / CH 以 <code>(trade_date, ticker)</code> 为逻辑键合理（横截面、供应商口径），但
        <strong>Track B 连续时序</strong>若仅按 ticker 串联，会在代码变更日（如 <code>FB→META</code>）发生序列断裂，或截面日出现「新旧代码」重复计数风险。</p>
      <table>
        <thead><tr><th>项</th><th>说明</th></tr></thead>
        <tbody>
          <tr><td>现状</td><td><code>all_tickers</code> 含 <code>composite_figi</code>、<code>cik</code>，但为 <strong>2k 快照</strong>，不足以单独支撑全历史映射</td></tr>
          <tr><td>勘误</td><td><code>GOOG</code>/<code>GOOGL</code> 多为<strong>不同股份类别</strong>，非单纯 rename；不可默认合并为一条序列，需按 share-class 策略分支</td></tr>
          <tr><td>落地</td><td>
            ① 维表 <code>dim_ticker_map</code>：<code>ticker</code>, <code>valid_from</code>, <code>valid_to</code>, <code>permanent_id</code>（优先 <code>composite_figi</code>，缺失时用 <code>cik</code>+交易所或自建 <code>sec_id</code>）<br>
            ② <code>panel_daily</code> 增加 <code>permanent_id</code>、<code>ticker_as_of</code>（当日有效代码）；CH <code>ORDER BY</code> 仍以 <code>(trade_date, ticker)</code> 服务 Track A<br>
            ③ Track B API：输入 <code>META</code> 时通过 map <strong>asof</strong> 拉取 <code>FB</code> 时代历史（同一 <code>permanent_id</code>）
          </td></tr>
          <tr><td>任务</td><td><code>PREPROC-011</code>（P1）— 基础架构组</td></tr>
        </tbody>
      </table>

      <h3>盲点二：退市收益惩罚（Delisting Return）</h3>
      <p><strong>采纳（强化 §4.5）。</strong> 动态 Universe 已要求含退市前历史，但若最后交易日无 bar 或仅从掩码剔除、不在 <code>returns_daily</code> 写入最后一跳收益，回测会<strong>静默躲过</strong>破产摘牌暴跌，夏普虚高。</p>
      <table>
        <thead><tr><th>退市类型</th><th>Layer 2 处理（示例策略，须写入 <code>delisting_policy</code> 元数据）</th></tr></thead>
        <tbody>
          <tr><td>破产 / 强制摘牌</td><td>最后交易日识别后写入<strong>惩罚收益</strong>（如 -90%～-100%）或 OTC 首日可观测价；禁止「无 bar = 无损失」</td></tr>
          <tr><td>并购现金收购</td><td>按公告/最后价映射<strong>正收益跳变</strong>，非一律 -100%</td></tr>
          <tr><td>主动退市转 OTC</td><td>若有 OTC 首笔，可作为清算价；否则策略化惩罚 + 敏感性分析</td></tr>
        </tbody>
      </table>
      <p><strong>勘误：</strong> Massive <strong>无</strong> CRSP 标准退市收益表；惩罚系数为<strong>回测政策</strong>而非供应商真值，需在文档与因子评估层一致声明。</p>
      <p>触发器：<code>all_tickers.delisted_utc</code>（全量后）、行情末 bar、<code>ipos</code>/公司行动交叉验证。</p>
      <p><strong>任务：</strong> <code>PREPROC-012</code>（P1）— 算法策略组 + 数据工程联合验收。</p>

      <h3>盲点三：动态总股本与市值（PIT Shares / Market Cap）</h3>
      <p><strong>采纳（强化 Stage 6）。</strong> 仅用季报 <code>basic_shares_outstanding</code> 作 90 天常数分母，会使 <code>market_cap = close × shares</code> 在披露日阶跃，PE/PB 截面扭曲。</p>
      <ul>
        <li><strong>优先级：</strong> 日频股本（若 REST <code>daily_market_summary</code> 或供应商字段可用）→ <code>stocks_floats</code> asof 事件表 → 季报 PiT 股数</li>
        <li><strong>与复权一致：</strong> 在 <code>splits</code> 执行日对历史 <code>shares_outstanding</code> 做与价格对称的<strong>股本逆向调整</strong>，避免「复权价 × 未调股本」</li>
        <li><strong>输出：</strong> <code>pit_shares_out</code>、<code>pit_market_cap</code>（可空）；元数据标注 <code>shares_source</code></li>
      </ul>
      <p><strong>任务：</strong> <code>PREPROC-013</code>（P1）— 数据工程组；依赖 P0 后 REST 日频字段盘点。</p>

      <h3>盲点四：ClickHouse Projections（单表双轨）</h3>
      <p><strong>采纳，优先于整表复制。</strong> 现有 §11 / CH-005 建议第二张表或 MV 解决 Track B；对 11TB 级宽表<strong>存储翻倍 + 双写校验</strong>成本过高。现代 CH 可用 <strong>Projection</strong> 在 <code>ORDER BY (trade_date, ticker)</code> 主表上附加 <code>(ticker, trade_date)</code> 物理索引。</p>
<pre>ALTER TABLE qs_massive.panel_daily
    ADD PROJECTION IF NOT EXISTS p_ticker_timeline
    (
        SELECT *
        ORDER BY (ticker, trade_date)
    );
ALTER TABLE qs_massive.panel_daily
    MATERIALIZE PROJECTION p_ticker_timeline;</pre>
      <p><strong>注意：</strong> 需 ClickHouse 版本支持 Projection（建议 ≥ 23.x）；物化投影耗时纳入首次导入计划；<code>CH-005</code> 降为备选（MV/副表仅当 Projection 不可用时）。</p>
      <p><strong>任务：</strong> <code>CH-006</code>（P1）— 平台 SRE；验收：单票 <code>WHERE ticker='NVDA' ORDER BY trade_date</code> 命中投影且延迟达标。</p>

      <h3>盲点五：成交条件码过滤协议（Condition Codes）</h3>
      <p><strong>采纳（封板 PREPROC-008）。</strong> 报告已指出 trades <code>conditions</code> 复杂；Stage 8 仅有原则性「剔除非 regular」，缺<strong>可执行的黑白名单</strong>，分钟 H/L 易被盘前盘后、均价单、延迟上报污染。</p>
      <p><strong>落地：</strong> 基于 <code>market_operations/condition_codes</code> 维表，由<strong>量化研究 + 数据工程</strong> 联合签署
        <em>Consolidated Tape Filtering Protocol</em>（版本化 YAML）：</p>
      <table>
        <thead><tr><th>规则层</th><th>行为</th></tr></thead>
        <tbody>
          <tr><td>参与 H/L/Close 聚合</td><td>仅 <strong>Regular Sale</strong> 类（白名单，以维表 <code>id</code> 映射为准，<strong>禁止</strong>未查表硬编码字母）</td></tr>
          <tr><td>盘前盘后 / 错时 / 衍生定价（评审示例 T,Z,U,I 等）</td><td>成交量可计入 <code>vol_alt</code>；成交价<strong>禁止</strong>进入 1min bar 的 high/low</td></tr>
          <tr><td>conditions=null</td><td>按 Massive 文档视为常规成交的一部分，纳入白名单默认集</td></tr>
        </tbody>
      </table>
      <p><strong>任务：</strong> <code>TASK-DC-005</code>（P1）— 产出 <code>config/tape_condition_filter.yaml</code> + 1 日 tick→1min 金样例回归。</p>

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

      <div class="box-warn">
        <strong>评审会追加行：</strong> 五条已写入 <a href="#sec-tasks">§14 任务总表</a>；Phase 3 路线图在 P0 通过后并行启动 P1 项。
      </div>
    </section>
'''

TASK_ROWS = """          <tr><td>PREPROC-011</td><td><span class="badge badge-p1">P1</span></td><td>Permanent ID（FIGI）血缘 <code>dim_ticker_map</code></td><td class="owner-cell">基础架构组</td><td class="status-cell">待办</td><td></td></tr>
          <tr><td>PREPROC-012</td><td><span class="badge badge-p1">P1</span></td><td>退市清算收益惩罚算子 + <code>delisting_policy</code></td><td class="owner-cell">算法策略组</td><td class="status-cell">待办</td><td></td></tr>
          <tr><td>PREPROC-013</td><td><span class="badge badge-p1">P1</span></td><td>日频动态股本 / 市值平滑（<code>pit_shares_out</code>）</td><td class="owner-cell">数据工程组</td><td class="status-cell">待办</td><td></td></tr>
          <tr><td>CH-006</td><td><span class="badge badge-p1">P1</span></td><td>panel_daily Projection 单表双轨（优先于 CH-005 整表复制）</td><td class="owner-cell">平台 SRE</td><td class="status-cell">待办</td><td></td></tr>
          <tr><td>TASK-DC-005</td><td><span class="badge badge-p1">P1</span></td><td>条件码过滤协议 YAML + 金样例</td><td class="owner-cell">量化研究组</td><td class="status-cell">待办</td><td></td></tr>
"""

CH006_ROW = '<tr><td><code>CH-006</code></td><td><span class="badge badge-p1">P1</span></td><td>panel_daily <strong>Projection</strong> <code>p_ticker_timeline</code>（单表双轨，优先于 CH-005）</td><td>§12a；<code>massive_ddl.sql</code> 内嵌 PROJECTION</td></tr>\n          '

CH005_NOTE = """      <p class="section-desc"><strong>CH-005 降级：</strong> 整表复制 / 物化视图 <code>panel_by_ticker</code> 仅当环境不支持 Projection 时启用；首选 <a href="#sec-blind-spots">CH-006</a>。</p>
"""


def patch_html(html: str) -> str:
    if "sec-blind-spots" in html:
        html = re.sub(
            r'\s*<section id="sec-blind-spots">.*?</section>\s*',
            "\n",
            html,
            count=1,
            flags=re.DOTALL,
        )

    # Insert before sec-tasks
    pat = r'(    <!-- 7\. Task table -->\s*<section id="sec-tasks">)'
    if not re.search(pat, html):
        raise SystemExit("sec-tasks anchor not found")
    html = re.sub(pat, SECTION + r"\n\1", html, count=1)

    nav_marker = '<li><a href="#sec-tasks">14. 任务总表</a></li>'
    if "sec-blind-spots" not in html.split("</nav>")[0]:
        html = html.replace(
            nav_marker,
            '<li><a href="#sec-blind-spots">12a. 工业级补盲点（采纳）</a></li>\n        ' + nav_marker,
        )

    if "PREPROC-011" not in html:
        html = html.replace(
            "          <tr><td>TASK-DC-001~004</td>",
            TASK_ROWS + "          <tr><td>TASK-DC-001~004</td>",
        )

    # CH engineering table
    if "CH-006" not in html or '<code>CH-006</code></td><td><span class="badge badge-p1">P1</span></td><td>panel_daily <strong>Projection</strong>' not in html:
        if "<code>CH-005</code></td><td><span class=\"badge badge-p2\">P2</span></td><td>Track B" in html:
            html = html.replace(
                "<code>CH-005</code></td><td><span class=\"badge badge-p2\">P2</span></td><td>Track B 物化视图或 <code>panel_by_ticker</code></td><td>时序查询 SLA</td></tr>",
                CH006_ROW.strip() + "\n          <tr><td><code>CH-005</code></td><td><span class=\"badge badge-p2\">P2</span></td><td>Track B 备选 MV（Projection 不可用时）</td><td>时序查询 SLA</td></tr>",
            )
        elif "CH-005" in html and "CH-006" not in html:
            html = html.replace(
                "</tbody>\n      </table>\n\n      <h3>11.10 验收</h3>",
                CH006_ROW + "        </tbody>\n      </table>\n\n      <h3>11.10 验收</h3>",
                1,
            )

    if "CH-005 降级" not in html and "11.5.2 单票时序" in html:
        html = html.replace(
            "<p>若时序查询成为瓶颈，增加物化视图 <code>panel_by_ticker</code>，<code>ORDER BY (ticker, trade_date)</code>。</p>",
            "<p>首选 <a href=\"#sec-blind-spots\">CH-006 Projection</a> <code>p_ticker_timeline</code>；仅当 CH 版本不支持时再建物化视图 <code>panel_by_ticker</code>。</p>",
        )

    # before-factor checklist items
    if "PREPROC-011" not in html.split("sec-before-factor")[1].split("sec-preproc")[0]:
        html = html.replace(
            "<li>确认未在 Layer 2 做 MAD/Barra（那是 factor_evaluation）</li>",
            "<li><code>PREPROC-011</code>：<code>permanent_id</code> / <code>dim_ticker_map</code> 已物化（Track B 需）</li>\n        <li><code>PREPROC-012</code>：<code>delisting_policy</code> 与退市惩罚收益已写入 <code>returns_daily</code></li>\n        <li><code>PREPROC-013</code>：<code>pit_shares_out</code> / 市值分母非 90 天常数（若用 PE/PB）</li>\n        <li>确认未在 Layer 2 做 MAD/Barra（那是 factor_evaluation）</li>",
        )

    # schema table - add permanent_id row
    if "permanent_id" not in html or html.count("permanent_id") < 2:
        html = html.replace(
            "<tr><td>坐标</td><td><code>trade_date</code>, <code>ticker</code>, <code>align_time</code></td>",
            "<tr><td>坐标</td><td><code>trade_date</code>, <code>ticker</code>, <code>permanent_id</code>, <code>align_time</code></td>",
            1,
        )

    # readmap platform row
    if "sec-blind-spots" not in html.split("readmap")[1][:2000] if "readmap" in html else True:
        html = html.replace(
            '平台 / ETL</td><td><a href="#sec-before-factor">★入模前</a>',
            '平台 / ETL</td><td><a href="#sec-blind-spots">§12a</a> <a href="#sec-before-factor">★入模前</a>',
        )

    # Phase 3 roadmap
    html = html.replace(
        "PREPROC-002~007 · FEAT-001~004 · CH-001~003",
        "PREPROC-002~007 · PREPROC-011~013 · FEAT-001~004 · CH-001~004 · CH-006 · TASK-DC-005",
    )

    return html


def patch_md(md: str) -> str:
    if "## 18. 工业级补盲点" in md:
        md = re.sub(r"\n## 18\. 工业级补盲点.*?(?=\n## |\Z)", "\n", md, flags=re.DOTALL)

    blind_md = """

## 18. 工业级补盲点（评审采纳）

> 详见团队 HTML `§12a`。五条均建议纳入 Layer 2 / CH，不替代 P0。

### 18.1 Permanent ID（PREPROC-011）

- **问题**：`ticker` 变更导致 Track B 序列断裂。
- **做法**：`dim_ticker_map` + `panel_daily.permanent_id`（优先 `composite_figi`）；`GOOG`/`GOOGL` 按股份类别策略，不盲目合并。
- **验收**：`META` 查询可连续拼接 `FB` 历史（同 `permanent_id`）。

### 18.2 退市收益惩罚（PREPROC-012）

- **问题**：末 bar 缺失 + 掩码剔除 → 回测躲过摘牌暴跌。
- **做法**：按退市类型写入 `returns_daily` 最后一跳；`delisting_policy` 元数据（非一律 -100%）。
- **验收**：破产案例回测末笔收益与政策一致；Massive 无 CRSP 表。

### 18.3 动态股本与市值（PREPROC-013）

- **问题**：季报股数 90 天常数 → PE/PB 阶跃。
- **做法**：日频股本优先；`splits` 同步调整历史股数；输出 `pit_shares_out`, `pit_market_cap`。
- **验收**：拆股日前后 `pit_market_cap` 无假跳（除真实市值变动）。

### 18.4 ClickHouse Projections（CH-006）

- 主表 `ORDER BY (trade_date, ticker)`；投影 `ORDER BY (ticker, trade_date)`。
- CH-005（副表/MV）降为备选。

### 18.5 条件码协议（TASK-DC-005）

- `config/tape_condition_filter.yaml`；H/L 仅白名单 regular；异类成交量 → `vol_alt`。
- **禁止**未对照 `condition_codes` 维表硬编码字母。

| ID | 优先级 |
|---|---|
| PREPROC-011 | P1 |
| PREPROC-012 | P1 |
| PREPROC-013 | P1 |
| CH-006 | P1 |
| TASK-DC-005 | P1 |

"""

    if "| PREPROC-010 |" in md and "| PREPROC-011 |" not in md:
        md = md.replace(
            "| PREPROC-010 | 质量监控 job | P1 | 截断/行数/拆股报警 |",
            "| PREPROC-010 | 质量监控 job | P1 | 截断/行数/拆股报警 |\n| PREPROC-011 | `dim_ticker_map` + `permanent_id` | P1 | FB→META 连续；FIGI 策略文档化 |\n| PREPROC-012 | 退市惩罚收益 + `delisting_policy` | P1 | 破产/并购分类型；无静默剔除 |\n| PREPROC-013 | `pit_shares_out` / 动态市值 | P1 | 拆股同步调股本；PE 无 90 天假线性 |",
        )

    # Enhance §4.5
    old_45 = "### 4.5 退市 / 破产收益处理（入模前需定义）"
    if old_45 in md and "PREPROC-012" not in md.split(old_45)[1][:800]:
        md = md.replace(
            "Massive **不**提供 CRSP 退市收益调整表；须在 `returns_daily` 元数据中记录 `delisting_policy`。",
            "Massive **不**提供 CRSP 退市收益调整表；须在 `returns_daily` 元数据中记录 `delisting_policy`。\n\n**工程化（PREPROC-012）**：识别 `last_trading_day`，按事件类型写入惩罚/清算收益；禁止仅依赖 universe 掩码剔除而不记收益。",
        )

    # Enhance tick section
    md = md.replace(
        "**任务**：`PREPROC-008`（P3）：文档化条件码位图 + 样例 pipeline。",
        "**任务**：`PREPROC-008`（P3）+ **`TASK-DC-005`**（P1）：`tape_condition_filter.yaml` 黑白名单 + 1 日金样例；H/L 仅 regular sale。",
    )

    # CH Track A/B
    if "CH-006" not in md and "Track B" in md:
        idx = md.find("### 2.4 ATR × 存储：ClickHouse 与排序键（Track A / B）")
        if idx >= 0:
            insert_at = md.find("\n---", idx)
            if insert_at > idx:
                md = (
                    md[:insert_at]
                    + "\n\n**CH-006（采纳）**：`panel_daily` 使用 **Projection** `p_ticker_timeline`（`ORDER BY (ticker, trade_date)`），避免 11TB 级整表复制。CH-005 副表仅作降级。\n"
                    + md[insert_at:]
                )

    md = md.rstrip() + blind_md
    if "v2.4" in md[:500]:
        md = md.replace(
            "| v2.4 |",
            "| v2.5 |",
            1,
        )
        if "v2.5" in md and "工业级补盲点" not in md.split("修订")[0][-200:]:
            pass
    # version line in header if exists
    md = re.sub(
        r"\*\*版本\*\*：v2\.\d+",
        "**版本**：v2.5",
        md,
        count=1,
    )
    return md


def main():
    html = HTML.read_text(encoding="utf-8")
    html = patch_html(html)
    HTML.write_text(html, encoding="utf-8")
    print("HTML updated, lines:", html.count("\n") + 1)

    if MD.exists():
        md = MD.read_text(encoding="utf-8")
        md = patch_md(md)
        MD.write_text(md, encoding="utf-8")
        print("MD updated")


if __name__ == "__main__":
    main()
