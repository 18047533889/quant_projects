#!/usr/bin/env python3
"""Merge substantive content from preprocessing MD into the single HTML deliverable."""
from __future__ import annotations

import re
from pathlib import Path

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"
MARKER = "<!-- sync_md_into_html -->"

ENGINE_CHECKLIST = """
    <!-- sync_md_into_html: engine checklist -->
    <section id="sec-engine-checklist">
      <h2><span class="sec-badge">17</span> 上线因子引擎前检查清单（入模闸门）</h2>
      <p class="section-desc">
        与 <a href="#sec-accept">§16 数据补全验收</a>、<a href="#sec-before-factor">★入模前 Stage 0–10</a> 区分：
        本节是<strong>允许调用 factor_engine / QuantaAlpha 之前</strong>必须勾完的闸门。
      </p>
      <h3>17.1 数据完整性</h3>
      <ul class="checklist">
        <li>P0 REST 截断已修复</li>
        <li>截断扫描 PASS（<a href="#sec-sre">§2c</a>）</li>
        <li><code>dividends</code>、<code>all_tickers</code> 非 2000/10000 封顶</li>
      </ul>
      <h3>17.2 证券池与 Universe</h3>
      <ul class="checklist">
        <li><code>security_master.is_cs_universe</code> 生效</li>
        <li><code>universe_daily</code> 按日动态生成（禁 2026 存活列表回灌历史）</li>
        <li>退市/破产收益政策写入 <code>returns_daily</code> 元数据</li>
      </ul>
      <h3>17.3 价量</h3>
      <ul class="checklist">
        <li><code>price_source</code> 单一且文档化（SIP 或 REST，禁止混 join）</li>
        <li>SIP 路径：日 K + <strong>分钟算子路径</strong> 已拆股复权（<a href="#sec-operator-adj">§3.2</a>、<code>fact_bars_adjusted_*</code>）</li>
        <li>未用 <code>sum(minute.volume)==day.volume</code> 误判数据损坏</li>
      </ul>
      <h3>17.4 基本面</h3>
      <ul class="checklist">
        <li>全部 asof on <code>filing_date</code> / <code>knowledge_ts</code>（禁 <code>period_end</code> 贴日）</li>
        <li>PiT 非 cleaned <code>keep_latest</code> 全历史（算法见 <a href="#sec-pit-algorithm">§9.5</a>）</li>
        <li>CIK/ticker 去重策略已定义</li>
        <li>LTM/PE/支付率口径文档化（<a href="#sec-stage6-ltm">§9.6 LTM</a>）</li>
      </ul>
      <h3>17.5 Panel 与泄露</h3>
      <ul class="checklist">
        <li><code>panel_daily</code> 物化分区可增量</li>
        <li>缺失 ≠ 0（left join 保持 NaN）</li>
        <li>信号/收益时间对齐已定义（<a href="#sec-returns-policy">§9.4 对齐表</a>）</li>
        <li>新闻/事件已映射 <code>signal_trade_date</code>（若使用）</li>
      </ul>
      <h3>17.6 明确不在 Layer 2</h3>
      <ul class="checklist">
        <li>MAD / Winsorize / Z-Score / Barra / IC·IR 仅在 <code>factor_evaluation</code></li>
      </ul>
    </section>
"""

RETURNS_LTM = """
    <!-- sync_md_into_html: returns + ltm -->
    <section id="sec-returns-policy">
      <h2><span class="sec-badge">9.4</span> 收益率序列与再投资政策（Stage 4）</h2>
      <table class="table-schema">
        <thead><tr><th>类型</th><th>定义</th><th>用途</th></tr></thead>
        <tbody>
          <tr><td><strong>价格收益</strong></td><td>复权价 close-to-close；<strong>不含</strong>常规现金分红</td><td>动量、技术、微观</td></tr>
          <tr><td><strong>总收益</strong></td><td>含分红再投资（CRSP 概念）</td><td>长期复利、红利因子</td></tr>
        </tbody>
      </table>
      <p><strong>价格收益（日）：</strong> <code>ret_price[t] = P_adj[t] / P_adj[t-1] - 1</code>。
        特殊现金分红（extraordinary）是否从价格收益剔除：核对 Massive <code>distribution_type</code>。</p>
      <p><strong>总收益：</strong> Massive 无 <code>cumfacpr</code> → Layer 2 用 <code>splits</code> + <code>dividends</code> 维护累计表；小分红可近似
        <code>ret_total ≈ ret_price + D_t / P_adj[t-1]</code>，生产应用严格累计因子表。</p>
      <table class="table-schema">
        <thead><tr><th>再投资假设（二选一，写入元数据）</th><th>说明</th></tr></thead>
        <tbody>
          <tr><td><code>div_reinvest_exdate</code></td><td>除息日再投资（FF 新版 CIZ）</td></tr>
          <tr><td><code>div_reinvest_month_end</code></td><td>月末再投资（旧 FIZ）</td></tr>
        </tbody>
      </table>
      <p><strong>输出 <code>returns_daily</code>：</strong> <code>ret_price</code>、<code>ret_total</code>（可空）、<code>div_reinvest_policy</code>。
        任务 <code>PREPROC-004</code>（P1）。</p>
      <h3>收益与信号对齐（防泄露）</h3>
      <table class="table-schema">
        <thead><tr><th>信号所用数据</th><th>收益区间</th><th>常见做法</th></tr></thead>
        <tbody>
          <tr><td>T 收盘后算出的因子</td><td>T+1 open→close 或 T+1 c2c</td><td><code>shift(1)</code> 或 next_td</td></tr>
          <tr><td>T 开盘前已知的基本面</td><td>T close-to-close</td><td><code>filing_date &lt; T</code> 09:30 ET</td></tr>
        </tbody>
      </table>
    </section>

    <section id="sec-stage6-ltm">
      <h2><span class="sec-badge">9.6</span> LTM 与估值比率（Stage 6）</h2>
      <p class="section-desc">在 <strong>PiT 对齐后</strong> 的序列上计算；仍是特征工程，不是截面 MAD。</p>
      <ol>
        <li>对每个 <code>(ticker, trade_date T)</code>，取 PiT 可见的最近 4 个不重叠 <code>quarterly</code> <code>period_end</code>；</li>
        <li>对 <code>revenue</code>、<code>net_income</code> 等 sum → <code>ltm_*</code>；</li>
        <li>不足 4 季：LTM 标 NaN 或降级 YTD（须在元数据声明）。</li>
      </ol>
      <table class="table-schema">
        <thead><tr><th>指标</th><th>分子</th><th>分母</th><th>约束</th></tr></thead>
        <tbody>
          <tr><td>PE</td><td>市值 = <code>close_adj * shares_out</code></td><td><code>ltm_eps</code></td><td>分母 ≤0 → PE=NaN</td></tr>
          <tr><td>PB</td><td>市值</td><td>最近 PiT 季 <code>total_equity</code></td><td>—</td></tr>
          <tr><td>ROA</td><td><code>ltm_net_income</code></td><td>同期 <code>total_assets</code></td><td>—</td></tr>
          <tr><td>股息支付率</td><td>DPS / <code>cash_div</code></td><td><strong>匹配财年</strong>的 LTM 净利润</td><td>禁止错季 EPS</td></tr>
        </tbody>
      </table>
      <p><code>shares_out</code>：优先 <code>stocks_floats</code> asof；否则日 K 隐含或财报股数。
        <code>financials_ratios</code> 仅作校验；生产因子推荐 <strong>自算 LTM + 市值</strong>（<code>PREPROC-006</code> P1）。</p>
    </section>
"""

STAGE8 = """
      <!-- sync_md_into_html: stage8 -->
      <h3 id="sec-stage8-minute">8. Stage 8：分钟线与 tick（汇总）</h3>
      <p>分钟复权定稿见 <a href="#sec-operator-adj">§3.2</a>；与日 K 关系见 <a href="#sec-quality">§2b §4.0.3</a>。</p>
      <table class="table-schema">
        <thead><tr><th>实证（2024-06-03）</th><th>数值</th></tr></thead>
        <tbody>
          <tr><td><code>sum(minute.volume)==day.volume</code></td><td><strong>1.9%</strong> ticker 完全一致</td></tr>
          <tr><td>误差 &lt;1%</td><td><strong>11.7%</strong></td></tr>
          <tr><td>median ratio sum(min)/day</td><td><strong>0.913</strong></td></tr>
        </tbody>
      </table>
      <p><strong>规则：</strong> 日频因子 → <code>fact_bars_adjusted_daily</code>；日内/<code>ts_*</code> → <code>fact_bars_adjusted_minute</code>；<strong>禁止</strong>用 minute 加总校验 day。</p>
      <table class="table-schema">
        <thead><tr><th>会话</th><th>美东</th><th>UTC（标准时示例）</th></tr></thead>
        <tbody>
          <tr><td>RTH</td><td>09:30–16:00 ET</td><td>14:30–21:00 UTC</td></tr>
          <tr><td>扩展</td><td>04:00–20:00 ET</td><td>按日历转换</td></tr>
        </tbody>
      </table>
      <p>AAPL 样例：分钟覆盖 UTC 08:00–23:59 → RTH 因子须先 filter。<code>window_start</code> 日 K 界为 <strong>04:00 UTC</strong>，不是 09:30 开盘。</p>
      <table class="table-schema">
        <thead><tr><th>rollup 指标</th><th>聚合</th><th>注意</th></tr></thead>
        <tbody>
          <tr><td>close</td><td>session 内 last(minute.close)</td><td>仍可能 ≠ day close</td></tr>
          <tr><td>volume</td><td>sum</td><td>与 day 偏差大</td></tr>
          <tr><td>VWAP</td><td>sum(price×vol)/sum(vol)</td><td>自定义</td></tr>
        </tbody>
      </table>
      <p><strong>tick：</strong> quotes+trades ~10 TB；cleaned 默认跳过；<code>conditions</code> ~24% null 不等于坏数据；
        过滤须查 <code>market_operations/condition_codes</code>；<code>PREPROC-008</code>（P3）文档化条件码。</p>
"""

FATAL_ERRORS = """
      <!-- sync_md_into_html: fatal errors -->
      <h3 id="sec-fatal-errors">1.3 常见致命错误（反模式，11 条）</h3>
      <table class="table-schema">
        <thead><tr><th>反模式</th><th>后果</th><th>正确做法</th></tr></thead>
        <tbody>
          <tr><td>用 <code>period_end</code> 贴财报到交易日</td><td>前视偏差</td><td><code>filing_date</code> + asof backward</td></tr>
          <tr><td>SIP 与 REST 日线混用</td><td>假动量、假估值</td><td>二选一，元数据 <code>price_source</code></td></tr>
          <tr><td>2026 存活 ticker 回灌 2008</td><td>幸存者偏差</td><td><code>universe_daily(D)</code> 动态</td></tr>
          <tr><td>未复权 SIP 跑多年动量</td><td>拆股假信号</td><td>复权或 REST；分钟算子见 §3.2</td></tr>
          <tr><td><code>sum(minute.vol)==day.vol</code> 对账</td><td>误判损坏</td><td>接受口径差</td></tr>
          <tr><td>cleaned 基本面 keep_latest 回测</td><td>重述前视</td><td>PiT 按 T 切片</td></tr>
          <tr><td>无 bar 填 0</td><td>假低价、假零盈利</td><td>left join 保持 NaN</td></tr>
          <tr><td>截断文件入 Panel</td><td>2024 因子全错</td><td>附录 H / §2c 门禁</td></tr>
          <tr><td><code>all_tickers</code> 稠密 T×N 网格</td><td>内存爆炸</td><td>稀疏锚点：从 day_aggs 出发</td></tr>
          <tr><td>MA(20)/\$1 当数据层必须</td><td>过度剔除</td><td>策略级 <code>universe_mask</code></td></tr>
          <tr><td>仅行数≠2000/10000 即放行</td><td>漏检略少未封顶</td><td>跨年行数 + settlement 审计</td></tr>
        </tbody>
      </table>
"""

JOIN_TABLE = """
<h3 id="sec-join-contract">D.3.1 完整 Join 规则表（Stage 7）</h3>
<table class="table-schema">
  <thead><tr><th>源</th><th>类型</th><th>键</th><th>方向</th><th>备注</th></tr></thead>
  <tbody>
    <tr><td><code>bars_adjusted</code></td><td>anchor</td><td>trade_date, ticker</td><td>—</td><td>价量锚点</td></tr>
    <tr><td><code>universe_daily</code></td><td>left</td><td>同上</td><td>—</td><td>过滤 <code>in_universe</code></td></tr>
    <tr><td><code>returns_daily</code></td><td>left</td><td>同上</td><td>—</td><td>—</td></tr>
    <tr><td>fundamentals PiT</td><td><strong>asof</strong></td><td>ticker</td><td>backward on filing</td><td>见 <a href="#sec-pit-algorithm">§9.5</a></td></tr>
    <tr><td><code>financials_ratios</code></td><td>asof</td><td>ticker</td><td>backward</td><td>非 PiT，作校验</td></tr>
    <tr><td><code>short_interest</code></td><td>asof</td><td>ticker</td><td>backward</td><td>滞后</td></tr>
    <tr><td><code>short_volume</code></td><td>asof/exact</td><td>ticker</td><td>backward</td><td>—</td></tr>
    <tr><td><code>dividends</code>/<code>splits</code></td><td>事件</td><td>ticker, date</td><td>—</td><td>复权用，不直接特征</td></tr>
    <tr><td><code>news</code></td><td>asof + map</td><td>ticker</td><td>backward → next_td</td><td>Stage 9</td></tr>
    <tr><td><code>daily_market_summary</code></td><td>—</td><td>—</td><td><strong>禁止混 join</strong></td><td>与 SIP 二选一</td></tr>
  </tbody>
</table>
"""

PIT_ALGO = """
      <!-- sync_md_into_html: pit -->
      <h3 id="sec-pit-algorithm">C.1.5 PiT 选股算法（可执行）</h3>
      <p>Gemini 写法 <code>max(filing_date ≤ t)</code> <strong>不完整</strong>：须按 <code>(period_end, timeframe)</code> 分组。</p>
      <p><code>Visible[i,t](q) = argmax filing_date ≤ t</code>，约束固定 <code>(i, q, timeframe)</code>。</p>
      <pre>def pit_fundamentals_at(T, fund_stream, ticker):
    # fund_stream: 单 ticker 按 filing_date 排序
    visible = fund_stream[fund_stream["filing_date"] &lt; T]
    if visible.empty:
        return None
    return visible.sort_values("filing_date").groupby(
        ["period_end", "timeframe"], as_index=False
    ).tail(1)</pre>
      <div class="box-danger">
        <strong>禁止</strong> 使用 cleaned 默认 <code>keep_latest_by_align_time</code> 全历史做回测。
        同 <code>period_end</code> 多 <code>filing_date</code>（重述）：在 t 只能用当时可见版本。
      </div>
      <p><strong>重述压力测试：</strong> 1999 回测若读到 2001 Enron 重述后数据 → 虚假预测破产。Massive 无 <code>revision_date</code>。</p>
"""

SCENARIOS_EXTRA = """
          <tr><td>红利 / 总收益因子</td><td>⚠️</td><td>Stage 3 分红口径 + <a href="#sec-returns-policy">§9.4</a> total return</td><td>再投资假设固定一种</td></tr>
          <tr><td>REST 快速动量（不跑 SIP 复权）</td><td>⚠️</td><td>0→2→7；价源 REST</td><td>与 SIP 路径二选一</td></tr>
          <tr><td>多因子评估入库</td><td>⚠️</td><td><strong>先完成 Stage 7 panel</strong></td><td>MAD/Barra/IC 在 factor_evaluation</td></tr>
"""

GEMINI_QUICK = """
      <h3 id="sec-gemini-quick">Gemini 七模块速查（对照 §12 详表）</h3>
      <table class="table-schema">
        <thead><tr><th>模块</th><th>核心主张</th><th>采纳</th><th>本手册落点</th></tr></thead>
        <tbody>
          <tr><td>1 P0 熔断</td><td>删 .ok、max_pages=None、停 cron</td><td>✅</td><td><a href="#sec-p0">§2</a></td></tr>
          <tr><td>2 截断扫描</td><td>跨年同比 &lt;50%</td><td>✅</td><td><a href="#sec-sre">§2c</a></td></tr>
          <tr><td>3 复权</td><td>拆股因子表</td><td>✅ 仅拆股至分红修好</td><td><a href="#sec-empirical">§1b</a>、PREPROC-003</td></tr>
          <tr><td>4 PiT</td><td>filing_date asof</td><td>✅ 按 period 分组</td><td><a href="#sec-pit-algorithm">§9.5</a></td></tr>
          <tr><td>5 Panel</td><td>稀疏锚点</td><td>✅ 禁稠密网格</td><td><a href="#sec-align">§9</a></td></tr>
          <tr><td>6 CH</td><td>物化宽表</td><td>✅ Track A/B</td><td><a href="#sec-clickhouse">§11</a></td></tr>
          <tr><td>7 任务分工</td><td>PREPROC 编号</td><td>✅</td><td><a href="#sec-tasks">§14</a></td></tr>
        </tbody>
      </table>
"""


def has_section(html: str, sec_id: str) -> bool:
    return f'id="{sec_id}"' in html and f'<section id="{sec_id}"' in html


def inject_once(html: str, sec_id: str, content: str, after_pat: str) -> str:
    if has_section(html, sec_id):
        return html
    m = re.search(after_pat, html, re.DOTALL)
    if not m:
        print(f"WARN: pattern not found for {marker}")
        return html
    pos = m.end()
    return html[:pos] + content + html[pos:]


def main():
    html = HTML.read_text(encoding="utf-8")

    # Single deliverable banner
    html = re.sub(
        r"(<p class=\"meta\">)(.*?)(</p>)",
        r"\1版本 v3.6 单文件交付（MD v2.6 已并入）· 2026-06-03 · 数据扫描截止 2026-03-05 · "
        r"<strong>仅以本 HTML 为准</strong>，姊妹 MD 为历史稿\3",
        html,
        count=1,
        flags=re.DOTALL,
    )

    # Remove external MD dependency in sec-preproc
    html = html.replace(
        "完整规范见 <strong>v2.4</strong>\n"
        "        <a href=\"美股原始数据预处理标准方案_因子入模前.md\">美股原始数据预处理标准方案_因子入模前.md</a>\n"
        "       （§2 ATR + Stage 0–10 + 附录 A）。",
        "完整规范已并入<strong>本 HTML</strong>（原 MD v2.6：§2 ATR + Stage 0–10 + 附录 A/C/D + §17 闸门）。",
    )

    # Broken §21.3 link
    html = html.replace(
        '与 <a href="美股原始数据预处理标准方案_因子入模前.md">预处理标准方案 §21.3</a>。',
        '与 <a href="#sec-engine-checklist">§17 入模闸门</a>、<a href="#sec-deep-research">§12f</a> 自检表。',
    )

    # Engine checklist before sec-preproc
    html = inject_once(
        html,
        "sec-engine-checklist",
        ENGINE_CHECKLIST,
        r"</section>\s*<!-- 6\. Preproc -->",
    )

    # Returns + LTM before sec-preproc (after before-factor if no engine section yet)
    anchor = (
        r'<section id="sec-engine-checklist">.*?</section>\s*'
        if has_section(html, "sec-engine-checklist")
        else r"</section>\s*<!-- 6\. Preproc -->"
    )
    html = inject_once(html, "sec-returns-policy", RETURNS_LTM, anchor)

    # Stage 8 under sec-p1 after operator-adj block
    if not has_section(html, "sec-stage8-minute"):
        html = inject_once(
            html,
            "sec-stage8-minute",
            STAGE8,
            r'(<h3 id="sec-operator-adj">.*?</table>\s*)',
        )

    # Fatal errors in sec-atr after 2.7 if present
    if not has_section(html, "sec-fatal-errors"):
        if "2.7 ATR 反模式" in html:
            html = inject_once(html, "sec-fatal-errors", FATAL_ERRORS, r"2\.7 ATR 反模式.*?</table>\s*")
        else:
            html = inject_once(html, "sec-fatal-errors", FATAL_ERRORS, r'<section id="sec-atr">')

    # Join table in sec-align before D.4
    if 'id="sec-join-contract"' not in html:
        html = html.replace(
            "<h3>D.4 Step 5：三种 Join 方式（按源选）</h3>",
            JOIN_TABLE + "\n<h3>D.4 Step 5：三种 Join 方式（按源选）</h3>",
            1,
        )

    # PiT in sec-fields after C.1 heading
    if 'id="sec-pit-algorithm"' not in html:
        html = inject_once(
            html,
            "sec-pit-algorithm",
            PIT_ALGO,
            r"(<h3>C\.1[^<]*</h3>\s*)",
        )

    # Scenarios extra rows
    if "多因子评估入库" not in html:
        html = html.replace(
            "<tr><td>读 CH 宽表挖因子</td>",
            SCENARIOS_EXTRA + "\n          <tr><td>读 CH 宽表挖因子</td>",
            1,
        )

    # Gemini quick table at start of sec-gemini
    if 'id="sec-gemini-quick"' not in html:
        html = inject_once(
            html,
            "sec-gemini-quick",
            GEMINI_QUICK,
            r'(<section id="sec-gemini">\s*<h2[^>]*>.*?</h2>\s*)',
        )

    # Readmap: engine checklist
    if "sec-engine-checklist" not in html.split("sec-readmap")[1][:2000] if "sec-readmap" in html else "":
        html = html.replace(
            '<a href="#sec-before-factor"><strong>★入模前</strong></a>',
            '<a href="#sec-engine-checklist"><strong>§17 闸门</strong></a> '
            '<a href="#sec-before-factor"><strong>★入模前</strong></a>',
            1,
        )

    # Cheatsheet: self-contained
    html = html.replace(
        "<tr><td>字段有哪些</td><td>附录 F、数据字典 md</td></tr>",
        "<tr><td>字段有哪些</td><td><a href=\"#sec-fields\">§7 附录 C</a>、<a href=\"#sec-data-catalog\">数据手册</a></td></tr>"
        "<tr><td>能否上因子引擎</td><td><a href=\"#sec-engine-checklist\">§17 闸门</a></td></tr>"
        "<tr><td>收益率/LTM 口径</td><td><a href=\"#sec-returns-policy\">§9.4</a>、<a href=\"#sec-stage6-ltm\">§9.6</a></td></tr>",
    )

    # Footer single deliverable
    html = html.replace(
        "Massive数据治理与改进行动清单.html v3.5 分钟算子须复权（§3.2）+ 全库数据手册",
        "Massive数据治理与改进行动清单.html v3.6 单文件交付（MD 已并入）",
    )
    html = html.replace(
        "Massive数据治理与改进行动清单.html v3.2 终审完整版",
        "Massive数据治理与改进行动清单.html v3.6 单文件交付（MD 已并入）",
    )

    HTML.write_text(html, encoding="utf-8")
    print("synced, lines:", len(html.splitlines()))


if __name__ == "__main__":
    main()
