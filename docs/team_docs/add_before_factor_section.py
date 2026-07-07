#!/usr/bin/env python3
"""Add comprehensive pre-factor data processing section to HTML."""
import re
from pathlib import Path

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"

SECTION = r'''
    <section id="sec-before-factor">
      <h2>核心：传入因子引擎之前必须完成的数据处理（Layer 2 全流程）</h2>
      <p class="section-desc">
        <code>cleaned_massive_data/</code> <strong>不能</strong>直接喂 QuantaAlpha / factor_engine。
        因子表达式读的是<strong>已对齐、已复权、已 PiT、已拼成宽表</strong>的输入，即 <code>panel_daily</code>（Parquet 或 ClickHouse）。
        下列步骤均在 <strong>Layer 2</strong> 完成；完成后才调用因子计算。
      </p>

      <div class="box-danger">
        <strong>入模后（不在 Layer 2）：</strong>截面 MAD/Winsorize、Z-Score、Barra/OLS 中性化、IC/IR/Rank IC。
        那些属于 <code>factor_evaluation</code>，在因子<strong>算出来之后</strong>做。
      </div>

      <h3>一、三层数据各自干什么</h3>
<pre>Layer 0  raw_massive_data/     供应商原始（SIP 不复权、REST 部分复权）
    ↓ massive_cleaning_framework.py
Layer 1  cleaned_massive_data/  单源：ticker + align_time + explode（不改 OHLCV）
    ↓ ATR 管道（Stage 0–10，本文）
Layer 2  panel_daily / CH       宽表 (trade_date, ticker) + pit_* + adj_* + 掩码
    ↓ factor_engine 表达式 / QuantaAlpha
因子值序列 → factor_evaluation（去极值、中性化、IC…）</pre>

      <h3>二、为什么 cleaned 还不够（必做 vs 不做）</h3>
      <table>
        <thead><tr><th>问题</th><th>cleaned 状态</th><th>因子入模前必须补做</th></tr></thead>
        <tbody>
          <tr><td>各源仍是独立目录</td><td>无跨源 panel</td><td>Stage 7 merge 成宽表</td></tr>
          <tr><td>SIP 日 K 不复权</td><td>保留 raw OHLCV</td><td>Stage 3 拆股复权（或换 REST 价源）</td></tr>
          <tr><td>财报 align_time=filing_date</td><td>未贴到每个交易日</td><td>Stage 5 PiT + asof backward</td></tr>
          <tr><td>用 period_end 会前视</td><td>字段仍在</td><td>禁止；只用 filing_date</td></tr>
          <tr><td>无每日可交易池</td><td>无 universe</td><td>Stage 2 动态 universe_daily</td></tr>
          <tr><td>缺失 = 无行</td><td>不填 0</td><td>left join 保持 NaN；volume=0 单独语义</td></tr>
          <tr><td>2024 截断财报</td><td>文件仍在盘上</td><td>Stage 0 隔离或 P0 重下</td></tr>
          <tr><td>新闻秒级时间</td><td>published_utc</td><td>Stage 9 → signal_trade_date</td></tr>
        </tbody>
      </table>

      <h3>三、Stage 0–10 逐步做什么（输入 → 输出 → 怎么做）</h3>
      <p>依赖顺序：<strong>0 → 1 → 2 → 3 → 4 → (5 → 6) → 7 → 10</strong>；8/9 按策略插入。PiT 表（5）应在 Panel merge（7）之前独立物化。</p>

      <table class="stage-table">
        <thead>
          <tr>
            <th>阶段</th><th>要处理什么</th><th>输入</th><th>输出表</th><th>具体怎么做</th><th>任务</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>0 门禁</strong></td>
            <td>数据能不能用</td>
            <td>raw 全库</td>
            <td>—</td>
            <td>L1 行数∈{2000,10000}；L2 同比&lt;50%；L3 业务覆盖（如 short 仅 1 日）。P0 未过：基本面 train≤2023-12-31</td>
            <td>TASK-DATA-*<br><a href="#sec-sre">§2c</a></td>
          </tr>
          <tr>
            <td><strong>1 资产过滤</strong></td>
            <td>截面同质性</td>
            <td>all_tickers + ticker_types</td>
            <td><code>security_master</code></td>
            <td><code>type=CS</code>，<code>market=stocks</code>，<code>locale=us</code>；剔 ETF/PFD/WARRANT；REIT/ADR 二次规则；过滤 ticker=<code>NA</code></td>
            <td>PREPROC-001</td>
          </tr>
          <tr>
            <td><strong>2 Universe</strong></td>
            <td>幸存者偏差</td>
            <td>每日 cleaned day_aggs(D)</td>
            <td><code>universe_daily</code></td>
            <td><strong>稀疏</strong>：仅「当日有 bar 的 CS 票」；含退市前历史；禁 2026 存活列表回灌 2008；MA(20)/\$1 放可选 <code>universe_mask</code></td>
            <td>PREPROC-002</td>
          </tr>
          <tr>
            <td><strong>3 复权</strong></td>
            <td>拆股/分红跳跃</td>
            <td>day_aggs + splits (+ dividends)</td>
            <td><code>bars_adjusted_daily</code></td>
            <td><strong>二选一价源：</strong>A) REST 日线（已拆股，无分红） B) SIP×splits。P0 前<strong>禁止</strong>分红复权（dividends 截断）。NVDA/TSLA/KO 单测</td>
            <td>PREPROC-003</td>
          </tr>
          <tr>
            <td><strong>4 收益</strong></td>
            <td>可比收益序列</td>
            <td>bars_adjusted</td>
            <td><code>returns_daily</code></td>
            <td><code>ret_price</code> 必算；<code>ret_total</code> 需 dividends 修好；元数据写清再投资假设；退市政策写入 metadata</td>
            <td>PREPROC-004</td>
          </tr>
          <tr>
            <td><strong>5 PiT 财报</strong></td>
            <td>防前视基本面</td>
            <td>三大报表 cleaned</td>
            <td><code>fundamentals_pit</code></td>
            <td>先 events 表；按 <code>(period_end, timeframe)</code> 分组；取 <code>filing_date ≤ trade_date</code> 最新版；<strong>禁</strong>全样本 keep_latest；可选 shift(1)</td>
            <td>PREPROC-005</td>
          </tr>
          <tr>
            <td><strong>6 LTM/比率</strong></td>
            <td>估值比口径</td>
            <td>pit 表 + 市值</td>
            <td>panel 列 <code>pit_pe</code> 等</td>
            <td>4 季滚动 LTM；PE 亏损→NaN；financials_ratios 仅作校验非 PiT</td>
            <td>PREPROC-006</td>
          </tr>
          <tr>
            <td><strong>7 Panel 物化</strong></td>
            <td>多源变一张宽表</td>
            <td>bars, returns, universe, pit, 可选 news/short</td>
            <td><code>panel_daily</code></td>
            <td>锚点 <code>(trade_date,ticker)</code>；merge 顺序见 <a href="#sec-align">§9.6</a>；逻辑键 <code>trade_date</code> 非混 align_time</td>
            <td>PREPROC-007</td>
          </tr>
          <tr>
            <td><strong>8 分钟/tick</strong></td>
            <td>日内价量</td>
            <td>minute / quotes / trades</td>
            <td>分钟 panel（独立）</td>
            <td>RTH 过滤；<strong>禁止</strong> sum(minute.vol)=day.vol；日频特征仅 T-1 asof 到分钟</td>
            <td>PREPROC-008</td>
          </tr>
          <tr>
            <td><strong>9 新闻/文本</strong></td>
            <td>事件时间映射</td>
            <td>news, risk_factors</td>
            <td><code>news_agg</code> 列</td>
            <td><code>published_utc</code> → 盘后则 <code>signal_trade_date</code>=下一交易日；解析 insights 情感</td>
            <td>PREPROC-009</td>
          </tr>
          <tr>
            <td><strong>10 监控+存储</strong></td>
            <td>可运维</td>
            <td>panel 分区</td>
            <td>Parquet + CH + meta</td>
            <td>行数环比±30%；拆股日 |r|&gt;50% 抽样；主键重复=0；导入 <a href="#sec-clickhouse">§11</a></td>
            <td>PREPROC-010, CH-*</td>
          </tr>
        </tbody>
      </table>

      <h3>四、宽表 merge 顺序（Stage 7 实操）</h3>
      <p>对每个 <code>trade_date</code> 分批或按年处理，<strong>禁止</strong>一次性 merge_asof 扫 11 TB cleaned 树。</p>
<pre>1. bars_adjusted_daily     ← 锚点（稀疏：仅有 bar 的行）
2. LEFT JOIN universe_daily   → in_universe_base
3. LEFT JOIN returns_daily    → ret_price, ret_total（exact）
4. ASOF JOIN fundamentals_pit → pit_* 列（knowledge_date ≤ trade_date）
5. LEFT JOIN financials_ratios / short_volume（exact 或 asof）
6. ASOF JOIN short_interest
7. LEFT JOIN news_agg on signal_trade_date
8. 可选 universe_mask（策略层流动性过滤）</pre>

      <h3>五、按因子类型：最小要完成哪些阶段</h3>
      <table>
        <thead><tr><th>你要算的因子</th><th>必经 Stage</th><th>可跳过</th><th>因子引擎读什么列</th></tr></thead>
        <tbody>
          <tr><td>动量 / 均线 / 波动（纯价量）</td><td>0→2→3→4→7</td><td>1,5,6,9</td><td>adj_close, ret_price, in_universe_base</td></tr>
          <tr><td>估值 / 质量 / 成长（基本面）</td><td>0→1→2→3→4→5→6→7</td><td>8,9</td><td>pit_revenue, pit_roe, pit_pe…</td></tr>
          <tr><td>REST 快速原型</td><td>0→2→7（价源 REST）</td><td>3 若用 REST</td><td>price_source=rest_split_only</td></tr>
          <tr><td>红利 / 总收益</td><td>0→3（含分红）→4→7</td><td>—</td><td>ret_total（dividends P0 后）</td></tr>
          <tr><td>日内 / 分钟</td><td>0→2→8</td><td>5,6 或仅 T-1 日频 asof</td><td>独立 minute panel</td></tr>
          <tr><td>新闻情绪</td><td>0→9→7</td><td>6</td><td>news_count_1d, news_sentiment_last</td></tr>
        </tbody>
      </table>
      <p>场景对照表见 <a href="#sec-scenarios">§3b</a>；各数据源改法见 <a href="#sec-datasets">§1a</a>。</p>

      <h3>六、缺失值与 volume=0 怎么处理</h3>
      <table>
        <thead><tr><th>情况</th><th>表现</th><th>因子入模前处理</th><th>禁止</th></tr></thead>
        <tbody>
          <tr><td>当日无成交</td><td>无 (D,ticker) 行</td><td>稀疏 panel 不生成行；或 left join NaN</td><td>填 0 当收盘价</td></tr>
          <tr><td>有 bar 但 volume=0</td><td>close 有效</td><td>volume 列=0；流动性因子可单独处理</td><td>删行当作退市</td></tr>
          <tr><td>两季报之间</td><td>pit_revenue=NaN</td><td>可选按 ticker PiT ffill（仅过去值）</td><td>bfill</td></tr>
          <tr><td>亏损无 PE</td><td>pit_pe=NaN</td><td>保持 NaN</td><td>填 0 或极大数</td></tr>
          <tr><td>报表科目不适用</td><td>inventories null</td><td>保持 NaN（结构性 null）</td><td>填 0</td></tr>
        </tbody>
      </table>

      <h3>七、防信息泄露（入模前必须遵守）</h3>
      <ul class="checklist">
        <li>基本面：<code>filing_date ≤ trade_date</code>（或更保守 &lt; trade_date / shift(1)）</li>
        <li>禁止用 <code>period_end</code> 作为可见日</li>
        <li>禁止用 cleaned 全历史 <code>keep_latest</code> 修订版回灌</li>
        <li>新闻：禁止用发布当日收盘算「当日」因子 → 用 <code>signal_trade_date</code></li>
        <li>分钟因子：日频门控变量只能来自 <strong>T-1</strong> 交易日</li>
        <li>两套日线（SIP vs REST）不得在同一 panel 混用</li>
        <li>P0 前：含 pit 列的训练截止 <strong>2023-12-31</strong></li>
      </ul>

      <h3>八、传入因子引擎时，输入应长什么样</h3>
      <p>factor_engine / QuantaAlpha 期望的 DataFrame（或 CH 表）<strong>每一行</strong>：</p>
      <table>
        <thead><tr><th>列组</th><th>必备列</th><th>说明</th></tr></thead>
        <tbody>
          <tr><td>键</td><td><code>trade_date</code>, <code>ticker</code></td><td>与表达式里的 symbol/datetime 映射一致</td></tr>
          <tr><td>价量</td><td><code>adj_open/high/low/close</code>, <code>adj_volume</code></td><td>已复权</td></tr>
          <tr><td>收益</td><td><code>ret_price</code></td><td>或由引擎从 adj_close 再算（需一致）</td></tr>
          <tr><td>掩码</td><td><code>in_universe_base</code></td><td>截面回测先 filter =1</td></tr>
          <tr><td>审计</td><td><code>price_source</code>, <code>batch_id</code></td><td>复现与排错</td></tr>
          <tr><td>基本面（若需要）</td><td><code>pit_*</code></td><td>已 asof，非 raw 报表列</td></tr>
        </tbody>
      </table>
      <p>读取方式：① <code>read_parquet(materialized_panel/...)</code> ② <code>SELECT * FROM qs_massive.panel_daily WHERE …</code> ③ <code>data_access/clickhouse_panel.py</code>（待建）</p>

      <h3>九、调用因子引擎之前的最终 Checklist</h3>
      <ul class="checklist">
        <li>Stage 0：截断扫描 PASS（<a href="#sec-accept">§16.4</a> 脚本）</li>
        <li>Stage 2：<code>universe_daily</code> 已生成；2008 池大于「仅存活列表」</li>
        <li>Stage 3：<code>price_source</code> 单一；NVDA 拆股日前后 adj_close 连续</li>
        <li>Stage 5：抽查任意 (ticker, date)，pit 的 knowledge_date ≤ trade_date</li>
        <li>Stage 7：<code>panel_daily</code> 无重复 (trade_date, ticker)；行数与锚点一致</li>
        <li>元数据：<code>adj_method</code>、<code>delisting_policy</code>、<code>reinvest_assumption</code> 已文档化</li>
        <li>CH（若用）：与 Parquet 同分区行数一致（<a href="#sec-clickhouse">§11.10</a>）</li>
        <li>确认未在 Layer 2 做 MAD/Barra（那是 factor_evaluation）</li>
      </ul>

      <h3>十、与代码组件的对应关系</h3>
      <table>
        <thead><tr><th>组件</th><th>在入模前流程中的角色</th><th>现状</th></tr></thead>
        <tbody>
          <tr><td><code>massive_cleaning_framework</code></td><td>Layer 1 清洗</td><td>✅ 已有</td></tr>
          <tr><td><code>build_panel_daily.py</code></td><td>Stage 7 物化</td><td>❌ 待建 FEAT-001</td></tr>
          <tr><td><code>CompositeDataSource</code></td><td>参考 merge_asof 逻辑</td><td>✅ 勿假设 PiT 算子已实现</td></tr>
          <tr><td><code>factor_engine</code></td><td>读 panel 算表达式</td><td>需绑定 Layer 2 列名</td></tr>
          <tr><td><code>factor_evaluation</code></td><td><strong>入模后</strong> IC/中性化</td><td>与 Massive 宽表分离</td></tr>
        </tbody>
      </table>

      <div class="box-info">
        <strong>相关章节：</strong>
        <a href="#sec-align">§9 多源对齐</a> ·
        <a href="#sec-atr">§6 ATR</a> ·
        <a href="#sec-pipeline">§10b 物化管道</a> ·
        <a href="#sec-clickhouse">§11 ClickHouse</a> ·
        <a href="#sec-preproc">§10 阶段索引</a>
      </div>
    </section>
'''


def main():
    html = HTML.read_text(encoding="utf-8")
    block = SECTION.strip() + "\n"

    html = re.sub(
        r'\s*<section id="sec-before-factor">.*?</section>\s*',
        "\n",
        html,
        count=1,
        flags=re.DOTALL,
    )

    # Insert after sec-align (core content before preproc index)
    pat = r'(    <section id="sec-align">.*?</section>\s*)'
    m = re.search(pat, html, flags=re.DOTALL)
    if not m:
        raise SystemExit("sec-align not found")
    html = html[: m.end()] + "\n" + block + html[m.end() :]

    nav = html.split("</nav>")[0]
    if "sec-before-factor" not in nav:
        html = html.replace(
            '<li><a href="#sec-align">9. 异构多源对齐实操（核心）</a></li>\n        <li><a href="#sec-preproc">',
            '<li><a href="#sec-align">9. 异构多源对齐实操</a></li>\n        <li><a href="#sec-before-factor"><strong>★ 因子入模前处理（必读）</strong></a></li>\n        <li><a href="#sec-preproc">',
        )

    # Fix duplicate title in sec-preproc
    html = html.replace(
        "<h2>10. 因子入模前预处理（Layer 2 索引）（Layer 2）</h2>",
        "<h2>10. 因子入模前预处理（Layer 2 阶段索引）</h2>",
    )
    # readmap - quant researcher first link to before-factor
    html = html.replace(
        '量化研究员</td><td><a href="#sec-datasets">§1a</a>',
        '量化研究员</td><td><a href="#sec-before-factor"><strong>★入模前</strong></a> <a href="#sec-datasets">§1a</a>',
    )
    html = html.replace(
        '平台 / ETL</td><td><a href="#sec-pipeline">§10b</a>',
        '平台 / ETL</td><td><a href="#sec-before-factor">★入模前</a> <a href="#sec-pipeline">§10b</a>',
    )

    # header purpose
    html = html.replace(
        "含 P0 补数、ATR、多源对齐",
        "含 <strong>因子入模前 Layer2 全流程</strong>、P0 补数、ATR、多源对齐",
    )
    html = html.replace("版本 v2.2 HTML 综合版", "版本 v2.3 HTML 综合版")

    # Add CSS for stage table if not present
    if ".stage-table" not in html:
        html = html.replace(
            "nav.toc.wide ul { columns: 2; }",
            "nav.toc.wide ul { columns: 2; }\n    table.stage-table td { vertical-align: top; font-size: 13px; }\n    table.stage-table th { font-size: 12px; }",
        )

    HTML.write_text(html, encoding="utf-8")
    print("before-factor section added, lines:", len(html.splitlines()))


if __name__ == "__main__":
    main()
