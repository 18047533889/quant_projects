#!/usr/bin/env python3
"""Round-3 blind spots + edge-case governance into HTML + MD."""
import re
from pathlib import Path

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"
MD = Path(__file__).resolve().parent / "美股原始数据预处理标准方案_因子入模前.md"

ROUND3 = r'''
    <section id="sec-blind-spots-3">
      <h2><span class="sec-badge">12c</span> 三轮补盲点（Feature Store 防腐 · 多频对齐）</h2>
      <p class="section-desc">
        对照 <em>Journal of Finance</em> / <em>RFS</em> 资产定价清洗标准与买方 Feature Store 实践，在物化落盘 / 导入 CH、传入
        <code>QuantaAlpha</code> / <code>factor_engine</code> <strong>前一秒</strong> 封板。
        与 <a href="#sec-blind-spots">§12a</a>、<a href="#sec-blind-spots-2">§12b</a> 正交。
      </p>

      <div class="box-warn">
        <strong>任务 ID 对照（避免与 §12b 混淆）：</strong>
        <code>ENG-011</code>、<code>PREPROC-014</code> 见 §12b；
        <code>PREPROC-015</code>（§12b）= 股本双轨阶梯；
        本节新增 <code>PREPROC-017</code>（拆股/分红<strong>成交量</strong>非对称）、
        <code>PREPROC-021</code>（Mask-First 掩码前置）、
        <code>PREPROC-022</code>（分钟级 acceptance 错位）；
        <code>PREPROC-016</code> 仍为 §12b 新闻去重。
      </div>

      <h3>盲点十一：代码回收与实体 ID（强化 PREPROC-011 / PREPROC-014）</h3>
      <p><strong>采纳。</strong> 纯 <code>ticker</code> 作跨源主键时，<code>APP</code> 等回收代码会把 American Apparel 时代价量/基本面吞进 AppLovin 长窗，产生<strong>僵尸实体污染</strong>。</p>
      <table>
        <thead><tr><th>项</th><th>规范</th></tr></thead>
        <tbody>
          <tr><td>实体底座</td><td><code>permanent_id</code>（优先 <code>composite_figi</code>，缺失 <code>cik</code>+交易所或自建 <code>sec_id</code>）+ <code>valid_from</code> / <code>valid_to</code></td></tr>
          <tr><td>字符归一</td><td><code>ticker_normalized</code>：剥离 <code>/ . -</code> 与空格（<code>BRK/B</code>、<code>BRK.B</code> → <code>BRKB</code>），见 <a href="#sec-blind-spots-2">§12b PREPROC-014</a></td></tr>
          <tr><td>禁止</td><td>合并 <code>GOOG</code>/<code>GOOGL</code> 等不同股份类别；Track B 经 map asof 拉历史代码</td></tr>
        </tbody>
      </table>
      <p><strong>任务：</strong> <code>PREPROC-011</code> + <code>PREPROC-014</code>（P1）。</p>

      <h3>盲点十二：Mask-First 掩码前置（PREPROC-021）</h3>
      <p><strong>采纳。</strong> 宽表拼完再 <code>filter(universe_mask)</code> 无法阻止 <code>ts_mean(close,20)</code> 等算子滑动窗口<strong>上游污染（Upstream Contamination）</strong>：停牌/截断脏价仍留在 buffer，后续 20 日 IC 虚高。</p>
      <ul>
        <li>Layer 2 生成 <code>compute_mask = universe_mask &amp; price_valid &amp; …</code>，与 OHLCV 同批落盘</li>
        <li>因子引擎：时序算子必须接收 <code>mask</code>；掩码外输入<strong>不得</strong>进入滑动缓存，输出强制 <code>NaN</code></li>
        <li>元数据：<code>mask_first_propagation=true</code>（QuantaAlpha 白盒公式层契约）</li>
      </ul>
      <p><strong>任务：</strong> <code>PREPROC-021</code>（P1）— 量化研究组 + 数据工程联合验收。</p>

      <h3>盲点十三：拆股/分红成交量非对称缩放（PREPROC-017）</h3>
      <p><strong>采纳。</strong> 与 §12b 股本双轨（<code>PREPROC-015</code>）并列：复权时<strong>拆股</strong>须价格×因子、成交量÷因子；<strong>现金分红</strong>只调价格类字段，<strong>成交量/换手率/笔数保持真值</strong>。</p>
      <table class="table-schema">
        <thead><tr><th>事件源</th><th>价格 OHLC</th><th>volume / transactions</th></tr></thead>
        <tbody>
          <tr><td><code>corporate_actions/splits</code></td><td>× <code>historical_adjustment_factor</code></td><td>÷ 同因子（名义成交额守恒）</td></tr>
          <tr><td><code>corporate_actions/dividends</code></td><td>× 分红调整因子（全收益轨）</td><td><strong>禁止</strong>乘除</td></tr>
        </tbody>
      </table>
      <p><strong>任务：</strong> <code>PREPROC-017</code>（P1，与 <code>PREPROC-003</code> SIP 复权同批验收）— 数据工程组。</p>

      <h3>盲点十四：分钟级日内披露错位（PREPROC-022）</h3>
      <p><strong>采纳。</strong> 日频用 <code>filing_date</code> asof 足够；<strong>1min</strong> 管道若整日广播，会在 09:30–11:14 提前获知 11:15 披露的 8-K/财报，日内未来函数。</p>
      <ul>
        <li>离散事件流增加 <code>knowledge_ts_utc</code>：优先 <code>sec_edgar_index</code> 的 <code>acceptance_time</code>（秒级，转 UTC）；缺失则 <code>published_utc</code>（新闻）；再缺失则 <code>filing_date T+1 00:00 UTC</code> 保守上界</li>
        <li>分钟网格：<code>merge_asof(direction='backward')</code> 在<strong>分钟/秒</strong>轴挂载，例 11:15:23 披露 → 最早 11:16:00 bar</li>
        <li><strong>禁止</strong>将日频 discrete 基本面平铺到当日盘前/开盘分钟</li>
      </ul>
      <p><strong>勘误：</strong> Massive 财报 raw <strong>无</strong> EDGAR acceptance 字段；<code>PREPROC-022</code> 依赖 <code>sec_edgar_index</code> 补全或外部 SEC 索引 join（P0 后启用）。</p>
      <p><strong>任务：</strong> <code>PREPROC-022</code>（P1）— 基础架构组。</p>

      <h3>三轮任务登记</h3>
      <table>
        <thead><tr><th>ID</th><th>阶段</th><th>摘要</th><th>负责人</th><th>验收红线</th></tr></thead>
        <tbody>
          <tr><td>PREPROC-011/014</td><td>Layer 1.5→2</td><td>实体 ID + ticker 归一（见 §12a/12b）</td><td>基础架构组</td><td><code>BRKB</code> 跨源无断链；无代码回收串线</td></tr>
          <tr><td>PREPROC-017</td><td>Layer 2 复权</td><td>拆股对偶缩放 vs 分红只调价量</td><td>数据工程组</td><td>派息后历史 vol 序列无乘除形变</td></tr>
          <tr><td>PREPROC-021</td><td>Layer 2 + 引擎契约</td><td>Mask-First 时序算子</td><td>量化研究组</td><td>停牌日价格不进入 rolling buffer</td></tr>
          <tr><td>PREPROC-022</td><td>Layer 2 多频</td><td>acceptance 秒级 asof → 分钟 bar</td><td>基础架构组</td><td>无盘中披露前视；金样例 8-K 11:15 测例</td></tr>
        </tbody>
      </table>
      <div class="box-info">
        已追加 <a href="#sec-tasks">§14 任务总表</a>；折叠代码见 <a href="#sec-code-round3">§10c.15</a>。
      </div>
    </section>
'''

ROUND4 = r'''
    <section id="sec-blind-spots-4">
      <h2><span class="sec-badge">12d</span> 极端异常清洗（熔断 · 停牌 · 仙股 · IPO · 幻影成交）</h2>
      <p class="section-desc">
        Layer 1.5 / Layer 2 在传入因子引擎前的<strong>最后一道微观结构防线</strong>：大盘熔断、个股 LULD/监管停牌、仙股复权负数、IPO 延迟开盘、休市幻影行。
      </p>

      <h3>极端一：大盘熔断 MWCB（ENG-012）</h3>
      <p><strong>采纳。</strong> 2020-03 等日 Level 1/2 全市场暂停 15 分钟，机械 ffill 会制造伪「零波动横盘」，污染日内波动率/Amihud。</p>
      <ul>
        <li>维表 <code>fact_market_halt_events</code>：登记 UTC 起止</li>
        <li>分钟 bar：允许价 ffill 昨收、<code>volume=0</code>，但必须 <code>is_market_halt=1</code></li>
        <li>时序算子：滚动窗口<strong>跳过</strong> halt 分钟（与 PREPROC-021 掩码联动）</li>
      </ul>
      <p><strong>任务：</strong> <code>ENG-012</code>（P1）— 数据工程组。</p>

      <h3>极端二：休市日与幻影成交 Ghost Trades（ENG-013）</h3>
      <p><strong>采纳。</strong> 周末/法定休市/早于 ET 04:00 的测试成交行会破坏日历效应与周频聚合。</p>
      <ul>
        <li>由全量 SIP 日 K 反推官方 \(\mathcal{T}\)（真值交易日历），持久化 <code>dim_calendar</code></li>
        <li>落盘前 <strong>Hard Delete</strong>：<code>trade_date ∉ 𝒯</code> 或非法时段行</li>
        <li>补全 <code>market_holidays</code> 历史（当前仅 2026–2027 不足）</li>
      </ul>
      <p><strong>任务：</strong> <code>ENG-013</code>（P1）— 平台 SRE 组。</p>

      <h3>极端三：个股 LULD / 监管停牌（PREPROC-018）</h3>
      <p><strong>采纳。</strong> 停牌须 reindex 补 bar：OHLC ffill 末价，<code>volume=0</code>，<code>is_ticker_halt=1</code>；RTH 累计停牌 &gt;60min → 当日 <code>universe_mask=False</code>（防生存者偏见「虚拟逃跌」）。</p>
      <p><strong>任务：</strong> <code>PREPROC-018</code>（P1）— 量化研究组。</p>

      <h3>极端四：仙股复权价格底座（PREPROC-019）</h3>
      <p><strong>采纳。</strong> 多次合股后复权价可 \(\le 0\) 或近零，触发 <code>close/open</code> 除零与 CH 索引异常。</p>
      <ul>
        <li>持久化前：<code>adj_close &lt;= 0</code> 或 \(&lt; 10^{-4}\) → floor clamp（如 <code>1e-5</code>）+ <code>price_clamped_flag=1</code></li>
        <li>审计：合股密集 ticker 清单进 QA 日报</li>
      </ul>
      <p><strong>任务：</strong> <code>PREPROC-019</code>（P1）— 基础架构组。</p>

      <h3>极端五：IPO 首日延迟开盘 Opening Cross（PREPROC-020）</h3>
      <p><strong>采纳。</strong> 09:30–首笔成交前<strong>无 bar</strong>，禁止 ffill「昨收」（新股无昨日）；须 <code>is_ipo_pre_open=1</code>，时序算子冷启动从首笔有效成交起算。</p>
      <ul>
        <li>锁定 <code>listing_trade_date</code>（<code>ipos_all</code> + 首 bar 交叉验证）</li>
        <li>Opening Cross 前分钟：价量不填充或仅标记，不参与 rolling</li>
      </ul>
      <p><strong>任务：</strong> <code>PREPROC-020</code>（P1）— 基础架构组。</p>

      <h3>极端异常任务登记</h3>
      <table>
        <thead><tr><th>ID</th><th>阶段</th><th>摘要</th><th>负责人</th><th>验收红线</th></tr></thead>
        <tbody>
          <tr><td>ENG-012</td><td>Layer 1.5</td><td>MWCB 事件表 + <code>is_market_halt</code></td><td>数据工程组</td><td>2020-03 熔断日滚动 vol 无伪平稳</td></tr>
          <tr><td>ENG-013</td><td>Layer 1.5</td><td>𝒯 硬滤 + Ghost 行删除</td><td>平台 SRE</td><td>非 𝒯 日期零行入库 CH</td></tr>
          <tr><td>PREPROC-018</td><td>Layer 2 分钟</td><td>LULD/监管停牌重构 + universe 联动</td><td>量化研究组</td><td>长停牌日 mask=False</td></tr>
          <tr><td>PREPROC-019</td><td>Layer 2</td><td>复权价 floor clamp</td><td>基础架构组</td><td>面板无 \(P_{adj}\le 0\)</td></tr>
          <tr><td>PREPROC-020</td><td>Layer 2 分钟</td><td>IPO 预开盘标签 + 冷启动</td><td>基础架构组</td><td>首日 09:30–open 无假 ffill</td></tr>
        </tbody>
      </table>
      <div class="box-info">
        五条已追加 <a href="#sec-tasks">§14</a>；与 §12c 一并构成<strong>入模前封板终审清单</strong>。代码 <a href="#sec-code-round3">§10c.15</a>。
      </div>
    </section>
'''

CODE_R3 = r'''
      <h3 id="sec-code-round3">10c.15 三轮补盲点 + 极端异常 · 折叠代码</h3>

      <details class="code-fold">
        <summary>PREPROC-017 · 拆股/分红成交量非对称（<code>adjust_bars.py</code> 片段）</summary>
        <pre><code>def apply_split(price, volume, factor):
    return price * factor, volume / factor  # factor = split_from/split_to 口径与 PREPROC-003 一致

def apply_dividend_price_only(price, volume, div_factor):
    return price * div_factor, volume  # volume 不变

# panel 列: adj_* 价格轨; adj_volume 仅吃 split 链; metadata: volume_adj_policy=split_only</code></pre>
      </details>

      <details class="code-fold">
        <summary>PREPROC-021 · Mask-First 契约（面板列 + 引擎伪代码）</summary>
        <pre><code># panel_daily 追加:
#   universe_mask UInt8
#   compute_mask UInt8  -- universe_mask &amp; has_bar &amp; not_halt &amp; ...

# factor_engine 时序算子:
def ts_mean(x, mask, window):
    # mask==0 的位置不 push 到 ring buffer; 输出 NaN
    ...</code></pre>
      </details>

      <details class="code-fold">
        <summary>PREPROC-022 · 分钟级 knowledge_ts asof（<code>pit_minute_asof.py</code>）</summary>
        <pre><code># events: knowledge_ts_utc from sec_edgar acceptance_time or news published_utc
# grid: minute bars timestamp (UTC)
# pd.merge_asof(events, bars, left_on='knowledge_ts_utc', right_on='timestamp',
#               direction='backward', tolerance='1min')
# 保守: ceil to next minute bar if sub-minute acceptance</code></pre>
      </details>

      <details class="code-fold">
        <summary>ENG-012 · <code>fact_market_halt_events</code> + 分钟 <code>is_market_halt</code></summary>
        <pre><code>CREATE TABLE IF NOT EXISTS qs_massive.fact_market_halt_events (
    halt_start_utc DateTime64(3,'UTC'),
    halt_end_utc   DateTime64(3,'UTC'),
    mwcb_level     LowCardinality(String),  -- L1|L2|L3
    note String
) ENGINE = MergeTree() ORDER BY halt_start_utc;</code></pre>
      </details>

      <details class="code-fold">
        <summary>ENG-013 · 真值交易日历硬滤（<code>calendar_hard_filter.py</code>）</summary>
        <pre><code>VALID_T = load_trading_days_from_sip_day_aggs()  # 𝒯
def keep_row(trade_date, ts_utc) -> bool:
    if trade_date not in VALID_T:
        return False  # Ghost / weekend / holiday
    if ts_et &lt; time(4, 0):  # 盘前非法窗口（按产品定义可调）
        return False
    return True</code></pre>
      </details>

      <details class="code-fold">
        <summary>PREPROC-018~020 · 停牌 / 价格底座 / IPO 预开盘列</summary>
        <pre><code># fact_bars_adjusted_minute 追加:
#   is_ticker_halt UInt8
#   is_market_halt UInt8
#   is_ipo_pre_open UInt8
#   price_clamped_flag UInt8
#   halt_minutes_rth UInt16  -- 累计 &gt; 60 -&gt; universe_mask=False @ EOD</code></pre>
      </details>
'''

TASK_ROWS = """
          <tr><td>PREPROC-017</td><td><span class=\"badge badge-p1\">P1</span></td><td>拆股/分红成交量非对称缩放（分红不调 vol）</td><td class=\"owner-cell\">数据工程组</td><td class=\"status-cell\">待办</td><td></td></tr>
          <tr><td>PREPROC-021</td><td><span class=\"badge badge-p1\">P1</span></td><td>Mask-First 掩码前置（防御时序算子上游污染）</td><td class=\"owner-cell\">量化研究组</td><td class=\"status-cell\">待办</td><td></td></tr>
          <tr><td>PREPROC-022</td><td><span class=\"badge badge-p1\">P1</span></td><td>分钟级 acceptance/published 秒级 asof 拼装</td><td class=\"owner-cell\">基础架构组</td><td class=\"status-cell\">待办</td><td></td></tr>
          <tr><td>ENG-012</td><td><span class=\"badge badge-p1\">P1</span></td><td>MWCB 大盘熔断事件 + is_market_halt 掩码</td><td class=\"owner-cell\">数据工程组</td><td class=\"status-cell\">待办</td><td></td></tr>
          <tr><td>ENG-013</td><td><span class=\"badge badge-p1\">P1</span></td><td>真值日历 𝒯 + Ghost Trades 硬删除</td><td class=\"owner-cell\">平台 SRE</td><td class=\"status-cell\">待办</td><td></td></tr>
          <tr><td>PREPROC-018</td><td><span class=\"badge badge-p1\">P1</span></td><td>LULD/监管停牌日内重构 + universe 联动</td><td class=\"owner-cell\">量化研究组</td><td class=\"status-cell\">待办</td><td></td></tr>
          <tr><td>PREPROC-019</td><td><span class=\"badge badge-p1\">P1</span></td><td>仙股复权价 floor clamp（防除零）</td><td class=\"owner-cell\">基础架构组</td><td class=\"status-cell\">待办</td><td></td></tr>
          <tr><td>PREPROC-020</td><td><span class=\"badge badge-p1\">P1</span></td><td>IPO 首日 Opening Cross 前 is_ipo_pre_open</td><td class=\"owner-cell\">基础架构组</td><td class=\"status-cell\">待办</td><td></td></tr>
"""

PANEL_COLS = """
          <tr><td><code>compute_mask</code></td><td>UInt8</td><td>Mask-First 计算白名单</td><td>PREPROC-021</td></tr>
          <tr><td><code>is_market_halt</code></td><td>UInt8</td><td>MWCB 挂起分钟</td><td>ENG-012</td></tr>
          <tr><td><code>is_ticker_halt</code></td><td>UInt8</td><td>个股停牌分钟</td><td>PREPROC-018</td></tr>
          <tr><td><code>is_ipo_pre_open</code></td><td>UInt8</td><td>IPO 首笔成交前</td><td>PREPROC-020</td></tr>
          <tr><td><code>price_clamped_flag</code></td><td>UInt8</td><td>复权价底座截断</td><td>PREPROC-019</td></tr>
          <tr><td><code>knowledge_ts_utc</code></td><td>DateTime64(3)</td><td>离散事件获知时刻（分钟 PiT）</td><td>PREPROC-022</td></tr>
"""

MD_SECTION = r'''

## 21. 三轮补盲点与极端异常清洗（2026-06-03 终审）

> HTML 详述：<a href="Massive数据治理与改进行动清单.html#sec-blind-spots-3">§12c</a>、<a href="Massive数据治理与改进行动清单.html#sec-blind-spots-4">§12d</a>。

| ID | 主题 | 要点 |
|----|------|------|
| PREPROC-017 | 成交量非对称 | split 调价量；dividend 只调价 |
| PREPROC-021 | Mask-First | 时序算子掩码前置，禁上游污染 |
| PREPROC-022 | 分钟 PiT | acceptance_time / published_utc 秒级 asof |
| ENG-012 | MWCB | is_market_halt，滚动窗口跳过 |
| ENG-013 | Ghost Trades | 真值日历 𝒯 硬删非交易日行 |
| PREPROC-018 | 个股停牌 | vol=0 + 长停牌 universe=False |
| PREPROC-019 | 仙股 floor | adj_price clamp |
| PREPROC-020 | IPO 冷启动 | is_ipo_pre_open |

**ID 说明：** §12b 的 `PREPROC-015` = 股本双轨；`PREPROC-016` = 新闻去重。本节 `PREPROC-017` 专指成交量协议（评审稿中曾另称 PREPROC-015，以本表为准）。
'''


def patch_html(html: str) -> str:
    if "sec-blind-spots-3" not in html:
        marker = (
            "        五条已追加 <a href=\"#sec-tasks\">§14 任务总表</a>；与 <a href=\"#sec-blind-spots\">§12a</a> 正交，一并构成入模前封板清单。\n"
            "      </div>\n\n    </section>\n\n    <!-- 7. Task table -->"
        )
        html = html.replace(
            marker,
            "        五条已追加 <a href=\"#sec-tasks\">§14 任务总表</a>；与 <a href=\"#sec-blind-spots\">§12a</a> 正交，一并构成入模前封板清单。\n"
            "      </div>\n\n    </section>\n\n" + ROUND3 + "\n" + ROUND4 + "\n\n    <!-- 7. Task table -->",
            1,
        )

    if 'id="sec-code-round3"' not in html:
        anchor = '<h3 id="sec-code-extra">10c.12 补充折叠（运维 · 校验 · SQL 模板）</h3>'
        if anchor in html:
            html = html.replace(anchor, CODE_R3 + "\n\n      " + anchor, 1)

    sec_tasks = re.search(
        r'(<section id="sec-tasks">.*?<tbody>)(.*?)(</tbody>)',
        html,
        flags=re.DOTALL,
    )
    if sec_tasks and "PREPROC-017" not in sec_tasks.group(2):
        html = (
            html[: sec_tasks.start(2)]
            + sec_tasks.group(2).rstrip()
            + TASK_ROWS
            + sec_tasks.group(3)
            + html[sec_tasks.end(3) :]
        )

    # before-factor checklist
    if "PREPROC-021" not in html:
        html = html.replace(
            "<li><code>PREPROC-016</code>：新闻 count 已去重（若启用 news 列）</li>",
            "<li><code>PREPROC-016</code>：新闻 count 已去重（若启用 news 列）</li>\n"
            "        <li><code>PREPROC-017</code>：分红不调整 volume；拆股对偶缩放</li>\n"
            "        <li><code>PREPROC-021</code>：<code>compute_mask</code> Mask-First 已落盘</li>\n"
            "        <li><code>PREPROC-022</code>：分钟 PiT 无日内 filing 广播</li>\n"
            "        <li><code>ENG-012/013</code>：MWCB 掩码 + 非 𝒯 行已剔除</li>\n"
            "        <li><code>PREPROC-018~020</code>：停牌/IPO/价格 floor 列齐全</li>",
        )

    # Phase 3 roadmap
    html = html.replace(
        "PREPROC-011~016 · ENG-011 · FEAT-001~004 · CH-001~007",
        "PREPROC-011~022 · PREPROC-017~021 · ENG-011~013 · FEAT-001~004 · CH-001~007",
    )

    # 10c.10 index
    if "PREPROC-021" not in html.split("10c.10")[1].split("10c.11")[0]:
        html = html.replace(
            "<tr><td>CH-007</td><td><a href=\"#sec-code-round2\">分钟 DDL</a></td><td>分钟 CH 分区</td></tr>",
            "<tr><td>CH-007</td><td><a href=\"#sec-code-round2\">分钟 DDL</a></td><td>分钟 CH 分区</td></tr>\n"
            "          <tr><td>017/021/022 · ENG-012/013 · 018~020</td><td><a href=\"#sec-code-round3\">§10c.15</a></td><td>三轮+极端异常</td></tr>",
        )

    # panel column dict (10c.6) — insert before delisting_policy if present
    if "<code>compute_mask</code>" not in html:
        for needle in (
            "<tr><td><code>delisting_policy</code></td>",
            "<tr><td><code>permanent_id</code></td>",
        ):
            if needle in html:
                html = html.replace(needle, PANEL_COLS + "          " + needle, 1)
                break

    # minute DDL extra columns in round2 fold
    if "is_ticker_halt" not in html:
        html = html.replace(
            "    is_early_close UInt8,\n    batch_id String",
            "    is_early_close UInt8,\n    is_market_halt UInt8,\n    is_ticker_halt UInt8,\n    is_ipo_pre_open UInt8,\n    price_clamped_flag UInt8,\n    batch_id String",
        )

    # version bump
    html = html.replace(
        "版本 v3.0 HTML 导航美化版",
        "版本 v3.1 HTML 三轮补盲点+极端异常版",
    )
    html = html.replace(
        "v3.0 导航美化版",
        "v3.1 三轮+极端异常版",
    )

    return html


def patch_md(md: str) -> str:
    if "## 21. 三轮补盲点" in md:
        return md
    if "## 20." in md:
        md = md + MD_SECTION
    else:
        md = md.rstrip() + "\n" + MD_SECTION
    md = re.sub(
        r"文档版本[^\n]*",
        "文档版本：v2.7（含 §21 三轮补盲点与极端异常）",
        md,
        count=1,
    )
    return md


def main():
    html = HTML.read_text(encoding="utf-8")
    html = patch_html(html)
    HTML.write_text(html, encoding="utf-8")
    if MD.exists():
        md = MD.read_text(encoding="utf-8")
        MD.write_text(patch_md(md), encoding="utf-8")
    print("round3+edge added, html lines:", html.count("\n") + 1)


if __name__ == "__main__":
    main()
