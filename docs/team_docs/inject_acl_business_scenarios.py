#!/usr/bin/env python3
"""Add business-scenario context (why we treat each ACL rule) to HTML handbook."""
from pathlib import Path
import re
import subprocess

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"
BEAUTIFY = Path(__file__).resolve().parent / "beautify_massive_html.py"

# Each tuple: (unique_anchor after <h3>, scenario_html) — skipped if anchor already followed by box-scenario
PATCHES = [
    (
        '<h3>ACL-1 · IPO 与 Spin-off 首日「延迟开盘时空黑洞」</h3>',
        """
      <div class="box-scenario">
        <strong>业务场景（为何要处理）：</strong>
        <p>美股新股与分拆上市首日，09:30 开盘钟响时往往<strong>尚无成交</strong>：订单在 LOB 中累积，直至 10:30～14:00 的
        <strong>Opening Cross</strong> 才出现首笔有效价量（如大型 Spin-off 子股、热门 IPO）。</p>
        <ul>
          <li><strong>典型例：</strong>分拆上市子股首日长时间无分钟 bar；传统 IPO 若用发行价 ffill 09:30–10:30 → 假「零波动横盘」，动量/波动率因子失真。</li>
          <li><strong>不做会怎样：</strong>大模型把「真空期」当成可交易连续序列，回测里可在根本无流动性的时段「成交」，产出 <strong>Fake Alpha</strong>。</li>
          <li><strong>团队要记住：</strong>这是<strong>交易机制</strong>问题，不是数据缺失；用 <code>is_ipo_pre_open</code> 标出真空，算子从首笔真实撮合再起窗。</li>
        </ul>
      </div>
""",
    ),
    (
        '<h3>ACL-2 · 长窗算子「数据预热期」（Burn-in）</h3>',
        """
      <div class="box-scenario">
        <strong>业务场景（为何要处理）：</strong>
        <p>白盒挖掘常演化出 <code>ts_mean(close, 252)</code>、<code>ts_mean(close, 1000)</code> 等长窗；研究若设定
        <strong>训练从 2016-01-01 开始</strong>，但数据管道也只从 2016 加载，则窗口内前 1～4 年<strong>没有足够历史</strong>喂饱滑动缓存。</p>
        <ul>
          <li><strong>典型例：</strong>1000 日窗口 ≈ 4 年；2016 起跑 → 至 2020 初全市场该因子仍为 NaN，浪费一半以上样本做「冷启动死锁」。</li>
          <li><strong>不做会怎样：</strong>误以为因子无效或随意用 0/截面均值填充 → 盲测 IC 与实盘逻辑完全脱节。</li>
          <li><strong>团队要记住：</strong>「论文可见区间」≠「磁盘加载起点」；Feature Store 须自动 <strong>Lookback Padding</strong>（<code>PREPROC-025</code>）。</li>
        </ul>
      </div>
""",
    ),
    (
        '<h3>ACL-3 · 个股 LULD / 监管停牌（Reindex + 价量非对称）</h3>',
        """
      <div class="box-scenario">
        <strong>业务场景（为何要处理）：</strong>
        <p>美股无 A 股式全天涨跌停，但有个股 <strong>LULD</strong>（5 分钟波动超阈值 → 暂停 5 分钟）及
        <strong>监管停牌</strong>（重大公告、调查，条件码 T1/T2）。停牌期间物理上无成交，SIP 常缺 bar。</p>
        <ul>
          <li><strong>典型例：</strong>财报暴雷或调查公告后盘中停牌数小时；复牌当日波动极大。</li>
          <li><strong>不做会怎样：</strong>直接 <strong>drop 停牌行</strong> → 事件回测默认「无法买卖」，组合在仿真里<strong>躲过复牌踩踏</strong>（生存者偏差）。</li>
          <li><strong>团队要记住：</strong>补回骨架行、价 ffill、量=0、打 <code>is_ticker_halt</code>；累计停牌过长则当日 <code>universe_mask=False</code>。</li>
        </ul>
      </div>
""",
    ),
    (
        '<h3>ACL-4 · 大盘熔断 MWCB</h3>',
        """
      <div class="box-scenario">
        <strong>业务场景（为何要处理）：</strong>
        <p>标普 500 日内跌幅触及阈值时，全市场触发 <strong>Market-Wide Circuit Breaker（MWCB）</strong>：
        Level 1/2 暂停交易 15 分钟，Level 3 可提前收市（如 <strong>2020 年 3 月疫情周连续四次熔断</strong>）。</p>
        <ul>
          <li><strong>典型例：</strong>熔断 15 分钟内全市场无价无量；若 ffill 成「极低波动横盘」→ 日内 realized vol、Amihud 等系统性偏低。</li>
          <li><strong>不做会怎样：</strong>模型把「监管强制静止」当成「市场平静」，系统性风险因子在极端日全面失真。</li>
          <li><strong>团队要记住：</strong>全市场事件用 <code>is_market_halt</code>；<code>ts_*</code> 滚动步长<strong>跳过</strong>熔断分钟（与个股 halt 区分）。</li>
        </ul>
      </div>
""",
    ),
    (
        '<h3>ACL-5 · 仙股与极端复权「负价 / 近零」</h3>',
        """
      <div class="box-scenario">
        <strong>业务场景（为何要处理）：</strong>
        <p>低价股为保主板上市资格常多次 <strong>合股（Reverse Split）</strong>；叠加现金分红后，在<strong>后复权</strong>长历史上
        价格可被压到接近 0 或为负（SIP 生产锚点本身不复权，复权在 Layer 2 才算）。</p>
        <ul>
          <li><strong>典型例：</strong>多次 1:50、1:100 合股仙股；高分红工业股（如历史上 GE 类路径）长周期复权分母趋零。</li>
          <li><strong>不做会怎样：</strong><code>close/open-1</code> 除零、Inf 污染截面 Z-Score，ClickHouse/分布式任务直接崩溃。</li>
          <li><strong>团队要记住：</strong>落盘前 floor + <code>price_clamped_flag</code> 是<strong>数值安全阀</strong>，不是「改行情真相」；策略可另剔 persistent clamp 标的。</li>
        </ul>
      </div>
""",
    ),
    (
        '<h3>ACL-6 · 成交量缩放非对称（拆股 vs 分红）</h3>',
        """
      <div class="box-scenario">
        <strong>业务场景（为何要处理）：</strong>
        <p><strong>拆股</strong>改变流通股数，历史价量须对偶调整以保名义成交额守恒；
        <strong>现金分红</strong>只调价格、<strong>不改变</strong>物理股本——二者对 volume 的影响完全不对称。</p>
        <ul>
          <li><strong>典型例：</strong>AAPL 1:4 拆股 → 价 ÷4、量 ×4；同日 MSFT 现金分红 → 只除息价，量不应 × 除息因子。</li>
          <li><strong>不做会怎样：</strong>分红也缩放 volume → 历史换手率、Amihud、量价相关等<strong>虚假放大</strong>，流动性因子不可信。</li>
          <li><strong>团队要记住：</strong>元数据写死 <code>volume_adj_policy=split_only</code>（<code>PREPROC-017</code>）。</li>
        </ul>
      </div>
""",
    ),
    (
        '<h3>ACL-7 · SIFMA 提前收盘（半日市骨架坍缩）</h3>',
        """
      <div class="box-scenario">
        <strong>业务场景（为何要处理）：</strong>
        <p><strong>SIFMA</strong> 建议下，感恩节翌日（Black Friday）、圣诞前夕、独立日前夕等美东 <strong>13:00 提前收市</strong>，
        当日合法 RTH 仅约 210 根分钟 bar，而非 390 根。</p>
        <ul>
          <li><strong>典型例：</strong>按 390 根骨架补 13:00–16:00 → 把「休市」当成「停牌 ffill + vol=0」→ 伪低波动、伪量能衰减。</li>
          <li><strong>不做会怎样：</strong>日内结构因子在全年少数半日市系统性偏，与大模型挖掘的「季节性」混淆。</li>
          <li><strong>团队要记住：</strong><code>dim_calendar.is_early_close</code> 驱动骨架<strong>物理截断</strong>，13:00 后禁止任何补 bar。</li>
        </ul>
      </div>
""",
    ),
    (
        '<h3>ACL-8 · 分钟级日内披露（秒级 PiT）</h3>',
        """
      <div class="box-scenario">
        <strong>业务场景（为何要处理）：</strong>
        <p>分钟级因子必须用「市场<strong>何时已知</strong>」对齐，而非「财报所属季度末」。
        8-K、盈余公告、并购等常在<strong>盘中</strong>发布（如 11:15:23），若用日级 <code>filing_date</code> 贴全天分钟 → 盘前即「看见」午后信息。</p>
        <ul>
          <li><strong>典型例：</strong>11:15 披露的 8-K；若 09:30–11:14 分钟线已带上该事件基本面 → 回测套利 <strong>未来函数</strong>。</li>
          <li><strong>不做会怎样：</strong>分钟多因子在样本内极高 IC，实盘无法复现（披露延迟与合规窗口）。</li>
          <li><strong>团队要记住：</strong>日频可用 <code>filing_date</code>；<strong>分钟因子</strong>必须用 <code>knowledge_ts_utc</code> / EDGAR <code>acceptance_time</code>（<code>PREPROC-022</code>）。</li>
        </ul>
      </div>
""",
    ),
    (
        '<h3>ACL-D1 · 财务极端畸变：负资产、负权益与零营收</h3>',
        """
      <div class="box-scenario">
        <strong>业务场景（为何要处理）：</strong>
        <p>破产重整、巨额亏损、研发期 Biotech、壳公司会出现<strong>负股东权益</strong>或<strong>多季零营收</strong>——
        这是真实财报状态，不是脏数据。</p>
        <ul>
          <li><strong>典型例：</strong>2020 年前后 Hertz 破产路径、AMC 等高波动 distressed；临床阶段药企收入为 0 仍上市交易。</li>
          <li><strong>不做会怎样：</strong><code>B/P = equity / cap</code> 符号翻转 → 截面 Rank 把濒临破产股排成「极致价值/成长」，污染估值与质量因子。</li>
          <li><strong>团队要记住：</strong>用掩码 <strong>标出</strong>而非填 0；估值类因子在截面层 <strong>Exclusion</strong>（<code>PREPROC-026</code>）。</li>
        </ul>
      </div>
""",
    ),
    (
        '<h3>ACL-D2 · Spinoff 分拆上市：非传统除权的价格断层</h3>',
        """
      <div class="box-scenario">
        <strong>业务场景（为何要处理）：</strong>
        <p>大型企业 <strong>Spinoff</strong>（如 GE 分拆 GE Vernova、辉瑞分拆 Viatris）在 Ex-date 母公司股价常
        <strong>技术性大跌 20%～40%</strong>——资产剥离，并非经营崩盘；标准 <code>splits</code>/<code>dividends</code> 表往往<strong>无对应因子</strong>。</p>
        <ul>
          <li><strong>不做会怎样：</strong>动量、跳空、波动率因子把「分拆除权」当成暴跌信号 → 假做空/假止损，回测夏普虚高。</li>
          <li><strong>团队要记住：</strong>分拆比率进公司行动流，历史价视同<strong>特殊非现金分红</strong>平滑（<code>PREPROC-027</code>）；与 ACL-1「子股首日 Opening Cross」是两件不同的事。</li>
        </ul>
      </div>
""",
    ),
    (
        '<h3>ACL-D3 · SEC EDGAR Batch-Filing：同日多期财报堆叠</h3>',
        """
      <div class="box-scenario">
        <strong>业务场景（为何要处理）：</strong>
        <p>EDGAR <strong>系统故障恢复</strong>、监管调查延迟披露后，公司可能在<strong>同一交易日</strong>一次性补交
        多期 10-K/10-Q（如积压的 2022 年报 + 2023 年报 + 多个季度报），各行 <code>filing_date</code> 相同。</p>
        <ul>
          <li><strong>不做会怎样：</strong>简单 <code>merge_asof</code> 或去重 → TTM 净利润/营收在披露日<strong>随机跳变</strong>，YoY 断裂，基本面动量因子出现非业务断崖。</li>
          <li><strong>团队要记住：</strong>同日落库多期时，tie-break 用 <code>period_end</code> 由远及近依次覆盖（<code>PREPROC-028</code>）。</li>
        </ul>
      </div>
""",
    ),
    (
        '<h3>ACL-D4 · Stock dividend vs split：股本非对称</h3>',
        """
      <div class="box-scenario">
        <strong>业务场景（为何要处理）：</strong>
        <p>美股常见 <strong>股票股利</strong>（如每 10 股送 1 股，价约降 10%、股本增 10%），在供应商表里常落在
        <code>dividends</code> 且 <code>cash_amount=0</code>——语义上更接近 <strong>拆股</strong> 而非现金分红。</p>
        <ul>
          <li><strong>不做会怎样：</strong>只调价格不调 <code>shares_out</code> → <code>market_cap = price × shares</code> 在除权日<strong>虚假蒸发</strong>，规模与流动性因子错乱。</li>
          <li><strong>团队要记住：</strong><code>stock_dividend</code> 路由 <code>splits</code> 管道，价量股本对偶缩放（<code>PREPROC-029</code>）。</li>
        </ul>
      </div>
""",
    ),
    (
        '<h3>ACL-B1 · 裁决作废交易（Erroneous / Adjudicated Trades）</h3>',
        """
      <div class="box-scenario">
        <strong>业务场景（为何要处理）：</strong>
        <p><strong>2024-06-03</strong> 纽交所 CTA 软件故障，<strong>BRK.A</strong>、<strong>BMO</strong> 等蓝筹盘中瞬间跌约
        <strong>99.97%</strong> 后，交易所与监管裁定该时段成交<strong>全部作废（Adjudicated/Cancelled）</strong>——
        现实中这些价格从未可成交。</p>
        <ul>
          <li><strong>不做会怎样：</strong>日内 high/low、最大回撤、振幅因子在巨型市值股上出现离谱极值，<strong>扭曲当日全截面 Rank</strong>。</li>
          <li><strong>团队要记住：</strong>读 <code>correction</code>/<code>conditions</code> + 分钟 MAD 回滚；打 <code>is_erroneous_glitch</code>（<code>PREPROC-030</code>）。与 Ghost 非交易日（<code>ENG-013</code>）不同。</li>
        </ul>
      </div>
""",
    ),
    (
        '<h3>ACL-B2 · 财年截止日变更与非标过渡季（Fiscal Year-End Shift）</h3>',
        """
      <div class="box-scenario">
        <strong>业务场景（为何要处理）：</strong>
        <p>公司可将财年从 <strong>12-31 改为 9-30</strong>（等），交接期须交 <strong>Transition Report（10-QT）</strong>，
        形成仅含 1～2 个月的「过渡季度」，Schema 仍可能标 <code>quarterly</code>。</p>
        <ul>
          <li><strong>不做会怎样：</strong>TTM = 最近四个 <code>quarterly</code> 简单相加 → 把 1 个月利润当成 3 个月，TTM 在披露日<strong>虚假暴跌 20%+</strong>，财务动量因子爆噪。</li>
          <li><strong>团队要记住：</strong>期间天数 &lt;75 或 &gt;105 天打 <code>is_nonstandard_duration</code>，TTM 做时间权重年化（<code>PREPROC-031</code>）。</li>
        </ul>
      </div>
""",
    ),
    (
        '<h3>ACL-B3 · 退市转 OTC 粉单：Tick Size 与幸存者偏差</h3>',
        """
      <div class="box-scenario">
        <strong>业务场景（为何要处理）：</strong>
        <p>破产、欺诈聆讯、长期低于 $1 等可导致 <strong>NYSE/NASDAQ 摘牌</strong>，股票转入 <strong>OTC 粉单</strong> 继续交易。
        主板最后交易日之后，最小报价单位由 <strong>$0.01 变为 $0.0001</strong>，微观结构「基因突变」。</p>
        <ul>
          <li><strong>典型例：</strong>危机股主板 Last Day 后仍在 OTC 成交；若 universe 直接删标的 → 回测无法在末日<strong>模拟强平</strong>，躲过清算期暴跌。</li>
          <li><strong>不做会怎样：</strong>严重 <strong>Survivorship Bias</strong>；高频 LOB 因子用错 tick 步长导致深度档位溢出/死锁。</li>
          <li><strong>团队要记住：</strong><code>delisting_return</code> 惩罚 + <code>tick_size_mask</code>（<code>PREPROC-032</code> 扩展 <code>012</code>）。</li>
        </ul>
      </div>
""",
    ),
    (
        '<h3>ACL-B4 · Form T / Late Prints 延迟上报（Out-of-Sequence）</h3>',
        """
      <div class="box-scenario">
        <strong>业务场景（为何要处理）：</strong>
        <p>场外大宗、暗盘（TRF）成交可在<strong>真实成交后数小时</strong>才进入 SIP。例如 14:00 成交的机构块，
        16:45（盘后）才带 <strong>Form T / Late Print</strong> 条件码上报——<code>sip_timestamp</code> 是 16:45，成交价却是 14:00 的现货价。</p>
        <ul>
          <li><strong>不做会怎样：</strong>16:45 分钟 bar 出现<strong>从未在盘面发生</strong>的长上下影线与放量脉冲，污染分钟波动率/影线类因子。</li>
          <li><strong>团队要记住：</strong>延迟单：量可计入、<strong>价不参与该分钟 H/L</strong>（<code>PREPROC-033</code> + <code>TASK-DC-005</code>）。</li>
        </ul>
      </div>
""",
    ),
    (
        '<h3>ACL-B5 · 新闻多源异步重发（News Duplicate Propagation）</h3>',
        """
      <div class="box-scenario">
        <strong>业务场景（为何要处理）：</strong>
        <p>另类舆情供应商（如 Benzinga 及交叉授权源）常对<strong>同一公司治理事件</strong>多次发稿：盘中首发后，
        盘后、次日盘前换标题/微改正文重发，<code>published_utc</code> 全新但语义重复。</p>
        <ul>
          <li><strong>典型例：</strong>15:45 大宗/并购公告 → 15:46 首条新闻 → 16:30、次日 08:30「重发」；若计数三次 → <code>news_count_1d</code> 虚增，情绪冲击被放大数倍。</li>
          <li><strong>不做会怎样：</strong>新闻爆发度、情绪衰减因子违反<strong>事件独立性</strong>，多模态因子过拟合「重发噪音」。</li>
          <li><strong>团队要记住：</strong>SimHash/3-gram 24h 去重，<code>news_dup_suppressed=1</code> 时不累加计数（<code>PREPROC-034</code>）。</li>
        </ul>
      </div>
""",
    ),
]

INTRO_ACL = """
      <div class="box-scenario" id="acl-read-how">
        <strong>本节怎么读（给工程 / 研究 / PM）：</strong>
        每条规则按 <strong>业务场景 → 技术陷阱（box-danger）→ 处理要点</strong> 组织。
        「业务场景」说明<strong>真实市场里发生了什么</strong>；「技术陷阱」说明<strong>不清洗时因子/回测会怎样坏</strong>。
        避免把掩码当成晦涩开关——掩码是交易机制与合规披露在数据层的<strong>显式契约</strong>。
      </div>
"""

TABLE_HEADER_OLD = """        <thead>
          <tr>
            <th>情形</th><th>如何识别</th><th>Layer 2 怎么处理</th><th>因子 / <code>ts_*</code> 怎么处理</th><th>禁止</th><th>任务</th>
          </tr>
        </thead>"""

TABLE_HEADER_NEW = """        <thead>
          <tr>
            <th>情形</th><th>典型业务场景（为何相关）</th><th>如何识别</th><th>Layer 2 怎么处理</th><th>因子 / <code>ts_*</code></th><th>禁止</th><th>任务</th>
          </tr>
        </thead>"""

# Short scenario blurbs for §3.3 key rows — applied via row-specific patches if we add column
EDGE_SCENARIO_COL = {
    "新股 / IPO": "Opening Cross 前无成交；真空期非缺失",
    "交易所错单": "2024-06-03 NYSE CTA；BRK.A 等作废闪崩",
    "财年变更": "10-QT 过渡季仅 1～2 月却标 quarterly",
    "退市转 OTC": "摘牌后 tick $0.0001；忌 universe 蒸发",
    "Form T": "14:00 成交 16:45 才进 SIP",
    "新闻异步重发": "同一公告多时段换标题重发",
    "负权益": "Hertz/AMC 类 distressed 报表",
    "Spinoff": "GE/Viatris 式技术性大跌非崩盘",
    "Batch-Filing": "EDGAR 恢复后同日多期补档",
    "Stock dividend": "送股在分红表但应走拆股逻辑",
    "待上市": "—",
    "退市 / 最后": "Last Day 须记 delisting_return",
    "个股 LULD": "5min 波动熔断暂停",
    "大盘熔断": "2020-03 _covid 四次 MWCB",
    "长窗算子": "train 2016 但 ts_mean(1000) 需更早数据",
    "半日市": "感恩节翌日 13:00 收市",
    "仙股": "多次合股后复权价近零",
}


def inject_scenarios(html: str) -> str:
    if 'id="acl-read-how"' not in html:
        html = html.replace(
            '<section id="sec-acl">\n      <h2>',
            '<section id="sec-acl">\n' + INTRO_ACL.strip() + '\n      <h2>',
            1,
        )
        # avoid duplicate if intro already before table
        if html.count('id="acl-read-how"') > 1:
            html = html.replace(INTRO_ACL.strip() + "\n      <h2>", "<h2>", 1)

    for anchor, snippet in PATCHES:
        if f"{anchor}\n      <div class=\"box-scenario\">" in html:
            continue
        if f"{anchor}\n      <div class='box-scenario'>" in html:
            continue
        if anchor in html:
            html = html.replace(anchor, anchor + snippet, 1)

    # §3.3 intro
    if "典型业务场景" not in html.split("sec-edge-cases")[1][:800] if "sec-edge-cases" in html else "":
        html = html.replace(
            "Layer 2 负责<strong>打标 + 掩码 + 元数据</strong>",
            "下表「典型场景」列说明<strong>为何要处理</strong>；Layer 2 负责<strong>打标 + 掩码 + 元数据</strong>",
            1,
        )

    # Add scenario column to §3.3 — insert after first <td><strong> in each row via manual row updates for main cases
    row_patches = [
        (
            '<td><strong>新股 / IPO / Spin-off 首日</strong></td>\n            <td><code>ipos.listing_date</code>',
            '<td><strong>新股 / IPO / Spin-off 首日</strong></td>\n            <td>首日常 10:30+ 才有 Opening Cross；非缺数据</td>\n            <td><code>ipos.listing_date</code>',
        ),
        (
            '<td><strong>交易所错单闪崩</strong></td>\n            <td><code>correction</code>',
            '<td><strong>交易所错单闪崩</strong></td>\n            <td>2024-06-03 CTA；BRK.A/BMO 作废成交</td>\n            <td><code>correction</code>',
        ),
        (
            '<td><strong>财年变更过渡季</strong></td>\n            <td>Duration',
            '<td><strong>财年变更过渡季</strong></td>\n            <td>财年改 9/30 等；10-QT 仅 1～2 月</td>\n            <td>Duration',
        ),
        (
            '<td><strong>退市转 OTC</strong></td>\n            <td><code>fact_delisting_events</code>',
            '<td><strong>退市转 OTC</strong></td>\n            <td>主板摘牌转粉单；忌回测「蒸发」</td>\n            <td><code>fact_delisting_events</code>',
        ),
        (
            '<td><strong>Form T / Late Prints</strong></td>\n            <td>条件码',
            '<td><strong>Form T / Late Prints</strong></td>\n            <td>盘中成交、盘后/延迟才进 SIP</td>\n            <td>条件码',
        ),
        (
            '<td><strong>新闻异步重发</strong></td>\n            <td>SimHash',
            '<td><strong>新闻异步重发</strong></td>\n            <td>同一公告多时段换标题重发</td>\n            <td>SimHash',
        ),
        (
            '<td><strong>负权益 / 零营收</strong></td>\n            <td><code>total_equity',
            '<td><strong>负权益 / 零营收</strong></td>\n            <td>Hertz/AMC、零营收 Biotech</td>\n            <td><code>total_equity',
        ),
        (
            '<td><strong>Spinoff 分拆 Ex-date</strong></td>\n            <td><code>spinoffs</code>',
            '<td><strong>Spinoff 分拆 Ex-date</strong></td>\n            <td>GE/Viatris 式技术性大跌</td>\n            <td><code>spinoffs</code>',
        ),
        (
            '<td><strong>Batch-Filing 同日多期</strong></td>\n            <td>同 <code>filing_date</code>',
            '<td><strong>Batch-Filing 同日多期</strong></td>\n            <td>EDGAR 故障后同日补多期财报</td>\n            <td>同 <code>filing_date</code>',
        ),
        (
            '<td><strong>Stock dividend</strong></td>\n            <td><code>distribution_type',
            '<td><strong>Stock dividend</strong></td>\n            <td>送 10% 股：价降、股本增</td>\n            <td><code>distribution_type',
        ),
    ]
    for old, new in row_patches:
        if old in html and "2024-06-03 CTA" not in html.split(old)[0][-200:]:
            html = html.replace(old, new, 1)

    # §ACL+ / §ACL++ short intros
    if "id=\"aclplus-read-how\"" not in html:
        html = html.replace(
            '<section id="sec-acl-deep">\n      <h2>',
            '<section id="sec-acl-deep">\n      <div class="box-scenario" id="aclplus-read-how">\n        <strong>§ACL+ 定位：</strong>解决「财报与公司行动<strong>语义</strong>」导致的假信号——不是行情缺 bar，而是<strong>科目含义</strong>或<strong>事件类型</strong>被误读。\n      </div>\n      <h2>',
            1,
        )
    if "id=\"aclpp-read-how\"" not in html:
        html = html.replace(
            '<section id="sec-acl-deep2">\n      <h2>',
            '<section id="sec-acl-deep2">\n      <div class="box-scenario" id="aclpp-read-how">\n        <strong>§ACL++ 定位：</strong>解决「基础设施与报送机制<strong>失真</strong>」——交易所故障、财年变更、OTC 微观、延迟成交、媒体重发；详见各条<strong>业务场景</strong>框。\n      </div>\n      <h2>',
            1,
        )

    # Summary tables: add 典型场景 column to ACL tables
    for old, new in [
        (
            '<thead><tr><th>#</th><th>极端情形</th><th>核心掩码/列</th><th>任务</th></tr></thead>',
            '<thead><tr><th>#</th><th>极端情形</th><th>典型业务场景</th><th>核心掩码/列</th><th>任务</th></tr></thead>',
        ),
        (
            '<tr><td>1</td><td>IPO / Spin-off 延迟 Opening Cross</td><td><code>is_ipo_pre_open</code>',
            '<tr><td>1</td><td>IPO / Spin-off 延迟 Opening Cross</td><td>首日常延迟撮合、无 09:30 bar</td><td><code>is_ipo_pre_open</code>',
        ),
        (
            '<tr><td>2</td><td>长窗算子预热期 Burn-in</td><td>调度',
            '<tr><td>2</td><td>长窗算子预热期 Burn-in</td><td>train 2016 但 ts_mean(1000) 需 2012+ 行情</td><td>调度',
        ),
        (
            '<tr><td>3</td><td>个股 LULD / 监管停牌</td><td><code>is_ticker_halt</code>',
            '<tr><td>3</td><td>个股 LULD / 监管停牌</td><td>5min 熔断、T1/T2 监管停牌无成交</td><td><code>is_ticker_halt</code>',
        ),
        (
            '<tr><td>4</td><td>大盘熔断 MWCB</td><td><code>is_market_halt</code>',
            '<tr><td>4</td><td>大盘熔断 MWCB</td><td>2020-03 疫情周全市场 15min 停滞</td><td><code>is_market_halt</code>',
        ),
        (
            '<tr><td>5</td><td>仙股 / 复权价 ≤0</td><td><code>price_clamped_flag</code>',
            '<tr><td>5</td><td>仙股 / 复权价 ≤0</td><td>多次合股 + 分红 → 后复权近零/负</td><td><code>price_clamped_flag</code>',
        ),
        (
            '<tr><td>6</td><td>成交量仅随拆股缩放</td><td><code>volume_adj_policy',
            '<tr><td>6</td><td>成交量仅随拆股缩放</td><td>拆股调量、现金分红不调量</td><td><code>volume_adj_policy',
        ),
        (
            '<tr><td>7</td><td>SIFMA 提前收盘（半日市）</td><td><code>expected_rth_bars</code>',
            '<tr><td>7</td><td>SIFMA 提前收盘（半日市）</td><td>Black Friday 等 13:00 收市</td><td><code>expected_rth_bars</code>',
        ),
        (
            '<tr><td>8</td><td>分钟级日内披露 PiT</td><td><code>knowledge_ts_utc</code>',
            '<tr><td>8</td><td>分钟级日内披露 PiT</td><td>盘中 8-K/盈余公告秒级披露</td><td><code>knowledge_ts_utc</code>',
        ),
    ]:
        if old in html and "典型业务场景" not in html[html.find(old) : html.find(old) + 200]:
            html = html.replace(old, new, 1)

    # ACL+ / ACL++ summary tables — add 典型业务场景 column header once per section
    for sect_marker, hdr_old, hdr_new in [
        (
            "sec-acl-deep",
            '<thead><tr><th>#</th><th>深水区情形</th><th>核心掩码/契约</th><th>任务</th></tr></thead>',
            '<thead><tr><th>#</th><th>深水区情形</th><th>典型业务场景</th><th>核心掩码/契约</th><th>任务</th></tr></thead>',
        ),
        (
            "sec-acl-deep2",
            '<thead><tr><th>#</th><th>情形</th><th>核心掩码</th><th>任务</th></tr></thead>',
            '<thead><tr><th>#</th><th>情形</th><th>典型业务场景</th><th>核心掩码</th><th>任务</th></tr></thead>',
        ),
    ]:
        if sect_marker in html:
            part = html.split(sect_marker, 1)[1].split("</section>", 1)[0]
            if "典型业务场景" not in part[:1200]:
                html = html.replace(hdr_old, hdr_new, 1)

    deep_rows = [
        (
            '<tr><td>D1</td><td>负权益 / 零营收报表畸变</td><td><code>is_negative_equity</code>',
            '<tr><td>D1</td><td>负权益 / 零营收报表畸变</td><td>Hertz/AMC、零营收 Biotech</td><td><code>is_negative_equity</code>',
        ),
        (
            '<tr><td>D2</td><td>Spinoff 分拆日价格断崖（非 split/div）</td><td>分拆比率',
            '<tr><td>D2</td><td>Spinoff 分拆日价格断崖（非 split/div）</td><td>GE/Viatris Ex-date 技术大跌</td><td>分拆比率',
        ),
        (
            '<tr><td>D3</td><td>SEC Batch-Filing 同日多期堆叠</td><td>排序',
            '<tr><td>D3</td><td>SEC Batch-Filing 同日多期堆叠</td><td>EDGAR 故障后同日补多期</td><td>排序',
        ),
        (
            '<tr><td>D4</td><td>Stock dividend vs cash dividend</td><td>路由',
            '<tr><td>D4</td><td>Stock dividend vs cash dividend</td><td>10 送 1 落在分红表</td><td>路由',
        ),
        (
            '<tr><td>D5</td><td>仙股合股后近零/负价（与 ACL-5 同）</td><td><code>price_clamped_flag</code>',
            '<tr><td>D5</td><td>仙股合股后近零/负价（与 ACL-5 同）</td><td>多次 1:100 合股</td><td><code>price_clamped_flag</code>',
        ),
    ]
    for old, new in deep_rows:
        if old in html:
            html = html.replace(old, new, 1)

    b_rows = [
        (
            '<tr><td>B1</td><td>裁决作废错单 / 交易所闪崩（如 2024-06-03 NYSE CTA）</td><td><code>is_erroneous_glitch</code>',
            '<tr><td>B1</td><td>裁决作废错单 / 交易所闪崩</td><td>2024-06-03 BRK.A/BMO 作废</td><td><code>is_erroneous_glitch</code>',
        ),
        (
            '<tr><td>B2</td><td>财年截止日变更 → 非标过渡季（10-QT）</td><td><code>is_nonstandard_duration</code>',
            '<tr><td>B2</td><td>财年截止日变更 → 非标过渡季</td><td>12/31→9/30 + 短过渡季</td><td><code>is_nonstandard_duration</code>',
        ),
        (
            '<tr><td>B3</td><td>主板退市转 OTC 粉单 · Tick Size 突变</td><td><code>tick_size_mask</code>',
            '<tr><td>B3</td><td>主板退市转 OTC 粉单</td><td>摘牌转粉单 $0.01→$0.0001</td><td><code>tick_size_mask</code>',
        ),
        (
            '<tr><td>B4</td><td>Form T / Late Prints 延迟上报污染分钟 H/L</td><td>条件码',
            '<tr><td>B4</td><td>Form T / Late Prints</td><td>14:00 成交 16:45 进 SIP</td><td>条件码',
        ),
        (
            '<tr><td>B5</td><td>新闻多源异步重发（SimHash 去重）</td><td><code>news_dup_suppressed</code>',
            '<tr><td>B5</td><td>新闻多源异步重发</td><td>同公告盘前/盘后重发</td><td><code>news_dup_suppressed</code>',
        ),
    ]
    for old, new in b_rows:
        if old in html:
            html = html.replace(old, new, 1)

    return html


def main():
    html = HTML.read_text(encoding="utf-8")

    # CSS for scenario boxes
    if ".box-scenario" not in html:
        html = html.replace(
            ".box-danger {",
            """.box-scenario {
      background: linear-gradient(135deg, #f0f7ff 0%, #e8f4fd 100%);
      border-left: 4px solid var(--accent, #2563eb);
      padding: 12px 16px;
      margin: 12px 0 16px;
      border-radius: 0 8px 8px 0;
      font-size: 0.92rem;
      line-height: 1.55;
    }
    .box-scenario strong { color: #1e40af; }
    .box-scenario ul { margin: 8px 0 0 1.1rem; }
    .box-danger {""",
            1,
        )

    html = inject_scenarios(html)
    html = re.sub(
        r"v3\.\d[^<]*",
        "v3.10 业务场景说明",
        html,
        count=1,
    )
    HTML.write_text(html, encoding="utf-8")
    print("business scenarios injected, lines:", html.count("\n") + 1)
    subprocess.run(["python3", str(BEAUTIFY)], check=True)


if __name__ == "__main__":
    main()
