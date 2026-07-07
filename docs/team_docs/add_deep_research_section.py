#!/usr/bin/env python3
"""Inject AI Deep Research report (adopted + corrections) into governance HTML."""
import re
from pathlib import Path

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"
MD = Path(__file__).resolve().parent / "美股原始数据预处理标准方案_因子入模前.md"

SECTION = r"""
    <section id="sec-deep-research">
      <h2><span class="sec-badge">12f</span> AI Deep Research 采纳与勘误（异构海量入模前沿预处理）</h2>
      <p class="section-desc">
        本节吸收外部深度研究报告中的<strong>可落地准则</strong>，并与本库 Massive 实证、既有 §12a–12e 任务对齐。
        已覆盖内容仅做索引；<strong>新增</strong>项登记为 <code>PREPROC-023/024</code>、<code>NLP-001~003</code>。
      </p>

      <div class="box-warn">
        <strong>Massive 勘误（报告泛化表述 → 本库正确做法）：</strong>
        <ul>
          <li><strong>复权：</strong>REST <code>daily_market_summary</code> 仅为<strong>拆股复权</strong>（API <code>adjusted=true</code>），<strong>不含现金分红</strong>；全收益回测需 SIP 未复权价 + <code>splits</code>/<code>dividends</code> 的 <code>historical_adjustment_factor</code>（<code>PREPROC-003/017</code>），勿照搬教科书 CRSP 公式覆盖供应商因子。</li>
          <li><strong>PiT：</strong>本库 cleaned 财报默认 <code>keep_latest_by_align_time</code> — <strong>不能</strong>直接用于严格回测；须 <code>fundamentals_pit</code>（<code>PREPROC-005</code> / <code>FEAT-002</code>）。FactSet/Compustat 级全版本 PiT 为<strong>目标架构</strong>，非当前 parquet 既有字段。</li>
          <li><strong>Layer 1 边界：</strong><code>massive_cleaning_framework.py</code> <strong>不改 OHLCV</strong>、不做跨源 join；报告中的「异常值剔除」若指 tick，属于 <strong>Layer 2 可选特征工程</strong>，且须区分集合竞价（见 <code>PREPROC-024</code>），与 §12d 截面 MAD（<code>PREPROC-010</code>，日频 panel）不是同一层。</li>
          <li><strong>分钟≠日 K：</strong>报告「98% 不一致」与本库 §1b 一致 — 已铁律化，禁止 <code>sum(minute.volume)==day.volume</code>。</li>
          <li><strong>截断 / .ok：</strong>P0 清单与 <code>TASK-ENG-001/002</code>、<code>TASK-DATA-001~003</code> 已覆盖；研究端暂冻结 2024 财报/dividends/risk_factors 直至重下验收。</li>
        </ul>
      </div>

      <h3>报告论点 → 本手册映射（避免重复阅读）</h3>
      <table class="table-schema">
        <thead><tr><th>报告主题</th><th>本库状态</th><th>详见</th></tr></thead>
        <tbody>
          <tr><td>Layer 0/1/2 物理隔离</td><td>✅ 已确立</td><td><a href="#sec-arch">§0</a>、<a href="#sec-before-factor">★入模前</a></td></tr>
          <tr><td>REST vs SIP 日 K 禁混</td><td>✅ 铁律</td><td><a href="#sec-data-catalog">§数据手册</a>、<a href="#sec-p1">§3</a></td></tr>
          <tr><td>filing_date PiT + asof backward</td><td>✅ + 任务</td><td><a href="#sec-time">§8</a>、<code>PREPROC-005</code></td></tr>
          <tr><td>财报重述勿用全样本 latest</td><td>✅ 已述</td><td><a href="#sec-empirical">§1b</a>、<code>FEAT-002</code></td></tr>
          <tr><td>新闻 tickers explode</td><td>✅ Layer 1</td><td><a href="#sec-data-catalog">§4.5.1 news</a></td></tr>
          <tr><td>MWCB / Ghost / 停牌 / IPO</td><td>✅ §12d</td><td><code>ENG-012/013</code>、<code>PREPROC-018~020</code></td></tr>
          <tr><td>分页 10k/2k + .ok 陷阱</td><td>✅ P0</td><td><a href="#sec-p0">§2</a>、<code>TASK-ENG-001</code></td></tr>
          <tr><td>「勿盲目清洗 tick」+ 集合竞价保护</td><td>🆕 采纳</td><td><code>PREPROC-024</code>、<code>TASK-DC-005</code></td></tr>
          <tr><td>同时间戳 quote/trade 聚合</td><td>🆕 采纳</td><td><code>PREPROC-023</code>（tick 特征层）</td></tr>
          <tr><td>HTML 样板剥离 + 工具选型</td><td>🆕 采纳</td><td><code>NLP-001</code>（待 10-K/8-K 下载）</td></tr>
          <tr><td>LLM 情绪前实体匿名化</td><td>🆕 采纳</td><td><code>NLP-002</code></td></tr>
          <tr><td>信源权威性加权</td><td>🆕 采纳</td><td><code>NLP-003</code></td></tr>
        </tbody>
      </table>

      <h3>一、高频量价：清洗边界（采纳 + 与 §12d 分工）</h3>
      <table class="table-schema">
        <thead><tr><th>类型</th><th>处理原则</th><th>Massive 落点</th></tr></thead>
        <tbody>
          <tr>
            <td><strong>损坏数据</strong></td>
            <td>价格≤0、成交量&lt;0、乱序截断 → <strong>硬删除</strong>（供应商故障）</td>
            <td>Layer 2 tick 管道；raw tick 未清洗</td>
          </tr>
          <tr>
            <td><strong>真异常 / 大宗 / 开收盘竞价</strong></td>
            <td><strong>禁止</strong>用全局 MAD/布林带当「坏 tick」删除；须用 <code>condition_codes</code> + 时段掩码</td>
            <td><code>PREPROC-024</code>：<code>is_opening_auction</code> / <code>is_closing_auction</code>；开盘 ~20s、收盘 ~3min 可配置豁免</td>
          </tr>
          <tr>
            <td><strong>同时间戳多笔</strong></td>
            <td>Quote：bid=max、ask=min（且 bid≤ask）、量求和；Trade：量求和 + VWAP</td>
            <td><code>PREPROC-023</code>；仅在做 tick 特征/微观因子时执行，<strong>非</strong> Layer 1 cleaned</td>
          </tr>
          <tr>
            <td><strong>跨粒度</strong></td>
            <td>日频因子只读 <code>day_aggs_v1</code>；分钟因子只读 <code>minute_aggs_v1</code></td>
            <td><a href="#sec-p1">§3 铁律③</a></td>
          </tr>
        </tbody>
      </table>

      <h3>二、复权：CRSP 思想 vs 本库字段</h3>
      <table class="table-schema">
        <thead><tr><th>概念</th><th>学术/CRSP</th><th>Massive 实现</th></tr></thead>
        <tbody>
          <tr><td>拆股</td><td>价格×因子、量÷因子；金额守恒</td><td><code>splits.historical_adjustment_factor</code> + <code>PREPROC-003</code>；REST 日线已拆股复权</td></tr>
          <tr><td>现金分红</td><td>股息乘数（非简单减法，防负价）</td><td><code>dividends.historical_adjustment_factor</code>（~20% null 需 fallback）；<code>PREPROC-017</code> <strong>分红不调 volume</strong></td></tr>
          <tr><td>价源铁律</td><td>单策略单一复权口径</td><td>生产锚点：<strong>SIP cleaned day_aggs</strong> + 自算复权；<strong>禁止</strong>与 REST 日线拼接</td></tr>
          <tr><td>精度</td><td>GE/TOPS 等高分红标的</td><td><code>PREPROC-019</code> floor clamp + <code>price_clamped_flag</code></td></tr>
        </tbody>
      </table>

      <h3>三、基本面 PiT 与 join（报告强调项 · 已部分落地）</h3>
      <table class="table-schema">
        <thead><tr><th>拼接模式</th><th>适用</th><th>键</th><th>实现</th></tr></thead>
        <tbody>
          <tr><td>Exact join</td><td>日/分钟价量 ↔ 同频衍生</td><td><code>align_time</code> + <code>ticker</code></td><td><code>Database_original.py</code> 价量 exact</td></tr>
          <tr><td>As-of backward</td><td>财报、宏观、文本事件</td><td><code>filing_date</code> → <code>align_time</code></td><td><code>composite_source.merge_asof(direction='backward')</code></td></tr>
          <tr><td>禁止</td><td>—</td><td><code>period_end</code> 作 asof 键</td><td>前视红线 · <a href="#sec-time">§8</a></td></tr>
          <tr><td>重述</td><td>同 period 多 filing</td><td>截止日可见<strong>最新披露版本</strong></td><td><code>PREPROC-005</code> / <code>FEAT-002</code>，勿 <code>keep_latest</code> 全历史</td></tr>
        </tbody>
      </table>
      <p class="section-desc">asof 后区间 NaN 的 <code>ffill</code> 仅允许使用<strong>过去</strong>已知值；事件研究新闻用 <code>PREPROC-009</code> 映射下一交易日。</p>

      <h3>四、非结构化文本：HTML → 情绪特征（新增管线）</h3>
      <table class="table-schema">
        <thead><tr><th>工具</th><th>特点</th><th>本库建议</th></tr></thead>
        <tbody>
          <tr><td>Beautiful Soup</td><td>站点定制、灵活</td><td>SEC EDGAR、彭博等<strong>固定版式</strong>定向规则</td></tr>
          <tr><td>Boilerpipe</td><td>快速、站点无关</td><td>大规模长尾网页正文抽取</td></tr>
          <tr><td>Fundus</td><td>高 F1、保留语义</td><td>新闻通稿默认引擎（优于 Trafilatura 激进裁剪）</td></tr>
          <tr><td>Trafilatura</td><td>激进删节点</td><td><strong>不推荐</strong>金融否定词场景（易情绪反转）</td></tr>
        </tbody>
      </table>
      <ul>
        <li><strong>Layer 1</strong>：新闻 <code>tickers[]</code> explode（已在 cleaning 配置）— 报告与 <a href="#sec-data-catalog">§4.5.1</a> 一致。</li>
        <li><strong>Layer 2 · NLP-001</strong>：10-K/8-K 下载（<code>FEAT-006</code>）后，样板剥离 → 纯文本列入库/进 CH。</li>
        <li><strong>Layer 2 · NLP-002</strong>：FinBERT/LLM 打分前，对目标公司名/ticker/高管做<strong>匿名占位</strong>，缓解训练语料前视与「注意力分散」。</li>
        <li><strong>Layer 2 · NLP-003</strong>：<code>source_authority_weight</code>（监管 &gt; 主流媒体 &gt; 社媒），禁止与 Reddit 散户帖等权混均。</li>
        <li>与 <code>PREPROC-016</code>（24h 去重）正交：先降噪 → 再去重 → 再打分。</li>
      </ul>

      <h3>五、工程红线（报告 P0 · 与本库一致）</h3>
      <table class="table-schema">
        <thead><tr><th>动作</th><th>命令/配置要点</th></tr></thead>
        <tbody>
          <tr><td>删假性 .ok</td><td><code>TASK-DATA-001</code> — 行数∈{2000,10000} 分区</td></tr>
          <tr><td>全量重拉</td><td><code>max_pages=None</code>、<code>--no-skip-existing</code>（<code>TASK-ENG-002</code>）</td></tr>
          <tr><td>暂停污染日更</td><td><code>daily_update_scheduler.sh</code> / <code>TASK-ENG-004</code></td></tr>
          <tr><td>研究降级</td><td>因子开发：价量用 SIP；基本面≤2023；禁 2024 截断表</td></tr>
        </tbody>
      </table>

      <h3>六、新增任务登记（Deep Research 专章）</h3>
      <table class="table-schema">
        <thead><tr><th>ID</th><th>阶段</th><th>摘要</th><th>优先级</th><th>验收</th></tr></thead>
        <tbody>
          <tr><td><code>PREPROC-023</code></td><td>Layer 2 tick</td><td>同时间戳 quote/trade 聚合协议（VWAP/买卖档）</td><td>P2</td><td>微观因子可复现；不改 cleaned parquet</td></tr>
          <tr><td><code>PREPROC-024</code></td><td>Layer 2 分钟/tick</td><td>集合竞价时段标记 + 禁止盲目 tick 删点</td><td>P1</td><td>开盘/收盘 bar 不被 MAD 当脏数据；condition_codes 可追溯</td></tr>
          <tr><td><code>NLP-001</code></td><td>Layer 2 文本</td><td>HTML 样板剥离（Fundus+SEC 定向规则）</td><td>P2</td><td>10-K/8-K 正文列非空；样板率抽检 &lt;5%</td></tr>
          <tr><td><code>NLP-002</code></td><td>Layer 2 文本</td><td>LLM/FinBERT 打分前实体匿名化</td><td>P2</td><td>样本外 IC 优于裸文本基线（内部 A/B）</td></tr>
          <tr><td><code>NLP-003</code></td><td>Layer 2 文本</td><td>信源权威性权重特征</td><td>P2</td><td>权重表版本化；与 sentiment 分列存储</td></tr>
        </tbody>
      </table>
      <div class="box-info">
        以上 5 项已同步至 <a href="#sec-tasks">§14 任务总表</a> 与 <a href="美股原始数据预处理标准方案_因子入模前.md">预处理标准方案 §21.3</a>。
      </div>
    </section>

"""

TASK_ROWS = """
          <tr><td>PREPROC-023</td><td><span class="badge badge-p2">P2</span></td><td>tick 同时间戳聚合（quote/trade VWAP 协议）</td><td class="owner-cell">数据工程组</td><td class="status-cell">待办</td><td></td></tr>
          <tr><td>PREPROC-024</td><td><span class="badge badge-p1">P1</span></td><td>集合竞价时段标记 + 禁止盲目 tick 异常剔除</td><td class="owner-cell">量化研究组</td><td class="status-cell">待办</td><td></td></tr>
          <tr><td>NLP-001</td><td><span class="badge badge-p2">P2</span></td><td>HTML 样板剥离（Fundus/BS4，10-K/8-K）</td><td class="owner-cell">量化研究组</td><td class="status-cell">待办</td><td></td></tr>
          <tr><td>NLP-002</td><td><span class="badge badge-p2">P2</span></td><td>LLM 情绪打分前实体匿名化</td><td class="owner-cell">量化研究组</td><td class="status-cell">待办</td><td></td></tr>
          <tr><td>NLP-003</td><td><span class="badge badge-p2">P2</span></td><td>新闻信源权威性权重 source_authority_weight</td><td class="owner-cell">量化研究组</td><td class="status-cell">待办</td><td></td></tr>
"""

TABLE_C = """
      <h3>表 C · AI Deep Research 采纳（§12f）</h3>
      <table class="table-schema">
        <thead><tr><th>阶段</th><th>任务 ID</th><th>处理项</th><th>负责人</th><th>验收红线</th></tr></thead>
        <tbody>
          <tr><td>Layer 2 tick</td><td><code>PREPROC-023</code></td><td>同时间戳 quote/trade 聚合（VWAP/档位）</td><td>数据工程组</td><td>微观特征可复现；不改 Layer1</td></tr>
          <tr><td>Layer 2 分钟</td><td><code>PREPROC-024</code></td><td>集合竞价豁免 + is_auction 标记</td><td>量化研究组</td><td>开收盘 bar 不被当脏 tick 删除</td></tr>
          <tr><td>Layer 2 NLP</td><td><code>NLP-001</code></td><td>HTML 样板剥离管线</td><td>量化研究组</td><td>依赖 FEAT-006 全文下载</td></tr>
          <tr><td>Layer 2 NLP</td><td><code>NLP-002</code></td><td>LLM 情绪实体匿名化</td><td>量化研究组</td><td>样本外优于裸文本</td></tr>
          <tr><td>Layer 2 NLP</td><td><code>NLP-003</code></td><td>信源权威性权重</td><td>量化研究组</td><td>权重表版本化</td></tr>
        </tbody>
      </table>
"""

MD_APPEND = """
### 21.3 AI Deep Research 采纳（2026-06）

| ID | 摘要 | 优先级 |
|----|------|--------|
| PREPROC-023 | tick 同时间戳 quote/trade 聚合 | P2 |
| PREPROC-024 | 集合竞价时段标记；禁止盲目 tick 删点 | P1 |
| NLP-001 | HTML 样板剥离（Fundus + SEC 定向） | P2 |
| NLP-002 | LLM/FinBERT 打分前实体匿名化 | P2 |
| NLP-003 | source_authority_weight 信源权重 | P2 |

**勘误：** REST 日线仅拆股复权；严格 PiT 须 `fundamentals_pit`；CRSP 乘数思想对齐 `historical_adjustment_factor`，勿与 REST 混源。详见治理清单 HTML `#sec-deep-research`。
"""


def main():
    html = HTML.read_text(encoding="utf-8")

    if "sec-deep-research" not in html:
        html = html.replace(
            "\n    <!-- 7. Task table -->\n    <section id=\"sec-tasks\">",
            "\n" + SECTION + "\n    <!-- 7. Task table -->\n    <section id=\"sec-tasks\">",
            1,
        )
    else:
        html = re.sub(
            r'<section id="sec-deep-research">.*?</section>',
            SECTION.strip(),
            html,
            count=1,
            flags=re.DOTALL,
        )

    if 'PREPROC-023</td><td><span class="badge badge-p2">P2</span></td><td>tick' not in html:
        html = html.replace(
            '<tr><td>CH-007</td><td><span class="badge badge-p2">P2</span></td><td>fact_bars_adjusted_minute CH 分区</td>',
            TASK_ROWS + '          <tr><td>CH-007</td><td><span class="badge badge-p2">P2</span></td><td>fact_bars_adjusted_minute CH 分区</td>',
            1,
        )

    if "表 C · AI Deep Research" not in html:
        html = html.replace(
            '<div class="box-info">\n        <strong>评审稿 ID 对照：</strong>',
            TABLE_C + '\n      <div class="box-info">\n        <strong>评审稿 ID 对照：</strong>',
            1,
        )
    html = html.replace(
        "汇总两轮评审共 <strong>15 项</strong>入模前任务",
        "汇总评审 + Deep Research 共 <strong>20 项</strong>入模前任务（表 A/B + <a href=\"#sec-deep-research\">表 C §12f</a>）",
    )

    html = html.replace(
        "版本 v3.3 HTML 全库数据形态手册",
        "版本 v3.4 Deep Research 采纳（12f）+ 全库数据手册",
    )
    if "v3.4" not in html:
        html = re.sub(
            r"版本 v3\.\d[^·]*",
            "版本 v3.4 Deep Research 采纳（12f）+ 全库数据手册",
            html,
            count=1,
        )

    HTML.write_text(html, encoding="utf-8")
    print("HTML updated, lines:", html.count("\n") + 1)

    if MD.exists():
        md = MD.read_text(encoding="utf-8")
        if "21.3 AI Deep Research" not in md:
            md = md.rstrip() + "\n" + MD_APPEND
            MD.write_text(md, encoding="utf-8")
            print("MD §21.3 appended")


if __name__ == "__main__":
    main()
