#!/usr/bin/env python3
"""Inject multimodal six-trap ACL (fundamentals/spinoff/batch/ADR/semantic news) into HTML."""
from pathlib import Path
import re
import subprocess

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"
BEAUTIFY = Path(__file__).resolve().parent / "beautify_massive_html.py"

MM_SECTION = r"""
    <section id="sec-acl-mm">
      <div class="box-scenario" id="acl-mm-read-how">
        <strong>§多模态六陷阱 · 怎么读：</strong>
        面向 <strong>日/分钟量价 + 三大报表 PiT + 另类新闻流</strong> 三条数据轨在入模前的交叉污染；
        与 <a href="#sec-acl">§ACL</a>（IPO/停牌/MWCB 等<strong>交易微观</strong>）、
        <a href="#sec-acl-deep2">§ACL++</a>（交易所故障/Form T/OTC 等<strong>基础设施</strong>）<strong>刻意不重复</strong>。
        条目 M1～M4 与 <a href="#sec-acl-deep">§ACL+</a> 任务同实施；M5～M6 为本章<strong>新增</strong>。
      </div>
      <h2><span class="sec-badge">MM</span> 多模态入模前六陷阱 — 财务畸变·分拆·Batch-Filing·ADR·跨语种舆情</h2>
      <p class="section-desc">
        结合资产定价顶刊清洗规约与 Millennium/Point72 级 Feature Store 契约：在宽表传入
        <code>factor_engine</code> / QuantaAlpha <strong>前一秒</strong>，对基本面结构畸变、Spinoff 非传统除权、
        SEC 批量回补、Stock Dividend 股本守恒、<strong>ADR 跨国时间错配</strong>、<strong>跨语种新闻语义重叠</strong> 强制卡死。
      </p>

      <table class="table-schema">
        <thead><tr><th>#</th><th>陷阱（多模态）</th><th>典型业务场景</th><th>核心掩码</th><th>任务</th></tr></thead>
        <tbody>
          <tr><td>M1</td><td>负权益 / 零营收报表畸变</td><td>Hertz、AMC、零营收 Biotech</td><td><code>is_negative_equity</code>, <code>is_zero_revenue</code></td><td><code>PREPROC-026</code></td></tr>
          <tr><td>M2</td><td>Spinoff 资产剥离 Ex-date</td><td>GE→Vernova、PFE→Viatris 技术大跌 30%</td><td><code>is_spinoff_ex_day</code></td><td><code>PREPROC-027</code></td></tr>
          <tr><td>M3</td><td>SEC Batch-Filing 同日多期</td><td>EDGAR 故障后一日补多期 10-K/10-Q</td><td><code>is_batch_filing</code></td><td><code>PREPROC-028</code></td></tr>
          <tr><td>M4</td><td>Stock dividend 非对称股本</td><td>10 送 1 躺在 dividends 表</td><td>路由 <code>splits</code></td><td><code>PREPROC-029</code></td></tr>
          <tr><td>M5</td><td>ADR 跨国多重上市时间错配</td><td>TSM、ASML；母所已收盘、美股未开盘</td><td><code>is_adr_asset</code>, <code>parent_close_fx_adj</code></td><td><code>PREPROC-035</code></td></tr>
          <tr><td>M6</td><td>跨语种新闻语义重复</td><td>同一调查：Benzinga + 中文/欧洲媒体</td><td><code>is_semantic_duplicate</code></td><td><code>PREPROC-036</code></td></tr>
        </tbody>
      </table>

      <h3>MM-1 · 负资产负债表与零营收（估值因子数理防御）</h3>
      <div class="box-scenario">
        <strong>业务场景：</strong>极端危机、大空头、LBO 等可使 <code>total_equity</code>（账面价值）为<strong>负</strong>；
        研发期 Biotech 可<strong>多季 revenue=0</strong>——均为真实报表，非脏行。
        <ul>
          <li><strong>陷阱：</strong><code>B/P = equity / cap</code>、<code>S/P</code> 分子为负 → 截面 Rank 把濒临破产股排成「极致低估成长」，毒化线性多因子基底。</li>
        </ul>
      </div>
      <h4>传入因子前处理</h4>
      <ul>
        <li>禁止填 0 或删行；物化 <code>is_negative_equity=1</code>、<code>is_zero_revenue=1</code></li>
        <li>估值/质量（ROE/ROA/B/P）：元数据强制 <strong>截面 Exclusion</strong> 或符号感知 winsorize</li>
      </ul>
      <p><strong>任务：</strong> <code>PREPROC-026</code>（P1）</p>

      <h3>MM-2 · Spinoff 分拆：非 split/div 的价格「技术性暴跌」</h3>
      <div class="box-scenario">
        <strong>业务场景：</strong>Ex-date 母公司资产依法剥离，开盘可<strong>跌 30%+</strong>；<code>splits</code> 无因子、<code>dividends</code> 捕不到。
        <ul>
          <li><strong>陷阱：</strong>裸 SIP → 假「技术面破位/强空」→ 回测误触发止损或虚假做空。</li>
        </ul>
      </div>
      <h4>传入因子前处理</h4>
      <ul>
        <li><code>corporate_actions</code> 解析 Spinoff 公告与剥离比率；Ex-date 打 <code>is_spinoff_ex_day=1</code></li>
        <li>R 轨：视同<strong>特殊非现金红利</strong>，Ex-date 前历史价 × 修正乘数前向平滑；动量算子可抹平该非交易跳空</li>
      </ul>
      <p><strong>任务：</strong> <code>PREPROC-027</code>（P1）+ <code>PREPROC-003</code></p>

      <h3>MM-3 · Batch-Filing：同日多期财报的 PiT 血缘</h3>
      <div class="box-scenario">
        <strong>业务场景：</strong>EDGAR 故障或调查延迟后，<strong>同一交易日</strong>批量公开积压的 10-K/10-Q，<code>filing_date</code> 相同。
        <ul>
          <li><strong>陷阱：</strong>简单 <code>merge_asof</code> / 按时间去重 → TTM 随机覆盖，阶梯函数断裂。</li>
        </ul>
      </div>
      <h4>传入因子前处理</h4>
      <ul>
        <li>同 ticker 同 <code>filing_date</code>：<strong>一级</strong> filing_date，<strong>二级 tie-breaker</strong> <code>period_end</code>（由远及近覆盖）</li>
        <li>当日出现批量回补 → <code>is_batch_filing=1</code>（驱动 CH/内存排序契约）</li>
      </ul>
      <p><strong>任务：</strong> <code>PREPROC-028</code>（P1）</p>

      <h3>MM-4 · Stock dividend：市值守恒的股本对偶</h3>
      <div class="box-scenario">
        <strong>业务场景：</strong>每 10 股送 1 股（10% stock dividend），常在 <code>dividends</code> 表且 <code>cash_amount=0</code>。
        <ul>
          <li><strong>陷阱：</strong>只调价不调股本 → <code>market_cap</code> 在除权日<strong>虚假蒸发 10%</strong>。</li>
        </ul>
      </div>
      <h4>传入因子前处理</h4>
      <ul>
        <li><code>stock_dividend</code> 且涉及股本变更 → <strong>切离 dividends</strong>，路由 <code>splits</code>（如 1:1.1）</li>
        <li>价 × 因子、股本 ÷ 因子（与 <code>PREPROC-017</code> volume 对偶一致）</li>
      </ul>
      <p><strong>任务：</strong> <code>PREPROC-029</code>（P1）</p>

      <h3>MM-5 · ADR 跨国多重上市：母所时间错配（新增）</h3>
      <div class="box-scenario">
        <strong>业务场景：</strong>美股 CS 池含大量 <strong>ADR</strong>（如 <code>TSM</code>、<code>ASML</code>）：RTH 在美股，基本面与母股在境外主所。
        母所常在美东<strong>盘前已收盘</strong>，财报/新闻 UTC 时间可能落在<strong>美股当日尚未开盘</strong>的日 K/分钟 grid 上。
        <ul>
          <li><strong>陷阱：</strong>未对齐 ADR parity → 分钟技术因子拟合「跨国时间扭曲虚假套利」。</li>
        </ul>
      </div>
      <h4>传入因子前处理</h4>
      <ul>
        <li><code>ticker_types</code> / issuer locale → <code>is_adr_asset=1</code></li>
        <li>Layer 2 挂载 <code>parent_close_fx_adj</code>（母所最新收盘×汇率）；1min 特征分母可做<strong>套利价差中性化</strong></li>
        <li>海外事件 <code>knowledge_ts_utc</code> 不得早于美股当日首笔 RTH bar（与 <code>PREPROC-022</code> 联动）</li>
      </ul>
      <p><strong>任务：</strong> <code>PREPROC-035</code>（P1）— 新增加密列 + ADR 元数据表。</p>

      <h3>MM-6 · 跨语种新闻：Entity+Subject 语义去重（新增）</h3>
      <div class="box-scenario">
        <strong>业务场景：</strong>同一事件（反垄断、重大技术）由 Benzinga/彭博与<strong>中文、欧洲</strong>媒体在不同时段报道；
        纯 SimHash 因语种不同<strong>无法重合</strong>。
        <ul>
          <li><strong>陷阱：</strong>时间轴上堆叠 2～3 次「独立舆情暴击」→ News Volume Shock / 情绪冲击被放大数倍。</li>
          <li>与 <a href="#sec-acl-deep2">ACL-B5</a> 同语种重发（<code>news_dup_suppressed</code> / <code>PREPROC-034</code>）<strong>正交</strong>：本节解决<strong>跨语种</strong>。</li>
        </ul>
      </div>
      <h4>传入因子前处理</h4>
      <ul>
        <li>升级去重：<strong>Entity + Subject</strong>（FinGPT/规则实体+事件主题），24h 窗口内同 Ticker 同一核心事件 → <code>is_semantic_duplicate=1</code></li>
        <li>情绪分可保留；<code>news_count_1d</code> 等计数<strong>不累加</strong></li>
      </ul>
      <p><strong>任务：</strong> <code>PREPROC-036</code>（P1）；与 <code>PREPROC-016</code>/<code>034</code> 分层实现。</p>

      <h3>多模态前置控制面（<code>panel_daily</code> 增补）</h3>
      <table class="table-schema">
        <thead><tr><th>数据轨</th><th>掩码列</th><th>业务含义</th><th>引擎硬红线</th></tr></thead>
        <tbody>
          <tr><td>基本面</td><td><code>is_negative_equity</code></td><td>账面权益为负</td><td>估值 B/P、ROE 等截面剔除，防符号反转</td></tr>
          <tr><td>公司行动</td><td><code>is_spinoff_ex_day</code></td><td>重大 Spinoff 执行日</td><td>动量/跳空：非交易性暴跌，可抹平断层</td></tr>
          <tr><td>基本面</td><td><code>is_batch_filing</code></td><td>单日批量回补多期财报</td><td>强制 <code>period_end</code> tie-break；TTM 血缘单调</td></tr>
          <tr><td>高频量价</td><td><code>is_adr_asset</code></td><td>ADR 存托凭证</td><td>1min 特征用 <code>parent_close_fx_adj</code> 价差中性化</td></tr>
          <tr><td>另类舆情</td><td><code>is_semantic_duplicate</code></td><td>跨语种同一事件重发</td><td>舆情计数归零；情绪分可保留</td></tr>
        </tbody>
      </table>
      <pre>ALTER TABLE qs_massive.panel_daily ADD COLUMN IF NOT EXISTS is_spinoff_ex_day      UInt8 DEFAULT 0;
ALTER TABLE qs_massive.panel_daily ADD COLUMN IF NOT EXISTS is_batch_filing         UInt8 DEFAULT 0;
ALTER TABLE qs_massive.panel_daily ADD COLUMN IF NOT EXISTS is_adr_asset              UInt8 DEFAULT 0;
ALTER TABLE qs_massive.panel_daily ADD COLUMN IF NOT EXISTS parent_close_fx_adj       Nullable(Float64);
ALTER TABLE qs_massive.panel_daily ADD COLUMN IF NOT EXISTS is_semantic_duplicate   UInt8 DEFAULT 0;
-- is_negative_equity / is_zero_revenue: PREPROC-026
-- news_dup_suppressed (同语种): PREPROC-034 · sec-acl-deep2</pre>

      <div class="box-info">
        <strong>闭环：</strong>多模态六陷阱 + <a href="#sec-acl">§ACL</a> + <a href="#sec-acl-deep2">§ACL++</a> →
        量价、三大报表 PiT、另类新闻在入模前<strong>分轨防腐</strong>；工程组将上表合入 CH 视图 / 物化脚本即可封板评审。
      </div>
    </section>
"""

TASK_ROWS = """          <tr><td>PREPROC-035</td><td><span class="badge badge-p1">P1</span></td><td>ADR 打标 is_adr_asset + parent_close_fx_adj 分钟价差中性化</td><td class="owner-cell">数据工程组</td><td class="status-cell">待办</td><td></td></tr>
          <tr><td>PREPROC-036</td><td><span class="badge badge-p1">P1</span></td><td>跨语种新闻 Entity+Subject 去重 is_semantic_duplicate</td><td class="owner-cell">量化研究组</td><td class="status-cell">待办</td><td></td></tr>
"""

EDGE_ROWS = """          <tr>
            <td><strong>ADR 跨国上市</strong></td>
            <td>TSM/ASML；母所先收盘、美股未开盘</td>
            <td><code>is_adr_asset</code>、<code>parent_close_fx_adj</code></td>
            <td>1min 价差中性化；PiT 不早于 RTH</td>
            <td>裸 ADR 价当本币</td>
            <td><code>PREPROC-035</code></td>
          </tr>
          <tr>
            <td><strong>跨语种新闻语义重复</strong></td>
            <td>同事件多语种媒体</td>
            <td><code>is_semantic_duplicate</code>；Entity+Subject</td>
            <td>news_count 不累加</td>
            <td>仅 SimHash 同语种</td>
            <td><code>PREPROC-036</code></td>
          </tr>
"""

COL_ROWS = """          <tr><td><code>is_spinoff_ex_day</code></td><td>UInt8</td><td>Spinoff 执行日</td><td>PREPROC-027</td></tr>
          <tr><td><code>is_batch_filing</code></td><td>UInt8</td><td>单日批量财报回补</td><td>PREPROC-028</td></tr>
          <tr><td><code>is_adr_asset</code></td><td>UInt8</td><td>ADR 存托凭证</td><td>PREPROC-035</td></tr>
          <tr><td><code>parent_close_fx_adj</code></td><td>Float64?</td><td>母所收盘（汇率换算）</td><td>PREPROC-035</td></tr>
          <tr><td><code>is_semantic_duplicate</code></td><td>UInt8</td><td>跨语种同一事件</td><td>PREPROC-036</td></tr>
"""


def main():
    html = HTML.read_text(encoding="utf-8")

    if '<section id="sec-acl-mm">' not in html:
        html = html.replace(
            '    <section id="sec-acl-deep">\n',
            MM_SECTION.strip() + '\n\n    <section id="sec-acl-deep">\n',
            1,
        )

    # sec-acl-deep pointer
    if "sec-acl-mm" not in html.split("sec-acl-deep")[1][:400] if "sec-acl-deep" in html else "":
        html = html.replace(
            '<section id="sec-acl-deep">\n      <div class="box-scenario" id="aclplus-read-how">',
            '<section id="sec-acl-deep">\n      <p class="muted">条目 D1～D4 与 <a href="#sec-acl-mm">§多模态六陷阱 MM-1～M4</a> 同义；本节保留任务索引。</p>\n      <div class="box-scenario" id="aclplus-read-how">',
            1,
        )

    if "PREPROC-035" not in html:
        m = (
            '<tr><td>PREPROC-034</td><td><span class="badge badge-p1">P1</span></td>'
            "<td>news_dup_suppressed + 24h SimHash（升级 016）</td>"
            '<td class="owner-cell">量化研究组</td><td class="status-cell">待办</td><td></td></tr>\n'
            "          <tr><td>PREPROC-011</td>"
        )
        if m in html:
            html = html.replace(m, m.replace("          <tr><td>PREPROC-011", TASK_ROWS + "          <tr><td>PREPROC-011"), 1)

    if "PREPROC-035</code></td>" not in html.split("sec-edge-cases")[1][:20000] if "sec-edge-cases" in html else "":
        html = html.replace(
            '<tr>\n            <td><strong>新股 / IPO / Spin-off 首日</strong></td>',
            EDGE_ROWS + '<tr>\n            <td><strong>新股 / IPO / Spin-off 首日</strong></td>',
            1,
        )

    if "<code>is_adr_asset</code></td><td>UInt8" not in html:
        if '<tr><td><code>news_dup_suppressed</code></td><td>UInt8</td><td>新闻重发抑制</td><td>PREPROC-034</td></tr>' in html:
            html = html.replace(
                '<tr><td><code>news_dup_suppressed</code></td><td>UInt8</td><td>新闻重发抑制</td><td>PREPROC-034</td></tr>',
                '<tr><td><code>news_dup_suppressed</code></td><td>UInt8</td><td>新闻重发抑制（同语种）</td><td>PREPROC-034</td></tr>\n' + COL_ROWS.strip(),
                1,
            )
        elif '<tr><td><code>compute_purified_mask</code></td><td>UInt8</td><td>净化总控' in html:
            html = html.replace(
                '<tr><td><code>compute_purified_mask</code></td><td>UInt8</td><td>净化总控',
                COL_ROWS.strip() + '\n          <tr><td><code>compute_purified_mask</code></td><td>UInt8</td><td>净化总控',
                1,
            )

    # acl summary row
    if "sec-acl-mm" not in html.split("sec-acl")[1][:4000] if "sec-acl" in html else "":
        html = html.replace(
            '<tr><td>+</td><td>深水五极端（财务/分拆/EDGAR）</td>',
            '<tr><td>+</td><td><a href="#sec-acl-mm">§多模态六陷阱</a>（量价+财报+新闻）</td><td><code>is_adr_asset</code> 等</td><td>MM <code>026~036</code></td></tr>\n'
            '          <tr><td>+</td><td>深水五极端（财务/分拆/EDGAR）</td>',
            1,
        )

    if "§多模态六陷阱" not in html:
        html = html.replace(
            '<a href="#sec-acl-deep"><strong>§ACL+</strong></a>',
            '<a href="#sec-acl-mm"><strong>§MM 多模态</strong></a> <a href="#sec-acl-deep"><strong>§ACL+</strong></a>',
            1,
        )

    if "多模态六陷阱" not in html and "防腐层八大极端" in html:
        html = html.replace(
            '<tr><td>财务/分拆/EDGAR 深水五极端</td>',
            '<tr><td>多模态六陷阱（财报+ADR+跨语种新闻）</td><td><a href="#sec-acl-mm">§MM</a></td></tr>\n          <tr><td>财务/分拆/EDGAR 深水五极端</td>',
            1,
        )

    if "PREPROC-035" not in html and "§ACL++" in html:
        html = html.replace(
            '<li><a href="#sec-acl-deep2">§ACL++</a> 已评审',
            '<li><a href="#sec-acl-mm">§多模态六陷阱</a> 已评审（<code>PREPROC-035~036</code> ADR/跨语种）</li>\n        <li><a href="#sec-acl-deep2">§ACL++</a> 已评审',
            1,
        )

    html = re.sub(
        r"v3\.\d[^<]*",
        "v3.12 多模态六陷阱+ADR+跨语种",
        html,
        count=1,
    )

    HTML.write_text(html, encoding="utf-8")
    print("ACL multimodal six injected, lines:", html.count("\n") + 1)

    bp = BEAUTIFY.read_text(encoding="utf-8")
    if '"sec-acl-deep"' in bp and '"sec-acl-mm"' not in bp:
        bp = bp.replace(
            '"sec-acl", "sec-acl-deep"',
            '"sec-acl", "sec-acl-mm", "sec-acl-deep"',
        )
        BEAUTIFY.write_text(bp, encoding="utf-8")
    subprocess.run(["python3", str(BEAUTIFY)], check=True)


if __name__ == "__main__":
    main()
