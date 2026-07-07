#!/usr/bin/env python3
"""Inject ACL deep-five-B (exchange/fiscal/OTC/late-print/news) into HTML deliverable."""
from pathlib import Path
import re
import subprocess

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"
BEAUTIFY = Path(__file__).resolve().parent / "beautify_massive_html.py"

DEEP2_SECTION = r"""
    <section id="sec-acl-deep2">
      <h2><span class="sec-badge">ACL++</span> 防腐层深水区五极端（B）— 交易所·财年·OTC·延迟成交·新闻重发</h2>
      <p class="section-desc">
        在 <a href="#sec-acl">§ACL</a>、<a href="#sec-acl-deep">§ACL+</a> 已覆盖的 IPO/停牌/MWCB/财务畸变等之外，
        还存在由<strong>交易所系统故障、财年变更过渡季、OTC 摘牌、Form T 延迟上报、新闻异步重发</strong>引发的隐蔽陷阱。
        必须在 Layer 1.5～2 物化或 CH 导入前卡死；条件码协议见 <code>TASK-DC-005</code> / <a href="#sec-code-assets">tape_condition_filter.yaml</a>。
      </p>

      <table class="table-schema">
        <thead><tr><th>#</th><th>情形</th><th>核心掩码</th><th>任务</th></tr></thead>
        <tbody>
          <tr><td>B1</td><td>裁决作废错单 / 交易所闪崩（如 2024-06-03 NYSE CTA）</td><td><code>is_erroneous_glitch</code></td><td><code>PREPROC-030</code></td></tr>
          <tr><td>B2</td><td>财年截止日变更 → 非标过渡季（10-QT）</td><td><code>is_nonstandard_duration</code></td><td><code>PREPROC-031</code></td></tr>
          <tr><td>B3</td><td>主板退市转 OTC 粉单 · Tick Size 突变</td><td><code>tick_size_mask</code>, <code>delisting_return</code></td><td><code>PREPROC-032</code></td></tr>
          <tr><td>B4</td><td>Form T / Late Prints 延迟上报污染分钟 H/L</td><td>条件码分流 resampling</td><td><code>PREPROC-033</code></td></tr>
          <tr><td>B5</td><td>新闻多源异步重发（SimHash 去重）</td><td><code>news_dup_suppressed</code></td><td><code>PREPROC-034</code></td></tr>
          <tr><td>—</td><td>SIFMA 提前收盘（与 ACL-7 同）</td><td><code>is_early_close</code></td><td><code>ENG-011</code></td></tr>
        </tbody>
      </table>

      <h3>ACL-B1 · 裁决作废交易（Erroneous / Adjudicated Trades）</h3>
      <div class="box-danger">
        <strong>盲区：</strong>2024-06-03 等 CTA 故障致 BRK.A/BMO 等盘中闪崩 99.97% 后官方作废；
        原始 <code>trades_v1</code>/<code>minute_aggs_v1</code> 仍含脏脉冲 → 日内 H/L、回撤、振幅截面失真。
      </div>
      <ul>
        <li>Layer 1.5：<code>trades_v1</code> 筛 <code>correction</code> + <code>conditions</code>，剔除作废/更正行（<code>TASK-DC-005</code>）</li>
        <li>分钟：滚动 5×MAD 偏离过滤器；裁决时段 OHLC <strong>回滚</strong> 至前一分钟公允价，成交量扣作废笔</li>
        <li>Layer 2：<code>is_erroneous_glitch=1</code>；<code>ts_*</code> 波动率算子<strong>不 push</strong> 该 bar</li>
      </ul>
      <p><strong>任务：</strong> <code>PREPROC-030</code>（P1）— 与 <code>ENG-013</code> Ghost 滤除正交。</p>

      <h3>ACL-B2 · 财年截止日变更与非标过渡季（Fiscal Year-End Shift）</h3>
      <div class="box-danger">
        <strong>盲区：</strong>12/31→9/30 等变更产生 1～2 月 <code>10-QT</code>，<code>timeframe=quarterly</code> 仍成立；
        机械 TTM 加总 4 行 → 把 1 个月当季当 3 个月，TTM 虚假暴跌 20%+。
      </div>
      <ul>
        <li>Duration = <code>period_end</code> 与上期差；&lt;75 或 &gt;105 日历天 → <code>is_nonstandard_duration=1</code></li>
        <li>TTM 前：<strong>时间权重线性年化缩放</strong>（Time-weighted Scaling），或该期排除出四期滚动加总</li>
      </ul>
      <p><strong>任务：</strong> <code>PREPROC-031</code>（P1）— <code>pit_fundamentals</code> / LTM 管线联动 <a href="#sec-stage6-ltm">§Stage6 LTM</a>。</p>

      <h3>ACL-B3 · 退市转 OTC 粉单：Tick Size 与幸存者偏差</h3>
      <div class="box-danger">
        <strong>盲区：</strong>主板 Last Day 后 Tick 从 $0.01→$0.0001；若在 universe 直接蒸发 → 回测躲过破产踩踏。
      </div>
      <ul>
        <li>状态机：<code>is_delisting_day=1</code> + <code>delisting_return</code> 惩罚（-100% 或 OTC 首日首价，<code>PREPROC-012</code> 扩展）</li>
        <li>若跟踪 OTC 分钟/tick：<code>tick_size_mask</code> 自适应 0.0001；LOB 重构禁止仍用 0.01</li>
        <li>禁止退市当日从截面「物理消失」而不记惩罚收益</li>
      </ul>
      <p><strong>任务：</strong> <code>PREPROC-032</code>（P1）— 扩展 <code>fact_delisting_events</code> + 元数据 <code>delisting_policy</code>。</p>

      <h3>ACL-B4 · Form T / Late Prints 延迟上报（Out-of-Sequence）</h3>
      <div class="box-danger">
        <strong>盲区：</strong>14:00 大宗 16:45 才进 SIP（Form T/Late）→ 按 16:45 重采样会造虚假长影线与 vol 脉冲。
      </div>
      <ul>
        <li><strong>Condition-Based Resampling Filter</strong>：延迟/乱序条件码成交 — <strong>volume 可累加</strong>，<strong>price 禁止参与该分钟 H/L</strong></li>
        <li>与 <code>PREPROC-024</code> 集合竞价标记、<code>TASK-DC-005</code> 条件码表联动</li>
      </ul>
      <p><strong>任务：</strong> <code>PREPROC-033</code>（P1）— 1min 物化管线硬约束。</p>

      <h3>ACL-B5 · 新闻多源异步重发（News Duplicate Propagation）</h3>
      <div class="box-danger">
        <strong>盲区：</strong>同一公告 15:46 / 16:30 / 次日 08:30 换标题重发 → <code>news_count_1d</code> 虚增 3 倍，破坏独立性。
      </div>
      <ul>
        <li>24h（可按 ticker）窗口：<code>title</code>/摘要 SimHash 或 3-gram Jaccard &gt;0.85 → <code>news_dup_suppressed=1</code></li>
        <li>情绪分可保留；<strong><code>news_count_*</code> 不累加</strong>（升级 <code>PREPROC-016</code> 为 P1 契约）</li>
      </ul>
      <p><strong>任务：</strong> <code>PREPROC-034</code>（P1）— 与 §10c.15 <code>PREPROC-016</code> 折叠代码同实现。</p>

      <h3>Layer 2 前置控制面（Purified Control Plane）</h3>
      <table class="table-schema">
        <thead><tr><th>层</th><th>列</th><th>含义</th><th>引擎红线</th></tr></thead>
        <tbody>
          <tr><td>L1.5</td><td><code>is_erroneous_glitch</code></td><td>交易所裁决作废闪崩分钟</td><td>滚动波动率跳过该 bar</td></tr>
          <tr><td>L1.5</td><td><code>is_early_close</code></td><td>SIFMA 13:00 提前收盘</td><td>骨架 210 bar，13:00 后禁止 ffill</td></tr>
          <tr><td>L2</td><td><code>is_nonstandard_duration</code></td><td>财年变更过渡季</td><td>TTM 触发时间权重缩放</td></tr>
          <tr><td>L2</td><td><code>news_dup_suppressed</code></td><td>24h 内媒体重发</td><td>舆情计数不累加</td></tr>
          <tr><td>L2</td><td><code>compute_purified_mask</code></td><td><strong>总控</strong>：无上游污染、可进 Ring Buffer</td><td>=0 冻结 <code>ts_*</code>；与 <code>compute_mask</code> 同义</td></tr>
        </tbody>
      </table>
      <pre>ALTER TABLE qs_massive.panel_daily ADD COLUMN IF NOT EXISTS is_erroneous_glitch    UInt8 DEFAULT 0;
ALTER TABLE qs_massive.panel_daily ADD COLUMN IF NOT EXISTS is_nonstandard_duration UInt8 DEFAULT 0;
ALTER TABLE qs_massive.panel_daily ADD COLUMN IF NOT EXISTS news_dup_suppressed     UInt8 DEFAULT 0;
ALTER TABLE qs_massive.panel_daily ADD COLUMN IF NOT EXISTS compute_purified_mask   UInt8 DEFAULT 1;

compute_purified_mask = compute_mask  -- 主控别名（推荐引擎统一字段名）
  AND NOT is_erroneous_glitch;
-- compute_mask 已含: halt, ipo_pre_open, price_clamped, universe, has_bar
-- 基本面 TTM 另读 is_nonstandard_duration; 新闻计数另读 news_dup_suppressed</pre>

      <div class="box-info">
        <strong>工业级闭环：</strong>§ACL + §ACL+ + 本节五极端 + <code>PREPROC-021</code>/<code>compute_purified_mask</code> →
        11TB 底座在传入 <code>factor_engine</code> 前一秒完成<strong>量价·基本面·舆情</strong>三线净化，可进入全链路终审封板。
      </div>
    </section>
"""

EDGE_ROWS = """          <tr>
            <td><strong>交易所错单闪崩</strong></td>
            <td><code>correction</code>+条件码；5×MAD 分钟偏离</td>
            <td>价回滚、vol 扣脏笔；<code>is_erroneous_glitch=1</code></td>
            <td>日内 H/L/波动率跳过该 bar</td>
            <td>保留 -99% 脏脉冲</td>
            <td><code>PREPROC-030</code></td>
          </tr>
          <tr>
            <td><strong>财年变更过渡季</strong></td>
            <td>Duration &lt;75 或 &gt;105 天</td>
            <td><code>is_nonstandard_duration=1</code>；TTM 时间权重缩放</td>
            <td>TTM/YoY 无虚假 20% 断崖</td>
            <td>机械加总 4 行 quarterly</td>
            <td><code>PREPROC-031</code></td>
          </tr>
          <tr>
            <td><strong>退市转 OTC</strong></td>
            <td><code>fact_delisting_events</code>；Last Day</td>
            <td><code>delisting_return</code>；<code>tick_size_mask</code> 0.0001</td>
            <td>禁 universe 蒸发躲过暴跌</td>
            <td>摘牌日直接 drop 标的</td>
            <td><code>PREPROC-032</code></td>
          </tr>
          <tr>
            <td><strong>Form T / Late Prints</strong></td>
            <td>条件码 Form T、Out-of-Sequence</td>
            <td>vol 可累加；价不进分钟 H/L</td>
            <td>16:45 无假影线</td>
            <td>按 sip 时间戳盲目 resample</td>
            <td><code>PREPROC-033</code></td>
          </tr>
          <tr>
            <td><strong>新闻异步重发</strong></td>
            <td>SimHash/Jaccard&gt;0.85 @24h</td>
            <td><code>news_dup_suppressed=1</code>；<code>news_count_*</code> 不累加</td>
            <td>舆情强度不被放大</td>
            <td>仅信供应商 id</td>
            <td><code>PREPROC-034</code></td>
          </tr>
"""

TASK_INSERT = """          <tr><td>PREPROC-030</td><td><span class="badge badge-p1">P1</span></td><td>裁决作废错单 + is_erroneous_glitch + 分钟 MAD 回滚</td><td class="owner-cell">数据工程组</td><td class="status-cell">待办</td><td></td></tr>
          <tr><td>PREPROC-031</td><td><span class="badge badge-p1">P1</span></td><td>财年过渡季 is_nonstandard_duration + TTM 时间权重缩放</td><td class="owner-cell">数据工程组</td><td class="status-cell">待办</td><td></td></tr>
          <tr><td>PREPROC-032</td><td><span class="badge badge-p1">P1</span></td><td>OTC 摘牌 tick_size_mask + delisting_return 扩展</td><td class="owner-cell">算法策略组</td><td class="status-cell">待办</td><td></td></tr>
          <tr><td>PREPROC-033</td><td><span class="badge badge-p1">P1</span></td><td>Form T/Late Prints：价不参与分钟 H/L</td><td class="owner-cell">数据工程组</td><td class="status-cell">待办</td><td></td></tr>
          <tr><td>PREPROC-034</td><td><span class="badge badge-p1">P1</span></td><td>news_dup_suppressed + 24h SimHash（升级 016）</td><td class="owner-cell">量化研究组</td><td class="status-cell">待办</td><td></td></tr>
"""

BLIND_ROWS = """          <tr><td>PREPROC-030</td><td>Layer 1.5/2</td><td>错单闪崩 MAD 回滚 + is_erroneous_glitch</td><td>数据工程组</td><td>BRK.A 类事件不污染日内极值</td></tr>
          <tr><td>PREPROC-031</td><td>Layer 2 PiT</td><td>非标过渡季 TTM 缩放</td><td>数据工程组</td><td>财年变更无虚假 TTM 断崖</td></tr>
          <tr><td>PREPROC-032</td><td>Layer 2</td><td>OTC tick_size + 退市惩罚</td><td>算法策略组</td><td>无摘牌幸存者偏差</td></tr>
          <tr><td>PREPROC-033</td><td>Layer 2 分钟</td><td>Late Print 条件码 resampling</td><td>数据工程组</td><td>延迟成交不造假影线</td></tr>
          <tr><td>PREPROC-034</td><td>Layer 2 NLP</td><td>news_dup_suppressed</td><td>量化研究组</td><td>重发不计入 news_count</td></tr>
"""

COL_ROWS = """          <tr><td><code>is_erroneous_glitch</code></td><td>UInt8</td><td>交易所裁决错单分钟</td><td>PREPROC-030</td></tr>
          <tr><td><code>is_nonstandard_duration</code></td><td>UInt8</td><td>财年变更过渡季</td><td>PREPROC-031</td></tr>
          <tr><td><code>news_dup_suppressed</code></td><td>UInt8</td><td>新闻重发抑制</td><td>PREPROC-034</td></tr>
          <tr><td><code>compute_purified_mask</code></td><td>UInt8</td><td>净化总控（≡ compute_mask 扩展）</td><td>PREPROC-021</td></tr>
"""


def main():
    html = HTML.read_text(encoding="utf-8")

    if '<section id="sec-acl-deep2">' not in html:
        html = html.replace(
            '    </section>\n\n<section id="sec-scenarios">',
            '    </section>\n\n' + DEEP2_SECTION.strip() + '\n\n    <section id="sec-scenarios">',
            1,
        )

    if 'href="#sec-acl-deep2"' not in html.split('id="sec-acl-deep"')[1][:600] if 'id="sec-acl-deep"' in html else "":
        html = html.replace(
            "11TB 本地库在传入 <code>factor_engine</code> 前一秒具备「数理防腐 + 财务防畸变 + 公司行动守恒」完整契约。",
            "11TB 本地库在传入 <code>factor_engine</code> 前一秒具备「数理防腐 + 财务防畸变 + 公司行动守恒」完整契约。"
            " 交易所/财年/OTC/延迟成交/新闻重发见 <a href=\"#sec-acl-deep2\">§ACL++</a>。",
            1,
        )

    if "微观结构八项见下表" in html and "§ACL++" not in html[:250000]:
        html = html.replace(
            "微观结构八项见下表；<strong>财务/分拆/EDGAR 深水五极端</strong>见 <a href=\"#sec-acl-deep\">§ACL+</a>。",
            "微观结构八项见下表；<a href=\"#sec-acl-deep\">§ACL+</a> 财务/分拆/EDGAR；"
            "<a href=\"#sec-acl-deep2\">§ACL++</a> 交易所/财年/OTC/延迟成交/新闻。",
            1,
        )

    if "PREPROC-030</code></td>" not in html.split("sec-edge-cases")[1][:15000] if "sec-edge-cases" in html else "":
        html = html.replace(
            '<tr>\n            <td><strong>负权益 / 零营收</strong></td>',
            EDGE_ROWS + '<tr>\n            <td><strong>负权益 / 零营收</strong></td>',
            1,
        )

    if "PREPROC-030" not in html or '<tr><td>PREPROC-030</td><td><span class="badge' not in html:
        marker = (
            '<tr><td>PREPROC-029</td><td><span class="badge badge-p1">P1</span></td>'
            "<td>stock_dividend 路由至 splits 管道</td>"
            '<td class="owner-cell">基础架构组</td><td class="status-cell">待办</td><td></td></tr>\n'
            "          <tr><td>PREPROC-011</td>"
        )
        if marker in html:
            html = html.replace(marker, marker.replace("          <tr><td>PREPROC-011", TASK_INSERT + "          <tr><td>PREPROC-011"), 1)

    if "PREPROC-030</td><td>Layer 1.5" not in html and "PREPROC-029</td><td>Layer 1.5" in html:
        html = html.replace(
            '<tr><td>PREPROC-029</td><td>Layer 1.5</td><td>stock_dividend → splits 路由</td><td>基础架构组</td><td>market_cap 无虚假蒸发</td></tr>\n        </tbody>',
            '<tr><td>PREPROC-029</td><td>Layer 1.5</td><td>stock_dividend → splits 路由</td><td>基础架构组</td><td>market_cap 无虚假蒸发</td></tr>\n' + BLIND_ROWS + '        </tbody>',
            1,
        )

    if '<tr><td><code>is_erroneous_glitch</code></td><td>UInt8</td><td>交易所裁决错单分钟</td><td>PREPROC-030</td></tr>' not in html:
        if '<tr><td><code>is_zero_revenue</code></td><td>UInt8</td><td>零营收标的</td><td>PREPROC-026</td></tr>' in html:
            html = html.replace(
                '<tr><td><code>is_zero_revenue</code></td><td>UInt8</td><td>零营收标的</td><td>PREPROC-026</td></tr>\n          <tr><td><code>knowledge_ts_utc</code>',
                '<tr><td><code>is_zero_revenue</code></td><td>UInt8</td><td>零营收标的</td><td>PREPROC-026</td></tr>\n' + COL_ROWS.strip() + '\n          <tr><td><code>knowledge_ts_utc</code>',
                1,
            )
        elif '<tr><td><code>compute_mask</code></td><td>UInt8</td><td>Mask-First 计算白名单</td><td>PREPROC-021</td></tr>' in html:
            html = html.replace(
                '<tr><td><code>compute_mask</code></td><td>UInt8</td><td>Mask-First 计算白名单</td><td>PREPROC-021</td></tr>',
                '<tr><td><code>compute_mask</code></td><td>UInt8</td><td>Mask-First 计算白名单</td><td>PREPROC-021</td></tr>\n' + COL_ROWS.strip(),
                1,
            )

    # extend sec-acl Mask-First pre block
    if "is_erroneous_glitch" not in html.split("Mask-First 契约")[1][:2500] if "Mask-First 契约" in html else "":
        html = html.replace(
            "  AND price_clamped_flag = 0\n  -- 可选: days_since_listing",
            "  AND price_clamped_flag = 0\n  AND NOT is_erroneous_glitch\n  -- compute_purified_mask 与 compute_mask 同义主控（§ACL++）\n  -- 可选: days_since_listing",
            1,
        )

    if "§ACL++ 深水五极端" not in html:
        html = html.replace(
            '<li><a href="#sec-acl-deep">§ACL+ 深水五极端</a> 已评审（<code>PREPROC-026~029</code>）</li>',
            '<li><a href="#sec-acl-deep">§ACL+ 深水五极端</a> 已评审（<code>PREPROC-026~029</code>）</li>\n        <li><a href="#sec-acl-deep2">§ACL++</a> 已评审（<code>PREPROC-030~034</code>、<code>compute_purified_mask</code>）</li>',
            1,
        )

    if 'href="#sec-acl-deep2"' not in html:
        html = html.replace(
            '<a href="#sec-acl-deep"><strong>§ACL+</strong></a>',
            '<a href="#sec-acl-deep"><strong>§ACL+</strong></a> <a href="#sec-acl-deep2"><strong>§ACL++</strong></a>',
            1,
        )

    if "交易所/财年/OTC" not in html and "防腐层八大极端" in html:
        html = html.replace(
            '<tr><td>财务/分拆/EDGAR 深水五极端</td><td><a href="#sec-acl-deep">§ACL+</a></td></tr>',
            '<tr><td>财务/分拆/EDGAR 深水五极端</td><td><a href="#sec-acl-deep">§ACL+</a></td></tr>\n          <tr><td>交易所/财年/OTC/延迟成交/新闻</td><td><a href="#sec-acl-deep2">§ACL++</a></td></tr>',
            1,
        )

    # ACL summary row
    if "026~029" in html and "030~034" not in html.split("sec-acl")[1][:3500] if "sec-acl" in html else True:
        html = html.replace(
            '<tr><td>+</td><td>深水五极端（财务/分拆/EDGAR）</td><td><code>is_negative_equity</code> 等</td><td><a href="#sec-acl-deep">§ACL+</a> <code>026~029</code></td></tr>',
            '<tr><td>+</td><td>深水五极端（财务/分拆/EDGAR）</td><td><code>is_negative_equity</code> 等</td><td><a href="#sec-acl-deep">§ACL+</a> <code>026~029</code></td></tr>\n'
            '          <tr><td>+</td><td>深水五极端 B（交易所/财年/OTC/新闻）</td><td><code>is_erroneous_glitch</code> 等</td><td><a href="#sec-acl-deep2">§ACL++</a> <code>030~034</code></td></tr>',
            1,
        )

    html = re.sub(
        r"v3\.\d[^<]*",
        "v3.9 ACL++交易所/财年/OTC封板",
        html,
        count=1,
    )

    HTML.write_text(html, encoding="utf-8")
    print("ACL++ deep five-B injected, lines:", html.count("\n") + 1)

    bp = BEAUTIFY.read_text(encoding="utf-8")
    if '"sec-acl-deep", "sec-scenarios"' in bp and '"sec-acl-deep2"' not in bp:
        bp = bp.replace(
            '"sec-acl-deep", "sec-scenarios"',
            '"sec-acl-deep", "sec-acl-deep2", "sec-scenarios"',
        )
        BEAUTIFY.write_text(bp, encoding="utf-8")
    subprocess.run(["python3", str(BEAUTIFY)], check=True)


if __name__ == "__main__":
    main()
