#!/usr/bin/env python3
"""Inject consolidated edge-case handbook into Massive HTML deliverable."""
from pathlib import Path
import re
import subprocess

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"
BEAUTIFY = Path(__file__).resolve().parent / "beautify_massive_html.py"

SECTION = """
    </section>

    <section id="sec-edge-cases">
      <h2><span class="sec-badge">3.3</span> 边界情形总手册（IPO · 停牌 · 长窗算子 · 退市等）</h2>
      <p class="section-desc">
        把分散在 <a href="#sec-quality">§2b</a>、<a href="#sec-blind-spots-4">§12d</a>、<a href="#sec-operator-adj">§3.2</a> 的规则收成<strong>一张决策表</strong>。
        Layer 2 负责<strong>打标 + 掩码 + 元数据</strong>；因子引擎负责<strong>尊重掩码 / 冷启动</strong>；禁止用 ffill 掩盖生命周期断点。
      </p>

      <table class="table-schema">
        <thead>
          <tr>
            <th>情形</th><th>如何识别</th><th>Layer 2 怎么处理</th><th>因子 / <code>ts_*</code> 怎么处理</th><th>禁止</th><th>任务</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>新股 / IPO 首日</strong></td>
            <td><code>ipos.listing_date</code>；首日首笔 minute/day bar 时间晚于 09:30</td>
            <td><code>list_date</code> 写入 <code>security_master</code>；分钟 <code>is_ipo_pre_open=1</code> 至首笔成交</td>
            <td>滚动/动量<strong>冷启动</strong>：窗口从首笔有效 bar 起；上市不足 N 日 → <code>universe_mask</code> 或因子 NaN</td>
            <td>用「昨收」ffill Opening Cross 前分钟</td>
            <td><code>PREPROC-020</code></td>
          </tr>
          <tr>
            <td><strong>待上市 / pending IPO</strong></td>
            <td><code>ipo_status=pending</code>，无 bar</td>
            <td>不进 <code>universe_daily</code>；无 bar 不造行</td>
            <td>不参与截面排名</td>
            <td>用 announced_date 当已上市</td>
            <td>策略过滤</td>
          </tr>
          <tr>
            <td><strong>退市 / 最后交易日</strong></td>
            <td>bar 序列终止；<code>fact_delisting_events</code></td>
            <td><code>is_delisting_day=1</code>；<code>delisting_return</code> + <code>delisting_policy</code> 元数据</td>
            <td>之后日期无行；收益链在末日闭合；禁 backfill 复活</td>
            <td>退市后 ffill 价格</td>
            <td><code>PREPROC-012</code></td>
          </tr>
          <tr>
            <td><strong>个股停牌</strong>（LULD / 监管）</td>
            <td>交易中断；tick 条件码 / 监管公告</td>
            <td>reindex 补 bar：OHLC ffill 末价，<code>volume=0</code>，<code>is_ticker_halt=1</code></td>
            <td>滚动窗口<strong>跳过</strong> halt bar 或 mask；RTH 累计停牌 &gt;60min → 当日 <code>universe_mask=False</code></td>
            <td>直接 drop 停牌日（生存者偏见）</td>
            <td><code>PREPROC-018</code></td>
          </tr>
          <tr>
            <td><strong>大盘熔断 MWCB</strong></td>
            <td>2020-03 等；全市场 15min 暂停</td>
            <td><code>fact_market_halt_events</code>；分钟 <code>is_market_halt=1</code></td>
            <td>日内波动/Amihud 滚动<strong>跳过</strong> halt 分钟</td>
            <td>把 halt 当成零波动横盘因子</td>
            <td><code>ENG-012</code></td>
          </tr>
          <tr>
            <td><strong>提前收盘 / 半日市</strong></td>
            <td>SIFMA；<code>dim_calendar.is_early_close</code></td>
            <td><code>expected_rth_bars</code>≠390（如 210）；<code>market_close_time_et=13:00</code></td>
            <td>分钟长窗按<strong>实际 bar 数</strong>，勿假定 390</td>
            <td>半日仍按全日分钟数归一</td>
            <td><code>ENG-011</code></td>
          </tr>
          <tr>
            <td><strong>当日无成交（无行）</strong></td>
            <td>sparse panel 缺 <code>(D,ticker)</code></td>
            <td>不生成行；left join → NaN</td>
            <td>区分「无数据」与「有 bar」；流动性因子 NaN</td>
            <td>填 0 当收盘价</td>
            <td>—</td>
          </tr>
          <tr>
            <td><strong>有 bar 但 volume=0</strong></td>
            <td>~0.8% 日样本</td>
            <td>保留行；<code>volume=0</code></td>
            <td>close 可算收益；turnover/Amihud 单独处理或 mask</td>
            <td>删行当退市</td>
            <td>策略定</td>
          </tr>
          <tr>
            <td><strong>长窗 <code>ts_*</code>（日频）</strong></td>
            <td><code>d=252/1000</code> 等</td>
            <td>读 <code>fact_bars_adjusted_daily</code>；拆股后 <strong>warm-up 重置</strong>（见下表）</td>
            <td>上市不足 <code>d</code> 交易日 → NaN 或 <code>min_listing_days</code> mask</td>
            <td>未复权 SIP 跑多年动量</td>
            <td><code>PREPROC-003</code></td>
          </tr>
          <tr>
            <td><strong>长窗 <code>ts_*</code>（分钟）</strong></td>
            <td><code>d=60~1000</code> bar</td>
            <td>读 <code>fact_bars_adjusted_minute</code>；尊重 <code>is_*_halt</code> / <code>is_ipo_pre_open</code></td>
            <td><code>d</code>=连续分钟数非日历日；跨 halt/IPO/拆股须复权+跳窗；半日市 bar 数变少</td>
            <td>cleaned 未复权 minute 直接进算子</td>
            <td><code>PREPROC-003</code>、<code>CH-007</code></td>
          </tr>
          <tr>
            <td><strong>拆股 / 合股</strong></td>
            <td><code>splits</code>；价量断崖</td>
            <td>复权物化；记录 <code>adj_method</code>；拆股日滚动窗口重置</td>
            <td>公司行动后 <code>ts_mean/std</code> 从新序列重算</td>
            <td>混 REST+SIP 两套价</td>
            <td><code>PREPROC-003</code></td>
          </tr>
          <tr>
            <td><strong>仙股 / 复权价 ≤0</strong></td>
            <td>多次 reverse split</td>
            <td>floor clamp + <code>price_clamped_flag=1</code></td>
            <td>含 flag 的 bar 可 mask；禁 <code>close/open</code> 除零</td>
            <td>负价进 CH</td>
            <td><code>PREPROC-019</code></td>
          </tr>
          <tr>
            <td><strong>代码变更 / 更名</strong></td>
            <td>FB→META；<code>dim_ticker_map</code></td>
            <td><code>permanent_id</code> 串联；Track B 用 <code>(permanent_id, time)</code></td>
            <td>勿把新旧 ticker 当两只独立股票拼长期序列</td>
            <td>仅按字符串 ticker 拼 10 年序列</td>
            <td><code>PREPROC-011</code></td>
          </tr>
          <tr>
            <td><strong>非交易日 / Ghost 行</strong></td>
            <td>周末、哀悼日、04:00 前测试成交</td>
            <td>Hard Delete；<code>trade_date ∉ 𝒯</code></td>
            <td>—</td>
            <td>周末行进 panel</td>
            <td><code>ENG-013</code></td>
          </tr>
          <tr>
            <td><strong>基本面不足 4 季 LTM</strong></td>
            <td>新上市 / 刚披露</td>
            <td><code>ltm_*</code>=NaN；元数据声明 YTD 降级</td>
            <td>PE/支付率分母 NaN</td>
            <td>用单季冒充 LTM</td>
            <td><code>PREPROC-006</code></td>
          </tr>
          <tr>
            <td><strong>2024 截断基本面</strong></td>
            <td>恰好 10000 行</td>
            <td>分区隔离或 <code>pit_*</code> 强制 NULL</td>
            <td>train ≤2023-12-31</td>
            <td>直接挖 2024 因子</td>
            <td><code>TASK-DATA-*</code></td>
          </tr>
        </tbody>
      </table>

      <h4>长参数窗口（<code>ts_*</code>）专表</h4>
      <table class="table-schema">
        <thead>
          <tr><th>参数</th><th>日频含义</th><th>分钟含义（RTH）</th><th>推荐政策</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><code>d=20</code></td><td>约 1 个月交易日</td><td>约 1 小时级（20 根 1min）</td>
            <td>新股上市 &lt;20 日可 mask 或允许短窗冷启动</td>
          </tr>
          <tr>
            <td><code>d=252</code></td><td>约 1 年</td><td>约 1 个交易日</td>
            <td>分钟上 <code>d=252</code> 是<strong>日内</strong>特征，勿与日频一年动量混称</td>
          </tr>
          <tr>
            <td><code>d=1000</code></td><td>约 4 年</td><td>约 2.6 个交易日</td>
            <td>分钟 <code>d=1000</code> 仍须<strong>复权</strong>；跨 IPO/停牌/拆股按上表跳窗</td>
          </tr>
        </tbody>
      </table>
      <div class="box-warn">
        <strong>因子引擎约定（须在表达式或引擎配置写明）：</strong>
        <ul>
          <li>默认 <code>min_periods = d</code>（窗口不满 → NaN），<strong>不要</strong>用 0 或截面均值硬填；</li>
          <li>可选 <code>min_listing_days</code>（如 60/252）：上市不足则整段因子 NaN；</li>
          <li>对 <code>is_ticker_halt</code> / <code>is_market_halt</code> / <code>is_ipo_pre_open</code>：实现「掩码感知滚动」或先过滤再 <code>ts_*</code>；</li>
          <li>拆股执行日：重置该股滚动状态（warm-up），避免拆股前后混在一个窗口。</li>
        </ul>
      </div>

      <h4>IPO 落地步骤（简）</h4>
      <ol>
        <li>从 <code>corporate_actions/ipos</code> 取 <code>listing_date</code>（与首 bar 交叉验证）；</li>
        <li>写入 <code>dim_security_master.list_date</code>；</li>
        <li>listing 日前：不进 universe；listing 当日：分钟 pre-open 打标；</li>
        <li>基本面：listing 前无 PiT 季报 → <code>pit_*</code> 为 NaN（正常）；</li>
        <li>长周期因子：可用 <code>days_since_listing &lt; d</code> 做 mask（策略或 Layer 2 派生列）。</li>
      </ol>

      <h4>停牌落地步骤（简）</h4>
      <ol>
        <li>检测：分钟无成交区间 + 监管/LULD 规则（或外部 halt 表）；</li>
        <li>补 bar：价格 ffill 最后有效价，<code>volume=0</code>，<code>is_ticker_halt=1</code>；</li>
        <li>累计 RTH 停牌 &gt;60 分钟：当日 <code>universe_mask=False</code>；</li>
        <li><code>ts_*</code>：窗口内排除 halt bar，或整日剔除（视因子定义）。</li>
      </ol>

      <p>细节与验收样例：<a href="#sec-blind-spots-4">§12d 极端异常</a> ·
        代码 <a href="#sec-code-round3">§10c.15</a> ·
        列契约 <a href="#sec-engineering">§10c.5</a> ·
        任务 <a href="#sec-tasks">§14</a>（<code>PREPROC-018~020</code>、<code>ENG-011~013</code>）。</p>
    </section>
"""

OPERATOR_APPEND = """
      <h4>3.2.1 长窗算子与生命周期交叉（必读）</h4>
      <p>算子参数 <code>d</code> 统计的是<strong>连续 bar 个数</strong>。生命周期事件（IPO、停牌、拆股、退市）会切断「可解释的连续序列」——
        必须在 Layer 2 用标志位表达，并在因子层选择<strong>跳窗 / 冷启动 / mask</strong>，见 <a href="#sec-edge-cases">§3.3</a>。</p>
"""

CHECKLIST_APPEND = """
      <h3>17.7 边界情形（IPO / 停牌 / 长窗）</h3>
      <ul class="checklist">
        <li><a href="#sec-edge-cases">§3.3</a> 决策表已评审：团队对 IPO/停牌/长窗政策一致</li>
        <li>IPO：<code>is_ipo_pre_open</code>、<code>list_date</code>、禁止 Opening Cross 前假 ffill</li>
        <li>停牌：<code>is_ticker_halt</code>、长停牌日 <code>universe_mask</code>、禁止 drop 停牌日</li>
        <li>熔断：<code>is_market_halt</code> / <code>ENG-012</code> 验收（2020-03 样例）</li>
        <li>长窗：分钟/日均用 <code>fact_bars_adjusted_*</code>；<code>min_periods</code> / <code>min_listing_days</code> 已配置</li>
        <li>拆股：滚动 warm-up 或等价重置；<code>adj_method</code> 元数据一致</li>
        <li>退市：<code>delisting_policy</code> + <code>delisting_return</code> 已写入</li>
      </ul>
"""

READMAP_SNIPPET = '<a href="#sec-edge-cases"><strong>§3.3 边界情形</strong></a> '


def main():
    html = HTML.read_text(encoding="utf-8")

    if '<section id="sec-edge-cases">' not in html:
        # upgrade h3-only block or insert fresh
        if 'id="sec-edge-cases"' in html and '<section id="sec-edge-cases">' not in html:
            html = re.sub(
                r'\s*<h3 id="sec-edge-cases">.*?</p>\s*(?=<h3 id="sec-stage8-minute">)',
                "\n" + SECTION + "\n      ",
                html,
                count=1,
                flags=re.DOTALL,
            )
        else:
            anchor = r'(<h3 id="sec-operator-adj">.*?</div>\s*)'
            if re.search(anchor, html, re.DOTALL):
                html = re.sub(anchor, r"\1" + SECTION, html, count=1, flags=re.DOTALL)
            else:
                html = html.replace(
                    '<section id="sec-scenarios">',
                    SECTION + '\n    <section id="sec-scenarios">',
                    1,
                )

    if "3.2.1 长窗算子与生命周期" not in html:
        html = html.replace(
            '<h3 id="sec-stage8-minute">',
            OPERATOR_APPEND + '\n      <h3 id="sec-stage8-minute">',
            1,
        )

    if "17.7 边界情形" not in html:
        html = html.replace(
            "      <h3>17.6 明确不在 Layer 2</h3>",
            "      <h3>17.6 明确不在 Layer 2</h3>",
            1,
        )
        html = html.replace(
            "      </ul>\n    </section>\n\n    \n    <!-- sync_md_into_html: returns",
            "      </ul>\n" + CHECKLIST_APPEND + "\n    </section>\n\n    \n    <!-- sync_md_into_html: returns",
            1,
        )

    if READMAP_SNIPPET not in html:
        html = html.replace(
            '<a href="#sec-operator-adj"><strong>§3.2 分钟复权</strong></a>',
            '<a href="#sec-operator-adj"><strong>§3.2 分钟复权</strong></a> ' + READMAP_SNIPPET,
            1,
        )

    # cheatsheet row
    if "IPO/停牌/长窗" not in html:
        html = html.replace(
            "<tr><td>能否上因子引擎</td>",
            '<tr><td>IPO/停牌/长窗怎么处理</td><td><a href="#sec-edge-cases">§3.3 边界情形手册</a></td></tr>\n          <tr><td>能否上因子引擎</td>',
            1,
        )

    HTML.write_text(html, encoding="utf-8")
    print("edge cases injected, lines:", html.count("\n") + 1)
    subprocess.run(["python3", str(BEAUTIFY)], check=True)


if __name__ == "__main__":
    main()
