#!/usr/bin/env python3
"""Inject ACL eight-pillars anti-corruption guide into HTML deliverable."""
from pathlib import Path
import re
import subprocess

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"
BEAUTIFY = Path(__file__).resolve().parent / "beautify_massive_html.py"

ACL_SECTION = r"""
    </section>

    <section id="sec-acl">
      <h2><span class="sec-badge">ACL</span> 防腐层（ACL）八大极端情形 — 入模前一秒定稿</h2>
      <p class="section-desc">
        Feature Store 的核心价值：在数据进入 <code>factor_engine</code> / QuantaAlpha 算子发酵前，
        用 <strong>Layer 1.5～2 防腐层（Anti-Corruption Layer）</strong> 把美股微观机制导致的断层、噪音与虚假信号在数据层消掉。
        忽略 IPO 真空、长窗冷启动、LULD/ MWCB、量价复权非对称等，大模型会拟合出<strong>无法成交的 Fake Alpha</strong>。
        快速决策表仍见 <a href="#sec-edge-cases">§3.3</a>；代码见 <a href="#sec-code-round3">§10c.15</a>、<a href="#sec-blind-spots-4">§12d</a>。
      </p>

      <table class="table-schema">
        <thead><tr><th>#</th><th>极端情形</th><th>核心掩码/列</th><th>任务</th></tr></thead>
        <tbody>
          <tr><td>1</td><td>IPO / Spin-off 延迟 Opening Cross</td><td><code>is_ipo_pre_open</code>, <code>list_date</code></td><td><code>PREPROC-020</code></td></tr>
          <tr><td>2</td><td>长窗算子预热期 Burn-in</td><td>调度 <code>load_start &lt; train_start</code></td><td><code>PREPROC-025</code></td></tr>
          <tr><td>3</td><td>个股 LULD / 监管停牌</td><td><code>is_ticker_halt</code>, <code>universe_mask</code></td><td><code>PREPROC-018</code></td></tr>
          <tr><td>4</td><td>大盘熔断 MWCB</td><td><code>is_market_halt</code></td><td><code>ENG-012</code></td></tr>
          <tr><td>5</td><td>仙股 / 复权价 ≤0</td><td><code>price_clamped_flag</code></td><td><code>PREPROC-019</code></td></tr>
          <tr><td>6</td><td>成交量仅随拆股缩放</td><td><code>volume_adj_policy=split_only</code></td><td><code>PREPROC-017</code></td></tr>
          <tr><td>7</td><td>SIFMA 提前收盘（半日市）</td><td><code>expected_rth_bars</code>, <code>is_early_close</code></td><td><code>ENG-011</code></td></tr>
          <tr><td>8</td><td>分钟级日内披露 PiT</td><td><code>knowledge_ts_utc</code></td><td><code>PREPROC-022</code></td></tr>
          <tr><td>+</td><td>Mask-First 算子契约</td><td><code>compute_mask</code></td><td><code>PREPROC-021</code></td></tr>
        </tbody>
      </table>

      <h3>ACL-1 · IPO 与 Spin-off 首日「延迟开盘时空黑洞」</h3>
      <div class="box-danger">
        <strong>盲区：</strong>09:30 无成交 Bar；机械 390 根骨架 ffill 无「昨收」可填；填发行价 → 假零波动横盘；留 NaN → 整日因子失效。
      </div>
      <ul>
        <li>用 <code>corporate_actions/ipos</code> + 首笔真实 bar 锁定 <code>listing_trade_date</code>（Spin-off 同理，禁母公司昨收 ffill）</li>
        <li>Opening Cross 前分钟：<code>is_ipo_pre_open=1</code>；价量<strong>禁止</strong> ffill 发行价或昨收</li>
        <li>算子：该掩码下滚动 buffer <strong>不 push</strong> 或从首笔撮合分钟冷启动（见 <code>PREPROC-021</code>）</li>
      </ul>

      <h3>ACL-2 · 长窗算子「数据预热期」（Burn-in）</h3>
      <div class="box-danger">
        <strong>盲区：</strong><code>train_period</code> 从 2016-01-01 起跑且只加载 2016 起行情 → <code>ts_mean(close,1000)</code> 前约 4 年全 NaN（冷启动死锁），样本效率暴跌。
      </div>
      <ul>
        <li><strong>物理解耦：</strong>「特征可见区间（Train/Test）」≠「行情加载起点（Warming）」</li>
        <li>设最长窗口 <code>K</code>（日频交易日数或分钟 bar 数）；请求 <code>train_start</code> 时自动向前取 <code>K</code> 个有效交易日加载（Lookback Padding）</li>
        <li>本库历史底座建议 ≥2010（或 2013）起，供 <code>K=1000+</code> 日频因子喂饱 Ring Buffer</li>
        <li>因子层仍可用 <code>min_listing_days</code> / 输出区间裁到 train_start 之后</li>
      </ul>
      <p><strong>任务：</strong> <code>PREPROC-025</code>（P1）— Feature Store 调度与 <code>factor_engine</code> 取数 API 契约。</p>

      <h3>ACL-3 · 个股 LULD / 监管停牌（Reindex + 价量非对称）</h3>
      <div class="box-danger">
        <strong>盲区：</strong>停牌分钟无 bar 若直接 drop → Track B 回测默认无法调仓，虚假躲过复牌暴跌（生存者偏差）。
      </div>
      <ul>
        <li>1min 物化：按当日 RTH 骨架 <code>reindex</code>（正常 390；半日 210，见 ACL-7）</li>
        <li>OHLC：<strong>ffill</strong> 末笔公允价；Volume/Transactions：<strong>填 0</strong>；<code>is_ticker_halt=1</code></li>
        <li>RTH 累计停牌 &gt;60min → 当日 <code>universe_mask=False</code></li>
        <li>禁止仅 drop 停牌行</li>
      </ul>

      <h3>ACL-4 · 大盘熔断 MWCB</h3>
      <div class="box-danger">
        <strong>盲区：</strong>2020-03 等 Level 1/2 全市场 15min 停滞；若当「低波动横盘」→ 日内波动率/Amihud 全错。
      </div>
      <ul>
        <li><code>fact_market_halt_events</code> 登记 UTC 起止</li>
        <li>分钟：价可 ffill、量=0、<code>is_market_halt=1</code></li>
        <li><code>ts_*</code> 滚动<strong>不计入</strong> halt 分钟步长（与 PREPROC-021 联动）</li>
        <li>Level 3 提早收市 → 联动 <code>expected_rth_bars</code>（ENG-011）</li>
      </ul>

      <h3>ACL-5 · 仙股与极端复权「负价 / 近零」</h3>
      <div class="box-danger">
        <strong>盲区：</strong>多次 reverse split + 分红 → <code>P_adj≤0</code> 或 &lt;1e-4 → 除零、Inf、CH/截面归一化崩溃。
      </div>
      <ul>
        <li>落盘前：<code>P_adj≤0</code> 或 &lt;1e-4 → floor（如 <code>1e-5</code>）+ <code>price_clamped_flag=1</code></li>
        <li>可选策略级剔除 persistent clamp 标的</li>
      </ul>

      <h3>ACL-6 · 成交量缩放非对称（拆股 vs 分红）</h3>
      <div class="box-danger">
        <strong>盲区：</strong>分红复权若也缩放 volume → 历史换手率/流动性特征虚假放大。
      </div>
      <ul>
        <li><code>splits</code>：价格 × 因子，<strong>volume ÷ 因子</strong>（名义成交额守恒）</li>
        <li><code>dividends</code>：只调价格类；<strong>volume / transactions 保持 raw</strong></li>
        <li>元数据强制：<code>volume_adj_policy = split_only</code>（<code>PREPROC-017</code>）</li>
      </ul>

      <h3>ACL-7 · SIFMA 提前收盘（半日市骨架坍缩）</h3>
      <div class="box-danger">
        <strong>盲区：</strong>机械 390 根 → 13:00 后缺 bar 被当「停牌」ffill+vol=0 → 伪日内结构噪音。
      </div>
      <ul>
        <li><code>dim_calendar</code>：<code>is_early_close</code>、<code>expected_rth_bars=210</code>、<code>market_close_time_et=13:00</code></li>
        <li>提前收盘日：骨架在 13:00 <strong>物理截断</strong>；13:00 后<strong>禁止</strong>任何 ffill/补 bar</li>
      </ul>

      <h3>ACL-8 · 分钟级日内披露（秒级 PiT）</h3>
      <div class="box-danger">
        <strong>盲区：</strong>仅用 <code>filing_date</code> 贴全天分钟 → 11:15 披露的 8-K 在 09:30–11:14 前视。
      </div>
      <ul>
        <li>事件流：<code>knowledge_ts_utc</code> 优先 <code>sec_edgar_index.acceptance_time</code>（秒级 UTC）；新闻用 <code>published_utc</code></li>
        <li>分钟 grid：asof 到 <strong>下一有效分钟 bar</strong>（如 11:15:23 → ≥11:16 bar）</li>
        <li>财报 raw 无 acceptance → 须 index join 或保守 <code>filing_date T+1 00:00 UTC</code>（见 §12c 盲点十四）</li>
        <li>日频 panel 仍用 <code>filing_date</code>；<strong>分钟因子</strong>必须用 ACL-8 表</li>
      </ul>

      <h3>Mask-First 契约（PREPROC-021）</h3>
      <p>在 <code>panel_daily</code> 与 <code>fact_bars_adjusted_minute</code> 固化掩码列；因子引擎 <strong>Mask-First Propagation</strong>：</p>
      <table class="table-schema">
        <thead><tr><th>列</th><th>粒度</th><th>含义</th></tr></thead>
        <tbody>
          <tr><td><code>is_market_halt</code></td><td>分钟</td><td>大盘 MWCB 挂起</td></tr>
          <tr><td><code>is_ticker_halt</code></td><td>分钟</td><td>个股 LULD/监管停牌</td></tr>
          <tr><td><code>is_ipo_pre_open</code></td><td>分钟</td><td>IPO/Spin-off Opening Cross 前</td></tr>
          <tr><td><code>price_clamped_flag</code></td><td>日/分钟</td><td>复权价 floor 截断</td></tr>
          <tr><td><code>compute_mask</code></td><td>日/分钟</td><td><strong>1=允许进入 ts_* 滑动 buffer</strong></td></tr>
        </tbody>
      </table>
      <pre>-- 日频宽表（qs_massive.panel_daily）与分钟表均须同语义
compute_mask = in_universe_base
  AND has_valid_bar
  AND NOT is_market_halt
  AND NOT is_ticker_halt
  AND NOT is_ipo_pre_open
  AND price_clamped_flag = 0
  -- 可选: days_since_listing >= min_listing_days</pre>
      <p>算子伪代码：<code>if compute_mask==0: do not push to ring buffer</code>。
        DDL 见 <a href="#sec-code-round3">§10c.15 PREPROC-021</a>；分钟列见 <code>CH-007</code> 折叠 DDL。</p>

      <div class="box-info">
        <strong>封板结论：</strong>上述八项 + Mask-First 在 Layer 2 物化完成、<code>§17</code> 闸门勾选后，
        宽表传入因子引擎方可称为「数理无前视、工程无死锁、业务可成交」的第一级底座。
      </div>
    </section>
"""

TASK_ROW = """          <tr><td>PREPROC-025</td><td><span class="badge badge-p1">P1</span></td><td>长窗 Burn-in：load_start 自动前推 K 个交易日/分钟</td><td class="owner-cell">平台/量化</td><td class="status-cell">待办</td><td></td></tr>
"""

CHECKLIST_ITEMS = """
        <li><a href="#sec-acl">§ACL 八大极端</a> 已评审；<code>PREPROC-025</code> Burn-in 已接入调度</li>
        <li><code>volume_adj_policy=split_only</code> 写入元数据（<code>PREPROC-017</code>）</li>
"""


def main():
    html = HTML.read_text(encoding="utf-8")

    # fix sec-edge-cases missing close (if scenarios follows without </section>)
    if '<section id="sec-edge-cases">' in html and '<section id="sec-acl">' not in html:
        html = html.replace(
            "\n    \n<section id=\"sec-scenarios\">",
            ACL_SECTION + "\n<section id=\"sec-scenarios\">",
            1,
        )
    elif "<section id=\"sec-acl\">" not in html:
        html = html.replace(
            '<section id="sec-scenarios">',
            ACL_SECTION.strip() + "\n\n    <section id=\"sec-scenarios\">",
            1,
        )

    if "PREPROC-025" not in html and "sec-tasks" in html:
        html = html.replace(
            "<tr><td>PREPROC-024</td>",
            TASK_ROW + "          <tr><td>PREPROC-024</td>",
            1,
        )

    if "§ACL 八大极端" not in html and "17.7 边界情形" in html:
        html = html.replace(
            "<li><code>PREPROC-022</code>：分钟 PiT 无日内 filing 广播</li>",
            "<li><code>PREPROC-022</code>：分钟 PiT 无日内 filing 广播</li>" + CHECKLIST_ITEMS,
            1,
        )

    # readmap
    if 'href="#sec-acl"' not in html.split("sec-readmap")[1][:2500] if "sec-readmap" in html else "":
        html = html.replace(
            '<a href="#sec-edge-cases"><strong>§3.3 边界情形</strong></a>',
            '<a href="#sec-edge-cases"><strong>§3.3</strong></a> <a href="#sec-acl"><strong>§ACL 八大极端</strong></a>',
            1,
        )

    # cheatsheet
    if "防腐层八大极端" not in html:
        html = html.replace(
            "<tr><td>IPO/停牌/长窗怎么处理</td>",
            '<tr><td>防腐层八大极端（入模前一秒）</td><td><a href="#sec-acl">§ACL</a> · <a href="#sec-edge-cases">§3.3</a></td></tr>\n          <tr><td>IPO/停牌/长窗怎么处理</td>',
            1,
        )

    html = html.replace(
        "v3.6 单文件交付",
        "v3.7 ACL八大极端+Mask-First",
        1,
    )

    HTML.write_text(html, encoding="utf-8")
    print("ACL injected, lines:", html.count("\n") + 1)

    # update beautify nav
    bp = BEAUTIFY.read_text(encoding="utf-8")
    if '"sec-edge-cases", "sec-scenarios"' in bp and '"sec-acl"' not in bp:
        bp = bp.replace(
            '"sec-edge-cases", "sec-scenarios"',
            '"sec-edge-cases", "sec-acl", "sec-acl-deep", "sec-scenarios"',
        )
        BEAUTIFY.write_text(bp, encoding="utf-8")
    subprocess.run(["python3", str(BEAUTIFY)], check=True)


if __name__ == "__main__":
    main()
