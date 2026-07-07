#!/usr/bin/env python3
"""Inject ACL deep-five (financial/corp-action/EDGAR) guide into HTML deliverable."""
from pathlib import Path
import re
import subprocess

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"
BEAUTIFY = Path(__file__).resolve().parent / "beautify_massive_html.py"

DEEP_SECTION = r"""
    <section id="sec-acl-deep">
      <h2><span class="sec-badge">ACL+</span> 防腐层深水区五极端 — 入模前一秒（财务·分拆·EDGAR）</h2>
      <p class="section-desc">
        在 <a href="#sec-acl">§ACL 八大极端</a>（IPO、停牌、MWCB、半日市等<strong>微观结构</strong>）之外，
        还存在由<strong>极端财务畸变、复杂公司行动、SEC 系统故障</strong>引发的隐蔽陷阱。
        必须在 Layer 1.5 / Layer 2 物化或导入 ClickHouse 前卡死；与 §3.3 决策表、<a href="#sec-blind-spots-4">§12d</a> 互补。
      </p>

      <table class="table-schema">
        <thead><tr><th>#</th><th>深水区情形</th><th>核心掩码/契约</th><th>任务</th></tr></thead>
        <tbody>
          <tr><td>D1</td><td>负权益 / 零营收报表畸变</td><td><code>is_negative_equity</code>, <code>is_zero_revenue</code></td><td><code>PREPROC-026</code></td></tr>
          <tr><td>D2</td><td>Spinoff 分拆日价格断崖（非 split/div）</td><td>分拆比率 → 特殊价格平滑</td><td><code>PREPROC-027</code></td></tr>
          <tr><td>D3</td><td>SEC Batch-Filing 同日多期堆叠</td><td>排序 <code>filing_date</code> + <code>period_end</code></td><td><code>PREPROC-028</code></td></tr>
          <tr><td>D4</td><td>Stock dividend vs cash dividend</td><td>路由至 <code>splits</code> 管道</td><td><code>PREPROC-029</code></td></tr>
          <tr><td>D5</td><td>仙股合股后近零/负价（与 ACL-5 同）</td><td><code>price_clamped_flag</code></td><td><code>PREPROC-019</code></td></tr>
        </tbody>
      </table>

      <h3>ACL-D1 · 财务极端畸变：负资产、负权益与零营收</h3>
      <div class="box-danger">
        <strong>盲区：</strong> distressed 标的（如 Hertz、AMC、研发期 biotech）可能出现 <code>total_equity &lt; 0</code>、
        多季 <code>revenue = 0</code>。直接算 B/P、S/P、ROE → 符号倒转，截面 Rank 把垃圾股排成「超级成长/深度价值」。
      </div>
      <ul>
        <li><strong>禁止</strong>暴力填 0 或删行；物化 <code>is_negative_equity=1</code>、<code>is_zero_revenue=1</code></li>
        <li>估值/质量因子横截面：对上述掩码执行 <strong>Exclusion</strong> 或符号感知 winsorize（元数据契约）</li>
        <li><code>compute_mask</code> 日频可仍=1（保留观测）；<strong>截面排序层</strong>必须读掩码（与 Mask-First 正交）</li>
      </ul>
      <p><strong>任务：</strong> <code>PREPROC-026</code>（P1）— Layer 2 基本面宽表 + 因子引擎截面契约。</p>

      <h3>ACL-D2 · Spinoff 分拆上市：非传统除权的价格断层</h3>
      <div class="box-danger">
        <strong>盲区：</strong> GE→Vernova、PFE→Viatris 等 Ex-date 母公司开盘可跌 30%+；
        <code>corporate_actions/splits</code> 无因子、<code>dividends</code> 也捕不到 → 假「崩盘动量」/ 假做空信号。
      </div>
      <ul>
        <li>Layer 1.5：解析 <code>spinoffs</code> / 公告文本，审计分拆比率与剥离资产价值</li>
        <li>R 轨量价：视同<strong>巨额特殊非现金分红</strong>，对 Ex-date 之前母公司历史价 × 修正乘数前向平滑</li>
        <li>与 ACL-1「Spin-off 首日 Opening Cross」区分：D2=母公司历史连续；ACL-1=新股/分拆标的首日分钟真空</li>
      </ul>
      <p><strong>任务：</strong> <code>PREPROC-027</code>（P1）— <code>corporate_actions</code> 清洗流 + <code>PREPROC-003</code> 联动。</p>

      <h3>ACL-D3 · SEC EDGAR Batch-Filing：同日多期财报堆叠</h3>
      <div class="box-danger">
        <strong>盲区：</strong> EDGAR 故障恢复后，同日批量申报 2022 10-K + 2023 10-K + 多个 10-Q，
        <code>filing_date</code>/<code>align_time</code> 相同；简单 <code>merge_asof</code> 或按时间去重 → TTM/YoY 随机覆盖、CH 排序不唯一。
      </div>
      <ul>
        <li>双时态 asof 前强制<strong>多级排序</strong>：同 ticker 同 <code>filing_date</code> 时，
          一级 <code>filing_date</code>，二级 tie-breaker <code>period_end</code>（会计期末，由远及近）</li>
        <li>Step Function 展开：按会计周期顺序依次覆盖，保证血缘单调递增</li>
        <li>与 <code>PREPROC-022</code>（分钟 acceptance）正交：D3=同日多期<strong>日频</strong>堆叠</li>
      </ul>
      <p><strong>任务：</strong> <code>PREPROC-028</code>（P1）— PiT 拼装管线 + ClickHouse 唯一性约束文档。</p>

      <h3>ACL-D4 · Stock dividend vs split：股本非对称</h3>
      <div class="box-danger">
        <strong>盲区：</strong> <code>distribution_type=stock_dividend</code>、<code>cash_amount=0</code> 躺在分红表；
        只调价不调股本 → <code>market_cap = price × shares</code> 虚假蒸发 10%。
      </div>
      <ul>
        <li>Layer 1.5：分红表语义漏斗 — <code>stock_dividend</code> 且涉及股本变更 → <strong>切离 dividends</strong>，路由 <code>splits</code></li>
        <li>译为 forward split（如 1:1.1）；价格 × 因子、股本 ÷ 因子（与 <code>PREPROC-017</code> volume 对偶一致）</li>
      </ul>
      <p><strong>任务：</strong> <code>PREPROC-029</code>（P1）— <code>corporate_actions/dividends</code> 路由规则。</p>

      <h3>ACL-D5 · 仙股多次合股：近零/负价（与 ACL-5 合并实施）</h3>
      <div class="box-danger">
        <strong>盲区：</strong> 连续 1:100 reverse split → 后复权历史价 → 1e-8 或负数 → 除零、Inf、Z-Score 爆炸。
      </div>
      <ul>
        <li>落盘前 1 秒：<code>P_adj ≤ 0</code> 或 &lt;1e-4 → floor <code>1e-5</code> + <code>price_clamped_flag=1</code></li>
        <li>实施见 <a href="#sec-acl">ACL-5</a>、<code>PREPROC-019</code>、§10c.15 折叠代码</li>
      </ul>

      <h3>防畸变掩码扩展（与 PREPROC-021 联动）</h3>
      <p>在 <code>panel_daily</code> / CH 物化契约中追加（与八大极端掩码<strong>同批落盘</strong>）：</p>
      <pre>ALTER TABLE qs_massive.panel_daily ADD COLUMN IF NOT EXISTS is_negative_equity UInt8 DEFAULT 0;
ALTER TABLE qs_massive.panel_daily ADD COLUMN IF NOT EXISTS is_zero_revenue     UInt8 DEFAULT 0;
-- compute_ready_mask 与 compute_mask 同语义（引擎统一读 compute_mask）

compute_mask = in_universe_base AND has_valid_bar
  AND NOT is_market_halt AND NOT is_ticker_halt AND NOT is_ipo_pre_open
  AND price_clamped_flag = 0;
-- 估值截面另读: WHERE is_negative_equity=0 AND is_zero_revenue=0</pre>
      <p>Mask-First：<code>compute_mask==0</code> 的 bar <strong>禁止 push</strong> 进 <code>ts_*</code> ring buffer；
        财务畸变标的使用<strong>截面剔除</strong>，不替代 halt/IPO 的时序跳过逻辑。</p>

      <div class="box-info">
        <strong>闭环：</strong>§ACL 八大 + 本节水五 + <code>PREPROC-021</code> →
        11TB 本地库在传入 <code>factor_engine</code> 前一秒具备「数理防腐 + 财务防畸变 + 公司行动守恒」完整契约。
      </div>
    </section>
"""

BLIND_ROWS = """          <tr><td>PREPROC-026</td><td>Layer 2 基本面</td><td>负权益/零营收掩码 + 截面 Exclusion</td><td>量化研究组</td><td>B/P 排序无符号倒转</td></tr>
          <tr><td>PREPROC-027</td><td>Layer 1.5/2</td><td>Spinoff 历史价格平滑</td><td>基础架构组</td><td>分拆 Ex-date 无假崩盘动量</td></tr>
          <tr><td>PREPROC-028</td><td>Layer 2 PiT</td><td>Batch-Filing 级联排序</td><td>基础架构组</td><td>同日多期 TTM/YoY 单调正确</td></tr>
          <tr><td>PREPROC-029</td><td>Layer 1.5</td><td>stock_dividend → splits 路由</td><td>基础架构组</td><td>market_cap 无虚假蒸发</td></tr>
"""

CHECKLIST = """
        <li><a href="#sec-acl-deep">§ACL+ 深水五极端</a> 已评审（<code>PREPROC-026~029</code>）</li>
"""

COL_ROWS = """          <tr><td><code>is_negative_equity</code></td><td>UInt8</td><td>账面权益≤0 或畸变</td><td>PREPROC-026</td></tr>
          <tr><td><code>is_zero_revenue</code></td><td>UInt8</td><td>连续零营收标的</td><td>PREPROC-026</td></tr>
"""


def main():
    html = HTML.read_text(encoding="utf-8")

    if '<section id="sec-acl-deep">' not in html:
        html = html.replace(
            '    </section>\n\n<section id="sec-scenarios">',
            '    </section>\n\n' + DEEP_SECTION.strip() + '\n\n<section id="sec-scenarios">',
            1,
        )
        if '<section id="sec-acl-deep">' not in html:
            html = html.replace(
                '    </section>\n\n    <section id="sec-scenarios">',
                '    </section>\n\n' + DEEP_SECTION.strip() + '\n\n    <section id="sec-scenarios">',
                1,
            )

    # sec-acl cross-link
    if "sec-acl-deep" not in html.split('id="sec-acl"')[1][:800] if 'id="sec-acl"' in html else "":
        html = html.replace(
            "快速决策表仍见 <a href=\"#sec-edge-cases\">§3.3</a>；代码见",
            "微观结构八项见下表；<strong>财务/分拆/EDGAR 深水五极端</strong>见 "
            "<a href=\"#sec-acl-deep\">§ACL+</a>。快速决策表仍见 <a href=\"#sec-edge-cases\">§3.3</a>；代码见",
            1,
        )

    # extend ACL mask table (only in sec-acl Mask-First table)
    if '<tr><td><code>is_zero_revenue</code></td><td>日</td>' not in html:
        html = html.replace(
            '<tr><td><code>compute_mask</code></td><td>日/分钟</td><td><strong>1=允许进入 ts_* 滑动 buffer</strong></td></tr>\n        </tbody>\n      </table>\n      <pre>-- 日频宽表',
            '<tr><td><code>is_negative_equity</code></td><td>日</td><td>负权益/畸变报表（截面估值剔除）</td></tr>\n'
            '          <tr><td><code>is_zero_revenue</code></td><td>日</td><td>零营收标的（截面估值剔除）</td></tr>\n'
            '          <tr><td><code>compute_mask</code></td><td>日/分钟</td><td><strong>1=允许进入 ts_* 滑动 buffer</strong>（<code>compute_ready_mask</code> 同义）</td></tr>\n'
            '        </tbody>\n      </table>\n      <pre>-- 日频宽表',
            1,
        )

    # §14 main task table (after PREPROC-020)
    if "PREPROC-026" not in html:
        marker = (
            '<tr><td>PREPROC-020</td><td><span class="badge badge-p1">P1</span></td>'
            "<td>IPO 首日 Opening Cross 前 is_ipo_pre_open</td>"
            '<td class="owner-cell">基础架构组</td><td class="status-cell">待办</td><td></td></tr>\n'
            "          <tr><td>PREPROC-011</td>"
        )
        if marker in html:
            insert = """          <tr><td>PREPROC-025</td><td><span class="badge badge-p1">P1</span></td><td>长窗 Burn-in：load_start 自动前推 K 个交易日/分钟</td><td class="owner-cell">平台/量化</td><td class="status-cell">待办</td><td></td></tr>
          <tr><td>PREPROC-026</td><td><span class="badge badge-p1">P1</span></td><td>负权益/零营收掩码 + 估值截面 Exclusion 契约</td><td class="owner-cell">量化研究组</td><td class="status-cell">待办</td><td></td></tr>
          <tr><td>PREPROC-027</td><td><span class="badge badge-p1">P1</span></td><td>Spinoff 分拆比率 → 母公司历史价格平滑</td><td class="owner-cell">基础架构组</td><td class="status-cell">待办</td><td></td></tr>
          <tr><td>PREPROC-028</td><td><span class="badge badge-p1">P1</span></td><td>Batch-Filing 同日多期级联排序（period_end tie-break）</td><td class="owner-cell">基础架构组</td><td class="status-cell">待办</td><td></td></tr>
          <tr><td>PREPROC-029</td><td><span class="badge badge-p1">P1</span></td><td>stock_dividend 路由至 splits 管道</td><td class="owner-cell">基础架构组</td><td class="status-cell">待办</td><td></td></tr>
"""
            html = html.replace(marker, marker.replace("          <tr><td>PREPROC-011", insert + "          <tr><td>PREPROC-011"), 1)

    # §3.3 edge-case rows
    edge_rows = """          <tr>
            <td><strong>负权益 / 零营收</strong></td>
            <td><code>total_equity&lt;0</code>；多季 <code>revenue=0</code></td>
            <td><code>is_negative_equity=1</code>、<code>is_zero_revenue=1</code>（禁止填 0）</td>
            <td>估值/质量因子截面 <strong>Exclusion</strong>；<code>ts_*</code> 仍可读 <code>compute_mask</code></td>
            <td>删行或填 0 掩盖畸变</td>
            <td><code>PREPROC-026</code></td>
          </tr>
          <tr>
            <td><strong>Spinoff 分拆 Ex-date</strong></td>
            <td><code>spinoffs</code> 公告；母股非 split/div 断崖</td>
            <td>分拆比率 → 特殊价格平滑（等同特殊 dividend）</td>
            <td>动量/跳空因子不因假崩盘污染</td>
            <td>裸 SIP 当真实崩盘</td>
            <td><code>PREPROC-027</code></td>
          </tr>
          <tr>
            <td><strong>Batch-Filing 同日多期</strong></td>
            <td>同 <code>filing_date</code> 多份 10-K/10-Q</td>
            <td>排序：<code>filing_date</code> + <code>period_end</code> tie-break</td>
            <td>TTM/YoY Step Function 单调覆盖</td>
            <td>简单去重/随机保留一期</td>
            <td><code>PREPROC-028</code></td>
          </tr>
          <tr>
            <td><strong>Stock dividend</strong></td>
            <td><code>distribution_type=stock_dividend</code></td>
            <td>路由 <code>splits</code>；价×因子、股本÷因子</td>
            <td><code>market_cap</code> 守恒</td>
            <td>仅调价不调股本</td>
            <td><code>PREPROC-029</code></td>
          </tr>
"""
    if "PREPROC-026</code></td>" not in html.split("sec-edge-cases")[1][:12000] if "sec-edge-cases" in html else "":
        html = html.replace(
            "        把分散在 <a href=\"#sec-quality\">§2b</a>",
            "        深水区（财务/分拆/EDGAR）见 <a href=\"#sec-acl-deep\">§ACL+</a>。把分散在 <a href=\"#sec-quality\">§2b</a>",
            1,
        )
        html = html.replace(
            '<tr>\n            <td><strong>待上市 / pending IPO</strong></td>',
            edge_rows + '<tr>\n            <td><strong>待上市 / pending IPO</strong></td>',
            1,
        )

    # §12d blind spot table
    if "PREPROC-028" not in html and "PREPROC-020</td><td>Layer 2 分钟</td>" in html:
        html = html.replace(
            '<tr><td>PREPROC-020</td><td>Layer 2 分钟</td><td>IPO 预开盘标签 + 冷启动</td><td>基础架构组</td><td>首日 09:30–open 无假 ffill</td></tr>\n        </tbody>',
            '<tr><td>PREPROC-020</td><td>Layer 2 分钟</td><td>IPO 预开盘标签 + 冷启动</td><td>基础架构组</td><td>首日 09:30–open 无假 ffill</td></tr>\n' + BLIND_ROWS + '        </tbody>',
            1,
        )

    # sec-acl footer cross-link
    if "§ACL+</a>" not in html.split("封板结论")[0][-500:] if "封板结论" in html else True:
        html = html.replace(
            "<strong>封板结论：</strong>上述八项 + Mask-First",
            "<strong>封板结论：</strong>上述八项 + <a href=\"#sec-acl-deep\">§ACL+ 深水五极端</a> + Mask-First",
            1,
        )

    # fix section indent if needed
    html = html.replace("\n<section id=\"sec-acl-deep\">", "\n    <section id=\"sec-acl-deep\">")

    # §10c.5 columns
    if html.count("is_negative_equity") < 3:
        if '<tr><td><code>price_clamped_flag</code></td><td>UInt8</td><td>复权价底座截断</td><td>PREPROC-019</td></tr>' in html:
            html = html.replace(
                '<tr><td><code>price_clamped_flag</code></td><td>UInt8</td><td>复权价底座截断</td><td>PREPROC-019</td></tr>',
                '<tr><td><code>price_clamped_flag</code></td><td>UInt8</td><td>复权价底座截断</td><td>PREPROC-019</td></tr>\n' + COL_ROWS.strip(),
                1,
            )

    # §17 checklist
    if "§ACL+ 深水五极端" not in html and "PREPROC-025</code> Burn-in" in html:
        html = html.replace(
            "<li><a href=\"#sec-acl\">§ACL 八大极端</a> 已评审；<code>PREPROC-025</code> Burn-in 已接入调度</li>",
            "<li><a href=\"#sec-acl\">§ACL 八大极端</a> 已评审；<code>PREPROC-025</code> Burn-in 已接入调度</li>" + CHECKLIST,
            1,
        )

    # readmap
    if 'href="#sec-acl-deep"' not in html:
        html = html.replace(
            '<a href="#sec-acl"><strong>§ACL 八大极端</strong></a>',
            '<a href="#sec-acl"><strong>§ACL</strong></a> <a href="#sec-acl-deep"><strong>§ACL+</strong></a>',
            1,
        )

    # cheatsheet row
    if "深水五极端" not in html and "防腐层八大极端" in html:
        html = html.replace(
            '<tr><td>防腐层八大极端（入模前一秒）</td><td><a href="#sec-acl">§ACL</a> · <a href="#sec-edge-cases">§3.3</a></td></tr>',
            '<tr><td>防腐层八大极端（入模前一秒）</td><td><a href="#sec-acl">§ACL</a> · <a href="#sec-edge-cases">§3.3</a></td></tr>\n          <tr><td>财务/分拆/EDGAR 深水五极端</td><td><a href="#sec-acl-deep">§ACL+</a></td></tr>',
            1,
        )

    html = re.sub(
        r"v3\.\d[^<]*单文件交付|v3\.\d ACL[^<]*",
        "v3.8 ACL+深水五极端",
        html,
        count=1,
    )

    HTML.write_text(html, encoding="utf-8")
    print("ACL+ deep five injected, lines:", html.count("\n") + 1)

    bp = BEAUTIFY.read_text(encoding="utf-8")
    if '"sec-acl", "sec-scenarios"' in bp and '"sec-acl-deep"' not in bp:
        bp = bp.replace(
            '"sec-acl", "sec-scenarios"',
            '"sec-acl", "sec-acl-deep", "sec-scenarios"',
        )
        BEAUTIFY.write_text(bp, encoding="utf-8")
    subprocess.run(["python3", str(BEAUTIFY)], check=True)


if __name__ == "__main__":
    main()
