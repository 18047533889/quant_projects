#!/usr/bin/env python3
"""Merge second AI preproc report into §12f with engineering judgment (no duplication)."""
from pathlib import Path
import re
import subprocess

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"
BEAUTIFY = Path(__file__).resolve().parent / "beautify_massive_html.py"

WAVE2 = """
      <h3>十一、第二轮 AI 报告裁量（2026 · 异构入模前沿预处理）</h3>
      <p class="section-desc">
        用户提供的第二份长篇 AI 报告与 §0/§6/§12f 前半<strong>高度重叠</strong>。
        下列为经 <a href="#sec-acl-judgment">§裁量</a> 筛选后<strong>值得保留或补强</strong>的增量；其余仅索引已有章节，不再重复粘贴。
      </p>
      <table class="table-schema">
        <thead><tr><th>报告主张</th><th>裁量</th><th>本库落点</th></tr></thead>
        <tbody>
          <tr><td>Layer 0/1/2 单源清洗与多源对齐解耦</td><td>✅ 已确立</td><td><a href="#sec-arch">§0</a>、<code>massive_cleaning_framework.py</code> / <code>composite_source.py</code></td></tr>
          <tr><td>禁止盲目 MAD 删 tick；保护开收盘竞价</td><td>✅ P1</td><td><code>PREPROC-024</code>、<code>TASK-DC-005</code></td></tr>
          <tr><td>Brownlees-Gallo 动态方差剔 tick</td><td>⚠️ 可选 P2</td><td>仅<strong>逐笔微观因子</strong>路径；分钟/日 K <strong>不做</strong>；与集合竞价豁免联测</td></tr>
          <tr><td>同时间戳 VWAP 聚合</td><td>✅ P2</td><td><code>PREPROC-023</code></td></tr>
          <tr><td>分钟 volume 求和 ≠ 日 volume</td><td>✅ 铁律</td><td><a href="#sec-p1">§3</a></td></tr>
          <tr><td>REST vs SIP 复权隔离</td><td>✅ 铁律</td><td><code>PREPROC-003</code></td></tr>
          <tr><td>Ticker 重用 → permanent_id</td><td>✅ P1</td><td><code>PREPROC-011</code>、<code>dim_ticker_map</code></td></tr>
          <tr><td>Shumway 退市 -30%/-55%</td><td>⚠️ 参考非真值</td><td><code>PREPROC-012</code>：Massive 无 CRSP；可用作<strong>敏感性分析</strong>默认，非供应商字段</td></tr>
          <tr><td>filing_date PiT + asof backward</td><td>✅ P0</td><td><code>PREPROC-005</code>、§8</td></tr>
          <tr><td>财务重述双时间轴</td><td>✅ 已述</td><td><code>FEAT-002</code>；禁 cleaned <code>keep_latest</code> 回测</td></tr>
          <tr><td>HTML 剥离 / news explode</td><td>✅ P2</td><td><code>NLP-001</code>、Layer 1 cleaning</td></tr>
          <tr><td>KG 情绪溢出赋权</td><td>❌ 暂不采纳</td><td>缺图谱数据源；记为 P3 研究项，不入默认面板</td></tr>
          <tr><td>LLM 实体匿名化</td><td>⚠️ 分场景</td><td>短新闻可 <code>NLP-002</code>；财报电话会等长文本<strong>慎脱敏</strong>（信息损耗 &gt; 前视收益时停用）</td></tr>
          <tr><td>10k/2k 截断 + 假 .ok</td><td>✅ P0</td><td><a href="#sec-p0">§2</a></td></tr>
          <tr><td>ADR 母股价差中性化</td><td>❌ 默认不做</td><td>见 <a href="#sec-acl-mm">§MM-5</a>：默认 universe 剔除 ADR</td></tr>
        </tbody>
      </table>
"""

SHUMWAY_ROWS = """          <tr><td>破产 / 强制摘牌（学术参考 Shumway 1997）</td><td>NYSE/AMEX 缺失末月收益时可参考 <strong>-30%</strong>、NASDAQ <strong>-55%</strong> 作<strong>敏感性默认</strong>；须写入 <code>delisting_policy</code>，<strong>非</strong> Massive 真值</td></tr>
          <tr><td>破产 / 强制摘牌（本库默认示例）</td><td>最后交易日识别后写入<strong>惩罚收益</strong>（如 -90%～-100%）或 OTC 首日可观测价；禁止「无 bar = 无损失」</td></tr>"""

SHUMWAY_OLD = """          <tr><td>破产 / 强制摘牌</td><td>最后交易日识别后写入<strong>惩罚收益</strong>（如 -90%～-100%）或 OTC 首日可观测价；禁止「无 bar = 无损失」</td></tr>"""

NLP2_OLD = """        <li><strong>Layer 2 · NLP-002</strong>：FinBERT/LLM 打分前，对目标公司名/ticker/高管做<strong>匿名占位</strong>，缓解训练语料前视、Look-Ahead-Bench 类「注意力分散」与预训练记忆渗漏。</li>"""

NLP2_NEW = """        <li><strong>Layer 2 · NLP-002</strong>：短标题/通稿可对 ticker/公司名<strong>匿名占位</strong>（缓解 LLM 预训练记忆）；<strong>财报电话会、长 8-K</strong>等慎脱敏——激进替换会破坏指代与数字语境（信息损耗可能超过前视收益）。须 A/B 验收后再默认开启。</li>"""

BG_ROW = """          <tr>
            <td><strong>Brownlees-Gallo 类动态方差</strong></td>
            <td>学术 tick 清洗；<strong>非</strong> Massive 默认路径。若做微观因子，在 <code>PREPROC-023</code> 之后可选层实现，且<strong>不得</strong>删除开收盘竞价 bar</td>
            <td>Layer 2 tick（P2 可选）</td>
          </tr>"""

JUDGMENT_ROW = """          <tr><td>§12f 异构入模 AI 报告（两轮）</td><td><strong>索引+裁量</strong></td><td>P0/P1 见 §12f；P2/P3 见 §12f 十一；勿重复读 AI 原文</td></tr>
"""

INTRO_OLD = """        本节吸收外部深度研究报告中的<strong>可落地准则</strong>，并与本库 Massive 实证、既有 §12a–12e 任务对齐。
        已覆盖内容仅做索引；<strong>新增</strong>项登记为 <code>PREPROC-023/024</code>、<code>NLP-001~003</code>。"""

INTRO_NEW = """        本节是外部 AI「异构海量入模预处理」报告的<strong>唯一收纳处</strong>（两轮内容已去重）。
        与 <a href="#sec-acl-judgment">§裁量</a> 联读：AI 叙述≠本库真值；Massive 实证勘误见下方黄框。
        首轮新增任务：<code>PREPROC-023/024</code>、<code>NLP-001~003</code>；ACL/MM 见 <a href="#sec-acl">§ACL</a> 系列。"""


def main():
    html = HTML.read_text(encoding="utf-8")

    if "十一、第二轮 AI 报告裁量" not in html:
        html = html.replace(
            "      <div class=\"box-info\">\n        以上 5 项已同步至",
            WAVE2.strip() + "\n\n      <div class=\"box-info\">\n        以上 5 项已同步至",
            1,
        )

    if INTRO_OLD in html:
        html = html.replace(INTRO_OLD, INTRO_NEW, 1)

    if SHUMWAY_OLD in html and "Shumway 1997" not in html:
        html = html.replace(SHUMWAY_OLD, SHUMWAY_ROWS, 1)

    if NLP2_OLD in html:
        html = html.replace(NLP2_OLD, NLP2_NEW, 1)

    if "Brownlees-Gallo" not in html and "<h3>一、高频量价" in html:
        html = html.replace(
            "          <tr>\n            <td><strong>跨粒度</strong></td>",
            BG_ROW + "\n          <tr>\n            <td><strong>跨粒度</strong></td>",
            1,
        )

    # fix duplicate Newspaper row
    html = html.replace(
        "                    <tr><td>Newspaper3k/4k</td><td>新闻抓取、元数据</td><td>通稿类站点；复杂 DOM 需抽检</td></tr>\n"
        "                    <tr><td>Newspaper3k/4k</td><td>新闻抓取、元数据</td><td>通稿类站点；复杂 DOM 需抽检</td></tr>",
        "          <tr><td>Newspaper3k/4k</td><td>新闻抓取、元数据</td><td>通稿类站点；复杂 DOM 需抽检</td></tr>",
        1,
    )

    if "§12f 异构入模 AI 报告" not in html and 'id="sec-acl-judgment"' in html:
        html = html.replace(
            "          <tr><td><code>compute_purified_mask</code></td><td><strong>合并命名</strong></td>",
            JUDGMENT_ROW + "          <tr><td><code>compute_purified_mask</code></td><td><strong>合并命名</strong></td>",
            1,
        )

    # sec-readmap link to 12f for architecture narrative
    if "sec-deep-research" not in html.split("sec-readmap")[1][:2500]:
        html = html.replace(
            "<tr><td>数据工程</td><td>",
            "<tr><td>架构/Pipeline</td><td><a href=\"#sec-arch\">§0</a> <a href=\"#sec-deep-research\">§12f</a></td><td>三层隔离+AI报告裁量索引</td></tr>\n          <tr><td>数据工程</td><td>",
            1,
        )

    html = re.sub(
        r"v3\.\d[^<]*",
        "v3.14 §12f 第二轮 AI 报告裁量合并",
        html,
        count=1,
    )

    HTML.write_text(html, encoding="utf-8")
    print("wave2 preproc merged, lines:", html.count("\n") + 1)
    subprocess.run(["python3", str(BEAUTIFY)], check=True)


if __name__ == "__main__":
    main()
