#!/usr/bin/env python3
"""Apply engineering judgment: trim AI hype, fix ADR/news priorities, add disclaimer."""
from pathlib import Path
import re
import subprocess

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"
BEAUTIFY = Path(__file__).resolve().parent / "beautify_massive_html.py"

JUDGMENT_SECTION = """
    <section id="sec-acl-judgment">
      <h2><span class="sec-badge">裁量</span> ACL / 多模态规约 · 工程裁量说明（必读）</h2>
      <p class="section-desc">
        手册中 §ACL / §MM / §ACL++ 大量条目来自<strong>外部 AI 草稿经人工整理</strong>，<strong>非经本库实证验收的绝对真理</strong>。
        实施前必须对照 <a href="#sec-data-catalog">§数据手册</a>、<a href="#sec-p0">§P0 红线</a> 与现有代码（<code>PREPROC-001</code> 等）做裁剪。
      </p>
      <table class="table-schema">
        <thead><tr><th>条目</th><th>裁量结论</th><th>优先级</th><th>说明</th></tr></thead>
        <tbody>
          <tr><td>M1 负权益/零营收</td><td><strong>采纳</strong></td><td>P1</td><td>经典量化坑；掩码+截面剔除，禁止填 0</td></tr>
          <tr><td>M2 Spinoff</td><td><strong>有条件采纳</strong></td><td>P1</td><td><code>is_spinoff_ex_day</code> 易落地；历史价平滑依赖 <code>corporate_actions</code> 是否有完整 spinoff 字段，缺则仅打标+动量跳日</td></tr>
          <tr><td>M3 Batch-Filing</td><td><strong>采纳</strong></td><td>P1</td><td><code>filing_date</code>+<code>period_end</code> 排序是 PiT 标准做法；75/105 天阈值为<strong>可调启发式</strong>，非 SEC 法定</td></tr>
          <tr><td>M4 Stock dividend</td><td><strong>采纳</strong></td><td>P1</td><td>与 <code>PREPROC-017</code> 一致；先审计 <code>dividends</code> 表 <code>distribution_type</code> 实际取值</td></tr>
          <tr><td>M5 ADR + 母股价差</td><td><strong>改口径</strong></td><td>P2</td><td>本库 <strong>无境外母所行情</strong>；默认策略与姊妹方案一致：<strong>PREPROC-001 识别 ADR 并从默认 universe 剔除</strong>。仅当团队<strong>单独采购母所数据</strong>且做 ADR 专题时，才实施 <code>parent_close_fx_adj</code></td></tr>
          <tr><td>M6 跨语种语义去重</td><td><strong>延后</strong></td><td>P2</td><td>新闻源 P0 未全量；先做 <code>PREPROC-034</code> 同语种 SimHash。Entity+Subject 需 NLP 管线，非当前阻塞项</td></tr>
          <tr><td>ACL++ 2024-06-03 错单</td><td><strong>采纳思路</strong></td><td>P1</td><td>事件真实；须在样例日回放 <code>trades_v1</code> 验证 <code>conditions</code> 是否已剔除</td></tr>
          <tr><td><code>compute_purified_mask</code></td><td><strong>合并命名</strong></td><td>—</td><td>与 <code>compute_mask</code> 同义；工程只维护<strong>一列</strong>，避免双总闸</td></tr>
        </tbody>
      </table>
      <div class="box-warn">
        <strong>勿照搬 AI 叙事：</strong>删去「Millennium/Point72/顶刊」等无法在本仓库核实的背书；掩码列的价值是<strong>可测试契约</strong>（金样例 + 单元断言），不是堆列名。
      </div>
    </section>
"""

MM5_OLD = """      <h3>MM-5 · ADR 跨国多重上市：母所时间错配（新增）</h3>
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
      <p><strong>任务：</strong> <code>PREPROC-035</code>（P1）— 新增加密列 + ADR 元数据表。</p>"""

MM5_NEW = """      <h3>MM-5 · ADR：默认剔除 vs 专题中性化（裁量后）</h3>
      <div class="box-scenario">
        <strong>业务场景（成立）：</strong><code>TSM</code>、<code>ASML</code> 等 ADR 的会计准则、披露时区与纯美股不同；PiT 与日 K 对齐更易出偏。
        <ul>
          <li><strong>本库事实：</strong>Massive 11TB 为<strong>美股 SIP + 美 SEC 基本面</strong>，<strong>不含</strong>台湾/欧洲母所行情。</li>
        </ul>
      </div>
      <div class="box-warn">
        <strong>默认策略（推荐）：</strong>在 <code>PREPROC-001</code> 用 CIK/名称/SIC 规则识别 ADR → <code>is_adr_asset=1</code> →
        <strong>默认 <code>in_universe_base=0</code></strong>（与预处理姊妹方案「ADR 剔除」一致）。不做 <code>parent_close_fx_adj</code>。
      </div>
      <h4>可选专题轨（P2，非默认）</h4>
      <ul>
        <li>仅当团队<strong>外接母所收盘价+FX</strong>且明确做 ADR 策略时：才物化 <code>parent_close_fx_adj</code> 与分钟价差中性化</li>
        <li>PiT：海外 <code>knowledge_ts_utc</code> 不得早于美股当日首 RTH bar（仍适用 <code>PREPROC-022</code>）</li>
      </ul>
      <p><strong>任务：</strong> <code>PREPROC-035</code>（<span class="badge badge-p2">P2</span>）— 默认仅 ADR 识别+universe 剔除；母股列属可选增强。</p>"""

MM6_OLD = """      <p><strong>任务：</strong> <code>PREPROC-036</code>（P1）；与 <code>PREPROC-016</code>/<code>034</code> 分层实现。</p>"""

MM6_NEW = """      <p><strong>任务：</strong> <code>PREPROC-036</code>（<span class="badge badge-p2">P2</span>）— 新闻 P0 全量后再做；优先完成 <code>034</code> 同语种去重。</p>"""

MM_DESC_OLD = """        结合资产定价顶刊清洗规约与 Millennium/Point72 级 Feature Store 契约：在宽表传入"""

MM_DESC_NEW = """        下列规约经 <a href="#sec-acl-judgment">§工程裁量</a> 筛选：在宽表传入"""

MM_TABLE_M5 = """          <tr><td>M5</td><td>ADR 跨国多重上市时间错配</td><td>TSM、ASML；母所已收盘、美股未开盘</td><td><code>is_adr_asset</code>, <code>parent_close_fx_adj</code></td><td><code>PREPROC-035</code></td></tr>"""

MM_TABLE_M5_NEW = """          <tr><td>M5</td><td>ADR 识别与 universe</td><td>默认剔除；母股价差为可选专题</td><td><code>is_adr_asset</code>（<code>parent_close_fx_adj</code> 可选）</td><td><code>PREPROC-035</code> P2</td></tr>"""

MM_TABLE_M6 = """          <tr><td>M6</td><td>跨语种新闻语义重复</td><td>同一调查：Benzinga + 中文/欧洲媒体</td><td><code>is_semantic_duplicate</code></td><td><code>PREPROC-036</code></td></tr>"""

MM_TABLE_M6_NEW = """          <tr><td>M6</td><td>跨语种新闻语义重复</td><td>同事件多语种（新闻 P0 后）</td><td><code>is_semantic_duplicate</code></td><td><code>PREPROC-036</code> P2</td></tr>"""

CTRL_ADR = """          <tr><td>高频量价</td><td><code>is_adr_asset</code></td><td>ADR 存托凭证</td><td>1min 特征用 <code>parent_close_fx_adj</code> 价差中性化</td></tr>"""

CTRL_ADR_NEW = """          <tr><td>Universe</td><td><code>is_adr_asset</code></td><td>ADR 标识</td><td>默认 <code>in_universe_base=0</code>；母股列仅 P2 专题</td></tr>"""

TASK_035 = """          <tr><td>PREPROC-035</td><td><span class="badge badge-p1">P1</span></td><td>ADR is_adr_asset + parent_close_fx_adj 分钟价差中性化</td>"""

TASK_035_NEW = """          <tr><td>PREPROC-035</td><td><span class="badge badge-p2">P2</span></td><td>ADR 识别+默认 universe 剔除；母股/FX 列为可选专题</td>"""

TASK_036 = """          <tr><td>PREPROC-036</td><td><span class="badge badge-p1">P1</span></td><td>跨语种 Entity+Subject 去重 is_semantic_duplicate</td>"""

TASK_036_NEW = """          <tr><td>PREPROC-036</td><td><span class="badge badge-p2">P2</span></td><td>跨语种 Entity+Subject 去重（新闻 P0 后；优先 034）</td>"""


def main():
    html = HTML.read_text(encoding="utf-8")

    if 'id="sec-acl-judgment"' not in html:
        html = html.replace(
            '    </section>\n<section id="sec-readmap">',
            '    </section>\n' + JUDGMENT_SECTION.strip() + '\n\n    <section id="sec-readmap">',
            1,
        )

    for old, new in [
        (MM5_OLD, MM5_NEW),
        (MM6_OLD, MM6_NEW),
        (MM_DESC_OLD, MM_DESC_NEW),
        (MM_TABLE_M5, MM_TABLE_M5_NEW),
        (MM_TABLE_M6, MM_TABLE_M6_NEW),
        (CTRL_ADR, CTRL_ADR_NEW),
        (TASK_035, TASK_035_NEW),
        (TASK_036, TASK_036_NEW),
    ]:
        if old in html:
            html = html.replace(old, new, 1)

    # MM-3 heuristic note
    if "75/105 天阈值为" not in html:
        html = html.replace(
            "<li>当日出现批量回补 → <code>is_batch_filing=1</code>（驱动 CH/内存排序契约）</li>",
            "<li>Duration 阈值（如 75/105 天）为<strong>可调启发式</strong>，实施时用历史样本校准</li>\n"
            "        <li>当日出现批量回补 → <code>is_batch_filing=1</code>（驱动 CH/内存排序契约）</li>",
            1,
        )

    # MM-2 spinoff data caveat
    if "spinoff 字段" not in html and "MM-2 · Spinoff" in html:
        html = html.replace(
            "<p><strong>任务：</strong> <code>PREPROC-027</code>（P1）+ <code>PREPROC-003</code></p>",
            "<p class=\"muted\">若 <code>corporate_actions</code> 无 spinoff 比率：至少打 <code>is_spinoff_ex_day</code>，动量类因子 Ex-date 跳窗；完整平滑待事件表补全。</p>\n"
            "      <p><strong>任务：</strong> <code>PREPROC-027</code>（P1）+ <code>PREPROC-003</code></p>",
            1,
        )

    # readmap link
    if "sec-acl-judgment" not in html.split("sec-readmap")[1][:1200]:
        html = html.replace(
            "<tr><td>量化研究员</td><td>",
            "<tr><td>全员</td><td><a href=\"#sec-acl-judgment\"><strong>§裁量</strong></a></td><td>AI 草稿 vs 本库可落地项</td></tr>\n"
            "          <tr><td>量化研究员</td><td>",
            1,
        )

    html = re.sub(
        r"v3\.\d[^<]*",
        "v3.13 工程裁量（AI 草稿已筛选）",
        html,
        count=1,
    )

    HTML.write_text(html, encoding="utf-8")
    print("engineering judgment applied, lines:", html.count("\n") + 1)

    bp = BEAUTIFY.read_text(encoding="utf-8")
    if '"sec-arch", "sec-readmap"' in bp and '"sec-acl-judgment"' not in bp:
        bp = bp.replace(
            '"sec-arch", "sec-readmap"',
            '"sec-arch", "sec-acl-judgment", "sec-readmap"',
        )
        BEAUTIFY.write_text(bp, encoding="utf-8")
    subprocess.run(["python3", str(BEAUTIFY)], check=True)


if __name__ == "__main__":
    main()
