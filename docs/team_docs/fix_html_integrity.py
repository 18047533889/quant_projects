#!/usr/bin/env python3
"""Deduplicate catalog, extend §12f, convert appendix JSON samples to tables."""
import re
from pathlib import Path

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"

DEEP_RESEARCH_APPEND = """
      <h3>七、CRSP 缩放公式（学术对照 · Layer 2 实现用）</h3>
      <table class="table-schema">
        <thead><tr><th>维度</th><th>公式（后复权思想）</th><th>Massive 落地</th></tr></thead>
        <tbody>
          <tr><td>调整后 Open/High/Low</td><td>原价 × (Adjusted Close / Unadjusted Close)</td><td><code>PREPROC-003</code> 自 SIP + splits；或 REST 已拆股复权 OHLC</td></tr>
          <tr><td>调整后 Volume</td><td>原量 ÷ (Adjusted Close / Unadjusted Close)</td><td><strong>仅拆股</strong>缩放；分红走 <code>PREPROC-017</code> 不调量</td></tr>
          <tr><td>指标 Warm-up</td><td>公司行动后滚动窗口重置</td><td>因子引擎在 split/ex-date 后重算 MA/波动率窗口</td></tr>
          <tr><td>高分红标的</td><td>防负价（GE/TOPS）</td><td><code>PREPROC-019</code> floor + <code>price_clamped_flag</code></td></tr>
        </tbody>
      </table>

      <h3>八、截断清单（报告 P0 · 与本库实证一致）</h3>
      <table class="table-schema">
        <thead><tr><th>文件模式</th><th>异常行数</th><th>对比</th><th>状态</th></tr></thead>
        <tbody>
          <tr><td><code>*_2024</code> 三大报表 + cash_flow</td><td>各 <strong>10,000</strong></td><td>2023 年 2.4万~4.2万</td><td><span class="badge badge-p0">P0 重下</span></td></tr>
          <tr><td><code>short_interest_2024</code></td><td><strong>10,000</strong>（仅 1 个 settlement 日）</td><td>2023: 470,159</td><td><span class="badge badge-p0">P0</span></td></tr>
          <tr><td><code>short_volume_2024</code></td><td><strong>10,000</strong></td><td>2025: 3,459,381</td><td><span class="badge badge-p0">P0</span></td></tr>
          <tr><td><code>dividends_{2003..2026}</code></td><td>每年 <strong>2,000</strong></td><td>2000–2002 正常个位数</td><td><span class="badge badge-p0">P0</span></td></tr>
          <tr><td><code>risk_factors_{2015..2026}</code></td><td>每年 <strong>2,000</strong></td><td>—</td><td><span class="badge badge-p0">P0</span></td></tr>
          <tr><td><code>news_all</code> / <code>all_tickers_all</code></td><td>各 <strong>2,000</strong></td><td>非全量宇宙</td><td><span class="badge badge-p1">P1</span></td></tr>
          <tr><td><code>sec_edgar_index_all</code></td><td><strong>10,000</strong></td><td>分页快照</td><td><span class="badge badge-p1">P1</span></td></tr>
          <tr><td><code>10k_sections</code> / <code>8k_text</code></td><td><strong>0 文件</strong></td><td>未纳入下载脚本</td><td><code>FEAT-006</code></td></tr>
        </tbody>
      </table>
      <p class="section-desc">根因：<code>async_universal_fetcher</code> 默认 <code>max_pages=10</code> × page_size 1000；<code>.ok</code> + <code>skip_existing</code> 假性锁死。见 <a href="#sec-p0">§2</a>、<code>TASK-ENG-002</code>。</p>

      <h3>九、基础设施与灾备（报告提及）</h3>
      <table class="table-schema">
        <thead><tr><th>路径 / 组件</th><th>用途</th><th>手册</th></tr></thead>
        <tbody>
          <tr><td><code>/home/yluel/share/project_data_backup/</code></td><td>tick 灾备 ~11 TB（quotes/trades 与主库对齐；无 day/基本面）</td><td><a href="#sec-arch">§0</a></td></tr>
          <tr><td><code>daily_update_scheduler.sh</code></td><td>日更 — <strong>修复前暂停</strong></td><td><code>TASK-ENG-004</code></td></tr>
          <tr><td><code>production_update_runner.py</code></td><td>增量写 workspace，未接主库</td><td><code>TASK-ENG-004</code></td></tr>
          <tr><td><code>composite_source.py</code> / <code>Database_original.py</code></td><td>asof / exact join</td><td><a href="#sec-align">§9</a></td></tr>
          <tr><td><code>materialized_panel/</code> Layer 1.5</td><td>物化中间层</td><td><a href="#sec-engineering">§10c</a></td></tr>
        </tbody>
      </table>

      <h3>十、入模前完整性自检（仅看 HTML 即可勾选）</h3>
      <table class="table-schema">
        <thead><tr><th>类别</th><th>检查项</th><th>章节</th></tr></thead>
        <tbody>
          <tr><td>数据形态</td><td>26 源路径/列/样例表齐全</td><td><a href="#sec-data-catalog">§数据手册</a></td></tr>
          <tr><td>价量</td><td>REST≠SIP；分钟≠日 K；tick 未盲目删竞价</td><td><a href="#sec-p1">§3</a>、<code>PREPROC-024</code></td></tr>
          <tr><td>复权</td><td>单一口径 + splits/dividends 因子</td><td><code>PREPROC-003/017</code></td></tr>
          <tr><td>基本面</td><td>filing_date PiT；禁 period_end；重述版本</td><td><a href="#sec-time">§8</a>、<code>PREPROC-005</code></td></tr>
          <tr><td>文本</td><td>explode；HTML 剥离；LLM 匿名；信源权重</td><td><code>NLP-001~003</code></td></tr>
          <tr><td>完整性</td><td>无 10,000/2,000 截断；无假 .ok</td><td><a href="#sec-p0">§2</a></td></tr>
          <tr><td>极端</td><td>熔断/停牌/IPO/仙股/幽灵成交</td><td><a href="#sec-blind-spots-4">§12d</a></td></tr>
          <tr><td>防腐</td><td>实体 ID、Mask-First、分钟 CH 分区</td><td><a href="#sec-blind-spots-3">§12c</a>、§12e 表 A/B/C</td></tr>
        </tbody>
      </table>
"""

INCOME_SAMPLE_TABLE = """
<details class="sample-fold"><summary>C.1.3 样例（raw · 表格）</summary><div class="fold-body">
<table class="table-schema"><thead><tr><th>字段</th><th>值（示意）</th></tr></thead><tbody>
<tr><td><code>cik</code></td><td>0000104169</td></tr>
<tr><td><code>tickers</code></td><td>["WMT"]</td></tr>
<tr><td><code>filing_date</code></td><td>2010-06-04</td></tr>
<tr><td><code>period_end</code></td><td>2009-04-30</td></tr>
<tr><td><code>timeframe</code></td><td>quarterly</td></tr>
<tr><td><code>revenue</code></td><td>94242000000.0</td></tr>
<tr><td><code>basic_earnings_per_share</code></td><td>0.26</td></tr>
</tbody></table>
</div></details>"""

CLEANED_SAMPLE_TABLE = """
<details class="sample-fold"><summary>C.1.4 cleaned 单行追加字段（表格示意）</summary><div class="fold-body">
<table class="table-schema"><thead><tr><th>列</th><th>示例值</th></tr></thead><tbody>
<tr><td><code>source</code></td><td>fundamentals/income_statement</td></tr>
<tr><td><code>ticker</code></td><td>WMT</td></tr>
<tr><td><code>align_time</code></td><td>2010-06-04T00:00:00Z</td></tr>
<tr><td><code>frequency</code></td><td>quarterly_annual</td></tr>
<tr><td><code>primary_key</code></td><td>（哈希）</td></tr>
</tbody></table>
</div></details>"""

RATIOS_SAMPLE_TABLE = """
<details class="sample-fold"><summary>样例（raw · 表格）</summary><div class="fold-body">
<table class="table-schema"><thead><tr><th>ticker</th><th>date</th><th>price</th><th>market_cap</th><th>PE</th><th>ROE</th></tr></thead><tbody>
<tr><td>A</td><td>2026-03-09</td><td>116.64</td><td>32962734255</td><td>25.55</td><td>0.1867</td></tr>
<tr><td>AA</td><td>2026-03-09</td><td>61.16</td><td>16136438621</td><td>13.95</td><td>0.1891</td></tr>
</tbody></table>
</div></details>"""

NEWS_SAMPLE_TABLE = """
<details class="sample-fold"><summary>样例（raw · 表格）</summary><div class="fold-body">
<table class="table-schema"><thead><tr><th>published_utc</th><th>tickers</th><th>title(截断)</th><th>sentiment</th></tr></thead><tbody>
<tr><td>2026-03-10T14:00:00Z</td><td>SYANY</td><td>Staff elected to AL Sydbank…</td><td>neutral</td></tr>
<tr><td>2026-03-10T13:40:26Z</td><td>FTCI</td><td>FTC Solar 1-GW Deal…</td><td>positive</td></tr>
</tbody></table>
</div></details>"""


def dedupe_data_catalog(html: str) -> str:
    marker = '<section id="sec-data-catalog">'
    while html.count(marker) > 1:
        first = html.find(marker)
        second = html.find(marker, first + 1)
        end = html.find("</section>", second)
        if end == -1:
            break
        end = html.find("\n", end + len("</section>"))
        html = html[:second] + html[end + 1 :]
    return html


def extend_deep_research(html: str) -> str:
    if "十、入模前完整性自检" in html:
        return html
    needle = "以上 5 项已同步至"
    idx = html.find('id="sec-deep-research"')
    if idx != -1 and needle in html[idx:]:
        pos = html.find(needle, idx)
        box = html.rfind('<div class="box-info">', idx, pos)
        if box != -1:
            return html[:box] + DEEP_RESEARCH_APPEND + "\n      " + html[box:]
    return html


def fix_deep_research_tools_table(html: str) -> str:
    old = "<tr><td>Boilerpipe</td>"
    if "Newspaper" in html[html.find("sec-deep-research") : html.find("sec-tasks")]:
        return html
    insert = """<tr><td>Newspaper3k/4k</td><td>新闻抓取、元数据</td><td>通稿类站点；复杂 DOM 需抽检</td></tr>
          """
    return html.replace(old, insert + "          " + old, 1)


def convert_json_samples(html: str) -> str:
    html = re.sub(
        r'<details class="sample-fold"><summary>C\.1\.3 样例（raw JSON 形态）</summary><div class="fold-body"><div class="json-block">\{.*?\}</div></div></details>',
        INCOME_SAMPLE_TABLE.strip(),
        html,
        count=1,
        flags=re.DOTALL,
    )
    html = re.sub(
        r'<details class="sample-fold"><summary>C\.1\.4 cleaned 单行追加字段（示意）</summary><div class="fold-body"><div class="json-block">\{.*?\}</div></div></details>',
        CLEANED_SAMPLE_TABLE.strip(),
        html,
        count=1,
        flags=re.DOTALL,
    )
    html = re.sub(
        r'<details class="sample-fold"><summary>JSON 样例</summary><div class="fold-body"><div class="json-block">\{\s*"ticker": "A".*?\}</div></div></details>',
        RATIOS_SAMPLE_TABLE.strip(),
        html,
        count=1,
        flags=re.DOTALL,
    )
    html = re.sub(
        r'<details class="sample-fold"><summary>JSON 样例</summary><div class="fold-body"><div class="json-block">\{\s*"id":.*?\}</div></div></details>',
        NEWS_SAMPLE_TABLE.strip(),
        html,
        count=1,
        flags=re.DOTALL,
    )
    return html


def fix_build_catalog_patch():
    path = Path(__file__).resolve().parent / "build_data_catalog_html.py"
    text = path.read_text(encoding="utf-8")
    if "while html.count(marker)" in text:
        return
    old = """    if "sec-data-catalog" in html:
        pat_cat = re.compile(
            r'<section id="sec-data-catalog">.*?</section>',
            re.DOTALL,
        )
        html = pat_cat.sub(catalog.strip(), html, count=1)"""
    new = """    if "sec-data-catalog" in html:
        pat_cat = re.compile(
            r'<section id="sec-data-catalog">.*?</section>\\s*',
            re.DOTALL,
        )
        html = pat_cat.sub(catalog.strip() + "\\n", html, count=0)
        # count=0: 去掉重复注入的整段 catalog"""
    if old in text:
        path.write_text(text.replace(old, new), encoding="utf-8")


def main():
    fix_build_catalog_patch()
    html = HTML.read_text(encoding="utf-8")
    html = dedupe_data_catalog(html)
    html = extend_deep_research(html)
    html = fix_deep_research_tools_table(html)
    html = convert_json_samples(html)
    HTML.write_text(html, encoding="utf-8")
    html = HTML.read_text(encoding="utf-8")
    print("catalog sections:", html.count('id="sec-data-catalog"'))
    print("json-block (excl CSS):", len(re.findall(r'<div class="json-block">', html)))
    print("sec-deep-research checks:", "十、入模前完整性自检" in html)
    print("lines:", html.count("\n") + 1)


if __name__ == "__main__":
    main()
