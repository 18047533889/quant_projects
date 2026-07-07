#!/usr/bin/env python3
"""Fill gaps in Massive HTML checklist."""
import re
from pathlib import Path

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"


def fix_toc(html: str) -> str:
    fixes = [
        ('<li><a href="#sec-cleaned">12. cleaned', '<li><a href="#sec-cleaned">13. cleaned'),
        ('<li><a href="#sec-tasks">14. 任务总表', '<li><a href="#sec-tasks">14. 任务总表'),  # ok
        ('<li><a href="#sec-cheatsheet">16. 一页纸', '<li><a href="#sec-cheatsheet">17. 一页纸'),
    ]
    for a, b in fixes:
        html = html.replace(a, b)
    # insert new toc entries after sec-p1
    ins = """        <li><a href="#sec-scenarios">3b. 场景矩阵（能否做因子）</a></li>
        <li><a href="#sec-sre">2c. SRE 三层下载门禁</a></li>
        <li><a href="#sec-pipeline">10b. Layer 2 物化管道</a></li>
"""
    if "sec-scenarios" not in html.split("<nav")[1].split("</nav>")[0]:
        html = html.replace(
            '        <li><a href="#sec-p1">3. P1 用法铁律</a></li>\n        <li><a href="#sec-download">',
            '        <li><a href="#sec-p1">3. P1 用法铁律</a></li>\n' + ins + '        <li><a href="#sec-download">',
        )
    return html


def remove_section(html: str, sec_id: str) -> str:
    return re.sub(
        rf'\s*<section id="{sec_id}">.*?</section>\s*',
        "\n",
        html,
        count=1,
        flags=re.DOTALL,
    )


def insert_after(html: str, after_id: str, block: str) -> str:
    pat = rf'(    <section id="{after_id}">.*?</section>\s*)'
    m = re.search(pat, html, flags=re.DOTALL)
    if not m:
        raise SystemExit(f"not found: {after_id}")
    return html[: m.end()] + "\n" + block + "\n" + html[m.end() :]


READMAP = """
    <section id="sec-readmap">
      <h2>读者导读（按角色）</h2>
      <table>
        <thead><tr><th>角色</th><th>必读章节</th><th>目标</th></tr></thead>
        <tbody>
          <tr><td>量化研究员</td><td><a href="#sec-empirical">§1b</a> <a href="#sec-p1">§3</a> <a href="#sec-scenarios">§3b</a> <a href="#sec-align">§9</a></td><td>知道能否做因子、价源与 PiT 红线</td></tr>
          <tr><td>数据工程</td><td><a href="#sec-p0">§2</a> <a href="#sec-sre">§2c</a> <a href="#sec-download">§4</a> <a href="#sec-quality">§2b</a></td><td>P0 补数、停 cron、三层门禁</td></tr>
          <tr><td>平台 / ETL</td><td><a href="#sec-pipeline">§10b</a> <a href="#sec-clickhouse">§11</a> <a href="#sec-preproc">§10</a></td><td>物化 Panel、导入 CH、日更</td></tr>
          <tr><td>评审 / PM</td><td><a href="#sec-tasks">§14</a> <a href="#sec-roadmap">§15</a> <a href="#sec-accept">§16</a></td><td>分工、阶段、验收</td></tr>
        </tbody>
      </table>
      <div class="box-warn">
        <strong>P0 前全局禁令：</strong>含基本面/做空/分红因子 — 训练与回测 <code>end_date ≤ 2023-12-31</code>；
        禁止用 workspace 增量目录；禁止 SIP+REST 混用。
      </div>
    </section>
"""

SRE = """
    <section id="sec-sre">
      <h2>2c. SRE 三层下载门禁（写 .ok 之前）</h2>
      <p class="section-desc">仅「行数 ≠ 2000/10000」<strong>不够</strong>；须叠加同比与业务覆盖检查（标准方案 §2.4）。</p>
      <table>
        <thead><tr><th>层级</th><th>规则</th><th>动作</th></tr></thead>
        <tbody>
          <tr><td><strong>L1 指纹</strong></td><td>行数 ∈ {2000, 10000}</td><td>不写 .ok；删旧 .ok；重下</td></tr>
          <tr><td><strong>L2 同比</strong></td><td>同年份 &lt; 上年同源 50%</td><td>报警；人工确认是否截断</td></tr>
          <tr><td><strong>L3 业务</strong></td><td>域内日期/集合覆盖异常</td><td>如 short_interest_2024 仅 1 个 settlement_date</td></tr>
        </tbody>
      </table>
<pre>def is_partition_suspect(path, prev_year_path):
    n = pq.read_metadata(path).num_rows
    if n in (2000, 10000):
        return True, "fingerprint"
    if prev_year_path and prev_year_path.exists():
        n_prev = pq.read_metadata(prev_year_path).num_rows
        if n &lt; 0.5 * n_prev:
            return True, "yoy_drop"
    return False, "ok"</pre>
      <p><strong>参数：</strong><code>max_pages=None</code>（勿用 999999）；重下 <code>--no-skip-existing</code>；cron 输出必须指向 <code>massive_parquet/raw_massive_data/</code>。</p>
    </section>
"""

SCENARIOS = """
    <section id="sec-scenarios">
      <h2>3b. 按场景：当前库能否直接做因子（§10.6）</h2>
      <table>
        <thead><tr><th>场景</th><th>能否直接用</th><th>必做步骤</th><th>注意</th></tr></thead>
        <tbody>
          <tr><td>日 K 技术因子（仅 OHLCV）</td><td>✅ 基本可用</td><td>universe、SIP 需复权</td><td>cleaned day_aggs 锚点</td></tr>
          <tr><td>日 K + ≤2023 基本面</td><td>✅</td><td>asof filing_date、PiT</td><td>禁 period_end</td></tr>
          <tr><td>日 K + 2024 基本面</td><td>❌</td><td>P0 重下三大报表</td><td>当前 10k 截断</td></tr>
          <tr><td>做空 2024</td><td>❌</td><td>重下 short_*</td><td>—</td></tr>
          <tr><td>分红复权 / 分红事件</td><td>❌</td><td>重下 dividends</td><td>每年 2k 封顶</td></tr>
          <tr><td>分钟 / 日内</td><td>✅ raw</td><td>RTH 过滤；勿校验 day volume</td><td>见 <a href="#sec-quality">§2b</a></td></tr>
          <tr><td>tick / LOB</td><td>⚠️</td><td>分区剪枝、条件码</td><td>无 cleaned</td></tr>
          <tr><td>新闻 / 风险 NLP</td><td>❌</td><td>重下 news、risk_factors</td><td>2k 快照</td></tr>
          <tr><td>10-K / 8-K 全文</td><td>❌</td><td>新下载任务</td><td>0 parquet</td></tr>
          <tr><td>读 CH 宽表挖因子</td><td>⚠️ 待建</td><td>PREPROC-007 + CH-002</td><td>见 <a href="#sec-clickhouse">§11</a></td></tr>
        </tbody>
      </table>
    </section>
"""

DOWNLOAD_DEEP = """
      <h3>4.1 下载代码审查（§10.9 摘要）</h3>
      <table>
        <thead><tr><th>脚本</th><th>用途</th><th>输出</th></tr></thead>
        <tbody>
          <tr><td><code>download_all_history.py</code></td><td>REST 全量</td><td>基本面、dividends、REST 日线</td></tr>
          <tr><td><code>download_history.py</code></td><td>S3 flatfiles</td><td>day/minute/tick</td></tr>
          <tr><td><code>async_universal_fetcher.py</code></td><td>日更（勿生产用）</td><td>默认 max_pages=10 → 1 万行封顶</td></tr>
        </tbody>
      </table>
      <h4>根因 A：max_pages 截断</h4>
      <table>
        <thead><tr><th>参数</th><th>最大行数</th><th>本地现象</th></tr></thead>
        <tbody>
          <tr><td>max_pages=2, limit=5000</td><td>10,000</td><td>2024 三大报表、short_*</td></tr>
          <tr><td>max_pages=2, limit=1000</td><td>2,000</td><td>dividends 每年、risk_factors、news</td></tr>
          <tr><td>max_pages=10（日更默认）</td><td>10,000</td><td>async_universal_fetcher L605</td></tr>
        </tbody>
      </table>
      <h4>根因 B：.ok + skip_existing 锁死坏分区</h4>
      <p>重下前必须删 <code>*.parquet.ok</code>，并 <code>--no-skip-existing</code>。</p>

      <h3>4.2 增量 / 日更链路（§10.10 — 当前不可用）</h3>
      <div class="box-danger">
        <strong>结论：</strong>增量链路<strong>不能</strong>替代全量补数；<strong>暂停</strong>
        <code>daily_update_scheduler.sh</code> 与 <code>minute_update_scheduler.sh</code>，直至输出改到主库。
      </div>
      <table>
        <thead><tr><th>问题</th><th>严重度</th><th>说明</th></tr></thead>
        <tbody>
          <tr><td>输出写到 workspace，非 massive_parquet</td><td>🔴 P0</td><td><code>production_update_runner.py</code> → lfl_workspace</td></tr>
          <tr><td>S3 增量只处理 .parquet，忽略 .csv.gz</td><td>🔴 P0</td><td>flatfiles 拉不下来</td></tr>
          <tr><td>SDK 增量硬编码 limit=1000</td><td>🔴 P0</td><td>天然截断</td></tr>
          <tr><td><code>incremental_update_master.py</code> 不存在</td><td>🟠 P1</td><td>cron/README 引用失效</td></tr>
          <tr><td>增量 merge 无业务主键 concat</td><td>🟠 P1</td><td>可能丢历史行</td></tr>
          <tr><td>未调用 massive_cleaning_framework</td><td>🟡 P2</td><td>与全量 cleaned 不一致</td></tr>
        </tbody>
      </table>
      <h4>目标态架构（生产下载）</h4>
<pre>massive_parquet/raw_massive_data/  ← 唯一主库
    ▲ REST: download_all_history (max_pages=None, L1-L3 后写 .ok)
    ▲ S3:   download_history (workers=8-16, local-root=.../us_stocks_sip/...)
    → massive_cleaning_framework → cleaned_massive_data/
❌ 暂停: production_update_runner / async_incremental_update 写 workspace</pre>
      <h4>主库 SIP 补最新日（示例）</h4>
<pre>python -m raw_data_layer.raw_data_fetching.run_pipeline download-history \\
  --year 2026 --month 3 \\
  --prefix-template "us_stocks_sip/day_aggs_v1/{year}/{month:02d}/" \\
  --local-root /home/yluel/share/projects/massive_parquet/raw_massive_data/us_stocks_sip/day_aggs_v1 \\
  --workers 16</pre>
"""

PIPELINE = """
    <section id="sec-pipeline">
      <h2>10b. Layer 2 物化管道（Parquet → ClickHouse）</h2>
      <p class="section-desc">对应 <code>FEAT-001</code> / <code>PREPROC-002~007</code>；<strong>在 P0 通过后</strong>再跑 2024+ 基本面列。</p>

      <h3>10b.1 端到端顺序（不可跳步）</h3>
<pre>Stage0 截断扫描 PASS
  → dim: calendar, security_master
  → universe_daily（稀疏，来自当日 day_aggs ∩ CS）
  → fact: bars_adjusted_daily（SIP×splits 或 REST 二选一）
  → fact: returns_daily
  → pit: fundamentals_events → fundamentals_pit（PiT 分组）
  → panel_daily 宽表（按 §9 merge 顺序）
  → 写 materialized_panel/year=YYYY/month=MM/*.parquet
  → CH-002 导入 qs_massive.panel_daily</pre>

      <h3>10b.2 Panel 构建逻辑（概念代码）</h3>
<pre># build_panel_daily.py — 骨架，生产需按年分批
def build_panel_for_dates(trade_dates, price_source="sip_adj"):
    bars = load_bars_adjusted(trade_dates)      # 锚点
    uni = load_universe_daily(trade_dates)
    ret = load_returns_daily(trade_dates)
    pit = load_fundamentals_pit()               # 长表
    panel = bars.merge(uni, on=["trade_date","ticker"], how="left")
    panel = panel.merge(ret, on=["trade_date","ticker"], how="left")
    panel = asof_join_pit(panel, pit)           # filing_date &lt;= trade_date
    panel["price_source"] = price_source
    panel["batch_id"] = make_batch_id()
    assert_no_duplicate_keys(panel, ["trade_date", "ticker"])
    write_parquet_hive(panel, "materialized_panel/layer2_factor_ready_panel/")
    return panel</pre>

      <h3>10b.3 因子入模前 vs 入模后（边界）</h3>
      <table>
        <thead><tr><th>层</th><th>做什么</th><th>不做什么</th></tr></thead>
        <tbody>
          <tr><td>Layer 2（本文档）</td><td>复权、Universe、PiT、panel 物化、CH 加载</td><td>MAD、Barra、IC/IR</td></tr>
          <tr><td>因子引擎</td><td>表达式、截面算子</td><td>—</td></tr>
          <tr><td>factor_evaluation</td><td>Winsorize、中性化、IC、入库</td><td>改 OHLCV 真相</td></tr>
        </tbody>
      </table>

      <h3>10b.4 产物路径一览</h3>
      <table>
        <thead><tr><th>产物</th><th>Parquet 路径</th><th>ClickHouse 表</th></tr></thead>
        <tbody>
          <tr><td>宽表</td><td><code>materialized_panel/layer2_factor_ready_panel/</code></td><td><code>qs_massive.panel_daily</code></td></tr>
          <tr><td>PiT 长表</td><td><code>.../pit_fundamentals/</code></td><td><code>qs_massive.pit_fundamentals</code></td></tr>
          <tr><td>Universe</td><td><code>.../universe_daily/</code></td><td><code>qs_massive.dim_universe_daily</code></td></tr>
        </tbody>
      </table>
    </section>
"""

CHEATSHEET_FIX = {
    "§10.6、附录 I.4": "#sec-scenarios",
    "§10.9.4、附录 H": "#sec-p0",
    "§4.0": "#sec-empirical",
    "§4.0.1": "#sec-quality",
    "§4.0.2、§10.9.2": "#sec-quality",
    "§10.10": "#sec-download",
    "§8、附录 E": "#sec-align",
    "附录 F、数据字典 md": "Massive原始数据报告.md",
}


def main():
    html = HTML.read_text(encoding="utf-8")

    html = fix_toc(html)

    for sid, block in [
        ("sec-readmap", READMAP),
        ("sec-sre", SRE),
        ("sec-scenarios", SCENARIOS),
        ("sec-pipeline", PIPELINE),
    ]:
        html = remove_section(html, sid)

    html = insert_after(html, "sec-arch", READMAP.strip() + "\n")
    html = insert_after(html, "sec-quality", SRE.strip() + "\n")
    html = insert_after(html, "sec-p1", SCENARIOS.strip() + "\n")
    html = insert_after(html, "sec-preproc", PIPELINE.strip() + "\n")

    if "4.1 下载代码审查" not in html:
        # insert before closing sec-download
        idx = html.find('    <section id="sec-download">')
        end = html.find('    </section>', idx)
        html = html[:end] + "\n" + DOWNLOAD_DEEP + html[end:]

    # arch: factor boundary
    if "因子入模后" not in html.split("sec-arch")[1][:2500]:
        html = html.replace(
            "</table>\n      <div class=\"box-info\">\n        <strong>执行载体：</strong>",
            """</table>
      <h3>0.3 因子入模前 / 入模后边界</h3>
      <table>
        <thead><tr><th>阶段</th><th>包含</th><th>不包含</th></tr></thead>
        <tbody>
          <tr><td>Layer 1 清洗</td><td>ticker、align_time、explode</td><td>复权、PiT、Panel</td></tr>
          <tr><td>Layer 2（本手册 §10–§11）</td><td>复权、Universe、PiT、panel、CH</td><td>MAD、Barra、IC/IR</td></tr>
          <tr><td>因子评估层</td><td>去极值、中性化、IC、入库</td><td>改 Massive 原始 OHLCV</td></tr>
        </tbody>
      </table>
      <div class="box-info">
        <strong>执行载体：</strong>""",
        )

    # header
    html = html.replace(
        "含 P0 补数、ATR（A/T/R）工程规范、基本面/新闻字段形态、异构多源对齐步骤、任务分工与验收。",
        "含 P0 补数、ATR、多源对齐、<strong>ClickHouse 存取</strong>、下载/cron 审查、场景矩阵、物化管道与任务验收。",
    )
    html = html.replace("版本 v2.0 HTML 综合版", "版本 v2.1 HTML 综合版")

    # cheatsheet links
    for old, new in CHEATSHEET_FIX.items():
        if old in html and new.startswith("#"):
            html = html.replace(f"<td>{old}</td>", f'<td><a href="{new}">本手册{new}</a></td>')

    # sec-tasks: add PREPROC/CH note row if missing pipeline ref
    if "sec-pipeline" not in html[html.find("sec-tasks") : html.find("sec-tasks") + 8000]:
        pass

    HTML.write_text(html, encoding="utf-8")
    print("filled gaps, lines:", len(html.splitlines()))


if __name__ == "__main__":
    main()
