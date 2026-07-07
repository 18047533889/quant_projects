#!/usr/bin/env python3
"""Add comprehensive per-dataset catalog to Massive HTML."""
import re
from pathlib import Path

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"

# (id, name, path, row_semantics, cleaned, status_badge, status_class, problems, layer2_use, how_to_fix, tasks)
DATASETS = [
    # 行情
    ("day_aggs", "SIP 日 K", "us_stocks_sip/day_aggs_v1/{YYYY}/{MM}/{date}.parquet",
     "1 ticker × 1 交易日 OHLCV", "✅", "可用", "green",
     "5657 日完整；<strong>不复权</strong>；拆股日价格断崖",
     "因子锚点首选；Layer2 需 PREPROC-003 拆股复权或改 REST 价源",
     "① 勿与 REST 日线混用 ② 长周期动量必须复权 ③ 勿 minute 加总校验 volume ④ T+1 用 download_history 补主库",
     "PREPROC-003, FEAT-008, FEAT-003"),
    ("minute_aggs", "SIP 分钟 K", "us_stocks_sip/minute_aggs_v1/{YYYY}/{MM}/{date}.parquet",
     "1 ticker × 1 分钟 bar", "✅", "可用", "green",
     "覆盖与日 K 同；~1.9% 与日 volume 一致；含盘前盘后",
     "日内因子；单独分钟 Panel，禁止校验 day volume",
     "① RTH 过滤（09:30–16:00 ET）② 与日频特征只用 T-1 asof ③ 可选 FEAT-007 tick 规范",
     "PREPROC-008"),
    ("daily_market", "REST 全市场日线", "aggregate_bars/daily_market_summary/daily_market_summary_{YYYY}.parquet",
     "1 ticker × 1 日；字段 T/o/h/l/v", "✅", "可用", "green",
     "2004–2026；<strong>仅拆股复权</strong>，不含分红（KO/AAPL 实证）",
     "长周期价量/动量可选价源；与 SIP 二选一",
     "① 文档化 price_source=rest_split_only ② 勿与 SIP 混 join ③ 无需再拆股复权",
     "PREPROC-003（若选此价源）"),
    ("quotes", "SIP 逐笔报价 tick", "us_stocks_sip/quotes_v1/...",
     "逐笔 quote", "❌ 无", "raw 可用", "amber",
     "~10 TB；无 cleaned；ask/bid=0 需过滤",
     "LOB/微观结构；直读 raw + 分区剪枝",
     "① 暂不进 CH ② 条件码过滤 ③ CAST_INT 注意类型 ④ P3 再考虑 cleaned",
     "FEAT-007, PREPROC-008"),
    ("trades", "SIP 逐笔成交 tick", "us_stocks_sip/trades_v1/...",
     "逐笔 trade", "❌ 无", "raw 可用", "amber",
     "~10 TB；conditions ~24% null 为正常",
     "同 quotes；与 streaming 目录勿混",
     "① 同 quotes ② workers 降至 8–16 防 S3 429",
     "FEAT-007"),
    # 基本面
    ("balance_sheet", "资产负债表", "fundamentals/balance_sheet/balance_sheet_{YYYY}.parquet",
     "1 报告期 1 表；tickers[]→ticker", "✅", "2024 不可用", "p0",
     "<strong>2024 恰好 10,000 行</strong>（2023: 24,150）；PiT 用 filing_date",
     "asof→pit_*；禁 period_end；train≤2023",
     "① 删 balance_sheet_2024.parquet.ok ② REST 重下 max_pages=None ③ 重跑 cleaning ④ 建 fundamentals_pit",
     "TASK-DATA-001~003, PREPROC-005"),
    ("income_statement", "利润表", "fundamentals/income_statement/income_statement_{YYYY}.parquet",
     "1 报告期 1 表", "✅", "2024 不可用", "p0",
     "2024 10,000 行（2023: 42,602）",
     "同 balance_sheet；revenue 等 pit 列",
     "同 balance_sheet 流程",
     "TASK-DATA-*, PREPROC-005"),
    ("cash_flow", "现金流量表", "fundamentals/cash_flow_statement/cash_flow_statement_{YYYY}.parquet",
     "1 报告期 1 表", "✅", "2024 不可用", "p0",
     "2024 10,000 行（2023: 42,526）",
     "同 balance_sheet",
     "同 balance_sheet 流程",
     "TASK-DATA-*, PREPROC-005"),
    ("financials_ratios", "财务比率（日截面）", "fundamentals/financials_ratios/financials_ratios_all.parquet",
     "1 ticker × 1 自然日", "✅", "快照", "amber",
     "单文件 ~5199 行；<strong>非 PiT 财报</strong>；PE ~56% null",
     "校验用；生产 PE 优先自算 LTM",
     "① 勿当 PiT ② 与市值 join 用 date ③ P0 不阻断但非全历史 panel",
     "PREPROC-006"),
    ("short_interest", "做空余额", "fundamentals/short_interest/short_interest_{YYYY}.parquet",
     "1 ticker × settlement_date", "✅", "2024 不可用", "p0",
     "2024 10k 行且仅 1 个 settlement_date",
     "asof backward 贴日频",
     "① 重下 2024 ② 验证 settlement 覆盖全年",
     "TASK-DATA-*, PREPROC-007"),
    ("short_volume", "每日做空量", "fundamentals/short_volume/short_volume_{YYYY}.parquet",
     "1 ticker × 1 日", "✅", "2024 不可用", "p0",
     "2024 10,000 行（2025: 3.4M+）",
     "exact join 到 panel",
     "① 重下 2024 ② exact merge",
     "TASK-DATA-*"),
    ("stocks_floats", "自由流通股本", "fundamentals/stocks_floats/...",
     "1 ticker × effective_date", "✅", "可用", "green",
     "事件低频；一般正常",
     "asof 或 exact；流通市值",
     "① 确认行数无 2k/10k ② 纳入 panel 可选",
     "PREPROC-007"),
    # 公司行动
    ("dividends", "现金分红", "corporate_actions/dividends/dividends_{YYYY}.parquet",
     "1 分红事件", "✅", "全系列截断", "p0",
     "<strong>2003–2026 每年恰好 2,000 行</strong>；haf 19.8% null",
     "分红复权/总收益；P0 前<strong>禁止</strong>分红复权",
     "① 删全部 dividends_*.ok ② 重下 2003–2026 ③ 重跑 cleaning ④ 再开 ret_total",
     "TASK-DATA-*, PREPROC-003/004"),
    ("splits", "拆股/合股", "corporate_actions/splits/splits_{YYYY}.parquet",
     "1 拆股事件", "✅", "可用", "green",
     "historical_adjustment_factor 完整；拆股复权必需",
     "SIP 复权乘积；REST 已复权可不用",
     "① 与 day_aggs 联用 ② NVDA/TSLA 单测",
     "PREPROC-003, FEAT-003"),
    ("ipos", "IPO", "corporate_actions/ipos/ipos_all.parquet",
     "1 IPO 事件", "✅", "部分可用", "amber",
     "announced_date 80.8% null；listing_date 13.2% null",
     "事件研究；非日频 panel 核心",
     "① 按策略过滤 ipo_status ② 非 P0",
     "—"),
    # 申报
    ("risk_factors", "10-K 风险因子文本", "filing/risk_factors/risk_factors_{YYYY}.parquet",
     "1 条风险文本", "✅", "按年 2k 截断", "p0",
     "2015+ 每年 <strong>2,000 行</strong>",
     "NLP；PiT filing_date",
     "① 按年重下 ② 重跑 cleaning",
     "TASK-DATA-*"),
    ("sec_edgar", "SEC 申报索引", "filing/sec_edgar_index/sec_edgar_index_all.parquet",
     "1 条申报", "✅", "快照 10k", "p0",
     "<strong>10,000 行</strong>；ticker ~48% 空（cleaned 丢）",
     "文本链接；非核心价量",
     "① 全量重下 ② 需要时补 ticker",
     "TASK-DATA-*"),
    ("risk_categories", "风险分类字典", "filing/risk_categories/...",
     "分类码表", "✅", "可用", "green",
     "字典表；行数小",
     "join risk_factors 分类",
     "① 登记 datasets.yaml",
     "TASK-DC-002"),
    ("10k_sections", "10-K 章节全文", "filing/10k_sections/",
     "—", "❌", "未下载", "p0",
     "<strong>0 个 parquet</strong>",
     "NLP 不可用",
     "① FEAT-006 纳入 DATASETS ② download-all-history",
     "FEAT-006"),
    ("8k_text", "8-K 全文", "filing/8k_text/",
     "—", "❌", "未下载", "p0",
     "0 文件",
     "事件 NLP",
     "同 10k",
     "FEAT-006"),
    # 新闻
    ("news", "新闻", "news/news/news_all.parquet",
     "1 新闻 × ticker（explode）", "✅", "2k 快照", "p0",
     "<strong>2,000 行</strong>；published_utc 秒级",
     "signal_trade_date；情感来自 insights[]",
     "① 重下全量 ② PREPROC-009 盘后→次日 ③ 勿进默认 panel 直至全量",
     "TASK-DATA-*, PREPROC-009"),
    # 元数据
    ("all_tickers", "证券主数据", "tickers/all_tickers/all_tickers_all.parquet",
     "1 ticker 快照", "✅", "2k 快照", "p0",
     "<strong>2,000 行</strong>；勿当全历史 universe",
     "security_master 辅助；universe 以 day_aggs 为准",
     "① 重下 ② PREPROC-001 CS 过滤 ③ 禁稠密网格",
     "TASK-DATA-*, PREPROC-001/002"),
    ("ticker_types", "证券类型字典", "tickers/ticker_types/...",
     "类型码表", "✅", "可用", "green",
     "完整",
     "CS/ETF 过滤 lookup",
     "① TASK-DC-002 登记",
     "PREPROC-001, TASK-DC-002"),
    ("exchanges", "交易所", "market_operations/exchanges/...",
     "交易所元数据", "✅", "可用", "green",
     "完整",
     "辅助过滤",
     "登记 yaml",
     "TASK-DC-002"),
    ("market_holidays", "休市日", "market_operations/market_holidays/...",
     "节假日", "✅", "仅 2026–27", "amber",
     "<strong>不能</strong>校验历史交易日",
     "勿单独依赖；日历从 day_aggs union",
     "① TASK-DC-004 建历史 calendar ② dim_calendar",
     "TASK-DC-004, PREPROC-007"),
    ("condition_codes", "成交条件码", "market_operations/condition_codes/...",
     "条件码字典", "✅", "可用", "green",
     "tick 过滤必备",
     "tick 清洗 lookup",
     "登记 yaml",
     "TASK-DC-002"),
]

BADGE = {"green": "badge-ok", "amber": "badge-p1", "p0": "badge-p0", "可用": "badge-ok", "快照": "badge-p1"}


def row_card(d):
    id_, name, path, sem, cleaned, status, cls, prob, l2, fix, tasks = d
    bc = BADGE.get(cls, "badge-p1")
    return f"""
      <div class="task-card" id="ds-{id_}">
        <div class="task-card-header">
          <strong>{name}</strong>
          <span class="badge {bc}">{status}</span>
          <span class="badge badge-p1">cleaned: {cleaned}</span>
        </div>
        <div class="task-card-body">
          <table>
            <tbody>
              <tr><th style="width:140px">路径</th><td><code>raw_massive_data/{path}</code><br><code>cleaned_massive_data/{path}</code>（若有）</td></tr>
              <tr><th>一行代表</th><td>{sem}</td></tr>
              <tr><th>当前情况</th><td>{prob}</td></tr>
              <tr><th>因子入模前要做什么</th><td>{l2}</td></tr>
              <tr><th>数据层怎么改</th><td>{fix}</td></tr>
              <tr><th>任务 ID</th><td><code>{tasks}</code></td></tr>
            </tbody>
          </table>
        </div>
      </div>
"""


def build_section():
    overview_rows = ""
    for d in DATASETS:
        id_, name, path, sem, cleaned, status, cls, prob, l2, fix, tasks = d
        bc = BADGE.get(cls, "badge-p0" if cls == "p0" else "badge-p1")
        overview_rows += f"""          <tr>
            <td><a href="#ds-{id_}">{name}</a></td>
            <td><code>{path.split('/')[0]}/…</code></td>
            <td>{cleaned}</td>
            <td><span class="badge {bc}">{status}</span></td>
            <td>{prob[:80]}{'…' if len(prob)>80 else ''}</td>
            <td><code>{tasks.split(',')[0].strip()}</code></td>
          </tr>
"""

    cards = {
        "行情（5）": ["day_aggs", "minute_aggs", "daily_market", "quotes", "trades"],
        "基本面（7）": ["balance_sheet", "income_statement", "cash_flow", "financials_ratios", "short_interest", "short_volume", "stocks_floats"],
        "公司行动（3）": ["dividends", "splits", "ipos"],
        "申报 / 文本（5）": ["risk_factors", "sec_edgar", "risk_categories", "10k_sections", "8k_text"],
        "新闻（1）": ["news"],
        "元数据（5）": ["all_tickers", "ticker_types", "exchanges", "market_holidays", "condition_codes"],
    }
    by_id = {d[0]: d for d in DATASETS}

    detail = ""
    for cat, ids in cards.items():
        detail += f"\n      <h3>{cat}</h3>\n"
        for i in ids:
            detail += row_card(by_id[i])

    return f"""
    <section id="sec-datasets">
      <h2>1a. 全数据源现状与改法（26 源逐项说明）</h2>
      <p class="section-desc">
        根路径：<code>/home/yluel/share/projects/massive_parquet/</code>。
        <strong>cleaned 只</strong>统一 ticker/align_time，不改 OHLCV 数值；
        跨源 Panel、复权、PiT、进 ClickHouse 见 <a href="#sec-pipeline">§10b</a>、<a href="#sec-clickhouse">§11</a>。
      </p>

      <div class="box-info">
        <strong>图例：</strong>
        <span class="badge badge-ok">可用</span> 可直接或经 Layer2 后用；
        <span class="badge badge-p1">快照/部分</span> 能用但有限制；
        <span class="badge badge-p0">P0 阻断</span> 必须先 TASK-DATA 补数。
      </div>

      <h3>1a.0 总览表（点击跳转到详情）</h3>
      <table>
        <thead>
          <tr>
            <th>数据集</th><th>路径前缀</th><th>cleaned</th><th>状态</th><th>核心问题（摘要）</th><th>首要任务</th>
          </tr>
        </thead>
        <tbody>
{overview_rows}
        </tbody>
      </table>

      <h3>1a.1 cleaned 层统一做了什么 / 没做什么</h3>
      <table>
        <thead><tr><th>做了</th><th>没做（必须在 Layer 2 / 因子层）</th></tr></thead>
        <tbody>
          <tr>
            <td>explode tickers[]；ticker 大写；align_time UTC；主键去重；丢空 ticker</td>
            <td>复权；PiT asof；Universe；跨源 panel；缺失填 0；截面去极值/Barra</td>
          </tr>
        </tbody>
      </table>

      <h3>1a.2 按数据集详情（现状 + 怎么改）</h3>
{detail}

      <h3>1a.3 按「你要做什么因子」快速选数据源</h3>
      <table>
        <thead><tr><th>目标</th><th>用哪些源</th><th>别用 / 先修</th></tr></thead>
        <tbody>
          <tr><td>日频技术因子</td><td>cleaned day_aggs + splits</td><td>混 REST；未复权 SIP</td></tr>
          <tr><td>日频 + 基本面（≤2023）</td><td>day_aggs + 三大报表 PiT</td><td>2024 报表；period_end</td></tr>
          <tr><td>总收益 / 红利</td><td>day_aggs + dividends（修好后）</td><td>当前 dividends 全截断</td></tr>
          <tr><td>做空因子</td><td>short_*（修 2024 后）</td><td>当前 2024</td></tr>
          <tr><td>新闻情绪</td><td>news（全量重下后）</td><td>当前 2k</td></tr>
          <tr><td>读 CH 挖因子</td><td>qs_massive.panel_daily</td><td>先 PREPROC-007 + CH-002</td></tr>
        </tbody>
      </table>
    </section>
"""


def main():
    html = HTML.read_text(encoding="utf-8")
    block = build_section()

    html = re.sub(
        r'\s*<section id="sec-datasets">.*?</section>\s*',
        "\n",
        html,
        count=1,
        flags=re.DOTALL,
    )

    # insert after sec-ok
    pat = r'(    <section id="sec-ok">.*?</section>\s*)'
    m = re.search(pat, html, flags=re.DOTALL)
    if not m:
        raise SystemExit("sec-ok not found")
    html = html[: m.end()] + block + html[m.end() :]

    if "sec-datasets" not in html.split("</nav>")[0]:
        html = html.replace(
            '<li><a href="#sec-ok">1. 当前可用数据</a></li>',
            '<li><a href="#sec-ok">1. 当前可用（摘要）</a></li>\n        <li><a href="#sec-datasets">1a. 全数据源现状与改法（26 源）</a></li>',
        )

    # Expand sec-ok intro
    html = html.replace(
        '<h2>1. 当前可用的数据（勿重复修）</h2>',
        '<h2>1. 当前可用的数据（摘要）</h2>',
    )
    html = html.replace(
        "以下数据经扫描可认为<strong>主库完整</strong>",
        "以下仅列<strong>相对完整</strong>、可优先投入人力的源；<strong>全部 26 源</strong>的现状、问题与改法见 <a href=\"#sec-datasets\">§1a</a>。",
    )

    html = html.replace("版本 v2.1 HTML 综合版", "版本 v2.2 HTML 综合版")
    HTML.write_text(html, encoding="utf-8")
    print("catalog added, lines:", len(html.splitlines()))


if __name__ == "__main__":
    main()
