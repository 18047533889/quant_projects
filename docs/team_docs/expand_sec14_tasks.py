#!/usr/bin/env python3
"""Expand §14 task table with plain-language descriptions."""
import re
from pathlib import Path

# (任务说明, 详见锚点)
TASKS = {
    "TASK-DATA-001": (
        "删除被截断数据旁边的 <code>.ok</code> 标记文件，让下载脚本能重新拉全量而不是跳过坏文件。",
        "#sec-p0",
    ),
    "TASK-DATA-002": (
        "重新下载 2024 财报、做空、分红等 P0 数据集（曾被限制为 10000/2000 行）。",
        "#sec-p0",
    ),
    "TASK-DATA-003": (
        "补数完成后重跑 Layer 1 清洗，并核对行数不再等于 10000/2000。",
        "#sec-p0",
    ),
    "TASK-ENG-001": (
        "改下载脚本：校验失败时<strong>不要</strong>写 <code>.ok</code>，避免把截断文件锁死。",
        "#sec-sre",
    ),
    "TASK-ENG-002": (
        "REST 下载默认取消 <code>max_pages</code> 上限，避免只下前两页。",
        "#sec-download",
    ),
    "TASK-ENG-004": (
        "暂停或修复会把数据写到错误路径的日更 cron，防止继续污染主库。",
        "#sec-download",
    ),
    "TASK-ENG-003": (
        "下载脚本支持 <code>--datasets</code> 只重下指定数据集，方便补数。",
        "#sec-download",
    ),
    "TASK-ENG-005": (
        "API 密钥改为环境变量，不要写死在代码里。",
        "#sec-download",
    ),
    "TASK-ENG-006": (
        "文档化 S3 并行下载参数和路径约定。",
        "#sec-download",
    ),
    "FEAT-001": (
        "开发 <code>build_panel_daily</code>：把 cleaned 数据拼成日频宽表 <code>panel_daily</code>（<strong>当前待建</strong>）。",
        "#sec-layer2-runbook",
    ),
    "FEAT-002": (
        "把三大报表按<strong>披露日</strong>对齐，生成 PiT 基本面视图（不能用 period_end 当信号日）。",
        "#sec-time",
    ),
    "FEAT-003": (
        "提供 SIP 未复权 K 线 + 拆股表的复权工具（或对接 Layer 2 物化结果）。",
        "#sec-adj-engineering",
    ),
    "PREPROC-002": (
        "生成<strong>每日股票池</strong> <code>dim_universe_daily</code>：哪天哪些股票允许进因子（不能只用「现在还活着的名单」回灌历史）。",
        "#sec-before-factor",
    ),
    "PREPROC-003": (
        "用拆股表把 SIP 日 K、分钟 K 调成复权价量，写入 <code>fact_bars_adjusted_*</code>；用 NVDA/TSLA 拆股日验收。",
        "#sec-adj-engineering",
    ),
    "PREPROC-005": (
        "把财报按披露日物化成 PiT 长表 <code>pit_fundamentals</code>，再 asof 贴到日线。",
        "#sec-time",
    ),
    "ENG-010": (
        "按 §10c 在 <code>materialized_panel/</code> 下落地 Layer 2 全部 Parquet 表 + 跑批脚本骨架（<strong>目录当前不存在</strong>）。",
        "#sec-engineering",
    ),
    "CH-001": (
        "在 ClickHouse 建库 <code>qs_massive</code> 及全部 Layer 2 表（DDL 见 §10c.11）。",
        "#sec-clickhouse",
    ),
    "CH-002": (
        "把 Parquet 宽表批量导入 ClickHouse，供 factor_engine SQL 读取。",
        "#sec-clickhouse",
    ),
    "CH-003": (
        "实现 T+1 日更：每天只追加/重写当月分区。",
        "#sec-clickhouse",
    ),
    "CH-004": (
        "让 data_access / factor_engine 通过统一 API 读 CH 宽表，不直接扫 cleaned。",
        "#sec-code-assets",
    ),
    "PREPROC-007": (
        "物化<strong>日频宽表</strong> <code>panel_daily</code>：价量+基本面+标志列拼在一起，这是 factor_engine 的主入口（<strong>待建</strong>）。",
        "#sec-before-factor",
    ),
    "PREPROC-001": (
        "建证券主数据：只保留普通股 CS；识别 ADR 并<strong>默认不进</strong>股票池。",
        "#sec-acl-mm",
    ),
    "PREPROC-004": (
        "计算并落盘日收益率表 <code>fact_returns_daily</code>（价格收益 / 可选总收益）。",
        "#sec-returns-policy",
    ),
    "PREPROC-006": (
        "在 PiT 对齐后算 LTM（滚动四季）和 PE/PB 等估值列。",
        "#sec-stage6-ltm",
    ),
    "PREPROC-010": (
        "Layer 2 跑批完成后自动生成质量日报（行数、重复键、截断扫描）。",
        "#sec-engineering",
    ),
    "FEAT-004": (
        "与 PREPROC-002 同义：Universe 表产品化接口。",
        "#sec-before-factor",
    ),
    "FEAT-005": (
        "数据质量监控看板或告警（截断、重复键、分区缺失）。",
        "#sec-quality",
    ),
    "FEAT-008": (
        "主库 SIP 日 K 的稳定日更 cron（写对路径、写对 .ok 逻辑）。",
        "#sec-download",
    ),
    "FEAT-006": (
        "下载 10-K / 8-K 全文（当前 0 文件，P2 扩展）。",
        "#sec-p0",
    ),
    "TASK-DC-001~004": (
        "维护 datasets.yaml、校验脚本、交易日历等数据目录配置（P2）。",
        "#sec-fields",
    ),
    "PREPROC-017": (
        "拆股时：价格 × 因子、成交量 ÷ 因子；<strong>现金分红只调价不调量</strong>。订单簿 size 同理。",
        "#sec-operator-adj-parity",
    ),
    "PREPROC-021": (
        "所有算子算之前必须先读 <code>compute_mask</code> 总开关；=0 的行不能进滚动窗口。",
        "#sec-acl",
    ),
    "PREPROC-022": (
        "盘中披露的 8-K/新闻：用<strong>秒级时间</strong>对齐，不能全天广播成 09:30 已知。",
        "#sec-acl",
    ),
    "ENG-012": (
        "登记大盘熔断（MWCB）时段，分钟 bar 打 <code>is_market_halt</code> 标志。",
        "#sec-acl",
    ),
    "ENG-013": (
        "维护真实交易日历；删掉周末/非交易日的「幽灵行」。",
        "#sec-acl",
    ),
    "PREPROC-018": (
        "个股停牌/LULD：补回分钟骨架，价 ffill、量=0，打停牌标志；停牌过久则当日出股票池。",
        "#sec-acl",
    ),
    "PREPROC-019": (
        "仙股多次合股后复权价可能 ≤0：落盘前 floor 并打标志，防止除零崩溃。",
        "#sec-acl",
    ),
    "PREPROC-020": (
        "IPO 首日 Opening Cross 前无成交：打 <code>is_ipo_pre_open</code>，禁止 ffill 发行价。",
        "#sec-acl",
    ),
    "PREPROC-025": (
        "长窗口因子（如 1000 日均线）需要比训练起点<strong>更早</strong>加载历史，跑批自动向前多取 K 日。",
        "#sec-acl",
    ),
    "PREPROC-026": (
        "负股东权益、零营收公司：打标志位，估值因子截面<strong>剔除</strong>，不要填 0 糊弄。",
        "#sec-acl-mm",
    ),
    "PREPROC-027": (
        "公司分拆（Spinoff）导致母股价格断崖：按分拆比率平滑历史价，不是真暴跌。",
        "#sec-acl-mm",
    ),
    "PREPROC-028": (
        "同一天挤进多份财报：按 filing_date + period_end 排序，避免 TTM 随机覆盖。",
        "#sec-acl-mm",
    ),
    "PREPROC-029": (
        "送股（10 送 1）应从 dividends 表路由到 splits 管道，保证市值守恒。",
        "#sec-acl-mm",
    ),
    "PREPROC-030": (
        "交易所作废的错单（如 BRK.A 闪崩）：回滚价格、打 <code>is_erroneous_glitch</code>。",
        "#sec-acl-deep2",
    ),
    "PREPROC-031": (
        "公司改财年导致过渡季只有 1～2 个月：打标志，TTM 用时间权重而非机械加四季。",
        "#sec-acl-deep2",
    ),
    "PREPROC-032": (
        "主板退市转 OTC 粉单：记录最后交易日收益，不要让回测里股票「蒸发消失」。",
        "#sec-acl-deep2",
    ),
    "PREPROC-033": (
        "延迟上报成交（Form T）：量可累加，但<strong>价不能</strong>进分钟最高/最低。",
        "#sec-acl-deep2",
    ),
    "PREPROC-034": (
        "同一新闻 24h 内重复发：SimHash 去重，计数和情绪分不重复累加。",
        "#sec-acl-deep2",
    ),
    "PREPROC-035": (
        "识别 ADR，默认不进 universe；母股价差中性化为可选专题（P2）。",
        "#sec-acl-mm",
    ),
    "PREPROC-036": (
        "同一事件多语种新闻：按实体+主题去重（P2，新闻补全后）。",
        "#sec-acl-mm",
    ),
    "PREPROC-037": (
        "把分钟 K 聚合成日频特征（realized vol、VWAP 等），写入 <code>fact_intraday_daily</code> 再 join 宽表。",
        "#sec-factor-freq-matrix",
    ),
    "PREPROC-038": (
        "1 分钟复权 K 聚合成 5/15/30 分钟，并传播 compute_mask（P2）。",
        "#sec-factor-freq-matrix",
    ),
    "FEAT-010": (
        "固定时刻截面（如每天 10:00 的 rank）专用表 <code>fact_minute_slice</code>。",
        "#sec-factor-freq-matrix",
    ),
    "FEAT-011": (
        "factor_engine 支持按 Track 读不同表 + 长窗口自动向前取历史 API。",
        "#sec-factor-freq-matrix",
    ),
    "PREPROC-039": (
        "每个入库因子登记元数据：用的哪条 Track、bar 频率、收益 horizon。",
        "#sec-factor-lifecycle",
    ),
    "PREPROC-040": (
        "做空数据半月才更新：加「距上次 settlement 几天」列，避免当 fresh 用。",
        "#sec-us-extremes",
    ),
    "PREPROC-041": (
        "维护 OPEX/FOMC/宏观发布等事件日历，join 到宽表（P2）。",
        "#sec-us-extremes",
    ),
    "PREPROC-042": (
        "分钟 RTH 过滤必须用美东时区（含夏令时切换），禁止写死 UTC 偏移。",
        "#sec-us-dst",
    ),
    "PREPROC-043": (
        "前一日跌超 10% 标 SSR 日（P2）。",
        "#sec-us-extremes",
    ),
    "PREPROC-044": (
        "一天多次停牌的 meme 股：打 <code>is_multi_halt_day</code> 等标志。",
        "#sec-us-extremes",
    ),
    "PREPROC-045": (
        "GOOG/GOOGL 等双类股：universe 去重，避免同一公司算两次（P2）。",
        "#sec-us-extremes",
    ),
    "PREPROC-046": (
        "建累计拆股因子审计表 <code>fact_cum_adj_factors</code>，复权链可回溯验收。",
        "#sec-adj-engineering",
    ),
    "PREPROC-047": (
        "某 ticker 发生新拆股时，该 ticker 全历史复权因子重算策略。",
        "#sec-layer2-runbook",
    ),
    "ENG-014": (
        "SPAC / 预发行（WI）等默认剔除出 universe（P2）。",
        "#sec-engineering",
    ),
    "PREPROC-011": (
        "ticker 更名（如 FB→META）用 <code>permanent_id</code> 串成一条长序列。",
        "#sec-edge-cases",
    ),
    "PREPROC-012": (
        "退市日写入惩罚收益策略，回测不能假装还能持有。",
        "#sec-edge-cases",
    ),
    "PREPROC-013": (
        "市值分母用 PiT 动态股本，不能 90 天不变常数。",
        "#sec-stage6-ltm",
    ),
    "CH-006": (
        "CH 宽表加 Projection，按 ticker 查长序列更快。",
        "#sec-clickhouse",
    ),
    "TASK-DC-005": (
        "tick 条件码过滤规则写入 yaml，团队统一协议。",
        "#sec-code-assets",
    ),
    "ENG-011": (
        "半日市（Black Friday 等）分钟 bar 只有 210 根而非 390 根。",
        "#sec-acl",
    ),
    "PREPROC-014": (
        "多源 join 前统一 ticker 后缀（如 .A / .WS 剥离规则）。",
        "#sec-atr",
    ),
    "PREPROC-015": (
        "股本变更分「有机增长」和「拆股/送股」，禁止 filing 之间 forward 插值。",
        "#sec-blind-spots-2",
    ),
    "PREPROC-016": (
        "新闻 SimHash 去重（与 PREPROC-034 合并实现）。",
        "#sec-acl-deep2",
    ),
    "PREPROC-023": (
        "tick 同一毫秒多笔：聚合成 VWAP 协议（P2）。",
        "#sec-blind-spots-3",
    ),
    "PREPROC-024": (
        "标记集合竞价时段；tick 异常剔除不能误伤开盘竞价。",
        "#sec-blind-spots-3",
    ),
    "NLP-001": (
        "10-K/8-K HTML 去样板再 NLP（P2，需先下载全文）。",
        "#sec-blind-spots",
    ),
    "NLP-002": (
        "LLM 打情绪分前对公司名匿名化（P2）。",
        "#sec-blind-spots",
    ),
    "NLP-003": (
        "新闻来源权威性权重（P2）。",
        "#sec-blind-spots",
    ),
    "CH-007": (
        "复权分钟 K 进 ClickHouse，按月+周分区（表当前待建）。",
        "#sec-clickhouse",
    ),
}

GROUPS = [
    ("A", "P0 · 数据补全（不修就不能做因子）", "badge-p0", [
        "TASK-DATA-001", "TASK-DATA-002", "TASK-DATA-003",
        "TASK-ENG-001", "TASK-ENG-002", "TASK-ENG-004",
    ]),
    ("B", "P0 · Layer 2 主链路（宽表 + 复权 + CH）", "badge-p0", [
        "ENG-010", "PREPROC-003", "PREPROC-002", "PREPROC-005", "PREPROC-007",
        "FEAT-001", "FEAT-002", "CH-001", "CH-002",
    ]),
    ("C", "P1 · 收益率 / 证券池 / 复权细节", "badge-p1", [
        "PREPROC-001", "PREPROC-004", "PREPROC-006", "PREPROC-010",
        "PREPROC-046", "PREPROC-047", "PREPROC-017",
        "FEAT-003", "FEAT-004", "FEAT-005",
        "CH-003", "CH-004", "FEAT-008",
    ]),
    ("D", "P1 · 入模前检查（ACL / 极端场景）", "badge-p1", [
        "PREPROC-021", "PREPROC-018", "PREPROC-019", "PREPROC-020", "PREPROC-025",
        "PREPROC-022", "PREPROC-024", "PREPROC-044",
        "ENG-011", "ENG-012", "ENG-013",
        "PREPROC-026", "PREPROC-027", "PREPROC-028", "PREPROC-029",
        "PREPROC-030", "PREPROC-031", "PREPROC-032", "PREPROC-033", "PREPROC-034",
        "PREPROC-042", "PREPROC-037",
    ]),
    ("E", "P1 · 平台 / 多频 / 因子元数据", "badge-p1", [
        "FEAT-010", "FEAT-011", "PREPROC-039", "PREPROC-040",
        "PREPROC-011", "PREPROC-012", "PREPROC-013", "PREPROC-014", "PREPROC-015",
        "CH-006", "CH-007", "TASK-DC-005",
    ]),
    ("F", "P2 · 增强 / 下载 / 新闻 / tick（可后排）", "badge-p2", [
        "TASK-ENG-003", "TASK-ENG-005", "TASK-ENG-006",
        "FEAT-006", "TASK-DC-001~004",
        "PREPROC-035", "PREPROC-036", "PREPROC-038", "PREPROC-041", "PREPROC-043",
        "PREPROC-045", "ENG-014", "PREPROC-016", "PREPROC-023",
        "NLP-001", "NLP-002", "NLP-003",
    ]),
]

def badge_for_id(tid):
    if tid.startswith("TASK-DATA") or tid in ("TASK-ENG-001", "TASK-ENG-002", "TASK-ENG-004"):
        return "badge-p0"
    if tid in ("ENG-010", "PREPROC-003", "PREPROC-002", "PREPROC-005", "PREPROC-007", "FEAT-001", "FEAT-002", "CH-001", "CH-002"):
        return "badge-p0"
    if "PREPROC-035" in tid or "PREPROC-036" in tid or tid.startswith("NLP-") or tid in ("FEAT-006", "TASK-DC-001~004", "PREPROC-038", "PREPROC-041", "PREPROC-043", "PREPROC-045", "ENG-014", "TASK-ENG-005", "TASK-ENG-006", "TASK-ENG-003", "PREPROC-016", "PREPROC-023"):
        return "badge-p2"
    return "badge-p1"

def row(tid, owner=""):
    desc, link = TASKS[tid]
    badge = badge_for_id(tid)
    owner_cell = f'<td class="owner-cell">{owner}</td>' if owner else '<td class="owner-cell"></td>'
    return (
        f'<tr><td><code>{tid}</code></td>'
        f'<td><span class="badge {badge}">{badge.replace("badge-","").upper()}</span></td>'
        f'<td class="task-desc">{desc}</td>'
        f'<td><a href="{link}">详见</a></td>'
        f'{owner_cell}'
        f'<td class="status-cell">待办</td><td></td></tr>'
    )

# owners from original table
OWNERS = {
    "ENG-010": "数据工程组",
    "PREPROC-017": "数据工程组",
    "PREPROC-021": "量化研究组",
    "PREPROC-022": "基础架构组",
    "ENG-012": "数据工程组",
    "ENG-013": "平台 SRE",
    "PREPROC-018": "量化研究组",
    "PREPROC-019": "基础架构组",
    "PREPROC-020": "基础架构组",
    "PREPROC-025": "平台/量化",
    "PREPROC-026": "量化研究组",
    "PREPROC-027": "基础架构组",
    "PREPROC-028": "基础架构组",
    "PREPROC-029": "基础架构组",
    "PREPROC-030": "数据工程组",
    "PREPROC-031": "数据工程组",
    "PREPROC-032": "算法策略组",
    "PREPROC-033": "数据工程组",
    "PREPROC-034": "量化研究组",
    "PREPROC-035": "数据工程组",
    "PREPROC-036": "量化研究组",
    "PREPROC-037": "数据工程组",
    "PREPROC-038": "数据工程组",
    "FEAT-010": "平台/量化",
    "FEAT-011": "平台/量化",
    "PREPROC-039": "量化研究组",
    "PREPROC-040": "数据工程组",
    "PREPROC-041": "数据工程组",
    "PREPROC-042": "数据工程组",
    "PREPROC-043": "量化研究组",
    "PREPROC-044": "量化研究组",
    "PREPROC-045": "基础架构组",
    "PREPROC-046": "数据工程组",
    "PREPROC-047": "数据工程组",
    "ENG-014": "数据工程组",
    "PREPROC-011": "基础架构组",
    "PREPROC-012": "算法策略组",
    "PREPROC-013": "数据工程组",
    "CH-006": "平台 SRE",
    "TASK-DC-005": "量化+数据",
    "ENG-011": "数据工程组",
    "PREPROC-014": "基础架构组",
    "PREPROC-015": "数据工程组",
    "PREPROC-016": "量化研究组",
    "PREPROC-023": "数据工程组",
    "PREPROC-024": "量化研究组",
    "NLP-001": "量化研究组",
    "NLP-002": "量化研究组",
    "NLP-003": "量化研究组",
    "CH-007": "平台 SRE",
}

def row_full(tid):
    return row(tid, OWNERS.get(tid, ""))

overview_rows = []
for letter, title, _, ids in GROUPS:
    overview_rows.append(
        f'<tr><td><strong>{letter}</strong></td><td>{title}</td>'
        f'<td>{len(ids)} 项</td><td>{", ".join(f"<code>{i}</code>" for i in ids[:4])}{"…" if len(ids)>4 else ""}</td></tr>'
    )

group_sections = []
for letter, title, badge_cls, ids in GROUPS:
    rows = "\n".join(row_full(i) for i in ids)
    open_attr = ' open' if letter in ("A", "B") else ""
    group_sections.append(f'''
      <details class="fold task-group"{open_attr}>
        <summary><strong>{letter}. {title}</strong>（{len(ids)} 项）</summary>
        <div class="fold-body">
        <table class="table-schema task-table">
        <thead><tr>
          <th>ID</th><th>优先级</th><th>任务说明</th><th>细节</th><th>负责人</th><th>状态</th><th>目标日期</th>
        </tr></thead>
        <tbody>
        {rows}
        </tbody>
        </table>
        </div>
      </details>''')

# remaining tasks not in groups
grouped_ids = {i for g in GROUPS for i in g[3]}
remaining = [k for k in TASKS if k not in grouped_ids]
if remaining:
    rows = "\n".join(row_full(i) for i in remaining)
    group_sections.append(f'''
      <details class="fold task-group">
        <summary><strong>F. 其他任务</strong>（{len(remaining)} 项）</summary>
        <div class="fold-body">
        <table class="table-schema task-table">
        <thead><tr>
          <th>ID</th><th>优先级</th><th>任务说明</th><th>细节</th><th>负责人</th><th>状态</th><th>目标日期</th>
        </tr></thead>
        <tbody>{rows}</tbody>
        </table>
        </div>
      </details>''')

new_section = f'''    <section id="sec-tasks">
      <h2><span class="sec-badge">14</span> 任务总表（评审会分工）</h2>
      <p class="section-desc">
        全手册待办汇总于此；任务说明见下表，实现细节点「详见」进正文。
        「负责人 / 状态 / 目标日期」列供评审会填写。
      </p>

      <div class="box-info" id="sec-tasks-howto">
        <strong>阅读顺序：</strong>
        <ol style="margin:8px 0 0;padding-left:1.3em">
          <li><strong>14.0 分组概览</strong>（补数 → 宽表 → 入模检查）</li>
          <li><strong>14.1 分组任务表</strong>；P0 的 A/B 组默认展开</li>
          <li>实现细节：点「详见」或查 §3e / §10c / §ACL</li>
        </ol>
        <p style="margin:10px 0 0"><strong>ID 前缀：</strong>
          <code>TASK-DATA</code> 补数 ·
          <code>TASK-ENG</code> 下载脚本 ·
          <code>FEAT</code> 产品能力 ·
          <code>PREPROC</code> Layer2 预处理 ·
          <code>CH</code> ClickHouse ·
          <code>ENG</code> 工程基础设施 ·
          <code>NLP</code> 文本（P2）
        </p>
        <p style="margin:6px 0 0"><strong>优先级：</strong>
          <span class="badge badge-p0">P0</span> 不修/不建就不能做生产因子 ·
          <span class="badge badge-p1">P1</span> 容易算歪或缺标志 ·
          <span class="badge badge-p2">P2</span> 增强项可后排
        </p>
      </div>

      <h3 id="sec-tasks-overview">14.0 分组概览</h3>
      <table class="table-schema">
        <thead><tr><th>组</th><th>阶段</th><th>项数</th><th>包含任务（示例）</th></tr></thead>
        <tbody>
        {"".join(overview_rows)}
        </tbody>
      </table>
      <p class="section-desc">推荐顺序：<strong>A → B → C → D → E</strong>；F 可后排。路线图见 <a href="#sec-roadmap">§15</a>，验收见 <a href="#sec-accept">§16</a>，入模闸门见 <a href="#sec-engine-checklist">§17</a>。</p>

      <h3 id="sec-tasks-detail">14.1 分组任务表（填负责人 / 状态）</h3>
      {"".join(group_sections)}
    </section>'''

path = Path("/home/yluel/share/projects/quantsociety_backend_project/docs/team_docs/Massive数据治理与改进行动清单.html")
html = path.read_text(encoding="utf-8")

html = re.sub(
    r'    <section id="sec-tasks">.*?</section>\s*\n\s*<!-- 8\. Roadmap -->',
    new_section + "\n\n    <!-- 8. Roadmap -->",
    html,
    count=1,
    flags=re.DOTALL,
)

# Add CSS for task-desc column
css_add = """
    .task-table .task-desc { font-size: 14px; line-height: 1.55; max-width: 520px; }
    .task-group { margin-bottom: 12px; }
"""
if ".task-table .task-desc" not in html:
    html = html.replace("</style>", css_add + "\n  </style>", 1)

html = html.replace("v3.31", "v3.32")
path.write_text(html, encoding="utf-8")
print("§14 updated, tasks:", len(TASKS))
