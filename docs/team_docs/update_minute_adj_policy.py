#!/usr/bin/env python3
"""Policy: minute bars for factor operators must be split-adjusted (Layer 2)."""
from pathlib import Path

HTML = Path(__file__).resolve().parent / "Massive数据治理与改进行动清单.html"
MD = Path(__file__).resolve().parent / "美股原始数据预处理标准方案_因子入模前.md"
CATALOG_PY = Path(__file__).resolve().parent / "build_data_catalog_html.py"

SEC_P1_BLOCK = """
      <h3 id="sec-operator-adj">3.2 因子算子（<code>ts_*</code> 长窗）与分钟复权定稿</h3>
      <p class="section-desc">
        <code>factor_engine</code> 中 <code>ts_mean(x, d)</code> 等的 <code>d</code> 是<strong>连续 bar 条数</strong>（非自然日）。
        日频 <code>d=1000</code> ≈ 四年交易日；分钟 <code>d=1000</code> 仅约数个交易日 — 但<strong>只要分钟 OHLC 进入算子，仍须与日频同一复权口径</strong>。
      </p>
      <table class="table-schema">
        <thead><tr><th>层级</th><th>分钟 K 存什么</th><th>因子引擎读什么</th></tr></thead>
        <tbody>
          <tr><td>Layer 1 <code>cleaned</code></td><td><strong>未复权</strong>（与供应商一致，不改 OHLCV）</td><td>❌ 禁止直接用于 <code>ts_*</code> 长窗因子</td></tr>
          <tr><td>Layer 2 物化</td><td><code>fact_bars_adjusted_minute/</code> 拆股复权 OHLCV</td><td>✅ <strong>默认唯一入口</strong>（与 <code>fact_bars_adjusted_daily</code> 同 <code>adj_method</code>）</td></tr>
          <tr><td>tick 微观</td><td>raw quotes/trades 可保留未复权</td><td>聚合成分钟/日后再走 Layer 2 复权；勿与 REST 日线混拼</td></tr>
        </tbody>
      </table>
      <div class="box-warn">
        <strong>定稿（2026-06）：</strong>因全库使用 WorldQuant 风格 <code>ts_*</code> 算子且参数可很长（如 1000 bar），
        <strong>分钟频入模前也必须拆股复权</strong>（<code>PREPROC-003</code> 扩展物化分钟表，<code>CH-007</code>）。
        未复权分钟仅允许：条件码/时段研究、tick 微观结构中间态，<strong>不得</strong>进入截面因子挖掘主路径。
      </div>
"""

REPLACEMENTS = [
    (
        "<strong>三条铁律（请贴到群公告）：</strong>\n        ① 两套日线不可混；② 基本面用 filing_date 不用 period_end；③ 不用 minute 加总校验 day。",
        "<strong>四条铁律（请贴到群公告）：</strong>\n        ① 两套日线不可混；② 基本面用 filing_date 不用 period_end；③ 不用 minute 加总校验 day；\n        ④ <strong>算子用价量（含分钟）须走复权物化</strong>（<code>fact_bars_adjusted_*</code>），禁止 cleaned 未复权 OHLC 直接进 <code>ts_*</code>。",
    ),
    (
        '<p>3. <strong>tick / 分钟线</strong>：一律不复权；若需要复权序列，只能自己用公司行动表回溯调整。</p>',
        '<p>3. <strong>tick / 分钟线</strong>：Layer 1 cleaned <strong>存未复权</strong>；<strong>因子算子路径</strong>须用 Layer 2 <code>fact_bars_adjusted_minute</code>（与日线同 <code>PREPROC-003</code> 拆股口径）。tick 微观可保留 raw，聚合后再复权。</p>',
    ),
    (
        "<tr><td>SIP <code>day_aggs</code> / minute</td><td><strong>不复权</strong> → 用 <code>splits.historical_adjustment_factor</code></td>",
        "<tr><td>SIP <code>day_aggs</code> / minute（cleaned）</td><td>存储未复权 → Layer 2 <strong>必须</strong> <code>PREPROC-003</code> 物化 <code>fact_bars_adjusted_*</code></td>",
    ),
    (
        "<tr><td>分钟 / 日内</td><td>✅ raw</td><td>RTH 过滤；勿校验 day volume</td><td>见 <a href=\"#sec-quality\">§2b</a></td></tr>",
        "<tr><td>分钟 / 日内（算子因子）</td><td>✅ 须复权物化</td><td><code>PREPROC-003</code> 分钟表 + RTH/<code>PREPROC-008</code>；勿校验 day volume</td><td><a href=\"#sec-operator-adj\">§3.2</a></td></tr>",
    ),
    (
        "<tr><td>跨粒度</td><td>日频因子只读 <code>day_aggs</code>；分钟因子只读 <code>minute</code></td><td><a href=\"#sec-p1\">§3 铁律③</a></td>",
        "<tr><td>跨粒度</td><td>日/分钟因子分别读 <code>fact_bars_adjusted_daily/minute</code></td><td><a href=\"#sec-p1\">§3</a>、<a href=\"#sec-operator-adj\">§3.2</a></td>",
    ),
    (
        "<tr><td>价源铁律</td><td>单策略单一复权口径</td><td>生产锚点：<strong>SIP cleaned day_aggs</strong> + 自算复权；<strong>禁止</strong>与 REST 日线拼接</td></tr>",
        "<tr><td>价源铁律</td><td>单策略单一复权口径</td><td>日+分钟均 <strong>SIP + splits</strong> 物化（<code>adj_method</code> 全库一致）；<strong>禁止</strong> REST 与 SIP 混拼</td></tr>",
    ),
    (
        "<tr><td>价量</td><td>REST≠SIP；分钟≠日 K；tick 未盲目删竞价</td><td><a href=\"#sec-p1\">§3</a>、<code>PREPROC-024</code></td></tr>",
        "<tr><td>价量</td><td>日/分钟算子须复权物化；分钟≠日 K 加总；tick 未盲目删竞价</td><td><a href=\"#sec-operator-adj\">§3.2</a>、<code>PREPROC-003/024</code></td></tr>",
    ),
    (
        "<tr><td>分钟物化 <code>fact_bars_adjusted_minute</code></td><td><strong>P2 可选</strong></td>",
        "<tr><td>分钟物化 <code>fact_bars_adjusted_minute</code></td><td><strong>P1 必做</strong>（算子因子）</td>",
    ),
    (
        "<tr><td>CH-007</td><td><span class=\"badge badge-p2\">P2</span></td><td>fact_bars_adjusted_minute CH 分区</td>",
        "<tr><td>CH-007</td><td><span class=\"badge badge-p1\">P1</span></td><td>fact_bars_adjusted_minute CH 分区（算子分钟因子入口）</td>",
    ),
    (
        "<tr><td>PREPROC-003</td><td><span class=\"badge badge-p0\">P0</span></td><td>SIP 复权模块 + 单测</td>",
        "<tr><td>PREPROC-003</td><td><span class=\"badge badge-p0\">P0</span></td><td>SIP 日+分钟拆股复权物化 + NVDA/TSLA 单测</td>",
    ),
    (
        '<td>1min×ticker</td><td>8</td><td>不复权</td><td>✅</td><td>✅</td><td>≠日K加总</td>',
        '<td>1min×ticker</td><td>8</td><td>cleaned 不复权；<strong>算子须 Layer2 复权</strong></td><td>✅</td><td>✅</td><td>≠日K加总；见 §3.2</td>',
    ),
    (
        "<tr><td>日内波动、成交量剖面、买卖价差、开盘竞价</td><td><strong>未复权</strong> + 时段/条件码（<code>PREPROC-008/024</code>）</td>",
        "<tr><td>纯 tick 微观（未入算子）</td><td>可 <strong>未复权</strong> + 条件码；<strong>分钟进算子须复权物化</strong></td>",
    ),
    (
        "<tr><td>5/15/60 分钟动量、跨夜收益、与复权日 panel join</td><td><strong>拆股复权</strong>（与日 K 同一套因子）；分红按策略单独开</td>",
        "<tr><td>分钟 OHLC 进入 <code>ts_*</code>（含长窗 d）</td><td><strong>必须</strong> <code>fact_bars_adjusted_minute</code>，与日线同 <code>adj_method</code></td>",
    ),
    (
        "<tr><td>minute✅ tick❌</td><td>✅ raw</td>",
        "<tr><td>minute✅ tick❌</td><td>算子用复权分钟；tick raw</td>",
    ),
]

CATALOG_REPLACEMENTS = [
    (
        '"adj_time": "不复权",\n            "cleaned": "✅",\n            "warn": "sum(minute.vol)≠day.vol",',
        '"adj_time": "cleaned 未复权；Layer2 算子须 fact_bars_adjusted_minute（PREPROC-003）",\n            "cleaned": "✅",\n            "warn": "sum(minute.vol)≠day.vol；禁止 cleaned 直接进 ts_*",',
    ),
]

MD_APPEND = """
### 21.4 分钟频算子因子须复权（2026-06 定稿）

| 层级 | 规则 |
|------|------|
| Layer 1 cleaned | 分钟 OHLCV **不改**（未复权存储） |
| Layer 2 / 因子引擎 | 凡 `ts_*` 等算子读取分钟价量 → **必须** `fact_bars_adjusted_minute`（与日线同 `PREPROC-003` / `adj_method`） |
| 长窗参数 | `d=1000` 在日频≈四年；在分钟仅数日 — 仍须复权以免窗内拆股假尖刺 |
| tick | 微观研究可用 raw；聚合到分钟后再复权 |

任务：`PREPROC-003`（扩展分钟物化）、`CH-007`（P1）。
"""


def main():
    html = HTML.read_text(encoding="utf-8")

    if 'id="sec-operator-adj"' not in html:
        for marker in (
            '<div class="box-warn">\n        <strong>三条铁律',
            '<div class="box-warn">\n        <strong>四条铁律',
        ):
            if marker in html:
                html = html.replace(marker, SEC_P1_BLOCK + "\n      " + marker, 1)
                break

    for old, new in REPLACEMENTS:
        if old in html:
            html = html.replace(old, new, 1)

    html = html.replace(
        "版本 v3.4 Deep Research 采纳（12f）+ 全库数据手册",
        "版本 v3.5 分钟算子须复权（§3.2）+ 全库数据手册",
    )

    HTML.write_text(html, encoding="utf-8")
    print("HTML updated")

    cat = CATALOG_PY.read_text(encoding="utf-8")
    for old, new in CATALOG_REPLACEMENTS:
        cat = cat.replace(old, new)
    CATALOG_PY.write_text(cat, encoding="utf-8")
    print("build_data_catalog_html.py updated")

    if MD.exists():
        md = MD.read_text(encoding="utf-8")
        if "21.4 分钟频算子" not in md:
            md = md.rstrip() + "\n" + MD_APPEND
            MD.write_text(md, encoding="utf-8")
            print("MD §21.4 appended")


if __name__ == "__main__":
    main()
