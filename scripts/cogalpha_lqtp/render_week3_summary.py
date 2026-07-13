#!/usr/bin/env python3
"""Generate Week3 volume-price factor summary with plain-Chinese explanations."""
from __future__ import annotations

import html as html_lib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.cogalpha_lqtp.report_html import (
    _enrich_topk_summary,
    _safe_float,
    _sanitize_eval_mode_label,
    resolve_compute_path,
)

DEFAULT_WORK_DIR = Path(__file__).resolve().parents[2] / "data/cogalpha_lqtp_production"

WEEK3_FACTORS = [
    "factor_persistence_ewma",
    "factor_vol_regime_vol_ratio",
    "factor_persistence",
    "factor_volume_abnormality_liquidity_gated",
    "factor_lag_vol_volatility_adjusted_gate",
    "factor_lag_response_vol_tool_adaptive",
    "factor_lag_volatility_regime_adapted",
    "factor_lag_vol_volatility_adjusted",
    "factor_vol_lag_ret_smoothed",
    "factor_adaptive_volume_smoothness",
    "factor_drawdown_depth_duration_range_vol",
    "factor_asym_vol_cont_gate_30",
    "factor_asym_ratio_pure",
    "factor_asym_vol_volume_cont_30",
    "factor_drawdown_vol_persistence",
    "factor_vol_gated_momentum_v2",
    "factor_range_asym_log_vol",
    "factor_resvol_volume_momentum",
    "factor_volume_price_divergence_cross_mut",
    "factor_range_asym_log_short",
]

# Plain-Chinese factor cards: theme, formula steps, direction hint
FACTOR_ZH: dict[str, dict[str, str]] = {
    "factor_persistence_ewma": {
        "theme": "收益路径平滑度（指数加权版）",
        "formula": (
            "① 算日收益率 r = close 相对昨收的涨跌幅；"
            "② 算相邻两日收益率变化 |r − 昨日r|；"
            "③ 对上述变化做 10 日指数加权平均 ema（近期权重更大）；"
            "④ 最后取负号：变化越小 → 走势越连贯 → 因子值越高。"
        ),
        "dsl": "-ema(abs(ts_delta(ts_pct(close, 1), 1)), 10)",
    },
    "factor_vol_regime_vol_ratio": {
        "theme": "高成交量下的振幅收缩",
        "formula": (
            "① 5 日平均振幅 ÷ 20 日平均振幅，前面加负号（短期振幅相对长期收窄时为正）；"
            "② 仅当当日成交量 > 30 日均量的 1.5 倍时才保留信号（放量才可信）；"
            "③ 全市场截面 rank 排序，得到 0~1 的分位值。"
        ),
        "dsl": "rank(-(ts_mean(high-low,5)/ts_mean(high-low,20)) * (volume>ts_mean(volume,30)*1.5))",
    },
    "factor_persistence": {
        "theme": "收益路径平滑度（简单滑动平均版）",
        "formula": (
            "与 persistence_ewma 逻辑相同，区别是第 ③ 步用 ts_mean（等权 10 日滑动平均）"
            "代替 ema，对近期突变不那么敏感。"
        ),
        "dsl": "-ts_mean(abs(ts_delta(ts_pct(close, 1), 1)), 10)",
    },
    "factor_volume_abnormality_liquidity_gated": {
        "theme": "成交量异常 × 流动性门控",
        "formula": (
            "① 当日成交量 ÷ 20 日成交量中位数，衡量放量/缩量程度；"
            "② 若 20 日平均成交额低于阈值（流动性差），因子置 0；"
            "③ 否则输出成交量相对中位数的偏离。"
        ),
        "dsl": "Python：rolling median / mean amount 门控",
    },
    "factor_lag_vol_volatility_adjusted_gate": {
        "theme": "滞后收益 × 对数成交量 × 波动调整（带风格门控）",
        "formula": (
            "① 5 日收益率 = close/5日前close − 1；"
            "② 对数(成交量相对 30 日均量的比值)；"
            "③ ATR(20)/close 作为波动率缩放；"
            "④ 三者相乘后做 EMA 平滑；高残差波动风格下用更长平滑窗口。"
        ),
        "dsl": "Python：ret_5 × log(vol_ratio) × ATR/close，EMA 门控",
    },
    "factor_lag_response_vol_tool_adaptive": {
        "theme": "量价滞后响应（自适应平滑）",
        "formula": (
            "与上一因子同类：5 日收益 × log(成交量状态比值) × ATR/close。"
            "区别在于：残差波动高时用 10 日 EMA，否则用 5 日 EMA，"
            "高噪环境下自动加大平滑。"
        ),
        "dsl": "Python：classify_volume_regime + 条件 EMA5/EMA10",
    },
    "factor_lag_volatility_regime_adapted": {
        "theme": "波动状态自适应的滞后量价",
        "formula": (
            "① 5 日收益 × log(成交量/EMA30) × ATR/close 构成原始信号；"
            "② 按波动状态切换 EMA 窗口（高波动拉长、低波动缩短），"
            "让因子在不同市场状态下保持可比的信噪比。"
        ),
        "dsl": "Python：vol regime 切换 EMA span",
    },
    "factor_lag_vol_volatility_adjusted": {
        "theme": "滞后收益 × 对数成交量 × ATR（DSL 版）",
        "formula": (
            "① 5 日收益率 close/delay(close,5)−1；"
            "② 乘以 log(volume/ema(volume,30))；"
            "③ 再乘以 ATR(20)/close；"
            "④ 整体做 5 日 ema 平滑。"
        ),
        "dsl": "ema((close/delay(close,5)-1)*log(volume/ema(volume,30))*(ATR/close), 5)",
    },
    "factor_vol_lag_ret_smoothed": {
        "theme": "量价滞后响应（平滑 DSL 版）",
        "formula": (
            "① 5 日收益 × log(成交量/20 日EMA成交量)；"
            "② 乘以 (1 + ATR/close) 放大高波动日；"
            "③ 5 日 ema 平滑输出。"
        ),
        "dsl": "ema((close/delay(close,5)-1)*log(volume/ema(volume,20))*(1+ATR/close), 5)",
    },
    "factor_adaptive_volume_smoothness": {
        "theme": "自适应成交量平滑度",
        "formula": (
            "① 算短期/长期成交量变异系数 CV 之比，衡量成交是否突然变得“毛躁”；"
            "② 高残差波动风格下加大权重；"
            "③ 输出经门控后的平滑度得分。"
        ),
        "dsl": "Python：CV_short/CV_long × style_gate",
    },
    "factor_drawdown_depth_duration_range_vol": {
        "theme": "回撤深度 × 回撤天数 × 振幅",
        "formula": (
            "① 相对 20 日最高价的回撤深度 = 1 − (最高−现价)/最高；"
            "② 过去 20 日里价格低于 20 日最高的天数占比（回撤持续度）；"
            "③ 两项相乘后再截面 rank；数值高表示浅回撤、停留时间短。"
        ),
        "dsl": "rank((1-(ts_max(close,20)-close)/ts_max(close,20))*(1-ts_sum(close<ts_max,20)/20))",
    },
    "factor_asym_vol_cont_gate_30": {
        "theme": "下跌/上涨波动不对称 × 连续量比",
        "formula": (
            "① 只取下跌日收益率算 60 日标准差（下跌波动）；"
            "② 只取上涨日收益率算 60 日标准差（上涨波动）；"
            "③ log(下跌波动/上涨波动) 刻画“跌时更乱还是涨时更乱”；"
            "④ 乘以 成交量/20 日均量，再 zscore 标准化。"
        ),
        "dsl": "zscore(log(safe_div(ts_std(下跌日收益,60), ts_std(上涨日收益,60)))*volume/ts_mean(volume,20))",
    },
    "factor_asym_ratio_pure": {
        "theme": "涨跌日振幅不对称（纯振幅比）",
        "formula": (
            "① 下跌日：把 high−low 振幅累加，除以下跌天数 → 下跌日平均振幅；"
            "② 上涨日：同样算法 → 上涨日平均振幅；"
            "③ log(下跌平均振幅/上涨平均振幅)，再截面 rank。"
        ),
        "dsl": "rank(log(下跌日振幅均值/上涨日振幅均值))",
    },
    "factor_asym_vol_volume_cont_30": {
        "theme": "波动不对称 × 成交量连续比（Python 版）",
        "formula": (
            "① 分别算下跌日、上涨日 30 日滚动波动率；"
            "② log(下跌波动/上涨波动)；"
            "③ 乘以 min(成交量/20 日均量, 3) 限制极端值。"
        ),
        "dsl": "Python：log(down_vol/up_vol) × clip(vol/MA20, 0, 3)",
    },
    "factor_drawdown_vol_persistence": {
        "theme": "回撤幅度 ÷ 波动率（持久性）",
        "formula": (
            "① 相对 60 日最高价的回撤比例；"
            "② 除以 20 日收益率标准差（波动率），做风险调整；"
            "③ 再与 ATR 相关项相乘，强调“深回撤+高波动”状态。"
        ),
        "dsl": "local_dsl：close/ts_max(close,60)-1 经 ts_std 缩放 × ATR 项",
    },
    "factor_vol_gated_momentum_v2": {
        "theme": "成交量门控动量",
        "formula": (
            "① 算 20 日价格动量（收益率）；"
            "② 仅当成交量高于 20 日均量的一定倍数时保留动量，否则削弱；"
            "③ 用振幅/波动过滤噪声。"
        ),
        "dsl": "Python：momentum × volume_gate × range_filter",
    },
    "factor_range_asym_log_vol": {
        "theme": "涨跌日振幅不对称 × 成交量（长窗口）",
        "formula": (
            "① 上涨日累计振幅 ÷ 上涨天数，下跌日同理；"
            "② 取 log(上涨平均/下跌平均)；"
            "③ 再乘以 log(成交量/20 日均量) 加权。"
        ),
        "dsl": "Python：log(up_range/down_range) × log(vol/MA20)",
    },
    "factor_resvol_volume_momentum": {
        "theme": "价格动量 × 相对成交量",
        "formula": (
            "① ROC(close, 20) = close/20日前close − 1，即 20 日涨跌幅；"
            "② 乘以 成交量/EMA(成交量,20)；"
            "③ 放量上涨或放量下跌都会放大信号。"
        ),
        "dsl": "ROC(close, 20) * safe_div(volume, ema(volume, 20))",
    },
    "factor_volume_price_divergence_cross_mut": {
        "theme": "量价背离（短窗口）",
        "formula": (
            "① 5 日价格变化 = close − 5日前close；"
            "② 乘以 成交量/20 日EMA成交量；"
            "③ 取负号：价涨量缩或价跌量增时信号强；"
            "④ tanh 压到 (−1,1) 防止极端值。"
        ),
        "dsl": "tanh(-(close-delay(close,5))*(volume/ema(volume,20)))",
    },
    "factor_range_asym_log_short": {
        "theme": "涨跌日振幅不对称（短窗口 log 比）",
        "formula": (
            "① 判断当日收阳/收阴（close 与昨收比较）；"
            "② 10 日内：上涨日振幅之和/上涨天数 vs 下跌日同理；"
            "③ log(上涨平均振幅/下跌平均振幅)，窗口 10 日、最少 5 日有效。"
        ),
        "dsl": "Python：10日涨跌日分别求 (high-low) 均值，再取 log 比",
    },
}

OPERATORS_ZH = [
    ("ts_pct(x, n)", "n 日收益率", "x 相对 n 天前 x 的涨跌幅，即 x/delay(x,n) − 1"),
    ("ts_mean(x, n)", "n 日滑动平均", "过去 n 个交易日 x 的算术平均值"),
    ("ema(x, n)", "n 日指数加权平均", "过去 n 日 x 的指数加权均值，越近的日期权重越大"),
    ("ts_std(x, n)", "n 日标准差", "过去 n 日 x 的样本标准差，衡量波动大小"),
    ("ts_delta(x, n)", "n 日差分", "x 减去 n 天前的 x"),
    ("delay(x, n)", "滞后 n 日", "取 n 个交易日之前的 x 值"),
    ("ts_max / ts_sum", "滚动最大/求和", "过去 n 日内 x 的最大值或累加和"),
    ("rank(x)", "截面排序", "同一交易日全市场股票对 x 排序，映射到 0~1 分位"),
    ("zscore(x)", "标准化", "(x − 均值) / 标准差，使不同股票可比"),
    ("safe_div(a, b)", "安全除法", "a÷b；分母为 0 或缺失时返回空值，避免除零"),
    ("where(条件, a, b)", "条件选择", "条件成立取 a，否则取 b（类似 if-else）"),
    ("log / abs / tanh", "数学函数", "自然对数、绝对值、双曲正切压缩（把大数压到 −1~1）"),
    ("cap(x, ε)", "下限截断", "把 x 限制在不小于 ε，防止 log(0)"),
    ("ATR(h,l,c,n)", "平均真实波幅", "n 日真实波幅的滑动平均，衡量价格波动幅度"),
    ("ROC(x, n)", "n 日变化率", "x 相对 n 天前的涨跌幅，同 ts_pct"),
]


def _esc(text: str) -> str:
    return html_lib.escape(text, quote=True)


def _load_rows(work_dir: Path) -> dict[str, dict[str, Any]]:
    prog = json.loads((work_dir / "production_progress.json").read_text(encoding="utf-8"))
    return {r["factor_name"]: r for r in prog.get("index_rows", [])}


def render_week3_summary(
    *,
    work_dir: Path | None = None,
    out_name: str = "week3_volume_price_summary.html",
) -> Path:
    work_dir = work_dir or DEFAULT_WORK_DIR
    by_name = _load_rows(work_dir)
    rows = [by_name[n] for n in WEEK3_FACTORS if n in by_name]
    missing = [n for n in WEEK3_FACTORS if n not in by_name]
    if missing:
        raise KeyError(f"Missing factors in progress: {missing}")

    ranked = sorted(rows, key=lambda r: _safe_float(r.get("mean_rank_ic")), reverse=True)
    ic_values = [_safe_float(r.get("mean_rank_ic")) for r in rows]
    ic_gt_003 = sum(1 for v in ic_values if v > 0.03)
    avg_rank_ic = sum(ic_values) / len(ic_values) if ic_values else 0.0
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    table_rows: list[str] = []
    for i, item in enumerate(ranked, 1):
        _, path_label, _, _ = resolve_compute_path(
            engine=str(item.get("engine") or ""),
            eval_route=str(item.get("eval_route") or ""),
            eval_mode=str(item.get("eval_mode") or ""),
        )
        topk = _enrich_topk_summary(
            {
                "total_return": item.get("backtest_total_ret"),
                "annualized_return": item.get("backtest_ann_ret"),
                "max_drawdown": item.get("backtest_max_drawdown"),
                "sharpe": item.get("backtest_sharpe"),
                "trading_days": item.get("backtest_trading_days"),
            }
        )
        fname = item.get("factor_name", "")
        zh = FACTOR_ZH.get(fname, {})
        theme = zh.get("theme", "")
        label = f"{fname}" + (f"（{theme}）" if theme else "")
        table_rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f'<td><a href="{item.get("report", "#")}">{_esc(label)}</a></td>'
            f"<td>{_esc(path_label)}</td>"
            f'<td>{_esc(_sanitize_eval_mode_label(str(item.get("eval_mode", ""))))}</td>'
            f'<td>{_safe_float(item.get("mean_rank_ic")):.4f}</td>'
            f'<td>{_safe_float(item.get("rank_icir")):.4f}</td>'
            f'<td>{_safe_float(item.get("rank_ic_positive_ratio")):.2%}</td>'
            f'<td>{_safe_float(item.get("long_short_sharpe")):.3f}</td>'
            f'<td>{_safe_float(topk.get("sharpe")):.3f}</td>'
            f'<td>{_safe_float(topk.get("total_return")):.2%}</td>'
            f'<td>{_safe_float(topk.get("max_drawdown")):.2%}</td>'
            f'<td>{_safe_float(topk.get("annualized_return")):.2%}</td>'
            "</tr>"
        )

    op_rows = "".join(
        f"<tr><td><code>{_esc(name)}</code></td><td>{_esc(short)}</td><td>{_esc(detail)}</td></tr>"
        for name, short, detail in OPERATORS_ZH
    )

    factor_cards: list[str] = []
    for i, item in enumerate(ranked, 1):
        fname = item["factor_name"]
        zh = FACTOR_ZH[fname]
        m_ic = _safe_float(item.get("mean_rank_ic"))
        m_icir = _safe_float(item.get("rank_icir"))
        ls = _safe_float(item.get("long_short_sharpe"))
        topk = _enrich_topk_summary(
            {
                "sharpe": item.get("backtest_sharpe"),
                "total_return": item.get("backtest_total_ret"),
            }
        )
        tk = _safe_float(topk.get("sharpe"))
        read = []
        if m_ic >= 0.05:
            read.append("RankIC 处于本批前列，因子值与次日收益的相关性较强。")
        elif m_ic >= 0.035:
            read.append("RankIC 中等偏上，具备一定选股区分度。")
        else:
            read.append("RankIC 略低但仍为正，更多作为辅助或组合因子。")
        if ls >= 2:
            read.append("多空夏普较高，说明最高组与最低组收益差稳定。")
        elif ls < 0.8:
            read.append("多空夏普偏低，分层两端差异不够稳定，需结合 TopK 一起看。")
        if tk >= 1.5:
            read.append("TopK 夏普突出，按因子值选前 10% 股票、次日开盘买入的回测表现较好。")
        elif tk < 0.5:
            read.append("TopK 夏普一般，实盘可降权或与其他因子组合使用。")
        factor_cards.append(
            f'<article class="fcard" id="{_esc(fname)}">'
            f'<h3>{i}. {_esc(fname)}</h3>'
            f'<p class="ftheme"><b>主题：</b>{_esc(zh["theme"])}</p>'
            f'<p><b>怎么算：</b>{_esc(zh["formula"])}</p>'
            f'<p class="fdsl"><b>公式：</b><code>{_esc(zh["dsl"])}</code></p>'
            f'<p class="fread"><b>结果怎么读：</b>{"".join(read)}</p>'
            f'<p class="flink"><a href="{_esc(item.get("report", "#"))}">查看完整图表 →</a></p>'
            "</article>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Week3 量价因子汇报 - 汇报人孙海崴</title>
  <style>
    :root {{ --fg:#18202a; --muted:#5d6878; --line:#d9dee7; --bg:#f6f8fb; --panel:#fff; --blue:#255f9e; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",Arial,sans-serif; color:var(--fg); background:var(--bg); line-height:1.65; }}
    header {{ padding:36px 48px 24px; background:#fff; border-bottom:1px solid var(--line); }}
    main {{ max-width:1400px; margin:0 auto; padding:24px; }}
    h1 {{ margin:0 0 8px; font-size:28px; }}
    h2 {{ margin:32px 0 14px; font-size:20px; border-left:4px solid var(--blue); padding-left:10px; }}
    h3 {{ margin:0 0 8px; font-size:16px; }}
    .muted {{ color:var(--muted); }}
    .cards {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; margin:20px 0; }}
    .metric {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; }}
    .metric b {{ display:block; font-size:26px; color:var(--blue); }}
    .metric span {{ color:var(--muted); font-size:13px; }}
    .notice {{ background:#eef6ff; border:1px solid #c9def5; padding:14px 16px; border-radius:8px; margin:16px 0; }}
    .notice-warn {{ background:#fffbeb; border:1px solid #f0e0a8; }}
    .explain {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px 18px; margin:12px 0; }}
    .explain ul {{ margin:8px 0 0 20px; padding:0; }}
    .explain li {{ margin:6px 0; }}
    table {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); font-size:13px; }}
    th,td {{ padding:8px 10px; border-bottom:1px solid #edf0f5; text-align:right; vertical-align:top; }}
    th:first-child,td:first-child, th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
    th {{ background:#f0f3f8; position:sticky; top:0; }}
    tr:hover td {{ background:#fbfcff; }}
    a {{ color:#165a9f; text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
    code {{ font-size:12px; background:#f4f6fa; padding:2px 5px; border-radius:4px; word-break:break-all; }}
    .fcard {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px 18px; margin:12px 0; }}
    .ftheme {{ color:var(--blue); margin:0 0 8px; }}
    .fdsl {{ font-size:13px; color:var(--muted); }}
    .fread {{ background:#f8fafc; padding:10px 12px; border-radius:6px; margin:10px 0; font-size:14px; }}
    .flink {{ margin:8px 0 0; font-size:13px; }}
    .toc {{ columns:2; column-gap:24px; font-size:14px; }}
    .toc a {{ display:block; margin:4px 0; }}
    @media (max-width:900px) {{ .cards {{ grid-template-columns:1fr 1fr; }} header {{ padding:20px; }} .toc {{ columns:1; }} }}
  </style>
</head>
<body>
<header>
  <h1>Week3 量价因子汇报 - 汇报人孙海崴</h1>
  <p class="muted">样本区间 2019-01-01 ~ 2026-06-30 · 共 20 个量价类因子 · 本批 Mean RankIC 算术平均 {avg_rank_ic:.4f} · 生成时间 {generated}</p>
</header>
<main>

  <section>
    <div class="cards">
      <div class="metric"><b>20</b><span>本页因子数</span></div>
      <div class="metric"><b>20</b><span>已完成评估</span></div>
      <div class="metric"><b>{ic_gt_003}</b><span>Mean RankIC &gt; 0.03</span></div>
      <div class="metric"><b>{avg_rank_ic:.4f}</b><span>本批 Mean RankIC 均值</span></div>
      <div class="metric"><b>0</b><span>失败/跳过</span></div>
    </div>
  </section>

  <section>
    <h2>一、表格里的指标是什么意思</h2>
    <div class="explain">
      <p>所有 IC 类指标都遵循同一对齐：<b>T 日收盘后得到因子值</b>，与 <b>T+1 日收益率</b>做相关。不使用未来数据。</p>
      <ul>
        <li><b>Mean RankIC</b>：每个交易日，把全市场股票的因子值做<b>截面排序</b>（名次），再与次日收益率的<b>Spearman 相关系数</b>；最后对所有交易日取平均。数值越大，说明“因子值高的股票，次日更容易涨”。本批大多在 0.03~0.06，属于可用区间。</li>
        <li><b>RankICIR</b>：Mean RankIC ÷ RankIC 的日度标准差，类似“信息比率”。越大说明 RankIC 不仅高，而且<b>每天比较稳定</b>，不是偶尔几天碰运气。</li>
        <li><b>RankIC 胜率</b>：RankIC &gt; 0 的交易日占比。例如 63% 表示约三分之二的日子里因子与次日收益正相关。</li>
        <li><b>LS Sharpe（多空夏普）</b>：按因子值把股票分成 10 组，<b>做多最高组、做空最低组</b>，用两组收益差算日度序列，再年化除以波动。这是本地 DuckDB 面板上的研究模拟，<b>不含交易成本</b>，主要看因子分层是否单调。</li>
        <li><b>TopK Sharpe / 累计 / 回撤 / 年化</b>：在 LQTP 平台回测。每个交易日选因子值最高的前 <b>10% 股票（最多 50 只）</b>，<b>次日开盘价等权买入</b>，含平台撮合与费用假设。累计收益、最大回撤、年化收益均来自该 TopK 组合净值曲线。</li>
      </ul>
    </div>
    <div class="notice notice-warn">
      <p><b>怎么同时看 RankIC 和 TopK？</b></p>
      <p>RankIC 衡量“排序对不对”；TopK 衡量“只买头部一组、按开盘成交”的真实组合表现。两者可能不一致：例如多空夏普高但 TopK 一般，可能是中间分组也有贡献；TopK 高但 RankIC 一般，可能是头部极端值驱动。汇报时建议两个都看。</p>
    </div>
  </section>

  <section>
    <h2>二、因子值是怎么算出来的（三条路径）</h2>
    <div class="explain">
      <ul>
        <li><b>LQTP 原生 DSL 直算</b>：公式直接提交 LQTP RunFactor，在平台用同一套算子（ts_mean、ema、rank 等）线上落值。本批中 persistence、vol_regime、drawdown、asym 等 6 个因子走此路径。</li>
        <li><b>自研 factor_engine DSL 落值</b>：Python 公式先转成 DSL，在本地 factor_engine + COS 行情上计算。算子名与 LQTP 对齐（如 safe_div、ema），但执行在本地。本批 6 个因子。</li>
        <li><b>纯 Python 落值</b>：公式含平台暂不支持的逻辑（如 talib、复杂门控），直接跑 Python 函数出因子值。本批 8 个因子。</li>
      </ul>
      <p>无论哪条路径，<b>评估口径统一</b>：因子面板写入 DuckDB 后，RankIC / 十组分层 / 多空在本地面板重算；TopK 另走 LQTP OPEN 回测。</p>
    </div>
  </section>

  <section>
    <h2>三、公式里常见算子（怎么算）</h2>
    <table>
      <thead><tr><th>算子</th><th>一句话</th><th>具体算法</th></tr></thead>
      <tbody>{op_rows}</tbody>
    </table>
    <p class="muted" style="margin-top:10px">说明：所有 ts_* 类算子都是<b>按单只股票自身的历史序列</b>滚动计算，不会用到未来日期；rank / zscore 是<b>同一交易日全市场横截面</b>运算。</p>
  </section>

  <section>
    <h2>四、因子排行（按 Mean RankIC 降序）</h2>
    <table>
      <thead>
        <tr>
          <th>#</th><th>因子</th><th>算值路径</th><th>评估模式</th>
          <th>Mean RankIC</th><th>RankICIR</th><th>RankIC 胜率</th>
          <th>LS Sharpe</th><th>TopK Sharpe</th><th>TopK 累计</th><th>TopK 最大回撤</th><th>TopK 年化</th>
        </tr>
      </thead>
      <tbody>{"".join(table_rows)}</tbody>
    </table>
  </section>

  <section>
    <h2>五、20 个因子逐一解读</h2>
    <p class="muted">按 RankIC 从高到低排列。每个因子说明：经济含义、计算步骤（白话）、原始公式、对本页数字的简要读法。</p>
    <nav class="toc muted">
      {"".join(f'<a href="#{_esc(r["factor_name"])}">{i}. {_esc(FACTOR_ZH[r["factor_name"]]["theme"])}</a>' for i, r in enumerate(ranked, 1))}
    </nav>
    {"".join(factor_cards)}
  </section>

</main>
</body>
</html>"""

    out_path = work_dir / "reports" / out_name
    out_path.write_text(html, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    path = render_week3_summary()
    print(f"Wrote {path}")
