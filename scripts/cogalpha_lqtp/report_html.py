#!/usr/bin/env python3
"""Generate HTML evaluation report from factor_engine values + LQTP metrics."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FACTOR_ANALYSIS_FIELDS = [
    "mean_ic",
    "std_ic",
    "icir",
    "ic_positive_ratio",
    "coverage",
    "long_short_return",
    "long_short_sharpe",
    "daily_ic",
    "daily_ls_returns",
    "trade_dates",
    "quote_times",
    "sample_counts",
    "coverages",
    "group_mean_returns",
    "group_pnls",
]

BACKTEST_RESULT_FIELDS = [
    "trade_date",
    "bod_net_asset",
    "eod_net_asset",
    "bod_market_value",
    "eod_market_value",
    "bod_long_market_value",
    "eod_long_market_value",
    "bod_short_market_value",
    "eod_short_market_value",
    "bod_cash",
    "eod_cash",
    "expect_buy_amount",
    "expect_sell_amount",
    "buy_amount",
    "sell_amount",
    "buy_execution_ratio",
    "sell_execution_ratio",
    "execution_ratio",
    "turnover_rate",
    "leverage",
    "commission",
    "net_asset_diff",
    "ret",
]


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _plot_daily_ic(payload: dict[str, Any]) -> str:
    dates = payload.get("trade_dates", [])
    ic = payload.get("daily_ic", [])
    if not dates or not ic:
        return ""
    frame = pd.DataFrame({"trade_date": dates, "daily_ic": ic})
    frame["dt"] = pd.to_datetime(frame["trade_date"].astype(str))
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(frame["dt"], frame["daily_ic"], linewidth=1.2)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_title("Daily Rank IC")
    ax.set_ylabel("IC")
    return _fig_to_base64(fig)


def _plot_cumulative_ls(payload: dict[str, Any]) -> str:
    dates = payload.get("trade_dates", [])
    ls = payload.get("daily_ls_returns", [])
    if not dates or not ls:
        return ""
    frame = pd.DataFrame({"trade_date": dates, "ls": ls})
    frame["dt"] = pd.to_datetime(frame["trade_date"].astype(str))
    frame["cum"] = (1.0 + frame["ls"]).cumprod() - 1.0
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(frame["dt"], frame["cum"], linewidth=1.2)
    ax.set_title("Cumulative Long-Short Return")
    ax.set_ylabel("Cum Return")
    return _fig_to_base64(fig)


def _plot_group_returns(payload: dict[str, Any]) -> str:
    groups = payload.get("group_pnls", [])
    if not groups:
        return ""
    fig, ax = plt.subplots(figsize=(8, 3))
    labels = []
    finals = []
    for item in groups:
        pnl = item.get("pnl", [])
        if not pnl:
            continue
        labels.append(f"G{item['group']}")
        finals.append(float(pd.Series(pnl).iloc[-1]))
    if not labels:
        return ""
    ax.bar(labels, finals)
    ax.set_title("Group PnL (End of Sample)")
    ax.set_ylabel("PnL")
    return _fig_to_base64(fig)


def _plot_backtest_nav(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    frame = pd.DataFrame(rows)
    frame["dt"] = pd.to_datetime(frame["trade_date"].astype(str))
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(frame["dt"], frame["eod_net_asset"], linewidth=1.2)
    ax.set_title("Backtest NAV")
    ax.set_ylabel("Net Asset")
    return _fig_to_base64(fig)


def _plot_coverage(payload: dict[str, Any]) -> str:
    dates = payload.get("trade_dates", [])
    cov = payload.get("coverages", [])
    if not dates or not cov:
        return ""
    frame = pd.DataFrame({"trade_date": dates, "coverage": cov})
    frame["dt"] = pd.to_datetime(frame["trade_date"].astype(str))
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(frame["dt"], frame["coverage"], linewidth=1.2)
    ax.set_title("Daily Coverage")
    ax.set_ylabel("Coverage")
    return _fig_to_base64(fig)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if abs(value) >= 1000 or (abs(value) < 1e-4 and value != 0):
            return f"{value:.6g}"
        return f"{value:.6f}"
    return str(value)


def _metrics_table(fields: list[str], data: dict[str, Any], title: str) -> str:
    rows = "".join(
        f"<tr><th>{name}</th><td>{_fmt(data.get(name))}</td></tr>"
        for name in fields
        if not isinstance(data.get(name), (list, dict))
    )
    return f"<h3>{title}</h3><table class='kv'>{rows}</table>"


def _series_table(payload: dict[str, Any]) -> str:
    dates = payload.get("trade_dates", [])
    if not dates:
        return ""
    frame = pd.DataFrame({"trade_date": dates})
    for col in ("daily_ic", "daily_ls_returns", "sample_counts", "coverages", "quote_times"):
        if col in payload and len(payload[col]) == len(dates):
            frame[col] = payload[col]
    return frame.to_html(index=False, float_format=lambda x: f"{x:.6f}")


def _backtest_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>无回测数据</p>"
    frame = pd.DataFrame(rows)
    cols = [c for c in BACKTEST_RESULT_FIELDS if c in frame.columns]
    return frame[cols].to_html(index=False, float_format=lambda x: f"{x:.6f}")


def render_factor_report(
    *,
    factor_name: str,
    dsl: str,
    analysis: dict[str, Any],
    backtest_rows: list[dict[str, Any]] | None,
    out_path: Path,
    eval_mode: str = "",
    materialize_meta: dict[str, Any] | None = None,
) -> None:
    ic_img = _plot_daily_ic(analysis)
    ls_img = _plot_cumulative_ls(analysis)
    group_img = _plot_group_returns(analysis)
    nav_img = _plot_backtest_nav(backtest_rows or [])
    cov_img = _plot_coverage(analysis)

    def img_tag(data: str, title: str) -> str:
        if not data:
            return ""
        return f'<figure><img alt="{title}" src="data:image/png;base64,{data}"/></figure>'

    summary_fields = [
        "mean_ic",
        "std_ic",
        "icir",
        "ic_positive_ratio",
        "coverage",
        "long_short_return",
        "long_short_sharpe",
    ]
    summary_cards = "".join(
        f'<div class="metric"><b>{_fmt(analysis.get(k))}</b><span>{k}</span></div>'
        for k in summary_fields
    )

    meta_block = ""
    if materialize_meta:
        meta_block = f"""
  <div class="card">
    <h2>本地落值 (factor_engine)</h2>
    <pre>{json.dumps(materialize_meta, ensure_ascii=False, indent=2)}</pre>
  </div>"""

    mode_note = {
        "analyze_factor_upload": "官方 AnalyzeFactor：上传 factor_engine 落值，由 LQTP 平台计算全部指标。",
        "local_values_lqtp_returns": "AnalyzeFactor 暂未开通；使用 factor_engine 落值 + LQTP 行情收益本地复算（不调用 LQTP RunFactor 重算因子）。",
    }.get(eval_mode, eval_mode or "unknown")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>{factor_name} · factor_engine → LQTP</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2937; max-width: 1200px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin: 16px 0; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .metric b {{ display: block; font-size: 18px; }}
    pre {{ background: #f8fafc; padding: 12px; overflow-x: auto; font-size: 13px; }}
    .charts {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
    img {{ max-width: 100%; border: 1px solid #e5e7eb; border-radius: 6px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }}
    table.kv th {{ width: 220px; background: #f8fafc; }}
    .note {{ color: #4b5563; font-size: 14px; }}
  </style>
</head>
<body>
  <h1>{factor_name}</h1>
  <p class="note">流程：factor_engine DSL 本地落值 → 上传 LQTP 评估 → 生成报告</p>
  <div class="card">
    <h2>Factor Engine DSL</h2>
    <pre>{dsl}</pre>
  </div>
  {meta_block}
  <div class="card">
    <h2>因子分析 (FactorAnalysis)</h2>
    <p class="note">评估模式：<b>{eval_mode or "unknown"}</b> — {mode_note}</p>
    <div class="metrics">{summary_cards}</div>
    {_metrics_table(FACTOR_ANALYSIS_FIELDS, analysis, "全部标量指标")}
    <div class="charts">
      {img_tag(ic_img, "Daily IC")}
      {img_tag(ls_img, "Cumulative LS")}
      {img_tag(group_img, "Group PnL")}
      {img_tag(cov_img, "Coverage")}
    </div>
    <h3>每日序列</h3>
    {_series_table(analysis)}
  </div>
  <div class="card">
    <h2>回测 (Backtest Result)</h2>
    <p class="note">权重来自 factor_engine 落值 top 分位，由 LQTP BacktestService 执行。</p>
    {_backtest_table(backtest_rows or [])}
    {img_tag(nav_img, "Backtest NAV")}
  </div>
  <div class="card">
    <h2>Raw Analysis JSON</h2>
    <pre>{json.dumps(analysis, ensure_ascii=False, indent=2)}</pre>
  </div>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def render_index(reports: list[dict[str, str]], out_path: Path) -> None:
    rows = "\n".join(
        f'<li><a href="{item["report"]}">{item["factor_name"]}</a> '
        f'· mode={item.get("eval_mode", "?")} '
        f'· IC={item.get("mean_ic", "n/a")} · ICIR={item.get("icir", "n/a")}</li>'
        for item in reports
    )
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/><title>CogAlpha factor_engine → LQTP</title></head>
<body>
  <h1>CogAlpha → factor_engine 落值 → LQTP 评估</h1>
  <p>算子命名与计算均使用 factor_engine；LQTP 仅负责 IC/分组/回测等评估。</p>
  <ul>{rows}</ul>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")
