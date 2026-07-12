#!/usr/bin/env python3
"""Generate HTML evaluation report: RankIC / groups / LS / TopK backtest."""
from __future__ import annotations

import base64
import html as html_lib
import io
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.cogalpha_lqtp.lqtp_client import summarize_backtest

DEFAULT_WORK_DIR = Path(__file__).resolve().parents[2] / "data/cogalpha_lqtp_production"


def _load_annotation(work_dir: Path | None, factor_name: str, annotation: dict[str, Any] | None) -> dict[str, Any]:
    if annotation is not None:
        return annotation
    if work_dir is None:
        work_dir = DEFAULT_WORK_DIR
    try:
        from scripts.cogalpha_lqtp.factor_annotations import load_factor_annotation

        return load_factor_annotation(work_dir, factor_name)
    except Exception:  # noqa: BLE001
        return {}


def _lookahead_badge(risk: str) -> str:
    colors = {
        "low": ("#dcfce7", "#166534", "低"),
        "medium": ("#fef9c3", "#854d0e", "中"),
        "high": ("#fee2e2", "#991b1b", "高"),
        "pending_fix": ("#fce7f3", "#9d174d", "待修复"),
    }
    bg, fg, label = colors.get(risk, ("#f3f4f6", "#374151", risk or "?"))
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:4px;'
        f'font-size:12px;font-weight:600">未来函数风险·{label}</span>'
    )


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if value != value:
            return "nan"
        if abs(value) >= 1000 or (abs(value) < 1e-4 and value != 0):
            return f"{value:.6g}"
        return f"{value:.6f}"
    return str(value)


def _fmt_pct(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if v != v:
        return "nan"
    return f"{v:.2%}"


def _rankic_chart_title(payload: dict[str, Any]) -> str:
    kind = payload.get("return_kind") or ""
    if kind == "close_to_close_T_plus_1":
        return "Daily RankIC (factor T → close(T+1)/close(T)-1)"
    if kind == "open_to_open_T_plus_2":
        return "Daily RankIC (factor T → open(T+1)→open(T+2))"
    return "Daily RankIC"


def _ls_chart_title(payload: dict[str, Any]) -> str:
    kind = payload.get("return_kind") or ""
    if kind == "close_to_close_T_plus_1":
        return "Cumulative Long−Short (G10−G1, close-to-close, net costs)"
    if kind == "open_to_open_T_plus_2":
        return "Cumulative Long−Short (G10−G1, open-to-open, net costs)"
    return "Cumulative Long−Short (G10−G1, net costs)"


def _plot_daily_rank_ic(payload: dict[str, Any]) -> str:
    dates = payload.get("trade_dates", [])
    ic = payload.get("daily_rank_ic") or payload.get("daily_ic", [])
    if not dates or not ic:
        return ""
    frame = pd.DataFrame({"trade_date": dates, "rank_ic": ic})
    frame["dt"] = pd.to_datetime(frame["trade_date"].astype(str))
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(frame["dt"], frame["rank_ic"], linewidth=1.0, color="#255f9e")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_title(_rankic_chart_title(payload))
    ax.set_ylabel("RankIC")
    return _fig_to_base64(fig)


def _plot_cumulative_ls(payload: dict[str, Any]) -> str:
    dates = payload.get("trade_dates", [])
    ls = payload.get("daily_ls_returns", [])
    if not dates or not ls:
        return ""
    frame = pd.DataFrame({"trade_date": dates, "ls": ls})
    frame["dt"] = pd.to_datetime(frame["trade_date"].astype(str))
    # Net G10−G1 daily return after turnover costs; cumprod is portfolio-style
    frame["cum"] = (1.0 + frame["ls"].clip(-0.5, 0.5)).cumprod() - 1.0
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(frame["dt"], frame["cum"], linewidth=1.2, color="#1b7a4b")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_title(_ls_chart_title(payload))
    ax.set_ylabel("Cumulative Return")
    return _fig_to_base64(fig)


def _plot_group_returns(payload: dict[str, Any]) -> str:
    groups = payload.get("group_pnls", [])
    if not groups:
        return ""
    fig, ax = plt.subplots(figsize=(8, 3.2))
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
    colors = ["#c0392b" if i == 0 else "#27ae60" if i == len(labels) - 1 else "#5d6d7e" for i in range(len(labels))]
    ax.bar(labels, finals, color=colors)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_title("Decile Cumulative Return (G1=low factor … G10=high factor)")
    ax.set_ylabel("Cum Return")
    return _fig_to_base64(fig)


def _plot_backtest_nav(rows: list[dict[str, Any]], title: str = "TopK Backtest NAV (OPEN)") -> str:
    if not rows:
        return ""
    frame = pd.DataFrame(rows).copy()
    frame["dt"] = pd.to_datetime(frame["trade_date"].astype(str))
    nav = pd.to_numeric(frame["eod_net_asset"], errors="coerce")
    # Normalize to 1.0 so the chart is readable (no matplotlib 1e6 offset).
    if "bod_net_asset" in frame.columns and pd.notna(frame["bod_net_asset"].iloc[0]):
        start = float(frame["bod_net_asset"].iloc[0])
    else:
        start = float(nav.iloc[0]) if len(nav) else float("nan")
    if start and start == start and start != 0:
        nav = nav / start
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(frame["dt"], nav, linewidth=1.2, color="#255f9e")
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("NAV (start=1)")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    return _fig_to_base64(fig)


def _metrics_cards(items: list[tuple[str, Any]]) -> str:
    parts = []
    for label, value in items:
        parts.append(
            f'<div class="metric"><b>{_fmt(value)}</b><span>{label}</span></div>'
        )
    return "".join(parts)


def _kv_table(rows: list[tuple[str, str]], title: str) -> str:
    body = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<h3>{title}</h3><table class='kv'><tbody>{body}</tbody></table>"


def _series_table_collapsed(payload: dict[str, Any]) -> str:
    dates = payload.get("trade_dates", [])
    if not dates:
        return ""
    frame = pd.DataFrame(
        {
            "trade_date": dates,
            "daily_rank_ic": payload.get("daily_rank_ic") or payload.get("daily_ic", []),
            "daily_ls_net": payload.get("daily_ls_returns", []),
            "sample_counts": payload.get("sample_counts", []),
        }
    )
    html = frame.to_html(index=False, float_format=lambda x: f"{x:.6f}")
    return f"<details><summary>每日序列（默认折叠）</summary>{html}</details>"


def render_factor_report(
    *,
    factor_name: str,
    dsl: str,
    analysis: dict[str, Any],
    backtest_rows: list[dict[str, Any]] | None,
    out_path: Path,
    eval_mode: str = "",
    materialize_meta: dict[str, Any] | None = None,
    ls_backtest_rows: list[dict[str, Any]] | None = None,
    topk_summary: dict[str, Any] | None = None,
    ls_summary: dict[str, Any] | None = None,
    python_code: str = "",
    work_dir: Path | None = None,
    annotation: dict[str, Any] | None = None,
) -> None:
    ann = _load_annotation(work_dir, factor_name, annotation)
    ic_img = _plot_daily_rank_ic(analysis)
    ls_img = _plot_cumulative_ls(analysis)
    group_img = _plot_group_returns(analysis)
    topk_rows = backtest_rows or []
    topk_img = _plot_backtest_nav(topk_rows, "TopK Long-Only Backtest NAV (LQTP OPEN)")
    ls_bt_rows = ls_backtest_rows or []
    ls_bt_img = _plot_backtest_nav(ls_bt_rows, "Long−Short Backtest NAV (LQTP OPEN, if available)")

    topk_summary = topk_summary or summarize_backtest(topk_rows)
    ls_summary = ls_summary or (summarize_backtest(ls_bt_rows) if ls_bt_rows else {})

    mean_rank_ic = analysis.get("mean_rank_ic", analysis.get("mean_ic"))
    std_rank_ic = analysis.get("std_rank_ic", analysis.get("std_ic"))
    rank_icir = analysis.get("rank_icir", analysis.get("icir"))
    rank_ic_pos = analysis.get("rank_ic_positive_ratio", analysis.get("ic_positive_ratio"))

    rank_cards = _metrics_cards(
        [
            ("Mean RankIC", mean_rank_ic),
            ("Std RankIC", std_rank_ic),
            ("RankICIR", rank_icir),
            ("RankIC 胜率", rank_ic_pos),
            ("LS Cum (G10−G1 net)", analysis.get("long_short_return")),
            ("LS Sharpe (net)", analysis.get("long_short_sharpe")),
        ]
    )

    topk_cards = _metrics_cards(
        [
            ("Total Return", topk_summary.get("total_return")),
            ("Ann. Return", topk_summary.get("annualized_return")),
            ("Max Drawdown", topk_summary.get("max_drawdown")),
            ("Sharpe", topk_summary.get("sharpe")),
            ("Volatility", topk_summary.get("volatility")),
            ("Calmar", topk_summary.get("calmar")),
            ("Win Rate", topk_summary.get("win_rate")),
            ("Avg Turnover", topk_summary.get("avg_turnover")),
            ("Total Commission", topk_summary.get("total_commission")),
            ("Trading Days", topk_summary.get("trading_days")),
        ]
    )

    meta_block = ""
    if materialize_meta:
        title = (
            "本地落值 (factor_engine)"
            if materialize_meta.get("engine") != "lqtp_dsl"
            else "LQTP 平台计算 (RunFactor)"
        )
        meta_block = f"""
  <div class="card">
    <h2>{title}</h2>
    <pre>{json.dumps(materialize_meta, ensure_ascii=False, indent=2)}</pre>
  </div>"""

    mode_note = {
        "analyze_factor_upload": "官方 AnalyzeFactor：上传落值，平台计算指标。",
        "local_values_lqtp_returns": (
            "本地 RankIC / 分层 / 多空：因子 T 日收盘后可得 → 前瞻收益 "
            "close(T+1)/close(T)-1（市面标准 close-to-close RankIC）。"
        ),
        "lqtp_run_factor": "DSL 与 LQTP 一致，RunFactor(analyze=True) 平台评估。",
    }.get(eval_mode, eval_mode or "unknown")

    signal_note = analysis.get("signal_lag_note") or ""
    return_kind = analysis.get("return_kind") or ""
    n_groups = analysis.get("n_groups") or 10
    comm_buy = analysis.get("commission_buy", 0.01)
    comm_sell = analysis.get("commission_sell", 0.01)

    formula_display = ann.get("formula_display") or ""
    formula_note = ann.get("formula_note") or ""
    rationale_zh = ann.get("rationale_zh") or ""
    lookahead_risk = ann.get("lookahead_risk") or ""
    lookahead_judgment = ann.get("lookahead_judgment_zh") or ""

    formula_block = ""
    dsl_clean = (dsl or "").strip()
    has_dsl = bool(dsl_clean and dsl_clean not in {"(python only)"})
    if formula_display or has_dsl or (python_code or "").strip():
        primary = formula_display if formula_display else dsl_clean
        src_note = ""
        if ann.get("formula_source") == "ast_translated":
            src_note = "（Python 自动翻译 DSL，便于阅读）"
        elif ann.get("formula_source") == "formula_text":
            src_note = "（来自 formula_text 文档）"
        elif ann.get("is_python_only") and formula_display:
            src_note = "（Python-only 因子 · 公式化表示）"
        formula_block = f"""
  <div class="card">
    <h2>公式 / 代码</h2>
    {f'<p class="note">{_lookahead_badge(lookahead_risk)} {html_lib.escape(formula_note or src_note)}</p>' if (lookahead_risk or formula_note or src_note) else ""}
    {f'<h3>公式化表示{html_lib.escape(src_note)}</h3><pre>{html_lib.escape(primary.strip())}</pre>' if primary.strip() else ""}
    {f'<h3>DSL（factor_engine）</h3><pre>{html_lib.escape(dsl_clean)}</pre>' if has_dsl and formula_display and formula_display.strip() != dsl_clean.strip() else ""}
    {f'<h3>Python 源码</h3><pre>{html_lib.escape((python_code or "").strip())}</pre>' if (python_code or "").strip() else ""}
    {"" if (primary.strip() or has_dsl or (python_code or "").strip()) else "<pre>（无公式/代码）</pre>"}
  </div>"""

    rationale_block = ""
    if rationale_zh or lookahead_judgment:
        rationale_block = f"""
  <div class="card">
    <h2>因子解读（中文）</h2>
    {f'<pre class="rationale">{html_lib.escape(rationale_zh)}</pre>' if rationale_zh else ""}
    {f'<p class="note"><b>未来函数审查</b>：{html_lib.escape(lookahead_judgment)}</p>' if lookahead_judgment and not rationale_zh else ""}
  </div>"""

    ls_bt_section = ""
    if ls_bt_rows:
        ls_cards = _metrics_cards(
            [
                ("Total Return", ls_summary.get("total_return")),
                ("Ann. Return", ls_summary.get("annualized_return")),
                ("Max Drawdown", ls_summary.get("max_drawdown")),
                ("Sharpe", ls_summary.get("sharpe")),
                ("Win Rate", ls_summary.get("win_rate")),
                ("Avg Turnover", ls_summary.get("avg_turnover")),
            ]
        )
        ls_bt_section = f"""
  <div class="card">
    <h2>平台多空回测（Top/Bottom 分位，LQTP OPEN）</h2>
    <p class="note">信号日 T → 权重落在 T+1，以开盘价成交；启用做空；双边手续费各 0.01%。</p>
    <div class="metrics">{ls_cards}</div>
    <figure>{{''}}</figure>
    {f'<img alt="LS Backtest NAV" src="data:image/png;base64,{ls_bt_img}"/>' if ls_bt_img else ''}
  </div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>{factor_name} · RankIC / 分层 / TopK</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; margin: 24px; color: #1f2937; max-width: 1100px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin: 16px 0; background: #fff; }}
    .metrics {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; }}
    .metric {{ background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 6px; padding: 10px; }}
    .metric b {{ display: block; font-size: 16px; color: #1e3a5f; }}
    .metric span {{ color: #6b7280; font-size: 12px; }}
    pre {{ background: #f8fafc; padding: 12px; overflow-x: auto; font-size: 12px; }}
    img {{ max-width: 100%; border: 1px solid #e5e7eb; border-radius: 6px; margin-top: 8px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }}
    table.kv th {{ width: 280px; background: #f8fafc; }}
    .note {{ color: #4b5563; font-size: 14px; }}
    pre.rationale {{ white-space: pre-wrap; line-height: 1.55; font-size: 13px; }}
    .hl {{ background: #eef6ff; border: 1px solid #c9def5; padding: 10px 12px; border-radius: 6px; }}
    details {{ margin-top: 12px; }}
    @media (max-width: 900px) {{ .metrics {{ grid-template-columns: 1fr 1fr; }} }}
  </style>
</head>
<body>
  <h1>{factor_name}</h1>
  <div class="hl note">
    <b>评估口径</b>：{mode_note}<br/>
    {signal_note}<br/>
    return_kind=<code>{return_kind}</code>
  </div>

  {formula_block or f'''
  <div class="card">
    <h2>公式 / 代码</h2>
    {f'<h3>Python 源码</h3><pre>{html_lib.escape((python_code or "").strip())}</pre>' if (python_code or "").strip() else ""}
    {f'<h3>DSL</h3><pre>{html_lib.escape((dsl or "").strip())}</pre>' if (dsl or "").strip() and (dsl or "").strip() not in {{"(python only)", ""}} else ""}
    {"" if ((python_code or "").strip() or ((dsl or "").strip() and (dsl or "").strip() not in {{"(python only)", ""}})) else "<pre>（无公式/代码）</pre>"}
  </div>'''}
  {rationale_block}
  {meta_block}

  <div class="card">
    <h2>① RankIC</h2>
    <p class="note">Spearman 秩相关；T 日因子 vs 次日 close-to-close 收益（标准 RankIC 口径）。</p>
    <div class="metrics">{rank_cards}</div>
    {f'<img alt="Daily RankIC" src="data:image/png;base64,{ic_img}"/>' if ic_img else ''}
  </div>

  <div class="card">
    <h2>② 分层（十分位 G1…G10）</h2>
    <p class="note">G1=因子最低，G10=因子最高；累计收益为各组 close-to-close 复利净值−1。</p>
    {f'<img alt="Group PnL" src="data:image/png;base64,{group_img}"/>' if group_img else ''}
  </div>

  <div class="card">
    <h2>③ 多空（G10 做多 − G1 做空）</h2>
    <p class="note">
      <b>计算步骤（DuckDB 面板，非 LQTP 平台回测）：</b><br/>
      1) 每个交易日 T，对全市场股票按因子值升序分成 {n_groups} 组（G1 最低 … G10 最高）；<br/>
      2) 各组内股票等权，组收益 = 组内个股 forward 收益均值，forward 收益 = close(T+1)/close(T)−1；<br/>
      3) 当日多空 gross = mean(G10 收益) − mean(G1 收益)；<br/>
      4) 扣费：假设每日单边换手 40%，日成本 = 2 × 40% × (买{comm_buy}% + 卖{comm_sell}%) = {2 * 0.4 * (float(comm_buy) + float(comm_sell)):.4f}% ；net = gross − 日成本；<br/>
      5) LS Sharpe = mean(日 net) / std(日 net) × √252；LS Cum = ∏(1+日 net) − 1。<br/>
      <b>注意</b>：这是研究用十分位多空，不是可交易的 TopK 组合；TopK 见下方 LQTP OPEN 回测。
    </p>
    {f'<img alt="Cumulative LS" src="data:image/png;base64,{ls_img}"/>' if ls_img else ''}
  </div>

  <div class="card">
    <h2>④ TopK 回测（LQTP 平台 · OPEN）</h2>
    <p class="note">
      每天取因子 Top 10%（最多 50 只）等权做多；信号日 T 的权重落到 T+1，
      以开盘价成交；买卖手续费各 0.01%；初始本金 1000 万。
      下图 NAV 已归一化到起点=1（平台原始净值为绝对金额）。
    </p>
    <div class="metrics">{topk_cards}</div>
    {f'<img alt="TopK NAV" src="data:image/png;base64,{topk_img}"/>' if topk_img else '<p class="note">无 TopK 回测结果（可能被跳过）。</p>'}
    {_kv_table(
        [
            ("total_return", _fmt_pct(topk_summary.get("total_return"))),
            ("annualized_return", _fmt_pct(topk_summary.get("annualized_return"))),
            ("max_drawdown", _fmt_pct(topk_summary.get("max_drawdown"))),
            ("sharpe", _fmt(topk_summary.get("sharpe"))),
            ("volatility", _fmt_pct(topk_summary.get("volatility"))),
            ("calmar", _fmt(topk_summary.get("calmar"))),
            ("win_rate", _fmt_pct(topk_summary.get("win_rate"))),
            ("avg_turnover", _fmt_pct(topk_summary.get("avg_turnover"))),
            ("total_commission", _fmt(topk_summary.get("total_commission"))),
            ("final_nav", _fmt(topk_summary.get("final_nav"))),
            ("trading_days", _fmt(topk_summary.get("trading_days"))),
        ],
        "TopK 回测指标明细",
    )}
  </div>

  {ls_bt_section}

  <div class="card">
    <h2>附录</h2>
    {_series_table_collapsed(analysis)}
    <details>
      <summary>Raw Analysis JSON</summary>
      <pre>{json.dumps({k: v for k, v in analysis.items() if k not in {"daily_ic", "daily_rank_ic", "daily_ls_returns", "daily_ls_gross", "group_pnls", "coverages", "sample_counts", "trade_dates", "quote_times"}}, ensure_ascii=False, indent=2)}</pre>
    </details>
  </div>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def load_analysis_from_factor_report(report_path: Path) -> dict[str, Any] | None:
    """Load RankIC/LS payload embedded in an existing factor HTML appendix."""
    if not report_path.is_file():
        return None
    text = report_path.read_text(encoding="utf-8", errors="replace")
    marker = '<summary>Raw Analysis JSON</summary>'
    idx = text.find(marker)
    if idx < 0:
        return None
    start = text.find("<pre>", idx)
    end = text.find("</pre>", start)
    if start < 0 or end < 0:
        return None
    blob = text[start + 5 : end].strip()
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None


def render_index(reports: list[dict[str, str]], out_path: Path) -> None:
    rows = "\n".join(
        f'<li><a href="{item["report"]}">{item["factor_name"]}</a> '
        f'· mode={item.get("eval_mode", "?")} '
        f'· RankIC={item.get("mean_ic", item.get("mean_rank_ic", "n/a"))} '
        f'· RankICIR={item.get("icir", item.get("rank_icir", "n/a"))}</li>'
        for item in reports
    )
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/><title>CogAlpha → LQTP</title></head>
<body>
  <h1>CogAlpha 因子评估索引</h1>
  <p>RankIC / 分层 / 多空 / TopK（开盘成交，无收盘当日成交）。</p>
  <ul>{rows}</ul>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def render_production_summary(
    *,
    rows: list[dict[str, Any]],
    out_path: Path,
    meta: dict[str, Any],
    failed: dict[str, str] | None = None,
) -> None:
    failed = failed or {}
    total = int(meta.get("catalog_total", len(rows) + len(failed)))
    completed = len(rows)
    fail_n = len(failed)

    def _sort_key(item: dict[str, Any]) -> float:
        return _safe_float(item.get("mean_rank_ic", item.get("mean_ic")))

    ranked = sorted(rows, key=_sort_key, reverse=True)
    pos_ic = sum(1 for r in rows if _safe_float(r.get("mean_rank_ic", r.get("mean_ic"))) > 0)
    pos_icir = sum(1 for r in rows if _safe_float(r.get("rank_icir", r.get("icir"))) > 0)

    table_rows = []
    for i, item in enumerate(ranked, 1):
        table_rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f'<td><a href="{item.get("report", "#")}">{item.get("factor_name", "")}</a></td>'
            f'<td>{item.get("eval_mode", "")}</td>'
            f'<td>{_safe_float(item.get("mean_rank_ic", item.get("mean_ic"))):.4f}</td>'
            f'<td>{_safe_float(item.get("rank_icir", item.get("icir"))):.4f}</td>'
            f'<td>{_safe_float(item.get("rank_ic_positive_ratio", item.get("ic_positive_ratio"))):.2%}</td>'
            f'<td>{_safe_float(item.get("long_short_sharpe")):.3f}</td>'
            f'<td>{_safe_float(item.get("backtest_sharpe")):.3f}</td>'
            f'<td>{_safe_float(item.get("backtest_total_ret")):.2%}</td>'
            f'<td>{_safe_float(item.get("backtest_max_drawdown")):.2%}</td>'
            f'<td>{_safe_float(item.get("backtest_ann_ret")):.2%}</td>'
            "</tr>"
        )

    fail_rows = "".join(
        f"<tr><td>{name}</td><td>{msg}</td></tr>" for name, msg in sorted(failed.items())
    )
    date_range = meta.get("date_range", ["", ""])
    generated = meta.get("generated_at", "")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>CogAlpha 因子生产评估汇总 · {date_range[0]} ~ {date_range[1]}</title>
  <style>
    :root {{ --fg:#18202a; --muted:#5d6878; --line:#d9dee7; --bg:#f6f8fb; --panel:#fff; --blue:#255f9e; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",Arial,sans-serif; color:var(--fg); background:var(--bg); line-height:1.55; }}
    header {{ padding:36px 48px 24px; background:#fff; border-bottom:1px solid var(--line); }}
    main {{ max-width:1400px; margin:0 auto; padding:24px; }}
    h1 {{ margin:0 0 8px; font-size:28px; }} h2 {{ margin:28px 0 12px; font-size:20px; }}
    .muted {{ color:var(--muted); }}
    .cards {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; margin:20px 0; }}
    .metric {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; }}
    .metric b {{ display:block; font-size:26px; color:var(--blue); }}
    .metric span {{ color:var(--muted); font-size:13px; }}
    .notice {{ background:#eef6ff; border:1px solid #c9def5; padding:12px 14px; border-radius:8px; margin:16px 0; }}
    table {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); font-size:13px; }}
    th,td {{ padding:8px 10px; border-bottom:1px solid #edf0f5; text-align:right; }}
    th:first-child,td:first-child, th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
    th {{ background:#f0f3f8; position:sticky; top:0; }}
    tr:hover td {{ background:#fbfcff; }}
    a {{ color:#165a9f; text-decoration:none; }}
    .fail td {{ text-align:left; color:#7a1f1f; }}
    @media (max-width:900px) {{ .cards {{ grid-template-columns:1fr 1fr; }} header {{ padding:20px; }} }}
  </style>
</head>
<body>
<header>
  <h1>CogAlpha 因子生产评估汇总</h1>
  <p class="muted">区间 {date_range[0]} ~ {date_range[1]} · RankIC(close) / 分层 / 多空 / TopK(OPEN)</p>
  <p class="muted">生成时间：{generated} · 进度 {completed}/{total} 完成 · {fail_n} 失败</p>
</header>
<main>
  <section>
    <div class="cards">
      <div class="metric"><b>{total}</b><span>目录因子总数</span></div>
      <div class="metric"><b>{completed}</b><span>已完成评估</span></div>
      <div class="metric"><b>{pos_ic}</b><span>Mean RankIC &gt; 0</span></div>
      <div class="metric"><b>{pos_icir}</b><span>RankICIR &gt; 0</span></div>
      <div class="metric"><b>{fail_n}</b><span>失败/跳过</span></div>
    </div>
    <div class="notice">
      <p><b>口径：</b>RankIC / 分层 / 多空用标准 close-to-close（T 日因子 → close(T+1)/close(T)-1）。
      TopK 回测 = LQTP 平台 OPEN 成交（Top10%≤50 只，信号 T → T+1 开盘）。</p>
    </div>
  </section>
  <section>
    <h2>因子排行（按 Mean RankIC 降序）</h2>
    <table>
      <thead>
        <tr>
          <th>#</th><th>因子</th><th>评估模式</th>
          <th>Mean RankIC</th><th>RankICIR</th><th>RankIC 胜率</th>
          <th>LS Sharpe</th><th>TopK Sharpe</th><th>TopK 累计</th><th>TopK 最大回撤</th><th>TopK 年化</th>
        </tr>
      </thead>
      <tbody>
        {''.join(table_rows)}
      </tbody>
    </table>
  </section>
  <section><h2>失败因子</h2><table class='fail'><thead><tr><th>因子</th><th>错误</th></tr></thead><tbody>{fail_rows}</tbody></table></section>
</main>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")
