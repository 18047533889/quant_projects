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

from scripts.cogalpha_lqtp.factor_report_zh import render_chart_guide, render_interpretation_block, signal_note_zh
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
        return "—"
    return f"{v:.2%}"


def _enrich_topk_summary(summary: dict[str, Any] | None) -> dict[str, Any]:
    """Fill annualized_return / calmar when only total_return (+ days) is known."""
    out = dict(summary or {})
    try:
        total = float(out.get("total_return"))
    except (TypeError, ValueError):
        total = float("nan")
    try:
        ann = float(out.get("annualized_return"))
    except (TypeError, ValueError):
        ann = float("nan")
    try:
        days = int(out.get("trading_days") or 0)
    except (TypeError, ValueError):
        days = 0
    if (ann != ann or out.get("annualized_return") is None) and total == total and total > -1.0:
        years = (days / 252.0) if days > 0 else (1815 / 252.0)
        if years > 0:
            ann = (1.0 + total) ** (1.0 / years) - 1.0
            out["annualized_return"] = ann
    try:
        max_dd = float(out.get("max_drawdown"))
    except (TypeError, ValueError):
        max_dd = float("nan")
    try:
        calmar = float(out.get("calmar"))
    except (TypeError, ValueError):
        calmar = float("nan")
    if (calmar != calmar or out.get("calmar") is None) and ann == ann and max_dd == max_dd and abs(max_dd) > 1e-12:
        out["calmar"] = float(ann / abs(max_dd))
    return out


def _has_topk_metrics(summary: dict[str, Any] | None) -> bool:
    if not summary:
        return False
    try:
        sh = float(summary.get("sharpe"))
        tr = float(summary.get("total_return"))
        return (sh == sh) or (tr == tr)
    except (TypeError, ValueError):
        return False


def _sanitize_eval_mode_label(eval_mode: str) -> str:
    s = str(eval_mode or "").strip()
    if not s:
        return "unknown"
    s = s.replace("duckdb_panel_close_to_close", "duckdb_panel")
    while "+topk_open+topk_open" in s:
        s = s.replace("+topk_open+topk_open", "+topk_open")
    return s


def _rankic_chart_title(payload: dict[str, Any]) -> str:
    kind = payload.get("return_kind") or ""
    if kind == "close_to_close_T_plus_1":
        return "Daily RankIC (factor T -> T+1 forward return)"
    if kind == "open_to_open_T_plus_2":
        return "Daily RankIC (factor T -> open(T+1)->open(T+2))"
    return "Daily RankIC"


def _ls_chart_title(payload: dict[str, Any]) -> str:
    kind = payload.get("return_kind") or ""
    if kind == "close_to_close_T_plus_1":
        return "Cumulative Long-Short (G10-G1, net)"
    if kind == "open_to_open_T_plus_2":
        return "Cumulative Long-Short (G10-G1, open-to-open, net)"
    return "Cumulative Long-Short (G10-G1, net)"


def _img_b64(b64: str, alt: str) -> str:
    if not b64:
        return ""
    return f'<img alt="{html_lib.escape(alt)}" src="data:image/png;base64,{b64}"/>'


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
    mean_abs = float(pd.Series(ls).abs().mean())
    if mean_abs > 0.02 or payload.get("return_kind") == "lqtp_platform_analyze":
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(
            0.5,
            0.5,
            f"LS series discarded (mean |daily LS|={mean_abs:.3%}, platform analyze).\n"
            "Recompute with local DuckDB panel.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            color="#991b1b",
        )
        ax.set_axis_off()
        ax.set_title(_ls_chart_title(payload))
        return _fig_to_base64(fig)
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
    if payload.get("return_kind") == "lqtp_platform_analyze" or max(abs(x) for x in finals) > 100.0:
        means = payload.get("group_mean_returns") or []
        fig, ax = plt.subplots(figsize=(8, 3.2))
        if len(means) == len(labels):
            colors = [
                "#c0392b" if i == 0 else "#27ae60" if i == len(labels) - 1 else "#5d6d7e"
                for i in range(len(labels))
            ]
            ax.bar(labels, means, color=colors)
            ax.axhline(0.0, color="gray", linewidth=0.8)
            ax.set_title("Decile Mean Daily Return (platform cum-PnL discarded)")
            ax.set_ylabel("Mean Daily Return")
        else:
            ax.text(
                0.5,
                0.5,
                "Group cum-PnL discarded (explosive / platform analyze).\n"
                "Recompute with local DuckDB panel.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="#991b1b",
            )
            ax.set_axis_off()
        return _fig_to_base64(fig)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    colors = ["#c0392b" if i == 0 else "#27ae60" if i == len(labels) - 1 else "#5d6d7e" for i in range(len(labels))]
    ax.bar(labels, finals, color=colors)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_title("Decile Cumulative Return (G1=low factor ... G10=high factor)")
    ax.set_ylabel("Cumulative Return")
    return _fig_to_base64(fig)


def _plot_backtest_nav(rows: list[dict[str, Any]], title: str = "TopK Backtest NAV (LQTP OPEN)") -> str:
    if not rows:
        return ""
    frame = pd.DataFrame(rows).copy()
    frame["dt"] = pd.to_datetime(frame["trade_date"].astype(str))
    nav = pd.to_numeric(frame["eod_net_asset"], errors="coerce")
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


# 算值路径：因子值怎么来的（与 RankIC/TopK 评估口径分开）
_COMPUTE_PATH: dict[str, tuple[str, str, str]] = {
    # key -> (短标签, 徽章色, 说明)
    "local_dsl": (
        "自研 factor_engine DSL 落值",
        "#0f766e",
        "Python 已转换成自研 DSL（ts_mean / protected_div / ema 等命名），"
        "用本地 factor_engine + COS A 股数据落因子值；不是 LQTP 平台直算。",
    ),
    "lqtp_dsl": (
        "LQTP 原生 DSL 直算",
        "#1d4ed8",
        "公式直接提交 LQTP RunFactor，用平台原生 DSL 在线上计算因子值；"
        "未走本地 factor_engine 落值。",
    ),
    "local_python": (
        "纯 Python 落值",
        "#a16207",
        "未转换成可执行 DSL（或仅有阅读用公式），直接跑 Python 因子函数落值。",
    ),
}


def resolve_compute_path(
    *,
    engine: str = "",
    eval_route: str = "",
    eval_mode: str = "",
    materialize_meta: dict[str, Any] | None = None,
) -> tuple[str, str, str, str]:
    """Return (route_key, short_label, color, detail)."""
    meta = materialize_meta or {}
    route = (
        str(eval_route or meta.get("eval_route") or "").strip()
        or str(engine or meta.get("engine") or "").strip()
    )
    if route in {"factor_engine", "local_dsl"}:
        key = "local_dsl"
    elif route in {"lqtp_dsl", "lqtp"}:
        key = "lqtp_dsl"
    elif route in {"python", "local_python"}:
        key = "local_python"
    elif "lqtp_values_duckdb" in str(eval_mode) or "lqtp_run_factor" in str(eval_mode):
        key = "lqtp_dsl"
    elif "duckdb" in str(eval_mode) or "local_values" in str(eval_mode):
        # Prefer engine when mode is ambiguous
        eng = str(engine or meta.get("engine") or "")
        if eng == "lqtp_dsl":
            key = "lqtp_dsl"
        elif eng == "python":
            key = "local_python"
        else:
            key = "local_dsl"
    else:
        key = "local_python" if route else "local_dsl"
    label, color, detail = _COMPUTE_PATH[key]
    return key, label, color, detail


def _compute_path_badge(label: str, color: str, detail: str) -> str:
    return (
        f'<div class="path-badge" style="border-color:{color};">'
        f'<span class="path-tag" style="background:{color};">{html_lib.escape(label)}</span>'
        f'<span class="path-detail">{html_lib.escape(detail)}</span>'
        f"</div>"
    )


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
    engine: str = "",
    eval_route: str = "",
) -> None:
    ann = _load_annotation(work_dir, factor_name, annotation)
    path_key, path_label, path_color, path_detail = resolve_compute_path(
        engine=engine,
        eval_route=eval_route,
        eval_mode=eval_mode,
        materialize_meta=materialize_meta,
    )
    path_badge = _compute_path_badge(path_label, path_color, path_detail)
    ic_img = _plot_daily_rank_ic(analysis)
    ls_img = _plot_cumulative_ls(analysis)
    group_img = _plot_group_returns(analysis)
    topk_rows = backtest_rows or []
    topk_img = _plot_backtest_nav(topk_rows, "TopK Long-Only Backtest NAV (LQTP OPEN)")
    ls_bt_rows = ls_backtest_rows or []
    ls_bt_img = _plot_backtest_nav(ls_bt_rows, "Long-Short Backtest NAV (LQTP OPEN)")

    topk_summary = _enrich_topk_summary(topk_summary or summarize_backtest(topk_rows))
    ls_summary = ls_summary or (summarize_backtest(ls_bt_rows) if ls_bt_rows else {})
    ls_summary = _enrich_topk_summary(ls_summary) if ls_summary else {}

    mean_rank_ic = analysis.get("mean_rank_ic", analysis.get("mean_ic"))
    std_rank_ic = analysis.get("std_rank_ic", analysis.get("std_ic"))
    rank_icir = analysis.get("rank_icir", analysis.get("icir"))
    rank_ic_pos = analysis.get("rank_ic_positive_ratio", analysis.get("ic_positive_ratio"))

    rank_cards = _metrics_cards(
        [
            ("Mean RankIC（均值）", mean_rank_ic),
            ("Std RankIC（标准差）", std_rank_ic),
            ("RankICIR", rank_icir),
            ("RankIC 胜率", rank_ic_pos),
            ("多空累计（G10−G1 净）", analysis.get("long_short_return")),
            ("多空 Sharpe（净）", analysis.get("long_short_sharpe")),
        ]
    )

    topk_cards = _metrics_cards(
        [
            ("累计收益", topk_summary.get("total_return")),
            ("年化收益", topk_summary.get("annualized_return")),
            ("最大回撤", topk_summary.get("max_drawdown")),
            ("Sharpe 比率", topk_summary.get("sharpe")),
            ("波动率", topk_summary.get("volatility")),
            ("Calmar 比率", topk_summary.get("calmar")),
            ("胜率", topk_summary.get("win_rate")),
            ("平均换手", topk_summary.get("avg_turnover")),
            ("总佣金", topk_summary.get("total_commission")),
            ("交易天数", topk_summary.get("trading_days")),
        ]
    )

    meta_block = ""
    if materialize_meta:
        title = f"落值元信息 · {path_label}"
        meta_block = f"""
  <div class="card">
    <h2>{html_lib.escape(title)}</h2>
    <pre>{json.dumps(materialize_meta, ensure_ascii=False, indent=2)}</pre>
  </div>"""

    # eval_mode may be "duckdb_panel_close_to_close+topk_open"
    mode_base = str(eval_mode or "").split("+", 1)[0]
    mode_note = {
        "analyze_factor_upload": "官方 AnalyzeFactor：上传落值，平台计算指标。",
        "local_values_lqtp_returns": (
            "本地 RankIC / 分层 / 多空：因子 T 日收盘后可得 → 与 T+1 日前瞻收益对齐。"
        ),
        "duckdb_panel_close_to_close": (
            "RankIC / 分层 / 多空：本地 DuckDB 面板。"
        ),
        "lqtp_run_factor": (
            "DSL 由 LQTP RunFactor 算值；若仍显示本模式，分层/多空可能来自平台 analyze（已知不可靠）。"
        ),
        "lqtp_values_duckdb_panel": (
            "因子值来自 LQTP RunFactor；RankIC / 分层 / 多空用本地 DuckDB "
            "面板重算（已丢弃平台 analyze 的 LS/groups）。"
        ),
    }.get(mode_base, _sanitize_eval_mode_label(eval_mode) if eval_mode else "unknown")

    signal_note = signal_note_zh(analysis.get("signal_lag_note") or "")
    eval_mode_display = _sanitize_eval_mode_label(eval_mode)
    n_groups = analysis.get("n_groups") or 10
    comm_buy = analysis.get("commission_buy", 0.01)
    comm_sell = analysis.get("commission_sell", 0.01)

    formula_display = ann.get("formula_display") or ""
    formula_note = ann.get("formula_note") or ""
    lookahead_risk = ann.get("lookahead_risk") or ""
    lookahead_judgment = ann.get("lookahead_judgment_zh") or ""

    interpretation_block = render_interpretation_block(
        factor_name,
        ann=ann,
        dsl=dsl,
        python_code=python_code,
        lookahead_risk=lookahead_risk,
        lookahead_judgment=lookahead_judgment,
    )

    formula_block = ""
    dsl_clean = (dsl or "").strip()
    has_dsl = bool(dsl_clean and dsl_clean not in {"(python only)"})
    if path_key == "lqtp_dsl":
        dsl_heading = "DSL（LQTP 平台直算）"
    elif path_key == "local_dsl":
        dsl_heading = "DSL（自研 factor_engine）"
    else:
        dsl_heading = "DSL（阅读用 / 未用于落值）"
    if formula_display or has_dsl or (python_code or "").strip():
        primary = formula_display if formula_display else dsl_clean
        src_note = ""
        if ann.get("formula_source") == "ast_translated":
            src_note = "（Python 自动翻译 DSL，便于阅读）"
        elif ann.get("formula_source") == "formula_text":
            src_note = "（来自 formula_text 文档）"
        elif ann.get("is_python_only") and formula_display:
            src_note = "（Python-only 因子 · 公式化表示）"
        show_fe_dsl = has_dsl and (
            path_key == "local_dsl"
            or (formula_display and formula_display.strip() != dsl_clean.strip())
            or path_key == "lqtp_dsl"
        )
        formula_block = f"""
  <div class="card">
    <h2>公式 / 代码</h2>
    <p class="note">下方为可读公式与可执行代码。若同时出现「公式化表示」与 DSL，以 DSL 为准用于落值。</p>
    {f'<p class="note">{_lookahead_badge(lookahead_risk)} {html_lib.escape(formula_note or src_note)}</p>' if (lookahead_risk or formula_note or src_note) else ""}
    {f'<h3>公式化表示{html_lib.escape(src_note)}</h3><pre>{html_lib.escape(primary.strip())}</pre>' if primary.strip() and primary.strip() != dsl_clean else ""}
    {f'<h3>{html_lib.escape(dsl_heading)}</h3><pre>{html_lib.escape(dsl_clean)}</pre>' if show_fe_dsl else ""}
    {f'<h3>Python 源码</h3><pre>{html_lib.escape((python_code or "").strip())}</pre>' if (python_code or "").strip() else ""}
    {"" if (primary.strip() or has_dsl or (python_code or "").strip()) else "<pre>（无公式/代码）</pre>"}
  </div>"""

    ls_bt_section = ""
    if ls_bt_rows:
        ls_cards = _metrics_cards(
            [
                ("累计收益", ls_summary.get("total_return")),
                ("年化收益", ls_summary.get("annualized_return")),
                ("最大回撤", ls_summary.get("max_drawdown")),
                ("Sharpe 比率", ls_summary.get("sharpe")),
                ("胜率", ls_summary.get("win_rate")),
                ("平均换手", ls_summary.get("avg_turnover")),
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
    ol.steps {{ margin: 8px 0 12px 22px; padding: 0; }}
    ol.steps li {{ margin: 6px 0; }}
    ul.op-list {{ margin: 8px 0 12px 22px; font-size: 14px; }}
    .theme-line {{ color: #1e3a5f; font-size: 16px; margin: 0 0 10px; }}
    .chart-guide {{ background: #f8fafc; padding: 10px 12px; border-radius: 6px; font-size: 14px; margin: 0 0 12px; }}
    .hl {{ background: #eef6ff; border: 1px solid #c9def5; padding: 10px 12px; border-radius: 6px; }}
    .path-badge {{ display:flex; gap:12px; align-items:flex-start; border:2px solid; border-radius:8px; padding:12px 14px; margin:12px 0 16px; background:#fff; }}
    .path-tag {{ flex:0 0 auto; color:#fff; font-weight:700; font-size:13px; padding:4px 10px; border-radius:4px; }}
    .path-detail {{ color:#374151; font-size:13px; line-height:1.5; }}
    details {{ margin-top: 12px; }}
    @media (max-width: 900px) {{ .metrics {{ grid-template-columns: 1fr 1fr; }} .path-badge {{ flex-direction:column; }} }}
  </style>
</head>
<body>
  <h1>{factor_name}</h1>
  {path_badge}
  <div class="hl note">
    <b>评估口径</b>：{mode_note}<br/>
    {html_lib.escape(signal_note)}<br/>
    算值路径=<code>{html_lib.escape(path_key)}</code>
    · 评估模式=<code>{html_lib.escape(eval_mode_display)}</code>
  </div>

  {formula_block or f'''
  <div class="card">
    <h2>公式 / 代码</h2>
    {f'<h3>Python 源码</h3><pre>{html_lib.escape((python_code or "").strip())}</pre>' if (python_code or "").strip() else ""}
    {f'<h3>{html_lib.escape(dsl_heading)}</h3><pre>{html_lib.escape((dsl or "").strip())}</pre>' if (dsl or "").strip() and (dsl or "").strip() not in {{"(python only)", ""}} else ""}
    {"" if ((python_code or "").strip() or ((dsl or "").strip() and (dsl or "").strip() not in {{"(python only)", ""}})) else "<pre>（无公式/代码）</pre>"}
  </div>'''}
  {interpretation_block}
  {meta_block}

  <div class="card">
    <h2>① RankIC（因子预测力）</h2>
    {render_chart_guide("rankic")}
    <div class="metrics">{rank_cards}</div>
    {_img_b64(ic_img, "Daily RankIC")}
  </div>

  <div class="card">
    <h2>② 分层回测（十分位 G1…G10）</h2>
    {render_chart_guide("decile")}
    {_img_b64(group_img, "Decile cumulative return")}
  </div>

  <div class="card">
    <h2>③ 多空（G10 做多 − G1 做空）</h2>
    {render_chart_guide("ls")}
    <p class="note">
      <b>计算步骤（DuckDB 面板，非 LQTP 平台回测）：</b><br/>
      1) 每个交易日 T，对全市场股票按因子值升序分成 {n_groups} 组（G1 最低 … G10 最高）；<br/>
      2) 各组内股票等权，组收益 = 组内个股 T+1 相对 T 的前瞻收益均值；<br/>
      3) 当日多空 gross = mean(G10 收益) − mean(G1 收益)；<br/>
      4) 扣费：假设每日单边换手 40%，日成本 = 2 × 40% × (买{comm_buy}% + 卖{comm_sell}%) = {2 * 0.4 * (float(comm_buy) + float(comm_sell)):.4f}% ；net = gross − 日成本；<br/>
      5) 多空 Sharpe = mean(日 net) / std(日 net) × √252；多空累计 = ∏(1+日 net) − 1。<br/>
      <b>注意</b>：这是研究用十分位多空，不是可交易的 TopK 组合；TopK 见下方。
    </p>
    {_img_b64(ls_img, "Cumulative long-short")}
  </div>

  <div class="card">
    <h2>④ TopK 回测（LQTP 平台 · 开盘价）</h2>
    {render_chart_guide("topk")}
    <p class="note">
      每天取因子 Top 10%（最多 50 只）等权做多；信号日 T 的权重落到 T+1，
      以开盘价成交；买卖手续费各 0.01%；初始本金 1000 万。
      下图 NAV 已归一化到起点=1（平台原始净值为绝对金额）。
    </p>
    <div class="metrics">{topk_cards}</div>
    {_img_b64(topk_img, "TopK NAV") if topk_img else (
        '<p class="note">有 TopK 指标摘要，但净值曲线未缓存（需重新跑 LQTP TopK 回测才会出现 NAV 图）。</p>'
        if _has_topk_metrics(topk_summary)
        else '<p class="note">无 TopK 回测结果（尚未跑或被跳过）。</p>'
    )}
    {_kv_table(
        [
            ("累计收益", _fmt_pct(topk_summary.get("total_return"))),
            ("年化收益", _fmt_pct(topk_summary.get("annualized_return"))),
            ("最大回撤", _fmt_pct(topk_summary.get("max_drawdown"))),
            ("Sharpe 比率", _fmt(topk_summary.get("sharpe"))),
            ("波动率", _fmt_pct(topk_summary.get("volatility"))),
            ("Calmar 比率", _fmt(topk_summary.get("calmar"))),
            ("胜率", _fmt_pct(topk_summary.get("win_rate"))),
            ("平均换手", _fmt_pct(topk_summary.get("avg_turnover"))),
            ("总佣金", _fmt(topk_summary.get("total_commission"))),
            ("期末净值", _fmt(topk_summary.get("final_nav"))),
            ("交易天数", _fmt(topk_summary.get("trading_days"))),
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
      <pre>{json.dumps(_analysis_json_export(analysis), ensure_ascii=False, indent=2)}</pre>
    </details>
  </div>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def _analysis_json_export(analysis: dict[str, Any]) -> dict[str, Any]:
    skip = {
        "daily_ic",
        "daily_rank_ic",
        "daily_ls_returns",
        "daily_ls_gross",
        "group_pnls",
        "coverages",
        "sample_counts",
        "trade_dates",
        "quote_times",
        "return_kind",
        "return_mode",
    }
    out: dict[str, Any] = {}
    for k, v in analysis.items():
        if k in skip:
            continue
        if k == "signal_lag_note":
            out[k] = signal_note_zh(str(v or ""))
        else:
            out[k] = v
    return out


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
    rows = []
    for item in reports:
        _, label, _, _ = resolve_compute_path(
            engine=str(item.get("engine") or ""),
            eval_route=str(item.get("eval_route") or ""),
            eval_mode=str(item.get("eval_mode") or ""),
        )
        rows.append(
            f'<li><a href="{item["report"]}">{item["factor_name"]}</a> '
            f'· <b>{label}</b> '
            f'· mode={_sanitize_eval_mode_label(str(item.get("eval_mode", "?")))} '
            f'· RankIC={item.get("mean_ic", item.get("mean_rank_ic", "n/a"))} '
            f'· RankICIR={item.get("icir", item.get("rank_icir", "n/a"))}</li>'
        )
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate"/>
<title>CogAlpha → LQTP</title></head>
<body>
  <h1>CogAlpha 因子评估索引</h1>
  <p>算值路径分三类：<b>自研 factor_engine DSL 落值</b> / <b>LQTP 原生 DSL 直算</b> / <b>纯 Python 落值</b>。
  RankIC / 分层 / 多空 / TopK（开盘成交）。</p>
  <ul>{chr(10).join(rows)}</ul>
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
        table_rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f'<td><a href="{item.get("report", "#")}">{item.get("factor_name", "")}</a></td>'
            f"<td>{path_label}</td>"
            f'<td>{_sanitize_eval_mode_label(str(item.get("eval_mode", "")))}</td>'
            f'<td>{_safe_float(item.get("mean_rank_ic", item.get("mean_ic"))):.4f}</td>'
            f'<td>{_safe_float(item.get("rank_icir", item.get("icir"))):.4f}</td>'
            f'<td>{_safe_float(item.get("rank_ic_positive_ratio", item.get("ic_positive_ratio"))):.2%}</td>'
            f'<td>{_safe_float(item.get("long_short_sharpe")):.3f}</td>'
            f'<td>{_safe_float(topk.get("sharpe")):.3f}</td>'
            f'<td>{_safe_float(topk.get("total_return")):.2%}</td>'
            f'<td>{_safe_float(topk.get("max_drawdown")):.2%}</td>'
            f'<td>{_safe_float(topk.get("annualized_return")):.2%}</td>'
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
  <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate"/>
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
  <p class="muted">区间 {date_range[0]} ~ {date_range[1]} · RankIC / 分层 / 多空 / TopK(OPEN)</p>
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
      <p><b>算值路径（因子值怎么算出来的）</b></p>
      <ul style="margin:6px 0 10px 18px;">
        <li><b>自研 factor_engine DSL 落值</b>：Python → 自研 DSL（ts_mean/protected_div 等）→ 本地 factor_engine + COS 数据落值</li>
        <li><b>LQTP 原生 DSL 直算</b>：公式直接丢给 LQTP RunFactor，平台线上算值</li>
        <li><b>纯 Python 落值</b>：没有可执行 DSL，直接跑 Python 函数落值</li>
      </ul>
      <p><b>评估口径：</b>RankIC / 分层 / 多空基于本地 DuckDB 面板（T 日因子 → T+1 日前瞻收益）。
      TopK 回测 = LQTP 平台 OPEN 成交（Top10%≤50 只，信号 T → T+1 开盘）。</p>
    </div>
  </section>
  <section>
    <h2>因子排行（按 Mean RankIC 降序）</h2>
    <table>
      <thead>
        <tr>
          <th>#</th><th>因子</th><th>算值路径</th><th>评估模式</th>
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
