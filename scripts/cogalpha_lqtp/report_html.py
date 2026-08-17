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
from scripts.cogalpha_lqtp.report_theme import (
    PALETTE,
    REPORT_CSS,
    SUMMARY_CSS,
    chart_grid,
    img_tag,
    metric_card,
    metrics_grid,
    section_block,
    setup_plot_style,
)

setup_plot_style()

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
            return "—"
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
    s = s.replace("duckdb_panel_vwap_to_vwap", "duckdb_panel")
    s = s.replace("duckdb_panel_close_to_close", "duckdb_panel")
    while "+topk_open+topk_open" in s:
        s = s.replace("+topk_open+topk_open", "+topk_open")
    return s


def _rankic_chart_title(payload: dict[str, Any]) -> str:
    kind = payload.get("return_kind") or ""
    if kind in {"vwap_to_vwap_T_plus_1", "vwap_to_vwap_T1_T2"}:
        return "Daily RankIC (factor T -> VWAP(T+1)->VWAP(T+2))"
    if kind == "close_to_close_T_plus_1":
        return "Daily RankIC (factor T -> close(T+1)/close(T)-1)"
    if kind == "open_to_open_T_plus_2":
        return "Daily RankIC (factor T -> open(T+1)->open(T+2))"
    return "Daily RankIC"


def _ls_chart_title(payload: dict[str, Any]) -> str:
    kind = payload.get("return_kind") or ""
    if kind in {"vwap_to_vwap_T_plus_1", "vwap_to_vwap_T1_T2"}:
        return "Cumulative Long-Short (G10-G1, VWAP→VWAP T+1→T+2, net)"
    if kind == "close_to_close_T_plus_1":
        return "Cumulative Long-Short (G10-G1, close→close, net)"
    if kind == "open_to_open_T_plus_2":
        return "Cumulative Long-Short (G10-G1, open-to-open, net)"
    return "Cumulative Long-Short (G10-G1, net)"


def _img_b64(b64: str, alt: str) -> str:
    return img_tag(b64, alt)


def _plot_daily_rank_ic(payload: dict[str, Any]) -> str:
    dates = payload.get("trade_dates", [])
    ic = payload.get("daily_rank_ic") or payload.get("daily_ic", [])
    if not dates or not ic:
        return ""
    frame = pd.DataFrame({"trade_date": dates, "rank_ic": ic})
    frame["dt"] = pd.to_datetime(frame["trade_date"].astype(str))
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(frame["dt"], frame["rank_ic"], linewidth=1.0, color=PALETTE["primary"], alpha=0.85)
    ax.fill_between(frame["dt"], frame["rank_ic"], 0, alpha=0.12, color=PALETTE["primary"])
    ax.axhline(0.0, color=PALETTE["neutral"], linewidth=0.8, linestyle="--")
    ax.grid(True, alpha=0.35)
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
    ax.plot(frame["dt"], frame["cum"], linewidth=1.4, color=PALETTE["positive"])
    ax.fill_between(frame["dt"], frame["cum"], 0, alpha=0.1, color=PALETTE["positive"])
    ax.axhline(0.0, color=PALETTE["neutral"], linewidth=0.8, linestyle="--")
    ax.grid(True, alpha=0.35)
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
                PALETTE["negative"] if i == 0 else PALETTE["positive"] if i == len(labels) - 1 else PALETTE["neutral"]
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
    colors = [
        PALETTE["negative"] if i == 0 else PALETTE["positive"] if i == len(labels) - 1 else PALETTE["neutral"]
        for i in range(len(labels))
    ]
    ax.bar(labels, finals, color=colors)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_title("Decile Cumulative Return (G1=low factor ... G10=high factor)")
    ax.set_ylabel("Cumulative Return")
    return _fig_to_base64(fig)


def _plot_ic_monthly_heatmap(heatmap: dict[str, Any]) -> str:
    years = heatmap.get("years") or []
    months = heatmap.get("months") or list(range(1, 13))
    matrix = heatmap.get("matrix") or []
    if not years or not matrix:
        return ""
    data = np.array(
        [[np.nan if v is None else float(v) for v in row] for row in matrix],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(10, max(3.0, 0.35 * len(years) + 1.5)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-0.08, vmax=0.08)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([f"{m}月" for m in months], fontsize=9)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels([str(y) for y in years], fontsize=9)
    ax.set_title("Monthly Mean RankIC Heatmap", fontweight="600")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    return _fig_to_base64(fig)


def _plot_rolling_ic(rolling: dict[str, Any]) -> str:
    dates = rolling.get("trade_dates") or []
    vals = rolling.get("rolling_ic") or []
    window = rolling.get("window", 20)
    if not dates or not vals:
        return ""
    frame = pd.DataFrame({"trade_date": dates, "ic": vals})
    frame["dt"] = pd.to_datetime(frame["trade_date"].astype(str))
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(frame["dt"], frame["ic"], linewidth=1.3, color=PALETTE["secondary"])
    ax.axhline(0.0, color=PALETTE["neutral"], linewidth=0.8, linestyle="--")
    ax.set_title(f"{window}-Day Rolling Mean RankIC", fontweight="600")
    ax.set_ylabel("RankIC")
    ax.grid(True, alpha=0.35)
    return _fig_to_base64(fig)


def _plot_ic_autocorr_curve(curve: dict[str, Any]) -> str:
    lags = curve.get("lags") or []
    values = curve.get("values") or []
    if not lags or not values:
        return ""
    fig, ax = plt.subplots(figsize=(8, 2.8))
    colors = [PALETTE["primary"] if (v == v and v >= 0) else PALETTE["negative"] for v in values]
    ax.bar(lags, values, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axhline(0.0, color=PALETTE["neutral"], linewidth=0.8)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorr")
    ax.set_title("IC Autocorrelation by Lag", fontweight="600")
    ax.grid(True, axis="y", alpha=0.35)
    return _fig_to_base64(fig)


def _plot_industry_ic_breakdown(breakdown: list[dict[str, Any]]) -> str:
    if not breakdown:
        return ""
    labels = [str(r.get("industry_code", "")) for r in breakdown]
    vals = [float(r.get("mean_ic", float("nan"))) for r in breakdown]
    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.28 * len(labels) + 1.5)))
    colors = [PALETTE["positive"] if (v == v and v >= 0) else PALETTE["negative"] for v in vals]
    y_pos = range(len(labels))
    ax.barh(list(y_pos), vals, color=colors, alpha=0.88, height=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0.0, color=PALETTE["neutral"], linewidth=0.8)
    ax.set_xlabel("Mean RankIC")
    ax.set_title("Industry-Level Mean RankIC (Top |IC|)", fontweight="600")
    ax.grid(True, axis="x", alpha=0.35)
    ax.invert_yaxis()
    return _fig_to_base64(fig)


def _metrics_cards(items: list[tuple[str, Any]]) -> str:
    cards = [metric_card(label, _fmt(value)) for label, value in items]
    cols = 5 if len(cards) >= 5 else 4 if len(cards) >= 4 else 3
    return metrics_grid(cards, cols=cols)


def _extended_metrics_cards(ext: dict[str, Any], analysis: dict[str, Any]) -> str:
    items: list[tuple[str, Any]] = []
    for key, label in (
        ("industry_neutral", "行业中性 Mean RankIC"),
        ("size_neutral", "市值中性 Mean RankIC"),
        ("industry_size_neutral", "行业+市值双中性 Mean RankIC"),
    ):
        block = ext.get(key) or {}
        if block.get("mean_rank_ic") is not None:
            items.append((label, block.get("mean_rank_ic")))
            items.append((f"{label} ICIR", block.get("rank_icir")))
    for label, k in (
        ("IC t 统计量", "ic_t_stat"),
        ("IC p 值", "ic_p_value"),
        ("IC 半衰期 (天)", "ic_half_life_days"),
        ("IC 衰减 λ", "ic_decay_lambda"),
        ("因子 Rank 换手", "factor_rank_turnover"),
        ("因子 Rank 自相关", "factor_rank_autocorr"),
        ("多空单边换手（日均）", "ls_mean_one_way_turnover"),
        ("多空日成本（日均）", "ls_mean_daily_cost"),
        ("每日覆盖率（均值）", "mean_daily_coverage"),
        ("每日覆盖率（中位）", "median_daily_coverage"),
        ("市值暴露 (corr)", "size_exposure_corr"),
        ("分层单调性", "decile_monotonicity"),
        ("分层 Spread", "decile_spread"),
        ("LS 最大回撤", "ls_max_drawdown"),
        ("LS 胜率", "ls_win_rate"),
        ("IC 自相关 lag1", "ic_autocorr_lag1"),
        ("IC 自相关 lag5", "ic_autocorr_lag5"),
        ("IC 自相关 lag20", "ic_autocorr_lag20"),
    ):
        val = ext.get(k) if k in ext else analysis.get(k)
        if val is None:
            continue
        try:
            if isinstance(val, float) and val != val:
                # Keep half-life visible as "—" rather than omitting the card.
                if k in {"ic_half_life_days", "ic_decay_lambda"}:
                    items.append((label, float("nan")))
                continue
        except (TypeError, ValueError):
            pass
        items.append((label, val))
    if not items:
        return ""
    return _metrics_cards(items)


def _extended_eval_section(analysis: dict[str, Any]) -> str:
    ext = analysis.get("extended_eval") or {}
    if not ext and not analysis.get("extended_eval_error"):
        return ""
    err = analysis.get("extended_eval_error")
    if err:
        return section_block(
            "extended",
            "④",
            "扩展评估",
            f'<p class="note">扩展评估失败：{html_lib.escape(str(err))}</p>',
        )
    heatmap = ext.get("ic_monthly_heatmap") or analysis.get("ic_monthly_heatmap") or {}
    heat_img = _plot_ic_monthly_heatmap(heatmap)
    rolling = ext.get("rolling_ic_20d") or analysis.get("rolling_ic_20d") or {}
    roll_img = _plot_rolling_ic(rolling)
    ac_curve = ext.get("ic_autocorr_curve") or analysis.get("ic_autocorr_curve") or {}
    ac_img = _plot_ic_autocorr_curve(ac_curve)
    ind_break = ext.get("industry_ic_breakdown") or analysis.get("industry_ic_breakdown") or []
    ind_img = _plot_industry_ic_breakdown(ind_break)
    cards = _extended_metrics_cards(ext, analysis)
    yearly = ext.get("ic_yearly_breakdown") or analysis.get("ic_yearly_breakdown") or []
    year_rows = []
    for y in yearly:
        year_rows.append(
            "<tr>"
            f"<td>{y.get('year')}</td>"
            f"<td>{_fmt(y.get('mean_rank_ic'))}</td>"
            f"<td>{_fmt(y.get('rank_icir'))}</td>"
            f"<td>{_fmt(y.get('rank_ic_positive_ratio'))}</td>"
            f"<td>{y.get('n_days')}</td>"
            "</tr>"
        )
    year_table = (
        "<h3>分年 RankIC</h3><table class='data'><thead><tr>"
        "<th>年份</th><th>Mean RankIC</th><th>RankICIR</th><th>正 IC 占比</th><th>交易日</th>"
        f"</tr></thead><tbody>{''.join(year_rows)}</tbody></table>"
        if year_rows
        else ""
    )
    neutral_note = (
        "<p class='note'>收益口径：VWAP→VWAP（收盘 T 信号 → vwap(T+2)/vwap(T+1)-1）。"
        "行业中性：申万一级（sw_l1）组内去均值；"
        "市值中性：每日 OLS 回归因子 ~ log(MarketCap) 取残差；"
        "双中性：先行业中性再市值中性。辅助数据来自 COS clean_data/ashare/lqtp_data。</p>"
    )
    charts = chart_grid(
        _img_b64(roll_img, "Rolling IC") if roll_img else "",
        _img_b64(ac_img, "IC autocorrelation") if ac_img else "",
        _img_b64(heat_img, "IC monthly heatmap") if heat_img else "",
        _img_b64(ind_img, "Industry IC breakdown") if ind_img else "",
    )
    body = f"{neutral_note}{cards}{year_table}{charts}"
    return section_block(
        "extended",
        "④",
        "扩展评估（中性 IC / 分年 / 显著性 / 半衰期 / 暴露度）",
        body,
        subtitle="行业/市值中性 · 分年 IC · IC 显著性 · 滚动 IC · 自相关 · 分行业 IC · 市值暴露 · 换手",
    )


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
    "local_panel": (
        "中性化面板落值（源侧未提供公式）",
        "#64748b",
        "候选池只交付了已中性化的因子面板数值；源侧未公开可回收 DSL/Python。"
        "本页 RankIC/分层/多空基于该面板本地 DuckDB 评估，不是平台公式直算。",
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
    eng = str(engine or meta.get("engine") or "").strip()
    has_formula = bool(str(meta.get("formula") or "").strip())
    if route in {
        "local_panel",
        "local_panel_values",
        "cos_panel",
        "panel",
    } or eng in {"local_panel", "cos_panel", "panel"}:
        key = "local_panel"
    elif route in {"factor_engine", "local_dsl", "fe_data_access_yearly"} and (
        has_formula or route == "fe_data_access_yearly"
    ):
        key = "local_dsl"
    elif route in {"factor_engine", "local_dsl"} and not has_formula:
        key = "local_panel"
    elif route in {"lqtp_dsl", "lqtp"}:
        key = "lqtp_dsl"
    elif route in {"python", "local_python"}:
        key = "local_python"
    elif "lqtp_values_duckdb" in str(eval_mode) or "lqtp_run_factor" in str(eval_mode):
        key = "lqtp_dsl"
    elif "duckdb" in str(eval_mode) or "local_values" in str(eval_mode):
        # Prefer engine when mode is ambiguous
        if eng == "lqtp_dsl":
            key = "lqtp_dsl"
        elif eng == "python":
            key = "local_python"
        elif eng in {"local_panel", "cos_panel", "panel"} or not has_formula:
            key = "local_panel"
        else:
            key = "local_dsl"
    else:
        if eng in {"local_panel", "cos_panel", "panel"} or (
            not has_formula and not str(meta.get("formula") or "").strip()
        ):
            key = "local_panel"
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
            ("多空单边换手（日均）", analysis.get("ls_mean_one_way_turnover")),
            ("多空日成本（日均）", analysis.get("ls_mean_daily_cost")),
            ("每日覆盖率（均值）", analysis.get("mean_daily_coverage")),
            ("每日覆盖率（中位）", analysis.get("median_daily_coverage")),
            ("日均重叠股票数", analysis.get("mean_overlap_names")),
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

    extended_section = _extended_eval_section(analysis)
    show_topk = bool(topk_rows) or _has_topk_metrics(topk_summary)
    topk_section = ""
    if show_topk:
        topk_section = f"""
  <div class="card">
    <h2>⑤ TopK 回测（LQTP 平台 · 开盘价）</h2>
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
  </div>"""

    meta_block = ""
    if materialize_meta:
        title = f"落值元信息 · {path_label}"
        meta_block = f"""
  <div class="card">
    <h2>{html_lib.escape(title)}</h2>
    <pre>{json.dumps(materialize_meta, ensure_ascii=False, indent=2)}</pre>
  </div>"""

    # eval_mode may be "duckdb_panel_vwap_to_vwap+topk_open"
    mode_base = str(eval_mode or "").split("+", 1)[0]
    mode_note = {
        "analyze_factor_upload": "官方 AnalyzeFactor：上传落值，平台计算指标。",
        "local_values_lqtp_returns": (
            "本地 RankIC / 分层 / 多空：因子 T 日收盘后可得 → 与 T+1 日前瞻收益对齐。"
        ),
        "duckdb_panel_vwap_to_vwap": (
            "RankIC / 分层 / 多空：本地 DuckDB 面板，收益口径 VWAP→VWAP（T+1→T+2）。"
        ),
        "duckdb_panel_vwap_to_vwap_t1t2": (
            "RankIC / 分层 / 多空：本地 DuckDB 面板，收益口径 VWAP→VWAP（T+1→T+2）。"
        ),
        "duckdb_panel_close_to_close": (
            "RankIC / 分层 / 多空：本地 DuckDB 面板，收益口径 close→close T+1。"
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
    elif path_key == "local_panel":
        dsl_heading = "公式（源侧未提供）"
    else:
        dsl_heading = "DSL（阅读用 / 未用于落值）"
    empty_formula_note = (
        "<p class='note'><b>源侧未公开公式</b>：候选池只交付了中性化面板数值"
        "（COS parquet），本地无法回收 DSL / Python。"
        "评估基于面板落值；详情页图表仍可用。</p>"
        "<pre>面板回测 · 无可展示公式/代码</pre>"
        if path_key == "local_panel"
        else "<pre>（无公式/代码）</pre>"
    )
    if formula_display or has_dsl or (python_code or "").strip():
        primary = formula_display if formula_display else dsl_clean
        src_note = ""
        if ann.get("formula_source") == "ast_translated":
            src_note = "（Python 自动翻译 DSL，便于阅读）"
        elif ann.get("formula_source") == "formula_text":
            src_note = "（来自 formula_text 文档）"
        elif ann.get("is_python_only") and formula_display:
            src_note = "（Python-only 因子 · 公式化表示）"
        # Always surface executable DSL when present (even if Python source is also shown).
        # Previously hidden for cos_panel / fe_retest routes → detail pages looked Python-only.
        show_fe_dsl = has_dsl and not dsl_clean.startswith("(python)")
        formula_block = f"""
  <div class="card">
    <h2>公式 / 代码</h2>
    <p class="note">下方为可读公式与可执行代码。若同时出现「公式化表示」与 DSL，以 DSL 为准用于落值。</p>
    {f'<p class="note">{_lookahead_badge(lookahead_risk)} {html_lib.escape(formula_note or src_note)}</p>' if (lookahead_risk or formula_note or src_note) else ""}
    {f'<h3>{html_lib.escape(dsl_heading)}</h3><pre>{html_lib.escape(dsl_clean)}</pre>' if show_fe_dsl else ""}
    {f'<h3>公式化表示{html_lib.escape(src_note)}</h3><pre>{html_lib.escape(primary.strip())}</pre>' if primary.strip() and primary.strip() != dsl_clean and not show_fe_dsl else ""}
    {f'<h3>Python 源码</h3><pre>{html_lib.escape((python_code or "").strip())}</pre>' if (python_code or "").strip() else ""}
    {"" if (primary.strip() or has_dsl or (python_code or "").strip()) else empty_formula_note}
  </div>"""
    elif path_key == "local_panel":
        formula_block = f"""
  <div class="card">
    <h2>公式 / 代码</h2>
    {empty_formula_note}
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
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html_lib.escape(factor_name)} · 因子评估报告</title>
  <style>{REPORT_CSS}</style>
</head>
<body>
<div class="page">
  <header class="hero">
    <h1>{html_lib.escape(factor_name)}</h1>
    <p class="hero-meta">RankIC · 分层 · 多空 · 扩展评估</p>
  </header>

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
    {"" if ((python_code or "").strip() or ((dsl or "").strip() and (dsl or "").strip() not in {{"(python only)", ""}})) else empty_formula_note}
  </div>'''}
  {interpretation_block}
  {meta_block}

  {section_block(
    "rankic",
    "①",
    "RankIC（因子预测力）",
    render_chart_guide("rankic") + rank_cards + _img_b64(ic_img, "Daily RankIC"),
  )}

  {section_block(
    "decile",
    "②",
    "分层回测（十分位 G1…G10）",
    render_chart_guide("decile") + _img_b64(group_img, "Decile cumulative return"),
  )}

  {section_block(
    "ls",
    "③",
    "多空（G10 做多 − G1 做空）",
    render_chart_guide("ls")
    + f'''<p class="note">
      <b>计算步骤（DuckDB 面板，非 LQTP 平台回测）：</b><br/>
      1) 每个交易日 T，对全市场股票按因子值升序分成 {n_groups} 组（G1 最低 … G10 最高）；<br/>
      2) 各组内股票等权，组收益 = 组内个股前瞻收益均值（VWAP T+1→T+2）；<br/>
      3) 当日多空 gross = mean(G10 收益) − mean(G1 收益)；<br/>
      4) 扣费：按 <b>真实</b> G10/G1 等权组合相对前日的单边换手
      （0.5·Σ|w_t−w_{{t-1}}|，多空两腿相加）× (买{comm_buy}% + 卖{comm_sell}%)；
      net = gross − 日成本。日均换手/成本见上方卡片；<br/>
      5) 多空 Sharpe = mean(日 net) / std(日 net) × √252；多空累计 = ∏(1+日 net) − 1。<br/>
      覆盖率 = 当日（因子∩收益）股票数 / 收益宇宙股票数。<br/>
      <b>注意</b>：这是研究用十分位多空曲线，用于观察因子单调性与分组收益。
    </p>'''
    + _img_b64(ls_img, "Cumulative long-short"),
  )}

  {extended_section}

  {topk_section}

  {ls_bt_section}

  <div class="card">
    <h2>附录</h2>
    {_series_table_collapsed(analysis)}
    <details>
      <summary>Raw Analysis JSON</summary>
      <pre>{json.dumps(_analysis_json_export(analysis), ensure_ascii=False, indent=2)}</pre>
    </details>
  </div>
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
        table_rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f'<td><a href="{item.get("report", "#")}">{item.get("factor_name", "")}</a></td>'
            f"<td>{path_label}</td>"
            f'<td>{_sanitize_eval_mode_label(str(item.get("eval_mode", "")))}</td>'
            f'<td>{_safe_float(item.get("mean_rank_ic", item.get("mean_ic"))):.4f}</td>'
            f'<td>{_safe_float(item.get("rank_icir", item.get("icir"))):.4f}</td>'
            f'<td>{_safe_float(item.get("industry_neutral_mean_rank_ic")):.4f}</td>'
            f'<td>{_safe_float(item.get("size_neutral_mean_rank_ic")):.4f}</td>'
            f'<td>{_safe_float(item.get("long_short_sharpe")):.3f}</td>'
            f'<td>{_safe_float(item.get("size_exposure_corr")):.3f}</td>'
            f'<td>{_safe_float(item.get("decile_monotonicity")):.3f}</td>'
            f'<td>{_safe_float(item.get("ls_max_drawdown")):.3f}</td>'
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
  <style>{SUMMARY_CSS}</style>
</head>
<body>
<header class="hero">
  <h1>CogAlpha 因子生产评估汇总</h1>
  <p class="muted">区间 {date_range[0]} ~ {date_range[1]} · RankIC / 分层 / 多空 / 中性IC / 半衰期</p>
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
      扩展评估含行业中性（sw_l1）、市值中性（log MarketCap OLS 残差）、IC 显著性、
      滚动 IC、IC 自相关、分行业 IC、市值暴露、分层单调性与 LS 最大回撤。</p>
    </div>
  </section>
  <section>
    <h2>因子排行（按 Mean RankIC 降序）</h2>
    <table>
      <thead>
        <tr>
          <th>#</th><th>因子</th><th>算值路径</th><th>评估模式</th>
          <th>Mean RankIC</th><th>RankICIR</th><th>行业中性IC</th><th>市值中性IC</th>
          <th>LS Sharpe</th><th>市值暴露</th><th>分层单调性</th><th>LS最大回撤</th>
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
