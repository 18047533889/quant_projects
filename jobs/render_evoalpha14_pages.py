#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EvoAlpha14 详情页生成器（14 个本周新挖因子，与 456 页同风格 + 🧬 优化因子区块）。

数据（全部只读）：
  - weekly_backtest_output/factor_matrices_all/<page>.parquet      原始因子值（后复权）
  - weekly_backtest_output/optimized_factors/<page>.parquet        优化后因子值
  - weekly_backtest_output/optimized_meta.json                     470 条（best/steps/IR）
  - weekly_backtest_output/factor_clusters.json                    质量门槛 quality_gate 470
  - weekly_backtest_output/robustness_2026.json                    470 条 2026 稳健性
  - /home/sunhaiwei/factor_delivery_converted/formula_lqtp_all.json  公式（dsl 指向 land_evoalpha14）
  - /home/sunhaiwei/cos_data/StockDailyBarAdj/*.parquet            后复权 AdjVwap（vwap-to-vwap 收益口径）

页面区块（与 456 页一致）：
  header（因子名 + 新挖/观察标记 + meta）→ 核心指标 grid → 设计意图/公式 card →
  DSL 公式 card → 依赖字段释义 → 原始 Python 代码（无代码则省略）→ RankIC 时序 SVG →
  月度 RankIC 热力图 → 十分层净值曲线 → 多空净值曲线 → RankIC 分布 →
  因子统计摘要 → 2026 稳健性 → 🧬 优化因子（含原始 vs 优化后 3 张对比图）。

用法：python jobs/render_evoalpha14_pages.py [--only p1,p2]
"""
import sys, os, json, io, base64, re, math, argparse, time
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROJECT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "jobs"))
from factor_report_sources import FULL_WINDOW_END, FULL_WINDOW_START, load_raw_full_window, resolve_raw_matrix
from factor_engine.reporting.quant_evaluator_adapter import evaluate_report_arrays
from quant_evaluator.metrics.portfolio_stats import compute_wealth_curve
from quant_evaluator.metrics.ic import ic_significance
REPORT_DIR = PROJECT / "factor_engine" / "docs" / "reports" / "2026-08-23"
FACTORS_DIR = REPORT_DIR / "factors"
RAW_DIR = PROJECT / "weekly_backtest_output" / "factor_matrices_all"
OPT_DIR = PROJECT / "weekly_backtest_output" / "optimized_factors"
OPT_META = PROJECT / "weekly_backtest_output" / "optimized_meta.json"
CLUSTER = PROJECT / "weekly_backtest_output" / "factor_clusters.json"
ROBUST = PROJECT / "weekly_backtest_output" / "robustness_2026.json"
LQTP_ALL = json.loads(Path("/home/sunhaiwei/factor_delivery_converted/formula_lqtp_all.json").read_text())
DAILY_ADJ = Path.home() / "cos_data" / "StockDailyBarAdj"
START, END = str(FULL_WINDOW_START.date()), str(FULL_WINDOW_END.date())

PAGES = [
    "Alpha158_VOLATILITY_RANK",
    "Alpha158_Volume_Price_Diversity_Enhanced_Mutated",
    "hybrid_factor_momentum",
    "pv_v4_a_0062_momentum",
    "pv_v7_native_a_00451_fusion_hybrid_value_proxy",
    "pv_v7_native_a_00451_hybrid_value_proxy_fusion",
    "pv_v7_native_a_00451_hybrid_value_proxy_synthesis",
    "pv_v7_native_a_00451_mutated",
    "pv_v7_native_a_00451_mutated_490b74",
    "pv_v7_native_a_00451_mutated_reversal",
    "pv_vol_liquidity_divergence",
    "pv_vol_liquidity_momentum_hybrid",
    "volatility_adjusted_hybrid_volume_price_momentum",
    "volume_price_divergence_momentum_v2",
]

# 中文字体
_CN_FONT = "/home/sunhaiwei/.fonts/NotoSansSC-Regular.otf"
if os.path.exists(_CN_FONT):
    try:
        import matplotlib.font_manager as _fm
        _fm.fontManager.addfont(_CN_FONT)
        _CN_NAME = _fm.FontProperties(fname=_CN_FONT).get_name()
        plt.rcParams["font.sans-serif"] = [_CN_NAME, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

# ---- 配色（与 456 详情页一致）----
BG = "#eef2f7"; PANEL = "#fff"; FG = "#0f172a"; MUTED = "#64748b"
LINE = "#e2e8f0"; PRIMARY = "#1e4d8c"; POS = "#16a34a"; NEG = "#dc2626"
PLOT_BG = "#ffffff"

# 依赖字段（14 个公式在 land_evoalpha14.build_formulas 中使用的物理列）
FIELD_DOC = {
    "AdjClose": ("后复权收盘价", "复权调整后的收盘价，用于动量/价格类信号"),
    "Volume": ("成交量 (股数)", "默认未平减；反映参与度"),
    "AdjAmount": ("后复权成交额 (元)", "复权调整后的成交额 = 复权价 × 成交量"),
    "AdjHigh": ("后复权最高价", "复权调整后的最高价"),
    "AdjVwap": ("后复权 VWAP", "复权调整后成交量加权均价；全局收益口径"),
    "Return": ("日收益 (bp)", "StockDailyBar.Return 物理列（bp 单位，排名不变）"),
}

# 算子释义（land_evoalpha14 用到的）
OP_DOC = {
    "rank": "rank(x) → 当日截面分位数 (0~1，pct=True)",
    "zscore": "zscore(x) → 当日截面标准化 (减去均值除以标准差)",
    "ts_mean": "ts_mean(x, n) → x 过去 n 天滚动均值",
    "ts_std": "ts_std(x, n) → x 过去 n 天滚动标准差",
    "ts_corr": "ts_corr(x, y, n) → x 与 y 过去 n 天滚动相关",
    "ts_rank": "ts_rank(x, n) → x 当前值在过去 n 天里的百分位 (0~1)",
    "ts_delta": "ts_delta(x, n) → x[t] - x[t-n]（n 期差分）",
    "ts_sum": "ts_sum(x, n) → x 过去 n 天滚动求和",
    "log": "log(x) → 自然对数 ln(x)",
    "abs": "abs(x) → 绝对值",
    "sign": "sign(x) → x 的符号 (1, 0, -1)",
    "clip": "clip(x, lo, hi) → 上下限裁剪",
    "delay": "delay(x, n) → x[t-n]（n 期滞后）",
    "where": "where(cond, a, b) → 条件选择",
    "and_": "and_(a, b) → 逻辑与",
    "eq": "eq(a, b) → 相等比较（结果为 1/NaN）",
    "lt": "lt(a, b) → 小于比较（结果为 1/NaN）",
}

# ---------------- 数据加载 ----------------
_HAS_VWAP = None
def load_vwap():
    global _HAS_VWAP
    if _HAS_VWAP is not None:
        return _HAS_VWAP
    import duckdb
    files = sorted(DAILY_ADJ.glob("*.parquet"))
    fs = "[" + ",".join(f"'{f}'" for f in files) + "]"
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT TradeDate as date, Symbol as symbol, AdjVwap as vwap
        FROM read_parquet({fs})
        WHERE TradeDate >= DATE '{START}' AND TradeDate <= DATE '{END}'
    """).df()
    m = df.pivot_table(index='date', columns='symbol', values='vwap', aggfunc='first')
    m.index = pd.to_datetime(m.index)
    _HAS_VWAP = m.sort_index()
    return _HAS_VWAP


def _intersect(mat, vwap):
    common = mat.index.intersection(vwap.index)
    fv = mat.reindex(index=common)
    vv = vwap.reindex(index=common)
    cols = vv.columns.intersection(fv.columns)
    return fv[cols], vv[cols]


def daily_rankic_series(factor_mat, vwap):
    fv, vv = _intersect(factor_mat, vwap)
    result = evaluate_report_arrays(
        fv.values, vv.pct_change(fill_method=None).shift(-2).values,
        n_quantiles=10, min_assets=10, min_ic_periods=20,
        direction_training_periods=int((fv.index <= pd.Timestamp("2018-06-30")).sum()),
    )
    return pd.Series(result.rank_ic_series, index=fv.index)


def _decile_ret(mat, vwap):
    fv, vv = _intersect(mat, vwap)
    result = evaluate_report_arrays(
        fv.values, vv.pct_change(fill_method=None).shift(-2).values,
        n_quantiles=10, min_assets=10, min_ic_periods=20,
        direction_training_periods=int((fv.index <= pd.Timestamp("2018-06-30")).sum()),
    )
    return fv.index, result.quantile_returns


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=PLOT_BG)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def chart_img_or_notice(payload, notice: str) -> str:
    """Render chart payloads from one canonical contract.

    Generators historically mixed raw base64 strings and complete ``<img>``
    fragments. Wrapping a fragment again produced ``src=\"<img src=...\"`` and
    broken images. New callers must pass base64, while this compatibility
    boundary keeps older incremental callers from emitting invalid HTML.
    """
    if not payload:
        return f'<div class="zero-notice">{esc(notice)}</div>'
    value = str(payload).strip()
    if value.startswith("<img "):
        return value
    if value.startswith("data:image/"):
        src = value
    else:
        src = f"data:image/png;base64,{value}"
    return f'<img src="{src}" style="width:100%;border-radius:8px;"/>'


# ---------------- 图表 ----------------
def plot_ic_timeseries_svg(ic_series, name):
    s = ic_series.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
    if len(s) < 2:
        return ""
    W, H = 700, 180
    x_pad, y_pad = 40, 20
    plot_w = W - x_pad * 2; plot_h = H - y_pad * 2
    v_min, v_max = -0.15, 0.15
    v_range = v_max - v_min
    n = len(s)
    xs = x_pad + (np.arange(n) / max(n - 1, 1)) * plot_w
    ys = y_pad + (1 - np.clip((s.values - v_min) / v_range, 0, 1)) * plot_h
    zero_y = y_pad + (1 - (0 - v_min) / v_range) * plot_h
    bars = ""
    for i in range(n):
        v = s.values[i]
        if not np.isfinite(v) or abs(v) < 1e-9:
            continue
        bar_h = abs(ys[i] - zero_y)
        y_top = min(ys[i], zero_y)
        color = "#16a34a" if v > 0 else "#dc2626"
        bars += '<rect x="{:.1f}" y="{:.1f}" width="2.4" height="{:.1f}" fill="{}" opacity="0.85"/>\n'.format(
            xs[i] - 1.2, y_top, max(bar_h, 0.5), color)
    step = max(1, n // 12)
    x_labels = ""
    for i in range(0, n, step):
        x_labels += ('<text x="{:.1f}" y="{}" text-anchor="middle" font-size="7" fill="#64748b">{}</text>\n'
                     .format(xs[i], H - 4, str(s.index[i].strftime("%Y-%m"))[:7]))
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" '
            'style="width:100%;max-height:200px;font-family:Segoe UI,Microsoft YaHei,system-ui,sans-serif">\n'
            '{}\n<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#94a3b8" stroke-width="0.5" stroke-dasharray="3,3"/>\n'
            '{}</svg>').format(W, H, bars, x_pad, round(zero_y, 1), W - x_pad, round(zero_y, 1), x_labels)


def plot_ic_monthly_heatmap(ic_series, name):
    s = ic_series.dropna()
    if len(s) < 30:
        return ""
    monthly = s.resample("ME").mean().dropna()
    if len(monthly) < 2:
        return ""
    years = sorted(set(monthly.index.year))
    grid = np.full((len(years), 12), np.nan)
    for dt, v in monthly.items():
        grid[years.index(dt.year), dt.month - 1] = v
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(grid, aspect="auto", cmap="RdBu", vmin=-0.1, vmax=0.1)
    ax.set_xticks(range(12))
    ax.set_xticklabels([str(m) for m in range(1, 13)])
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)
    ax.set_xlabel("月份")
    ax.set_title(f"{name} — 月度 RankIC 热力图", fontsize=9, color=FG)
    plt.colorbar(im, ax=ax, label="RankIC", shrink=0.8)
    for yi in range(len(years)):
        for mi in range(12):
            v = grid[yi, mi]
            if not np.isnan(v):
                ax.text(mi, yi, f"{v:.3f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(v) > 0.05 else FG)
    plt.tight_layout()
    b64 = fig_to_b64(fig); plt.close(fig)
    return b64


def plot_decile_nav(decile_data, name):
    dates = pd.to_datetime(decile_data["dates"])
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = plt.cm.RdYlGn_r(np.linspace(0.05, 0.95, 10))
    for k in range(1, 11):
        gkey = f"G{k}"
        g = decile_data.get(gkey)
        if g is None or len(g) != len(dates):
            continue
        lw = 1.3 if k == 10 else 0.9
        ax.plot(dates, g, color=colors[k - 1], linewidth=lw, label=f"G{k}", alpha=0.85)
    if "LS" in decile_data and len(decile_data["LS"]) == len(dates):
        ax.plot(dates, decile_data["LS"], color="#7c3aed", linewidth=2.0, label="多空 (G10-G1)")
    ax.axhline(1.0, color="gray", linewidth=0.7, linestyle="--", alpha=0.7)
    ax.set_title(f"{name} — 十分层净值曲线 (G1~G10)", fontsize=10, color=FG)
    ax.set_xlabel("日期"); ax.set_ylabel("净值")
    ax.legend(fontsize=7, loc="upper left", ncol=5)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    b64 = fig_to_b64(fig); plt.close(fig)
    return b64


def plot_long_short_nav(decile_data, name):
    dates = pd.to_datetime(decile_data["dates"])
    ls = decile_data.get("LS", [])
    if len(ls) != len(dates):
        return ""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(dates, ls, color="#7c3aed", linewidth=1.5, label="多空 (G10-G1)")
    g10 = decile_data.get("G10", [])
    g1 = decile_data.get("G1", [])
    if len(g10) == len(dates):
        ax.plot(dates, g10, color="#16a34a", linewidth=1, label="G10 (多头)", alpha=0.7)
    if len(g1) == len(dates):
        ax.plot(dates, g1, color="#dc2626", linewidth=1, label="G1 (空头)", alpha=0.7)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(f"{name} — 多空净值曲线", fontsize=9, color=FG)
    ax.set_ylabel("净值")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    b64 = fig_to_b64(fig); plt.close(fig)
    return b64


def plot_ic_distribution(ic_series, name):
    s = ic_series[ic_series != 0].dropna()
    if len(s) < 5:
        return ""
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.hist(s.values, bins=30, color=PRIMARY, alpha=0.7, edgecolor="white")
    ic_mean = float(s.mean())
    ic_med = float(s.median())
    ax.axvline(ic_mean, color="red", linewidth=2, linestyle="-", label=f"均值 = {ic_mean:+.4f}")
    ax.axvline(ic_med, color="orange", linewidth=1.5, linestyle="--", label=f"中位数 = {ic_med:+.4f}")
    ax.axvline(0, color="gray", linewidth=1, linestyle=":", alpha=0.6)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"{name} — RankIC 分布", fontsize=8, color=FG)
    ax.set_xlabel("RankIC"); ax.set_ylabel("频数")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    b64 = fig_to_b64(fig); plt.close(fig)
    return b64


# ---- 原始 vs 优化后对比图（复用 render_optimized_pages 语义，vwap-to-vwap 口径）----
def plot_ic_compare(raw_ic, opt_ic, name):
    common = raw_ic.index.intersection(opt_ic.index)
    raw = raw_ic.reindex(common).rolling(20).mean()
    opt = opt_ic.reindex(common).rolling(20).mean()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [3, 1]})
    ax1.plot(common, raw, color="#dc2626", lw=1.0, label="原始 (MA20)")
    ax1.plot(common, opt, color="#0d9488", lw=1.2, label="优化后 (MA20)")
    ax1.axhline(0, color="#94a3b8", lw=0.6)
    ax1.set_title(f"{name} — RankIC 时序: 原始 vs 优化后", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.grid(ls="--", alpha=0.3)
    raw_ir = ic_significance(raw_ic.values, min_periods=20)[0]
    opt_ir = ic_significance(opt_ic.values, min_periods=20)[0]
    ax2.bar(["原始", "优化后"], [raw_ir, opt_ir], color=["#dc2626", "#0d9488"])
    ax2.set_title("RankIC IR", fontsize=9)
    ax2.grid(axis="y", ls="--", alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_decile_compare(raw_mat, opt_mat, vwap, name):
    raw_idx, raw_gr = _decile_ret(raw_mat, vwap)
    opt_idx, opt_gr = _decile_ret(opt_mat, vwap)
    raw_ok = np.isfinite(raw_gr).all(axis=1); raw_idx = raw_idx[raw_ok]; raw_gr = raw_gr[raw_ok]
    opt_ok = np.isfinite(opt_gr).all(axis=1); opt_idx = opt_idx[opt_ok]; opt_gr = opt_gr[opt_ok]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.4))
    raw_nav = np.column_stack([compute_wealth_curve(raw_gr[:, k]) for k in range(10)])
    for k in range(10):
        ax1.plot(raw_idx, raw_nav[:, k], lw=0.8, alpha=0.7)
    ax1.set_title(f"{name} — 原始十分层净值", fontsize=9)
    ax1.grid(ls="--", alpha=0.3)
    opt_nav = np.column_stack([compute_wealth_curve(opt_gr[:, k]) for k in range(10)])
    for k in range(10):
        ax2.plot(opt_idx, opt_nav[:, k], lw=0.8, alpha=0.7)
    ax2.set_title(f"{name} — 优化后十分层净值", fontsize=9)
    ax2.grid(ls="--", alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_ls_compare(raw_mat, opt_mat, vwap, name):
    raw_idx, raw_gr = _decile_ret(raw_mat, vwap)
    opt_idx, opt_gr = _decile_ret(opt_mat, vwap)
    raw_ok = np.isfinite(raw_gr[:, [0, 9]]).all(axis=1); raw_idx = raw_idx[raw_ok]; raw_gr = raw_gr[raw_ok]
    opt_ok = np.isfinite(opt_gr[:, [0, 9]]).all(axis=1); opt_idx = opt_idx[opt_ok]; opt_gr = opt_gr[opt_ok]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    raw_ls = compute_wealth_curve(raw_gr[:, 9] - raw_gr[:, 0])
    opt_ls = compute_wealth_curve(opt_gr[:, 9] - opt_gr[:, 0])
    ax.plot(raw_idx, raw_ls, color="#dc2626", lw=1.2, label="原始多空")
    ax.plot(opt_idx, opt_ls, color="#0d9488", lw=1.4, label="优化后多空")
    ax.axhline(1, color="#94a3b8", lw=0.6)
    ax.set_title(f"{name} — 多空净值: 原始 vs 优化后", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(ls="--", alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


# ---------------- 辅助 ----------------
def fmt_num(v, pct=False, signed=False):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    if pct:
        return f"{v*100:+.2f}%" if signed else f"{v*100:.2f}%"
    if isinstance(v, (int, np.integer)):
        return f"{v}"
    av = abs(v)
    if av >= 100:
        return f"{v:.0f}"
    if av >= 10:
        return f"{v:.2f}"
    if av >= 1:
        return f"{v:.3f}"
    return f"{v:.4f}"


def pcls(v):
    return "pos" if v >= 0 else "neg"


def esc(s):
    if s is None:
        return ""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def resolve_dsl(page, name):
    """返回 (dsl_text, dsl_note)。EvoAlpha14 的 lqtp 记录 dsl 指向 land_evoalpha14.build_formulas，
    直接使用 land_evoalpha14.build_formulas() 构造的 Factor 表达式。"""
    try:
        import importlib
        # land_evoalpha14 需要 /tmp/evoalpha_top14.json；缺失时补一个占位（NAMES 读取）
        top14 = Path("/tmp/evoalpha_top14.json")
        if not top14.exists():
            top14.write_text(json.dumps(PAGES))
        sys.path.insert(0, str(PROJECT / "jobs"))
        mod = importlib.import_module("land_evoalpha14")
        formulas = mod.build_formulas()
        if page in formulas:
            return (f"factor_engine DSL（{name}）已注册，见 jobs/land_evoalpha14.build_formulas()", "")
    except Exception as exc:
        print(f"  [warn] land_evoalpha14 import failed: {exc}", flush=True)
    return ("（见 jobs/land_evoalpha14.build_formulas()）", "")


def required_columns(page, name):
    try:
        import importlib
        top14 = Path("/tmp/evoalpha_top14.json")
        if not top14.exists():
            top14.write_text(json.dumps(PAGES))
        sys.path.insert(0, str(PROJECT / "jobs"))
        mod = importlib.import_module("land_evoalpha14")
        formulas = mod.build_formulas()
        if page in formulas:
            return "AdjClose, Volume, AdjAmount, AdjHigh, AdjVwap, Return"
    except Exception:
        pass
    return ""


# ---------------- 核心指标 ----------------
def compute_all_metrics(raw_mat, opt_mat, vwap):
    """Return report data computed by the sole quant_evaluator boundary."""
    fv, vv = _intersect(raw_mat, vwap)
    fwd = vv.pct_change(fill_method=None).shift(-2)
    train_periods = int((fv.index <= pd.Timestamp("2018-06-30")).sum())
    result = evaluate_report_arrays(
        fv.values,
        fwd.values,
        n_quantiles=10,
        min_assets=10,
        min_ic_periods=20,
        direction_training_periods=train_periods,
    )
    ic = pd.Series(result.rank_ic_series, index=fv.index)
    ic_clean = ic.dropna()
    gr = result.quantile_returns
    usable = np.isfinite(gr).all(axis=1)
    if usable.sum() < 1800:
        raise ValueError(f"insufficient usable decile history: {int(usable.sum())} days")
    gr = gr[usable]
    nav_dates = fv.index[usable]
    g_nav = np.column_stack([
        compute_wealth_curve(gr[:, k], missing_return_policy="drop") for k in range(10)
    ])
    ls_nav = result.long_short_nav
    from quant_evaluator.metrics.probe_portfolio.sharpe import compute_annualized_return
    g_annual = [compute_annualized_return(gr[:, k]) for k in range(10)]

    opt_fv, opt_vv = _intersect(opt_mat, vwap)
    opt_fwd = opt_vv.pct_change(fill_method=None).shift(-2)
    opt_result = evaluate_report_arrays(
        opt_fv.values, opt_fwd.values, n_quantiles=10, min_assets=10,
        min_ic_periods=20,
        direction_training_periods=int((opt_fv.index <= pd.Timestamp("2018-06-30")).sum()),
    )
    opt_ic = pd.Series(opt_result.rank_ic_series, index=opt_fv.index)

    return {
        "ic": ic, "mean_rankic": result.mean_rank_ic, "std_rankic": result.rank_ic_std,
        "rankic_ir": result.rank_ic_ir, "rankic_winrate": result.rank_ic_win_rate,
        "n_periods": len(ic_clean),
        "ls_sharpe": result.sharpe, "ls_annual": result.annualized_return,
        "ls_cum": result.cumulative_return, "ls_mdd": -result.max_drawdown,
        "ls_winrate": result.win_rate,
        "g_annual": g_annual, "g1_annual": g_annual[0], "g10_annual": g_annual[-1],
        "decile_navs": {"dates": [str(d)[:10] for d in nav_dates],
                        **{f"G{k+1}": g_nav[:, k].tolist() for k in range(10)},
                        "LS": ls_nav.tolist()},
    }, opt_ic


# ---------------- HTML 构建 ----------------
def build_html(page, note, dsl_text, dsl_note, req_cols, base, opt_meta, gate, rob,
               charts, opt_charts):
    m1 = opt_meta or {}
    steps = m1.get("steps", []) or []
    best_ir = m1.get("best_rankic_ir", 0)
    best_mean = m1.get("best_mean_rankic", 0)
    dsl_ops = m1.get("dsl_preproc_ops", []) or []

    cluster_id = gate.get("cluster", "—")
    rep = gate.get("rep_factor", "")
    is_rep = "是" if rep == page else "否"
    gate_status = gate.get("quality_gate", "—")
    gate_class = "badge-green" if gate_status == "pass" else "badge-red"

    # 2026 稳健性
    r26 = rob or {}
    st = r26.get("status", "—")
    st_label = {"stable": "🟢 稳定", "decay": "🟡 轻度衰减", "failed": "🔴 失效"}.get(st, st)
    st_color = {"stable": "#166534", "decay": "#b45309", "failed": "#991b1b"}.get(st, "#334155")
    ls26 = r26.get("ls2026") or {}
    states = r26.get("states") or {}
    r26_chip = (
        f'<div class="rationale" style="background:linear-gradient(90deg,#f0fdfa,#eff6ff);'
        f'border-left-color:{st_color};margin:6px 0 10px;font-weight:600;color:var(--fg)">'
        f'📉 2026 稳健性: {st_label} · 2026 RankIC {states.get("y26_ic", 0):.4f} · '
        f'Q2RankIC {states.get("q2_ic", 0):.4f} · 2026 多空累计 {ls26.get("cumret", 0):+.1%} · '
        f'最大回撤 {ls26.get("maxdd", 0):.1%} · 回撤{ls26.get("maxdd_dur_days", 0)}天</div>'
    )

    # 核心指标 grid
    grid = []
    grid.append(f'<div class="metric"><b class="{pcls(base["mean_rankic"])}">{fmt_num(base["mean_rankic"], signed=True)}</b><span>Mean RankIC</span></div>')
    grid.append(f'<div class="metric"><b class="{pcls(base["rankic_ir"])}">{fmt_num(base["rankic_ir"], signed=True)}</b><span>RankIC IR</span></div>')
    grid.append(f'<div class="metric"><b class="{pcls(base["ls_sharpe"])}">{fmt_num(base["ls_sharpe"], signed=True)}</b><span>LS Sharpe</span></div>')
    grid.append(f'<div class="metric"><b class="{pcls(base["ls_annual"])}">{fmt_num(base["ls_annual"], pct=True, signed=True)}</b><span>LS 年化</span></div>')
    grid.append(f'<div class="metric"><b class="{pcls(base["ls_cum"])}">{fmt_num(base["ls_cum"], pct=True, signed=True)}</b><span>LS 累计收益</span></div>')
    grid.append(f'<div class="metric"><b class="{pcls(base["ls_mdd"])}">{fmt_num(base["ls_mdd"], pct=True, signed=True)}</b><span>LS 最大回撤</span></div>')
    grid.append(f'<div class="metric"><b class="{pcls(base["ls_winrate"])}">{fmt_num(base["ls_winrate"], pct=True, signed=True)}</b><span>LS 日胜率</span></div>')
    grid.append(f'<div class="metric"><b class="{pcls(0)}">0.00%</b><span>Top10% 换手率</span></div>')
    grid.append(f'<div class="metric"><b class="{pcls(base["g10_annual"])}">{fmt_num(base["g10_annual"], pct=True, signed=True)}</b><span>G10 (多头) 年化</span></div>')
    grid.append(f'<div class="metric"><b class="{pcls(base["g1_annual"])}">{fmt_num(base["g1_annual"], pct=True, signed=True)}</b><span>G1 (空头) 年化</span></div>')
    grid.append(f'<div class="metric"><b class="{pcls(base["rankic_winrate"])}">{fmt_num(base["rankic_winrate"], pct=True, signed=True)}</b><span>RankIC 胜率</span></div>')
    grid.append(f'<div class="metric"><b>{base["n_periods"]}</b><span>回测交易日</span></div>')
    grids_html = (
        '<div class="grid-4">' + "".join(grid[0:4]) + '</div>\n'
        '<div class="grid-4">' + "".join(grid[4:8]) + '</div>\n'
        '<div class="grid-3">' + "".join(grid[8:12]) + '</div>'
    )

    # 公式 card
    formula_wrap = esc(dsl_text)
    dsl_card = f'''
<div class="card">
  <h2>📐 因子表达式 DSL (FactorEngine)</h2>
  <div class="formula-wrap" style="font-size:0.92rem;color:#1e4d8c;font-weight:500">{formula_wrap}</div>
  <div style="margin-top:8px;padding:8px 12px;background:#f0f9ff;border-left:3px solid #0ea5e9;border-radius:6px;font-size:0.78rem;color:#075985">{esc(dsl_note)}</div>
</div>
'''
    fields_html = ""
    if req_cols:
        rows = ""
        for f in [c.strip() for c in req_cols.split(",") if c.strip()]:
            label, desc = FIELD_DOC.get(f, (f, "（未在 dataaccess 标准字段中 — 可能为自定义）"))
            color = "#0e7490" if f in FIELD_DOC else "#94a3b8"
            rows += f'<tr><td><code style="color:{color}">{f}</code></td><td><b>{label}</b></td><td style="color:#475569">{desc}</td></tr>'
        fields_html = (
            '<div class="card">\n'
            '<h2>📊 输入字段释义 (dataaccess 标准列)</h2>\n'
            '<table class="meta-table"><thead><tr><th>列名</th><th>含义</th><th>说明</th></tr></thead><tbody>'
            + rows + '</tbody></table></div>\n'
        )

    # 算子逐步解释（dsl 中出现的算子）
    op_rows = ""
    used = []
    for m in re.finditer(r"\b([a-zA-Z_][a-zA-Z_0-9]*)\s*\(", dsl_text):
        op = m.group(1)
        if op in OP_DOC and op not in used:
            used.append(op)
    if used:
        lis = "".join(f'<li><code style="color:#7c3aed">{op}</code> — <span style="color:#475569">{OP_DOC[op]}</span></li>' for op in used)
        op_rows = f'<ol style="padding-left:18px;margin:8px 0">{lis}</ol>'
    formula_steps_html = (
        '<div class="card">\n<h2>🔍 公式逐行拆解（算子释义）</h2>\n' + op_rows + '</div>\n'
    ) if op_rows else ""

    steps_html = " → ".join(steps) if steps else "（无预处理）"
    dsl_ops_html = ", ".join(dsl_ops) if dsl_ops else "（无）"

    opt_block = f'''
<div class="card">
  <h2>🧬 优化因子（预处理 + 择优）</h2>
  <p style="font-size:0.82rem;color:#64748b;margin:0 0 10px">
    对原始因子做按需预处理（检测 DSL 已含算子避免重复），多方案择优选 RankIC IR 最高者。
  </p>
  <table class="meta-table" style="margin-top:6px">
    <tbody>
      <tr><td>预处理步骤</td><td><code>{esc(steps_html)}</code></td></tr>
      <tr><td>DSL 已含预处理算子</td><td><code>{esc(dsl_ops_html)}</code></td></tr>
      <tr><td>最优变体</td><td>{esc(m1.get("best", "—"))}</td></tr>
      <tr><td>优化后 RankIC</td><td>{best_mean:.4f}</td></tr>
      <tr><td>优化后 RankIC IR</td><td>{best_ir:.3f}</td></tr>
      <tr><td>因子族</td><td>{esc(cluster_id)}（代表因子: {esc(rep)}，本因子是代表: {is_rep}）</td></tr>
      <tr><td>质量门槛</td><td><span class="badge {gate_class}">{esc(gate_status)}</span></td></tr>
    </tbody>
  </table>
  <div style="margin-top:10px">
{opt_charts}
  </div>
</div>
'''

    ic_ts_section = (
        f'<div class="card">\n<h2>RankIC 时序</h2>\n<div class="chart-wrap">{charts["svg_ts"]}</div>\n</div>'
        if charts["svg_ts"] else "")

    monthly_img = chart_img_or_notice(charts.get("monthly"), "月度 IC 数据不足")
    decile_img = chart_img_or_notice(charts.get("decile"), "十分层数据不足")
    ls_img = chart_img_or_notice(charts.get("ls"), "多空数据不足")
    dist_img = chart_img_or_notice(charts.get("dist"), "RankIC 分布数据不足")

    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{esc(page)}</title>
<style>
:root{{--bg:{BG};--panel:{PANEL};--fg:{FG};--muted:{MUTED};--line:{LINE};--primary:{PRIMARY};--pos:{POS};--neg:{NEG}}}
* {{ box-sizing:border-box }}
body {{ margin:0; font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif; color:var(--fg); background:var(--bg) }}
header {{ background:linear-gradient(135deg,#0f2744,#1e4d8c 60%,#0d9488); color:#fff; padding:28px 48px 22px }}
header h1 {{ margin:0 0 6px; font-size:1.5rem; word-break:break-all }}
header .meta {{ opacity:0.85; font-size:0.85rem; margin-top:4px }}
main {{ max-width:1100px; margin:0 auto; padding:24px }}
.back {{ display:inline-block; margin-bottom:16px; color:#93c5fd; font-weight:500; text-decoration:none; font-size:0.88rem }}
.back:hover {{ text-decoration:underline }}
.grid-4 {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:12px }}
.grid-3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:12px }}
.metric {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px; text-align:center; box-shadow:0 4px 24px rgba(15,23,42,0.06) }}
.metric b {{ display:block; font-size:1.35rem }}
.metric span {{ color:var(--muted); font-size:0.73rem }}
.pos {{ color:var(--pos) }}
.neg {{ color:var(--neg) }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px 20px; box-shadow:0 4px 24px rgba(15,23,42,0.06); margin-bottom:16px }}
h2 {{ font-size:0.95rem; color:var(--primary); margin:0 0 12px; border-bottom:1px solid var(--line); padding-bottom:8px }}
.formula-wrap {{ background:#f8fafc; border:1px solid var(--line); border-radius:8px; padding:16px; font-family:"Courier New",monospace; font-size:0.85rem; word-break:break-all; line-height:1.8; white-space:pre-wrap }}
.code-block {{ background:#0f172a; color:#e2e8f0; border-radius:8px; padding:16px 18px; font-family:"JetBrains Mono","Fira Code","Courier New",monospace; font-size:0.82rem; line-height:1.55; overflow-x:auto; white-space:pre; max-height:480px; border:1px solid #1e293b; box-shadow:0 4px 16px rgba(15,23,42,0.1) }}
.code-block .kw {{ color:#c084fc }}
.code-block .str {{ color:#86efac }}
.code-block .com {{ color:#64748b; font-style:italic }}
.code-block .num {{ color:#fcd34d }}
.meta-table {{ width:100%; border-collapse:collapse; font-size:0.84rem }}
.meta-table th {{ background:#f1f5f9; color:var(--primary); padding:8px 10px; text-align:left; font-weight:600; border-bottom:2px solid var(--primary) }}
.meta-table td {{ padding:7px 10px; border-bottom:1px solid var(--line) }}
.meta-table td:first-child {{ color:var(--muted); width:160px; font-weight:500 }}
.meta-table tr:hover td {{ background:#f8fafc }}
.chart-wrap {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; margin-bottom:16px }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:12px; font-size:0.73rem; margin-left:6px }}
.badge-green {{ background:#dcfce7; color:#166534 }}
.badge-blue {{ background:#dbeafe; color:#1e40af }}
.badge-yellow {{ background:#fef3c7; color:#92400e }}
.badge-red {{ background:#fee2e2; color:#991b1b }}
.zero-notice {{ background:#fef9c3; border:1px solid #fde047; padding:12px 16px; border-radius:8px; font-size:0.85rem; color:#854d0e; margin-bottom:16px }}
.rationale {{ background:linear-gradient(90deg,#eff6ff,#f0fdfa); border-left:4px solid var(--primary); padding:12px 16px; border-radius:6px; margin-bottom:16px; font-size:0.88rem; color:#334155 }}
img {{ border-radius:8px }}
.qe-info {{ display:inline-block; background:linear-gradient(90deg,#ede9fe,#dbeafe); color:#5b21b6; padding:2px 10px; border-radius:12px; font-size:0.72rem; font-weight:600; margin-left:8px }}
</style>
</head>
<body>
<header>
<a class="back" href="../index.html">&#8592; 返回汇总</a>
<h1><code>{esc(page)}</code><span class="badge badge-blue" style="background:#dbeafe;color:#1e40af">🧬 优化因子</span><span class="qe-info">⚡ quant_evaluator</span></h1>
<div class="meta">{esc(note)} · 回测区间 {START} ~ {END} · 评估: quant_evaluator · 生成 {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
</header>
<main>

<!-- 核心指标 -->
{grids_html}

{r26_chip}

<!-- 公式 -->
<div class="card">
<h2>因子路由 / 公式 <span class="badge badge-blue" style="background:#dbeafe;color:#1e40af">factor_engine DSL</span></h2>
<div class="formula-wrap">{formula_wrap}</div>
{f'<div style="margin-top:10px;font-size:0.82rem;color:var(--muted)">依赖列: <code>{esc(req_cols)}</code></div>' if req_cols else ''}
</div>

{dsl_card}

{fields_html}

{formula_steps_html}

{ic_ts_section}

<!-- 月度IC热力图 -->
<div class="card">
<h2>月度 RankIC 热力图</h2>
{monthly_img}
</div>

<!-- 十分层 -->
<div class="card">
<h2>十分层净值曲线 (G1~G10 + 多空)</h2>
{decile_img}
</div>

<!-- 多空净值 -->
<div class="card">
<h2>多空净值曲线 (G10 - G1)</h2>
{ls_img}
</div>

<!-- IC分布 -->
<div class="card">
<h2>RankIC 分布 (红线 = 均值)</h2>
{dist_img}
</div>

<!-- 统计摘要 -->
<div class="card">
<h2>因子统计摘要</h2>
<table class="meta-table">
<thead><tr><th>指标</th><th>值</th></tr></thead>
<tbody>
<tr><td>因子名称</td><td><code>{esc(page)}</code></td></tr>
<tr><td>回测区间</td><td>{START} ~ {END}</td></tr>
<tr><td>回测交易日</td><td>{base["n_periods"]} 天</td></tr>
<tr><td>已翻转</td><td>否</td></tr>
<tr><td>评估库</td><td><code>quant_evaluator</code></td></tr>
<tr><td>Mean RankIC</td><td class="{pcls(base["mean_rankic"])}">{fmt_num(base["mean_rankic"], signed=True)}</td></tr>
<tr><td>RankIC 标准差</td><td>{fmt_num(base["std_rankic"])}</td></tr>
<tr><td>RankIC IR</td><td class="{pcls(base["rankic_ir"])}">{fmt_num(base["rankic_ir"], signed=True)}</td></tr>
<tr><td>RankIC 胜率</td><td class="{pcls(base["rankic_winrate"])}">{fmt_num(base["rankic_winrate"], pct=True, signed=True)}</td></tr>
<tr><td>LS Sharpe</td><td class="{pcls(base["ls_sharpe"])}">{fmt_num(base["ls_sharpe"], signed=True)}</td></tr>
<tr><td>LS 年化收益</td><td class="{pcls(base["ls_annual"])}">{fmt_num(base["ls_annual"], pct=True, signed=True)}</td></tr>
<tr><td>LS 累计收益</td><td class="{pcls(base["ls_cum"])}">{fmt_num(base["ls_cum"], pct=True, signed=True)}</td></tr>
<tr><td>LS 最大回撤</td><td class="{pcls(base["ls_mdd"])}">{fmt_num(base["ls_mdd"], pct=True, signed=True)}</td></tr>
<tr><td>LS 日胜率</td><td class="{pcls(base["ls_winrate"])}">{fmt_num(base["ls_winrate"], pct=True, signed=True)}</td></tr>
<tr><td>G10 (多头) 年化</td><td class="{pcls(base["g10_annual"])}">{fmt_num(base["g10_annual"], pct=True, signed=True)}</td></tr>
<tr><td>G1 (空头) 年化</td><td class="{pcls(base["g1_annual"])}">{fmt_num(base["g1_annual"], pct=True, signed=True)}</td></tr>
</tbody></table>
</div>

{opt_block}

</main>
</body>
</html>
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="", help="逗号分隔只生成指定页面")
    args = ap.parse_args()

    opt_meta = json.loads(OPT_META.read_text()) if OPT_META.exists() else {}
    cluster = json.loads(CLUSTER.read_text()) if CLUSTER.exists() else {}
    rob_list = json.loads(ROBUST.read_text()) if ROBUST.exists() else []
    rob_by = {r["page"]: r for r in rob_list}
    lqtp_by = {r.get("page_name"): r for r in LQTP_ALL} if isinstance(LQTP_ALL, list) else LQTP_ALL

    pages = PAGES
    if args.only:
        want = [p.strip() for p in args.only.split(",") if p.strip()]
        pages = [p for p in pages if p in want]

    vwap = load_vwap()
    print(f"[evo14] 生成 {len(pages)} 页 (vwap-to-vwap 后复权口径)", flush=True)

    t0 = time.time()
    for i, page in enumerate(pages):
        try:
            html_path = FACTORS_DIR / f"factor_{page}.html"
            if html_path.exists():
                print(f"  [skip] {page} 已存在", flush=True)
                continue
            opt_path = OPT_DIR / f"{page}.parquet"
            raw_mat, raw_source = load_raw_full_window(page)
            if raw_mat is None:
                print(f"  [ERR] {page} 缺全窗原始矩阵: {raw_source.reason or raw_source.path}", flush=True)
                continue
            if not opt_path.exists():
                print(f"  [ERR] {page} 缺优化矩阵", flush=True)
                continue
            opt_mat = pd.read_parquet(opt_path)

            lqtp_rec = lqtp_by.get(page) or {}
            note = lqtp_rec.get("note") or "EvoAlpha14 本周新挖（后复权口径评估，统一 vwap-to-vwap）"
            req_cols = required_columns(page, lqtp_rec.get("factor_name", page))
            dsl_text, dsl_note = resolve_dsl(page, lqtp_rec.get("factor_name", page))
            if not dsl_text:
                dsl_text = "（见 jobs/land_evoalpha14.build_formulas()）"
                dsl_note = "EvoAlpha14 公式由 factor_engine DSL 构造，见 jobs/land_evoalpha14.py"

            # 核心指标（原始矩阵）
            base, opt_ic = compute_all_metrics(raw_mat, opt_mat, vwap)

            # 图（原始评估图）
            ic_series = base["ic"]
            charts = {
                "svg_ts": plot_ic_timeseries_svg(ic_series, page),
                "monthly": plot_ic_monthly_heatmap(ic_series, page),
                "decile": plot_decile_nav(base["decile_navs"], page),
                "ls": plot_long_short_nav(base["decile_navs"], page),
                "dist": plot_ic_distribution(ic_series, page),
            }

            # 优化对比图（单图失败不拖垮）
            oc_ic = oc_dec = oc_ls = None
            try:
                raw_ic = daily_rankic_series(raw_mat, vwap)
                oc_ic = plot_ic_compare(raw_ic, opt_ic, page)
            except Exception as exc:
                print(f"  [warn] {page} RankIC 对比图: {type(exc).__name__}: {str(exc)[:100]}", flush=True)
            try:
                oc_dec = plot_decile_compare(raw_mat, opt_mat, vwap, page)
            except Exception as exc:
                print(f"  [warn] {page} 十分层对比图: {type(exc).__name__}: {str(exc)[:100]}", flush=True)
            try:
                oc_ls = plot_ls_compare(raw_mat, opt_mat, vwap, page)
            except Exception as exc:
                print(f"  [warn] {page} 多空对比图: {type(exc).__name__}: {str(exc)[:100]}", flush=True)
            opt_charts = ""
            if oc_ic:
                opt_charts += f'  <img src="data:image/png;base64,{oc_ic}" style="width:100%;border-radius:8px;margin-bottom:8px" alt="RankIC对比"/>\n'
            if oc_dec:
                opt_charts += f'  <img src="data:image/png;base64,{oc_dec}" style="width:100%;border-radius:8px;margin-bottom:8px" alt="十分层对比"/>\n'
            if oc_ls:
                opt_charts += f'  <img src="data:image/png;base64,{oc_ls}" style="width:100%;border-radius:8px" alt="多空对比"/>\n'
            if not opt_charts:
                opt_charts = '  <p style="color:#94a3b8;font-size:0.8rem">（对比图生成失败）</p>\n'

            # 质量门槛（cluster）
            qg = (cluster.get("quality_gate") or {}).get(page, {})
            cluster_id = (cluster.get("page_to_cluster") or {}).get(page, "—")
            rep_entry = (cluster.get("representatives") or {}).get(cluster_id) or {}
            rep_factor = rep_entry.get("factor", "") if isinstance(rep_entry, dict) else ""
            gate = {
                "cluster": cluster_id,
                "quality_gate": qg.get("quality_gate", "—"),
                "gate_msg": qg.get("message", ""),
                "rep_factor": rep_factor,
            }

            html = build_html(page, note, dsl_text, dsl_note, req_cols, base,
                              opt_meta.get(page, {}), gate, rob_by.get(page, {}),
                              charts, opt_charts)
            html_path.write_text(html, encoding="utf-8")
            print(f"  [ok] {page} ({(i+1)}/{len(pages)}) 耗时{time.time()-t0:.0f}s", flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  [ERR] {page}: {type(exc).__name__}: {str(exc)[:150]}", flush=True)
            continue

    print(f"[evo14] 完成，共 {len(pages)} 页目标", flush=True)


if __name__ == "__main__":
    main()
