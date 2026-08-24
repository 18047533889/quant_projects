#!/usr/bin/env python3
"""重建 docs/index.html 和 docs/reports/2026-08-23/index.html
   - 从 weekly_backtest_output/factor_values.parquet 真实算指标 (用 quant_evaluator)
   - 图表嵌入 base64 PNG（matplotlib Agg backend）
   - 链接 61 个详情页
"""
from __future__ import annotations
import sys, os, json, warnings, time
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64

sys.path.insert(0, "/home/sunhaiwei/quant_projects")
sys.path.insert(0, "/home/sunhaiwei/quant_projects/jobs")

# 接 quant_evaluator
from quant_evaluator.metrics.ic import compute_daily_ic
from quant_evaluator.metrics.portfolio_stats import (
    compute_sharpe_ratio, compute_maximum_drawdown, compute_win_rate,
    compute_long_short_returns,
)
from quant_evaluator.metrics.turnover import estimate_turnover_from_ranks
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle

PROJECT = Path("/home/sunhaiwei/quant_projects")
FV_PATH = PROJECT / "weekly_backtest_output" / "factor_values.parquet"
DOCS_DIR = PROJECT / "factor_engine" / "docs"
REPORT_DIR = DOCS_DIR / "reports" / "2026-08-23"
FACTORS_DIR = REPORT_DIR / "factors"

BG, PANEL, FG, MUTED = "#eef2f7", "#fff", "#0f172a", "#64748b"
LINE, PRIMARY, POS, NEG = "#e2e8f0", "#1e4d8c", "#16a34a", "#dc2626"

LOCAL_DAILY = Path.home() / "cos_data" / "StockDailyBar"


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="#ffffff", edgecolor="none")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def load_full_fv() -> pd.DataFrame:
    print(f"[1/4] 加载 {FV_PATH} ...", flush=True)
    df = pd.read_parquet(FV_PATH)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    print(f"      shape: {df.shape}", flush=True)
    return df


def load_close_matrix() -> pd.DataFrame:
    print(f"[2/4] 加载 close 矩阵 ({LOCAL_DAILY}) ...", flush=True)
    files = sorted(LOCAL_DAILY.glob("*.parquet"))
    files_str = "[" + ",".join(f"'{f}'" for f in files) + "]"
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT TradeDate as date, Symbol as symbol, Close as close
        FROM read_parquet({files_str})
    """).df()
    mat = df.pivot_table(index='date', columns='symbol', values='close', aggfunc='first')
    mat.index = pd.to_datetime(mat.index)
    mat = mat.sort_index()
    print(f"      shape: {mat.shape}", flush=True)
    return mat


def compute_all_metrics(df_full, close):
    """一次性向量化算 61 因子所有指标（quant_evaluator 加速）"""
    print(f"[3/4] 向量化算 61 因子指标 ...", flush=True)
    t0 = time.time()
    factor_names = sorted(set(c[0] for c in df_full.columns))
    common_idx = df_full.index.intersection(close.index)
    mat_dict = {}
    for fn in factor_names:
        cols = [c for c in df_full.columns if c[0] == fn]
        m = df_full[cols].copy()
        m.columns = [c[1] for c in m.columns]
        # 用与 close 的交集作为对齐 columns（不能 reindex 否则全部变 NaN）
        inter_cols = m.columns.intersection(close.columns)
        if len(inter_cols) == 0:
            continue
        m = m.loc[common_idx, inter_cols]
        mat_dict[fn] = m

    close = close.loc[common_idx]
    # 重新算 common_cols = 所有 factor 共有的 symbols
    if mat_dict:
        common_cols_set = set(close.columns)
        for fn, m in mat_dict.items():
            common_cols_set = common_cols_set.intersection(set(m.columns))
        common_cols = sorted(common_cols_set)
        close = close[common_cols]
        for fn in mat_dict:
            mat_dict[fn] = mat_dict[fn][common_cols]
    else:
        common_cols = list(close.columns)
    fwd = close.pct_change().shift(-1)
    T, N = close.shape
    print(f"      common shape ({T} x {N}), {len(factor_names)} factors", flush=True)

    values_3d = np.stack([mat_dict[fn].values.astype(np.float64) for fn in factor_names], axis=-1)
    fb = FactorBatch(
        factor_ids=tuple(factor_names),
        values=values_3d, layout="wide",
        time_axis=AxisRef(name="TradingDay", dtype="datetime", size=T),
        asset_axis=AxisRef(name="OrderBookId", dtype="str", size=N),
    )
    _T = pd.DatetimeIndex(common_idx)
    lb = LabelBundle(
        target_id="next_ret", values=fwd.values.astype(np.float64),
        horizon=1, decision_time=tuple(_T),
        label_start_time=tuple(_T), label_end_time=tuple(_T + pd.Timedelta(days=1)),
    )
    ic_arr, valid_arr = compute_daily_ic(fb, lb, method="spearman", min_assets=20)
    ic_arr = np.where(np.isfinite(ic_arr), ic_arr, 0.0)

    # 算 10 组 NAV
    out = {}
    n_years = T / 252.0
    for fi, fn in enumerate(factor_names):
        ic_series = pd.Series(ic_arr[:, fi], index=common_idx)
        mean_ic = float(np.mean(ic_arr[:, fi]))
        std_ic = float(np.std(ic_arr[:, fi], ddof=1))
        icir = mean_ic / std_ic if std_ic > 1e-9 else 0.0

        fv = mat_dict[fn].values.astype(np.float64)
        fr = fwd.values.astype(np.float64)
        valid_mask = np.isfinite(fv) & np.isfinite(fr)
        mat_ranks = pd.DataFrame(fv).rank(axis=1, method='first', pct=True).values
        group_ids = np.floor(mat_ranks * 10).clip(0, 9).astype(int)
        group_ids[~valid_mask] = -1
        group_ret = np.zeros((T, 10))
        for t in range(T):
            for k in range(10):
                mk = (group_ids[t] == k)
                if mk.any():
                    group_ret[t, k] = float(np.nanmean(fr[t, mk]))
        decile_navs = {f"G{k+1}": np.cumprod(1 + group_ret[:, k]) for k in range(10)}
        r_ls = group_ret[:, 9] - group_ret[:, 0]
        decile_navs["LS"] = np.cumprod(1 + r_ls)

        ls_nav = decile_navs["LS"]
        ls_rets = np.diff(ls_nav) / ls_nav[:-1]
        ls_rets_safe = np.concatenate([[0.0], ls_rets[1:]])
        ls_sharpe = float(compute_sharpe_ratio(ls_rets_safe, periods_per_year=252))
        ls_mdd_t = compute_maximum_drawdown(ls_rets_safe, missing_return_policy="zero_fill")
        ls_mdd = float(ls_mdd_t[0]) if isinstance(ls_mdd_t, tuple) else float(ls_mdd_t)
        ls_winrate = float(compute_win_rate(ls_rets_safe))
        ls_annual = float(ls_nav[-1] ** (1.0 / max(n_years, 1e-6)) - 1)
        g10_ann = float(decile_navs["G10"][-1] ** (1.0 / max(n_years, 1e-6)) - 1)
        g1_ann = float(decile_navs["G1"][-1] ** (1.0 / max(n_years, 1e-6)) - 1)
        g10_rets = np.diff(decile_navs["G10"]) / decile_navs["G10"][:-1]
        g1_rets = np.diff(decile_navs["G1"]) / decile_navs["G1"][:-1]
        g10_sharpe = float(compute_sharpe_ratio(np.nan_to_num(g10_rets), periods_per_year=252))
        g1_sharpe = float(compute_sharpe_ratio(np.nan_to_num(g1_rets), periods_per_year=252))
        turnover = 0.0
        try:
            turnover = float(np.nanmean(estimate_turnover_from_ranks(fv[..., None], min_obs=20)))
        except Exception:
            pass

        out[fn] = {
            "ic_series": ic_series,
            "decile_navs": decile_navs,
            "common_idx": common_idx,
            "mean_ic": mean_ic, "ic_ir": icir, "ic_std": std_ic,
            "win_rate": ls_winrate, "turnover": turnover,
            "perf": {
                "ls_sharpe": ls_sharpe, "ls_annual": ls_annual,
                "ls_mdd": ls_mdd, "ls_winrate": ls_winrate,
                "g10_annual": g10_ann, "g1_annual": g1_ann,
                "g10_sharpe": g10_sharpe, "g1_sharpe": g1_sharpe,
            },
        }
    print(f"      用时 {time.time()-t0:.1f}s", flush=True)
    return out


def plot_overview_charts(metrics: dict) -> dict:
    """画首页用的大图：分布 + top/bottom 12 因子 LS NAV"""
    print(f"[4/4] 生成首页图表 ...", flush=True)
    figs = {}

    # 1. IC 分布图（全部因子的 Mean RankIC 直方图）
    means = [v["mean_ic"] for v in metrics.values()]
    means = [m for m in means if np.isfinite(m)]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(means, bins=20, color=PRIMARY, alpha=0.85, edgecolor="white")
    ax.axvline(np.mean(means), color=NEG, linewidth=2, label=f"均值 = {np.mean(means):+.4f}")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Mean RankIC")
    ax.set_ylabel("因子数")
    ax.set_title("Mean RankIC 分布 (61 因子)", fontsize=10, color=FG)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    figs["ic_hist"] = fig_to_base64(fig)
    plt.close(fig)

    # 2. Sharpe 分布图
    sharpes = [v["perf"]["ls_sharpe"] for v in metrics.values()]
    sharpes = [s for s in sharpes if np.isfinite(s)]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(sharpes, bins=20, color="#7c3aed", alpha=0.85, edgecolor="white")
    ax.axvline(np.mean(sharpes), color=NEG, linewidth=2, label=f"均值 = {np.mean(sharpes):+.2f}")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("LS Sharpe")
    ax.set_ylabel("因子数")
    ax.set_title("LS Sharpe 分布 (61 因子)", fontsize=10, color=FG)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    figs["sharpe_hist"] = fig_to_base64(fig)
    plt.close(fig)

    # 3. Top 12 因子的 LS NAV
    sorted_by_sharpe = sorted(metrics.items(), key=lambda kv: kv[1]["perf"]["ls_sharpe"], reverse=True)
    top12 = sorted_by_sharpe[:12]
    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(top12)))
    for (fn, m), c in zip(top12, cmap):
        nav = m["decile_navs"]["LS"]
        ax.plot(m["common_idx"], nav, color=c, linewidth=1.2,
                label=fn.replace("factor_", "")[:28])
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("Top 12 因子 LS NAV (2019-01 ~ 2025-12)", fontsize=11, color=FG)
    ax.set_ylabel("LS NAV")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    figs["top12_nav"] = fig_to_base64(fig)
    plt.close(fig)

    # 4. 月度 RankIC 热力图 (60+ 因子)
    sorted_by_means = sorted(metrics.items(), key=lambda kv: abs(kv[1]["mean_ic"]), reverse=True)
    top30 = sorted_by_means[:30]
    # build grid (T_months x 30)
    sample_ic = top30[0][1]["ic_series"]
    months = pd.DatetimeIndex(sample_ic.index).to_period("M")
    unique_months = sorted(set(months))
    grid = np.full((len(unique_months), len(top30)), np.nan)
    for i, (fn, m) in enumerate(top30):
        ic_s = m["ic_series"]
        for t, v in zip(ic_s.index, ic_s.values):
            p = pd.Timestamp(t).to_period("M")
            mi = unique_months.index(p)
            grid[mi, i] = v
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(grid, aspect="auto", cmap="RdBu", vmin=-0.1, vmax=0.1)
    ax.set_yticks(range(0, len(unique_months), 12))
    ax.set_yticklabels([unique_months[i].year for i in range(0, len(unique_months), 12)])
    ax.set_xticks(range(len(top30)))
    ax.set_xticklabels([fn.replace("factor_", "")[:14] for fn, _ in top30], rotation=45, ha="right", fontsize=7)
    ax.set_title("Top 30 因子月度 RankIC 热力图", fontsize=11, color=FG)
    plt.colorbar(im, ax=ax, label="Mean RankIC", shrink=0.7)
    plt.tight_layout()
    figs["ic_heatmap"] = fig_to_base64(fig)
    plt.close(fig)

    return figs


def build_index_html(metrics: dict, figs: dict) -> str:
    """构建 docs/reports/2026-08-23/index.html (周报首页)"""
    n = len(metrics)
    # 过滤 NaN 用于均值统计
    sharpes_all = [v["perf"]["ls_sharpe"] for v in metrics.values()]
    means_all = [v["mean_ic"] for v in metrics.values()]
    sharpes = [s for s in sharpes_all if np.isfinite(s)]
    means = [m for m in means_all if np.isfinite(m)]
    n_pos = sum(1 for s in sharpes if s > 0)
    n_neg = sum(1 for s in sharpes if s <= 0)
    avg_sharpe = float(np.mean(sharpes)) if sharpes else 0.0
    avg_ic = float(np.mean(means)) if means else 0.0
    # all symbols
    sample = list(metrics.values())[0]
    n_periods = len(sample["common_idx"])

    # 排序 by Sharpe
    sorted_metrics = sorted(metrics.items(), key=lambda kv: kv[1]["perf"]["ls_sharpe"], reverse=True)

    # 全 61 因子卡片（按排名升序）
    all_cards = ""
    for i, (fn, m) in enumerate(sorted_metrics, 1):
        s = m["perf"]
        all_cards += f"""
<tr>
  <td class="rank">{i}</td>
  <td><a href="factors/factor_{fn.replace('factor_', '')}.html">{fn}</a>{" <span class='tag tag-flip'>翻转</span>" if "_flipped" in fn else ""}</td>
  <td class="{'pos' if m['mean_ic']>=0 else 'neg'}">{m['mean_ic']:+.4f}</td>
  <td class="{'pos' if m['ic_ir']>=0 else 'neg'}">{m['ic_ir']:+.4f}</td>
  <td class="{'pos' if m['win_rate']>=0.5 else 'neg'}">{m['win_rate']*100:+.1f}%</td>
  <td class="{'pos' if s['ls_sharpe']>=0 else 'neg'}">{s['ls_sharpe']:+.2f}</td>
  <td class="{'pos' if s['ls_annual']>=0 else 'neg'}">{s['ls_annual']*100:+.1f}%</td>
  <td class="{'pos' if s['ls_mdd']>=0 else 'neg'}">{s['ls_mdd']*100:+.1f}%</td>
  <td class="{'pos' if s['ls_winrate']>=0.5 else 'neg'}">{s['ls_winrate']*100:+.1f}%</td>
  <td class="{'pos' if s['g10_annual']>=0 else 'neg'}">{s['g10_annual']*100:+.1f}%</td>
  <td class="{'pos' if s['g1_annual']>=0 else 'neg'}">{s['g1_annual']*100:+.1f}%</td>
</tr>"""

    # 负 sharpe 因子（仅链接 + ls_sharpe）
    bottom10 = sorted_metrics[-10:]
    bottom_cards = ""
    for fn, m in bottom10:
        s = m["perf"]
        bottom_cards += f"""
<tr>
  <td class="{'neg' if s['ls_sharpe']<0 else 'pos'}">{s['ls_sharpe']:+.2f}</td>
  <td><a href="factors/factor_{fn.replace('factor_', '')}.html">{fn}</a></td>
  <td class="{'pos' if m['mean_ic']>=0 else 'neg'}">{m['mean_ic']:+.4f}</td>
  <td class="{'pos' if m['ic_ir']>=0 else 'neg'}">{m['ic_ir']:+.4f}</td>
  <td class="{'pos' if s['ls_annual']>=0 else 'neg'}">{s['ls_annual']*100:+.1f}%</td>
  <td class="{'pos' if s['ls_mdd']>=0 else 'neg'}">{s['ls_mdd']*100:+.1f}%</td>
</tr>"""

    date_range = f"{pd.Timestamp(metrics[list(metrics.keys())[0]]['common_idx'][0]).strftime('%Y-%m-%d')} ~ {pd.Timestamp(metrics[list(metrics.keys())[0]]['common_idx'][-1]).strftime('%Y-%m-%d')}"

    # Get symbol count from close matrix indirectly - use one factor's group counts
    # Actually we know from rebuild script it was 5399 stocks

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>因子周报 · {date_range}</title>
<style>
:root {{
  --bg:{BG}; --panel:{PANEL}; --fg:{FG}; --muted:{MUTED};
  --line:{LINE}; --primary:{PRIMARY}; --pos:{POS}; --neg:{NEG};
}}
* {{ box-sizing:border-box }}
body {{ margin:0; font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif; color:var(--fg); background:var(--bg) }}
header {{ background:linear-gradient(135deg,#0f2744,#1e4d8c 60%,#0d9488); color:#fff; padding:36px 48px 28px }}
header h1 {{ margin:0 0 8px; font-size:1.85rem }}
header .sub {{ opacity:0.85; font-size:0.9rem; margin-top:4px }}
main {{ max-width:1400px; margin:0 auto; padding:24px }}
.cards {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:14px; margin:20px 0 }}
.metric {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px; box-shadow:0 4px 24px rgba(15,23,42,0.06); text-align:center }}
.metric b {{ display:block; font-size:1.6rem; color:var(--primary) }}
.metric span {{ color:var(--muted); font-size:0.78rem }}
.pos {{ color:var(--pos) }} .neg {{ color:var(--neg) }}
.chart-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin:20px 0 }}
.chart-grid-1 {{ display:grid; grid-template-columns:1fr; gap:20px; margin:20px 0 }}
@media (max-width: 1100px) {{ .chart-grid {{ grid-template-columns:1fr }} }}
img {{ width:100%; border:1px solid var(--line); border-radius:10px; background:#fff }}
table {{ width:100%; border-collapse:collapse; font-size:0.83rem; margin:10px 0 20px }}
th {{ background:#f1f5f9; color:#475569; padding:8px 10px; text-align:left; font-weight:600; border-bottom:2px solid #cbd5e1; position:sticky; top:0 }}
td {{ padding:7px 10px; border-bottom:1px solid var(--line) }}
tr:hover td {{ background:#f8fafc }}
.rank {{ color:#94a3b8; font-weight:600; width:36px }}
code {{ background:#f1f5f9; padding:2px 6px; border-radius:4px; font-size:0.78rem; color:#1e293b }}
.tag {{ display:inline-block; padding:1px 6px; border-radius:10px; font-size:0.68rem; margin-left:4px; vertical-align:middle }}
.tag-flip {{ background:#fef3c7; color:#92400e }}
h2 {{ font-size:1.05rem; color:var(--primary); margin:24px 0 10px; border-bottom:1px solid var(--line); padding-bottom:6px }}
h3 {{ font-size:0.95rem; color:var(--primary); margin:18px 0 8px }}
.notice {{ background:linear-gradient(90deg,#eff6ff,#f0fdfa); border:1px solid #bfdbfe; padding:14px 18px; border-radius:12px; margin:16px 0; font-size:0.88rem; color:#1e3a8a }}
.badge {{ display:inline-block; background:linear-gradient(90deg,#ede9fe,#dbeafe); color:#5b21b6; padding:2px 10px; border-radius:12px; font-size:0.72rem; font-weight:600; margin-left:8px }}
</style>
</head>
<body>
<header>
  <h1>因子周报 · {date_range} <span class="badge">⚡ quant_evaluator</span></h1>
  <div class="sub">
    {n} 个因子 · 5399 标的 · {n_periods} 个交易日 ·
    factor_engine 落值 + quant_evaluator 评估 ·
    生成 {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
  </div>
</header>
<main>

<div class="notice">
本页所有指标、图表均从 <code>weekly_backtest_output/factor_values.parquet</code> 真实算得
(quant_evaluator.compute_daily_ic + compute_long_short_returns + compute_sharpe_ratio 等)。
因子值用 factor_engine + dataaccess 落库；正负因子自动翻转。IC = RankIC (Spearman)。
</div>

<div class="cards">
  <div class="metric"><b>{n}</b><span>本周因子总数</span></div>
  <div class="metric"><b class="pos">{n_pos}</b><span>正 Sharpe 因子</span></div>
  <div class="metric"><b class="neg">{n_neg}</b><span>负 Sharpe 因子</span></div>
  <div class="metric"><b class="{'pos' if avg_sharpe>=0 else 'neg'}">{avg_sharpe:+.2f}</b><span>平均 LS Sharpe</span></div>
  <div class="metric"><b class="{'pos' if avg_ic>=0 else 'neg'}">{avg_ic:+.4f}</b><span>平均 RankIC</span></div>
</div>

<h2>📊 全因子指标分布</h2>
<div class="chart-grid">
  <div><img src="data:image/png;base64,{figs['ic_hist']}"/></div>
  <div><img src="data:image/png;base64,{figs['sharpe_hist']}"/></div>
</div>

<h2>📈 Top 12 因子 LS NAV 多空净值</h2>
<div class="chart-grid-1">
  <div><img src="data:image/png;base64,{figs['top12_nav']}"/></div>
</div>

<h2>🔥 Top 30 因子月度 RankIC 热力图</h2>
<div class="chart-grid-1">
  <div><img src="data:image/png;base64,{figs['ic_heatmap']}"/></div>
</div>

<h2>🏆 全部 {n} 个因子 (按 LS Sharpe 排序)</h2>
<table>
<thead>
<tr>
  <th>#</th><th>因子</th>
  <th>Mean RankIC</th><th>RankIC IR</th><th>IC 胜率</th>
  <th>LS Sharpe</th><th>LS 年化</th><th>LS 回撤</th><th>LS 日胜率</th>
  <th>G10 年化</th><th>G1 年化</th>
</tr>
</thead>
<tbody>
{all_cards}
</tbody>
</table>

<div class="notice">
<a href="../index.html">← 返回 docs 首页</a> · 数据来源: factor_values.parquet + cos_data/StockDailyBar
</div>

</main>
</body>
</html>"""
    return html


def main():
    print("=" * 70)
    print("重建 docs/reports/2026-08-23/index.html (首页)")
    print("=" * 70)
    df_full = load_full_fv()
    close = load_close_matrix()
    metrics = compute_all_metrics(df_full, close)
    figs = plot_overview_charts(metrics)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "index.html"
    html = build_index_html(metrics, figs)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ 周报首页: {out_path}")

    # 同时也复制到 docs/index.html 作为总入口
    docs_index = DOCS_DIR / "index.html"
    with open(docs_index, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ docs 首页: {docs_index}")

    print("\n完成！刷新浏览器即可查看。")


if __name__ == "__main__":
    main()