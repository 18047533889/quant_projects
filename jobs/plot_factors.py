#!/usr/bin/env python3
"""
全套因子分析图表
"""

import json, sys, warnings as _warn
_warn.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "vectorbt_qs"))
import weekly_factor_backtest as _wfb

plt.rcParams.update({
    "axes.facecolor": "#0f1419",
    "figure.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#e6e8ea",
    "xtick.color": "#71767b",
    "ytick.color": "#71767b",
    "text.color": "#e6e8ea",
    "grid.color": "#21262d",
    "grid.linewidth": 0.5,
    "font.family": ["PingFang SC", "Microsoft YaHei", "Arial"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})
sns.set_style("darkgrid")

OUT = _PROJECT_ROOT / "weekly_backtest_output"
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# 加载数据
# ============================================================
print("加载数据...")
with open(OUT / "backtest_results.json") as f:
    results = json.load(f)

fv = pd.read_parquet(str(OUT / "factor_values.parquet"))
print(f"因子值: {fv.shape}")

# 构建最终因子列表（翻转版替代原版）
factor_list = []
for name, res in results.items():
    if not res.get("success"):
        continue
    if name.endswith("_flipped"):
        factor_list.append((name, res, True))
    else:
        flipped = name + "_flipped"
        if flipped not in results:
            factor_list.append((name, res, False))

factor_list.sort(key=lambda x: -x[1]["stats"].get("Sharpe Ratio", 0))

# 最终因子名
final_names = [f[0] for f in factor_list]
all_names = list(results.keys())

# ============================================================
# 计算 IC 时序
# ============================================================
print("计算 IC/RankIC 时序...")

# 加载行情
symbols = _wfb._get_stock_universe_from_sample(symbols_limit=500)
market_data = _wfb.load_market_data(symbols, "2024-01-02", "2025-12-31")
close = market_data["close"]
# 对齐到真实交易日交集
common_idx = close.index.intersection(fv.index)
close = close.loc[common_idx]
fv_aligned = fv.loc[common_idx]
fwd = close.pct_change().shift(-1)

# 只对最终因子列表计算
ic_series = {}
ric_series = {}
for nm in final_names[:30]:  # 最多 30 个
    if nm not in fv_aligned.columns.get_level_values(0):
        continue
    sub = fv_aligned[nm]
    ic_s = []
    ric_s = []
    dates_ic = []
    for dt in common_idx:
        fv_row = sub.loc[dt].dropna()
        ret_row = fwd.loc[dt, fv_row.index].dropna()
        common = fv_row.index.intersection(ret_row.index)
        if len(common) < 10:
            ic_s.append(np.nan)
            ric_s.append(np.nan)
        else:
            x = fv_row[common].values
            y = ret_row[common].values
            x = np.nan_to_num(x, nan=0)
            y = np.nan_to_num(y, nan=0)
            ic = np.corrcoef(x, y)[0, 1]
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() > 3:
                rank_ic = float(pd.Series(x[valid]).corr(pd.Series(y[valid])))
            else:
                rank_ic = 0
            ic_s.append(ic)
            ric_s.append(rank_ic if not np.isnan(rank_ic) else 0)
        dates_ic.append(dt)
    ic_series[nm] = pd.Series(ic_s, index=pd.DatetimeIndex(dates_ic))
    ric_series[nm] = pd.Series(ric_s, index=pd.DatetimeIndex(dates_ic))

ic_ts_df = pd.DataFrame(ic_series)
ric_ts_df = pd.DataFrame(ric_series)

# 月度 IC
monthly_ic = ic_ts_df.resample("M").mean()
monthly_ric = ric_ts_df.resample("M").mean()

print(f"IC 时序: {ic_ts_df.shape}, 月度: {monthly_ic.shape}")


# ============================================================
# 计算 Top10 净值曲线
# ============================================================
print("计算净值曲线...")

def calc_nav():
    navs = {}
    for nm in final_names[:10]:
        if nm not in fv_aligned.columns.get_level_values(0):
            continue
        sub = fv_aligned[nm]  # (dates × symbols) DataFrame
        top_n = max(5, int(len(sub.columns) * 0.2))
        weight_mat = pd.DataFrame(0.0, index=sub.index, columns=sub.columns)
        for dt in sub.index:
            row = sub.loc[dt]
            valid = row.dropna()  # 去除 NaN 的股票
            if len(valid) < top_n:
                continue
            top_s = valid.nlargest(top_n)
            weight_mat.loc[dt, top_s.index] = 1.0 / top_n
        weight_mat = weight_mat.ffill().fillna(0)
        close_sub = close.loc[sub.index, weight_mat.columns]
        rets_strat = (weight_mat.shift(1) * close_sub.pct_change().fillna(0)).sum(axis=1)
        navs[nm] = (1 + rets_strat).cumprod()
    return pd.DataFrame(navs)


nav_df = calc_nav()
bench_nav = close.loc[nav_df.index].pct_change().fillna(0).mean(axis=1)
bench_nav = (1 + bench_nav).cumprod()

print(f"净值: {nav_df.shape}")


# ============================================================
# 相关性矩阵（Top 20）
# ============================================================
print("计算相关性矩阵...")
top20_names = [nm for nm in final_names[:20] if nm in fv_aligned.columns.get_level_values(0)]
corr_vals = {}
for n1 in top20_names:
    for n2 in top20_names:
        sub1 = fv_aligned[n1].mean(axis=1)
        sub2 = fv_aligned[n2].mean(axis=1)
        corr_vals[(n1, n2)] = sub1.corr(sub2)

corr_mat = pd.DataFrame(index=top20_names, columns=top20_names)
for (n1, n2), v in corr_vals.items():
    corr_mat.loc[n1, n2] = v

print(f"相关性: {corr_mat.shape}")


# ============================================================
# 绘图
# ============================================================

# ---- 图1: IC 时序 (多线图) ----
print("图1: IC 时序...")
fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
ax1, ax2 = axes

top_factors = [nm for nm in final_names[:8] if nm in ic_ts_df.columns]
colors = plt.cm.tab10(np.linspace(0, 1, len(top_factors)))

for nm, c in zip(top_factors, colors):
    label = nm.replace("factor_", "").replace("_flipped", " [F]")
    ax1.plot(ic_ts_df.index, ic_ts_df[nm].rolling(20, min_periods=5).mean(),
             label=label, color=c, linewidth=1.5, alpha=0.9)
    ax2.plot(ric_ts_df.index, ric_ts_df[nm].rolling(20, min_periods=5).mean(),
             label=label, color=c, linewidth=1.5, alpha=0.9)

ax1.axhline(0, color="white", linewidth=0.5, alpha=0.3)
ax2.axhline(0, color="white", linewidth=0.5, alpha=0.3)
ax1.set_title("IC 时序 (20日均线, Top 8 因子)", color="white")
ax2.set_title("RankIC 时序 (20日均线, Top 8 因子)", color="white")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02),
           frameon=False, fontsize=8)
plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(str(OUT / "fig_ic_timeseries.png"), dpi=120, bbox_inches="tight")
plt.close()
print("  saved: fig_ic_timeseries.png")


# ---- 图2: IC 热力图 (Top 20 × 月) ----
print("图2: IC 热力图...")
top20_ic_names = [nm for nm in final_names[:20] if nm in monthly_ic.columns]
labels = [n.replace("factor_", "").replace("_flipped", "[F]") for n in top20_ic_names]
hm = monthly_ic[top20_ic_names].T
hm.index = labels

fig, ax = plt.subplots(figsize=(18, 12))
vmax = max(abs(hm.min().min()), abs(hm.max().max()), 0.02)
sns.heatmap(hm, cmap="RdYlGn", center=0, vmin=-vmax, vmax=vmax,
            annot=False, fmt=".3f", ax=ax, linewidths=0.3,
            cbar_kws={"label": "IC"})
ax.set_title("月度 IC 热力图 (Top 20 因子)", color="white", pad=10)
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
fig.savefig(str(OUT / "fig_ic_heatmap.png"), dpi=120, bbox_inches="tight")
plt.close()
print("  saved: fig_ic_heatmap.png")


# ---- 图3: 净值曲线 (Top 10) ----
print("图3: 净值曲线...")
fig, ax = plt.subplots(figsize=(14, 7))

nav_colors = plt.cm.tab10(np.linspace(0, 1, len(nav_df.columns)))
for nm, c in zip(nav_df.columns, nav_colors):
    label = nm.replace("factor_", "").replace("_flipped", " [F]")
    ax.plot(nav_df.index, nav_df[nm], label=label, color=c, linewidth=1.8, alpha=0.85)

ax.plot(bench_nav.index, bench_nav, color="white", linewidth=2.5,
        linestyle="--", label="Benchmark", alpha=0.7)
ax.axhline(1.0, color="white", linewidth=0.5, alpha=0.3)
ax.set_title("Top 10 因子净值曲线 (等权 Top20% 多空)", color="white")
ax.set_ylabel("净值")
ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig(str(OUT / "fig_nav_curves.png"), dpi=120, bbox_inches="tight")
plt.close()
print("  saved: fig_nav_curves.png")


# ---- 图4: 相关性矩阵 ----
print("图4: 相关性矩阵...")
corr_labels = [n.replace("factor_", "").replace("_flipped", "[F]") for n in top20_names]
corr_disp = corr_mat.astype(float)
corr_disp.index = corr_labels
corr_disp.columns = corr_labels

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.zeros_like(corr_disp, dtype=bool)
sns.heatmap(corr_disp, cmap="coolwarm", center=0, vmin=-1, vmax=1,
            annot=True, fmt=".2f", ax=ax, linewidths=0.3,
            annot_kws={"size": 7},
            cbar_kws={"label": "Correlation"})
ax.set_title("因子相关性矩阵 (Top 20)", color="white", pad=10)
plt.xticks(rotation=45, ha="right", fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()
fig.savefig(str(OUT / "fig_correlation.png"), dpi=120, bbox_inches="tight")
plt.close()
print("  saved: fig_correlation.png")


# ---- 图5: 统计分布 ----
print("图5: 统计分布...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5a: Sharpe 分布
ax = axes[0, 0]
sharpes = [results[n]["stats"].get("Sharpe Ratio", 0) for n in final_names]
colors_sh = ["#3ba55c" if s > 0 else "#f4212e" for s in sharpes]
ax.barh(range(len(sharpes)), sorted(sharpes), color="steelblue", alpha=0.8)
ax.axvline(0, color="white", linewidth=0.5)
ax.set_title("Sharpe 分布（按 Sharpe 排序）", color="white")
ax.set_xlabel("Sharpe Ratio")

# 5b: IC vs RankIC 散点
ax = axes[0, 1]
ics = [results[n]["stats"].get("IC", 0) for n in final_names]
rics = [results[n]["stats"].get("Rank IC", 0) for n in final_names]
sc = ax.scatter(ics, rics, c=sharpes, cmap="RdYlGn",
                vmin=-2, vmax=4, s=50, alpha=0.8, edgecolors="none")
ax.axhline(0, color="white", linewidth=0.5, alpha=0.3)
ax.axvline(0, color="white", linewidth=0.5, alpha=0.3)
ax.set_xlabel("IC")
ax.set_ylabel("Rank IC")
ax.set_title("IC vs RankIC", color="white")
plt.colorbar(sc, ax=ax, label="Sharpe")

# 5c: 年化收益 vs 最大回撤
ax = axes[1, 0]
ann_rets = [results[n]["stats"].get("Annualized Return (%)", 0) for n in final_names]
max_dds = [-results[n]["stats"].get("Max Drawdown (%)", 0) for n in final_names]
ax.scatter(max_dds, ann_rets, c=sharpes, cmap="RdYlGn",
           vmin=-2, vmax=4, s=50, alpha=0.8, edgecolors="none")
ax.axhline(0, color="white", linewidth=0.5, alpha=0.3)
ax.set_xlabel("Max Drawdown (%)")
ax.set_ylabel("Annualized Return (%)")
ax.set_title("收益 vs 回撤", color="white")
for i, nm in enumerate(final_names):
    if sharpes[i] > 2 or sharpes[i] < -1:
        short = nm.replace("factor_", "").replace("_flipped", "[F]")
        ax.annotate(short[:15], (max_dds[i], ann_rets[i]), fontsize=5, alpha=0.6)

# 5d: IC 月度均值分布
ax = axes[1, 1]
ic_means = monthly_ic.mean(axis=1)
ax.bar(ic_means.index, ic_means.values,
       color=["#3ba55c" if v > 0 else "#f4212e" for v in ic_means.values],
       alpha=0.8, width=20)
ax.axhline(0, color="white", linewidth=0.5)
ax.set_title("月度 IC 均值（所有 Top20 因子平均）", color="white")
ax.set_ylabel("Avg IC")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)

plt.tight_layout()
fig.savefig(str(OUT / "fig_statistics.png"), dpi=120, bbox_inches="tight")
plt.close()
print("  saved: fig_statistics.png")


# ---- 图6: IC_IR 时序 ----
print("图6: IC_IR 滚动...")
ic_ir = ic_ts_df.rolling(60, min_periods=20).mean() / ic_ts_df.rolling(60, min_periods=20).std()
fig, ax = plt.subplots(figsize=(16, 6))
top_ir = [nm for nm in final_names[:6] if nm in ic_ir.columns]
colors_ir = plt.cm.tab10(np.linspace(0, 1, len(top_ir)))
for nm, c in zip(top_ir, colors_ir):
    label = nm.replace("factor_", "").replace("_flipped", " [F]")
    ax.plot(ic_ir.index, ic_ir[nm], label=label, color=c, linewidth=1.5, alpha=0.85)
ax.axhline(0, color="white", linewidth=0.5, alpha=0.3)
ax.set_title("IC_IR 时序 (60日滚动, Top 6 因子)", color="white")
ax.set_ylabel("IC / IC_std")
ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig(str(OUT / "fig_icir.png"), dpi=120, bbox_inches="tight")
plt.close()
print("  saved: fig_icir.png")

print("\n所有图表生成完毕！")
print(f"  fig_ic_timeseries.png  - IC/RankIC 时序")
print(f"  fig_ic_heatmap.png     - IC 月度热力图")
print(f"  fig_nav_curves.png     - 净值曲线")
print(f"  fig_correlation.png    - 相关性矩阵")
print(f"  fig_statistics.png    - 统计分布")
print(f"  fig_icir.png          - IC_IR 滚动")
