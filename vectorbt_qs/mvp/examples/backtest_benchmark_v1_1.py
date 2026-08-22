"""
benchmark gtja191_alpha191_meanvar_hist_v1_1 回测 + 可视化输出

输入: benchmarks/.../v1_1/target_positions.parquet (mean-variance 优化权重)
输出: vectorbt_qs/examples/output/  (PNG 图表 + CSV 报告)
"""
import sys, os
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(_PROJ))
sys.path.insert(0, os.path.join(_PROJ, "vectorbt"))

import pandas as pd
import numpy as np
from pathlib import Path

from vectorbt_qs.mvp.engine.runner import run_backtest, portfolio_report

# ============================================================
# 配置
# ============================================================
BENCHMARK_PATH = (
    r"D:\quantsociety\whh_local_workspace\benchmarks\gtja191_alpha191_meanvar_hist"
    r"\results\gtja191_alpha191_meanvar_hist_v1_1\target_positions.parquet"
)
OUTPUT_DIR = Path(_PROJ) / "examples" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INIT_CASH = 10_000_000
CONFIG = {"init_cash": INIT_CASH, "freq": "1D", "fees": 0, "fixed_fees": 0}

# ============================================================
# 1. 加载仓位
# ============================================================
print("=" * 60)
print("  gtja191 alpha191 meanvar_hist v1_1 回测")
print("=" * 60)

tw = pd.read_parquet(BENCHMARK_PATH)
n_stocks_day = (tw > 0).sum(axis=1)
print(f"  权重矩阵: {tw.shape[0]} 天 x {tw.shape[1]} 只")
print(f"  日期范围: {tw.index.min().date()} ~ {tw.index.max().date()}")
print(f"  日均持仓: {n_stocks_day.mean():.0f} 只 (min={n_stocks_day.min()}, max={n_stocks_day.max()})")
print(f"  权重范围: [{tw.values[tw.values > 0].min():.4f}, {tw.values.max():.4f}]")

# ============================================================
# 2. 回测
# ============================================================
print("\n[回测中...]")
pf = run_backtest("ashare", tw, config=CONFIG)
print("回测完成!")

# ============================================================
# 3. 绩效报告 → CSV
# ============================================================
stats = portfolio_report(pf)
stats.to_csv(OUTPUT_DIR / "performance_stats.csv", header=True)
print(f"\n绩效报告已保存: {OUTPUT_DIR / 'performance_stats.csv'}")

keys_show = [
    "Start Value", "End Value", "Total Return [%]",
    "Sharpe Ratio", "Max Drawdown [%]", "Total Trades",
    "Win Rate [%]", "Expectancy", "Avg Winning Trade", "Avg Losing Trade",
]
print("\n关键指标:")
for k in keys_show:
    v = stats.get(k, "N/A")
    if isinstance(v, float):
        print(f"  {k}: {v:,.2f}")
    else:
        print(f"  {k}: {v}")

# ============================================================
# 4. 可视化 → PNG
# ============================================================
print("\n[生成图表...]")

# 设置中文字体
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# --- 4a. 净值曲线 + 回撤 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                gridspec_kw={"height_ratios": [3, 1]})

value = pf.value()
ax1.plot(value.index, value.values / INIT_CASH, color="#1f77b4", linewidth=1)
ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
ax1.set_ylabel("Net Value (x Initial)", fontsize=11)
ax1.set_title("gtja191 alpha191 meanvar_hist v1_1 — Equity Curve", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3)

dd = pf.drawdown()
ax2.fill_between(dd.index, dd.values * 100, 0, color="#d62728", alpha=0.6)
ax2.set_ylabel("Drawdown %", fontsize=11)
ax2.set_xlabel("Date", fontsize=11)
ax2.grid(True, alpha=0.3)

ret = stats.get("Total Return [%]", 0)
dd_max = stats.get("Max Drawdown [%]", 0)
sharpe = stats.get("Sharpe Ratio", 0)
fig.suptitle(
    f"Return: {ret:.1f}%  |  Max DD: {dd_max:.1f}%  |  Sharpe: {sharpe:.2f}",
    fontsize=11, color="gray", y=0.99,
)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "equity_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  净值曲线: {OUTPUT_DIR / 'equity_curve.png'}")

# --- 4b. 月度收益热力图 ---
returns_series = pf.returns()
if isinstance(returns_series, pd.DataFrame):
    returns_series = returns_series.iloc[:, 0]
monthly_returns = returns_series.resample("ME").apply(lambda x: (1 + x).prod() - 1)
monthly_table = monthly_returns.to_frame("return")
monthly_table["year"] = monthly_table.index.year
monthly_table["month"] = monthly_table.index.month
pivot = monthly_table.pivot(index="year", columns="month", values="return") * 100

fig, ax = plt.subplots(figsize=(14, len(pivot) * 0.5 + 2))
im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-15, vmax=15)
ax.set_xticks(range(12))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
ax.set_yticks(range(len(pivot)))
ax.set_yticklabels(pivot.index)
for i in range(len(pivot)):
    for j in range(12):
        v = pivot.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 8 else "black")
ax.set_title("Monthly Returns (%) — gtja191 alpha191 meanvar_hist v1_1", fontsize=14, fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "monthly_returns.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  月度收益: {OUTPUT_DIR / 'monthly_returns.png'}")

# --- 4c. 年度收益柱状图 ---
yearly = monthly_table.groupby("year")["return"].apply(lambda x: (1 + x).prod() - 1) * 100

fig, ax = plt.subplots(figsize=(12, 5))
colors = ["#d62728" if v < 0 else "#2ca02c" for v in yearly.values]
ax.bar(yearly.index.astype(str), yearly.values, color=colors, edgecolor="white")
ax.axhline(y=0, color="gray", linewidth=0.8)
for i, v in enumerate(yearly.values):
    ax.text(i, v + (2 if v >= 0 else -4), f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax.set_title("Annual Returns — gtja191 alpha191 meanvar_hist v1_1", fontsize=14, fontweight="bold")
ax.set_ylabel("Return %", fontsize=11)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "annual_returns.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  年度收益: {OUTPUT_DIR / 'annual_returns.png'}")

# --- 4d. 持仓数量变化 ---
fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(n_stocks_day.index, n_stocks_day.values, alpha=0.4, color="#1f77b4")
ax.plot(n_stocks_day.index, n_stocks_day.values, linewidth=0.8, color="#1f77b4")
ax.axhline(y=n_stocks_day.mean(), color="gray", linestyle="--", alpha=0.5,
           label=f"Mean: {n_stocks_day.mean():.0f}")
ax.set_ylabel("Positions", fontsize=11)
ax.set_title("Daily Position Count — gtja191 alpha191 meanvar_hist v1_1", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "position_count.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  持仓数量: {OUTPUT_DIR / 'position_count.png'}")

# ============================================================
# 5. 汇总
# ============================================================
print(f"\n{'='*60}")
print(f"  全部输出: {OUTPUT_DIR}")
print(f"    equity_curve.png      — 净值曲线 + 回撤")
print(f"    monthly_returns.png   — 月度收益热力图")
print(f"    annual_returns.png    — 年度收益柱状图")
print(f"    position_count.png    — 每日持仓数量")
print(f"    performance_stats.csv — 完整绩效指标")
print(f"{'='*60}")
