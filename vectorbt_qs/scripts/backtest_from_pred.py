"""
pred.parquet 预测 → 目标权重 → vectorbt_qs 回测

方案：每天选 pred 最高的 Top-50 股票，等权 2% 做多。

用法：
    cd vectorbt_qs
    $env:DATA_ACCESS_SKIP_COS_MIRROR = "1"
    $env:ASHARE_PARQUET_ROOT = "D:\quantsociety\whh_local_workspace\lqtp_data"
    python scripts/backtest_from_pred.py
"""
import sys, os
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_PROJ))
sys.path.insert(0, os.path.join(_PROJ, "vectorbt"))

import pandas as pd
import numpy as np
from pathlib import Path

from vectorbt_qs.mvp.data.adapter import load_market_data
from vectorbt_qs.mvp.engine.runner import run_backtest, portfolio_report

# ============================================================
# 配置
# ============================================================
MODEL = "OLS"          # OLS / Ridge / Lasso / ElasticNet / PLS / IC_weighted / ICIR_weighted / zscore_weighted
HORIZON = "h5"         # h1 / h5 / h10 / h20
TOP_N = 50             # 每天选 Top-N 股票
WEIGHT_PER_STOCK = 0.02  # 每只等权 2%（50×2%=100%）
INIT_CASH = 10_000_000   # 初始资金 1000 万
OUTPUTS_DIR = Path(_PROJ).parent / "outputs"

# ============================================================
# 1. 读取预测数据 → pivot 宽表
# ============================================================
print(f"=== 1. 读取预测: {MODEL}_{HORIZON} ===")
pred_file = OUTPUTS_DIR / f"{MODEL}_{HORIZON}_pred.parquet"
pred_long = pd.read_parquet(pred_file)
print(f"  原始: {pred_long.shape[0]} 行, {pred_long['Symbol'].nunique()} 只股票")

# pivot: TradeDate × Symbol → pred
pred_wide = pred_long.pivot(index="TradeDate", columns="Symbol", values="pred")
pred_wide = pred_wide.sort_index().sort_index(axis=1)
print(f"  宽表: {pred_wide.shape[0]} 天 × {pred_wide.shape[1]} 只")
print(f"  日期范围: {pred_wide.index.min().date()} ~ {pred_wide.index.max().date()}")

# ============================================================
# 2. 生成目标权重：每日 Top-50 pred，等权 2%
# ============================================================
print(f"\n=== 2. 生成目标权重 (Top-{TOP_N}, 等权 {WEIGHT_PER_STOCK*100:.0f}%) ===")

target_weights = pd.DataFrame(0.0, index=pred_wide.index, columns=pred_wide.columns)

for date in pred_wide.index:
    row = pred_wide.loc[date].dropna()
    if len(row) == 0:
        continue
    top_symbols = row.nlargest(TOP_N).index
    target_weights.loc[date, top_symbols] = WEIGHT_PER_STOCK

nonzero_count = (target_weights != 0).sum(axis=1)
print(f"  日均持仓: {nonzero_count.mean():.1f} 只")
print(f"  总调仓次数: {(target_weights != 0).sum().sum()}")

# ============================================================
# 3. 加载行情（与预测日期对齐）
# ============================================================
print("\n=== 3. 加载行情 ===")
start = str(target_weights.index.min().date())
end = str(target_weights.index.max().date())

# 取所有在权重中出现过的股票
all_symbols = sorted(target_weights.columns[(target_weights != 0).any(axis=0)].tolist())
print(f"  需要行情: {len(all_symbols)} 只, {start} ~ {end}")

data = load_market_data("ashare", symbols=all_symbols, start=start, end=end)
close = data["close"]
print(f"  close: {close.shape[0]} 天 × {close.shape[1]} 只")

# ============================================================
# 4. 对齐 + 回测
# ============================================================
print("\n=== 4. 执行回测 ===")
pf = run_backtest(
    "ashare", target_weights,
    symbols=all_symbols,
    start=start, end=end,
    config={
        "init_cash": INIT_CASH,
        "freq": "1D",
        "slippage": 0.001,
        "fees": 0.0,
        "fixed_fees": 0.0,
    },
)
print("回测完成!")

# ============================================================
# 5. 输出结果（CSV + 图表）
# ============================================================
RESULT_PREFIX = f"backtest_{MODEL}_{HORIZON}"
print(f"\n=== 5. 保存结果到 {OUTPUTS_DIR} ===")

stats = portfolio_report(pf)

# --- 5a. 绩效汇总 CSV ---
summary = pd.DataFrame({"指标": stats.index, "数值": stats.values})
summary_path = OUTPUTS_DIR / f"{RESULT_PREFIX}_summary.csv"
summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
print(f"  绩效汇总: {summary_path}")

# --- 5b. 每日权益 CSV ---
equity = pf.value()
equity.name = "equity"
equity_path = OUTPUTS_DIR / f"{RESULT_PREFIX}_equity.csv"
equity.to_csv(equity_path, encoding="utf-8-sig")
print(f"  每日权益: {equity_path}")

# --- 5c. 权益曲线 + 回撤图 ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                         gridspec_kw={"height_ratios": [3, 1]})

# 上图：权益曲线
ax1 = axes[0]
ax1.plot(equity.index, equity.values / 1e4, color="#1f77b4", linewidth=1.0,
         label=f"Equity ({equity.iloc[-1]/1e4:.0f}万)")
ax1.set_ylabel("权益 (万元)", fontsize=11)
ax1.set_title(
    f"{MODEL}_{HORIZON}  Top-{TOP_N} 等权 {WEIGHT_PER_STOCK*100:.0f}%  |  "
    f"初始 {INIT_CASH/1e4:.0f}万  |  "
    f"收益 {stats.get('Total Return [%]', 0):+.1f}%  |  "
    f"Sharpe {stats.get('Sharpe Ratio', 0):.2f}",
    fontsize=12,
)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

# 下图：回撤
ax2 = axes[1]
dd = (equity / equity.cummax() - 1) * 100
ax2.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.4,
                 label=f"MaxDD: {stats.get('Max Drawdown [%]', 0):.1f}%")
ax2.set_ylabel("回撤 (%)", fontsize=11)
ax2.set_xlabel("日期", fontsize=11)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax2.legend(loc="lower left")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
chart_path = OUTPUTS_DIR / f"{RESULT_PREFIX}_equity.png"
fig.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  权益曲线: {chart_path}")

# --- 5d. 控制台摘要 ---
print(f"\n{'='*50}")
print(f"  {MODEL}_{HORIZON}  Top-{TOP_N} 等权 {WEIGHT_PER_STOCK*100:.0f}%")
print(f"  初始资金: {INIT_CASH:,.0f}")
for k in ["Start Value", "End Value", "Total Return [%]",
          "Sharpe Ratio", "Max Drawdown [%]", "Total Trades",
          "Win Rate [%]", "Expectancy"]:
    v = stats.get(k, "N/A")
    if isinstance(v, float):
        print(f"  {k}: {v:,.2f}")
    else:
        print(f"  {k}: {v}")
print(f"{'='*50}")
