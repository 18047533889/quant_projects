"""测试 vectorbt_qs 目标仓位模式回测"""
import sys, os
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_PROJ))
sys.path.insert(0, os.path.join(_PROJ, "vectorbt"))

import pandas as pd
import numpy as np

from vectorbt_qs.mvp.data.adapter import load_market_data
from vectorbt_qs.mvp.engine.runner import run_backtest, portfolio_report

# 1. 加载数据
print("=== 1. 加载行情 ===")
data = load_market_data(
    "ashare",
    symbols=["000001.SZ", "000002.SZ", "600519.SH", "000858.SZ", "601318.SH"],
    start="2025-01-01",
    end="2026-03-31",
)
close = data["close"]
print(f"close: {close.shape[0]} 天 x {close.shape[1]} 只")

# 2. 生成模拟目标权重（等权二选一，每天换仓）
print("\n=== 2. 生成目标权重 ===")
target_weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
rng = np.random.default_rng(42)
for i, date in enumerate(close.index):
    chosen = rng.choice(close.columns, size=min(2, len(close.columns)), replace=False)
    target_weights.loc[date, chosen] = 0.25
print(f"非零权重条目: {(target_weights != 0).sum().sum()}")

# 3. 执行回测（使用统一入口）
print("\n=== 3. 执行回测 ===")
pf = run_backtest(
    "ashare", target_weights,
    config={"init_cash": 10_000_000.0, "freq": "1D"},
)
print("回测完成!")

# 4. 绩效报告
print("\n=== 4. 绩效报告 ===")
stats = portfolio_report(pf)
keys = ["Start Value", "End Value", "Total Return [%]",
        "Sharpe Ratio", "Max Drawdown [%]", "Total Trades"]
for k in keys:
    v = stats.get(k, "N/A")
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")
