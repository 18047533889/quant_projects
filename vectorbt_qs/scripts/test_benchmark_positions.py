"""测试 benchmarks 各版本 target_positions 作为仓位输入"""
import sys, os
_PROJ = r"D:\quantsociety\whh_local_workspace\vectorbt_qs"
sys.path.insert(0, os.path.dirname(_PROJ))
sys.path.insert(0, os.path.join(_PROJ, "vectorbt"))

import pandas as pd
from vectorbt_qs.mvp.engine.runner import run_backtest, portfolio_report

BENCH_DIR = r"D:\quantsociety\whh_local_workspace\benchmarks\gtja191_alpha191_meanvar_hist\results"

versions = {
    "v1 (等权)":     f"{BENCH_DIR}\\gtja191_alpha191_meanvar_hist_v1\\riskfolio\\target_positions.parquet",
    "v1_1 (优化)":   f"{BENCH_DIR}\\gtja191_alpha191_meanvar_hist_v1_1\\target_positions.parquet",
}

for label, path in versions.items():
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    tw = pd.read_parquet(path)
    n_stocks_day = (tw > 0).sum(axis=1)
    print(f"  权重: {tw.shape[0]}天 x {tw.shape[1]}只")
    print(f"  日均持仓: {n_stocks_day.mean():.0f} 只")

    pf = run_backtest(
        "ashare", tw,
        config={"init_cash": 10_000_000, "freq": "1D", "fees": 0, "fixed_fees": 0},
    )

    stats = portfolio_report(pf)
    keys = ["Start Value", "End Value", "Total Return [%]",
            "Sharpe Ratio", "Max Drawdown [%]", "Total Trades"]
    for k in keys:
        v = stats.get(k, "N/A")
        if isinstance(v, float):
            print(f"  {k}: {v:,.2f}")
        else:
            print(f"  {k}: {v}")
