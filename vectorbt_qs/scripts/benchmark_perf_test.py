"""
benchmark 全量回测性能测试：B001 gtja191_alpha191_topn_v1
2294 天 × 5122 只股票，Top-50 等权

输出: vectorbt_qs/docs/benchmark_performance_report.md
"""
import sys, os, time
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_PROJ))
sys.path.insert(0, os.path.join(_PROJ, "vectorbt"))

import pandas as pd
from pathlib import Path
from datetime import datetime

from vectorbt_qs.mvp.data.adapter import load_market_data
from vectorbt_qs.mvp.engine.runner import run_backtest, portfolio_report

# ============================================================
BENCHMARK_PATH = (
    r"D:\quantsociety\whh_local_workspace\benchmarks\gtja191_alpha191_topn"
    r"\results\gtja191_alpha191_topn_v1\target_positions.parquet"
)
DOCS_DIR = Path(_PROJ) / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = DOCS_DIR / "benchmark_performance_report.md"

timings = {}
t0_total = time.time()

# ============================================================
# 1. 读取 target_positions
# ============================================================
t0 = time.time()
tw = pd.read_parquet(BENCHMARK_PATH)
timings["读取 target_positions"] = time.time() - t0
n_stocks_day = (tw > 0).sum(axis=1)

# ============================================================
# 2. 回测（内部自动加载行情）
# ============================================================
t0 = time.time()
pf = run_backtest(
    "ashare", tw,
    config={"init_cash": 10_000_000, "freq": "1D", "fees": 0, "fixed_fees": 0},
)
timings["回测 (含行情加载)"] = time.time() - t0

# ============================================================
# 3. 绩效
# ============================================================
t0 = time.time()
stats = portfolio_report(pf)
timings["绩效报告"] = time.time() - t0

timings["总计"] = time.time() - t0_total

start = str(tw.index.min().date())
end = str(tw.index.max().date())

# ============================================================
# 5. 生成报告
# ============================================================
lines = []
lines.append("# vectorbt_qs Benchmark 性能报告")
lines.append("")
lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
lines.append(f"> 环境: Windows, Python 3.11, conda whh_local_workspace")
lines.append("")
lines.append("## 测试对象")
lines.append("")
lines.append("| 项目 | 值 |")
lines.append("|---|---|")
lines.append(f"| Benchmark | B001 gtja191_alpha191_topn_v1 |")
lines.append(f"| 信号源 | GTJA191 Alpha191 |")
lines.append(f"| 优化器 | topn_long_only_equal_weight |")
lines.append(f"| 权重矩阵 | {tw.shape[0]} 天 × {tw.shape[1]} 只 |")
lines.append(f"| 日期范围 | {start} ~ {end} |")
lines.append(f"| 日均持仓 | {n_stocks_day.mean():.0f} 只 (min={n_stocks_day.min()}, max={n_stocks_day.max()}) |")
lines.append("")
lines.append("## 耗时分析")
lines.append("")
lines.append("| 步骤 | 耗时 (秒) | 占比 |")
lines.append("|---|---|---|")
for step, elapsed in timings.items():
    pct = elapsed / timings["总计"] * 100
    lines.append(f"| {step} | {elapsed:.1f} | {pct:.1f}% |")
lines.append("")
lines.append("## 回测绩效")
lines.append("")
lines.append("| 指标 | 值 |")
lines.append("|---|---|")
keys = [
    "Start Value", "End Value", "Total Return [%]",
    "Sharpe Ratio", "Max Drawdown [%]", "Total Trades",
    "Win Rate [%]", "Expectancy", "Annual Return [%]",
    "Annual Volatility [%]", "Calmar Ratio",
]
for k in keys:
    v = stats.get(k, "N/A")
    if isinstance(v, float):
        lines.append(f"| {k} | {v:,.2f} |")
    else:
        lines.append(f"| {k} | {v} |")

lines.append("")
lines.append("## 系统信息")
lines.append("")
lines.append(f"| 项目 | 值 |")
lines.append(f"|---|---|")
lines.append(f"| 行情数据源 | data_access (DuckDB) + lqtp_data 本地 parquet |")
lines.append(f"| 行情加载列数 | 11 列 (OHLCV + 涨跌停 + 复权因子) |")
lines.append(f"| 数据量 | ~2300 个 parquet 文件, ~6GB |")
lines.append(f"| vectorbt 版本 | 1.x (源码) |")
lines.append(f"| 约束层 | A股停牌过滤 + 涨跌停价格裁剪 + 费率矩阵 |")

report = "\n".join(lines)
REPORT_PATH.write_text(report, encoding="utf-8")
print(report)
print(f"\n报告已保存: {REPORT_PATH}")
