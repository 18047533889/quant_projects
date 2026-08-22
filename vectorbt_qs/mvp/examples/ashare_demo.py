"""
A 股组合回测 Demo

演示流程：
1. 加载行情数据（从本地 parquet 或 COS）
2. 生成模拟目标权重（随机 / 指数型 / 均线交叉）
3. 应用 A 股约束（停牌/涨跌停/手续费）
4. 执行回测
5. 输出绩效报告

运行方式:
    cd /home/wuhaohai/vectorbt_qs
    python -m vectorbt_qs.mvp.examples.ashare_demo
"""

import sys
import os
import pandas as pd
import numpy as np

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from vectorbt_qs.mvp.data.adapter import load_ashare_daily_bar
from vectorbt_qs.mvp.engine.runner import run_backtest, portfolio_report


# ============================================================
# 模拟目标权重生成器
# ============================================================

def random_weights(
    close: pd.DataFrame,
    n_stocks: int = 10,
    weight_per_stock: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    随机生成目标权重矩阵

    每天随机选 n_stocks 只做多，每只 weight_per_stock。
    固定种子保证可复现。

    参数
    ----
    n_stocks : 每天持仓股票数
    weight_per_stock : 每只权重
    seed : 随机种子
    """
    rng = np.random.default_rng(seed)
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    for i, date in enumerate(close.index):
        chosen = rng.choice(close.columns, size=min(n_stocks, len(close.columns)), replace=False)
        weights.loc[date, chosen] = weight_per_stock

    return weights


def index_like_weights(
    close: pd.DataFrame,
    base_weights: pd.Series,
    noise_std: float = 0.01,
    turnover_prob: float = 0.1,
    weight_per_stock: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成类似指数持仓 + 随机扰动的目标权重

    以 base_weights 为基准（如沪深300成分股权重），
    每天有 turnover_prob 概率换仓，加上噪声。

    参数
    ----
    base_weights : 基准权重（如等权: pd.Series(1/n, index=tickers)）
    noise_std : 权重噪声标准差
    turnover_prob : 每日换仓概率
    weight_per_stock : 每只目标权重（覆盖 base_weights 的值）
    seed : 随机种子
    """
    rng = np.random.default_rng(seed)
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    # 初始持仓
    current = base_weights[base_weights > 0].copy()
    current = current / current.sum() * (weight_per_stock * len(current))

    for i, date in enumerate(close.index):
        # 以 turnover_prob 概率换仓
        if rng.random() < turnover_prob or i == 0:
            chosen = rng.choice(
                close.columns,
                size=min(len(base_weights[base_weights > 0]), len(close.columns)),
                replace=False,
            )
            new_w = pd.Series(0.0, index=close.columns)
            new_w[chosen] = weight_per_stock
            # 添加噪声
            noise = rng.normal(0, noise_std, len(close.columns))
            new_w = new_w + pd.Series(noise, index=close.columns)
            new_w = new_w.clip(lower=0)
            current = new_w

        weights.loc[date] = current

    return weights


def ma_cross_weights(
    close: pd.DataFrame,
    fast: int = 10,
    slow: int = 30,
    weight: float = 0.05,
) -> pd.DataFrame:
    """
    均线交叉策略生成目标权重

    快线上穿慢线 → target_weight = +weight
    快线下穿慢线 → target_weight =  0.0
    """
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    weights[fast_ma > slow_ma] = weight
    weights = weights.iloc[slow:]  # 去掉均线未形成的前 N 天
    return weights


# ============================================================
# 第 2 步：运行回测
# ============================================================

def main():
    # ======== 配置 ========
    MARKET = "ashare"
    SYMBOLS = None  # None = 加载全部（建议 None，自动限制前 30 只）
    START = "2025-01-01"    # COS 真实数据范围
    END = "2025-06-30"
    INIT_CASH = 1_000_000.0

    # 数据源: "cos" = 直接从 COS 读（通过 clean-cos CLI） | "local" = 读本地 data/ 目录
    DATA_SOURCE = "cos"

    # 权重模式: "random" | "index_like" | "ma_cross"
    WEIGHT_MODE = "random"

    print("=" * 60)
    print(f"  A 股 MVP 回测 Demo [{WEIGHT_MODE}]")
    print(f"  数据源: {DATA_SOURCE} | {START} ~ {END}")
    print("=" * 60)

    # --- 切换数据源 ---
    if DATA_SOURCE == "cos":
        from mvp.data.adapter import set_data_root
        set_data_root("ashare", "cos://qs-cold/clean_data/ashare/lqtp_data")
        print("\n⚠️  首次从 COS 读取会下载并缓存到本地，请耐心等待 (~2-5 分钟/年)")
    # DATA_SOURCE == "local" → 使用默认的 data/ 目录

    # --- 加载数据 ---
    print("\n[1/4] 加载行情数据...")
    data = load_ashare_daily_bar(symbols=SYMBOLS, start=START, end=END)
    close = data["Close_adj"]  # 复权收盘价

    if SYMBOLS is None and close.shape[1] > 50:
        print(f"  ⚠ 标的过多 ({close.shape[1]} 只)，建议限制 SYMBOLS")
        # 取前 30 只演示
        close = close.iloc[:, :30]
        for key in ["is_suspend", "high_limit", "low_limit"]:
            if key in data and not data[key].empty:
                common_cols = data[key].columns.intersection(close.columns)
                data[key] = data[key][common_cols]

    print(f"  数据形状: {close.shape[0]} 天 × {close.shape[1]} 个标的")
    print(f"  日期范围: {close.index[0].date()} ~ {close.index[-1].date()}")

    # --- 生成目标权重 ---
    print(f"\n[2/4] 生成目标权重 (模式: {WEIGHT_MODE})...")

    if WEIGHT_MODE == "random":
        target_weights = random_weights(close, n_stocks=10, weight_per_stock=0.05)
    elif WEIGHT_MODE == "index_like":
        base_w = pd.Series(1.0 / len(close.columns), index=close.columns)
        target_weights = index_like_weights(close, base_w, noise_std=0.01, turnover_prob=0.15)
    elif WEIGHT_MODE == "ma_cross":
        target_weights = ma_cross_weights(close, weight=0.05)
        target_weights = target_weights.reindex(close.index).fillna(0.0)
    else:
        raise ValueError(f"未知权重模式: {WEIGHT_MODE}")

    avg_long = (target_weights > 0).sum(axis=1).mean()
    avg_short = (target_weights < 0).sum(axis=1).mean()
    print(f"  日均做多: {avg_long:.1f} 只, 日均做空: {avg_short:.1f} 只")
    print(f"  目标权重矩阵: {target_weights.shape}")

    # --- 执行回测 ---
    print("\n[3/4] 执行回测...")
    pf = run_backtest(
        MARKET,
        target_weights,
        config={
            "init_cash": INIT_CASH,
            "size_granularity": 100,          # A 股 100 股一手
            "slippage": 0.001,                # 滑点 0.1%
        },
    )

    # --- 输出结果 ---
    print("\n[4/4] 回测结果\n")
    stats = portfolio_report(pf)
    for k, v in stats.items():
        print(f"  {k:30s}: {v}")

    # --- 可选：画图（需额外安装 plotly）---
    try:
        pf.plot_value(title="A 股组合净值").show()
        pf.plot_drawdowns(title="回撤").show()
    except Exception as e:
        print(f"\n  ⚠ 绘图失败（可能未安装 plotly）: {e}")

    # --- 输出 trades 详情 ---
    trades = pf.trades.records_readable
    if len(trades) > 0:
        print(f"\n  共 {len(trades)} 笔交易")
        print(trades[["Entry Timestamp", "Exit Timestamp", "PnL", "Return"]].head(10))

    print("\n" + "=" * 60)
    print("  回测完成")
    print("=" * 60)

    return pf


if __name__ == "__main__":
    pf = main()
