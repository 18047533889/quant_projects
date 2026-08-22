"""
美股组合回测 Demo

演示流程：
1. 加载行情数据（PanelDaily）
2. 生成模拟目标权重（随机 / 指数型 / 动量多空）
3. 应用美股约束（SSR）
4. 执行回测
5. 对比多组策略

运行方式:
    cd /home/wuhaohai/vectorbt_qs
    python -m vectorbt_qs.mvp.examples.us_stock_demo
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from vectorbt_qs.mvp.data.adapter import load_us_daily_bar
from vectorbt_qs.mvp.engine.runner import run_backtest, portfolio_report, compare_reports


# ============================================================
# 模拟目标权重生成器
# ============================================================

def random_weights_long_short(
    close: pd.DataFrame,
    n_long: int = 8,
    n_short: int = 4,
    weight_per_stock: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    随机多空目标权重

    每天随机选 n_long 只做多，n_short 只做空。
    """
    rng = np.random.default_rng(seed)
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    all_cols = close.columns.tolist()

    for i, date in enumerate(close.index):
        shuffled = rng.permutation(all_cols)
        n = n_long + n_short
        chosen = shuffled[:min(n, len(all_cols))]
        longs = chosen[:min(n_long, len(chosen))]
        shorts = chosen[n_long:n_long + n_short] if len(chosen) > n_long else []
        weights.loc[date, longs] = weight_per_stock
        weights.loc[date, shorts] = -weight_per_stock

    return weights


def random_weights_long_only(
    close: pd.DataFrame,
    n_stocks: int = 12,
    weight_per_stock: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """随机纯做多"""
    rng = np.random.default_rng(seed)
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    for i, date in enumerate(close.index):
        chosen = rng.choice(close.columns, size=min(n_stocks, len(close.columns)), replace=False)
        weights.loc[date, chosen] = weight_per_stock

    return weights


def index_like_weights_ls(
    close: pd.DataFrame,
    noise_std: float = 0.01,
    turnover_prob: float = 0.15,
    n_long: int = 10,
    n_short: int = 3,
    weight_per_stock: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    指数型多空权重：稳定持仓 + 低换手 + 噪声
    """
    rng = np.random.default_rng(seed)
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    # 初始随机选择一批做多 + 做空
    all_cols = close.columns.tolist()
    long_pool = rng.choice(all_cols, size=n_long, replace=False).tolist()
    short_pool = rng.choice([c for c in all_cols if c not in long_pool], size=n_short, replace=False).tolist()

    for i, date in enumerate(close.index):
        if rng.random() < turnover_prob or i == 0:
            long_pool = rng.choice(all_cols, size=n_long, replace=False).tolist()
            remaining = [c for c in all_cols if c not in long_pool]
            short_pool = rng.choice(remaining, size=n_short, replace=False).tolist() if len(remaining) >= n_short else []

        for ticker in long_pool:
            noise = rng.normal(0, noise_std)
            weights.loc[date, ticker] = max(0, weight_per_stock + noise)
        for ticker in short_pool:
            noise = rng.normal(0, noise_std)
            weights.loc[date, ticker] = min(0, -weight_per_stock + noise)

    return weights


# ============================================================
# 主流程
# ============================================================

def main():
    MARKET = "us"
    SYMBOLS = None
    START = "2025-01-01"
    END = "2025-06-30"
    INIT_CASH = 1_000_000.0

    # 数据源: "cos" = 直接从 COS 读 | "local" = 读本地 data/ 目录
    DATA_SOURCE = "cos"

    # 权重模式: 'random' | 'index_like'
    WEIGHT_MODE = "random"

    print("=" * 60)
    print(f"  美股 MVP 回测 Demo [{WEIGHT_MODE}]")
    print(f"  数据源: {DATA_SOURCE} | {START} ~ {END}")
    print("=" * 60)

    # --- 切换数据源 ---
    if DATA_SOURCE == "cos":
        from mvp.data.adapter import set_data_root
        set_data_root("us_stock", "cos://qs-cold/clean_data/us_stock/massive_data")
        print("\n⚠️  首次从 COS 读取会下载并缓存到本地，请耐心等待 (~3-8 分钟/年)")

    # --- 加载数据 ---
    print("\n[1/4] 加载行情数据...")
    data = load_us_daily_bar(symbols=SYMBOLS, start=START, end=END, use_panel=True)
    close = data["close"]

    if SYMBOLS is None and close.shape[1] > 50:
        print(f"  ⚠ 标的过多 ({close.shape[1]} 只)，限制前 30 只")
        close = close.iloc[:, :30]

    print(f"  数据形状: {close.shape[0]} 天 × {close.shape[1]} 个标的")
    print(f"  日期范围: {close.index[0].date()} ~ {close.index[-1].date()}")

    # --- 生成多组目标权重 ---
    print(f"\n[2/4] 生成目标权重 (模式: {WEIGHT_MODE})...")

    if WEIGHT_MODE == "random":
        target_weights_ls = random_weights_long_short(close, n_long=8, n_short=4)
        target_weights_lo = random_weights_long_only(close, n_stocks=12)
    elif WEIGHT_MODE == "index_like":
        target_weights_ls = index_like_weights_ls(close, n_long=10, n_short=3, turnover_prob=0.15)
        target_weights_lo = random_weights_long_only(close, n_stocks=12)
    else:
        raise ValueError(f"未知权重模式: {WEIGHT_MODE}")

    for name, tw in [("多空", target_weights_ls), ("纯做多", target_weights_lo)]:
        l = (tw > 0).sum(axis=1).mean()
        s = (tw < 0).sum(axis=1).mean()
        print(f"  {name}: 日均做多 {l:.1f} 只, 日均做空 {s:.1f} 只")

    # --- 执行回测 ---
    print("\n[3/4] 执行回测...")
    config = {
        "init_cash": INIT_CASH,
        "slippage": 0.0005,
    }

    pf_ls = run_backtest(MARKET, target_weights_ls, config=config)
    pf_lo = run_backtest(MARKET, target_weights_lo, config=config)

    print(f"\n  多空策略: {len(pf_ls.trades.records_readable)} 笔交易, 最终净值 ${pf_ls.final_value():,.0f}")
    print(f"  纯做多:   {len(pf_lo.trades.records_readable)} 笔交易, 最终净值 ${pf_lo.final_value():,.0f}")

    # --- 输出结果 ---
    print("\n[4/4] 回测结果对比\n")
    comparison = compare_reports({"多空随机": pf_ls, "纯做多随机": pf_lo})
    print(comparison.to_string())

    # --- trades ---
    trades = pf_ls.trades.records_readable
    if len(trades) > 0:
        long_trades = trades[trades["Direction"] == "Long"]
        short_trades = trades[trades["Direction"] == "Short"]
        print(f"\n  多空策略交易明细:")
        print(f"    做多: {len(long_trades)} 笔, 胜率 {long_trades['Return'].gt(0).mean():.1%}")
        print(f"    做空: {len(short_trades)} 笔, 胜率 {short_trades['Return'].gt(0).mean():.1%}")

    print("\n" + "=" * 60)
    print("  回测完成")
    print("=" * 60)

    return pf_ls, pf_lo


if __name__ == "__main__":
    pf_ls, pf_lo = main()
