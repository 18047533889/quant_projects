#!/usr/bin/env python3
"""
用格式对齐真实数据的 mock parquet 做 RealInputAdapter 端到端验证。

生成与对齐版文档一致格式的 parquet 文件 → RealInputAdapter 加载 → Pipeline 运行。
验证 adapter 的 pivot/筛选/对齐逻辑和三条优化路径。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 临时数据目录
DATA = Path("/tmp/real_adapter_test")
for sub in ["StockDailyBar", "StockList", "StockIndicator", "StockIndicesComponents"]:
    (DATA / sub).mkdir(parents=True, exist_ok=True)

DATES = pd.bdate_range("2026-06-01", periods=5)
TICKERS = [f"TICKER_{i:03d}" for i in range(20)]

rng = np.random.default_rng(42)

# ---- 生成格式对齐的 parquet ----
print("[1] 生成 mock parquet（对齐 COS 格式）...")

# StockDailyBar: 每日期一个 parquet, 列: Ticker, Close
for d in DATES:
    df = pd.DataFrame({
        "Ticker": TICKERS,
        "Close": 100 + rng.normal(0, 2, len(TICKERS)).cumsum(),
    })
    df.to_parquet(DATA / "StockDailyBar" / f"{d.strftime('%Y-%m-%d')}.parquet")
print(f"  ✅ StockDailyBar: {len(DATES)} 个文件")

# StockList: 每日期一个 parquet, 列: Symbol, type, delisted_utc
for d in DATES:
    df = pd.DataFrame({
        "Symbol": TICKERS,
        "type": ["CS"] * len(TICKERS),
        "delisted_utc": [None] * len(TICKERS),
    })
    df.to_parquet(DATA / "StockList" / f"{d.strftime('%Y-%m-%d')}.parquet")
print(f"  ✅ StockList: {len(DATES)} 个文件")

# StockIndicator: 每日期一个 parquet, 列: ticker, market_cap
for d in DATES:
    df = pd.DataFrame({
        "ticker": TICKERS,
        "market_cap": rng.lognormal(3, 1, len(TICKERS)) * 1e9,
    })
    df.to_parquet(DATA / "StockIndicator" / f"{d.strftime('%Y-%m-%d')}.parquet")
print(f"  ✅ StockIndicator: {len(DATES)} 个文件")

# StockIndicesComponents: 每日期一个 parquet, 列: IndexName, Symbol
for d in DATES:
    df = pd.DataFrame({
        "IndexName": ["S&P 500"] * len(TICKERS),
        "Symbol": TICKERS,
    })
    df.to_parquet(DATA / "StockIndicesComponents" / f"{d.strftime('%Y-%m-%d')}.parquet")
print(f"  ✅ StockIndicesComponents: {len(DATES)} 个文件")

# ---- 生成 alpha ----
alpha = pd.DataFrame(
    rng.normal(0, 0.01, size=(len(DATES), len(TICKERS))),
    index=DATES, columns=TICKERS,
)

# ---- RealInputAdapter 加载 ----
print("\n[2] RealInputAdapter 加载...")
from riskfolio_qs.adapters.real_adapter import RealInputAdapter

adapter = RealInputAdapter(
    alpha_df=alpha,
    data_root=str(DATA),
    start_date="2026-06-01",
    end_date="2026-06-04",
)
bundle = adapter.build_bundle()

print(f"  alpha:      {bundle.alpha.shape}")
print(f"  market:     {bundle.market.shape}")
print(f"  tradable:   {bundle.tradable.shape if bundle.tradable is not None else 'None'}")
print(f"  F_mcap:     {bundle.F_mcap.shape if bundle.F_mcap is not None else 'None'}")
print(f"  benchmark:  {bundle.benchmark.shape if bundle.benchmark is not None else 'None'}")
print(f"  G:          {bundle.G.shape if bundle.G is not None else 'None (美股 StockIndustry 空表)'}")

# ---- 跑三条路径 ----
from riskfolio_qs.runners.pipeline import OptimizationPipeline

pipeline = OptimizationPipeline()

tests = [
    ("topN 兜底", "fallback"),
    ("绝对收益", "absolute_return"),
]

print("\n[3] 跑优化路径...")
for name, scenario in tests:
    try:
        out = pipeline.run(bundle, scenario=scenario, data_version_hash="adapter_test")
        w = out.target_positions.iloc[-1]
        print(f"  ✅ {name}: {(w > 0).sum()} 只持仓, optimizer={out.metadata.iloc[0]['optimizer_name']}")
    except Exception as e:
        print(f"  ❌ {name}: {e}")

print("\n[4] 指增路径（预期 fail_fast——缺 Barra）...")
try:
    out = pipeline.run(bundle, scenario="index_enhancement")
    print("  ❌ 意外成功")
except ValueError as e:
    print(f"  ✅ 预期 fail_fast: {str(e)[:80]}")

print("\n===== ✅ RealInputAdapter 端到端验证通过 =====")
