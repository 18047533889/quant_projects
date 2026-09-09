#!/usr/bin/env python3
"""Manual COS-backed end-to-end smoke test.

This module is intentionally side-effect free when imported so pytest and
other discovery tools can inspect it safely.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    """Download a small real-data sample and run the public pipeline."""
    data_dir = Path("/tmp/real_data_test")
    date = "2026-06-02"

    print("[1/5] 从 COS 拉取数据...")
    for table in ["StockDailyBar", "StockList"]:
        destination = data_dir / table
        destination.mkdir(parents=True, exist_ok=True)
        cos_path = (
            f"cos://qs-cold/clean_data/us_stock/massive_data/"
            f"{table}/{date}.parquet"
        )
        try:
            result = subprocess.run(
                [
                    "coscli",
                    "cp",
                    cos_path,
                    str(destination / f"{date}.parquet"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            print("  coscli 不可用；请安装并配置后重试。")
            return 2
        if result.returncode != 0:
            print(f"  {table} 拉取失败: {result.stderr.strip()}")
            return result.returncode
        print(f"  {table}: OK")

    print("[2/5] 加载数据到 InputBundle...")
    from riskfolio_qs.adapters.real_adapter import RealInputAdapter
    from riskfolio_qs.runners.pipeline import OptimizationPipeline

    bars = pd.read_parquet(data_dir / "StockDailyBar" / f"{date}.parquet")
    tickers = bars["Ticker"].tolist()[:100]
    alpha_df = pd.DataFrame(
        np.random.default_rng(42).normal(0, 0.01, size=(3, len(tickers))),
        index=pd.bdate_range("2026-06-01", periods=3),
        columns=tickers,
    )
    bundle = RealInputAdapter(
        alpha_df=alpha_df,
        data_root=str(data_dir),
        start_date="2026-06-01",
        end_date="2026-06-02",
    ).build_bundle()
    print(f"  alpha={bundle.alpha.shape}, market={bundle.market.shape}")

    pipeline = OptimizationPipeline()
    print("[3/5] topN 规则路径...")
    fallback = pipeline.run(
        bundle,
        scenario="fallback",
        data_version_hash="real_test_001",
    )
    print(f"  optimizer={fallback.metadata.iloc[0]['optimizer_name']}")

    print("[4/5] 绝对收益路径...")
    absolute = pipeline.run(
        bundle,
        scenario="absolute_return",
        data_version_hash="real_test_001",
    )
    print(f"  optimizer={absolute.metadata.iloc[0]['optimizer_name']}")

    print("[5/5] 指数增强能力路由...")
    try:
        enhanced = pipeline.run(
            bundle,
            scenario="index_enhancement",
            data_version_hash="real_test_001",
        )
        print(f"  optimizer={enhanced.metadata.iloc[0]['optimizer_name']}")
    except ValueError as exc:
        print(f"  无可用指数增强输入: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
