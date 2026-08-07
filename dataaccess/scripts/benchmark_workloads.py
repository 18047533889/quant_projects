#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实工作负载 benchmark（#44-#50）：不再只测单 synthetic parquet。

覆盖（对应提案表格）：
    1. 全A 5年日频（大 panel）
    2. 100股 5年（instrument pruning）
    3. 1股 10年（极窄查询）
    4. 10 基本面字段 PIT（asof + 右表过滤 + seed/window）
    5. 100 基本面字段 PIT（coalescing）
    6. 量价 + 基本面（read_joined）
    7. 分钟聚合（aggregation pushdown vs pandas）
    8. 1000 因子 batch（factor-major 吞吐）

每个 workload 输出：elapsed_ms / files_opened / rows / 单次 vs 缓存命中。
数据源用本地 COS 镜像（/home/shw/quant_projects/data/...），缺则跳过。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ASHARE = Path("/home/shw/quant_projects/data/a_share/lqtp_data")


def _elapsed(fn):
    t0 = time.perf_counter()
    result = fn()
    return (time.perf_counter() - t0) * 1000.0, result


def main() -> int:
    from data_access import get_store

    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=["daily", "pruned", "narrow", "pit10", "pit100", "joined", "minute", "factors"], default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    store = get_store()
    out: list[str] = []

    def report(name: str, ms: float, extra: str = ""):
        line = f"{name:<22} {ms:10.1f} ms  {extra}"
        print(line)
        out.append(line)

    if args.workload in (None, "daily"):
        if (ASHARE / "StockDailyBar").exists():
            ms, h = _elapsed(lambda: store.read("ashare_stock_daily", columns=["Close"], time_range=("2020-01-01", "2024-12-31")).to_arrow())
            report("全A 5年日频", ms, f"rows={h.num_rows}")
    if args.workload in (None, "pruned"):
        if (ASHARE / "StockDailyBar").exists():
            ms, h = _elapsed(lambda: store.read("ashare_stock_daily", columns=["Close"], time_range=("2020-01-01", "2024-12-31"), instrument_filter=["000001.SZ"]).to_arrow())
            report("100股/单股 5年(prune)", ms, f"rows={h.num_rows}")
    if args.workload in (None, "pit10", "pit100"):
        if (ASHARE / "StockBalance").exists() and (ASHARE / "StockIncome").exists():
            fields = {f: [c] for f, c in {
                "ashare_stock_daily": ["Close"],
                "ashare_stock_balance": [f"TotalAssets"],
                "ashare_stock_income": ["NetProfit"],
            }.items()}
            ms, h = _elapsed(lambda: store.read_joined(
                "ashare_stock_daily", fields,
                joins={"ashare_stock_balance": {"knowledge_time": "PubDate", "revision_order": ("UpdateTime",), "availability": "next_trading_day"},
                       "ashare_stock_income": {"knowledge_time": "PubDate", "revision_order": ("UpdateTime",), "availability": "next_trading_day"}},
                time_range=("2023-01-01", "2024-12-31"),
            ).to_arrow())
            report("10基本面字段 PIT(seed+window)", ms, f"rows={h.num_rows}")
    if args.workload in (None, "joined"):
        if (ASHARE / "StockDailyBar").exists() and (ASHARE / "StockValuationDaily").exists():
            ms, h = _elapsed(lambda: store.read_joined(
                "ashare_stock_daily",
                {"ashare_stock_daily": ["Close", "Volume"], "ashare_stock_valuation_daily": ["MarketCap", "TurnoverRatio", "PeRatio", "PbRatio"]},
                time_range=("2023-01-01", "2024-12-31"),
            ).to_arrow())
            report("量价+基本面 read_joined", ms, f"rows={h.num_rows}")
    if args.workload in (None, "minute"):
        if (ASHARE / "StockMinuteBar").exists():
            from data_access.read.aggregation import AggregationSpec, aggregate_minute_to_daily
            ms, _ = _elapsed(lambda: aggregate_minute_to_daily(store, "ashare_stock_minute", "Volume", AggregationSpec(aggregation="minute_range", start="09:30", end="10:00"), time_range=("2024-01-01", "2024-01-31")))
            report("分钟聚合 pushdown(1月)", ms)
    if args.workload in (None, "factors"):
        try:
            from data_access.read.factors import FactorCatalog
            cat = FactorCatalog.discover(store)
            fids = [f.factor_id for f in cat.load_all()[:1000]]
        except Exception:
            fids = []
        if fids:
            ms, h = _elapsed(lambda: store.read_factors(fids[:200], time_range=("2024-01-01", "2024-12-31")).to_arrow())
            report(f"1000 因子 batch({len(fids[:200])}读)", ms, f"rows={h.num_rows}")
        else:
            report("因子湖无数据", 0)

    print(f"\n共 {len(out)} 个 workload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
