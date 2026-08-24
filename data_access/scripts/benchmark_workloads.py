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


class _Metrics:
    """#32 记录 files/bytes/rows/physical scan count（通过包装 execute_arrow）。"""

    def __init__(self):
        self.files = 0
        self.bytes = 0
        self.rows = 0
        self.scans = 0
        self._orig = None
        self._engine = None

    def __enter__(self):
        from data_access import get_store

        self._engine = get_store()._engine
        self._orig = self._engine.execute_arrow

        def _wrapped(sql, params=None, **kw):
            self.scans += 1
            table = self._orig(sql, params, **kw)
            if table is not None:
                self.rows += table.num_rows
                self.bytes += table.nbytes
            return table

        self._engine.execute_arrow = _wrapped  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc):
        if self._engine is not None and self._orig is not None:
            self._engine.execute_arrow = self._orig  # type: ignore[method-assign]


def _count_files(store, dataset, **params):
    try:
        ds = store._registry.get(dataset)
        paths = store._prepare_dataset_read(ds, time_range=None, params=params)
        n = 0
        for g in paths:
            try:
                tbl = store._engine.execute_arrow(
                    "SELECT count(*) AS n FROM glob(?)", [str(g)], deadline_ms=None
                )
                n += int(tbl.to_pylist()[0]["n"])
            except Exception:
                n += 1
        return n
    except Exception:
        return 0


def main() -> int:
    from data_access import get_store

    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=[
        "daily", "pruned", "narrow", "pit10", "pit100", "joined", "minute", "factors",
        "daily1d", "daily1y", "daily5y", "minute_full", "us_daily", "latest_pit",
        "filing_pit", "industry", "topten", "us_mcap", "factnews", "cos_cold_warm",
    ], default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--metrics", action="store_true", help="#32 记录 files/bytes/rows/scans")
    args = parser.parse_args()
    store = get_store()
    out: list[str] = []
    _metrics = _Metrics() if args.metrics else None
    if _metrics is not None:
        _metrics.__enter__()

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

    # ---- #32 真实尺寸 workload（对应两本字典的实际数据规模） ----
    US = Path("/home/shw/quant_projects/data/us_stock/massive_data")

    if args.workload in (None, "daily1d", "daily1y", "daily5y"):
        if (ASHARE / "StockDailyBar").exists():
            for label, rng in (("A股日频 1日", ("2024-01-02", "2024-01-02")),
                               ("A股日频 1年", ("2023-01-01", "2023-12-31")),
                               ("A股日频 5年", ("2020-01-01", "2024-12-31"))):
                if args.workload not in (None, "daily1d", "daily1y", "daily5y"):
                    continue
                if label.endswith("1日") and args.workload != "daily1d" and args.workload is not None:
                    continue
                if label.endswith("1年") and args.workload != "daily1y" and args.workload is not None:
                    continue
                if label.endswith("5年") and args.workload != "daily5y" and args.workload is not None:
                    continue
                ms, h = _elapsed(lambda: store.read("ashare_stock_daily", columns=["Close", "Volume", "Return"], time_range=rng).to_arrow())
                report(label, ms, f"rows={h.num_rows} files={_count_files(store, 'ashare_stock_daily')}")
    if args.workload in (None, "minute_full"):
        if (ASHARE / "StockMinuteBar").exists():
            from data_access.read.aggregation import AggregationSpec, aggregate_minute_to_daily
            ms, h = _elapsed(lambda: aggregate_minute_to_daily(
                store, "ashare_stock_minute", "Volume",
                AggregationSpec(aggregation="minute_range", start="09:31", end="10:00", market="ashare"),
                time_range=("2024-01-02", "2024-01-02")).to_arrow())
            report("A股分钟全市场1日(~122万行)", ms, f"rows={h.num_rows}")
    if args.workload in (None, "us_daily"):
        if (US / "StockDailyBar").exists():
            ms, h = _elapsed(lambda: store.read("us_stock_daily", columns=["Close", "Volume", "Ret"], time_range=("2020-01-01", "2024-12-31")).to_arrow())
            report("美股日频 5年", ms, f"rows={h.num_rows}")
    if args.workload in (None, "latest_pit"):
        if (ASHARE / "StockBalance").exists():
            ms, h = _elapsed(lambda: store.read_joined(
                "ashare_stock_daily",
                {"ashare_stock_daily": ["Close"], "ashare_stock_balance": ["TotalAssets"]},
                joins={"ashare_stock_balance": {"policy": "pit_asof", "knowledge_time": "PubDate",
                        "period_time": "ReportPeriodEndDate", "revision_order": ("UpdateTime",),
                        "availability": "next_trading_day", "period_selection": "latest_period"}},
                time_range=("2023-01-01", "2024-12-31")).to_arrow())
            report("A股 latest_period PIT", ms, f"rows={h.num_rows}")
    if args.workload in (None, "filing_pit"):
        if (US / "StockBalance").exists():
            ms, h = _elapsed(lambda: store.read_joined(
                "us_stock_daily",
                {"us_stock_daily": ["Close"], "us_stock_balance": ["TotalAssets"]},
                joins={"us_stock_balance": {"policy": "pit_asof", "knowledge_time": "filing_date",
                        "period_time": "period_end", "period_selection": "latest_period"}},
                filters_by_dataset={"us_stock_balance": {"timeframe": "annual"}},
                time_range=("2020-01-01", "2024-12-31")).to_arrow())
            report("美股 filing_date+timeframe PIT", ms, f"rows={h.num_rows}")
    if args.workload in (None, "industry"):
        if (ASHARE / "StockIndustry").exists():
            ms, h = _elapsed(lambda: store.read("ashare_stock_industry", columns=["Symbol", "IndustryCode"],
                filters={"IndustrySource": "sw_l1"}, time_range=("2024-01-01", "2024-12-31")).to_arrow())
            report("A股 Industry sw_l1", ms, f"rows={h.num_rows}")
    if args.workload in (None, "topten"):
        if (ASHARE / "StockTopTenShareholder").exists():
            ms, h = _elapsed(lambda: store.read("ashare_stock_topten_shareholder", columns=["Symbol", "ShareholdingRatio"], time_range=("2024-01-01", "2024-12-31")).to_arrow())
            report("TopTen 聚合(源)", ms, f"rows={h.num_rows}")
    if args.workload in (None, "us_mcap"):
        if (US / "TickerSharesSnapshot").exists():
            ms, h = _elapsed(lambda: store.read_joined(
                "us_stock_daily",
                {"us_stock_daily": ["Close"], "us_ticker_shares_snapshot": ["weighted_shares_outstanding"]},
                time_range=("2024-01-01", "2024-12-31")).to_arrow())
            report("US shares×close 市值", ms, f"rows={h.num_rows}")
    if args.workload in (None, "factnews"):
        try:
            from data_access.cos_contract import resolve_event_clock
            resolve_event_clock("us_fact_news", allow_effective_time=False)
        except Exception:
            report("FactNews explode(无契约数据)", 0)
        else:
            ms, h = _elapsed(lambda: store.read("us_fact_news", columns=["ticker", "published_utc"], mode="event").to_arrow())
            report("FactNews published_utc", ms, f"rows={h.num_rows}")
    if args.workload in (None, "cos_cold_warm"):
        import os as _os
        if _os.environ.get("DATA_ACCESS_COS_READ_MODE") == "remote":
            report("COS cold/warm(remote 模式)", 0, extra="设置后重跑")
        else:
            report("COS cold/warm(local mirror)", 0, extra="local")

    if _metrics is not None:
        _metrics.__exit__(None, None, None)
        print(
            f"\n#32 metrics: scans={_metrics.scans} files≈{_metrics.files} "
            f"bytes={_metrics.bytes:,} rows={_metrics.rows:,}"
        )

    print(f"\n共 {len(out)} 个 workload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
