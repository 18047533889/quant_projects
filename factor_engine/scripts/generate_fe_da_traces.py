#!/usr/bin/env python3
"""R30 §68 —— 生成 FE_DA_1000/10000_FACTOR_TRACE.json evidence。

证明 R30 核心成功指标：
    factor_count ↑        （1000 / 10000）
    physical_scan_count 不线性 ↑   （source_group_count << factor_count）

用 FE 批量 IR（FactorSourcePlan → FactorBatchDataPlan → FieldRequestCoalescer），
纯 Python 无数据读取。输出到 dataaccess/evidence/factor_engine/r30/。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "factor_engine"))

from factor_engine.planner.factor_source_plan import FactorSourcePlan
from factor_engine.planner.factor_batch_plan import plan_from_factors
from factor_engine.planner.field_request_coalescer import (
    FieldRequest,
    FieldRequestCoalescer,
    RequestCompatibilityKey,
)

# 20 个 dataset × 少量概念：10000 个因子共享少数字段。
DATASETS = {
    "ashare_stock_daily": ("price.close", "price.open", "price.high", "price.low",
                           "price.vwap", "volume", "amount"),
    "ashare_stock_income": ("financial.revenue", "financial.net_income.attributable"),
    "ashare_stock_balance": ("financial.total_assets", "financial.total_liabilities"),
    "ashare_stock_valuation": ("valuation.market_cap", "valuation.pe", "valuation.pb"),
    "ashare_industry": ("fundamental.industry"),
    "us_stock_daily": ("price.close", "price.vwap", "volume"),
    "us_stock_income": ("financial.revenue",),
}
MARKETS = {"ashare": 0, "us": 1}


def make_factor_plan(fid: int, market: str) -> FactorSourcePlan:
    datasets = list(DATASETS)
    # 每因子用 1-3 个 dataset 的字段，market 决定 dataset 集合。
    if market == "ashare":
        pools = datasets[:5]
    else:
        pools = [d for d in datasets if d.startswith("us")]
    used = pools[(fid * 7) % len(pools) : (fid * 7) % len(pools) + 2]
    if not used:
        used = [pools[0]]
    fields = []
    for ds in used:
        fields.extend(DATASETS[ds][:2])
    leaf = tuple(fields)
    return FactorSourcePlan(
        factor_id=f"F{fid:05d}",
        market=market,
        leaf_concepts=leaf,
        source_datasets=tuple(used),
        required_frequency="daily",
        required_grain=("trade_date", "instrument"),
        pit_requirements={"fidelity": "knowledge_date"},
        price_basis="backward_adjusted",
        aggregations=(),
        joins=(),
        coverage_requirements={"min_coverage": 0.8},
        timeframe="daily",
        universe="csi300" if market == "ashare" else "sp500",
        security_scope="research",
    )


def coalesce_from_factors(plans, time_range) -> dict:
    reqs = []
    for p in plans:
        for ds in p.source_datasets:
            reqs.append(
                FieldRequest(
                    key=RequestCompatibilityKey(
                        dataset=ds,
                        source_snapshot="S123",
                        market=p.market,
                        time_range=time_range,
                        instrument_universe=p._extra.get("universe"),
                        pit_policy=str(p.pit_requirements.get("fidelity")),
                        timeframe=p._extra.get("timeframe", "daily"),
                        security_digest="sec1",
                        frequency=p.required_frequency,
                        price_basis=p.price_basis,
                        aggregation_semantics="raw",
                    ),
                    fields=p.leaf_concepts,
                )
            )
    co = FieldRequestCoalescer()
    co.coalesce(reqs)
    return co.to_dict()


def trace_factor_count(n: int) -> dict:
    plans = [make_factor_plan(i, "ashare" if i % 4 else "us") for i in range(n)]
    time_range = ("2016-01-01", "2026-01-01")
    batch = plan_from_factors(
        plans, time_range=time_range, universe_id="csi300", market="ashare"
    )
    summary = batch.group_summary()
    co = coalesce_from_factors(plans, time_range)
    return {
        "factor_count": summary["factor_count"],
        "unique_concepts": summary["unique_concepts"],
        "source_group_count": summary["source_group_count"],
        "physical_scan_count": summary["physical_scan_count"],
        "scan_amplification": round(
            summary["physical_scan_count"] / max(summary["factor_count"], 1), 4
        ),
        "coalescer": {
            "input_requests": co["input_requests"],
            "merged_requests": co["merged_requests"],
            "saved_scans": co["saved_scans"],
            "field_union_total": co["field_union_total"],
        },
        "acceptance": {
            "source_group_count_lt_factor_count": summary["source_group_count"]
            < summary["factor_count"],
        },
    }


def main() -> None:
    out_dir = Path(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "dataaccess/evidence/factor_engine/r30",
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, name in ((1000, "FE_DA_1000_FACTOR_TRACE.json"),
                    (10000, "FE_DA_10000_FACTOR_TRACE.json")):
        trace = trace_factor_count(n)
        path = out_dir / name
        path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{name}: factor={trace['factor_count']} "
              f"groups={trace['source_group_count']} "
              f"scans={trace['physical_scan_count']} "
              f"amp={trace['scan_amplification']} "
              f"coalesce_saved={trace['coalescer']['saved_scans']}")
    print("Wrote evidence to", out_dir)


if __name__ == "__main__":
    main()
