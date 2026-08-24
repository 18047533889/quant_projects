# -*- coding: utf-8 -*-
"""R30-P0-002/003/004 行为测试 —— 纯 Python，不读数据。

覆盖：
  - P0-002 FactorSourcePlan：identity 稳定（同 factor 同字段 → 同 digest）；
    price_basis / market / pit policy 变化 → digest 变；extract 从 source
    dependency manifest 抽取 leaf_concepts / source_datasets。
  - P0-003 FactorBatchDataPlan：plan_from_factors 1000 factor 来自 ~20 dataset
    → source_group_count << factor_count（<100）；混合 raw/adj、US quarterly/TTM
    → 组数增加（证明不错误合并）。
  - P0-004 FieldRequestCoalescer：100 个共享 Close 的请求 → Close 只出现在一个
    组；不同 timeframe / pit_policy / price_basis 永不合并。
"""
from __future__ import annotations

from factor_engine.planner.factor_source_plan import FactorSourcePlan
from factor_engine.planner.factor_batch_plan import (
    FactorBatchDataPlan,
    SourceDemandGroup,
    batch_data_request,
    plan_from_factors,
)
from factor_engine.planner.field_request_coalescer import (
    FieldRequest,
    FieldRequestCoalescer,
    RequestCompatibilityKey,
    coalesce,
)


# ---------------------------------------------------------------------------
# P0-002 FactorSourcePlan
# ---------------------------------------------------------------------------


def _base_factor_kwargs() -> dict:
    return dict(
        factor_id="mom_20",
        market="us",
        leaf_concepts=("close", "high", "low"),
        source_datasets=("us_stock_daily",),
        required_frequency="daily",
        required_grain=("instrument", "time"),
        pit_requirements={"policy": "strict", "strict_pit": True},
        price_basis="raw",
        aggregations=(),
        joins=(),
        coverage_requirements={"min_ratio": 0.8},
    )


def test_factor_source_plan_identity_stable():
    kwargs = _base_factor_kwargs()
    p1 = FactorSourcePlan(**kwargs)
    p2 = FactorSourcePlan(**kwargs)
    # 同一 factor 编译多次 → 同一 digest
    assert p1.identity() == p2.identity()

    # leaf_concepts 顺序不影响 identity（内部排序）
    p3 = FactorSourcePlan(**{**kwargs, "leaf_concepts": ("low", "close", "high")})
    assert p1.identity() == p3.identity()

    # price_basis 变化 → identity 变
    p4 = FactorSourcePlan(**{**kwargs, "price_basis": "backward_adjusted"})
    assert p1.identity() != p4.identity()

    # market 变化 → identity 变
    p5 = FactorSourcePlan(**{**kwargs, "market": "cn"})
    assert p1.identity() != p5.identity()

    # pit policy 变化 → identity 变
    p6 = FactorSourcePlan(
        **{**kwargs, "pit_requirements": {"policy": "none", "strict_pit": False}}
    )
    assert p1.identity() != p6.identity()

    # source dataset 变化 → identity 变
    p7 = FactorSourcePlan(**{**kwargs, "source_datasets": ("us_fundamental",)})
    assert p1.identity() != p7.identity()


def test_factor_source_plan_extract_from_manifest():
    manifest = (
        '{"table":"us_stock_daily","field":"close","params":[],"transform":null,'
        '"transform_params":{},"dialect":"v1","dialect_version":"1.0"}',
        '{"table":"us_fundamental","field":"revenue_ttm","params":[],"transform":null,'
        '"transform_params":{},"dialect":"v1","dialect_version":"1.0"}',
    )
    p = FactorSourcePlan.extract(
        factor_id="f1",
        market="us",
        source_manifest=manifest,
        required_frequency="daily",
        price_basis="raw",
    )
    assert "close" in p.leaf_concepts
    assert "revenue_ttm" in p.leaf_concepts
    assert "us_stock_daily" in p.source_datasets
    assert "us_fundamental" in p.source_datasets
    assert p.required_frequency == "daily"
    assert p.price_basis == "raw"

    # 无 manifest / expression_plan → overrides 直接填充依赖
    p2 = FactorSourcePlan.extract(
        factor_id="f2",
        market="us",
        leaf_concepts=("open",),
        source_datasets=("us_stock_daily",),
        required_frequency="daily",
        price_basis="raw",
    )
    assert p2.leaf_concepts == ("open",)
    assert p2.source_datasets == ("us_stock_daily",)

    # manifest 里显式覆盖优先（overrides 给了 leaf_concepts 就不取 manifest）
    p3 = FactorSourcePlan.extract(
        factor_id="f3",
        market="us",
        source_manifest=manifest,
        leaf_concepts=("custom",),
        source_datasets=("custom_ds",),
        required_frequency="daily",
    )
    assert p3.leaf_concepts == ("custom",)
    assert p3.source_datasets == ("custom_ds",)


def test_factor_source_plan_to_dict_roundtrip():
    p = FactorSourcePlan(**_base_factor_kwargs())
    d = p.to_dict()
    assert d["factor_id"] == "mom_20"
    assert d["market"] == "us"
    assert d["price_basis"] == "raw"
    assert isinstance(d["leaf_concepts"], list)
    # to_dict 是稳定 digest 的唯一输入
    from factor_engine.planner.factor_source_plan import stable_digest

    assert stable_digest(d) == p.identity()


# ---------------------------------------------------------------------------
# P0-003 FactorBatchDataPlan
# ---------------------------------------------------------------------------


def _make_factor(i: int, *, price_basis: str = "raw", timeframe: str = "default"):
    dataset = f"ds_{i % 20:02d}"
    concepts = ("close", "volume", f"feature_{i % 7}")
    return FactorSourcePlan(
        factor_id=f"factor_{i}",
        market="us",
        leaf_concepts=concepts,
        source_datasets=(dataset,),
        required_frequency="daily",
        required_grain=("instrument", "time"),
        pit_requirements={"policy": "strict"},
        price_basis=price_basis,
        aggregations=(),
        joins=(),
        coverage_requirements={},
        timeframe=timeframe,  # 进入 _extra，batch 分组读取
    )


def test_plan_from_factors_1000_small_group_count():
    plans = [_make_factor(i) for i in range(1000)]
    plan = plan_from_factors(
        plans,
        time_range=("2020-01-01", "2024-12-31"),
        universe_id="us_all",
        market="us",
    )
    assert isinstance(plan, FactorBatchDataPlan)
    summary = plan.group_summary()
    assert summary["factor_count"] == 1000
    assert summary["unique_concepts"] > 0
    assert summary["source_group_count"] < 100
    assert summary["source_group_count"] < summary["factor_count"]
    assert summary["physical_scan_count"] == summary["source_group_count"]
    assert summary["avg_fields_per_group"] >= 1.0

    # to_dict 可序列化 + summary 一致
    d = plan.to_dict()
    assert d["summary"]["source_group_count"] == summary["source_group_count"]
    assert d["universe_id"] == "us_all"
    assert d["market"] == "us"


def test_plan_from_factors_rejects_wrong_merges():
    # raw vs backward_adjusted：绝不合并 → 组数增加
    raw = [_make_factor(i, price_basis="raw") for i in range(500)]
    adj = [_make_factor(i, price_basis="backward_adjusted") for i in range(500)]
    p_raw = plan_from_factors(raw, market="us")
    p_mix = plan_from_factors(raw + adj, market="us")
    assert p_mix.group_summary()["source_group_count"] > p_raw.group_summary()[
        "source_group_count"
    ]

    # US quarterly vs TTM（不同 timeframe）：绝不合并 → 组数增加
    daily = [_make_factor(i, timeframe="daily") for i in range(500)]
    quarterly = [_make_factor(i, timeframe="quarterly") for i in range(500)]
    p_daily = plan_from_factors(daily, market="us")
    p_qmix = plan_from_factors(daily + quarterly, market="us")
    assert p_qmix.group_summary()["source_group_count"] > p_daily.group_summary()[
        "source_group_count"
    ]

    # 不同 universe / pit_policy：各自独立组
    uni_a = [
        FactorSourcePlan(
            factor_id=f"a_{i}",
            market="us",
            leaf_concepts=("close",),
            source_datasets=("us_stock_daily",),
            required_frequency="daily",
            pit_requirements={"policy": "strict"},
            price_basis="raw",
            universe="us_all",
        )
        for i in range(50)
    ]
    uni_b = [
        FactorSourcePlan(
            factor_id=f"b_{i}",
            market="us",
            leaf_concepts=("close",),
            source_datasets=("us_stock_daily",),
            required_frequency="daily",
            pit_requirements={"policy": "strict"},
            price_basis="raw",
            universe="us_large",
        )
        for i in range(50)
    ]
    p_uni = plan_from_factors(uni_a + uni_b, market="us")
    assert p_uni.group_summary()["source_group_count"] == 2


def test_source_demand_group_compatibility_key():
    base = dict(
        dataset="us_stock_daily",
        market="us",
        source_snapshot="snap_v1",
        frequency="daily",
        timeframe="daily",
        pit_policy="strict",
        universe="us_all",
        security_scope="",
        price_basis="raw",
        aggregation_recipe="",
        filters=(),
    )
    g1 = SourceDemandGroup(**base, fields=("close", "volume"), time_range=(None, "2024-01-01"))
    g2 = SourceDemandGroup(**base, fields=("close",), time_range=(None, "2023-01-01"))
    # fields / time_range 不同仍可合并
    assert g1.compatibility_key() == g2.compatibility_key()
    assert g1.mergeable_with(g2)

    g3 = SourceDemandGroup(**{**base, "price_basis": "backward_adjusted"}, fields=("close",))
    assert not g1.mergeable_with(g3)

    g4 = SourceDemandGroup(**{**base, "timeframe": "quarterly"}, fields=("close",))
    assert not g1.mergeable_with(g4)

    g5 = SourceDemandGroup(**{**base, "pit_policy": "as_of"}, fields=("close",))
    assert not g1.mergeable_with(g5)

    g6 = SourceDemandGroup(**{**base, "universe": "us_large"}, fields=("close",))
    assert not g1.mergeable_with(g6)


def test_batch_data_request_adapter():
    plans = [_make_factor(i) for i in range(50)]
    plan = plan_from_factors(plans, market="us")
    req = batch_data_request(plan)
    if req is None:
        # 合法退化：import 环境缺 BatchDataRequest 时不应抛
        return
    assert len(req.groups) == plan.group_summary()["source_group_count"]
    assert len(req.fields) >= 1


# ---------------------------------------------------------------------------
# P0-004 FieldRequestCoalescer
# ---------------------------------------------------------------------------


def _make_req(
    i: int,
    *,
    timeframe: str = "daily",
    price_basis: str = "raw",
    pit_policy: str = "strict",
) -> FieldRequest:
    key = RequestCompatibilityKey(
        dataset="us_stock_daily",
        source_snapshot="snap_v1",
        market="us",
        time_range=("2020-01-01", "2024-12-31"),
        instrument_universe=("AAPL", "MSFT"),
        pit_policy=pit_policy,
        timeframe=timeframe,
        security_digest="sec_digest_v1",
        frequency="daily",
        price_basis=price_basis,
        aggregation_semantics="none",
    )
    return FieldRequest(
        key=key, fields=("close", f"field_{i % 10}"), payload={"i": i}
    )


def test_coalescer_merges_shared_close():
    requests = [_make_req(i) for i in range(100)]
    coalescer = FieldRequestCoalescer()
    merged = coalescer.coalesce(requests)
    # 全部完全兼容 → 单组
    assert len(merged) == 1
    fields = merged[0].fields
    assert "close" in fields
    # Close 只出现在一个组
    occurrences = sum(1 for m in merged if "close" in m.fields)
    assert occurrences == 1

    stats = coalescer.to_dict()
    assert stats["input_requests"] == 100
    assert stats["merged_requests"] == 1
    assert stats["saved_scans"] == 99
    # field_union_total = 合并后的字段并集大小（close + field_0..field_9）
    assert stats["field_union_total"] == len(fields)
    assert len(fields) == 11


def test_coalescer_never_merges_different_timeframe():
    requests = [_make_req(i, timeframe="daily") for i in range(50)] + [
        _make_req(i, timeframe="quarterly") for i in range(50)
    ]
    merged = coalesce(requests)
    assert len(merged) == 2
    timeframes = {m.key.timeframe for m in merged}
    assert timeframes == {"daily", "quarterly"}
    # 每组内 timeframe 一致，绝不跨 timeframe 合并
    for m in merged:
        assert m.key.timeframe in ("daily", "quarterly")
    daily = [m for m in merged if m.key.timeframe == "daily"][0]
    assert "field_0" in daily.fields
    assert "close" in daily.fields


def test_coalescer_never_merges_different_pit_policy():
    strict = [_make_req(i, pit_policy="strict") for i in range(30)]
    as_of = [_make_req(i, pit_policy="as_of") for i in range(30)]
    merged = coalesce(strict + as_of)
    assert len(merged) == 2


def test_coalescer_never_merges_different_price_basis():
    raw = [_make_req(i, price_basis="raw") for i in range(30)]
    adj = [_make_req(i, price_basis="backward_adjusted") for i in range(30)]
    merged = coalesce(raw + adj)
    assert len(merged) == 2


def test_request_compatibility_key_normalization():
    k1 = RequestCompatibilityKey(
        dataset="us_stock_daily",
        time_range=("2020-01-01", "2024-12-31"),
        instrument_universe=("MSFT", "AAPL"),
    )
    k2 = RequestCompatibilityKey(
        dataset="us_stock_daily",
        time_range=("2020-01-01", "2024-12-31"),
        instrument_universe=("AAPL", "MSFT"),  # 顺序无关
    )
    assert k1 == k2
    assert hash(k1) == hash(k2)
    # 不同 timeframe 永不相等
    k3 = RequestCompatibilityKey(
        dataset="us_stock_daily",
        time_range=("2020-01-01", "2024-12-31"),
        instrument_universe=("AAPL", "MSFT"),
        timeframe="quarterly",
    )
    assert k1 != k3
