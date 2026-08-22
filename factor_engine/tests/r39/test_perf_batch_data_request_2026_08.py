# -*- coding: utf-8 -*-
"""R39-P0-PERF-002/003/004 行为测试 —— 真实探针，不是 grep/module-existence。

覆盖：
  - PERF-002：ColumnSourceBinding 直接从列解析出 typed scope，secondary 字段
    正确拆分；旧「market 塞进 SourceScopeId(dataset=...)」类型混用回归消失。
  - PERF-003：BatchSourceResolver per-scope adapter；每个 dataset 独立
    ScanCost evidence（两个 adapter → 两个 cost source，不共享 anchor estimator）。
  - PERF-004：TimeRange (None, end) / (start, None) 原样保留 + 产生 prune bound。
  - PERF-002 hard-gate：SOURCE_REF_WITHOUT_TYPED_SOURCE_BINDING == 0。
"""
from __future__ import annotations

from types import SimpleNamespace

from planner.batch_data_request import (
    BatchSourceResolver,
    ScanCostUnavailable,
    build_batch_data_request,
)
from planner.logical_plan import PlanNode
from planner.physical_factor_dag import SourceScopeId
from planner.source_binding import (
    ColumnSourceBinding,
    TimeRange,
    discover_column_source_bindings,
)
from api.source_ref import (
    encode_source_ref,
    make_source_ref,
)


def _sref(table, field, *, dataset=None, market=None):
    """构造一个可解码的 SourceRef 列名。"""
    return encode_source_ref(
        make_source_ref(table, field, dataset=dataset, market=market)
    )


def _column(name):
    return PlanNode(op="column", attrs={"name": name}, inputs=())


def _plan(*cols):
    return PlanNode(
        op="ts_mean",
        attrs={"window": 5},
        inputs=tuple(_column(c) for c in cols),
    )


def _dag(root_plan):
    return SimpleNamespace(
        shared_nodes={},
        roots=(SimpleNamespace(factor_name="f1", root=root_plan),),
    )


class _CostAdapter:
    """per-scope cost adapter probe：记录每次 estimate_scan_cost 的 kwargs。"""

    def __init__(self, dataset, storage_kind="data_access"):
        self.dataset = dataset
        self.storage_kind = storage_kind
        self.calls: list[dict] = []

    def estimate_scan_cost(self, **kw):
        self.calls.append(dict(kw))
        return SimpleNamespace(
            selected_bytes=100,
            projection_bytes=50,
            estimated_rows=10,
            file_count=1,
            remote=False,
            instrument_count=1,
        )


class _AnchorSource:
    dataset = "ashare_daily"
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    instrument_filter = ["000001"]

    def __init__(self, *, adapters=None):
        self.calls: list[dict] = []
        self.fund = _CostAdapter("fundamental_quarterly")
        self.industry = _CostAdapter("stock_industry")
        self.sources = {
            "fund": self.fund,
            "industry": self.industry,
        }

    def estimate_scan_cost(self, **kw):
        self.calls.append(dict(kw))
        return SimpleNamespace(
            selected_bytes=1000,
            projection_bytes=500,
            estimated_rows=100,
            file_count=3,
            remote=False,
            instrument_count=2,
        )


def test_column_source_binding_typed_scope_and_field_split():
    ref_fund = _sref("StockIncome", "NetProfit", dataset="fundamental_quarterly", market="ashare")
    ref_ind = _sref("StockIndustry", "IndustrySource", dataset="stock_industry", market="ashare")
    req = build_batch_data_request(
        _AnchorSource(),
        analyses={},
        dag=_dag(_plan("close", ref_fund, ref_ind)),
        ctx=SimpleNamespace(market="ashare"),
    )

    groups = {g.dataset: g for g in req.groups}
    assert set(groups) == {"ashare_daily", "fundamental_quarterly", "stock_industry"}

    anchor = groups["ashare_daily"]
    assert anchor.fields == ("close",)
    assert isinstance(anchor.source_scope, SourceScopeId)

    fund = groups["fundamental_quarterly"]
    assert fund.fields == ("NetProfit",)
    assert fund.source_scope.dataset == "fundamental_quarterly"
    assert fund.source_scope.market == "ashare"

    ind = groups["stock_industry"]
    assert ind.fields == ("IndustrySource",)
    assert ind.source_scope.dataset == "stock_industry"


def test_secondary_group_dataset_is_real_not_market_regression():
    """旧 bug：market 字符串被塞进 ``SourceScopeId(dataset=...)``。

    现在 secondary group 的 dataset 必须是真实 dataset，不是 market。
    """
    ref_fund = _sref("StockIncome", "NetProfit", dataset="fundamental_quarterly", market="ashare")
    req = build_batch_data_request(
        _AnchorSource(),
        analyses={},
        dag=_dag(_plan("close", ref_fund)),
        ctx=SimpleNamespace(market="ashare"),
    )
    fund = next(g for g in req.groups if g.dataset != "ashare_daily")
    assert fund.dataset == "fundamental_quarterly"  # 不是 "ashare"
    assert fund.source_scope.dataset == "fundamental_quarterly"
    assert "ashare" not in {g.dataset for g in req.groups}
    assert fund.source_scope.market == "ashare"


def test_same_dataset_ref_goes_to_anchor():
    """与 anchor 同 dataset+market 的 SourceRef 走 anchor 源（避免同 scope 双 group）。"""
    ref = _sref("DailyBar", "Close", dataset="ashare_daily", market="ashare")
    req = build_batch_data_request(
        _AnchorSource(),
        analyses={},
        dag=_dag(_plan("close", ref)),
        ctx=SimpleNamespace(market="ashare"),
    )
    assert len(req.groups) == 1
    assert req.groups[0].dataset == "ashare_daily"
    assert "close" in req.groups[0].fields


def test_anchor_fallback_no_source_ref():
    """无任何 SourceRef 时，anchor group 兜底收全部列。"""
    req = build_batch_data_request(
        _AnchorSource(),
        analyses={},
        dag=_dag(_plan("close", "volume", "open")),
        ctx=SimpleNamespace(market="ashare"),
    )
    assert len(req.groups) == 1
    assert req.groups[0].dataset == "ashare_daily"
    assert set(req.groups[0].fields) == {"close", "volume", "open"}
    assert req.groups[0].scan_cost is not None


def test_batch_source_resolver_per_scope_estimator():
    """PERF-003：两个 adapter → 两个独立 cost source，不共享 anchor estimator。"""
    src = _AnchorSource()
    req = build_batch_data_request(
        src,
        analyses={},
        dag=_dag(_plan("close", _sref("StockIncome", "NetProfit", dataset="fundamental_quarterly", market="ashare"))),
        ctx=SimpleNamespace(market="ashare"),
    )
    anchor = next(g for g in req.groups if g.dataset == "ashare_daily")
    fund = next(g for g in req.groups if g.dataset == "fundamental_quarterly")

    # anchor estimator 只为 anchor group 估算过一次 —— 绝不为 secondary 复用。
    assert len(src.calls) == 1
    assert src.calls[0]["fields"] == ("close",)
    assert anchor.cost_estimator.__self__ is src
    assert anchor.cost_dataset == "ashare_daily"

    # secondary 用自己 adapter 的 estimator。
    assert fund.cost_estimator.__self__ is src.fund
    assert fund.cost_dataset == "fundamental_quarterly"
    assert len(src.fund.calls) == 1
    assert src.fund.calls[0]["fields"] == ("NetProfit",)
    assert src.fund.calls[0]["dataset"] == "fundamental_quarterly"
    assert src.fund.calls[0]["time_range"] == ("2024-01-01", "2024-12-31")


def test_resolver_missing_secondary_adapter_records_unavailable():
    """secondary 源没有可用 adapter → 记录独立 ScanCostUnavailable（绝不复用 anchor）。"""
    src = _AnchorSource()
    req = build_batch_data_request(
        src,
        analyses={},
        dag=_dag(_plan("close", _sref("StockUnknown", "X", dataset="unknown_table", market="ashare"))),
        ctx=SimpleNamespace(market="ashare"),
    )
    unavail = next(g.scan_cost_unavailable for g in req.groups if g.dataset == "unknown_table")
    assert isinstance(unavail, ScanCostUnavailable)
    assert "no secondary adapter" in unavail.reason
    # anchor 仍正常估到，不受影响。
    assert any(g.scan_cost is not None for g in req.groups)


def test_batch_source_resolver_direct():
    resolver = BatchSourceResolver(_AnchorSource(), market="ashare")
    anchor_scope = SourceScopeId(dataset="ashare_daily", market="ashare")
    assert resolver.resolve_source(anchor_scope).dataset == "ashare_daily"

    fund_scope = SourceScopeId(dataset="fundamental_quarterly", market="ashare")
    assert resolver.resolve_source(fund_scope).dataset == "fundamental_quarterly"

    missing_scope = SourceScopeId(dataset="no_such_table", market="ashare")
    assert resolver.resolve_source(missing_scope) is None


def test_time_range_end_only_preserved():
    """PERF-004：(None, end) 保留，端界进入 prune 边界（不是全量）。"""
    src = _AnchorSource()
    src.start_date = None
    src.end_date = "2025-12-31"

    req = build_batch_data_request(src, analyses={}, dag=_dag(_plan("close")), ctx=SimpleNamespace(market="ashare"))
    g = req.groups[0]
    assert g.time_range == TimeRange(None, "2025-12-31")
    assert g.time_range.as_tuple() == (None, "2025-12-31")
    assert g.to_dict()["time_range"] == [None, "2025-12-31"]
    # prune bound：estimator 收到的 end 是非 None 上界。
    assert src.calls[0]["time_range"] == (None, "2025-12-31")
    assert src.calls[0]["time_range"][1] == "2025-12-31"


def test_time_range_start_only_preserved():
    """PERF-004：(start, None) 保留，端界进入 prune 边界（不是全量）。"""
    src = _AnchorSource()
    src.start_date = "2025-01-01"
    src.end_date = None

    req = build_batch_data_request(src, analyses={}, dag=_dag(_plan("close")), ctx=SimpleNamespace(market="ashare"))
    g = req.groups[0]
    assert g.time_range == TimeRange("2025-01-01", None)
    assert g.time_range.as_tuple() == ("2025-01-01", None)
    assert g.to_dict()["time_range"] == ["2025-01-01", None]
    assert src.calls[0]["time_range"] == ("2025-01-01", None)
    assert src.calls[0]["time_range"][0] == "2025-01-01"


def test_time_range_source_none_dates_gives_none():
    src = _AnchorSource()
    src.start_date = None
    src.end_date = None
    req = build_batch_data_request(src, analyses={}, dag=_dag(_plan("close")), ctx=SimpleNamespace(market="ashare"))
    assert req.groups[0].time_range is None
    assert src.calls[0]["time_range"] is None


def test_hard_gate_counter_zero_on_typed_binding():
    src = _AnchorSource()
    req = build_batch_data_request(
        src,
        analyses={},
        dag=_dag(_plan("close", _sref("StockIncome", "NetProfit", dataset="fundamental_quarterly", market="ashare"))),
        ctx=SimpleNamespace(market="ashare"),
    )
    assert req.hard_gate_counters["SOURCE_REF_WITHOUT_TYPED_SOURCE_BINDING"] == 0


def test_hard_gate_counter_counts_unbound_source_ref():
    """SourceRef 前缀存在但解码不出 typed binding → 计 1，不静默丢进 anchor。"""
    malformed = "__fe_source_ref_v1__not-a-valid-payload"
    src = _AnchorSource()
    req = build_batch_data_request(
        src,
        analyses={},
        dag=_dag(_plan("close", malformed)),
        ctx=SimpleNamespace(market="ashare"),
    )
    assert req.hard_gate_counters["SOURCE_REF_WITHOUT_TYPED_SOURCE_BINDING"] == 1
    # 该列退化为 anchor 列（correctness 可继续），但性能证据已记录。
    assert malformed in req.groups[0].fields


def test_discover_binding_result_counts():
    ref_fund = _sref("StockIncome", "NetProfit", dataset="fundamental_quarterly", market="ashare")
    res = discover_column_source_bindings([_plan("close", ref_fund)])
    assert res.source_ref_columns_seen == 1
    assert res.source_ref_columns_bound == 1
    binding = res.bindings[ref_fund]
    assert isinstance(binding, ColumnSourceBinding)
    assert binding.dataset == "fundamental_quarterly"
    assert binding.field == "NetProfit"
    assert binding.market == "ashare"
    assert binding.source_scope.dataset == "fundamental_quarterly"
    assert binding.source_scope.market == "ashare"


def test_multi_field_coalesce_same_scope():
    """同 dataset+同 transform → 合并进同一 group，多个 field 一起估一次。"""
    ref_a = _sref("StockIncome", "NetProfit", dataset="fundamental_quarterly", market="ashare")
    ref_b = _sref("StockIncome", "TotalRevenue", dataset="fundamental_quarterly", market="ashare")
    src = _AnchorSource()
    req = build_batch_data_request(
        src,
        analyses={},
        dag=_dag(_plan("close", ref_a, ref_b)),
        ctx=SimpleNamespace(market="ashare"),
    )
    fund = next(g for g in req.groups if g.dataset == "fundamental_quarterly")
    assert set(fund.fields) == {"NetProfit", "TotalRevenue"}
    assert len(src.fund.calls) == 1
    assert set(src.fund.calls[0]["fields"]) == {"NetProfit", "TotalRevenue"}
