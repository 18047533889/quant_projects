# -*- coding: utf-8 -*-
"""R44: node-level incremental FactorEngine — asset-scope / AffectedDomain.

在既有 1-D 时间窗口（``AffectedWindow``）之外，为每个受影响节点附加 asset
维度（:class:`AffectedDomain`）。单标的的 source change 经过跨截面/分组算子后，
根的 asset 作用域会升级到整截面 / 组 / 全宇宙；时间窗口保持不变。
"""
from __future__ import annotations

import pytest

from factor_engine.ir.nodes import IRNode
from factor_engine.runtime.change_impact import (
    AssetScope,
    AffectedDomain,
    AffectedWindow,
    affected_root_domain,
    affected_root_window,
    compute_affected_domains,
    compute_change_impact,
)
from factor_engine.runtime.incremental_scheduler import DataEvent, normalize_data_event


def _col(name="close"):
    return IRNode("column", attrs={"name": name})


def _ts_mean(col, window=20):
    return IRNode("ts_mean", (col,), attrs={"window": window})


# ---------------------------------------------------------------------------
# 1. 单标的有限变化 → 根域 ONE_INSTRUMENT，时间窗口与 affected_root_window 一致
# ---------------------------------------------------------------------------
def test_single_instrument_finite_change_is_one_instrument():
    plan = _ts_mean(_col(), window=20)
    root = affected_root_domain(plan, field="close", changed_start="2024-01-10")
    assert root is not None
    assert root.asset_scope == AssetScope.ONE_INSTRUMENT
    assert root.affected_instruments == ()
    assert root.group_key is None
    # 时间窗口与既有 affected_root_window 完全一致。
    win = affected_root_window(plan, field="close", changed_start="2024-01-10")
    assert root.time_range == win
    assert root.time_range.start == "2024-01-10"
    assert root.time_range.end == "2024-02-06"  # window=20 -> impact 19 bdays


# ---------------------------------------------------------------------------
# 2. 同一变化喂给 cs_rank → 根域 FULL_UNIVERSE
# ---------------------------------------------------------------------------
def test_cs_rank_escalates_to_full_universe():
    plan = IRNode("cs_rank_01", (_ts_mean(_col(), window=20),))
    root = affected_root_domain(plan, field="close", changed_start="2024-01-10")
    assert root is not None
    assert root.asset_scope == AssetScope.FULL_UNIVERSE
    # 时间窗口不变（cs_rank 自身 forward impact = 0，仍继承 ts_mean 的窗口）。
    win = affected_root_window(plan, field="close", changed_start="2024-01-10")
    assert root.time_range == win


# ---------------------------------------------------------------------------
# 3. 同一变化喂给 group 算子 → GROUP 作用域
# ---------------------------------------------------------------------------
def test_group_op_escalates_to_group():
    plan = IRNode("group_rank", (_ts_mean(_col(), window=20),))
    root = affected_root_domain(plan, field="close", changed_start="2024-01-10")
    assert root is not None
    assert root.asset_scope == AssetScope.GROUP


def test_group_op_with_group_key_attr():
    plan = IRNode(
        "group_rank",
        (_ts_mean(_col(), window=20),),
        attrs={"group": "sw_l1=bank"},
    )
    root = affected_root_domain(plan, field="close", changed_start="2024-01-10")
    assert root is not None
    assert root.asset_scope == AssetScope.GROUP
    assert root.group_key == "sw_l1=bank"


# ---------------------------------------------------------------------------
# 4. DataEvent INSTRUMENT_SET 作用域传播
# ---------------------------------------------------------------------------
def test_data_event_instrument_set_propagates():
    event = DataEvent(
        dataset="ashare_stock_daily",
        column="close",
        updated_date="2024-01-10",
        affected_instruments=("600000.SH", "000001.SZ"),
        asset_scope="INSTRUMENT_SET",
    )
    plan = _ts_mean(_col(), window=20)
    root = affected_root_domain(
        plan, field="close", changed_start="2024-01-10", event=event
    )
    assert root is not None
    assert root.asset_scope == AssetScope.INSTRUMENT_SET
    assert root.affected_instruments == ("600000.SH", "000001.SZ")


def test_data_event_dict_coercible():
    event = {
        "dataset": "ashare_stock_daily",
        "column": "close",
        "updated_date": "2024-01-10",
        "affected_instruments": ["600000.SH"],
        "asset_scope": "INSTRUMENT_SET",
    }
    plan = _ts_mean(_col(), window=20)
    root = affected_root_domain(
        plan, field="close", changed_start="2024-01-10", event=event
    )
    assert root is not None
    assert root.asset_scope == AssetScope.INSTRUMENT_SET
    assert root.affected_instruments == ("600000.SH",)


# ---------------------------------------------------------------------------
# 5. 全宇宙变化事件 → 根 FULL_UNIVERSE
# ---------------------------------------------------------------------------
def test_universe_change_event_full_universe():
    event = DataEvent(
        dataset="ashare_stock_daily",
        column="close",
        updated_date="2024-01-10",
        asset_scope="FULL_UNIVERSE",
    )
    plan = _ts_mean(_col(), window=20)
    root = affected_root_domain(
        plan, field="close", changed_start="2024-01-10", event=event
    )
    assert root is not None
    assert root.asset_scope == AssetScope.FULL_UNIVERSE


# ---------------------------------------------------------------------------
# 6. GLOBAL child 强制 GLOBAL parent
# ---------------------------------------------------------------------------
def test_global_child_forces_global_parent():
    # 一个 GLOBAL 作用域的分支 + 一个 ONE_INSTRUMENT 分支 → 根取最严重 GLOBAL。
    global_col = IRNode("column", attrs={"name": "close"})
    global_branch = IRNode("ts_mean", (global_col,), attrs={"window": 5})
    local_col = IRNode("column", attrs={"name": "close"})
    local_branch = IRNode("ts_mean", (local_col,), attrs={"window": 5})
    plan = IRNode("add", (global_branch, local_branch))
    event = DataEvent(
        dataset="ashare_stock_daily",
        column="close",
        updated_date="2024-01-10",
        asset_scope="GLOBAL",
    )
    root = affected_root_domain(
        plan, field="close", changed_start="2024-01-10", event=event
    )
    assert root is not None
    assert root.asset_scope == AssetScope.GLOBAL


# ---------------------------------------------------------------------------
# 7. 向后兼容：compute_change_impact 输出不变
# ---------------------------------------------------------------------------
def test_backward_compat_compute_change_impact_unchanged():
    plan = IRNode("add", (_ts_mean(_col(), window=20), _ts_mean(_col(), window=5)))
    before = compute_change_impact(plan, field="close", changed_start="2024-01-10")
    # 直接调用（与 R44 无关的既有路径）应返回相同 AffectedWindow 列表。
    assert isinstance(before, list)
    assert all(isinstance(w, AffectedWindow) for w in before)
    # 与 affected_root_window 摘要一致。
    assert before[-1] == affected_root_window(
        plan, field="close", changed_start="2024-01-10"
    )


# ---------------------------------------------------------------------------
# 8. 用 IRNode 手工构建 plan 树
# ---------------------------------------------------------------------------
def test_irnode_plan_tree_domains():
    col = _col()
    ts = _ts_mean(col, window=20)
    cs = IRNode("cs_rank_01", (ts,))
    plan = IRNode("add", (cs, _ts_mean(_col(), window=5)))
    domains = compute_affected_domains(plan, field="close", changed_start="2024-01-10")
    assert len(domains) >= 1
    by_op = {d.op: d for d in domains}
    assert "add" in by_op
    assert by_op["add"].asset_scope == AssetScope.FULL_UNIVERSE  # cs 分支升级
    assert by_op["cs_rank_01"].asset_scope == AssetScope.FULL_UNIVERSE
    assert by_op["ts_mean"].asset_scope == AssetScope.ONE_INSTRUMENT


# ---------------------------------------------------------------------------
# DataEvent 新字段：to_dict / normalize_data_event 往返
# ---------------------------------------------------------------------------
def test_data_event_new_fields_roundtrip():
    ev = DataEvent(
        dataset="d",
        column="close",
        updated_date="2024-01-10",
        affected_instruments=("a", "b"),
        asset_scope="GROUP",
        group_key="sw_l1=bank",
    )
    d = ev.to_dict()
    assert d["affected_instruments"] == ["a", "b"]
    assert d["asset_scope"] == "GROUP"
    assert d["group_key"] == "sw_l1=bank"
    ev2 = normalize_data_event(d)
    assert ev2.affected_instruments == ("a", "b")
    assert ev2.asset_scope == "GROUP"
    assert ev2.group_key == "sw_l1=bank"


def test_data_event_new_fields_default_none():
    ev = DataEvent(dataset="d", column="close", updated_date="2024-01-10")
    assert ev.affected_instruments is None
    assert ev.asset_scope is None
    assert ev.group_key is None
    d = ev.to_dict()
    assert d["affected_instruments"] is None
    assert d["asset_scope"] is None
    assert d["group_key"] is None
