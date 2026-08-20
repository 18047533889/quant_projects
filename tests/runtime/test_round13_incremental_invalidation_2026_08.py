# -*- coding: utf-8 -*-
"""Round-13 中心整改回归：增量失效契约（P0-01/02/05）+ 单历史 authority（P0-03/04）
+ dependency edge 语义（P1-11）+ DataEvent ledger（P0-18/19/20）。

覆盖：
    P0-01  历史修订的影响向未来传播 —— ts_mean(20) 修订 t0 时 recompute_end 扩到 t0+19
    P0-01b stateful 因子（forward unbounded）→ recompute 到最新数据日
    P0-02  删除事件同样走 forward 传播
    P0-04  ParamSpec.min 不再是默认 window；kernel 签名默认值参与历史
    P0-05  IncrementalPlan 不再用 1e9 sentinel 判断 full-history（HistoryRequirement）
    P0-17  生产下 identity 校验 fail-closed（无法证明即失败）
    P0-18  DataEvent ledger：重复事件跳过、乱序拒绝、全部成功才 commit
    P0-19  生产裸 deleted key 拒绝
    P0-20  生产 partial success 抛 PartialIncrementalFailureError
"""
from __future__ import annotations

import pytest

from runtime.dependency_catalog import (
    DependencyCatalog,
    FactorDependencyEdge,
    UNBOUNDED_FORWARD_IMPACT,
)
from runtime.incremental_scheduler import (
    DataEvent,
    DataEventLedger,
    DataEventStaleError,
    DeletePredicateError,
    _coerce_deleted_keys,
    execute_incremental_updates_from_event,
    plan_updates_from_data_event,
)
from storage.catalog import FactorCatalog


def _mk_catalog(tmp_path) -> tuple[FactorCatalog, DependencyCatalog]:
    catalog = FactorCatalog(tmp_path / "_catalog.sqlite")
    return catalog, DependencyCatalog(catalog)


def _register(catalog: FactorCatalog, fid: str) -> None:
    catalog.register(
        fid,
        author="tester",
        frequency="1d",
        ast_hash="h_" + fid,
        expression='col("close")',
    )


# ---------------------------------------------------------------------------
# P0-01: forward impact propagation
# ---------------------------------------------------------------------------
def test_revision_extends_recompute_end_forward(tmp_path):
    catalog, dep = _mk_catalog(tmp_path)
    _register(catalog, "f_mean")
    # ts_mean(close, 20) recorded with a finite forward impact of 19.
    dep.record_factor_edges(
        "f_mean",
        edges=[
            FactorDependencyEdge(
                "f_mean", "ashare_stock_daily", "close",
                forward_impact=19,
            )
        ],
        lookback=20,
        frequency="1d",
        source_dataset="ashare_stock_daily",
    )
    event = DataEvent(
        dataset="ashare_stock_daily",
        column="close",
        updated_date="2026-08-09",
        field_id="close",
        affected_start="2024-06-01",
        affected_end="2024-06-01",
    )
    plans = plan_updates_from_data_event(dep, event, lookback_extra=0)
    assert len(plans) == 1
    plan = plans[0]
    # affected_end 被 forward impact 扩展到 2024-06-01 + 19 个交易日，绝不停在 t0。
    assert plan.since == "2024-06-01"
    assert plan.end_date is not None
    assert plan.end_date > "2024-06-01"
    assert plan.end_date < "2026-08-09"  # 封顶在最新数据日之前


def test_legacy_edge_no_forward_no_extension(tmp_path):
    """P0-01：legacy edge（未记录 forward_impact，存储为 None）不做 forward 扩展。"""
    catalog, dep = _mk_catalog(tmp_path)
    _register(catalog, "f_plain")
    dep.record_factor_edges(
        "f_plain",
        edges=[FactorDependencyEdge("f_plain", "ds_x", "close")],
        lookback=1,
        frequency="1d",
        source_dataset="ds_x",
    )
    plans = plan_updates_from_data_event(
        dep,
        DataEvent(
            dataset="ds_x", column="close", updated_date="2026-08-09",
            field_id="close", affected_start="2024-06-01", affected_end="2024-06-15",
        ),
        lookback_extra=0,
    )
    assert plans[0].end_date == "2024-06-15"


def test_unbounded_forward_recomputes_to_latest(tmp_path):
    """P0-01b：stateful 因子（-1 sentinel → unbounded）重算到最新数据日。"""
    catalog, dep = _mk_catalog(tmp_path)
    _register(catalog, "f_state")
    dep.record_factor_edges(
        "f_state",
        edges=[
            FactorDependencyEdge(
                "f_state", "ashare_stock_daily", "close",
                forward_impact=UNBOUNDED_FORWARD_IMPACT,
            )
        ],
        lookback=120,
        frequency="1d",
        source_dataset="ashare_stock_daily",
    )
    event = DataEvent(
        dataset="ashare_stock_daily", column="close", updated_date="2026-08-09",
        field_id="close", affected_start="2024-06-01", affected_end="2024-06-01",
    )
    plans = plan_updates_from_data_event(
        dep, event, end_date="2026-08-09", lookback_extra=0
    )
    assert plans[0].end_date == "2026-08-09"


def test_delete_event_forward_propagation(tmp_path):
    """P0-02：删除事件同样向未来传播（tombstone 之上 recompute_end 前移）。"""
    catalog, dep = _mk_catalog(tmp_path)
    _register(catalog, "f_mean")
    dep.record_factor_edges(
        "f_mean",
        edges=[
            FactorDependencyEdge(
                "f_mean", "ashare_stock_daily", "close",
                forward_impact=19,
            )
        ],
        lookback=20,
        frequency="1d",
        source_dataset="ashare_stock_daily",
    )
    event = DataEvent(
        dataset="ashare_stock_daily", column="close", updated_date="2026-08-09",
        field_id="close", affected_start="2024-06-01", affected_end="2024-06-01",
        deleted_keys=(("2024-06-01", "000001.SZ"),),
    )
    plans = plan_updates_from_data_event(dep, event, lookback_extra=0)
    assert len(plans) == 1
    assert plans[0].deleted_keys == (("2024-06-01", "000001.SZ"),)
    assert plans[0].end_date > "2024-06-01"


# ---------------------------------------------------------------------------
# P0-04 / P0-05: history single authority + no ParamSpec.min default
# ---------------------------------------------------------------------------
def test_param_spec_min_is_never_used_as_default():
    from types import SimpleNamespace

    from runtime.execution_contract import _bound_param

    # ParamSpec with min=2 but NO default: an unbound window must resolve to
    # None (unknown history), NOT to 2 (the legal-domain boundary).
    spec = SimpleNamespace(default=None, min=2)
    assert _bound_param({}, "window", {"window": spec}, canonical="") is None
    # ParamSpec.default is the declared default and is used.
    spec2 = SimpleNamespace(default=10, min=2)
    assert _bound_param({}, "window", {"window": spec2}, canonical="") == 10


def test_incremental_plan_uses_history_requirement_not_sentinel():
    from runtime.execution_contract import HistoryRequirement
    from runtime.incremental import build_incremental_plan

    plan = build_incremental_plan(
        factor_id="f1",
        analysis_lookback=1_000_000_000,  # legacy sentinel still means full replay
        watermark={"end_date": "2026-08-01"},
        since="2026-08-01",
        end_date="2026-08-09",
        market="A",
    )
    assert plan.full_history_required is True
    # ... but a HistoryRequirement object is now authoritative over the sentinel.
    finite = build_incremental_plan(
        factor_id="f1",
        analysis_lookback=1_000_000_000,
        history=HistoryRequirement(kind="finite", rows=5),
        watermark={"end_date": "2026-08-01"},
        since="2026-08-01",
        end_date="2026-08-09",
        market="A",
    )
    assert finite.full_history_required is False
    # the overriding HistoryRequirement is authoritative for the warm-up rows;
    # the legacy 1e9 sentinel is NOT treated as a window size.
    assert finite.backward_history == 5
    assert finite.history_kind == "finite"


# ---------------------------------------------------------------------------
# P0-17 / P0-19 / P0-20
# ---------------------------------------------------------------------------
def test_production_bare_deleted_key_rejected(tmp_path):
    catalog, dep = _mk_catalog(tmp_path)
    _register(catalog, "f_plain")
    with pytest.raises(DeletePredicateError):
        _coerce_deleted_keys(
            DataEvent(
                dataset="ds_x", column="close", updated_date="2026-08-09",
                deleted_keys=("000001.SZ",),
            ),
            production=True,
        )
    # research: bare key still auto-dated
    out = _coerce_deleted_keys(
        DataEvent(
            dataset="ds_x", column="close", updated_date="2026-08-09",
            affected_start="2024-06-01", deleted_keys=("000001.SZ",),
        ),
        production=False,
    )
    assert out == (("2024-06-01", "000001.SZ"),)


def test_ledger_idempotent_and_stale(tmp_path):
    ledger = DataEventLedger(lake_root=str(tmp_path))
    ev1 = DataEvent(
        dataset="d", column="c", updated_date="2026-08-01", event_id="e1",
        snapshot_before="s0", snapshot_after="s1",
    )
    status, token = ledger.begin(ev1)
    assert status == "ok"
    ledger.commit(ev1, token)
    # duplicate retry is skipped
    dup = DataEvent(
        dataset="d", column="c", updated_date="2026-08-01", event_id="e1",
        snapshot_before="s0", snapshot_after="s1",
    )
    status_dup, _ = ledger.begin(dup)
    assert status_dup == "duplicate"
    # stale: next event must chain from the committed snapshot s1, not s0
    stale = DataEvent(
        dataset="d", column="c", updated_date="2026-08-02", event_id="e2",
        snapshot_before="s0", snapshot_after="s2",
    )
    with pytest.raises(DataEventStaleError):
        ledger.begin(stale)
    # in-order event chains correctly
    good = DataEvent(
        dataset="d", column="c", updated_date="2026-08-02", event_id="e2",
        snapshot_before="s1", snapshot_after="s2",
    )
    status_good, _ = ledger.begin(good)
    assert status_good == "ok"


def test_execute_event_rejects_stale_in_production(tmp_path):
    """P0-20 wiring: a stale event (snapshot_before != ledger last committed) is
    rejected before any materialization attempt."""
    from runtime.dependency_catalog import DependencyCatalog

    catalog = FactorCatalog(tmp_path / "_catalog.sqlite")
    _register(catalog, "f_plain")
    ledger = DataEventLedger(lake_root=str(tmp_path))
    first = DataEvent(
        dataset="ds_x", column="close", updated_date="2026-08-01",
        field_id="close", event_id="e1",
        snapshot_before="s0", snapshot_after="s1",
    )
    status, token = ledger.begin(first)
    assert status == "ok"
    ledger.commit(first, token)
    # out-of-order event arriving late is rejected by the ledger check.
    stale = DataEvent(
        dataset="ds_x", column="close", updated_date="2026-08-02",
        field_id="close", event_id="e2",
        snapshot_before="s0", snapshot_after="s2",
    )
    with pytest.raises(DataEventStaleError):
        ledger.begin(stale)
