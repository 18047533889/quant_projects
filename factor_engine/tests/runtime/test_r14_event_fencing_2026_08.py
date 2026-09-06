# -*- coding: utf-8 -*-
"""R14 复查 P0-2：DataEventLedger lease takeover 的 **fencing** 测试。

上一轮只加了 lease TTL：stale pending 可被其他 worker takeover，但被接管后旧
worker 仍可 ``commit``/``reject``（``commit`` 从不校验 owner/attempt/fence）→
A 醒来 publish+commit、B 也 publish+commit = 双写。``owner``/``attempt_id`` 也只是
记录信息，没有成为 fencing token；dict/API 送事件时这两个字段直接丢。

本文件覆盖 destructive gate 清单：
  #4  A begin fence=1 → lease 过期 → B takeover fence=2 → A commit → MUST FAIL
  #5  A lease lost → A publish（renew 门）→ MUST FAIL / 不可见
  #6  B commit → 事件可见**恰好一次**
  #7  dict DataEvent owner/attempt/fence round-trip 保留
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import pandas as pd

from factor_engine.runtime.incremental_scheduler import (
    DataEvent,
    DataEventLeaseLostError,
    DataEventLedger,
    _ledger_now_iso,
    execute_incremental_updates_from_event,
)
from factor_engine.storage.materializer import ParquetMaterializer
from factor_engine.storage.trading_calendar import (
    TradingCalendar,
    clear_trading_calendar_cache,
    register_trading_calendar,
)


@pytest.fixture(autouse=True)
def _production_calendar():
    """Install a deterministic, explicitly sourced A-share calendar for R14."""
    clear_trading_calendar_cache()
    register_trading_calendar(
        "ashare",
        TradingCalendar(
            list(pd.bdate_range("2026-07-01", "2026-09-30")),
            anchor_policy="previous_trade_day",
            source="explicit_test_fixture",
            snapshot="r14-2026-08",
            version="1",
            timezone="Asia/Shanghai",
        ),
    )
    yield
    clear_trading_calendar_cache()


def _ev(
    event_id: str,
    *,
    owner: str | None = None,
    attempt_id: str | None = None,
    fencing_epoch: int | None = None,
    dataset: str = "d",
    field: str = "c",
) -> DataEvent:
    return DataEvent(
        dataset=dataset,
        column=field,
        field_id=field,
        updated_date="2026-08-01",
        event_id=event_id,
        owner=owner,
        attempt_id=attempt_id,
        fencing_epoch=fencing_epoch,
    )


def _append_raw_pending(
    ledger: DataEventLedger,
    event_id: str,
    *,
    owner: str,
    attempt_id: str,
    fencing_epoch: int,
    reserved_at: str,
) -> None:
    """直接往 ledger 文件写一条 pending 记录（模拟 B 接管 / A 旧预留）。"""
    rec = {
        "event_id": event_id,
        "dataset": "d",
        "field": "c",
        "sequence": None,
        "status": "pending",
        "snapshot_before": None,
        "snapshot_after": None,
        "received_at": "2026-08-01",
        "owner": owner,
        "attempt_id": attempt_id,
        "fencing_epoch": fencing_epoch,
        "reserved_at": reserved_at,
    }
    with open(ledger._path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        fh.flush()


def _expire_lease(ledger: DataEventLedger, event_id: str, owner: str, attempt_id: str, fence: int) -> None:
    """把当前 pending 的 lease 改成过期（写一条旧 reserved_at 的同 key 记录，
    append-only ledger 取最后一条）。"""
    _append_raw_pending(
        ledger,
        event_id,
        owner=owner,
        attempt_id=attempt_id,
        fencing_epoch=fence,
        reserved_at="2026-08-01T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# destructive gate #4 / #6：stale owner commit 被 fencing 拒绝；B commit 恰好一次
# ---------------------------------------------------------------------------


def test_r14_stale_owner_commit_rejected_b_commits_once(tmp_path):
    """A begin fence=1 → lease 过期 → B takeover fence=2 → A commit MUST FAIL
    （DataEventLeaseLostError）；B commit → 事件可见恰好一次。"""
    ledger = DataEventLedger(lake_root=str(tmp_path / "lake"))
    st_a, tok_a = ledger.begin(_ev("e1", owner="A", attempt_id="a1"))
    assert st_a == "ok"
    assert tok_a.fencing_epoch == 1

    # A 卡住 > lease；B 来 begin → takeover（fence 单调递增）
    _expire_lease(ledger, "e1", owner="A", attempt_id="a1", fence=1)
    st_b, tok_b = ledger.begin(_ev("e1", owner="B", attempt_id="b1"))
    assert st_b == "ok"
    assert tok_b.fencing_epoch == 2, "takeover 必须单调递增 fence"

    # A 醒来 commit（用旧 token）→ 必须被拒
    with pytest.raises(DataEventLeaseLostError, match="接管|失效"):
        ledger.commit(_ev("e1", owner="A", attempt_id="a1"), tok_a)
    assert not ledger.is_committed("e1")

    # B commit → 成功；committed 记录恰好一条（destructive gate #6）
    ledger.commit(_ev("e1", owner="B", attempt_id="b1"), tok_b)
    assert ledger.is_committed("e1")
    raw = (tmp_path / "lake" / ".event_ledger.jsonl").read_text()
    committed = [
        l for l in raw.splitlines()
        if '"e1"' in l and '"committed"' in l
    ]
    assert len(committed) == 1


def test_r14_takeover_fence_monotonic(tmp_path):
    """连续两次 takeover：fence 1 → 2 → 3，绝不回落。"""
    ledger = DataEventLedger(lake_root=str(tmp_path / "lake"))
    _, tok_a = ledger.begin(_ev("e1", owner="A", attempt_id="a1"))
    assert tok_a.fencing_epoch == 1
    _expire_lease(ledger, "e1", "A", "a1", 1)
    _, tok_b = ledger.begin(_ev("e1", owner="B", attempt_id="b1"))
    assert tok_b.fencing_epoch == 2
    _expire_lease(ledger, "e1", "B", "b1", 2)
    _, tok_c = ledger.begin(_ev("e1", owner="C", attempt_id="c1"))
    assert tok_c.fencing_epoch == 3


def test_r14_stale_owner_reject_rejected(tmp_path):
    """A lease 丢失后 A reject → DataEventLeaseLostError（stale worker 不得 reject）。"""
    ledger = DataEventLedger(lake_root=str(tmp_path / "lake"))
    _, tok_a = ledger.begin(_ev("e1", owner="A", attempt_id="a1"))
    _expire_lease(ledger, "e1", "A", "a1", 1)
    st_b, _ = ledger.begin(_ev("e1", owner="B", attempt_id="b1"))
    assert st_b == "ok"
    with pytest.raises(DataEventLeaseLostError, match="接管|失效"):
        ledger.reject(_ev("e1", owner="A", attempt_id="a1"), "stale A", token=tok_a)


# ---------------------------------------------------------------------------
# destructive gate #5：A lease 丢失后 A 的 publish（renew 门）被拒
# ---------------------------------------------------------------------------


def test_r14_stale_owner_renew_rejected(tmp_path):
    """A lease 丢失后 A renew（publish 前的 fencing 门）→ DataEventLeaseLostError，
    且 pending 仍归 B（A 的续租不可能刷新 B 的记录）。"""
    ledger = DataEventLedger(lake_root=str(tmp_path / "lake"))
    _, tok_a = ledger.begin(_ev("e1", owner="A", attempt_id="a1"))
    _expire_lease(ledger, "e1", "A", "a1", 1)
    _, tok_b = ledger.begin(_ev("e1", owner="B", attempt_id="b1"))
    assert tok_b.fencing_epoch == 2
    with pytest.raises(DataEventLeaseLostError, match="接管|失效"):
        ledger.renew(tok_a)
    rec = ledger._records["e1"]
    assert rec["owner"] == "B" and rec["fencing_epoch"] == 2


def test_r14_commit_without_token_fails(tmp_path):
    """无 fencing token 的 commit → fail-closed（旧代码没有 fencing，任意 commit）。"""
    ledger = DataEventLedger(lake_root=str(tmp_path / "lake"))
    st, _ = ledger.begin(_ev("e1", owner="A", attempt_id="a1"))
    assert st == "ok"
    with pytest.raises(DataEventLeaseLostError, match="无 fencing token"):
        ledger.commit(_ev("e1", owner="A", attempt_id="a1"), None)


def test_r14_renew_heartbeat_refreshes_lease(tmp_path):
    """正常 holder 的 renew：续租 + fencing 通过（不被误 takeover）。"""
    ledger = DataEventLedger(lake_root=str(tmp_path / "lake"))
    _, tok = ledger.begin(_ev("e1", owner="A", attempt_id="a1"))
    ledger.renew(tok)  # 不抛
    rec = ledger._records["e1"]
    from factor_engine.runtime.incremental_scheduler import _ledger_pending_is_stale

    assert _ledger_pending_is_stale(rec) is False


# ---------------------------------------------------------------------------
# destructive gate #7：dict DataEvent owner/attempt/fence round-trip
# ---------------------------------------------------------------------------


def test_r14_dict_roundtrip_preserves_fencing_fields():
    """dict → normalize → to_dict 保留 owner/attempt_id/fencing_epoch（旧代码丢失）。"""
    from factor_engine.runtime.incremental_scheduler import normalize_data_event

    raw = {
        "dataset": "d",
        "column": "c",
        "updated_date": "2026-08-01",
        "event_id": "e1",
        "owner": "A",
        "attempt_id": "a1",
        "fencing_epoch": 3,
    }
    ev = normalize_data_event(raw)
    assert ev.owner == "A"
    assert ev.attempt_id == "a1"
    assert ev.fencing_epoch == 3
    out = ev.to_dict()
    assert out["owner"] == "A"
    assert out["attempt_id"] == "a1"
    assert out["fencing_epoch"] == 3
    # 二次 normalize 仍保留
    ev2 = normalize_data_event(out)
    assert ev2.fencing_epoch == 3
    # token 序列化 round-trip
    ledger = DataEventLedger(lake_root=None)
    _st, token = ledger.begin(ev2)
    from factor_engine.runtime.incremental_scheduler import ReservationToken

    d = token.to_dict()
    token2 = ReservationToken(**d)
    assert token2 == token


# ---------------------------------------------------------------------------
# destructive gate #5（scheduler 级）：A lease 丢失 → A 的 publish 被拒/不可见
# ---------------------------------------------------------------------------


def test_r14_execute_flow_stale_worker_publish_gate(tmp_path, monkeypatch):
    """A 的 lease 被接管后，调度器在 stage 后 renew（publish 门）抛
    DataEventLeaseLostError → **零 publish**（A 发布被拒，读者看不到新因子）。"""
    from factor_engine.api.dsl_parser import parse_factor
    from factor_engine.ir.analyzer import Analyzer
    from factor_engine.runtime.dependency_catalog import DependencyCatalog, FactorDependencyEdge
    from factor_engine.storage.catalog import compute_ir_hash

    _force_production(monkeypatch)
    monkeypatch.setenv("DATA_EVENT_PRODUCTION_AUTO_PUBLISH", "1")

    ir = Analyzer(production=True, market="ashare").lower(
        parse_factor('field("close")').expr
    ).ir
    ast_hash = compute_ir_hash(ir)
    lake = tmp_path / "lake"
    catalog = ParquetMaterializer(lake_root=lake).catalog
    dep = DependencyCatalog(catalog)
    for fid in ("f_a", "f_b"):
        catalog.register(
            fid, author="tester", frequency="1d", ast_hash=ast_hash,
            expression='field("close")',
            data_source_config={"type": "parquet", "dataset": "ds_x"},
        )
        dep.record_factor_edges(
            fid,
            edges=[FactorDependencyEdge(fid, "shared_ds", "close")],
            lookback=2, frequency="1d", source_dataset="shared_ds",
        )
        dep.record_full_factor_definition(
            fid, expression='field("close")', frequency="1d",
            data_source_config={"type": "parquet", "dataset": "ds_x"},
        )

    control = {"staged": []}

    def fake_engine_factory(full_def, *, lake_root=None, market=None, run_mode=None):
        eng = MagicMock()

        def _mi(factor, *, factor_id, since=None, end_date=None, expression=None,
                frequency=None, description=None, deleted_keys=None, **kw):
            control["staged"].append(factor_id)
            if factor_id == "f_a":
                # 注入：B 在 A 计算期间接管（写一条更高 fence 的 pending）
                led_b = DataEventLedger(lake_root=lake_root)
                _append_raw_pending(
                    led_b, "ev_fence",
                    owner="B", attempt_id="b1", fencing_epoch=2,
                    reserved_at=_ledger_now_iso(),
                )
            return {"materialization": {"factor_id": factor_id, "rows_written": 1}}

        eng.materialize_incremental.side_effect = _mi
        return eng

    store = _FakeStore()
    monkeypatch.setattr("data_access.get_store", lambda: store)
    ev = DataEvent(
        dataset="shared_ds", column="close", updated_date="2026-08-09",
        field_id="close", revision_kind="update", event_id="ev_fence",
    )
    with pytest.raises(DataEventLeaseLostError, match="接管|失效"):
        execute_incremental_updates_from_event(
            None, ev, lake_root=lake, market="ashare", engine_factory=fake_engine_factory
        )
    # A 未 publish 任何因子（stage 后被 fencing 门拦下）→ 读者看到 ZERO 新因子
    assert control["staged"] == ["f_a"]
    assert store.published == []
    assert store.staged == []


def _force_production(monkeypatch) -> None:
    monkeypatch.setattr(
        "factor_engine.runtime.production_policy.is_production_mode", lambda: True
    )


class _FakeStore:
    """fake DA store：记录 staged / published。"""

    def __init__(self):
        self.staged: list[str] = []
        self.published: list[str] = []

    def upsert(self, dataset, table, upsert_on=None, partition_by=None,
               factor_id=None, **kw):
        self.staged.append(factor_id)
        return {"rows_upserted": table.num_rows}

    def resolve_dataset_path(self, dataset, factor_id=None):
        return Path("/tmp/fake_published")

    def publish_from_staging(self, *args, **kwargs):
        self.published.append(kwargs.get("factor_id"))
        return {"rows": 1}
