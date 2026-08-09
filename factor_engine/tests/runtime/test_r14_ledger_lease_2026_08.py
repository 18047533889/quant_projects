# -*- coding: utf-8 -*-
"""R14 #5 P1：DataEventLedger ``pending`` 预留的 lease / stale takeover。

外部 AI 复查：``pending`` 没有 lease/TTL——worker 在 ``begin(event)`` 后 crash，
其他 worker 永远得到 ``in_flight``，事件永久卡死（且 scheduler 把 ``in_flight``
返回成 ``"ledger_status": "duplicate"``）。

R14 #5 P1 修复：
  * pending 记录携带 ``owner`` / ``attempt_id`` / ``reserved_at``；
  * ``begin`` 对 lease 过期的 pending 直接 takeover（append 新预留，返回 ok）；
  * 无 ``reserved_at`` 的 legacy pending 保守不 takeover；
  * ``execute_incremental_updates_from_event`` 对 ``in_flight`` 返回真实的
    ``ledger_status``（不再塌缩成 "duplicate"）。
"""

from __future__ import annotations

import json

import pytest

from runtime.incremental_scheduler import (
    DataEvent,
    DataEventLedger,
    _ledger_pending_is_stale,
    execute_incremental_updates_from_event,
)


def _ev(
    event_id: str,
    *,
    owner: str | None = None,
    attempt_id: str | None = None,
    dataset: str = "d",
    field: str = "c",
    snapshot_before: str | None = None,
) -> DataEvent:
    return DataEvent(
        dataset=dataset,
        column=field,
        field_id=field,
        updated_date="2026-08-01",
        event_id=event_id,
        owner=owner,
        attempt_id=attempt_id,
        snapshot_before=snapshot_before,
    )


def _append_raw_pending(ledger: DataEventLedger, event: DataEvent, reserved_at: str) -> None:
    """直接往 ledger 文件写一条 pending 记录（含指定 reserved_at，模拟旧/陈旧预留）。"""
    rec = {
        "event_id": event.event_id,
        "dataset": event.dataset,
        "field": event.field,
        "sequence": event.sequence,
        "status": "pending",
        "snapshot_before": event.snapshot_before,
        "snapshot_after": None,
        "received_at": event.updated_date,
        "owner": event.owner,
        "attempt_id": event.attempt_id,
        "reserved_at": reserved_at,
    }
    with open(ledger._path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        fh.flush()


def test_r14_pending_carries_lease_fields(tmp_path):
    ledger = DataEventLedger(lake_root=str(tmp_path / "lake"))
    ev = _ev("e1", owner="w1", attempt_id="a1")
    status, token = ledger.begin(ev)
    assert status == "ok"
    assert token is not None
    assert token.owner == "w1"
    assert token.attempt_id == "a1"
    assert token.fencing_epoch == 1
    assert token.expires_at, "token 必须带 expires_at"
    rec = ledger._records["e1"]
    assert rec["status"] == "pending"
    assert rec["owner"] == "w1"
    assert rec["attempt_id"] == "a1"
    assert rec["fencing_epoch"] == 1, "pending 必须带 fencing_epoch（fencing 校验）"
    assert rec["reserved_at"], "pending 必须带 reserved_at（lease 过期判定）"


def test_r14_fresh_pending_returns_in_flight(tmp_path):
    ledger = DataEventLedger(lake_root=str(tmp_path / "lake"))
    ev = _ev("e2", owner="w1", attempt_id="a1")
    status, _ = ledger.begin(ev)
    assert status == "ok"
    # 同一 worker 换 attempt 重复 begin（尚未 crash）→ 仍在预留期内 → in_flight
    status2, token2 = ledger.begin(_ev("e2", owner="w1", attempt_id="a2"))
    assert status2 == "in_flight"
    assert token2 is None


def test_r14_same_attempt_rebegin_refreshes_lease(tmp_path):
    """同一 worker 同一 attempt 重入 → 幂等续租（返回原 attempt/fence，非 in_flight）。"""
    ledger = DataEventLedger(lake_root=str(tmp_path / "lake"))
    ev = _ev("e2", owner="w1", attempt_id="a1")
    status, token = ledger.begin(ev)
    assert status == "ok"
    status2, token2 = ledger.begin(_ev("e2", owner="w1", attempt_id="a1"))
    assert status2 == "ok"
    assert token2.attempt_id == "a1"
    assert token2.fencing_epoch == token.fencing_epoch


def test_r14_stale_pending_taken_over(tmp_path):
    ledger = DataEventLedger(lake_root=str(tmp_path / "lake"))
    ev = _ev("e3", owner="crashed-worker", attempt_id="a1")
    # 旧 worker 在 begin 后 crash：预留 2 小时前（远大于默认 3600s lease）
    _append_raw_pending(ledger, ev, reserved_at="2026-08-01T00:00:00+00:00")
    # 新 worker 现在（2026-08-09）begin → lease 过期 → takeover
    status, token = ledger.begin(_ev("e3", owner="w2", attempt_id="b1"))
    assert status == "ok"
    assert token is not None
    assert token.attempt_id == "b1"
    assert token.fencing_epoch == 1  # legacy pending 无 fence → 接管从 1 起
    rec = ledger._records["e3"]
    assert rec["owner"] == "w2"
    assert rec["attempt_id"] == "b1"
    assert rec["fencing_epoch"] == 1


def test_r14_legacy_pending_without_reserved_at_not_taken_over(tmp_path):
    ledger = DataEventLedger(lake_root=str(tmp_path / "lake"))
    ev = _ev("e4")
    _append_raw_pending(ledger, ev, reserved_at="")
    ledger._reload()
    assert _ledger_pending_is_stale(ledger._records["e4"]) is False
    status, token = ledger.begin(_ev("e4"))
    assert status == "in_flight"
    assert token is None


def test_r14_execute_flow_reports_in_flight_not_duplicate(tmp_path):
    """scheduler 对 in_flight 必须返回真实 ledger_status，不再塌缩成 duplicate。"""
    from runtime.incremental_scheduler import _ledger_now_iso

    lake = tmp_path / "lake"
    # 预置一条**fresh** pending 预留（模拟其他 worker 正在处理、lease 未过期）
    ledger = DataEventLedger(lake_root=str(lake))
    _append_raw_pending(
        ledger, _ev("ev_inflight"), reserved_at=_ledger_now_iso()
    )
    out = execute_incremental_updates_from_event(
        None,
        _ev("ev_inflight"),
        lake_root=str(lake),
    )
    assert out["ledger_status"] == "in_flight"
    assert out["factor_count"] == 0
