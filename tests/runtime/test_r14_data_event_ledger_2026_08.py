# -*- coding: utf-8 -*-
"""R14 #5 DataEventLedger 事务语义测试。

旧实现问题（外部 AI 复查）：
  * ``last_committed_snapshot`` 返回**第一个** committed record 而非最新 chain
    head → 事件链 e1→e2→e3 时把 e1 的 snapshot 当 chain head，把合法的
    ``snapshot_before=s2`` 误判 stale；
  * ledger 文件损坏被静默跳过/清空、append ``OSError`` 被直接吞掉；
  * 无 multiprocess reservation/CAS。

R14 #5 改动：
  * ``last_committed_snapshot`` 取 sequence 最大的 chain head（sequence 空时取
    文件顺序最后者）；``sequence`` 参与乱序校验；
  * 中段损坏 fail loud（``DataEventLedgerCorruptionError``），仅尾部半行（崩溃
    mid-append）截断恢复；append OSError 上抛不再吞；
  * ``begin``/``commit``/``reject`` 在 flock 内 reload+append —— 并发 worker
    同一 event 只有一个能 begin（pending 即预留）。
"""

from __future__ import annotations

import json

import pytest

from factor_engine.runtime.incremental_scheduler import (
    DataEvent,
    DataEventLedger,
    DataEventLedgerCorruptionError,
    DataEventStaleError,
)


def _ev(
    event_id: str,
    *,
    snapshot_before: str | None = None,
    snapshot_after: str | None = None,
    sequence: int | None = None,
    dataset: str = "d",
    field: str = "c",
) -> DataEvent:
    return DataEvent(
        dataset=dataset,
        column=field,
        field_id=field,
        updated_date="2026-08-01",
        event_id=event_id,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
        sequence=sequence,
    )


def _ledger_path(tmp_path) -> dict:
    path = tmp_path / "lake" / ".event_ledger.jsonl"
    return {"path": path}


def test_r14_chain_head_is_latest_committed(tmp_path):
    """三事件链 s0→s1→s2→s3：last_committed 必须是 s3，不是 e1 的 s1。"""
    ledger = DataEventLedger(lake_root=str(tmp_path / "lake"))
    e1 = _ev("e1", snapshot_before="s0", snapshot_after="s1", sequence=1)
    e2 = _ev("e2", snapshot_before="s1", snapshot_after="s2", sequence=2)
    e3 = _ev("e3", snapshot_before="s2", snapshot_after="s3", sequence=3)
    for e in (e1, e2, e3):
        status, token = ledger.begin(e)
        assert status == "ok"
        ledger.commit(e, token)
    assert ledger.last_committed_snapshot("d", "c") == "s3"


def test_r14_chain_head_selected_by_sequence(tmp_path):
    """chain head 取 sequence 最大者，而不是第一个/最后插入者。"""
    ledger = DataEventLedger(lake_root=str(tmp_path / "lake"))
    e_early = _ev("e_early", snapshot_before="s0", snapshot_after="s1", sequence=5)
    status, token = ledger.begin(e_early)
    assert status == "ok"
    ledger.commit(e_early, token)
    # 同一链上 sequence 更大但先被处理的保留先提交 —— head 必须是 sequence 大的
    e_late = _ev("e_late", snapshot_before="s1", snapshot_after="s2", sequence=9)
    status, token = ledger.begin(e_late)
    assert status == "ok"
    ledger.commit(e_late, token)
    assert ledger.last_committed_snapshot("d", "c") == "s2"


def test_r14_out_of_order_sequence_rejected(tmp_path):
    ledger = DataEventLedger(lake_root=str(tmp_path / "lake"))
    e1 = _ev("e1", snapshot_before="s0", snapshot_after="s1", sequence=10)
    status, token = ledger.begin(e1)
    assert status == "ok"
    ledger.commit(e1, token)
    # sequence 不大于 chain head → 乱序
    with pytest.raises(DataEventStaleError, match="sequence"):
        ledger.begin(_ev("e2", snapshot_before="s1", snapshot_after="s2", sequence=10))
    with pytest.raises(DataEventStaleError, match="sequence"):
        ledger.begin(_ev("e2", snapshot_before="s1", snapshot_after="s2", sequence=9))


def test_r14_two_workers_same_event_cas(tmp_path):
    """两个 worker 同时处理同一事件：只有一个能 begin（pending 预留）。"""
    ledger_a = DataEventLedger(lake_root=str(tmp_path / "lake"))
    ledger_b = DataEventLedger(lake_root=str(tmp_path / "lake"))
    ev = _ev("e1", snapshot_before="s0", snapshot_after="s1")
    status_a, token_a = ledger_a.begin(ev)
    assert status_a == "ok"  # worker A 拿到预留
    status_b, _ = ledger_b.begin(ev)
    assert status_b == "in_flight"  # worker B 被 CAS 拒绝
    ledger_a.commit(ev, token_a)
    # 已提交后再来 → duplicate（幂等重试跳过）
    status_dup, _ = ledger_b.begin(ev)
    assert status_dup == "duplicate"


def test_r14_mid_file_corruption_fails_loud(tmp_path):
    path = tmp_path / "lake"
    path.mkdir(parents=True)
    ledger = DataEventLedger(lake_root=str(path))
    e1 = _ev("e1", snapshot_before="s0", snapshot_after="s1")
    _, token = ledger.begin(e1)
    ledger.commit(e1, token)
    # 中段插入一行坏 JSON（后面还有合法行）→ 重新加载必须 fail loud，
    # 绝不静默跳过。坏行若是**最后**一行，会被当作崩溃残留截断——所以这里
    # 再追加一行合法记录，让坏行落在中间。
    with open(path / ".event_ledger.jsonl", "a", encoding="utf-8") as fh:
        fh.write("{broken-json-line\n")
        fh.write(json.dumps({"event_id": "e2", "status": "committed"}) + "\n")
    with pytest.raises(DataEventLedgerCorruptionError):
        DataEventLedger(lake_root=str(path))


def test_r14_trailing_partial_line_truncated(tmp_path):
    """崩溃 mid-append 的尾部半行 → 截断恢复，已提交记录保留。"""
    path = tmp_path / "lake"
    path.mkdir(parents=True)
    ledger = DataEventLedger(lake_root=str(path))
    e1 = _ev("e1", snapshot_before="s0", snapshot_after="s1")
    _, token = ledger.begin(e1)
    ledger.commit(e1, token)
    # 尾部半行（无 \n，模拟崩溃中断写）
    with open(path / ".event_ledger.jsonl", "ab") as fh:
        fh.write(b'{"event_id": "e_partial", "status": "committ')
    ledger2 = DataEventLedger(lake_root=str(path))
    assert ledger2.is_committed("e1")
    assert not ledger2.is_committed("e_partial")
    # 截断后文件以合法 JSON 行结尾
    raw = (path / ".event_ledger.jsonl").read_bytes()
    assert raw.rstrip(b"\n").endswith(b"}")
    # 恢复后仍可正常提交
    e2 = _ev("e2", snapshot_before="s1", snapshot_after="s2")
    status, _ = ledger2.begin(e2)
    assert status == "ok"


def test_r14_append_failure_propagates(tmp_path, monkeypatch):
    """append OSError 不再被吞：落盘失败必须上抛，绝不静默继续。"""
    lake = tmp_path / "lake"
    lake.mkdir(parents=True)
    ledger = DataEventLedger(lake_root=str(lake))
    ev = _ev("e1", snapshot_before="s0", snapshot_after="s1")
    status, token = ledger.begin(ev)
    assert status == "ok"

    def _boom(self, rec):
        raise OSError("disk full")

    monkeypatch.setattr(DataEventLedger, "_append_unlocked", _boom)
    with pytest.raises(OSError, match="disk full"):
        ledger.commit(ev, token)


def test_r14_rejected_event_retry_allowed(tmp_path):
    """partial failure → reject；重跑同一 event 允许重新 begin（幂等收敛）。"""
    ledger = DataEventLedger(lake_root=str(tmp_path / "lake"))
    ev = _ev("e1", snapshot_before="s0", snapshot_after="s1")
    status, token = ledger.begin(ev)
    assert status == "ok"
    ledger.reject(ev, "2 downstream factor(s) failed", token=token)
    assert not ledger.is_committed("e1")
    # 重试：重新 begin（pending 覆盖旧的 rejected），允许再次处理
    status2, token2 = ledger.begin(ev)
    assert status2 == "ok"
    ledger.commit(ev, token2)
    assert ledger.is_committed("e1")
