"""audit 单元测试：落盘路径优先级、JSONL 格式、读事件默认不记、失败不影响调用方。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_access import audit


@pytest.fixture
def isolated_audit(tmp_path, monkeypatch):
    """把审计日志重定向到 tmp_path，并设好 namespace/operator，避免用真实环境。"""
    log_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("QUANT_AUDIT_LOG", str(log_path))
    monkeypatch.setenv("QUANT_RUN_NAMESPACE", "testns")
    monkeypatch.setenv("QUANT_OPERATOR", "tester@team")
    # 清 audit_reads 环境变量残留
    monkeypatch.delenv("QUANT_AUDIT_READS", raising=False)
    yield log_path


def _read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_write_event_recorded(isolated_audit):
    audit.record(op="write", dataset="ds1", ok=True, rows=100, mode="overwrite",
                 paths=["/tmp/a.parquet"], elapsed_ms=12.3)
    records = _read_lines(isolated_audit)
    assert len(records) == 1
    r = records[0]
    assert r["op"] == "write"
    assert r["dataset"] == "ds1"
    assert r["ok"] is True
    assert r["rows"] == 100
    assert r["mode"] == "overwrite"
    assert r["paths"] == ["/tmp/a.parquet"]
    assert r["elapsed_ms"] == 12.3
    assert r["namespace"] == "testns"
    assert r["operator"] == "tester@team"
    assert "ts" in r


def test_read_event_skipped_by_default(isolated_audit):
    audit.record(op="read", dataset="ds1", ok=True, rows=5)
    assert not isolated_audit.exists() or isolated_audit.read_text() == ""


def test_read_event_recorded_when_opted_in(isolated_audit, monkeypatch):
    monkeypatch.setenv("QUANT_AUDIT_READS", "true")
    audit.record(op="read", dataset="ds1", ok=True, rows=5)
    records = _read_lines(isolated_audit)
    assert len(records) == 1
    assert records[0]["op"] == "read"


def test_failure_record_includes_error(isolated_audit):
    audit.record(op="write", dataset="ds1", ok=False, error="boom!" * 50)
    r = _read_lines(isolated_audit)[0]
    assert r["ok"] is False
    assert r["error"].startswith("boom!")
    # 500 字符截断
    assert len(r["error"]) <= 500


def test_operator_none_when_unset(isolated_audit, monkeypatch):
    monkeypatch.delenv("QUANT_OPERATOR", raising=False)
    audit.record(op="write", dataset="ds1", ok=True)
    r = _read_lines(isolated_audit)[0]
    assert r["operator"] is None


def test_audit_failure_does_not_raise(tmp_path, monkeypatch):
    """审计写失败（比如磁盘只读）不能把主流程搞挂。"""
    # 指到一个不允许创建的路径
    monkeypatch.setenv("QUANT_AUDIT_LOG", "/nonexistent/path/that/cannot/exist/audit.jsonl")
    monkeypatch.setenv("QUANT_RUN_NAMESPACE", "testns")
    # 不抛就算通过
    audit.record(op="write", dataset="ds1", ok=True, rows=1)


def test_timer_measures_elapsed():
    with audit.AuditTimer() as t:
        import time
        time.sleep(0.01)
    assert t.elapsed_ms >= 10.0
