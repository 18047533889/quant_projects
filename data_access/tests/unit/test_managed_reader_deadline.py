"""read/managed_reader.py + engine deadline 取消单测。"""
from __future__ import annotations

import time

import pytest

from data_access.core.engine import DuckDBEngine
from data_access.core.exceptions import DeadlineExceeded
from data_access.read.managed_reader import ManagedBatchReader


def test_managed_reader_normal_completion():
    eng = DuckDBEngine(threads=2, enable_object_cache=False)
    eng._conn.execute("CREATE TABLE t AS SELECT range AS a FROM range(100)")
    reader = eng.execute_reader("SELECT * FROM t", batch_size=10)
    assert isinstance(reader, ManagedBatchReader)
    total = sum(b.num_rows for b in reader)
    assert total == 100
    assert reader._closed is True


def test_managed_reader_early_break_explicit_close():
    eng = DuckDBEngine(threads=2, enable_object_cache=False)
    eng._conn.execute("CREATE TABLE t AS SELECT range AS a FROM range(100)")
    reader = eng.execute_reader("SELECT * FROM t", batch_size=10)
    for _batch in reader:
        break
    # break 不会自动触发（迭代协议不通知），但显式 close 幂等释放
    assert reader._closed is False
    reader.close()
    assert reader._closed is True


def test_managed_reader_context_manager():
    eng = DuckDBEngine(threads=2, enable_object_cache=False)
    eng._conn.execute("CREATE TABLE t AS SELECT range AS a FROM range(10)")
    with eng.execute_reader("SELECT * FROM t", batch_size=5) as reader:
        assert sum(b.num_rows for b in reader) == 10
    assert reader._closed is True


def test_deadline_exceeded():
    eng = DuckDBEngine(threads=2, enable_object_cache=False)
    start = time.perf_counter()
    with pytest.raises(DeadlineExceeded):
        eng.execute_arrow(
            "SELECT SUM(x) FROM (SELECT range AS x FROM range(200000000))",
            deadline_ms=50,
        )
    assert (time.perf_counter() - start) < 5.0


def test_no_deadline_normal():
    eng = DuckDBEngine(threads=2, enable_object_cache=False)
    eng._conn.execute("CREATE TABLE t AS SELECT range AS a FROM range(100)")
    assert eng.execute_arrow("SELECT COUNT(*) FROM t").column(0)[0].as_py() == 100
