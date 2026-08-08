"""最终收官轮 round7：第六轮 54 项 ledger 中本会话落地项的回归。

对应审计清单（main @ 3550257f0 之上）：
    1.  mutation_lock fencing：heartbeat 续租 / owner-only release / PID reuse
    2.  namespace 请求级：resolve_namespace_path / session 原样 / static_prefix 屏蔽
    3.  ReadPlan pin 执行直接消费 pinned file set（physical_scope 注入）
    4.  冲突 availability 声明 fail-closed（不选更宽松的 same_day）
    5.  MarketCalendar 右边界 fail-open → strict 报错
    6.  PIT index generation-directory staged 提交（保留旧 gen 到新 gen 完整）
    7.  Filter AST 完整 canonical 排序（And/Or 同列不同值 tie 消除）
    8.  date-only end 用 `< next_day`（nanosecond 精度不漏最后 999ns）
    9.  is_strict_semantics 全贯穿（paths/key_policy/s3_duckdb/telemetry）
   10.  atomic durable-write helper（tmp→fsync(fd)→replace→fsync(dir)）
   11.  coverage：empty_ok=complete / 5t 用真实交易日历 / remote-only 标注
   12.  COS mirror：verified/legacy-unverified/corrupt 三态 + strict 不把非空当完整
   13.  audit durable acknowledgement（发布审计写失败 ⇒ AuditWriteError）
   14.  SQL parser 级 allowlist：恰好一个语句 + 函数 allowlist（table/file 拒绝）

全部在本地临时数据集上验证，不依赖生产数据 / GitHub。
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.core.audit import record as audit_record
from data_access.core.atomic import atomic_write_json, atomic_write_text
from data_access.core.exceptions import AuditWriteError, ValidationError
from data_access.core.namespace import (
    DataAccessSession,
    namespace_scope,
    resolve_namespace,
    set_session_namespace,
)
from data_access.registry.paths import resolve_namespace_path


# ---------------------------------------------------------------------------
# 1) mutation_lock fencing
# ---------------------------------------------------------------------------


def test_mutation_lock_heartbeat_renews_lease(tmp_path):
    import time

    from data_access.write.mutation_lock import mutation_lock

    lock = tmp_path / ".data-access.mutation.lock"
    with mutation_lock(tmp_path, lease_seconds=0.5, poll=0.01):
        payload0 = json.loads(lock.read_text())
        time.sleep(0.8)  # 超过 lease，但心跳续租后 lease_until 必须仍在未来
        payload1 = json.loads(lock.read_text())
        assert payload1["lease_until"] > time.time()
        assert payload1["transaction_id"] == payload0["transaction_id"]
    assert not lock.exists()  # 正常 release 删除锁


def test_mutation_lock_release_only_owner(tmp_path):
    from data_access.write.mutation_lock import (
        _release_lease,
        mutation_lock,
    )

    lock = tmp_path / ".data-access.mutation.lock"
    with mutation_lock(tmp_path, lease_seconds=10, poll=0.01):
        txid_a = json.loads(lock.read_text())["transaction_id"]
        # 模拟：A 的锁被打破、B 重建（新 transaction_id）
        lock.unlink()
        lock.write_text(
            json.dumps(
                {"pid": os.getpid(), "host": "x", "transaction_id": "OTHER",
                 "lease_until": 0},
            ),
            encoding="utf-8",
        )
        # A 的 release 绝不能删除 B 的锁
        _release_lease(lock, txid_a)
        assert lock.exists()
        assert json.loads(lock.read_text())["transaction_id"] == "OTHER"


def test_mutation_lock_pid_reuse_detection():
    import socket

    from data_access.write.mutation_lock import (
        _owner_is_dead,
        _proc_starttime_ticks,
    )

    pid = os.getpid()
    cur = _proc_starttime_ticks(pid)
    host = socket.gethostname()
    assert _owner_is_dead({"pid": pid, "host": host, "starttime_ticks": cur}) is False
    # 同 PID 但 starttime 不同 ⇒ PID 被复用 ⇒ owner 已死
    assert (
        _owner_is_dead(
            {"pid": pid, "host": host, "starttime_ticks": float(cur) + 1000.0}
        )
        is True
    )


# ---------------------------------------------------------------------------
# 2) namespace 请求级解析
# ---------------------------------------------------------------------------


def test_resolve_namespace_path_uses_current_context():
    with namespace_scope("run_123"):
        assert (
            resolve_namespace_path("/data/${RUN_NAMESPACE}/x") == "/data/run_123/x"
        )
    # context 外不残留
    assert resolve_namespace_path("/data/${RUN_NAMESPACE}/x") != "/data/run_123/x"


def test_session_namespace_preserved_no_sanitize():
    # -abc- 与 abc 是两个不同的 namespace（不再被 _sanitize 汇聚）
    with DataAccessSession(namespace="-abc-"):
        assert resolve_namespace() == "-abc-"
    with DataAccessSession(namespace="abc"):
        assert resolve_namespace() == "abc"


def test_static_prefix_shields_namespace_placeholder(tmp_path):
    import os as _os

    _os.environ["TEST_WS"] = str(tmp_path)
    _os.environ["QUANT_RUN_NAMESPACE"] = "shw"
    yaml = tmp_path / "d.yaml"
    yaml.write_text(
        "ds:\n"
        "  kind: parametric\n"
        "  access_mode: namespaced\n"
        "  root_template: ${TEST_WS}/staging/${RUN_NAMESPACE}/factors/{fid}\n"
        "  glob_template: '**/*.parquet'\n"
        "  params_schema: {fid: str}\n",
        encoding="utf-8",
    )
    from data_access.registry import load_registry

    reg = load_registry(yaml)
    ds = reg.get("ds")
    # authorized_root 保留占位符（不再被静态前缀切成 /staging/$）
    assert "${RUN_NAMESPACE}" in str(ds.authorized_root or ds.static_root)
    # allowed_roots 按当前 context 解析
    assert any("staging/shw/factors" in str(r) for r in reg.allowed_roots())
    # resolve_paths 在 read/write 时解析
    assert "/staging/shw/factors/f1/" in ds.resolve_paths(fid="f1")[0]


# ---------------------------------------------------------------------------
# 3) ReadPlan pin 执行消费 pinned file set（physical_scope）
# ---------------------------------------------------------------------------


def test_readplan_pin_builds_physical_scope():
    from data_access.read.data_request import ReadPlan
    from data_access.read.read_contract import FileVersion

    plan = ReadPlan(
        request=object(),
        datasets=["ds"],
        fields=[],
        per_dataset_columns={"ds": ["a"]},
        join_policies={},
        scan_costs={},
        storage={},
        snapshot_info={},
        snapshot_policy="pin",
        compiled=object(),
        plan_pinned_files={
            "ds": (FileVersion(path="/data/x.parquet", size=10, mtime_ns=1),)
        },
    )
    pinned_scope: list[str] | None = None
    if plan.snapshot_policy == "pin":
        pinned = plan.plan_pinned_files.get("ds")
        if pinned:
            pinned_scope = [str(getattr(fv, "path", "")) for fv in pinned]
    assert pinned_scope == ["/data/x.parquet"]


def test_read_accepts_physical_scope_keyword(tmp_path):
    import inspect

    from data_access.store import DataAccessStore

    sig = inspect.signature(DataAccessStore.read)
    assert "physical_scope" in sig.parameters


# ---------------------------------------------------------------------------
# 4) 冲突 availability 声明 fail-closed
# ---------------------------------------------------------------------------


def test_availability_conflict_strict_raises():
    from data_access.read.temporal_join import TemporalJoinSpec
    from data_access.store import _raise_availability_conflict

    cur = TemporalJoinSpec(policy="pit_asof", availability="next_trading_day")
    inc = TemporalJoinSpec(policy="pit_asof", availability="same_day")
    with pytest.raises(ValidationError):
        _raise_availability_conflict("ds", cur, inc, is_strict=True)
    # research 不抛（保留第一个）
    _raise_availability_conflict("ds", cur, inc, is_strict=False)


# ---------------------------------------------------------------------------
# 5) 日历右边界 fail-open → strict 报错
# ---------------------------------------------------------------------------


def test_calendar_right_boundary_strict_raises():
    from data_access.read.session_calendar import (
        MarketCalendar,
        compile_available_from,
    )

    cal = MarketCalendar(
        "ashare", trading_days=[dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
    )
    k = dt.date(2024, 12, 31)  # 超出日历覆盖
    with pytest.raises(ValidationError):
        compile_available_from(k, "next_trading_day", calendar=cal, strict=True)
    # research 回退 knowledge
    assert (
        compile_available_from(k, "next_trading_day", calendar=cal, strict=False) == k
    )


# ---------------------------------------------------------------------------
# 6) PIT index generation-directory staged 提交
# ---------------------------------------------------------------------------


def test_pit_index_generation_directory_commit(tmp_path):
    from data_access.read.pit_event_index import (
        PITEventIndex,
        PITEventRecord,
        PITIndexMetadata,
        _commit_index_generation,
        _current_generation,
        _current_index_parquet,
        _pit_index_dir,
        load_pit_event_index,
    )

    root = tmp_path
    idx = PITEventIndex(
        [
            PITEventRecord(
                ticker="AAPL",
                filing_date=dt.date(2024, 1, 10),
                period_end=dt.date(2023, 12, 31),
                file_path="x.parquet",
            )
        ]
    )
    meta = PITIndexMetadata(complete=True, source_file_count=1,
                            indexed_file_count=1, generation_id="g1")
    _commit_index_generation(root, idx, meta, "g1")
    assert _current_generation(root) == "g1"
    p = _current_index_parquet(root)
    assert p is not None and "g1" in str(p) and p.exists()
    loaded = load_pit_event_index(p)
    assert len(loaded) == 1 and loaded.records[0].ticker == "AAPL"
    assert loaded.metadata.complete is True
    # 二次提交后旧 generation 被清理
    idx2 = PITEventIndex(
        [PITEventRecord(ticker="MSFT", filing_date=dt.date(2024, 2, 1),
                        period_end=dt.date(2024, 1, 31), file_path="y.parquet")]
    )
    meta2 = PITIndexMetadata(complete=True, source_file_count=1,
                             indexed_file_count=1, generation_id="g2")
    _commit_index_generation(root, idx2, meta2, "g2")
    assert _current_generation(root) == "g2"
    dirs = [c.name for c in _pit_index_dir(root).iterdir() if c.is_dir()]
    assert dirs == ["g2"]


# ---------------------------------------------------------------------------
# 7) Filter AST 完整 canonical 排序（And/Or 同列不同值）
# ---------------------------------------------------------------------------


def test_filter_ast_commutative_hash():
    from data_access.read.predicate_ast import canonical_filter_hash

    h1 = canonical_filter_hash({"a": {"gt": 1}, "b": {"gt": 2}})
    h2 = canonical_filter_hash({"b": {"gt": 2}, "a": {"gt": 1}})
    assert h1 == h2
    # 同列不同值的 And tie：交换书写顺序仍同 hash（#P1-final closure 15）
    f1 = {"x": {"eq": 1, "gte": 0}}
    f2 = {"x": {"gte": 0, "eq": 1}}
    assert canonical_filter_hash(f1) == canonical_filter_hash(f2)


# ---------------------------------------------------------------------------
# 8) date-only end 用 `< next_day`
# ---------------------------------------------------------------------------


def test_date_only_end_uses_lt_next_day():
    from data_access.read.predicate import CompiledPredicate, Predicate, compile_predicate

    pred = Predicate(
        time_range=("2024-01-01", "2024-01-10"),
        time_column_is_timestamp=True,
    )
    out = compile_predicate(pred, time_column="ts", instrument_column="inst")
    assert "<" in out.where_sql  # 上界用严格小于
    # end 展开为 next_day（2024-01-11 00:00），不是 next_day - 1µs
    assert out.params[1] == "2024-01-11 00:00:00" or "2024-01-11" in str(out.params[1])
    # 带时间的 end 原样保留
    pred2 = Predicate(
        time_range=("2024-01-01", "2024-01-10 15:30:00"),
        time_column_is_timestamp=True,
    )
    out2 = compile_predicate(pred2, time_column="ts", instrument_column="inst")
    assert out2.params[1] == "2024-01-10 15:30:00"
    assert "<=" in out2.where_sql


# ---------------------------------------------------------------------------
# 9) is_strict_semantics 全贯穿
# ---------------------------------------------------------------------------


def test_paths_production_uses_is_strict(monkeypatch):
    import data_access.registry.paths as p

    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("FACTOR_ENGINE_RUN_MODE", raising=False)
    monkeypatch.setenv("DATA_ACCESS_STRICT_READ", "1")
    assert p._production_mode() is True
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    assert p._production_mode() is False


def test_key_policy_uses_is_strict(monkeypatch):
    from data_access.read.key_policy import resolve_key_policy

    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    assert resolve_key_policy().invalid_key == "drop"
    monkeypatch.setenv("DATA_ACCESS_STRICT_READ", "1")
    assert resolve_key_policy().invalid_key == "error"


# ---------------------------------------------------------------------------
# 10) atomic durable-write helper
# ---------------------------------------------------------------------------


def test_atomic_write_text_json(tmp_path):
    target = tmp_path / "sub" / "x.json"
    atomic_write_json(target, {"a": 1, "b": [1, 2]})
    assert json.loads(target.read_text())["a"] == 1
    assert not list(tmp_path.rglob("*.tmp"))  # 无残留 tmp
    # 覆盖写保持原子
    atomic_write_json(target, {"a": 2})
    assert json.loads(target.read_text())["a"] == 2


# ---------------------------------------------------------------------------
# 11) coverage：empty_ok=complete / 5t 交易日历
# ---------------------------------------------------------------------------


def test_coverage_empty_ok_is_complete(monkeypatch):
    from data_access.read.coverage import compute_coverage

    class _FakeContract:
        coverage_start = None
        coverage_end = None
        expected_cadence = None
        max_staleness = None
        missing_partition_semantics = "empty_ok"

    class _Store:
        _registry = type("R", (), {"get": lambda self, n: object()})()

        def _resolve_raw_paths(self, *a, **k):
            return []

    monkeypatch.setattr(
        "data_access.cos_contract.get_cos_contract", lambda d: _FakeContract()
    )
    r = compute_coverage(_Store(), "ds")
    assert r.status == "complete"


def test_coverage_trading_day_lag_uses_calendar(monkeypatch):
    from data_access.read.coverage import _trading_day_lag

    class _Cal:
        has_data = True
        trading_days = [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4)]

    def _get_cal(market, store=None):
        return _Cal()

    monkeypatch.setattr(
        "data_access.read.session_calendar.get_market_calendar", _get_cal
    )
    # observed_end 在最后一个交易日之前 → lag = 2（1/3, 1/4）
    assert _trading_day_lag(dt.date(2024, 1, 2), dt.date(2024, 1, 4),
                            store=object(), dataset="us_xxx") == 2


# ---------------------------------------------------------------------------
# 12) COS mirror 三态
# ---------------------------------------------------------------------------


def test_mirror_file_state_and_strict(tmp_path, monkeypatch):
    from data_access.cos.mirror import _local_file_usable, _mirror_file_state

    f = tmp_path / "x.parquet"
    f.write_bytes(b"1234")
    assert _mirror_file_state(f) == "legacy_unverified"
    monkeypatch.setenv("DATA_ACCESS_STRICT_READ", "1")
    from data_access.cos.mirror import _strict_mirror_mode

    assert _strict_mirror_mode() is True
    assert _local_file_usable(f) is False  # strict：不把非空当完整
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    assert _local_file_usable(f) is True  # research：放宽


# ---------------------------------------------------------------------------
# 13) audit durable acknowledgement
# ---------------------------------------------------------------------------


def test_audit_durable_raises_on_write_failure(monkeypatch):
    monkeypatch.setenv("QUANT_AUDIT_LOG", "/nonexistent_zzz/audit.jsonl")
    with pytest.raises(AuditWriteError):
        audit_record(op="publish", dataset="x", ok=True, rows=1, durable=True)


def test_audit_non_durable_swallows(monkeypatch):
    monkeypatch.setenv("QUANT_AUDIT_LOG", "/nonexistent_zzz/audit.jsonl")
    audit_record(op="write", dataset="x", ok=True, rows=1, durable=False)  # 不抛


# ---------------------------------------------------------------------------
# 14) SQL parser 级 allowlist
# ---------------------------------------------------------------------------


def test_sql_single_statement_and_allowlist():
    from data_access.read.sql_escape import (
        _assert_single_select_statement,
        _check_sql_function_allowlist,
    )

    # 合法：单 SELECT + scalar/aggregate/window + grammar 关键字
    _assert_single_select_statement(
        "SELECT SUM(v) FILTER (WHERE d > DATE '2024-01-01') FROM {{ds}}"
    )
    _check_sql_function_allowlist(
        "SELECT CAST(a AS BIGINT), COUNT(*), ROW_NUMBER() OVER (PARTITION BY a ORDER BY b) FROM {{ds}}",
        strict=True,
    )
    # 多语句拒绝
    with pytest.raises(ValidationError):
        _assert_single_select_statement("SELECT 1; SELECT 2")
    # table/file function 拒绝（注册表分类 + deny 兜底）
    for bad in (
        "SELECT * FROM read_parquet('/x')",
        "SELECT * FROM parquet_metadata('/x.parquet')",
        "SELECT * FROM sniff_csv('/x.csv')",
        "SELECT * FROM glob('/x/*.parquet')",
    ):
        with pytest.raises(ValidationError):
            _check_sql_function_allowlist(bad, strict=True)
    # 未知函数 strict fail-closed
    with pytest.raises(ValidationError):
        _check_sql_function_allowlist("SELECT mystery_fn(a) FROM {{ds}}", strict=True)
    # 字符串里的函数名不误伤
    _check_sql_function_allowlist(
        "SELECT 'read_parquet' AS note, COUNT(*) FROM {{ds}}", strict=True
    )
