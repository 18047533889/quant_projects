"""#7 第七轮 final closure（用户 12 项新增问题/二阶回归）回归测试。

覆盖：
    item 1   mutation lock 续租/release 的 TOCTOU（fd inode 身份）
    item 2   StorageSpec URI scheme → backend（不再 fallback LOCAL；矛盾拒绝）
    item 3   PathAuthorizer 按请求 namespace 延迟解析
    item 4   SemanticField.join_policy strict enum（拼写错 fail-closed）
    item 5   next_bar 消费 effective segments（美股 early close）
    item 6   未知 market fail-closed（不再静默当美国市场）
    item 7   availability_latency 只接受 exact int（1.9 拒绝）
    item 8   strict_sequence ordered 拆规则（fields/order_by 拒绝 set/dict）
    item 9   TimePartitionSpec / PartitionSpec __post_init__（typo 不 fallback）
    item 10  无 manifest 日历缓存用文件 snapshot token（数据更新即失效）
    item 11  QueryBudget 自身 invariant validation（负数/NaN/bool 拒绝）
    item 12  StorageSpec.options 深冻结
"""

import datetime as _dt
import json
import os
from pathlib import Path

import pytest

from data_access.core.exceptions import ValidationError


# ---------------------------------------------------------------------------
# 1) mutation lock：续租经 fd（inode 身份），旧 owner 绝不动新 inode
# ---------------------------------------------------------------------------


def test_mutation_lock_renew_does_not_corrupt_new_inode(tmp_path):
    from data_access.write.mutation_lock import _renew_lease

    lock = tmp_path / ".data-access.mutation.lock"
    # A 持有锁（fd = A 的 inode）
    fd_a = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
    try:
        payload_a = {
            "pid": os.getpid(), "host": "x", "transaction_id": "AAAA",
            "acquired_at": 0, "lease_until": 1,
        }
        os.write(fd_a, (json.dumps(payload_a) + "\n").encode())
        os.fsync(fd_a)
        # B 打破 A 的锁并重建（新 inode、新 transaction_id）
        lock.unlink()
        lock.write_text(
            json.dumps({"transaction_id": "OTHER", "lease_until": 9999999999}),
            encoding="utf-8",
        )
        # A 的心跳续租（拿着 A 的 fd）绝不能截断 B 的新锁
        renewed = _renew_lease(fd_a, lock, "AAAA", 60.0)
        assert renewed is False  # 锁已不属于 A → 停止续租
        assert json.loads(lock.read_text())["transaction_id"] == "OTHER"
        assert json.loads(lock.read_text())["lease_until"] == 9999999999
    finally:
        os.close(fd_a)


def test_mutation_lock_released_marker_is_breakable(tmp_path):
    import time

    from data_access.write.mutation_lock import _can_break_lock, mutation_lock

    lock = tmp_path / ".data-access.mutation.lock"
    with mutation_lock(tmp_path, lease_seconds=60, poll=0.01):
        pass  # release 写 released 标记（不再 unlink）
    assert lock.exists()
    assert json.loads(lock.read_text())["released"] is True
    # released 锁立即可打破（下次 acquisition 清理）
    assert _can_break_lock(lock, stale_after=3600, hard_break=7200) is True
    # 正常 acquisition 能继续拿锁（打破 released → 重建）
    with mutation_lock(tmp_path, lease_seconds=60, poll=0.01):
        assert json.loads(lock.read_text()).get("released") is None
        assert json.loads(lock.read_text())["transaction_id"]


# ---------------------------------------------------------------------------
# 2) StorageSpec：URI scheme → backend；矛盾拒绝
# ---------------------------------------------------------------------------


def test_storage_spec_infers_backend_from_scheme():
    from data_access.core.storage import StorageSpec

    assert StorageSpec.from_yaml("cos://bucket/path").type == "cos"
    assert StorageSpec.from_yaml("s3://bucket/path").type == "s3"
    assert StorageSpec.from_yaml("https://host/x").type == "http"
    assert StorageSpec.from_yaml({"uri": "cos://b/p"}).type == "cos"
    assert StorageSpec.from_yaml({"root": "/data/x"}).type == "local"
    assert StorageSpec.from_yaml("local").type == "local"


def test_storage_spec_rejects_unknown_scheme_and_contradiction():
    from data_access.core.storage import StorageSpec

    for bad in [
        "oss://bucket/x",            # 有 scheme 但未实现
        {"uri": "cos://b/p", "type": "local"},  # scheme 与 type 矛盾
        {"uri": "s3://b/p", "type": "cos"},
    ]:
        with pytest.raises(ValidationError):
            StorageSpec.from_yaml(bad)


def test_storage_spec_options_deep_frozen():
    from types import MappingProxyType

    from data_access.core.storage import StorageSpec

    spec = StorageSpec.from_yaml(
        {"type": "local", "options": {"nested": [1, 2], "k": "v"}}
    )
    assert isinstance(spec.options, MappingProxyType)
    assert spec.options["nested"] == (1, 2)  # list → tuple
    with pytest.raises(TypeError):
        spec.options["foo"] = "bar"
    with pytest.raises(TypeError):
        spec.options["nested"][0] = 99


# ---------------------------------------------------------------------------
# 3) PathAuthorizer：按请求 namespace 延迟解析
# ---------------------------------------------------------------------------


def test_path_authorizer_resolves_namespace_per_request():
    from data_access.core.namespace import namespace_scope
    from data_access.registry.paths import PathAuthorizer

    auth = PathAuthorizer(["/data/${RUN_NAMESPACE}/ws"])
    with namespace_scope("alpha"):
        assert auth.resolve_and_authorize("/data/alpha/ws/x.parquet")
        with pytest.raises(ValidationError):
            auth.resolve_and_authorize("/data/beta/ws/x.parquet")
    with namespace_scope("beta"):
        assert auth.resolve_and_authorize("/data/beta/ws/x.parquet")
        with pytest.raises(ValidationError):
            auth.resolve_and_authorize("/data/alpha/ws/x.parquet")


# ---------------------------------------------------------------------------
# 4) join_policy strict enum
# ---------------------------------------------------------------------------


def test_semantic_field_join_policy_strict():
    from data_access.read.semantic_catalog import parse_semantic_field

    f = parse_semantic_field(
        "x", {"dataset": "d", "physical_name": "c", "join_policy": "pit_asof_backward"}
    )
    assert f.join_policy == "pit_asof_backward"
    with pytest.raises(ValidationError):
        parse_semantic_field(
            "x", {"dataset": "d", "physical_name": "c", "join_policy": "pit_asof_backword"}
        )


# ---------------------------------------------------------------------------
# 5) next_bar 消费 effective segments（美股 early close）
# ---------------------------------------------------------------------------


def _ny(d: _dt.date, hhmm: str) -> _dt.datetime:
    from zoneinfo import ZoneInfo

    h, m = (int(p) for p in hhmm.split(":"))
    return _dt.datetime(d.year, d.month, d.day, h, m, tzinfo=ZoneInfo("America/New_York"))


def test_next_bar_respects_early_close():
    from data_access.read.session_calendar import MarketCalendar, build_us_session

    ec = {_dt.date(2026, 7, 3)}
    sess = build_us_session(early_close_dates=ec)
    cal = MarketCalendar(
        "us",
        trading_days=[_dt.date(2026, 7, 2), _dt.date(2026, 7, 3), _dt.date(2026, 7, 6)],
        session=sess,
    )
    d = _dt.date(2026, 7, 3)
    # 13:00 实际收盘 → next_bar 必须跳下一交易日 09:30（不能 13:01）
    assert cal.available_from(_ny(d, "13:00"), "next_bar") == _dt.datetime(2026, 7, 6, 9, 30)
    # 12:59 → 13:00（early-close 最后一根）
    assert cal.available_from(_ny(d, "12:59"), "next_bar") == _dt.datetime(2026, 7, 3, 13, 0)
    # 13:30（常规 09:30-16:00 内但已收盘）→ 下一交易日
    assert cal.available_from(_ny(d, "13:30"), "next_bar") == _dt.datetime(2026, 7, 6, 9, 30)
    # elapsed_index 日期感知
    assert sess.elapsed_index(_dt.time(14, 0), on=d) is None
    assert sess.elapsed_index(_dt.time(14, 0)) == 270


# ---------------------------------------------------------------------------
# 6) 未知 market fail-closed
# ---------------------------------------------------------------------------


def test_canonicalize_market_rejects_unknown():
    from data_access.read.session_calendar import _calendar_dataset_for, canonicalize_market

    assert canonicalize_market("nyse") == "us"
    assert canonicalize_market("a_share") == "ashare"
    for bad in ["europe", "abc", "uss"]:
        with pytest.raises(ValidationError):
            canonicalize_market(bad)
    with pytest.raises(ValidationError):
        _calendar_dataset_for("europe")


# ---------------------------------------------------------------------------
# 7) availability_latency 只接受 exact int
# ---------------------------------------------------------------------------


def test_latency_rejects_float():
    from data_access.read.temporal_join import parse_join_spec

    assert parse_join_spec({"policy": "pit_asof", "availability_latency": 3}).availability_latency == 3
    assert parse_join_spec({"availability_latency": "2"}).availability_latency == 2
    for bad in [1.9, "1.9", 0.5, True, -1, "-1"]:
        with pytest.raises(ValidationError):
            parse_join_spec({"policy": "asof", "availability_latency": bad})


# ---------------------------------------------------------------------------
# 8) strict_sequence ordered 拆规则
# ---------------------------------------------------------------------------


def test_strict_sequence_ordered_split():
    from data_access.read.data_request import DataRequest
    from data_access.read.predicate import strict_sequence

    # ordered：set/dict/generator 拒绝，list/tuple 保序
    for bad in [{"a", "b"}, {"a": 1}, (x for x in ["a"])]:
        with pytest.raises(ValidationError):
            strict_sequence(bad, name="x", ordered=True)
    assert strict_sequence(["b", "a"], name="x", ordered=True) == ("b", "a")
    # unordered：set 接受但 canonical 排序
    assert strict_sequence({"zz", "aa"}, name="x") == ("aa", "zz")
    with pytest.raises(ValidationError):
        strict_sequence({"k": 1}, name="x")
    # DataRequest：fields/order_by 顺序是语义 → set 拒绝
    with pytest.raises(ValidationError):
        DataRequest(fields={"close", "open"})


# ---------------------------------------------------------------------------
# 9) TimePartitionSpec / PartitionSpec __post_init__
# ---------------------------------------------------------------------------


def test_partition_spec_self_validating():
    from data_access.read.partition_planner import PartitionSpec, TimePartitionSpec

    TimePartitionSpec(frequency="daily")
    with pytest.raises(ValidationError):
        TimePartitionSpec(frequency="daliy")  # typo 不再静默 fallback daily
    with pytest.raises(ValidationError):
        PartitionSpec(time=TimePartitionSpec(frequency="daliy"))
    with pytest.raises(ValidationError):
        PartitionSpec(hive=(1, 2))


# ---------------------------------------------------------------------------
# 10) 无 manifest 日历缓存用文件 snapshot token
# ---------------------------------------------------------------------------


def test_calendar_file_token_tracks_data_version(tmp_path):
    from data_access.read.session_calendar import _calendar_file_token

    cal_dir = tmp_path / "Calendar"
    cal_dir.mkdir(parents=True)
    (cal_dir / "cal.parquet").write_bytes(b"v1")

    class FakeDS:
        root = str(cal_dir)
        glob = "*.parquet"

    class FakeReg:
        def get(self, name):
            return FakeDS()

    class FakeStore:
        registry = FakeReg()

    t1 = _calendar_file_token(FakeStore(), "us")
    assert t1 and t1.startswith("files:")
    (cal_dir / "cal.parquet").write_bytes(b"v2-updated")
    import time

    time.sleep(0.01)  # mtime_ns 分辨率内确保变化
    t2 = _calendar_file_token(FakeStore(), "us")
    assert t2 != t1
    # remote/未同步 → None（调用方兜底）
    import shutil

    shutil.rmtree(cal_dir)
    assert _calendar_file_token(FakeStore(), "us") is None


# ---------------------------------------------------------------------------
# 11) QueryBudget 自身 invariant validation
# ---------------------------------------------------------------------------


def test_query_budget_self_validating():
    from data_access.read.query_budget import QueryBudget

    QueryBudget(max_rows=1000)
    QueryBudget(max_elapsed_ms=5000.5)
    for kw in [
        dict(max_rows=-1),
        dict(max_rows=True),
        dict(max_scan_files=-5),
        dict(max_elapsed_ms=float("nan")),
        dict(max_elapsed_ms=float("inf")),
        dict(max_elapsed_ms=True),
        dict(max_result_bytes=0),
        dict(require_columns="yes"),
    ]:
        with pytest.raises(ValidationError):
            QueryBudget(**kw)
