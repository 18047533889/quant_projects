# -*- coding: utf-8 -*-
"""R39-P1-PERF-027: spool threshold 自适应 —— 行为测试（真实探针，非 grep）。

覆盖：
  - :func:`decide_spool_or_keep` 各分支：小 shard KEEP、writer 快 + 内存足 KEEP、
    merge 很近 + 内存足 KEEP、否则 SPOOL；writer 队列不富余 / merge 很远 → SPOOL。
  - :func:`adaptive_spool_threshold`：默认回基线 512MB、环境变量覆盖、内存/磁盘
    推导上下浮动。
  - ``ShardExecutor.spool_or_keep``：传 policy 时走新路径（计数器变化、KEEP/SPOOL
    都覆盖旧 512MB 阈值）；不传 policy 时行为与原 512MB 完全一致。
  - :func:`default_spool_policy_factory`：取不到任何信号 → 诚实 fail-safe 固定阈值；
    有 broker live_headroom / sink 队列余量 / merge lifetime → 自适应。
  - policy SPOOL + 低 spool 盘吞吐 → 优先 parquet。
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pandas as pd
import pytest

import factor_engine.runtime.shard_executor as se
import factor_engine.runtime.spool_policy as sp
from factor_engine.runtime.shard_executor import ArrowSpoolRef, ShardExecutor, SpooledShard


# ---------------------------------------------------------------------------
# 测试工具
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_counters():
    sp.reset_spool_policy_counters()
    yield
    sp.reset_spool_policy_counters()


def _series(lo: str, hi: str) -> pd.Series:
    idx = pd.MultiIndex.from_product(
        [pd.date_range(lo, hi, freq="D"), ["A001", "A002"]],
        names=["timestamp", "instrument"],
    )
    return pd.Series([float(i % 3) for i in range(len(idx))], index=idx)


def _decision(spool_or_keep_result: bool, reason: str = "test") -> sp.SpoolDecision:
    return sp.SpoolDecision(
        decision=sp.DECISION_SPOOL if spool_or_keep_result else sp.DECISION_KEEP,
        reason=reason,
        threshold_bytes=sp.SPOOL_THRESHOLD_BYTES_BASELINE,
        adaptive_threshold=True,
    )


class _FakeQueue:
    def __init__(self, max_bytes: int, current_bytes: int = 0) -> None:
        self.max_bytes = max_bytes
        self.current_bytes = current_bytes


class _FakeSink:
    """最小 sink 替身：只暴露 _worker_queues / queue 的字节容量。"""

    def __init__(self, max_bytes: int = 4 * 1024**3, current_bytes: int = 0) -> None:
        q = _FakeQueue(max_bytes, current_bytes)
        self._worker_queues = [q]
        self.queue = q


class _FakeBroker:
    def __init__(self, live_headroom: int | None, disk_busy: float = 0.0) -> None:
        self._lh = live_headroom
        self._busy = disk_busy

    def snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(live_headroom=self._lh, disk_busy=self._busy)


def _patch_no_signals(monkeypatch) -> None:
    """让所有 best-effort 探测都失败 → factory 走 fail-safe。"""
    import factor_engine.runtime.resource_governor as rg

    monkeypatch.setattr(rg, "spill_disk_speed_class", lambda: "unknown")
    monkeypatch.setattr(rg, "live_memory_headroom_bytes", lambda: None)
    monkeypatch.delenv(sp.ENV_MERGE_LIFETIME, raising=False)


# ---------------------------------------------------------------------------
# decide_spool_or_keep：各分支
# ---------------------------------------------------------------------------


def test_decide_small_shard_keep():
    d = sp.decide_spool_or_keep(
        16 * 1024**2,  # 16MB < 64MB
        live_headroom=None,
        writer_queue_headroom=None,
        spool_disk_throughput_bytes_per_s=500 * 1024**2,
        estimated_merge_lifetime_s=None,
    )
    assert d.decision == sp.DECISION_KEEP
    assert d.reason == "small_shard_below_min_keep"


def test_decide_writer_fast_and_memory_ample_keep():
    d = sp.decide_spool_or_keep(
        200 * 1024**2,  # > 64MB
        live_headroom=8 * 1024**3,
        writer_queue_headroom=4 * 1024**3,
        spool_disk_throughput_bytes_per_s=None,
        estimated_merge_lifetime_s=None,
    )
    assert d.decision == sp.DECISION_KEEP
    assert d.reason == "writer_fast_memory_ample"


def test_decide_merge_imminent_and_memory_ample_keep():
    d = sp.decide_spool_or_keep(
        200 * 1024**2,
        live_headroom=8 * 1024**3,
        writer_queue_headroom=0,  # writer 队列不富余，但 merge 近且内存足 → KEEP
        spool_disk_throughput_bytes_per_s=None,
        estimated_merge_lifetime_s=1.0,  # < 5s
    )
    assert d.decision == sp.DECISION_KEEP
    assert d.reason == "merge_imminent_memory_ample"


def test_decide_otherwise_spool():
    d = sp.decide_spool_or_keep(
        200 * 1024**2,
        live_headroom=None,
        writer_queue_headroom=None,
        spool_disk_throughput_bytes_per_s=None,
        estimated_merge_lifetime_s=None,
    )
    assert d.decision == sp.DECISION_SPOOL
    assert d.reason == "spool_to_disk"


def test_decide_writer_queue_not_plentiful_spools():
    # writer 队列余量 < shard（哪怕 live_headroom 富余）→ SPOOL
    d = sp.decide_spool_or_keep(
        200 * 1024**2,
        live_headroom=8 * 1024**3,
        writer_queue_headroom=50 * 1024**2,
        spool_disk_throughput_bytes_per_s=None,
        estimated_merge_lifetime_s=None,
    )
    assert d.decision == sp.DECISION_SPOOL


def test_decide_merge_far_spools():
    # merge 很远（>= 5s）且 writer 不富余 → SPOOL
    d = sp.decide_spool_or_keep(
        200 * 1024**2,
        live_headroom=8 * 1024**3,
        writer_queue_headroom=0,
        spool_disk_throughput_bytes_per_s=None,
        estimated_merge_lifetime_s=30.0,
    )
    assert d.decision == sp.DECISION_SPOOL


# ---------------------------------------------------------------------------
# adaptive_spool_threshold：默认 / 环境覆盖 / 推导
# ---------------------------------------------------------------------------


def test_adaptive_threshold_default_baseline_512mb(monkeypatch):
    monkeypatch.delenv(sp.ENV_SPOOL_THRESHOLD, raising=False)
    assert sp.adaptive_spool_threshold() == 512 * 1024**2
    # 无信号 → 基线不变
    assert sp.adaptive_spool_threshold(None, None) == 512 * 1024**2


def test_adaptive_threshold_env_override(monkeypatch):
    monkeypatch.setenv(sp.ENV_SPOOL_THRESHOLD, str(128 * 1024**2))
    assert sp.adaptive_spool_threshold() == 128 * 1024**2
    # 环境覆盖优先于推导
    assert sp.adaptive_spool_threshold(headroom_fraction=0.9, disk_throughput=2 * 1024**3) == 128 * 1024**2


def test_adaptive_threshold_invalid_env_ignored(monkeypatch):
    monkeypatch.setenv(sp.ENV_SPOOL_THRESHOLD, "not-a-number")
    assert sp.adaptive_spool_threshold() == 512 * 1024**2


def test_adaptive_threshold_memory_high_doubles(monkeypatch):
    monkeypatch.delenv(sp.ENV_SPOOL_THRESHOLD, raising=False)
    # live headroom 相对 hard limit 比例 >= 0.5 → 阈值翻倍（更多留内存）
    assert sp.adaptive_spool_threshold(headroom_fraction=0.75) == 1024 * 1024**2


def test_adaptive_threshold_memory_tight_halves(monkeypatch):
    monkeypatch.delenv(sp.ENV_SPOOL_THRESHOLD, raising=False)
    assert sp.adaptive_spool_threshold(headroom_fraction=0.10) == 256 * 1024**2


def test_adaptive_threshold_disk_fast_halves(monkeypatch):
    monkeypatch.delenv(sp.ENV_SPOOL_THRESHOLD, raising=False)
    # 盘快（>= 1GB/s）→ spool 更划算 → 阈值减半（更早 spool）
    assert sp.adaptive_spool_threshold(disk_throughput=2 * 1024**3) == 256 * 1024**2


def test_adaptive_threshold_disk_slow_doubles(monkeypatch):
    monkeypatch.delenv(sp.ENV_SPOOL_THRESHOLD, raising=False)
    assert sp.adaptive_spool_threshold(disk_throughput=50 * 1024**2) == 1024 * 1024**2


def test_adaptive_threshold_floor_at_min_keep(monkeypatch):
    monkeypatch.delenv(sp.ENV_SPOOL_THRESHOLD, raising=False)
    # 组合推导结果不能低于 MIN_KEEP_THRESHOLD_BYTES（64MB）
    got = sp.adaptive_spool_threshold(headroom_fraction=0.10, disk_throughput=2 * 1024**3)
    assert got >= sp.MIN_KEEP_THRESHOLD_BYTES


# ---------------------------------------------------------------------------
# spool_or_keep：policy 路径 vs 原固定 512MB 路径
# ---------------------------------------------------------------------------


def test_spool_or_keep_no_policy_fixed_threshold_behavior(monkeypatch):
    # 默认路径：spool_threshold 参数仍为 512MB，行为不变
    sig = inspect.signature(ShardExecutor.spool_or_keep)
    assert sig.parameters["spool_threshold"].default == 512 * 1024**2
    assert se.SPOOL_THRESHOLD_BYTES == 512 * 1024**2

    ex = ShardExecutor()
    # < 512MB → KEEP（原样返回）
    monkeypatch.setattr(se, "_estimate_bytes", lambda result: 256 * 1024**2)
    out = ex.spool_or_keep("dummy-keep")
    assert out == "dummy-keep"
    assert ex.summary()["spooled"] == 0
    # > 512MB → SPOOL
    monkeypatch.setattr(se, "_estimate_bytes", lambda result: 600 * 1024**2)
    out2 = ex.spool_or_keep(_series("2024-01-01", "2024-01-10"))
    assert isinstance(out2, SpooledShard)
    assert ex.summary()["spooled"] == 1
    # 无 policy → 不记录 spool_policy 计数器
    assert sp.get_spool_decision_count() == 0
    assert sp.get_spool_keep_count() == 0


def test_spool_or_keep_policy_keep_overrides_fixed_threshold(monkeypatch):
    ex = ShardExecutor()
    # 600MB > 512MB（固定阈值会 SPOOL），但 policy 说 KEEP → 保留
    monkeypatch.setattr(se, "_estimate_bytes", lambda result: 600 * 1024**2)
    policy = lambda n: _decision(False, reason="test_keep")
    out = ex.spool_or_keep(_series("2024-01-01", "2024-01-10"), policy=policy)
    assert isinstance(out, pd.Series)
    assert ex.summary()["spooled"] == 0
    assert sp.get_spool_keep_count() == 1
    assert sp.get_spool_decision_count() == 0


def test_spool_or_keep_policy_spool_overrides_fixed_threshold(monkeypatch):
    ex = ShardExecutor()
    # 100MB < 512MB（固定阈值会 KEEP），但 policy 说 SPOOL → 落盘
    monkeypatch.setattr(se, "_estimate_bytes", lambda result: 100 * 1024**2)
    policy = lambda n: _decision(True, reason="test_spool")
    out = ex.spool_or_keep(_series("2024-01-01", "2024-01-10"), policy=policy)
    assert isinstance(out, SpooledShard)
    assert ex.summary()["spooled"] == 1
    assert sp.get_spool_decision_count() == 1
    assert sp.get_spool_keep_count() == 0
    # adaptive_threshold=True → 自适应计数 +1
    assert sp.get_adaptive_threshold_applied_count() == 1


def test_spool_or_keep_policy_decision_counters_accumulate(monkeypatch):
    ex = ShardExecutor()
    monkeypatch.setattr(se, "_estimate_bytes", lambda result: 100 * 1024**2)
    # 2×SPOOL + 1×KEEP
    ex.spool_or_keep(_series("2024-01-01", "2024-01-05"), policy=lambda n: _decision(True))
    ex.spool_or_keep(_series("2024-02-01", "2024-02-05"), policy=lambda n: _decision(True))
    ex.spool_or_keep(_series("2024-03-01", "2024-03-05"), policy=lambda n: _decision(False))
    assert sp.get_spool_decision_count() == 2
    assert sp.get_spool_keep_count() == 1
    assert sp.get_adaptive_threshold_applied_count() == 3


def test_spool_or_keep_policy_arrow_roundtrip_still_works():
    # policy SPOOL + 吞吐高（>= 100MB/s）→ Arrow IPC 路径保持 PERF-024 语义
    ex = ShardExecutor()
    policy = lambda n: sp.SpoolDecision(
        decision=sp.DECISION_SPOOL,
        reason="test",
        threshold_bytes=sp.SPOOL_THRESHOLD_BYTES_BASELINE,
        spool_disk_throughput=500 * 1024**2,
        adaptive_threshold=True,
    )
    out = ex.spool_or_keep(_series("2024-01-01", "2024-01-10"), policy=policy)
    assert isinstance(out, ArrowSpoolRef)
    loaded = ex.load_partial(out)
    pd.testing.assert_series_equal(loaded.sort_index(), _series("2024-01-01", "2024-01-10").sort_index(), check_names=False)


def test_spool_or_keep_policy_slow_disk_prefers_parquet(monkeypatch):
    # policy SPOOL + 低 spool 盘吞吐（< 100MB/s）→ 优先 parquet（即使可 Arrow 转换）
    ex = ShardExecutor()
    monkeypatch.setattr(se, "_estimate_bytes", lambda result: 100 * 1024**2)
    policy = lambda n: sp.SpoolDecision(
        decision=sp.DECISION_SPOOL,
        reason="test",
        threshold_bytes=sp.SPOOL_THRESHOLD_BYTES_BASELINE,
        spool_disk_throughput=30 * 1024**2,
        adaptive_threshold=True,
    )
    out = ex.spool_or_keep(_series("2024-01-01", "2024-01-10"), policy=policy)
    assert isinstance(out, SpooledShard)
    assert not isinstance(out, ArrowSpoolRef)
    assert out.spool_format == "parquet"


# ---------------------------------------------------------------------------
# default_spool_policy_factory：best-effort 探测 + 诚实 fail-safe
# ---------------------------------------------------------------------------


def test_default_factory_fail_safe_fixed_threshold(monkeypatch):
    _patch_no_signals(monkeypatch)
    policy = sp.default_spool_policy_factory()
    # 取不到任何信号 → 原固定 512MB 语义
    keep = policy(256 * 1024**2)
    assert keep.decision == sp.DECISION_KEEP
    assert keep.reason == "fail_safe_fixed_threshold"
    assert keep.adaptive_threshold is False
    spooled = policy(600 * 1024**2)
    assert spooled.decision == sp.DECISION_SPOOL
    assert spooled.reason == "fail_safe_fixed_threshold"
    assert spooled.threshold_bytes == 512 * 1024**2


def test_default_factory_broker_live_headroom_derived():
    # broker 提供 live_headroom + 显式 writer 余量 → writer 快 + 内存足 → KEEP
    broker = _FakeBroker(live_headroom=8 * 1024**3, disk_busy=0.1)
    policy = sp.default_spool_policy_factory(
        broker=broker, writer_queue_headroom=4 * 1024**3
    )
    d = policy(200 * 1024**2)
    assert d.decision == sp.DECISION_KEEP
    assert d.live_headroom == 8 * 1024**3
    assert d.writer_queue_headroom == 4 * 1024**3
    assert d.adaptive_threshold is True


def test_default_factory_sink_writer_queue_headroom():
    # sink 队列余量（max - current）被读取
    sink = _FakeSink(max_bytes=4 * 1024**3, current_bytes=3 * 1024**3)
    policy = sp.default_spool_policy_factory(
        sink=sink, live_headroom=8 * 1024**3
    )
    d = policy(200 * 1024**2)
    # 队列余量 = 1GB > 200MB，live 8GB > 200MB → writer fast → KEEP
    assert d.writer_queue_headroom == 1 * 1024**3
    assert d.decision == sp.DECISION_KEEP


def test_default_factory_merge_imminent_keep(monkeypatch):
    _patch_no_signals(monkeypatch)
    # 显式 merge lifetime（< 5s）+ live_headroom 足 → merge imminent KEEP
    policy = sp.default_spool_policy_factory(
        live_headroom=8 * 1024**3,
        estimated_merge_lifetime_s=2.0,
    )
    d = policy(200 * 1024**2)
    assert d.decision == sp.DECISION_KEEP
    assert d.reason == "merge_imminent_memory_ample"


def test_default_factory_env_merge_lifetime(monkeypatch):
    _patch_no_signals(monkeypatch)
    monkeypatch.setenv(sp.ENV_MERGE_LIFETIME, "1.5")
    policy = sp.default_spool_policy_factory(live_headroom=8 * 1024**3)
    d = policy(200 * 1024**2)
    assert d.estimated_merge_lifetime == 1.5
    assert d.decision == sp.DECISION_KEEP


def test_default_factory_writer_not_plentiful_spools(monkeypatch):
    _patch_no_signals(monkeypatch)
    broker = _FakeBroker(live_headroom=8 * 1024**3, disk_busy=0.1)
    policy = sp.default_spool_policy_factory(broker=broker, writer_queue_headroom=10 * 1024**2)
    d = policy(200 * 1024**2)
    # writer 队列余量 10MB < 200MB → 不富余 → SPOOL
    assert d.decision == sp.DECISION_SPOOL
    assert d.reason == "spool_to_disk"


# ---------------------------------------------------------------------------
# 端到端：factory policy 接入 spool_or_keep（计数器 + 行为）
# ---------------------------------------------------------------------------


def test_spool_or_keep_with_factory_policy_integration(monkeypatch):
    _patch_no_signals(monkeypatch)
    ex = ShardExecutor()
    monkeypatch.setattr(se, "_estimate_bytes", lambda result: 100 * 1024**2)
    policy = sp.default_spool_policy_factory(
        broker=_FakeBroker(live_headroom=8 * 1024**3),
        writer_queue_headroom=4 * 1024**3,
    )
    # writer fast + 内存足 → KEEP（尽管 100MB 走自适应；计数器 KEEP+1）
    out = ex.spool_or_keep(_series("2024-01-01", "2024-01-10"), policy=policy)
    assert isinstance(out, pd.Series)
    assert ex.summary()["spooled"] == 0
    assert sp.get_spool_keep_count() == 1

    # 换一个 memory 紧的 broker → SPOOL（计数器 SPOOL+1，自适应计数 +1）
    tight = sp.default_spool_policy_factory(
        broker=_FakeBroker(live_headroom=10 * 1024**2),
        writer_queue_headroom=4 * 1024**3,
    )
    out2 = ex.spool_or_keep(_series("2024-02-01", "2024-02-10"), policy=tight)
    assert isinstance(out2, SpooledShard)
    assert sp.get_spool_decision_count() == 1
    assert sp.get_adaptive_threshold_applied_count() == 2


def test_default_factory_fail_safe_used_in_spool_or_keep(monkeypatch):
    _patch_no_signals(monkeypatch)
    ex = ShardExecutor()
    monkeypatch.setattr(se, "_estimate_bytes", lambda result: 300 * 1024**2)
    policy = sp.default_spool_policy_factory()
    out = ex.spool_or_keep(_series("2024-01-01", "2024-01-10"), policy=policy)
    # 300MB < 512MB fail-safe → KEEP
    assert isinstance(out, pd.Series)
    # fail-safe 决策不是自适应阈值 → adaptive_threshold_applied_count 不增
    assert sp.get_adaptive_threshold_applied_count() == 0
    assert sp.get_spool_keep_count() == 1
