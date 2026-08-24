# -*- coding: utf-8 -*-
"""R39-PERF-040+045 收口：sink join timeout 随 batch_size 缩放。

回归（真实 bug，benchmark 暴露）：
    batch_size 默认 64 后，一次 writer flush（execute_materialize_batch 写
    64 个因子）可远超旧固定 10s。旧 ``finish()`` 的 ``join(timeout=10.0)``
    会把活着的慢 writer 误判为致命（R38 P0-045 abort generation），丢弃
    全部已接受结果。本测试锁定：
      1. ``join_timeout`` 默认 = max(10s, batch_size × 2s)；
      2. 显式 ``join_timeout`` 覆盖默认；
      3. writer 真死锁（永不返回）时仍能触发 fatal（P0-045 语义保留）；
      4. 慢但活的 writer（每项超过旧 10s 上限）在 batch=64 下不再被误杀。
"""
from __future__ import annotations

import threading
import time

import pytest

from factor_engine.runtime.streaming_result_sink import StreamingResultSink, _PER_ITEM_JOIN_BUDGET_S


class _Item:
    """最小 ResultItem 替身（sink 只访问 .name/.bytes/.value/.meta）。"""

    def __init__(self, name: str, value: object = None, b: int = 8):
        self.name = name
        self.value = value
        self.bytes = b
        self.meta = {}


class _Rec:
    def __init__(self):
        self.calls: list[list] = []
        self.committed = 0


def _make_sink(batch_size, join_timeout=None, writer=None, writer_threads=1):
    rec = _Rec()

    def w(batch):
        rec.calls.append(list(batch))
        rec.committed += len(batch)

    sink = StreamingResultSink(
        writer=writer or w,
        batch_size=batch_size,
        writer_threads=writer_threads,
        join_timeout=join_timeout,
    )
    return sink, rec


def test_join_timeout_default_scales_with_batch_size():
    """batch_size=64 → 默认 join timeout = 64 × 2s = 128s（> 旧固定 10s）。"""
    sink, _ = _make_sink(batch_size=64)
    assert sink._join_timeout == 64 * _PER_ITEM_JOIN_BUDGET_S
    assert sink._join_timeout > 10.0


def test_join_timeout_small_batch_keeps_10s_floor():
    """batch_size=1 → 默认仍 10s（旧语义不变）。"""
    sink, _ = _make_sink(batch_size=1)
    assert sink._join_timeout == 10.0


def test_join_timeout_explicit_override():
    """显式 join_timeout 覆盖默认。"""
    sink, _ = _make_sink(batch_size=64, join_timeout=3.0)
    assert sink._join_timeout == 3.0


def test_slow_but_alive_writer_batch64_not_killed():
    """慢但活的 writer（每项 ~0.15s，64 项 batch 共 ~10s）不再被 10s 误杀。"""

    def slow_writer(batch):
        time.sleep(0.15 * len(batch))

    sink, rec = _make_sink(batch_size=64, writer=slow_writer)
    # 慢 writer 也必须记录（覆盖 _make_sink 默认 writer 的记录逻辑）
    committed = [0]

    def recording(batch):
        slow_writer(batch)
        committed[0] += len(batch)

    sink = StreamingResultSink(
        writer=recording,
        batch_size=64,
        writer_threads=1,
    )
    sink.start()
    for i in range(64):
        sink.submit(f"f{i}", object(), b=8)
    t0 = time.monotonic()
    sink.finish()  # 不抛异常 = 通过
    assert time.monotonic() - t0 >= 9.0  # 确实等到了慢 writer 写完
    assert committed[0] == 64


def test_dead_writer_still_fatal():
    """真死锁（writer 永不返回）仍触发 P0-045 fatal（join 超时 → abort）。"""
    marker = {"done": False}
    release = threading.Event()

    def dead_writer(batch):
        release.wait(60)

    sink, rec = _make_sink(batch_size=1, writer=dead_writer, join_timeout=1.0)
    sink.start()
    sink.submit("f0", object(), b=8)
    with pytest.raises(RuntimeError, match="alive after join"):
        sink.finish()
    marker["done"] = True
    assert marker["done"]


def test_small_fast_batch_unchanged():
    """batch_size=1 快 writer 正常路径不受影响。"""
    sink, rec = _make_sink(batch_size=1)
    sink.start()
    for i in range(3):
        sink.submit(f"f{i}", object(), b=8)
    sink.finish()
    assert rec.committed == 3
    assert sum(len(c) for c in rec.calls) == 3


def test_join_timeout_for_finish_covers_backlog_batch1():
    """batch=1 + 大量剩余：finish 动态放宽 join timeout，不误杀活 writer。

    回归基线场景（8e9893b5，R39 前）：batch=1 下 writer 逐项慢写，finish 时
    队列仍有大量剩余 → 旧固定 10s 必然 abort generation。
    """
    sink, rec = _make_sink(batch_size=1)
    assert sink._join_timeout == 10.0  # 默认下限
    for i in range(30):
        sink.submit(f"f{i}", object(), b=8)
    # 模拟 close 瞬间队列仍有 30 项剩余 → 预算 = 30 × 2s = 60s（> 10s）
    t = sink._join_timeout_for_finish()
    assert t >= 60.0
    assert t > 10.0


def test_backlog_slow_writer_batch1_finishes():
    """batch=1 慢 writer（0.2s/项，30 项 = 6s > 固定 10s 场景的反面）不误杀。"""

    def slow_writer(batch):
        time.sleep(0.2 * len(batch))

    committed = [0]

    def recording(batch):
        slow_writer(batch)
        committed[0] += len(batch)

    sink = StreamingResultSink(writer=recording, batch_size=1, writer_threads=1)
    sink.start()
    for i in range(30):
        sink.submit(f"f{i}", object(), b=8)
    t0 = time.monotonic()
    sink.finish()
    assert time.monotonic() - t0 >= 5.0  # 等到了全部写完
    assert committed[0] == 30
