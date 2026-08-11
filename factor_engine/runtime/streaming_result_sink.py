# -*- coding: utf-8 -*-
"""R27-105..109 + R33-P0-048..051: BoundedResultQueue + StreamingResultSink。

R33 升级
    - P0-048：队列改用 ``collections.deque``（O(1) pop，不再 list.pop(0) O(n)）。
    - P0-049：writer 状态机 ACTIVE / RETRYING / FAILED / DRAINED；**只** retry
      transient IO（TimeoutError/ConnectionError/OSError 等），permanent（schema/
      invalid factor/disk full/确定性 bug）→ FAILED → fatal。
    - P0-050：``finish`` 必须证明 ``accepted == committed + failed``、队列空、
      writer 线程全部终止、``fatal_error is None``，否则抛错（生产 run 不能谎报成功）。
    - P0-051：partition-aware writer —— 同一 partition 永远同一 writer 线程
      （无并发写同 factor/date 竞态），不同 partition 并行。
    - P0-046：``backpressure_ratio`` 供 scheduler admission 消费（已有字段保留）。
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

#: writer 状态机（R33-P0-049）
WS_ACTIVE = "ACTIVE"
WS_RETRYING = "RETRYING"
WS_FAILED = "FAILED"
WS_DRAINED = "DRAINED"

#: 只 retry transient IO；permanent（schema/invalid factor/disk full/确定性写 bug）
#: 直接 FAILED（R33-P0-049 retry taxonomy）。
_RETRYABLE_EXC = (
    TimeoutError,
    ConnectionError,
)
_RETRY_MARKERS = ("transient", "temporary lock", "remote retryable", "retryable")
_PERMANENT_MARKERS = (
    "schema",
    "invalid factor",
    "disk full",
    "no space",
    "deterministic",
)
_MAX_RETRIES = 3


def _bytes_of(value: Any) -> int:
    try:
        from runtime.resource_governor import estimate_object_bytes

        return estimate_object_bytes(value)
    except Exception:
        return 0


def _classify_write_error(exc: BaseException) -> str:
    """writer 错误分类：transient（可 retry）vs permanent（FAILED）。"""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if any(m in msg for m in _PERMANENT_MARKERS):
        return "permanent"
    if isinstance(exc, _RETRYABLE_EXC) or any(m in msg for m in _RETRY_MARKERS):
        return "transient"
    if "oom" in msg or "out of memory" in msg or name == "memoryerror":
        return "permanent"
    return "transient"  # 默认只重试有限次；超预算仍 FAILED（见 writer 循环）


def _should_flush(
    batch_count: int,
    batch_bytes: int,
    oldest_age_s: float,
    *,
    batch_size: int,
    target_batch_bytes: int | None,
    max_batch_age_s: float | None,
) -> bool:
    """R39-PERF-040 flush 谓词：bytes + factor_count + latency budget 任一触发。

    ``target_batch_bytes``/``max_batch_age_s`` 为 ``None`` 时对应维度不参与判断
    （保持旧 count-only 行为）。纯函数，便于单测。
    """
    if batch_count >= batch_size:
        return True
    if target_batch_bytes is not None and batch_bytes >= target_batch_bytes:
        return True
    if max_batch_age_s is not None and oldest_age_s >= max_batch_age_s:
        return True
    return False


@dataclass
class ResultItem:
    """一次结果交付（name + value + 元数据）。"""

    name: str
    value: Any
    bytes: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


class BoundedResultQueue:
    """按 **bytes** 上限的有界结果队列（R27-105）。

    R33-P0-048：内部用 ``collections.deque``（O(1) pop）。``backpressure_ratio``
    反映队列占用比例，scheduler 据此降 admission（R33-P0-046）。
    """

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max(1, int(max_bytes))
        self._items: deque[ResultItem] = deque()
        self._current_bytes = 0
        self._lock = threading.Condition()
        self._closed = False
        self._write_backpressure_seconds = 0.0

    @property
    def backpressure_ratio(self) -> float:
        with self._lock:
            return self._current_bytes / self.max_bytes if self.max_bytes else 0.0

    @property
    def current_bytes(self) -> int:
        with self._lock:
            return self._current_bytes

    @property
    def queued_count(self) -> int:
        with self._lock:
            return len(self._items)

    def put(self, item: ResultItem, *, timeout: float = 600.0) -> bool:
        """放入结果；队列满（bytes）时阻塞等待消费者（backpressure，R27-106）。"""
        if item.bytes <= 0:
            item.bytes = _bytes_of(item.value)
        with self._lock:
            if self._closed:
                return False
            while self._current_bytes + item.bytes > self.max_bytes:
                t0 = time.monotonic()
                if not self._lock.wait(timeout):
                    return False
                self._write_backpressure_seconds += time.monotonic() - t0
            self._items.append(item)
            self._current_bytes += item.bytes
            self._lock.notify()
            return True

    def get(self, *, timeout: float = 1.0) -> ResultItem | None:
        """取下一个结果；队列空且未 closed 时短暂阻塞。O(1) pop（deque）。"""
        deadline = time.monotonic() + timeout
        with self._lock:
            while not self._items and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._lock.wait(min(remaining, 0.2))
            if not self._items:
                return None
            item = self._items.popleft()
            self._current_bytes = max(0, self._current_bytes - item.bytes)
            self._lock.notify()
            return item

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._lock.notify_all()

    def set_target_bytes(self, new_target: int) -> None:
        """R38 P0-043（§17）：弹性缩容/放宽队列字节目标。

        缩容时不丢已有 items；producer 在 ``current + item.bytes > max_bytes``
        时阻塞等待消费降到新 target 以下；恢复后（target 变大）自动放宽。
        """
        with self._lock:
            self.max_bytes = max(1, int(new_target))
            self._lock.notify_all()

    def drain(self) -> list[ResultItem]:
        """取出全部剩余结果（run 结束收尾）。"""
        with self._lock:
            out = list(self._items)
            self._items.clear()
            self._current_bytes = 0
        return out

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_bytes": self.max_bytes,
                "current_bytes": self._current_bytes,
                "queued_count": len(self._items),
                "backpressure_ratio": round(self._current_bytes / self.max_bytes, 4) if self.max_bytes else 0.0,
                "write_backpressure_seconds": round(self._write_backpressure_seconds, 3),
                "closed": self._closed,
            }


class _WriterWorker:
    """R33-P0-049/051：单 writer 状态机 + retry budget。

    同一 worker 独占其 partition 集合（partition hash → worker 映射固定），
    保证同 partition 单 writer、不同 partition 并行。permanent 错误 → FAILED →
    记录 ``fatal_error``（scheduler/上层据此 abort）。transient 只重试
    ``_MAX_RETRIES`` 次。
    """

    def __init__(
        self,
        *,
        worker_id: int,
        writer: Callable[[list[ResultItem]], None],
        queue: BoundedResultQueue,
        batch_size: int,
        partition_key: Callable[[ResultItem], str] | None,
        on_fatal: Callable[[BaseException], None] | None = None,
        target_batch_bytes: int | None = None,
        max_batch_age_s: float | None = None,
    ) -> None:
        self.worker_id = worker_id
        self._writer = writer
        self._queue = queue
        self._batch_size = max(1, batch_size)
        self._target_batch_bytes = (
            max(1, int(target_batch_bytes)) if target_batch_bytes is not None else None
        )
        self._max_batch_age_s = (
            float(max_batch_age_s) if max_batch_age_s is not None else None
        )
        self._partition_key = partition_key
        # P0-023：worker fatal 时立即回调 sink（atomic set fatal + close queues），
        # 不等 finish()——否则 compute 还往死掉的 writer 队列塞结果，最后阻塞到
        # queue 满甚至等 600 秒。
        self._on_fatal = on_fatal
        self.state = WS_ACTIVE
        self.fatal_error: BaseException | None = None
        self.committed = 0
        self.failed = 0
        self.retried = 0
        self._lock = threading.Lock()
        self._retry_map: dict[str, int] = {}

    def _run(self) -> None:
        """writer 循环（由 sink 的线程调用）。

        R39-PERF-040：flush 条件 = factor_count **或** 累积 bytes **或**
        latency budget（batch 内最老 item 等待时长）任一触发，不再只看数量。
        """
        while True:
            item = self._queue.get()
            if item is None and self._queue._closed:
                break
            if item is None:
                continue
            batch = [item]
            batch_bytes = max(0, item.bytes)
            batch_started = time.monotonic()
            while not _should_flush(
                len(batch),
                batch_bytes,
                time.monotonic() - batch_started,
                batch_size=self._batch_size,
                target_batch_bytes=self._target_batch_bytes,
                max_batch_age_s=self._max_batch_age_s,
            ):
                nxt = self._queue.get(timeout=0.05)
                if nxt is None:
                    break
                batch.append(nxt)
                batch_bytes += max(0, nxt.bytes)
            self._write_batch(batch)
            if self.state == WS_FAILED:
                # fatal：本 worker 停止取新任务（其余 worker 仍可继续）。
                return

    def _write_batch(self, batch: list[ResultItem]) -> None:
        attempts = 0
        while attempts < _MAX_RETRIES:
            attempts += 1
            try:
                self._writer(batch)
                with self._lock:
                    self.committed += len(batch)
                    self.state = WS_ACTIVE
                return
            except Exception as exc:  # noqa: BLE001
                kind = _classify_write_error(exc)
                with self._lock:
                    self.retried += 1
                if kind == "permanent" or attempts >= _MAX_RETRIES:
                    with self._lock:
                        self.state = WS_FAILED
                        self.fatal_error = exc
                        self.failed += len(batch)
                    # P0-023：**立即**传播到 sink（atomic set fatal + close 全部
                    # queues + 通知），scheduler 下一次 submit 立刻被拒、停止 admission。
                    if self._on_fatal is not None:
                        try:
                            self._on_fatal(exc)
                        except Exception:
                            pass
                    return
                # transient：指数退避后重试同一 batch。
                time.sleep(min(0.05 * (2 ** attempts), 1.0))


class StreamingResultSink:
    """compute→writer 的流式 sink（R27-107 pipeline 重叠）。

    R33-P0-048..051：deque 队列 + 每 worker 状态机 + retry budget + fatal 传播 +
    ``finish`` 证明写完。partition-aware：同一 partition 永远同一 worker。
    """

    def __init__(
        self,
        *,
        writer: Callable[[list[ResultItem]], None],
        queue_bytes: int = 4 * 1024**3,
        batch_size: int = 1,
        writer_threads: int | None = None,
        partition_key: Callable[[ResultItem], str] | None = None,
        target_batch_bytes: int | None = None,
        max_batch_age_s: float | None = None,
    ) -> None:
        self._writer = writer
        self._batch_size = max(1, batch_size)
        self._target_batch_bytes = (
            max(1, int(target_batch_bytes)) if target_batch_bytes is not None else None
        )
        self._max_batch_age_s = (
            float(max_batch_age_s) if max_batch_age_s is not None else None
        )
        self._partition_key = partition_key
        # R39-PERF-041：writer_threads 自动选择。同 partition（无 partition_key 时
        # 视为单 partition 域）→ 单 writer；partition-aware → ``min(2, cpu)``，
        # 显式 ``writer_threads`` 覆盖自动选择（保持旧接口向后兼容：显式传 1 仍 1）。
        if writer_threads is None:
            if partition_key is None:
                writer_threads = 1
            else:
                writer_threads = min(2, os.cpu_count() or 1)
        self._writer_threads = max(1, int(writer_threads))
        self._threads: list[threading.Thread] = []
        # R33-P0-051：每 worker 一个独立队列（同 partition → 同 worker → 单 writer）。
        self._worker_queues: list[BoundedResultQueue] = [
            BoundedResultQueue(queue_bytes) for _ in range(self._writer_threads)
        ]
        # 兼容旧接口：``sink.queue`` 指向 worker-0 队列（单 writer 时即唯一队列）。
        self.queue = self._worker_queues[0]
        self._workers: list[_WriterWorker] = []
        self._lock = threading.Lock()
        self._accepted = 0
        self._fatal_error: BaseException | None = None
        self._drained = False
        self._started = False
        # R39-PERF-042：动态 least-loaded ownership。partition key 首次出现时
        # 交给当前负载最低的 worker，之后 pin（同一 partition 永不并发写）。
        self._partition_owners: dict[str, int] = {}
        self._route_lock = threading.Lock()

    def _least_loaded_writer(self) -> int:
        """当前队列字节占用最低的 worker（tie → 最小 id，确定性强）。"""
        return min(
            range(self._writer_threads),
            key=lambda i: (self._worker_queues[i].current_bytes, i),
        )

    def _route_worker(self, item: ResultItem) -> int:
        """R39-PERF-042：动态 least-loaded ownership（替代静态 hash）。

        同一 partition 首次出现 → 分配给 least-loaded worker 并 pin；此后该
        partition 恒同一 worker。partition epoch 结束（sink 收尾 drain）后随
        sink 释放，保证 ``same partition never concurrent write``。
        """
        if self._partition_key is None:
            return 0
        key = self._partition_key(item)
        with self._route_lock:
            owner = self._partition_owners.get(key)
            if owner is None:
                owner = self._least_loaded_writer()
                self._partition_owners[key] = owner
            return owner

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        for i in range(self._writer_threads):
            worker = _WriterWorker(
                worker_id=i,
                writer=self._writer,
                queue=self._worker_queues[i],
                batch_size=self._batch_size,
                partition_key=self._partition_key,
                on_fatal=self._set_fatal,
                target_batch_bytes=self._target_batch_bytes,
                max_batch_age_s=self._max_batch_age_s,
            )
            self._workers.append(worker)
            t = threading.Thread(target=worker._run, daemon=True, name=f"r27-writer-{i}")
            t.start()
            self._threads.append(t)

    @property
    def backpressure_ratio(self) -> float:
        """R33-P0-046：聚合 backpressure = 各 worker 队列占用最大值。"""
        if not self._worker_queues:
            return 0.0
        return max(q.backpressure_ratio for q in self._worker_queues)

    @property
    def fatal_error(self) -> BaseException | None:
        """P0-023：writer 已 fatal 的异常（scheduler 据此停止 admission）。"""
        with self._lock:
            return self._fatal_error

    def set_target_bytes(self, total: int) -> None:
        """R38 P0-043/044（§17）：弹性调整 sink 总字节目标。

        ``total`` 是全部 worker 共享的总额（P0-044：不是每 worker 各拿 full
        budget）——按 worker 数均分到各队列。缩容不丢已有 items。
        """
        total = max(1, int(total))
        per = max(1, total // max(1, len(self._worker_queues)))
        for q in self._worker_queues:
            q.set_target_bytes(per)

    def submit(self, name: str, value: Any, **meta: Any) -> bool:
        """R33-P0-049：提交结果。writer 已 FAILED 时拒绝新提交（fatal 传播）。"""
        if self._fatal_error is not None:
            return False
        item = ResultItem(name=name, value=value, meta=dict(meta))
        with self._lock:
            self._accepted += 1
        idx = self._route_worker(item)
        return self._worker_queues[idx].put(item)

    def finish(self) -> None:
        """R33-P0-050：收尾并**证明写完**。

        - close 所有 worker 队列 → join writer 线程 → drain 补写；
        - 任何 worker FAILED → 抛 fatal error（不静默）；
        - ``accepted != committed + failed`` → 抛错（有 item 未落盘）。
        """
        for q in self._worker_queues:
            q.close()
        for t in self._threads:
            t.join(timeout=10.0)
        # R38 P0-045（§17）：join 超时后 writer 线程仍 alive → **fatal**（abort
        # generation），**不** main 线程并发 drain 补写（避免并发消费/写竞态）。
        alive = [t for t in self._threads if t.is_alive()]
        if alive:
            self._set_fatal(
                RuntimeError(
                    f"writer thread(s) alive after join(timeout=10): "
                    f"{len(alive)} alive — abort generation, no manual drain write "
                    "(R38-P0-045: live writer after join is fatal)"
                )
            )
            self._drained = False
            raise RuntimeError(
                "StreamingResultSink.finish: writer thread alive after join — "
                "generation aborted (R38-P0-045)"
            )
        # join 成功后队列中残留 item 由 main 补写（此时无并发 writer）。
        remaining: list[ResultItem] = []
        for q in self._worker_queues:
            remaining.extend(q.drain())
        if remaining:
            try:
                self._writer(remaining)
                with self._lock:
                    if self._workers:
                        self._workers[0].committed += len(remaining)
                    else:
                        self._accepted = max(0, self._accepted - len(remaining))
            except Exception as exc:  # noqa: BLE001
                self._set_fatal(exc)
        self._drained = True
        # R33-P0-049/050：worker 级 FAILED（fatal_error）必须传播到 sink 级——
        # 任一 writer 死掉，整个 publish 失败（生产 run 不能谎报成功）。
        worker_fatal = next(
            (w.fatal_error for w in self._workers if w.fatal_error is not None),
            None,
        )
        if worker_fatal is not None:
            self._set_fatal(worker_fatal)
        if self._fatal_error is not None:
            raise RuntimeError(
                f"StreamingResultSink.finish: writer fatal — {type(self._fatal_error).__name__}: "
                f"{self._fatal_error}"
            ) from self._fatal_error
        committed = sum(w.committed for w in self._workers)
        failed = sum(w.failed for w in self._workers)
        with self._lock:
            accepted = self._accepted
        if accepted != committed + failed:
            raise RuntimeError(
                f"StreamingResultSink.finish: NOT durable — accepted={accepted} "
                f"committed={committed} failed={failed} (queue empty={not any(q.queued_count for q in self._worker_queues)})"
            )

    def _set_fatal(self, exc: BaseException) -> None:
        """P0-023：atomic set fatal + **立即 close 全部 worker 队列** + 通知。

        close 后 ``submit()`` 的 ``put`` 直接返回 False（queue closed），producer
        立即停止往死掉的 writer 塞结果；同时唤醒阻塞在 ``put``/``get`` 的线程，
        不等到 finish() 才暴露 writer 已死。
        """
        with self._lock:
            if self._fatal_error is not None:
                return
            self._fatal_error = exc
        for q in self._worker_queues:
            try:
                q.close()
            except Exception:
                pass

    def summary(self) -> dict[str, Any]:
        with self._lock:
            committed = sum(w.committed for w in self._workers)
            failed = sum(w.failed for w in self._workers)
            retried = sum(w.retried for w in self._workers)
            failed_workers = [w.worker_id for w in self._workers if w.state == WS_FAILED]
            return {
                "queue": self.queue.summary(),
                "accepted": self._accepted,
                "committed": committed,
                "failed": failed,
                "retried": retried,
                "writer_threads": self._writer_threads,
                "batch_size": self._batch_size,
                "target_batch_bytes": self._target_batch_bytes,
                "max_batch_age_s": self._max_batch_age_s,
                "partition_owner_count": len(self._partition_owners),
                "writer_states": [w.state for w in self._workers],
                "failed_workers": failed_workers,
                "fatal_error": (
                    f"{type(self._fatal_error).__name__}: {self._fatal_error}"
                    if self._fatal_error is not None
                    else None
                ),
                "durable_finish": self._drained
                and self._fatal_error is None
                and self._accepted == committed + failed,
            }
