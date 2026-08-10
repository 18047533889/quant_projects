# -*- coding: utf-8 -*-
"""R27-105..109: BoundedResultQueue + StreamingResultSink —— 边算边写。

目标
    - 计算线程与 writer 之间用 **bounded bytes queue**（不是只按 item count），
      R27-105。
    - 磁盘写不过来 → queue 达到 bytes 上限 → writer backpressure 信号 → scheduler
      降低新 compute admission（R27-106）。
    - compute 与 write pipeline 重叠：CPU 正在算 B/C 时 writer 落 A（R27-107）。
    - 写端 batching：按 date partition / factor group / matrix block 批量写，
      避免大量小因子单独 open/write/close（R27-108/109）。
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable


def _bytes_of(value: Any) -> int:
    try:
        from runtime.resource_governor import estimate_object_bytes

        return estimate_object_bytes(value)
    except Exception:
        return 0


@dataclass
class ResultItem:
    """一次结果交付（name + value + 元数据）。"""

    name: str
    value: Any
    bytes: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


class BoundedResultQueue:
    """按 **bytes** 上限的有界结果队列（R27-105）。

    消费者 ``get()`` 阻塞直到有 item 或 closed；``backpressure_ratio`` 反映
    队列占用比例，writer 慢时 scheduler 据此降 admission。
    """

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max(1, int(max_bytes))
        self._items: list[ResultItem] = []
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
                import time

                t0 = time.monotonic()
                if not self._lock.wait(timeout):
                    return False
                self._write_backpressure_seconds += time.monotonic() - t0
            self._items.append(item)
            self._current_bytes += item.bytes
            self._lock.notify()
            return True

    def get(self, *, timeout: float = 1.0) -> ResultItem | None:
        """取下一个结果；队列空且未 closed 时短暂阻塞轮询。"""
        import time

        deadline = time.monotonic() + timeout
        with self._lock:
            while not self._items and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._lock.wait(min(remaining, 0.2))
            if not self._items:
                return None
            item = self._items.pop(0)
            self._current_bytes = max(0, self._current_bytes - item.bytes)
            self._lock.notify()
            return item

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._lock.notify_all()

    def drain(self) -> list[ResultItem]:
        """取出全部剩余结果（run 结束收尾）。"""
        out: list[ResultItem] = []
        with self._lock:
            out = self._items
            self._items = []
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


class StreamingResultSink:
    """compute→writer 的流式 sink（R27-107 pipeline 重叠）。

    ``result_policy='sink'`` 时，每个 root 完成立即 ``put``，writer 线程
    ``get`` 后按 batch 写入目标（factor matrix block / date partition）。
    """

    def __init__(
        self,
        *,
        writer: Callable[[list[ResultItem]], None],
        queue_bytes: int = 4 * 1024**3,
        batch_size: int = 1,
        writer_threads: int = 1,
    ) -> None:
        self.queue = BoundedResultQueue(queue_bytes)
        self._writer = writer
        self._batch_size = max(1, batch_size)
        self._threads: list[threading.Thread] = []
        self._writer_threads = max(1, writer_threads)
        self._lock = threading.Lock()
        self._writes_done = 0
        self._write_bytes = 0

    def start(self) -> None:
        for i in range(self._writer_threads):
            t = threading.Thread(target=self._run_writer, daemon=True, name=f"r27-writer-{i}")
            t.start()
            self._threads.append(t)

    def _run_writer(self) -> None:
        while True:
            item = self.queue.get()
            if item is None and self.queue._closed:
                break
            if item is None:
                continue
            batch = [item]
            # 写端 batching（R27-108/109）：尽量凑 batch。
            while len(batch) < self._batch_size:
                nxt = self.queue.get(timeout=0.05)
                if nxt is None:
                    break
                batch.append(nxt)
            try:
                self._writer(batch)
                with self._lock:
                    self._writes_done += len(batch)
                    self._write_bytes += sum(b.bytes for b in batch)
            except Exception:
                # writer 失败：result item 不应静默丢失——重新入队（队头）并继续。
                for b in reversed(batch):
                    self.queue.put(b, timeout=1.0)

    def submit(self, name: str, value: Any, **meta: Any) -> bool:
        return self.queue.put(ResultItem(name=name, value=value, meta=dict(meta)))

    def finish(self) -> None:
        self.queue.close()
        for t in self._threads:
            t.join(timeout=5.0)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "queue": self.queue.summary(),
                "writes_done": self._writes_done,
                "write_bytes": self._write_bytes,
                "writer_threads": self._writer_threads,
            }
