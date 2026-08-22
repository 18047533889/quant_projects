"""Process-wide absolute deadline scheduler for cancellable DataAccess work."""

from __future__ import annotations

import heapq
import itertools
import threading
import time
from collections.abc import Callable


class DeadlineManager:
    """Schedule all query interrupts on one daemon timer thread."""

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, str]] = []
        self._active: dict[str, tuple[int, Callable[[], None]]] = {}
        self._fired: set[str] = set()
        self._sequence = itertools.count()
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._thread_count = 0

    @property
    def watchdog_thread_created_count(self) -> int:
        with self._condition:
            return self._thread_count

    def register_at(
        self, token: str, absolute_deadline: float, cancel: Callable[[], None]
    ) -> None:
        with self._condition:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run,
                    daemon=True,
                    name="dataaccess-deadline-manager",
                )
                self._thread.start()
                self._thread_count += 1
            sequence = next(self._sequence)
            self._active[token] = (sequence, cancel)
            heapq.heappush(self._heap, (absolute_deadline, sequence, token))
            self._condition.notify_all()

    def cancel(self, token: str) -> bool:
        with self._condition:
            return self._active.pop(token, None) is not None

    def fired(self, token: str) -> bool:
        with self._condition:
            return token in self._fired

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._heap:
                    self._condition.wait()
                deadline, sequence, token = self._heap[0]
                wait = deadline - time.monotonic()
                if wait > 0:
                    self._condition.wait(wait)
                    continue
                heapq.heappop(self._heap)
                active = self._active.get(token)
                if active is None or active[0] != sequence:
                    continue
                del self._active[token]
                self._fired.add(token)
                cancel = active[1]
            try:
                cancel()
            except Exception:
                pass


_MANAGER = DeadlineManager()


def get_deadline_manager() -> DeadlineManager:
    return _MANAGER
