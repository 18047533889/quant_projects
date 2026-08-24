# -*- coding: utf-8 -*-
"""R36 §168/169/170/196：RunPeakSampler —— 只统计本次 run 窗口的真实峰值。

R36 P0-029（§169）：当前 ``rss_peak_run`` 把进程**历史 lifetime peak**
（``ru_maxrss``）混入 run peak，不能用于精确训练 task memory model。修复：
run start 起专门 sampler，run finish 停——只统计该 run 时间窗口内
``process_family_pss`` / ``process_family_rss`` 的峰值（§168 true run peak；
§13/15 PSS 比 RSS 更准，跨进程共享页不重复计）。

多 task 并发时无法精确逐 byte 归属单 task（§171），第一版只做 run 级峰值；
per-shape 归因由 isolated calibration runs（§172）+ concurrent marginal model
后续补充。
"""
from __future__ import annotations

import threading
import time
from typing import Any


def _family_memory() -> tuple[int, int]:
    """当前 process family (pss, rss)；优先 PSS（smaps_rollup）。"""
    try:
        from factor_engine.runtime.resource_governor import process_family_rss_bytes

        rss = process_family_rss_bytes(prefer_pss=False) or 0
        pss = process_family_rss_bytes(prefer_pss=True) or rss
        return int(pss), int(rss)
    except Exception:
        try:
            import psutil  # type: ignore[import-untyped]

            rss = int(psutil.Process().memory_info().rss)
            return rss, rss
        except Exception:
            return 0, 0


class RunPeakSampler:
    """run 级峰值采样器：start → 后台定时采样 → stop 返回 run 窗口峰值。

    只统计 ``start()`` 到 ``stop()`` 之间的采样（基线取 start 时 family 内存，
    峰值 = max(采样 family memory - baseline, 0) 的增量峰值，避免把进程既有
    常驻内存算进本 run）。
    """

    def __init__(self, *, interval_s: float = 0.5) -> None:
        self._interval = max(0.05, float(interval_s))
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._baseline_pss = 0
        self._baseline_rss = 0
        self._peak_pss = 0
        self._peak_rss = 0
        self._samples = 0
        self._started = False
        self._started_at_ms = 0.0

    @property
    def started(self) -> bool:
        with self._lock:
            return self._started

    def start(self) -> None:
        """开始采样：记录基线，启动后台线程。"""
        with self._lock:
            if self._started:
                return
            self._baseline_pss, self._baseline_rss = _family_memory()
            self._peak_pss = 0
            self._peak_rss = 0
            self._samples = 0
            self._started = True
            self._started_at_ms = time.monotonic() * 1000.0
            self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="r36-run-peak", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval):
            pss, rss = _family_memory()
            with self._lock:
                self._samples += 1
                self._peak_pss = max(self._peak_pss, max(0, pss - self._baseline_pss))
                self._peak_rss = max(self._peak_rss, max(0, rss - self._baseline_rss))

    def stop(self) -> dict[str, Any]:
        """停止采样，返回本 run 窗口峰值（不混入 lifetime peak）。"""
        if not self._started:
            return {"started": False, "peak_family_pss": 0, "peak_family_rss": 0}
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(2.0, self._interval * 2 + 1.0))
        with self._lock:
            self._started = False
            end_pss, end_rss = _family_memory()
            return {
                "started": True,
                "started_at_ms": round(self._started_at_ms, 3),
                "peak_family_pss": self._peak_pss,
                "peak_family_rss": self._peak_rss,
                "baseline_family_pss": self._baseline_pss,
                "baseline_family_rss": self._baseline_rss,
                "end_family_pss": end_pss,
                "end_family_rss": end_rss,
                "samples": self._samples,
            }


#: 便捷入口：run 级采样上下文。
class run_peak:
    """``with run_peak() as sampler: ...`` → 结束后 sampler.stop() 已调用。"""

    def __init__(self, *, interval_s: float = 0.5) -> None:
        self.sampler = RunPeakSampler(interval_s=interval_s)

    def __enter__(self) -> RunPeakSampler:
        self.sampler.start()
        return self.sampler

    def __exit__(self, *exc) -> None:
        self.sampler.stop()
