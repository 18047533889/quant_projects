# -*- coding: utf-8 -*-
"""R38 P0-025/026/050/051（§10/§20）：DA governor locking + standalone cap +
DataReadSession 并发 isolation。

行为探针：
    - DataReadSession 并发不互相覆盖 resolution cache（§R38_DATAREADSESSION_CONCURRENT_ISOLATION）；
    - DA standalone 自动 safe cap（§R38_DA_STANDALONE_MEMORY_BOUNDED）；
    - 动态 setter 与 admit() 同锁、cap 收缩不 kill incumbents（
      §R38_DA_DYNAMIC_LIMIT_THREAD_SAFE）。
"""
from __future__ import annotations

import os
import threading

import pytest

from data_access.runtime.resource_governor import (
    GlobalResourceGovernor,
    ResourceReservation,
    reset_global_governor,
)


# ---------------------------------------------------------------------------
# DA standalone safe cap（P0-025）
# ---------------------------------------------------------------------------


def test_da_standalone_safe_cap(monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_AUTO_BOUND_MEMORY", "1")
    gov = GlobalResourceGovernor()
    assert gov._max_memory is not None, "standalone production 不能 None（无限）"
    assert gov._max_memory > 0
    assert gov._max_scan is not None
    # P0-025 fail-closed：**默认开启**——未设置 env 也必须有 safe cap（不无限）。
    monkeypatch.delenv("DATA_ACCESS_AUTO_BOUND_MEMORY", raising=False)
    reset_global_governor()
    gov2 = GlobalResourceGovernor()
    assert gov2._max_memory is not None, "默认（未设置）也必须有 safe cap（fail-closed）"
    assert gov2._max_memory > 0
    # 显式关闭才允许无上限（测试 / 无上限部署显式选择）。
    monkeypatch.setenv("DATA_ACCESS_AUTO_BOUND_MEMORY", "0")
    reset_global_governor()
    gov3 = GlobalResourceGovernor()
    assert gov3._max_memory is None


# ---------------------------------------------------------------------------
# dynamic cap 与 admit 同锁（P0-026）
# ---------------------------------------------------------------------------


def test_da_dynamic_cap_locking_over_target():
    gov = GlobalResourceGovernor(max_total_reserved_memory=1000, max_total_scan_bytes_inflight=1000)
    gov.admit(ResourceReservation(query_id="a", principal_id="p", estimated_memory=800, estimated_scan_bytes=100))
    # cap 收缩到低于当前 reservation → 不 kill incumbent；标记 over_current_target。
    gov.set_max_total_reserved_memory(500)
    assert gov._over_current_target is True
    # incumbent 仍在 active（不 kill）。
    assert gov.active_count() == 1
    # 新 admission 被阻止。
    with pytest.raises(Exception):
        gov.admit(ResourceReservation(query_id="b", principal_id="p", estimated_memory=100, estimated_scan_bytes=100))
    # incumbent release 后恢复正常。
    gov.release("a")
    assert gov._over_current_target is False
    gov.admit(ResourceReservation(query_id="b", principal_id="p", estimated_memory=100, estimated_scan_bytes=100))


def test_da_setter_thread_safe():
    gov = GlobalResourceGovernor(max_total_reserved_memory=10**9)
    errors: list[BaseException] = []

    def _writer():
        for i in range(50):
            try:
                gov.set_max_total_reserved_memory(10**9 - i * 1000)
                gov.set_max_total_scan_bytes_inflight(10**9 - i * 1000)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    def _admitter():
        for i in range(50):
            try:
                gov.admit(ResourceReservation(query_id=f"q{i}", principal_id="p", estimated_memory=1, estimated_scan_bytes=1))
                gov.release(f"q{i}")
            except Exception:
                pass

    threads = [threading.Thread(target=_writer), threading.Thread(target=_admitter)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


# ---------------------------------------------------------------------------
# DataReadSession 并发 isolation（P0-050/051）
# ---------------------------------------------------------------------------


def test_datareadsession_concurrent_cache_isolation():
    from data_access.read.read_session import DataReadSession
    from data_access.runtime.read_session_context import (
        get_resolution_cache,
    )

    class _Store:
        def __init__(self):
            self._resolution_cache = None
            self._reads = []

        def lock_calendars(self):
            pass

        def read(self, *args, **kwargs):
            # 读取当前 request-scoped cache（写入标记）。
            cache = get_resolution_cache()
            if cache is not None:
                cache[f"hit_{args[0] if args else '?'}"] = True
            self._reads.append(1)
            return {"ok": True}

    store = _Store()
    results: list[dict] = []
    barrier = threading.Barrier(2)

    def _worker(name: str):
        # 用真实 DataReadSession：A/B 各自绑定自己的 resolution cache。
        with DataReadSession(store, request_id=name) as s:
            barrier.wait()
            # 自己的 cache 写入标记（模拟 job 内 resolve）。
            cache = get_resolution_cache()
            cache[f"k_{name}"] = name
            s.read(name)
            barrier.wait()
            results.append({"name": name, "cache": dict(cache)})
        # A 退出后（B 仍可能运行）：B 的 cache 仍在自己的 ContextVar 里。
        barrier.wait()
        results.append({"name": name + "_after_exit", "store_resolution_cache": store._resolution_cache})

    a = threading.Thread(target=_worker, args=("A",))
    b = threading.Thread(target=_worker, args=("B",))
    a.start(); b.start(); a.join(); b.join()
    # 各自 request-scoped cache 隔离（顺序不定，按 name 找）。
    caches = {r["name"]: r["cache"] for r in results if "cache" in r}
    assert "k_A" in caches["A"] and "k_B" in caches["B"], f"caches={caches}"
    # Store 全局属性不被 session 覆盖（§35：Store 没有被 session 覆盖全局字段）。
    for r in results:
        if "store_resolution_cache" in r:
            assert r["store_resolution_cache"] is None
