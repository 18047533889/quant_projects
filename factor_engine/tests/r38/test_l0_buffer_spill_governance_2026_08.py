# -*- coding: utf-8 -*-
"""P0-012/013/014/033: L0 单一 owner + spill reload admission + 生命周期 + 成本决策。

验证：
    - Governor 高压逐出走 GovernedBufferStore（L0 唯一 owner），pinned 不逐出；
    - reload 放回 backing 前必须进预算（放不下逐出腾位；仍放不下保持 disk-backed）；
    - release 删除 spill 文件（不积累 orphan parquet）；SpillStore 启动 scavenge 孤儿；
    - LRU 逐出消费 recompute_cost（高成本 spill / 低成本 drop，should_spill 真实生效）；
    - ExpressionCache 在 governed_store 模式下是纯 adapter（不双套 accounting）。
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest

from runtime.buffer_store import GovernedBufferStore
from runtime.spill_store import SpillStore


def _frame(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame(np.arange(n * 3, dtype=float).reshape(n, 3),
                        index=pd.bdate_range("2026-01-01", periods=n),
                        columns=["A", "B", "C"])


def test_l0_evict_hook_routes_to_buffer_store_and_skips_pinned():
    backing: dict = {}
    store = GovernedBufferStore(backing, budget_bytes=1000,
                                spill_store=SpillStore(root_dir=os.path.join("/tmp", "fe_spill_t1")))
    store.put("a", _frame(10))  # ~2400 bytes > 1000
    store.put("b", _frame(10))
    store.pin("b")
    freed = store.evict_if_over_budget(1200)  # governor 高压 → L0 唯一 owner
    assert freed > 0
    # pinned 不逐出；"b" 仍在，LRU "a" 被逐（或 spill/drop）。
    assert "b" in backing
    assert "a" not in backing or store._spill_refs.get("a") is not None


def test_reload_respects_budget():
    store = GovernedBufferStore({}, budget_bytes=500,
                                spill_store=SpillStore(root_dir=os.path.join("/tmp", "fe_spill_t2")))
    ref = store._spill_store.spill(_frame(20), key="big", execution_id="e1")
    store._spill_refs["big"] = ref
    # 预算 500 放不下 20x3 float64 frame（~4800B）：逐出腾位仍放不下 → 不缓存。
    value = store.get_ref("big")
    assert value is not None  # 本次仍返回
    assert "big" not in store._backing  # 但保持 disk-backed（没塞回内存）
    assert store._spill_refs["big"] is not None  # reload 依据仍在


def test_release_deletes_spill_file_and_orphan_scavenger():
    root = os.path.join("/tmp", "fe_spill_t3")
    ss = SpillStore(root_dir=root)
    ref = ss.spill(_frame(5), key="k", execution_id="e2")
    assert os.path.exists(ref.path)
    store = GovernedBufferStore({}, budget_bytes=10**9, spill_store=ss, execution_id="e2")
    store._spill_refs["k"] = ref
    store.release("k")
    assert not os.path.exists(ref.path)  # release 删除 spill 文件（P0-013）
    # 孤儿文件（未追踪）被启动 scavenge 清理。
    orphan = os.path.join(root, "orphan_old.parquet")
    with open(orphan, "wb") as fh:
        fh.write(b"x" * 8)
    old = time.time() - 7200
    os.utime(orphan, (old, old))
    assert ss.cleanup_orphans(ttl_seconds=3600) >= 1
    assert not os.path.exists(orphan)


def test_evict_lru_cost_decision():
    # 高 recompute 成本 → spill；低成本 → drop。
    store = GovernedBufferStore({}, budget_bytes=10**9,
                                spill_store=SpillStore(root_dir=os.path.join("/tmp", "fe_spill_t4")),
                                execution_id="e3")
    store.put("cheap", _frame(5), recompute_cost_ms=0.0)
    store.put("costly", _frame(5), recompute_cost_ms=10000.0)
    store._evict_lru(10**9)  # 全部逐出（needed 巨大）
    # cheap 被 drop（无 spill ref），costly 被 spill（有 spill ref）。
    assert "cheap" not in store._backing and store._spill_refs.get("cheap") is None
    assert "costly" not in store._backing and store._spill_refs.get("costly") is not None
    assert store._spill_refs["costly"].execution_id == "e3"


def test_expression_cache_is_pure_adapter():
    from cache.expression_cache import ExpressionCache

    backing: dict = {}
    store = GovernedBufferStore(backing, budget_bytes=10**9)
    ec = ExpressionCache(backing, governed_store=store)
    ec.set("sid", _frame(4))
    assert "sid" in backing
    assert ec.get("sid") is not None
    ec.release("sid")
    assert "sid" not in backing
    # 不再维护自己的 accounting（纯 adapter）。
    assert ec._governed_store is store
    assert ec._bytes == 0
