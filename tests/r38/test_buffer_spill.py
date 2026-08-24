# -*- coding: utf-8 -*-
"""R38 P0-029/030/032/033 + P1-035/036/037（§12/§13/§14）：BufferStore / spill。

行为探针：
    - put 拒绝不静默（BufferPutResult status，§R38_BUFFER_PUT_REFUSAL_NOT_SILENT）；
    - production 无 raw shared_result_cache fallback（§R38_ZERO_PRODUCTION_RAW_CSE_FALLBACK）；
    - 真实 spill 往返（write → reload == 原值，checksum 校验，§R38_REAL_SPILL_RELOAD_PASS）；
    - spill checksum 失败检测；
    - reconciliation 大 keyset 抽样（不再天然巨大漂移）；
    - get() 持锁 + LRU 淘汰（P1-035/036）。
"""
from __future__ import annotations

import pandas as pd
import pytest

from factor_engine.runtime.buffer_store import (
    STATUS_MEMORY,
    STATUS_RECOMPUTE,
    STATUS_REFUSED,
    STATUS_SPILLED,
    GovernedBufferStore,
)
from factor_engine.runtime.spill_store import SpillStore


def test_put_refusal_not_silent():
    store = GovernedBufferStore({}, budget_bytes=100)
    assert store.put("a", "x" * 40, bytes_=40).status == STATUS_MEMORY
    res = store.put("b", "y" * 1000, bytes_=1000)
    assert res.status in (STATUS_RECOMPUTE, STATUS_REFUSED), res.status
    assert store.summary()["refused"] >= 1


def test_put_spills_over_budget_when_spool():
    store = GovernedBufferStore(
        {},
        budget_bytes=100,
        spill_store=SpillStore(),
    )
    res = store.put("big", pd.Series([1.0, 2.0, 3.0]), bytes_=10**6, spool=True)
    assert res.status == STATUS_SPILLED, res.status
    # spill 后 backing 不常驻，但 get_ref 可 reload。
    assert store.get_ref("big") is not None
    val = store.get_ref("big")
    assert isinstance(val, pd.Series)


def test_spill_reload_roundtrip():
    store = SpillStore()
    s = pd.Series([1.0, 2.0, 3.0], index=pd.MultiIndex.from_product(
        [[pd.Timestamp("2024-01-01")], ["A", "B", "C"]]
    ))
    ref = store.spill(s, key="s1", source_identity="cse")
    assert store.verify(ref)
    reloaded = store.reload(ref)
    # MultiIndex 在 parquet 中保留；名字可能因中间列名变化 → check_names=False。
    pd.testing.assert_series_equal(reloaded, s, check_names=False)


def test_spill_checksum_failure_detected():
    import os

    store = SpillStore()
    s = pd.Series([1.0, 2.0, 3.0])
    ref = store.spill(s, key="s2")
    # 篡改文件内容 → checksum 校验失败。
    with open(ref.path, "ab") as fh:
        fh.write(b"corrupt")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        store.reload(ref)


def test_no_production_raw_cse_fallback():
    from types import SimpleNamespace

    import factor_engine.runtime.batch_service as bs

    # production ctx：只有 shared_result_cache（raw dict），无 buffer store →
    # _materialize_shared_subplan 必须 fail-closed（不写 raw）。
    ctx = SimpleNamespace(
        shared_result_cache={},
        shared_buffers=None,
        expression_cache=None,
        runtime_stats={},
        run_mode="production",
    )
    backend = SimpleNamespace(
        supports_lazy_shared=False,
        execute=lambda sub, c: pd.Series([1.0]),
    )
    from factor_engine.runtime.production_policy import is_production_mode

    with pytest.raises(RuntimeError, match="fail-closed|governed"):
        bs._materialize_shared_subplan(backend, SimpleNamespace(op="ts_mean"), ctx, "sid1")
    assert "sid1" not in ctx.shared_result_cache, "production 不得写 raw shared_result_cache"


def test_reconciliation_large_keyset_no_false_drift():
    from factor_engine.runtime.buffer_store import _estimate_bytes

    store = GovernedBufferStore({}, budget_bytes=10**9)
    total = 0
    for i in range(2000):
        s = pd.Series([1.0, 2.0, 3.0])
        b = _estimate_bytes(s)
        store.put(f"k{i:04d}", s, bytes_=b)
        total += b
    rec = store.reconciliation(sample_limit=512)
    assert rec["total_keys"] == 2000
    assert rec["extrapolated"] is True
    # accounted == 全量实际（每项真实字节），抽样+外推后 drift 应接近 0（不再
    # 因「只比前 512 项」天然产生巨大漂移）。
    assert rec["accounted_bytes"] == total
    assert rec["drift_bytes"] <= total * 0.1


def test_lru_eviction_most_stale_first():
    store = GovernedBufferStore({}, budget_bytes=100)
    store.put("a", "a" * 40, bytes_=40)
    store.put("b", "b" * 40, bytes_=40)
    # 访问 a（bump last_access）→ a 不是最久未用。
    store.get("a")
    store.put("c", "c" * 40, bytes_=40)  # 需淘汰一个 → 淘汰 b（最久未用）
    assert store.get("b") is None
    assert store.get("a") is not None
