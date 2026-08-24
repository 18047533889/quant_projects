# -*- coding: utf-8 -*-
"""R40 #61/#64 测试：SQL production-safe 参数域认证合并 + capability 有界 LRU。"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# #61：is_sql_production_safe / effective_sql_production_safe 合并参数域认证
# ---------------------------------------------------------------------------


def test_sql_production_safe_requires_parameter_domain_cert():
    from factor_engine.backend.sql_tiers import (
        effective_sql_production_safe,
        is_sql_production_safe,
    )
    from factor_engine.runtime.parameter_domain_store import (
        CertificationKey,
        get_parameter_domain_store,
        reset_parameter_domain_store,
    )

    reset_parameter_domain_store()
    try:
        store = get_parameter_domain_store()
        # 白名单里有 column，但无 duckdb_sql 参数域认证 → production-safe=False
        assert is_sql_production_safe("column") is False
        # 给 column 加 duckdb_sql 认证 → True（白名单 ∩ 参数域认证）
        store.certify_point(
            CertificationKey(
                canonical="column", backend="duckdb_sql",
                source_context="memory", grain="daily",
            ),
            passed=True,
        )
        store._loaded = True
        assert is_sql_production_safe("column") is True
        assert effective_sql_production_safe("column") is True
        # literal 在白名单但无认证 → False
        assert is_sql_production_safe("literal") is False
    finally:
        reset_parameter_domain_store()


def test_sql_production_safe_fail_closed_without_store():
    """store 未装载（无认证证据）→ 一律 False（fail-closed，不静默放行）。"""
    from factor_engine.backend.sql_tiers import is_sql_production_safe
    from factor_engine.runtime.parameter_domain_store import reset_parameter_domain_store

    reset_parameter_domain_store()
    try:
        assert is_sql_production_safe("column") is False
    finally:
        reset_parameter_domain_store()


# ---------------------------------------------------------------------------
# #64：SQL emitter capability 缓存 entry/byte 有界 LRU
# ---------------------------------------------------------------------------


def test_sql_emitter_capability_cache_bounded_lru():
    from factor_engine.backend.sql_tiers import _BoundedLRU, duckdb_downgrade_cache_info

    cache = _BoundedLRU(max_entries=2, max_bytes=2000)
    cache.put("k1", frozenset({"a"}))
    cache.put("k2", frozenset({"b"}))
    cache.put("k3", frozenset({"c"}))
    info = cache.info()
    assert info["entries"] <= 2          # 超限按 LRU 逐出
    assert cache.get("k1") is None       # 最旧被淘汰
    assert cache.get("k3") is not None
    assert info["max_entries"] == 2
    assert info["max_bytes"] == 2000

    # 全局 DuckDB downgrade 缓存也是 _BoundedLRU（entry/byte 双界）。
    gi = duckdb_downgrade_cache_info()
    assert "max_entries" in gi and "max_bytes" in gi
    assert gi["entries"] <= gi["max_entries"]
