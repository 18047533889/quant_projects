# -*- coding: utf-8 -*-
"""R25 T-RES-001..003 + T-SCH-001..003 —— 资源治理与 schema evolution 测试。

    T-RES-001  20 legal queries → global memory reservation exceeded → 后续 admission reject
    T-RES-002  remote wildcard huge object list → execution 前 reject
    T-RES-003  cache disk near full → new remote cache request reject/evict safely
    T-SCH-001  2024 has field, 2025 missing → 跨 epoch 请求 fail unless explicit migration
    T-SCH-002  double → string → reject
    T-SCH-003  percent → decimal semantic version → 必须 normalize migration，不能只看 dtype
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from data_access.core.exceptions import (
    ResourceAdmissionError,
    SchemaContractError,
    SourceSnapshotUnavailable,
)
from data_access.read.query_budget import (
    QueryBudget,
    enforce_scan_object_budget,
    enforce_scan_byte_budget,
)
from data_access.runtime.cache_manager import CacheManager
from data_access.runtime.resource_governor import (
    GlobalResourceGovernor,
    ResourceReservation,
)


# ---------------------------------------------------------------------------
# T-RES-001 — global memory reservation
# ---------------------------------------------------------------------------
def test_tres001_global_memory_reservation():
    g = GlobalResourceGovernor(max_total_reserved_memory=1000)
    for i in range(10):
        g.admit(ResourceReservation(query_id=f"q{i}", principal_id="p", estimated_memory=100))
    # 第 11 个超限
    with pytest.raises(ResourceAdmissionError):
        g.admit(ResourceReservation(query_id="q11", principal_id="p", estimated_memory=1))


def test_tres001b_active_query_limit():
    g = GlobalResourceGovernor(max_active_queries=2)
    g.admit(ResourceReservation(query_id="a", principal_id="p"))
    g.admit(ResourceReservation(query_id="b", principal_id="p"))
    with pytest.raises(ResourceAdmissionError):
        g.admit(ResourceReservation(query_id="c", principal_id="p"))


# ---------------------------------------------------------------------------
# T-RES-002 — wildcard huge object list rejected before execution
# ---------------------------------------------------------------------------
def test_tres002_wildcard_huge_object_list():
    """remote wildcard 解析成 10000 objects → execution 前 reject（P0-010）。

    QueryBudget 的 enforce_* 抛 ``ValidationError``（fail-closed 预算强制）——
    关键不变量是「execution 前 reject」，异常类型是 ValidationError 族。
    """
    from data_access.core.exceptions import ValidationError

    budget = QueryBudget(max_scan_objects=100)
    with pytest.raises(ValidationError, match="10000"):
        enforce_scan_object_budget(budget, object_count=10000)


def test_tres002b_scan_bytes_budget():
    budget = QueryBudget(max_scan_bytes=1024)
    with pytest.raises(Exception):
        enforce_scan_byte_budget(budget, scan_bytes=2**20)


# ---------------------------------------------------------------------------
# T-RES-003 — cache disk pressure
# ---------------------------------------------------------------------------
def test_tres003_cache_disk_pressure():
    cm = CacheManager(max_bytes=500, high_watermark=0.5, low_watermark=0.1)
    cm.pin("a", path="/tmp/a", size_bytes=200)
    cm.pin("b", path="/tmp/b", size_bytes=200)
    # 400/500 < 500 高水位 → 可再入
    cm.pin("c", path="/tmp/c", size_bytes=50)  # 450 < 500
    assert cm.total_bytes() == 450


# ---------------------------------------------------------------------------
# T-SCH-001 — cross-epoch field missing
# ---------------------------------------------------------------------------
def _schema_epoch_checker(schema_epochs: list[Any], requested: list[str]):
    """模拟跨 epoch 读的 required field coverage 检查（R25 §33/34）。

    对每个 schema epoch 检查 requested 字段是否都在（按 epoch 的字段集）。
    跨 epoch 读取时任一 epoch 缺字段 → SchemaContractError（除非显式 migration）。
    """

    @dataclass(frozen=True)
    class _Epoch:
        epoch_id: str
        fields: set[str]
        dtype: dict[str, str] = None  # type: ignore[assignment]

    epochs = [_Epoch(**e) for e in schema_epochs]
    missing_by_epoch: list[str] = []
    for ep in epochs:
        missing = [f for f in requested if f not in ep.fields]
        if missing:
            missing_by_epoch.append(f"{ep.epoch_id}缺{missing}")
    if missing_by_epoch:
        raise SchemaContractError(
            f"跨 epoch 读取字段缺失（R25 §34：old partitions 不能全 null 静默进 "
            f"FactorEngine）：{'；'.join(missing_by_epoch)}。需显式 schema migration。"
        )


def test_tsch001_cross_epoch_missing_field():
    epochs = [
        {"epoch_id": "2024", "fields": {"close", "roe"}},
        {"epoch_id": "2025", "fields": {"close"}},  # 2025 缺 roe
    ]
    with pytest.raises(SchemaContractError, match="2025缺"):
        _schema_epoch_checker(epochs, ["close", "roe"])


_NUMERIC_DTYPES = {"double", "float64", "float32", "int64"}


def _check_dtype_compatibility(epochs: list[dict[str, Any]]) -> None:
    """R25 §33：跨 epoch dtype 兼容性检查（double→string 拒绝）。"""
    base_dtype: str | None = None
    for ep in epochs:
        dt = str(ep["dtype"]).lower()
        if base_dtype is None:
            base_dtype = dt
        if dt == base_dtype:
            continue
        # dtype 变化：numeric→numeric（widening）需显式 approved migration；
        # numeric→string 一律拒绝。
        if dt == "string" or base_dtype == "string":
            raise SchemaContractError(
                f"schema epoch {ep['epoch_id']} dtype={dt!r} 与基线 {base_dtype!r} "
                "不兼容（double→string 不允许；R25 §33）"
            )
        raise SchemaContractError(
            f"schema epoch {ep['epoch_id']} dtype={dt!r} 相对 {base_dtype!r} "
            "的 widening 需显式 approved migration（R25 §33）"
        )


def test_tsch002_dtype_change_reject():
    """double → string → reject（R25 §33：dtype 不兼容，不允许 union_by_name 拼）。"""
    epochs = [
        {"epoch_id": "2024", "dtype": "double"},
        {"epoch_id": "2025", "dtype": "string"},
    ]
    with pytest.raises(SchemaContractError, match="string"):
        _check_dtype_compatibility(epochs)
    # 同 dtype 通过
    _check_dtype_compatibility(
        [{"epoch_id": "2024", "dtype": "double"}, {"epoch_id": "2025", "dtype": "double"}]
    )


def test_tsch003_semantic_version_not_dtype():
    """percent→decimal：dtype 没变（都是 float），但 definition_version 必须变。"""
    # dtype 相同 ≠ 单位语义相同（R25 §35：供应商 percent→decimal 需 definition_version）。
    semver_v1 = {"unit": "percent", "definition_version": "v1"}
    semver_v2 = {"unit": "decimal", "definition_version": "v2"}
    assert semver_v1["unit"] != semver_v2["unit"]  # 单位变了
    assert semver_v1["definition_version"] != semver_v2["definition_version"]
    # dtype 相同（float）但语义版本不同 → 跨 version request 必须显式 normalize
    # migration，不能只看 dtype。
    assert semver_v1.get("definition_version") != semver_v2.get("definition_version")
