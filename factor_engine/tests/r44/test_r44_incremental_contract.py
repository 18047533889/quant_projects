# -*- coding: utf-8 -*-
"""R44-IncrementalContract: node-level incremental FactorEngine 契约测试。

覆盖 :mod:`runtime.incremental_contract` 的：
  1. ``classify_incremental_mode`` 对代表性算子的自动分类。
  2. ``resolve_incremental_contract`` 的解析优先级 / fail-closed 回退。
  3. 显式 ``register_incremental_contract`` 覆盖自动分类 + 重复声明报错。
  4. ``incremental_capability_matrix`` 的完整性与字段集。
  5. ``node_incremental_slot`` 对 IR 节点的契约解析。
"""
from __future__ import annotations

import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.ir.nodes import IRNode
from factor_engine.runtime.incremental_contract import (
    CrossSectionScope,
    IncrementalContract,
    IncrementalContractResolutionError,
    IncrementalMode,
    StatePartitionAxis,
    classify_incremental_mode,
    incremental_capability_matrix,
    node_incremental_slot,
    register_incremental_contract,
    resolve_incremental_contract,
)
from factor_engine.stateful_contract import StatefulCheckpointRegistry

load_all()


def _registered_once(*, unique: str) -> None:
    """每个 canonical 只注册一次（测试模块级共享注册表；幂等跳过）。"""
    from factor_engine.runtime.incremental_contract import _REGISTRY

    if unique in _REGISTRY:
        raise AssertionError(f"duplicate registration fixture for {unique!r}")


# ---------------------------------------------------------------------------
# 1. 自动分类器
# ---------------------------------------------------------------------------
def test_classify_ts_ema_checkpointed_state() -> None:
    """ts_ema 在 StatefulCheckpointRegistry 且 state_model=recursive → CHECKPOINTED_STATE。"""
    assert StatefulCheckpointRegistry.get("ts_ema") is not None
    assert classify_incremental_mode("ts_ema", {"span": 20}) is IncrementalMode.CHECKPOINTED_STATE


def test_classify_ts_mean_finite_window() -> None:
    """ts_mean(window=20)：有限滚动窗口 → FINITE_WINDOW。"""
    assert classify_incremental_mode("ts_mean", {"window": 20}) is IncrementalMode.FINITE_WINDOW


def test_classify_elementwise_not_full_replay() -> None:
    """逐行算子（log / ts_delta）绝不落入 FULL_REPLAY / CHECKPOINTED_STATE。"""
    for canonical in ("log", "ts_delta"):
        mode = classify_incremental_mode(canonical, {"d": 1} if canonical == "ts_delta" else None)
        assert mode is not IncrementalMode.FULL_REPLAY, canonical
        assert mode is not IncrementalMode.CHECKPOINTED_STATE, canonical


def test_classify_unknown_fails_closed_to_full_replay() -> None:
    """无法解析的 canonical → FULL_REPLAY（fail-closed）。"""
    assert classify_incremental_mode("__r44_unknown_op__") is IncrementalMode.FULL_REPLAY


# ---------------------------------------------------------------------------
# 2. resolve_incremental_contract
# ---------------------------------------------------------------------------
def test_resolve_ts_ema_fields() -> None:
    """ts_ema(span=20) → CHECKPOINTED_STATE / PER_INSTRUMENT / 无界 forward_impact。"""
    contract = resolve_incremental_contract("ts_ema", {"span": 20})
    assert contract.incremental_mode is IncrementalMode.CHECKPOINTED_STATE
    assert contract.state_partition_axis is StatePartitionAxis.PER_INSTRUMENT
    assert contract.forward_impact is None
    assert contract.state_model == "recursive"
    assert contract.checkpoint_schema == "ema_state.v2"
    assert contract.checkpoint_schema_version == "ema_state.v2"


def test_resolve_ts_mean_fields() -> None:
    """ts_mean(window=20) → FINITE_WINDOW，backward_history >= 1。"""
    contract = resolve_incremental_contract("ts_mean", {"window": 20})
    assert contract.incremental_mode is IncrementalMode.FINITE_WINDOW
    assert contract.backward_history >= 1
    assert contract.forward_impact is not None and contract.forward_impact >= 1


def test_resolve_unknown_research_sets_resolution_error() -> None:
    """research 模式：无法解析 → FULL_REPLAY + resolution_error 已设置。"""
    contract = resolve_incremental_contract("__r44_unknown_op__")
    assert contract.incremental_mode is IncrementalMode.FULL_REPLAY
    assert contract.incremental_certified is False
    assert contract.resolution_error


def test_resolve_unknown_production_raises() -> None:
    """production 模式：解析失败抛 IncrementalContractResolutionError。"""
    from factor_engine.runtime.incremental_contract import IncrementalContractResolutionError as E

    with pytest.raises(E):
        resolve_incremental_contract("__r44_unknown_op__", production=True)


# ---------------------------------------------------------------------------
# 3. 显式声明覆盖自动分类 + 重复声明报错
# ---------------------------------------------------------------------------
def test_registration_overrides_classifier_and_duplicate_raises() -> None:
    """显式声明覆盖自动分类；重复声明抛 IncrementalContractResolutionError。"""
    _registered_once(unique="__r44_declared_override__")
    register_incremental_contract(
        "__r44_declared_override__",
        incremental_mode=IncrementalMode.FINITE_WINDOW,
        backward_history=5,
        forward_impact=7,
        state_partition_axis=StatePartitionAxis.GLOBAL,
        cross_section_scope=CrossSectionScope.PER_DATE,
        revision_policy="affected_domain",
        incremental_certified=True,
    )
    # 覆盖自动分类器（自动分类对未知 canonical 应为 FULL_REPLAY）。
    assert classify_incremental_mode("__r44_declared_override__") is IncrementalMode.FULL_REPLAY
    contract = resolve_incremental_contract("__r44_declared_override__")
    assert contract.incremental_mode is IncrementalMode.FINITE_WINDOW
    assert contract.backward_history == 5
    assert contract.forward_impact == 7
    assert contract.incremental_certified is True
    assert contract.state_partition_axis is StatePartitionAxis.GLOBAL
    # 重复声明是 drift，必须报错。
    with pytest.raises(IncrementalContractResolutionError):
        register_incremental_contract(
            "__r44_declared_override__",
            incremental_mode=IncrementalMode.STATELESS,
        )


def test_incremental_contract_to_dict_serializes_enums_as_values() -> None:
    """to_dict 把枚举序列化为 .value 字符串。"""
    contract = resolve_incremental_contract("ts_mean", {"window": 20})
    data = contract.to_dict()
    assert data["incremental_mode"] == "FINITE_WINDOW"
    assert data["state_partition_axis"] == "GLOBAL"
    assert data["cross_section_scope"] == "PER_DATE"
    assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# 4. IncrementalCapabilityMatrix
# ---------------------------------------------------------------------------
def test_capability_matrix_covers_every_catalog_canonical_once() -> None:
    """矩阵对 operator catalog 每个 canonical 恰好一行，且每行含 10 个字段。"""
    canonicals = sorted(OperatorRegistry._catalog)
    matrix = incremental_capability_matrix()
    assert len(matrix) == len(canonicals) == len({row["canonical"] for row in matrix})
    fields = (
        "canonical", "incremental_mode", "incremental_certified",
        "backward_history", "forward_impact", "state_model",
        "state_partition_axis", "cross_section_scope",
        "checkpoint_schema", "checkpoint_schema_version",
    )
    for row in matrix:
        assert set(row) == set(fields), row
        assert row["canonical"] in canonicals
        assert row["incremental_mode"] in {
            mode.value for mode in IncrementalMode
        }, row
    # 显式列子集（前 5 个）也工作。
    subset = incremental_capability_matrix(canonicals=["ts_mean", "ts_ema", "log"])
    assert [row["canonical"] for row in subset] == ["ts_mean", "ts_ema", "log"]


# ---------------------------------------------------------------------------
# 5. node_incremental_slot
# ---------------------------------------------------------------------------
def test_node_incremental_slot_ts_mean_finite_window() -> None:
    """IR 节点 ts_mean(window=20) → FINITE_WINDOW。"""
    node = IRNode(
        op="ts_mean",
        inputs=(IRNode(op="column", attrs={"name": "close"}),),
        attrs={"window": 20},
    )
    contract = node_incremental_slot(node)
    assert isinstance(contract, IncrementalContract)
    assert contract.incremental_mode is IncrementalMode.FINITE_WINDOW
    assert contract.backward_history >= 1


def test_node_incremental_slot_positional_literal_maps_param() -> None:
    """位置 literal 子节点按 param_names 映射（等价 _node_params 技术）。"""
    node = IRNode(
        op="ts_mean",
        inputs=(
            IRNode(op="column", attrs={"name": "close"}),
            IRNode(op="literal", attrs={"value": 20}),
        ),
        attrs={},
    )
    contract = node_incremental_slot(node)
    assert contract.incremental_mode is IncrementalMode.FINITE_WINDOW
    assert contract.backward_history >= 1
