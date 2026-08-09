# -*- coding: utf-8 -*-
"""R14 #4：事件增量 rebuild 恢复完整 canonical execution scope。

外部 AI 复查第二轮 P0（上轮 #3 的残口）：``factor_from_catalog_info`` 只调
``parse_factor``，不恢复 ``decision_time_policy / market / calendar``，而
``_verify_factor_semantic_identity`` 只检查「actual 非空时不同」——declared
``decision_policy="eod"`` + rebuilt ``actual=""`` 静默通过（fail-open）。

R14 #4 修复：
  * rebuild 后给 Factor 挂 ``FactorExecutionScopeHint``（market/universe_id/
    frequency/calendar_id/decision_time_policy 取自 full definition）；
  * ``_scope_from_factor`` 消费该 hint → canonical scope 与落库时一致；
  * ``_verify_factor_semantic_identity`` 对 market/universe/frequency/calendar/
    decision_policy 全量 fail-closed：declared 有值 + actual 缺失 → production 拒绝。
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from api import rank
from api.columns import col
from runtime.engine import _scope_from_factor
from runtime.incremental_scheduler import (
    FactorSemanticIdentityMismatch,
    _verify_factor_semantic_identity,
    factor_from_catalog_info,
)


def _full_def(**overrides) -> dict:
    """full definition：只带 scope 字段，不带 ast_hash。

    不声明 surface/dialect（走默认 daily/native，避开并发会话 cleaned_operators
    WIP 的 flaky load_all 治理检查）；不声明 ast_hash——Expr 重建路径下
    ``compute_ir_hash`` 不可用，且本组测试目标是 scope 校验，不是 hash。
    """
    d = {
        "factor_id": "f_scope",
        "expression": 'rank(col("close"))',
        "market": "A",
        "universe": "CSI300",
        "frequency": "1d",
        "calendar": "SSE",
        "decision_policy": "eod",
    }
    d.update(overrides)
    return d


def test_r14_rebuild_restores_decision_policy_scope():
    factor = factor_from_catalog_info(_full_def())
    assert factor.semantic_identity is not None
    hint = factor.semantic_identity
    assert hint.decision_time_policy == "eod"
    assert hint.market == "A"
    assert hint.universe_id == "CSI300"
    assert hint.frequency == "1d"
    assert hint.calendar_id == "SSE"
    # _scope_from_factor 消费 hint → canonical scope 与落库时一致（不是空决策策略）
    scope = _scope_from_factor(factor)
    assert scope.decision_time_policy == "eod"
    assert scope.market == "A"
    assert scope.universe_id == "CSI300"
    assert scope.calendar_id == "SSE"


def test_r14_rebuild_without_scope_fields_keeps_plain_factor():
    """full definition 无 scope 字段（legacy）→ 不伪造 hint，仍返回普通 Factor。"""
    factor = factor_from_catalog_info(
        {"factor_id": "f", "expression": 'rank(col("close"))'}
    )
    assert factor.semantic_identity is None
    assert factor.source_expr == 'rank(col("close"))'


def test_r14_verify_production_fails_when_declared_policy_missing():
    """declared decision_policy=eod + rebuilt 无 hint → production fail-closed。"""
    full_def = _full_def()
    factor = factor_from_catalog_info(full_def)
    # 把 hint 摘掉（模拟旧 rebuild：只 parse_factor，不恢复 scope）
    factor = dataclasses.replace(factor, semantic_identity=None)
    with pytest.raises(FactorSemanticIdentityMismatch, match="decision_policy"):
        _verify_factor_semantic_identity(factor, full_def, production=True)
    # research：不抛（warning 路径）
    _verify_factor_semantic_identity(factor, full_def, production=False)


def test_r14_verify_production_passes_when_scope_restored():
    """rebuild 恢复 hint 且与 catalog 一致 → production 通过。"""
    full_def = _full_def()
    factor = factor_from_catalog_info(full_def)
    _verify_factor_semantic_identity(factor, full_def, production=True)


def test_r14_verify_production_rejects_wrong_market():
    """hint 恢复了但 market 与 catalog 不一致 → production 拒绝。"""
    full_def = _full_def()
    factor = factor_from_catalog_info(full_def)
    factor = dataclasses.replace(
        factor, semantic_identity=dataclasses.replace(
            factor.semantic_identity, market="US"
        )
    )
    with pytest.raises(FactorSemanticIdentityMismatch, match="market"):
        _verify_factor_semantic_identity(factor, full_def, production=True)
