# -*- coding: utf-8
"""R20-036..165 identity / CSE-scope / typed-hash audit tests (R20-036..039 .. R20-102).

This agent's scope covers the optimizer differential, scope fail-closed, unified
execution identity, CSE scope binding and typed-hash rectifications.  Each test
is self-contained — no full ``load_all()`` (slow + concurrent-session-broken).

Covers (by R20 id):

- R20-036..039 — OptimizerDifferentialAudit (reference vs optimized, same data).
- R20-040..043 — RewriteTemporalProof (no future shift / no non-causal deps /
  no session-close -> intraday / history not underestimated).
- R20-044..046 — scoped-universe classifier fail-closed (UNKNOWN_SCOPE, not
  "not cross-sectional").
- R20-047..051 — canonicalize_execution_config: unsupported source config
  production hard fail (no ``str()`` fallback).
- R20-052..055 — ExecutionSemanticIdentityV2 projection chain invariant
  (identity -> CSE scope -> cache namespace -> checkpoint -> factor_version).
- R20-056..057 — FactorExecutionScope default market is UNKNOWN (""), not "A".
- R20-058..061 — CSE scope binds source_dependency_hash; secondary SourceRef
  differences isolate CSE / cache.
- R20-062..066 — universe membership hash (as-of resolved, not label compare);
  ``*_ALL`` whole-market requires market-family consistency.
- R20-067..072 — production contract-resolution gate (operator + field catalog).
- R20-073..078 — semantic attrs / identity reject repr / default=str fallback.
- R20-079..082 — Typed IR / Logical / Optimized / Physical hash layering.
- R20-083..087 — lineage layered hashes + backend route summary.
- R20-088..090 — PhysicalExecutionContract.
- R20-091..093 — SQL SourceRef three-state (NOT / VALID / MALFORMED).
- R20-094..099 — materialized_series typed boundary + downsample validation.
- R20-100..102 — canonical_target_index unification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from planner.dag import FactorExecutionScope
from planner.logical_plan import PlanNode, canonical_target_index
from planner.physical_plan import (
    DownsampleContractError,
    ExecKind,
    MaterializedSeriesContract,
    PhysicalExecutionContract,
    PhysicalNode,
    build_physical_execution_contract,
    materialized_series_contract,
    validate_downsample_contract,
)
from planner.cse import (
    PlanScopeCategory,
    UnresolvedScopeError,
    apply_cse,
    assert_cse_contracts_resolved,
    assert_plan_scope_resolved,
    classify_plan_scope,
    cse_scope_key,
)
from planner.plan_hash import (
    PlanHashKind,
    PlanSemanticAttrTypeError,
    POST_OPTIMIZATION_TEMPORAL_PROOF_PASS,
    RewriteTemporalProof,
    UnresolvedContractError,
    OptimizerDifferentialAudit,
    assert_plan_contracts_resolved,
    build_backend_route_summary,
    lineage_plan_hashes,
    logical_plan_hash,
    optimized_plan_hash,
    physical_plan_hash,
    source_expression_hash,
    structural_key,
    typed_ir_semantic_hash,
    typed_ir_structural_hash,
)
from planner.source_dependencies import (
    is_whole_market_label,
    universe_membership_hash,
)
from runtime.factor_identity import (
    ExecutionSemanticIdentityV2,
    FactorSemanticIdentity,
    IdentityHashTypeError,
    assert_identity_projection_chain,
    assert_projection_chain_distinct,
    projection_chain,
)
from storage.data_scope import (
    ExecutionConfigTypeError,
    canonicalize_execution_config,
    compute_execution_cache_scope,
    execution_scope_key,
)


# ---------------------------------------------------------------------------
# 构造 helpers
# ---------------------------------------------------------------------------
def _col(name: str = "close", **attrs: object) -> PlanNode:
    return PlanNode(op="column", attrs={"name": name, **attrs}, inputs=[])


def _lit(value: object) -> PlanNode:
    return PlanNode(op="literal", attrs={"value": value}, inputs=[])


def _walk(root: PlanNode):
    for c in root.inputs:
        yield from _walk(c)
    yield root


_SEM = {"grain": ("1d",), "price_basis": "RAW", "unit": "px", "available_at": "eod"}


# ---------------------------------------------------------------------------
# R20-036..039: OptimizerDifferentialAudit
# ---------------------------------------------------------------------------
def _eval_plan(plan: PlanNode, data: dict[str, pd.Series]) -> Any:
    """极小计划求值器（供 differential 测试；无需注册表）。"""
    if plan.op == "literal":
        return float(plan.attrs["value"])
    if plan.op == "column":
        return data[plan.attrs["name"]]
    if plan.op == "add":
        return _eval_plan(plan.inputs[0], data) + _eval_plan(plan.inputs[1], data)
    if plan.op == "subtract":
        return _eval_plan(plan.inputs[0], data) - _eval_plan(plan.inputs[1], data)
    if plan.op == "multiply":
        return _eval_plan(plan.inputs[0], data) * _eval_plan(plan.inputs[1], data)
    if plan.op == "divide":
        return _eval_plan(plan.inputs[0], data) / _eval_plan(plan.inputs[1], data)
    raise ValueError(f"unsupported op for differential test: {plan.op}")


class _IdentityOpt:
    """reference-style optimizer: 不做任何 rewrite。"""

    def optimize(self, plan: PlanNode, *, production: bool = False) -> PlanNode:
        return plan


class _FoldConstOpt:
    """optimizer-style: 常量折叠 add(lit, lit) -> lit。"""

    def optimize(self, plan: PlanNode, *, production: bool = False) -> PlanNode:
        if plan.op == "add" and len(plan.inputs) == 2:
            a, b = plan.inputs
            if a.op == "literal" and b.op == "literal":
                return _lit(float(a.attrs["value"]) + float(b.attrs["value"]))
        return plan


class _WrongValueOpt:
    """buggy optimizer: 把结果替换为错误值（用于检测 mismatch）。"""

    def optimize(self, plan: PlanNode, *, production: bool = False) -> PlanNode:
        return _lit(99.0)


def _ts_data() -> dict[str, pd.Series]:
    idx = pd.date_range("2024-01-01", periods=4)
    return {"close": pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)}


def test_r20_036_differential_passes_for_semantic_preserving_rewrite():
    plan = PlanNode(op="add", inputs=[_lit(1), _lit(2)])
    data = _ts_data()
    audit = OptimizerDifferentialAudit(
        optimizer=_FoldConstOpt(), reference_optimizer=_IdentityOpt()
    )
    result = audit.audit(plan, executor=lambda p: _eval_plan(p, data))
    assert result.passed, result.summary()


def test_r20_037_differential_detects_value_mismatch():
    plan = PlanNode(op="add", inputs=[_lit(1), _lit(2)])
    data = _ts_data()
    audit = OptimizerDifferentialAudit(
        optimizer=_WrongValueOpt(), reference_optimizer=_IdentityOpt()
    )
    result = audit.audit(plan, executor=lambda p: _eval_plan(p, data))
    assert not result.passed
    attrs = [m.attribute for m in result.mismatches]
    assert any("finite_values" in a for a in attrs)


def test_r20_038_differential_detects_semantic_attr_change():
    ref = PlanNode(
        op="ts_mean",
        inputs=[_col(), _lit(2)],
        semantic_attrs={"available_at": "eod"},
    )
    opt = PlanNode(
        op="ts_mean",
        inputs=[_col(), _lit(2)],
        semantic_attrs={"available_at": "session_open"},
    )

    class _SemShift:
        def optimize(self, plan, *, production=False):
            return opt

    data = _ts_data()
    audit = OptimizerDifferentialAudit(
        optimizer=_SemShift(), reference_optimizer=_IdentityOpt()
    )
    result = audit.audit(ref, executor=lambda p: data["close"])
    assert not result.passed
    assert any("semantic.available_at" in m.attribute for m in result.mismatches)


def test_r20_039_differential_detects_new_source_dependency():
    from api.source_ref import make_source_ref, encode_source_ref

    ref_name = encode_source_ref(make_source_ref("StockIncome", "NetProfit"))
    ref = _col(name=ref_name)
    other = _col(name=encode_source_ref(make_source_ref("StockBalance", "TotalAssets")))

    class _AddDep:
        def optimize(self, plan, *, production=False):
            # 引入 reference 中没有的二级 SourceRef 依赖
            return PlanNode(op="add", inputs=[plan, other])

    data = _ts_data()
    audit = OptimizerDifferentialAudit(
        optimizer=_AddDep(), reference_optimizer=_IdentityOpt()
    )
    result = audit.audit(ref, executor=lambda p: data["close"])
    assert not result.passed
    assert any("source_dependencies" in m.attribute for m in result.mismatches)


# ---------------------------------------------------------------------------
# R20-040..043: RewriteTemporalProof
# ---------------------------------------------------------------------------
def test_r20_040_temporal_proof_passes_identical_plans():
    ref = PlanNode(
        op="ts_mean", inputs=[_col(), _lit(2)], semantic_attrs={"available_at": "eod"}
    )
    opt = PlanNode(
        op="ts_mean", inputs=[_col(), _lit(2)], semantic_attrs={"available_at": "eod"}
    )
    proof = RewriteTemporalProof()
    result = proof.prove(ref, opt)
    assert result.passed, result.violations
    assert result.marker == POST_OPTIMIZATION_TEMPORAL_PROOF_PASS


def test_r20_041_temporal_proof_rejects_availability_shift():
    ref = PlanNode(
        op="ts_mean", inputs=[_col(), _lit(2)], semantic_attrs={"available_at": "eod"}
    )
    opt = PlanNode(
        op="ts_mean",
        inputs=[_col(), _lit(2)],
        semantic_attrs={"available_at": "session_open"},
    )
    result = RewriteTemporalProof().prove(ref, opt)
    assert not result.passed
    assert any("session-close" in v or "available_at" in v for v in result.violations)


def test_r20_042_temporal_proof_rejects_non_causal_new_dependency():
    from api.source_ref import make_source_ref, encode_source_ref

    ref = _col(name=encode_source_ref(make_source_ref("StockIncome", "NetProfit")))
    new_dep = _col(name=encode_source_ref(make_source_ref("StockBalance", "TotalAssets")))
    opt = PlanNode(op="add", inputs=[ref, new_dep])
    result = RewriteTemporalProof().prove(ref, opt)
    assert not result.passed
    assert any("new source dependencies" in v for v in result.violations)


def test_r20_043_temporal_proof_rejects_history_underestimation():
    ref = PlanNode(
        op="ts_mean",
        inputs=[_col(), _lit(2)],
        semantic_attrs={"available_at": "eod", "history": 20},
    )
    opt = PlanNode(
        op="ts_mean",
        inputs=[_col(), _lit(2)],
        semantic_attrs={"available_at": "eod", "history": 5},
    )
    result = RewriteTemporalProof().prove(ref, opt)
    assert not result.passed
    assert any("history requirement underestimated" in v for v in result.violations)


# ---------------------------------------------------------------------------
# R20-044..046: scoped-universe classifier fail-closed
# ---------------------------------------------------------------------------
def test_r20_044_scope_classifier_cross_sectional():
    rank = PlanNode(op="rank", inputs=[_col()])
    assert classify_plan_scope(rank) is PlanScopeCategory.CROSS_SECTIONAL
    assert classify_plan_scope(PlanNode(op="cs_regression", inputs=[_col(), _col("r")])) is PlanScopeCategory.CROSS_SECTIONAL


def test_r20_045_scope_classifier_not_cross_sectional():
    ts = PlanNode(op="ts_mean", inputs=[_col(), _lit(2)])
    assert classify_plan_scope(ts) is PlanScopeCategory.NOT_CROSS_SECTIONAL
    assert classify_plan_scope(None) is PlanScopeCategory.NOT_CROSS_SECTIONAL


def test_r20_046_unknown_scope_fails_closed_in_production(monkeypatch):
    """registry lookup 异常 → UNKNOWN_SCOPE（不是「不是横截面」），production hard fail。

    用 fake ``cleaned_operators.registry`` 注入 sys.modules，避免触发真实
    registry（并发会话下 load_all 间歇性 broken）。
    """
    import sys
    import types

    fake_registry = types.ModuleType("cleaned_operators.registry")

    class _BrokenRegistry:
        @staticmethod
        def resolve_canonical(op):
            raise RuntimeError("registry broken")

        @staticmethod
        def get(op):
            return None

    fake_registry.OperatorRegistry = _BrokenRegistry
    fake_pkg = types.ModuleType("cleaned_operators")
    fake_pkg.registry = fake_registry
    monkeypatch.setitem(sys.modules, "cleaned_operators.registry", fake_registry)
    monkeypatch.setitem(sys.modules, "cleaned_operators", fake_pkg)

    plan = PlanNode(op="mystery_op", inputs=[_col()])
    assert classify_plan_scope(plan) is PlanScopeCategory.UNKNOWN_SCOPE
    with pytest.raises(UnresolvedScopeError):
        assert_plan_scope_resolved(plan, production=True)
    # research 下返回 UNKNOWN_SCOPE（不静默回落成 NOT_CROSS_SECTIONAL）
    assert (
        assert_plan_scope_resolved(plan, production=False)
        is PlanScopeCategory.UNKNOWN_SCOPE
    )


# ---------------------------------------------------------------------------
# R20-047..051: canonicalize_execution_config
# ---------------------------------------------------------------------------
def test_r20_047_unsupported_source_config_object_hard_fails():
    class Weird:
        pass

    with pytest.raises(ExecutionConfigTypeError):
        canonicalize_execution_config({"weird": Weird()})
    # 绝不 str() 兜底 —— str 后会是 <... object at 0x...> 不稳定表示
    with pytest.raises(ExecutionConfigTypeError):
        canonicalize_execution_config_strict = canonicalize_execution_config
        canonicalize_execution_config_strict({"weird": lambda: None})


def test_r20_048_canonical_config_handles_known_types():
    from datetime import date, datetime
    from enum import Enum
    from pathlib import Path

    class E(Enum):
        X = "x"

    cfg = {
        "dataset": "ashare_daily",
        "fields": {"close": "close_px"},
        "params": {"lookback": 20},
        "when": date(2024, 1, 1),
        "now": datetime(2024, 1, 1, 10, 0),
        "p": Path("/tmp/x"),
        "enum": E.X,
    }
    canon = canonicalize_execution_config(cfg)
    assert canon["enum"] == "x"
    assert canon["when"] == "2024-01-01"
    # 确定性
    assert canonicalize_execution_config(cfg) == canon


def test_r20_469_unsupported_source_config_production_hard_fail_no_stringify():
    """R20-469：unsupported source config 在 production 下 hard fail，不 stringify。"""
    class CallableConfig:
        def __call__(self):  # pragma: no cover
            pass

    # engine 旧 `_stable_config` 会对未知对象 `str(value)`（含内存地址）。
    # canonicalize_execution_config 必须 fail-closed。
    with pytest.raises(ExecutionConfigTypeError):
        canonicalize_execution_config({"on_complete": CallableConfig()})


# ---------------------------------------------------------------------------
# R20-052..055: ExecutionSemanticIdentityV2 投影链
# ---------------------------------------------------------------------------
def _identity(**overrides):
    base = dict(
        ir_hash="ir",
        operator_contract_hash="op",
        field_contract_hash="field",
        source_contract_hash="src",
        source_dependency_hash="dep",
        universe_membership_hash="members",
        market="A",
        frequency="1d",
    )
    base.update(overrides)
    return ExecutionSemanticIdentityV2(**base)


def test_r20_052_identity_projection_chain_is_deterministic():
    chain1 = assert_identity_projection_chain(_identity())
    chain2 = assert_identity_projection_chain(_identity())
    assert chain1 == chain2


def test_r20_053_identity_projection_chain_changes_with_membership():
    a = assert_identity_projection_chain(_identity(universe_membership_hash="m1"))
    b = assert_identity_projection_chain(_identity(universe_membership_hash="m2"))
    assert a != b
    assert a["cse_scope"] != b["cse_scope"]
    assert a["cache_namespace"] != b["cache_namespace"]
    assert a["factor_version"] != b["factor_version"]


def test_r20_054_identity_projection_chain_distinct_for_different_identity():
    assert assert_projection_chain_distinct(_identity(), _identity(market="US"))
    assert assert_projection_chain_distinct(
        _identity(), _identity(source_dependency_hash="other-dep")
    )


def test_r20_055_factor_semantic_identity_roundtrip_preserves_membership():
    fid = FactorSemanticIdentity(
        ir_hash="a", operator_contract_hash="b", field_contract_hash="c",
        source_contract_hash="d", source_dependency_hash="e",
        universe_membership_hash="m1",
    )
    v2 = ExecutionSemanticIdentityV2.from_factor_identity(fid)
    assert v2.universe_membership_hash == "m1"
    restored = FactorSemanticIdentity.from_dict(fid.to_dict())
    assert restored.universe_membership_hash == "m1"
    assert restored.identity_digest() == fid.identity_digest()


# ---------------------------------------------------------------------------
# R20-056..057: FactorExecutionScope 默认 market 不能是 A
# ---------------------------------------------------------------------------
def test_r20_056_factor_execution_scope_default_market_is_unknown():
    scope = FactorExecutionScope()
    assert scope.market == ""
    assert scope.market != "A"


def test_r20_057_factor_execution_scope_default_does_not_reintroduce_a():
    # 手动 DAG 构造不得重新引入 A 股默认污染
    scope = FactorExecutionScope(frequency="1d", universe_id="US")
    assert scope.market == ""
    # 显式声明 market 时保留
    assert FactorExecutionScope(market="A").market == "A"


# ---------------------------------------------------------------------------
# R20-058..061: CSE scope 绑定真实二级 SourceRef 依赖
# ---------------------------------------------------------------------------
def _source_ref_column(table: str, field: str) -> PlanNode:
    from api.source_ref import encode_source_ref, make_source_ref

    name = encode_source_ref(make_source_ref(table, field))
    return _col(name=name)


def test_r20_470_secondary_source_ref_different_factors_do_not_share_cse_scope():
    """R20-470：secondary SourceRef 不同的 factor 不得跨 dependency CSE 共享。"""
    from planner.source_dependencies import source_dependency_hash

    f1 = _source_ref_column("StockIncome", "NetProfit")
    f2 = _source_ref_column("StockBalance", "TotalAssets")
    h1 = source_dependency_hash(f1)
    h2 = source_dependency_hash(f2)
    assert h1 != h2

    scope = FactorExecutionScope(frequency="1d", universe_id="ALL")
    k1 = cse_scope_key(scope, source_dependency_hash=h1)
    k2 = cse_scope_key(scope, source_dependency_hash=h2)
    assert k1 != k2

    # cache namespace 同样隔离
    ns1 = compute_execution_cache_scope(None, source_dependencies=(h1,))
    ns2 = compute_execution_cache_scope(None, source_dependencies=(h2,))
    assert ns1 != ns2


def test_r20_059_same_source_dependency_same_cse_scope():
    from planner.source_dependencies import source_dependency_hash

    f1 = _source_ref_column("StockIncome", "NetProfit")
    f2 = _source_ref_column("StockIncome", "NetProfit")
    h = source_dependency_hash(f1)
    assert source_dependency_hash(f2) == h
    scope = FactorExecutionScope()
    assert cse_scope_key(scope, source_dependency_hash=h) == cse_scope_key(
        scope, source_dependency_hash=h
    )


def test_r20_060_source_dependency_hash_enters_plan_key():
    """structural_key / plan_cache_key 已把二级 SourceRef 依赖编入（源列名即
    SourceRef 身份，不同依赖 = 不同 column name = 不同 key）。"""
    f1 = _source_ref_column("StockIncome", "NetProfit")
    f2 = _source_ref_column("StockBalance", "TotalAssets")
    assert structural_key(f1) != structural_key(f2)
    assert structural_key(_col("close") != structural_key(f1)


# ---------------------------------------------------------------------------
# R20-062..066: universe membership hash
# ---------------------------------------------------------------------------
def test_r20_062_membership_hash_binds_actual_membership():
    h_a = universe_membership_hash(
        "CSI300", as_of="2024-01-01", membership=["600000.SH", "600036.SH"]
    )
    h_b = universe_membership_hash(
        "CSI300", as_of="2024-01-01", membership=["600000.SH", "600036.SH"]
    )
    assert h_a == h_b
    h_c = universe_membership_hash(
        "CSI300", as_of="2024-01-01", membership=["600000.SH"]
    )
    assert h_a != h_c


def test_r20_063_membership_hash_sensitive_to_as_of():
    h1 = universe_membership_hash(
        "CSI300", as_of="2024-01-01", membership=["600000.SH"]
    )
    h2 = universe_membership_hash(
        "CSI300", as_of="2024-06-01", membership=["600000.SH", "601318.SH"]
    )
    assert h1 != h2


def test_r20_064_whole_market_requires_market_family_consistency():
    assert is_whole_market_label("ALL") is True
    assert is_whole_market_label("") is True
    assert is_whole_market_label("A", market="A") is True
    assert is_whole_market_label("ASHARE_ALL", market="A") is True
    assert is_whole_market_label("US_MASSIVE_ALL", market="US") is True
    assert is_whole_market_label("US_NASDAQ_ALL", market="US") is True
    # market family 不一致 → 不是 whole market（fail-closed）
    assert is_whole_market_label("ASHARE_ALL", market="US") is False
    assert is_whole_market_label("US_MASSIVE_ALL", market="A") is False
    assert is_whole_market_label("US_NASDAQ_ALL", market="A") is False
    assert is_whole_market_label("CSI300", market="A") is False


def test_r20_065_membership_hash_into_cache_namespace():
    ns1 = compute_execution_cache_scope(None, universe_membership_hash="m1")
    ns2 = compute_execution_cache_scope(None, universe_membership_hash="m2")
    assert ns1 != ns2


def test_r20_066_universe_membership_hash_deterministic_whole_market():
    a = universe_membership_hash("ASHARE_ALL")
    b = universe_membership_hash("ASHARE_ALL")
    assert a == b


# ---------------------------------------------------------------------------
# R20-067..072: production contract-resolution gate
# ---------------------------------------------------------------------------
def test_r20_067_unresolved_operator_contract_production_fails(monkeypatch):
    from planner import plan_hash as ph

    plan = PlanNode(op="some_op", inputs=[_col()])
    # 模拟 registry lookup 失败 → semantic_version="unregistered"
    monkeypatch.setattr(
        ph, "_operator_semantic_contract", lambda op: {"semantic_version": "unregistered"}
    )
    with pytest.raises(UnresolvedContractError):
        assert_plan_contracts_resolved(plan, production=True)
    # research 显式放行（unresolved 计划 research 可用）
    assert_plan_contracts_resolved(plan, production=False)


def test_r20_068_field_catalog_unavailable_production_fails(monkeypatch):
    from planner import plan_hash as ph

    plan = PlanNode(op="ts_mean", inputs=[_col()])
    # 只有 operator contract resolved，但 field catalog hash 失败
    monkeypatch.setattr(
        ph, "_operator_semantic_contract", lambda op: {"semantic_version": "1.0"}
    )
    monkeypatch.setattr(
        "fields.compute_field_catalog_hash", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(UnresolvedContractError):
        assert_plan_contracts_resolved(plan, production=True)


def test_r20_069_resolved_contract_production_passes(monkeypatch):
    from planner import plan_hash as ph

    plan = PlanNode(op="ts_mean", inputs=[_col()])
    monkeypatch.setattr(
        ph, "_operator_semantic_contract", lambda op: {"semantic_version": "1.0"}
    )
    assert_plan_contracts_resolved(plan, production=True)


def test_r20_070_cse_contract_gate_fails_closed(monkeypatch):
    from planner import plan_hash as ph

    plan = PlanNode(op="some_op", inputs=[_col()])
    monkeypatch.setattr(
        ph, "_operator_semantic_contract", lambda op: {"semantic_version": "unregistered"}
    )
    with pytest.raises(UnresolvedContractError):
        assert_cse_contracts_resolved(plan, production=True)
    assert_cse_contracts_resolved(plan, production=False)


def test_r20_071_structural_key_still_works_for_resolved_plans(monkeypatch):
    from planner import plan_hash as ph

    plan = PlanNode(op="ts_mean", inputs=[_col()])
    monkeypatch.setattr(
        ph, "_operator_semantic_contract", lambda op: {"semantic_version": "1.0"}
    )
    key = structural_key(plan)
    assert isinstance(key, str) and len(key) == 64


def test_r20_072_optimized_plan_hash_production_gate(monkeypatch):
    from planner import plan_hash as ph

    plan = PlanNode(op="some_op", inputs=[_col()])
    monkeypatch.setattr(
        ph, "_operator_semantic_contract", lambda op: {"semantic_version": "unregistered"}
    )
    with pytest.raises(UnresolvedContractError):
        optimized_plan_hash(plan, production=True)
    # research 允许
    assert isinstance(optimized_plan_hash(plan, production=False), str)


# ---------------------------------------------------------------------------
# R20-073..078: repr / default=str 收口
# ---------------------------------------------------------------------------
def test_r20_073_semantic_attr_unsupported_object_raises():
    bad = PlanNode(op="column", attrs={"name": "x"}, semantic_attrs={"obj": object()})
    from planner.plan_hash import _semantic_digest

    with pytest.raises(PlanSemanticAttrTypeError):
        _semantic_digest(bad)


def test_r20_074_semantic_attr_session_close_label_normalized():
    from planner.plan_hash import _semantic_digest

    from ir.types import SessionClose
    n = PlanNode(
        op="column", attrs={"name": "close"},
        semantic_attrs={"available_at": SessionClose()},
    )
    assert _semantic_digest(n) is not None
    # 与字符串 label 归一一致
    n_str = PlanNode(
        op="column", attrs={"name": "close"},
        semantic_attrs={"available_at": "SessionClose:session_close"},
    )
    assert _semantic_digest(n) == _semantic_digest(n_str)


def test_r20_075_identity_payload_unsupported_object_raises():
    import runtime.factor_identity as fi

    with pytest.raises(IdentityHashTypeError):
        fi._stable_hash({"x": object()})


def test_r20_076_identity_scalar_unsupported_object_raises():
    import runtime.factor_identity as fi

    with pytest.raises(IdentityHashTypeError):
        fi._scalar(object())


def test_r20_077_semantic_attr_known_types_ok():
    from planner.plan_hash import _semantic_digest

    n = PlanNode(
        op="column", attrs={"name": "x"},
        semantic_attrs={
            "grain": ("1d",),
            "available_at": "eod",
            "lattice": {"kind": "PriceRaw"},
            "tags": frozenset({"a", "b"}),
        },
    )
    assert _semantic_digest(n) is not None


def test_r20_078_typed_semantic_value_nested_unsupported_raises():
    from planner.plan_hash import _typed_semantic_value

    with pytest.raises(PlanSemanticAttrTypeError):
        _typed_semantic_value({"inner": {"deep": object()}})


# ---------------------------------------------------------------------------
# R20-079..082: Typed IR / Logical / Optimized / Physical hash layering
# ---------------------------------------------------------------------------
def test_r20_079_hash_kinds_are_distinct_namespaces():
    plan = PlanNode(op="add", inputs=[_col(), _lit(1)])
    structural = structural_key(plan)
    assert typed_ir_structural_hash(plan) != structural
    assert logical_plan_hash(plan) != structural
    assert typed_ir_semantic_hash(plan) != typed_ir_structural_hash(plan)
    assert logical_plan_hash(plan) != optimized_plan_hash(plan, production=False)


def test_r20_080_typed_ir_semantic_hash_differs_on_semantic_attrs():
    a = PlanNode(op="add", inputs=[_col(), _lit(1)], semantic_attrs={"price_basis": "RAW"})
    b = PlanNode(op="add", inputs=[_col(), _lit(1)], semantic_attrs={"price_basis": "CONTINUOUS"})
    assert typed_ir_semantic_hash(a) != typed_ir_semantic_hash(b)
    # 结构哈希忽略 semantic attrs
    assert typed_ir_structural_hash(a) == typed_ir_structural_hash(b)


def test_r20_081_physical_plan_hash_distinct():
    from planner.physical_plan import PhysicalPlan

    plan = PlanNode(op="add", inputs=[_col(), _lit(1)])
    pp1 = PhysicalPlan(root=plan, sql_subtrees={}, fully_sql=False)
    pp2 = PhysicalPlan(root=plan, sql_subtrees={"sid": plan}, fully_sql=False)
    assert physical_plan_hash(pp1) != physical_plan_hash(pp2)


def test_r20_082_optimized_plan_hash_kind_prefixed():
    plan = PlanNode(op="add", inputs=[_col(), _lit(1)])
    h = optimized_plan_hash(plan, production=False)
    assert isinstance(h, str) and len(h) == 64
    assert h != logical_plan_hash(plan)


# ---------------------------------------------------------------------------
# R20-083..087: lineage layered hashes + backend route summary
# ---------------------------------------------------------------------------
def test_r20_083_lineage_plan_hashes_are_layered():
    plan = PlanNode(op="add", inputs=[_col(), _lit(1)])
    hashes = lineage_plan_hashes(plan, production=False)
    assert set(hashes) == {
        "typed_ir_hash", "logical_plan_hash", "optimized_plan_hash",
        "physical_plan_hash", "execution_semantic_hash",
    }
    assert hashes["typed_ir_hash"] != hashes["logical_plan_hash"]
    assert hashes["logical_plan_hash"] != hashes["optimized_plan_hash"]


def test_r20_084_lineage_backend_route_summary():
    from planner.physical_plan import PhysicalPlan

    sql_node = PlanNode(op="ts_mean", inputs=[_col()])
    py_node = PlanNode(op="log", inputs=[_col()])
    phys = PhysicalNode(kind=ExecKind.SQL, plan=sql_node, sid="s1")
    sub = PhysicalNode(kind=ExecKind.PYTHON, plan=py_node)
    # build_backend_route_summary consumes a PhysicalPlan-like with root + subtrees
    pp = PhysicalPlan(root=sql_node, sql_subtrees={"s1": sql_node})
    summary = build_backend_route_summary(pp, fallback_events=[{"op": "log", "route": "fallback"}])
    bp = summary["backend_path_summary"]
    assert bp["fallback_count"] == 1
    assert bp["native_count"] >= 1
    assert bp["per_operator_routes"]["ts_mean"] == "sql"


def test_r20_085_lineage_physical_hash_from_physical_plan():
    from planner.physical_plan import PhysicalPlan

    plan = PlanNode(op="add", inputs=[_col(), _lit(1)])
    pp = PhysicalPlan(root=plan, sql_subtrees={}, fully_sql=True)
    h = physical_plan_hash(pp)
    assert isinstance(h, str) and len(h) == 64


# ---------------------------------------------------------------------------
# R20-088..090: PhysicalExecutionContract
# ---------------------------------------------------------------------------
def test_r20_088_physical_execution_contract_built():
    node = PhysicalNode(
        kind=ExecKind.SQL,
        plan=PlanNode(op="ts_mean", inputs=[_col()], semantic_attrs=dict(_SEM)),
    )
    contract = build_physical_execution_contract(node, source_snapshot="snap-1")
    assert isinstance(contract, PhysicalExecutionContract)
    assert contract.native_backend == "sql"
    assert contract.input_grain == "1d"
    assert contract.availability == "eod"
    assert contract.source_snapshot == "snap-1"


def test_r20_089_physical_node_carries_contract():
    plan = PlanNode(op="literal", attrs={"value": 1.0})
    contract = build_physical_execution_contract(PhysicalNode(kind=ExecKind.PYTHON, plan=plan))
    node = PhysicalNode(kind=ExecKind.PYTHON, plan=plan, contract=contract)
    assert node.contract is contract


def test_r20_090_contract_to_dict_serializable():
    contract = PhysicalExecutionContract(
        output_semantic_digest="d", native_backend="sql",
        input_grain="1d", output_grain="1d", availability="eod",
    )
    d = contract.to_dict()
    assert d["native_backend"] == "sql"
    assert d["output_grain"] == "1d"


# ---------------------------------------------------------------------------
# R20-091..093: SQL SourceRef 三态
# ---------------------------------------------------------------------------
def _valid_source_ref_name(table: str, field: str) -> str:
    from api.source_ref import encode_source_ref, make_source_ref

    return encode_source_ref(make_source_ref(table, field))


def test_r20_091_source_ref_three_states():
    from planner.sql_lowerer import SourceRefStatus, classify_source_ref

    valid = _col(name=_valid_source_ref_name("StockIncome", "NetProfit"))
    assert classify_source_ref(valid) is SourceRefStatus.VALID_SOURCE_REF

    plain = _col(name="close")
    assert classify_source_ref(plain) is SourceRefStatus.NOT_SOURCE_REF

    malformed = _col(name="__fe_source_ref_v1__%%%bad%%%")
    assert classify_source_ref(malformed) is SourceRefStatus.MALFORMED_SOURCE_REF


def test_r20_092_malformed_source_ref_production_hard_fails():
    from planner.sql_lowerer import _contains_source_ref

    malformed = _col(name="__fe_source_ref_v1__%%%bad%%%")
    # research: 按 opaque 处理（非 SQL-capable）
    assert _contains_source_ref(malformed, production=False) is True
    # production: hard fail
    with pytest.raises(ValueError, match="malformed SourceRef"):
        _contains_source_ref(malformed, production=True)


def test_r20_093_valid_and_plain_source_refs_sql_detection():
    from planner.sql_lowerer import _contains_source_ref

    valid = _col(name=_valid_source_ref_name("StockIncome", "NetProfit"))
    assert _contains_source_ref(valid, production=True) is True
    assert _contains_source_ref(_col("close"), production=True) is False
    # 嵌套子树
    root = PlanNode(op="add", inputs=[_col("close"), valid])
    assert _contains_source_ref(root, production=True) is True


# ---------------------------------------------------------------------------
# R20-094..099: materialized_series typed boundary + downsample validation
# ---------------------------------------------------------------------------
def test_r20_094_materialized_series_typed_contract():
    n = PlanNode(
        op="materialized_series", attrs={"sid": "abc"},
        semantic_attrs={
            "grain": ("1d",), "frequency": "1d", "available_at": "eod",
            "unit": "px", "price_basis": "RAW",
        },
    )
    contract = materialized_series_contract(n)
    assert isinstance(contract, MaterializedSeriesContract)
    assert contract.frequency == "1d"
    assert contract.available_at == "eod"
    assert contract.price_basis == "RAW"


def test_r20_095_materialized_series_contract_fallback():
    n = PlanNode(op="materialized_series", attrs={"sid": "abc"})
    fb = MaterializedSeriesContract(
        grain_kind="daily", frequency="1d", calendar="SSE", timezone="Asia/Shanghai"
    )
    contract = materialized_series_contract(n, fallback=fb)
    assert contract.frequency == "1d"
    assert contract.calendar == "SSE"


def test_r20_096_downsample_rejects_upsampling():
    idx = pd.date_range("2024-01-01", periods=5)
    panel = pd.DataFrame(np.arange(10).reshape(5, 2), index=idx, columns=["A", "B"])
    with pytest.raises(DownsampleContractError):
        validate_downsample_contract(panel.iloc[:3], declared_grain="1d", template_panel=panel.iloc[:1])


def test_r20_097_downsample_rejects_non_datetime_index():
    df = pd.DataFrame({"x": [1, 2, 3]})
    panel = pd.DataFrame({"x": [1]}, index=pd.date_range("2024-01-01", periods=1))
    with pytest.raises(DownsampleContractError):
        validate_downsample_contract(df, declared_grain="1d", template_panel=panel)


def test_r20_098_downsample_rejects_duplicate_timestamps():
    dup = pd.DataFrame(
        {"x": [1, 1]}, index=pd.to_datetime(["2024-01-01", "2024-01-01"])
    )
    panel = pd.DataFrame({"x": [1, 2]}, index=pd.date_range("2024-01-01", periods=2))
    with pytest.raises(DownsampleContractError):
        validate_downsample_contract(dup, declared_grain="1d", template_panel=panel)


def test_r20_099_downsample_timezone_mismatch_rejected():
    idx_utc = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    panel = pd.DataFrame({"x": [1, 2, 3]}, index=idx_utc)
    with pytest.raises(DownsampleContractError):
        validate_downsample_contract(
            panel, declared_grain="1d", template_panel=panel, timezone="Asia/Shanghai"
        )
    # 匹配时区放行
    validate_downsample_contract(
        panel, declared_grain="1d", template_panel=panel, timezone="UTC"
    )


# ---------------------------------------------------------------------------
# R20-100..102: canonical_target_index 统一
# ---------------------------------------------------------------------------
def test_r20_100_canonical_target_index_series():
    idx = pd.date_range("2024-01-01", periods=3)
    s = pd.Series([1, 2, 3], index=idx)
    assert canonical_target_index(s) is idx


def test_r20_101_canonical_target_index_index():
    idx = pd.date_range("2024-01-01", periods=3)
    assert canonical_target_index(idx) is idx


def test_r20_102_canonical_target_index_consistent_across_branches():
    """1-D ndarray 分支必须用 canonical_target_index(template) 而非 template.index
    —— 两种模板（Series / Index）得到同一 target。"""
    idx = pd.date_range("2024-01-01", periods=3)
    series_target = canonical_target_index(pd.Series([0, 0, 0], index=idx))
    index_target = canonical_target_index(idx)
    assert series_target.equals(index_target)
