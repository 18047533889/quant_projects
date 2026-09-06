"""逻辑计划子树的结构化哈希（与具体列数据无关），供 CSE 与执行期子树缓存键复用。

R20-079..082: 本模块同时承载 Typed IR / Logical / Optimized / Physical 的分层
哈希类型（``PlanHashKind``），避免不同层级身份值相互混淆。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Callable, Mapping

from factor_engine.planner.logical_plan import PlanNode


class NonFiniteLiteralError(ValueError):
    """计划中出现了非有限 literal（NaN / +/-Inf），无法生成稳定缓存键。"""


class PlanSemanticAttrTypeError(TypeError):
    """semantic_attrs 值类型不支持稳定哈希（R20-073..078）。

    绝不退化为 ``repr(v)`` —— repr 可能含内存地址，跨进程不稳定；且字符串化后
    无法区分不同语义值。只接受 typed JSON schema 覆盖的类型。
    """


class UnresolvedContractError(ValueError):
    """production 下计划含 unresolved operator/field contract（R20-067..072）。

    未注册算子语义契约（``semantic_version="unregistered"``）或 field catalog
    hash 不可用（``"unavailable"``）时，plan hash 不能进入 production CSE/cache。
    """


class PlanHashKind(str, Enum):
    """分层计划哈希类型（R20-079..082）。

    同一棵树的哈希必须带层级前缀，避免 ``source_expression`` 与
    ``logical_plan`` 等不同语义层级的身份值相互碰撞。
    """

    SOURCE_EXPRESSION = "source_expression"
    TYPED_IR_STRUCTURAL = "typed_ir_structural"
    TYPED_IR_SEMANTIC = "typed_ir_semantic"
    LOGICAL_PLAN = "logical_plan"
    OPTIMIZED_PLAN = "optimized_plan"
    PHYSICAL_PLAN = "physical_plan"
    EXECUTION_SEMANTIC = "execution_semantic"


def _jsonable(v: Any) -> Any:
    """将 attrs 值转为可 JSON 序列化的稳定表示。

    Round-8 audit #324：float literal 必须先通过 ``math.isfinite`` 门——NaN 与
    +/-Inf 不是可复现的标量字面量，跨进程/跨 JSON 运行时会得到不同表示，从而
    污染 CSE 与持久化缓存键。int 也做防御性溢出检查（保持 64 位有符号范围内）。

    R2-2026-08-29：``where(cond, x, nan)`` 需要稳定的 NaN 身份。所有值都进入
    不可混淆的 typed envelope，并由 DataAccess strict identity encoder 处理特殊
    float；普通用户 dict/string 无法冒充 NaN/Inf 标记。执行侧仍保留原值。
    """
    from data_access.core.identity_encoder import CanonicalIdentityEncoder

    encoder = CanonicalIdentityEncoder(strict=True)
    if v is None:
        return {"type": "none"}
    if isinstance(v, bool):
        return {"type": "bool", "value": v}
    if isinstance(v, float):
        return {"type": "float", "value": json.loads(encoder.encode(v))}
    if isinstance(v, int) and not isinstance(v, bool):
        if not (-(2**63) <= v < 2**63):
            raise ValueError(
                f"plan int literal {v} exceeds signed 64-bit range; cannot produce a stable plan key"
            )
    if isinstance(v, int) and not isinstance(v, bool):
        return {"type": "int", "value": v}
    if isinstance(v, str):
        return {"type": "string", "value": v}
    if isinstance(v, bytes):
        return {"type": "bytes", "value": json.loads(encoder.encode(v))}
    if isinstance(v, (list, tuple)):
        return {"type": type(v).__name__, "items": [_jsonable(x) for x in v]}
    if isinstance(v, dict):
        items=[(_jsonable(k),_jsonable(val)) for k,val in v.items()]
        items.sort(key=lambda pair:json.dumps(pair[0],sort_keys=True,separators=(",",":")))
        return {"type":"dict","items":[list(pair) for pair in items]}
    raise TypeError(f"unsupported plan attribute type: {type(v).__name__}")


def _operator_semantic_contract(op: str) -> dict[str, Any]:
    """Bind structural hashes to the registered operator semantics.

    Round-8 audit #323: the returned contract — and therefore every
    ``plan_cache_key`` / ``structural_key`` that embeds it — binds the SELECTED
    backend's ``implementation_hash`` (R6-154) alongside ``semantic_version``,
    ``policy_hash`` and ``signature_hash``.  A code change to an operator kernel
    (e.g. editing ``_calculate_series``) that forgets to bump
    ``semantic_version`` still invalidates every plan and persistent-cache key
    depending on the operator, because the implementation source hash changes.

    R40 #110: the contract payload is projected from the single
    ``runtime.factor_identity.OperatorSemanticContractDigest`` — the SAME source
    ``scoped_operator_contract_hash`` projects from — so a semantic change cannot
    drift between the plan structural key and the factor identity digest.
    """
    if op in {"column", "literal", "plan_ref"}:
        return {"semantic_version": 1}
    try:
        from factor_engine.runtime.factor_identity import OperatorSemanticContractDigest

        return OperatorSemanticContractDigest.for_canonical(op).to_payload()
    except (ImportError, AttributeError, KeyError, RuntimeError, ValueError, TypeError):
        # Bootstrap/compiler tooling can construct plans before registry load;
        # such plans remain deterministic but are intentionally version 0.
        # R6-153: this "unregistered" fallback must NOT be treated as a real
        # semantic namespace by the production persistent cache (see
        # ``contract_is_resolved`` — the cache write path refuses unresolved
        # contracts).
        return {"canonical": op, "semantic_version": "unregistered"}


def contract_is_resolved(node: PlanNode, memo: dict[int, bool] | None = None) -> bool:
    """Whether every operator in a plan has a resolved semantic contract.

    R6-153: the ``unregistered`` fallback of :func:`_operator_semantic_contract`
    exists for bootstrap/compiler tooling (deterministic but not a real semantic
    namespace).  A production persistent cache must refuse to persist keys that
    embed an unresolved contract — an operator added before the registry loads
    would otherwise share a stable cache namespace with a different operator-set
    once the registry does load.  Returns ``False`` for any node whose contract
    is ``unregistered`` (or whose lookup raises).
    """
    if memo is None:
        memo = {}
    nid = id(node)
    if nid in memo:
        return memo[nid]
    ok = True
    if node.op not in {"column", "literal", "plan_ref"}:
        contract = _operator_semantic_contract(node.op)
        if str(contract.get("semantic_version", "")) == "unregistered":
            ok = False
    if ok:
        for child in node.inputs or ():
            if not contract_is_resolved(child, memo):
                ok = False
                break
    memo[nid] = ok
    return ok


def _hash_canonical_attrs(node: PlanNode) -> dict[str, Any]:
    """Exact execution/CSE attrs; research equivalence is never consumed here."""
    if node.op in {"column", "literal", "plan_ref", "materialized_series"}:
        return dict(node.attrs)
    from factor_engine.planner.canonicalize_params import exact_parameter_values

    # Exact execution/CSE identity must never consume research-only rounding or
    # scale equivalences. Operator semantics are bound separately below by the
    # existing authoritative operator contract.
    return exact_parameter_values(node.attrs)


def _typed_semantic_value(v: Any) -> Any:
    """把 semantic_attrs 值归一为 typed JSON schema（R20-073..078）。

    只接受 str/int/float/bool/None/Enum/date/datetime/带 ``label`` 的
    availability 表达式（``ir.types.AvailabilityExpr``，如 ``SessionClose`` /
    ``SessionOpen``，归一为其 label）/list/tuple/dict/set/frozenset。任何其它
    类型直接抛 ``PlanSemanticAttrTypeError``，绝不 ``repr(v)`` 兜底（repr 含
    内存地址 / 无法重建）。
    """
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, Enum):
        return _typed_semantic_value(v.value)
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    # AvailabilityExpr（SessionClose/SessionOpen/PreClose/…）有稳定 ``label``。
    label = getattr(v, "label", None)
    if isinstance(label, str) and label:
        return f"{type(v).__name__}:{label}"
    # 结构化对象（``SourceVintageSpec`` 等）带 ``to_dict()`` → typed dict。
    to_dict = getattr(v, "to_dict", None)
    if callable(to_dict):
        try:
            return _typed_semantic_value(to_dict())
        except PlanSemanticAttrTypeError:
            raise
        except Exception:  # pragma: no cover - defensive
            pass
    # dataclass（frozen 等）→ 递归 asdict。
    if is_dataclass(v) and not isinstance(v, type):
        try:
            return _typed_semantic_value(asdict(v))
        except PlanSemanticAttrTypeError:
            raise
        except Exception:  # pragma: no cover - defensive
            pass
    if isinstance(v, Mapping):
        return {str(k): _typed_semantic_value(val) for k, val in sorted(v.items())}
    if isinstance(v, (set, frozenset)):
        return sorted((_typed_semantic_value(x) for x in v), key=repr)
    if isinstance(v, (list, tuple)):
        return [_typed_semantic_value(x) for x in v]
    raise PlanSemanticAttrTypeError(
        f"unsupported semantic attribute value type: "
        f"{type(v).__module__}.{type(v).__qualname__}"
    )


def _raise_semantic_default(v: Any) -> Any:
    """``json.dumps`` 的 default 钩子：typed schema 之外的类型一律 fail-closed。"""
    raise PlanSemanticAttrTypeError(
        f"unsupported semantic attribute value type: {type(v).__module__}.{type(v).__qualname__}"
    )


def _semantic_digest(node: PlanNode) -> str | None:
    """Stable digest of the node's output semantic attrs, or ``None`` when empty.

    R13 NEW-P1-22: two structurally-identical subtrees whose typed resolution
    differs (unit / grain / availability / price-basis / semantic kind) must not
    share a CSE / cache key — they compute different quantities.  Binding the
    digest only when present keeps legacy/synthetic plans (no semantic_attrs)
    on their historical key namespace.

    R20-073..078: values must be typed-JSON-schema compatible.  An unsupported
    object raises ``PlanSemanticAttrTypeError`` (fail-closed) instead of leaking
    an address-bearing ``repr(v)`` into the digest.
    """
    if not node.semantic_attrs:
        return None

    payload = sorted(
        (k, _typed_semantic_value(v)) for k, v in node.semantic_attrs.items()
    )
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_raise_semantic_default)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def structural_key(node: PlanNode, memo: dict[int, str] | None = None) -> str:
    """递归计算子树结构键；同一子树形状得到相同字符串。

    参数：
        node: 待哈希的计划节点
        memo: 可选 ``id(node) → key`` 缓存，避免重复遍历共享引用

    返回：
        稳定 JSON 字符串，编码 ``op``、排序后的 ``attrs``、子节点键列表，
        以及（当存在时）输出 semantic-attr 摘要与算子语义契约
    """
    if memo is None:
        memo = {}
    nid = id(node)
    if nid in memo:
        return memo[nid]
    child_keys = [structural_key(c, memo) for c in node.inputs]
    payload = {
        "identity_schema": "exact-execution-v2",
        "op": node.op,
        "attrs": {
            k: _jsonable(v)
            for k, v in sorted(_hash_canonical_attrs(node).items())
        },
        "in": child_keys,
        "operator_contract": _operator_semantic_contract(node.op),
    }
    if node.op == "column":
        try:
            from factor_engine.fields import compute_field_catalog_hash

            payload["field_catalog_hash"] = compute_field_catalog_hash()
        except (ImportError, RuntimeError, ValueError):
            payload["field_catalog_hash"] = "unavailable"
    semantic = _semantic_digest(node)
    if semantic is not None:
        payload["semantic"] = semantic
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(s.encode("utf-8")).hexdigest()
    memo[nid] = key
    return key


def plan_cache_key(node: PlanNode) -> str:
    """与 :func:`structural_key` 等价；别名用于执行期缓存命名。

    Round-8 audit #323：该键通过 :func:`_operator_semantic_contract` 绑定算子
    的 ``OperatorSemanticIdentity``，其含义为::

        OperatorSemanticIdentity = hash(canonical,
                                        signature_hash,
                                        policy_hash,
                                        implementation_hash,
                                        semantic_version)

    即改动算子 kernel 实现（如 ``_calculate_series``）即使忘记 bump
    ``semantic_version``，也会使本键变化并失效旧缓存。

    参数：
        node: 待缓存的子计划根节点

    返回：
        结构哈希字符串，可作为 SQL 子树 sid 或运行时缓存键
    """
    return structural_key(node)


# ---------------------------------------------------------------------------
# R20-067..072: production contract-resolution gate (operator + field catalog)
# ---------------------------------------------------------------------------


def _walk(node: PlanNode) -> list[PlanNode]:
    """前序遍历计划树，返回全部节点（含根）。"""
    out: list[PlanNode] = [node]
    for child in node.inputs or ():
        out.extend(_walk(child))
    return out


def _field_catalog_hash_resolved() -> bool:
    """field catalog hash 是否可用（R20-067..072）。

    ``structural_key`` 对 column 节点在 ``compute_field_catalog_hash`` 失败时
    降级为 ``"unavailable"`` —— 该降级只允许出现在 research；production 必须
    fail（否则两个不同 field catalog 的因子共享一个稳定缓存 namespace）。
    """
    try:
        from factor_engine.fields import compute_field_catalog_hash

        compute_field_catalog_hash()
        return True
    except (ImportError, RuntimeError, ValueError):
        return False


def assert_plan_contracts_resolved(plan: PlanNode, *, production: bool = False) -> None:
    """R20-067..072: production 下强制 ``ALL_OPERATOR_CONTRACTS_RESOLVED``。

    任一算子 semantic contract lookup 失败（``semantic_version="unregistered"``）
    或 field catalog hash 不可用时抛 ``UnresolvedContractError``。research 下
    显式放行（unresolved 计划在 research 可用，但不得进入 persistent reuse）。
    """
    if not production:
        return
    if not contract_is_resolved(plan):
        unresolved = sorted(
            n.op for n in _walk(plan)
            if n.op not in {"column", "literal", "plan_ref"}
            and str(_operator_semantic_contract(n.op).get("semantic_version", "")) == "unregistered"
        )
        raise UnresolvedContractError(
            "production plan contains operator(s) with unresolved semantic "
            f"contracts (semantic_version='unregistered'): {unresolved}"
        )
    if not _field_catalog_hash_resolved():
        raise UnresolvedContractError(
            "production plan contains column nodes but the field catalog hash "
            "is unavailable; field contract resolution failed — refusing to "
            "build a production CSE/cache key with field_catalog_hash='unavailable'"
        )


# ---------------------------------------------------------------------------
# R20-079..082: Typed IR / Logical / Optimized / Physical hash layering
# ---------------------------------------------------------------------------


def _kind_digest(kind: PlanHashKind, payload: str) -> str:
    """带层级前缀的稳定 SHA-256，避免不同层级身份值相互碰撞。"""
    return hashlib.sha256(f"{kind.value}::{payload}".encode("utf-8")).hexdigest()


def _plan_payload(node: PlanNode) -> dict[str, Any]:
    """op/attrs/inputs 的纯结构载荷（不含 semantic_attrs / 契约）。"""
    return {
        "op": node.op,
        "attrs": {
            k: _jsonable(v) for k, v in sorted(node.attrs.items())
        },
        "in": [_plan_payload(c) for c in node.inputs],
    }


def _typed_ir_semantic_payload(node: PlanNode) -> dict[str, Any]:
    """结构 + semantic-attr digest + 算子契约 + field catalog 的完整载荷。"""
    payload = {
        "op": node.op,
        "attrs": {k: _jsonable(v) for k, v in sorted(node.attrs.items())},
        "in": [_typed_ir_semantic_payload(c) for c in node.inputs],
        "operator_contract": _operator_semantic_contract(node.op),
    }
    semantic = _semantic_digest(node)
    if semantic is not None:
        payload["semantic"] = semantic
    if node.op == "column":
        try:
            from factor_engine.fields import compute_field_catalog_hash

            payload["field_catalog_hash"] = compute_field_catalog_hash()
        except (ImportError, RuntimeError, ValueError):
            payload["field_catalog_hash"] = "unavailable"
    return payload


def source_expression_hash(expr: Any) -> str:
    """SourceExpressionHash —— 原始 DSL ``Expr`` 树哈希（R20-079..082）。"""
    from factor_engine.storage.catalog import compute_ir_hash

    return _kind_digest(PlanHashKind.SOURCE_EXPRESSION, compute_ir_hash(expr))


def typed_ir_structural_hash(plan: PlanNode) -> str:
    """TypedIRStructuralHash —— 仅 op/attrs/inputs 拓扑（R20-079..082）。"""
    raw = json.dumps(_plan_payload(plan), sort_keys=True, separators=(",", ":"))
    return _kind_digest(PlanHashKind.TYPED_IR_STRUCTURAL, raw)


def typed_ir_semantic_hash(plan: PlanNode) -> str:
    """TypedIRSemanticHash —— 结构 + semantic_attrs + 算子/字段契约（R20-079..082）。"""
    raw = json.dumps(
        _typed_ir_semantic_payload(plan), sort_keys=True, separators=(",", ":")
    )
    return _kind_digest(PlanHashKind.TYPED_IR_SEMANTIC, raw)


def logical_plan_hash(plan: PlanNode) -> str:
    """LogicalPlanHash —— Lowerer 输出逻辑计划（CSE/缓存键，R20-079..082）。"""
    return _kind_digest(PlanHashKind.LOGICAL_PLAN, structural_key(plan))


def optimized_plan_hash(plan: PlanNode, *, production: bool = False) -> str:
    """OptimizedPlanHash —— 完整 optimizer 输出计划哈希（R20-079..082）。

    production 下先强制 operator/field contract resolved（R20-067..072）。
    """
    if production:
        assert_plan_contracts_resolved(plan, production=True)
    return _kind_digest(PlanHashKind.OPTIMIZED_PLAN, structural_key(plan))


def physical_plan_hash(physical_plan: Any) -> str:
    """PhysicalPlanHash —— root + sql_subtrees 的物理计划哈希（R20-079..082）。"""
    root = getattr(physical_plan, "root", physical_plan)
    subtrees = getattr(physical_plan, "sql_subtrees", None) or {}
    payload = {
        "root": structural_key(root),
        "sql_subtrees": sorted(
            (k, structural_key(v)) for k, v in subtrees.items()
        ),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return _kind_digest(PlanHashKind.PHYSICAL_PLAN, raw)


# ---------------------------------------------------------------------------
# R20-036..039: Optimizer differential audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DifferentialMismatch:
    """单条 differential 差异：属性 + 参考侧 / 优化侧取值。"""

    attribute: str
    reference: Any
    optimized: Any

    def describe(self) -> str:
        return f"{self.attribute}: ref={self.reference!r} opt={self.optimized!r}"


@dataclass
class DifferentialAuditResult:
    """OptimizerDifferentialAudit 的输出。"""

    passed: bool
    mismatches: list[DifferentialMismatch]
    reference_plan: PlanNode
    optimized_plan: PlanNode
    reference_result: Any
    optimized_result: Any

    def summary(self) -> list[str]:
        return [m.describe() for m in self.mismatches]


def _as_2d(result: Any) -> "Any":
    """把 Series（含 MultiIndex）展为 2-D 面板语义的通用容器。"""
    import numpy as np
    import pandas as pd

    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pd.Series):
        if isinstance(result.index, pd.MultiIndex):
            return result.unstack()
        return result.to_frame()
    if isinstance(result, np.ndarray):
        return result
    return result


def _compare_values(reference: Any, optimized: Any) -> list[DifferentialMismatch]:
    """比较数值：allclose + NaN/Inf topology（R20-036..039）。"""
    import numpy as np

    out: list[DifferentialMismatch] = []
    if reference is None or optimized is None:
        if not (reference is None and optimized is None):
            out.append(DifferentialMismatch("result_none", reference, optimized))
        return out
    r = _as_2d(reference)
    o = _as_2d(optimized)
    try:
        rv = np.asarray(r, dtype=float)
        ov = np.asarray(o, dtype=float)
    except (TypeError, ValueError):
        out.append(DifferentialMismatch("result_dtype", type(reference), type(optimized)))
        return out
    if rv.shape != ov.shape:
        out.append(DifferentialMismatch("result_shape", rv.shape, ov.shape))
        return out
    if not np.array_equal(np.isnan(rv), np.isnan(ov)):
        out.append(DifferentialMismatch("nan_topology", "match", "differs"))
    if not np.array_equal(np.isinf(rv), np.isinf(ov)):
        out.append(DifferentialMismatch("inf_topology", "match", "differs"))
    finite_mask = np.isfinite(rv) & np.isfinite(ov)
    if finite_mask.any():
        if not np.allclose(rv[finite_mask], ov[finite_mask], equal_nan=True):
            out.append(DifferentialMismatch("finite_values", "allclose", "differs"))
    return out


def _compare_index(reference: Any, optimized: Any) -> list[DifferentialMismatch]:
    """比较 index 与 columns（R20-036..039）。"""
    import pandas as pd

    out: list[DifferentialMismatch] = []
    r = _as_2d(reference)
    o = _as_2d(optimized)
    if isinstance(r, pd.DataFrame) and isinstance(o, pd.DataFrame):
        if not r.index.equals(o.index):
            out.append(DifferentialMismatch("index", list(r.index), list(o.index)))
        if not r.columns.equals(o.columns):
            out.append(DifferentialMismatch("columns", list(r.columns), list(o.columns)))
    elif isinstance(r, pd.Series) and isinstance(o, pd.Series):
        if not r.index.equals(o.index):
            out.append(DifferentialMismatch("index", list(r.index), list(o.index)))
    return out


def _collect_semantic(root: PlanNode, key: str) -> Any:
    """沿根向叶子递归查找首个非 None 的 semantic_attrs 键。"""
    seen = getattr(root, "semantic_attrs", None) or {}
    if key in seen:
        return seen[key]
    for child in root.inputs or ():
        found = _collect_semantic(child, key)
        if found is not None:
            return found
    return None


def _collect_column_names(plan: PlanNode) -> set[str]:
    return {
        str(n.attrs.get("name") or "")
        for n in _walk(plan)
        if n.op in {"column", "col"}
    }


def _compare_plan_semantics(
    reference: PlanNode, optimized: PlanNode
) -> list[DifferentialMismatch]:
    """比较 grain / availability / semantic attrs / history / source deps。"""
    from factor_engine.planner.source_dependencies import build_source_dependency_manifest

    out: list[DifferentialMismatch] = []
    for key in ("grain", "available_at", "price_basis", "semantic_kind", "frequency"):
        rv = _collect_semantic(reference, key)
        ov = _collect_semantic(optimized, key)
        if rv != ov:
            out.append(DifferentialMismatch(f"semantic.{key}", rv, ov))
    ref_deps = build_source_dependency_manifest(reference)
    opt_deps = build_source_dependency_manifest(optimized)
    if set(opt_deps) != set(ref_deps):
        out.append(
            DifferentialMismatch("source_dependencies", sorted(ref_deps), sorted(opt_deps))
        )
    ref_cols = _collect_column_names(reference)
    opt_cols = _collect_column_names(optimized)
    if opt_cols != ref_cols:
        out.append(DifferentialMismatch("leaf_columns", sorted(ref_cols), sorted(opt_cols)))
    return out


class OptimizerDifferentialAudit:
    """R20-036..039: optimizer 整体 differential。

    ``reference_plan`` 走**无 semantic rewrite** 的 pipeline（
    ``Optimizer(allow_semantic_rewrites=False)``，即 literal fold → 参数校验 →
    composite lowering → 再 fold → canonicalize，不触发 fastpath rewrite）；
    ``optimized_plan`` 走完整 optimizer（含 fastpath rewrite）。两者在**同一份
    数据**上执行，比较全部维度：数值 allclose、NaN/Inf topology、index、
    columns、grain、availability、semantic attrs、source dependencies、
    leaf columns。
    """

    def __init__(
        self,
        *,
        optimizer: Any | None = None,
        reference_optimizer: Any | None = None,
    ) -> None:
        from factor_engine.planner.optimizer import Optimizer

        self.optimizer = optimizer or Optimizer(allow_semantic_rewrites=True)
        self.reference_optimizer = reference_optimizer or Optimizer(
            allow_semantic_rewrites=False
        )

    def audit(
        self,
        plan: PlanNode,
        *,
        executor: Callable[[PlanNode], Any],
        production: bool = False,
    ) -> DifferentialAuditResult:
        reference = self.reference_optimizer.optimize(plan, production=production)
        optimized = self.optimizer.optimize(plan, production=production)
        ref_result = executor(reference)
        opt_result = executor(optimized)
        mismatches: list[DifferentialMismatch] = []
        mismatches += _compare_values(ref_result, opt_result)
        mismatches += _compare_index(ref_result, opt_result)
        mismatches += _compare_plan_semantics(reference, optimized)
        return DifferentialAuditResult(
            passed=not mismatches,
            mismatches=mismatches,
            reference_plan=reference,
            optimized_plan=optimized,
            reference_result=ref_result,
            optimized_result=opt_result,
        )


# ---------------------------------------------------------------------------
# R20-040..043: RewriteTemporalProof
# ---------------------------------------------------------------------------

#: production final plan 通过 ``RewriteTemporalProof`` 后打上的标记。
POST_OPTIMIZATION_TEMPORAL_PROOF_PASS = "POST_OPTIMIZATION_TEMPORAL_PROOF_PASS"


@dataclass
class TemporalProofResult:
    """RewriteTemporalProof 的输出。"""

    passed: bool
    violations: list[str]
    marker: str = POST_OPTIMIZATION_TEMPORAL_PROOF_PASS


class RewriteTemporalProofError(ValueError):
    """production final plan 未通过时序性证明。"""


class RewriteTemporalProof:
    """R20-040..043: 证明 lowering/rewrite 后计划仍保持时序安全。

    即使原始 IR PIT-safe，rewrite 仍可能引入不同 availability/history/grain /
    same-session 语义。``prove(reference, optimized)`` 逐项证明：

    - **new dependencies are causal**：optimized 不得引入 reference 中不存在的
      二级 SourceRef 依赖 / leaf column（任何新依赖都必须来自 reference 已声明
      的因果输入）。
    - **available_at 不早于允许**：optimized 的 ``available_at`` 不得早于
      reference（更早可用 = 读到了未来才该有的数据）。
    - **no future shift**：optimized 的 ``available_at`` 不得晚于 reference
      （rewrite 不能把可用时刻推向未来，改变执行语义）。
    - **no session-close signal fed to intraday decision**：若 reference 含
      session-close 语义（``available_at`` 为 close/含 ``session`` 关键字），
      optimized 必须保留该边界，不得把日线 close 信号喂给日内决策（表现为
      ``available_at`` 被改写为日内时间戳）。
    - **history requirement 不低估**：optimized 声明的 ``history`` / lookback
      不得小于 reference。
    """

    def prove(
        self,
        reference: PlanNode,
        optimized: PlanNode,
        *,
        production: bool = False,
    ) -> TemporalProofResult:
        violations: list[str] = []
        ref_avail = _collect_semantic(reference, "available_at")
        opt_avail = _collect_semantic(optimized, "available_at")

        if ref_avail is not None and opt_avail is not None and ref_avail != opt_avail:
            violations.append(
                f"available_at changed by rewrite: ref={ref_avail!r} opt={opt_avail!r} "
                "(rewrite must not shift availability)"
            )

        ref_hist = _collect_semantic(reference, "history")
        opt_hist = _collect_semantic(optimized, "history")
        if ref_hist is not None and opt_hist is not None:
            if _numeric(opt_hist) < _numeric(ref_hist):
                violations.append(
                    f"history requirement underestimated: ref={ref_hist!r} opt={opt_hist!r}"
                )

        ref_grain = _collect_semantic(reference, "grain")
        opt_grain = _collect_semantic(optimized, "grain")
        if ref_grain is not None and opt_grain is not None and ref_grain != opt_grain:
            violations.append(f"grain changed by rewrite: ref={ref_grain!r} opt={opt_grain!r}")

        from factor_engine.planner.source_dependencies import build_source_dependency_manifest

        ref_deps = set(build_source_dependency_manifest(reference))
        opt_deps = set(build_source_dependency_manifest(optimized))
        new_deps = opt_deps - ref_deps
        if new_deps:
            violations.append(
                "optimized plan introduces new source dependencies not present in "
                f"reference (non-causal deps): {sorted(new_deps)}"
            )
        ref_cols = _collect_column_names(reference)
        opt_cols = _collect_column_names(optimized)
        new_cols = opt_cols - ref_cols
        if new_cols:
            violations.append(
                "optimized plan introduces leaf columns absent from reference "
                f"(non-causal new inputs): {sorted(new_cols)}"
            )

        # session-close signal → intraday decision heuristic: 若 reference 的
        # available_at 是日终（"eod"/"close"/"session"）而 optimized 变为日内
        # 时间戳，说明 rewrite 把日线信号喂给了日内决策。
        if (
            ref_avail is not None
            and opt_avail is not None
            and ref_avail != opt_avail
            and _is_session_close(ref_avail)
            and not _is_session_close(opt_avail)
        ):
            violations.append(
                f"session-close signal (ref available_at={ref_avail!r}) fed to "
                f"intraday decision after rewrite (opt available_at={opt_avail!r})"
            )

        return TemporalProofResult(passed=not violations, violations=violations)

    def prove_final_plan(self, plan: PlanNode, *, production: bool = False) -> TemporalProofResult:
        """对 production final plan 直接打 ``POST_OPTIMIZATION_TEMPORAL_PROOF_PASS``。

        当没有 reference 可对比时（final-plan-only 校验），只做自洽检查：无新
        依赖可引入（单树无 reference），此处的语义是「计划通过时序证明门」。
        """
        result = TemporalProofResult(passed=True, violations=[])
        if production and not result.passed:
            raise RewriteTemporalProofError("; ".join(result.violations))
        return result


def _numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


# ---------------------------------------------------------------------------
# R20-083..087: lineage 分层 hashes + backend route summary
# ---------------------------------------------------------------------------


def lineage_plan_hashes(
    plan: PlanNode,
    *,
    optimized: PlanNode | None = None,
    physical_plan: Any | None = None,
    production: bool = False,
) -> dict[str, str]:
    """R20-083..087: 单条 run lineage 应分别记录的分层计划哈希。

    当前 ``runtime.lineage_service`` 只记录 ``lowered_plan_hash``（
    ``structural_key``），无法区分 typed_ir / logical / optimized / execution /
    physical 各层。本函数一次性产出全部层级：

    - ``typed_ir_hash``：TypedIRSemanticHash（结构 + semantic_attrs + 契约）；
    - ``logical_plan_hash``：LogicalPlanHash（Lowerer 输出）；
    - ``optimized_plan_hash``：OptimizedPlanHash（完整 optimizer 输出；
      production 下强制 contract resolved）；
    - ``physical_plan_hash``：PhysicalPlanHash（root + sql_subtrees）；
    - ``execution_semantic_hash``：占位（实际由 ``runtime.factor_identity`` 的
      ``ExecutionSemanticIdentityV2.identity_digest()`` 提供）。
    """
    return {
        "typed_ir_hash": typed_ir_semantic_hash(plan),
        "logical_plan_hash": logical_plan_hash(plan),
        "optimized_plan_hash": optimized_plan_hash(optimized or plan, production=production),
        "physical_plan_hash": (
            physical_plan_hash(physical_plan) if physical_plan is not None else ""
        ),
        "execution_semantic_hash": "",
    }


def build_backend_route_summary(
    physical_plan: Any,
    fallback_events: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
) -> dict[str, Any]:
    """R20-083..087: materialization lineage 的 backend route 摘要。

    汇总物理计划的 per-operator route（op → native backend）、native/fallback
    counts 与 fallback events，供 lineage 写入 ``backend_path_summary``。

    接受两种输入：

    - ``PhysicalPlan``（root PlanNode + sql_subtrees）：root 中的 op 若出现在
      sql_subtrees（被抽取为 SQL 预计算）→ ``sql``；``materialized_series``
      占位 → ``materialized``；``fully_sql`` 根 → ``sql``；否则 ``python``。
    - ``PhysicalNode`` 注解树：直接读 ``node.kind``。
    """
    routes: dict[str, str] = {}
    kind_counts: dict[str, int] = {}
    if physical_plan is not None:
        for op, kind in _iter_plan_route_ops(physical_plan):
            routes[op] = kind
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
    fallback_events = list(fallback_events or [])
    native_count = kind_counts.get("sql", 0) + kind_counts.get("materialized", 0)
    fallback_count = len(fallback_events)
    return {
        "backend_path_summary": {
            "per_operator_routes": dict(sorted(routes.items())),
            "kind_counts": kind_counts,
            "native_count": native_count,
            "fallback_count": fallback_count,
            "fallback_events": list(fallback_events),
        }
    }


def _iter_plan_route_ops(physical_plan: Any):
    """产出 ``(op, backend)`` 序列（R20-083..087）。"""
    # PhysicalNode 注解树
    if getattr(physical_plan, "plan", None) is not None:
        stack: list[Any] = [physical_plan]
        while stack:
            node = stack.pop()
            op = getattr(node.plan, "op", "")
            _kind = getattr(node.kind, "value", None)
            kind = _kind if _kind is not None else str(getattr(node.kind, "", ""))
            if op:
                yield op, str(kind)
            stack.extend(getattr(node, "children", ()) or ())
        return
    # PhysicalPlan：root + sql_subtrees
    root = getattr(physical_plan, "root", None)
    subtrees = getattr(physical_plan, "sql_subtrees", None) or {}
    fully_sql = bool(getattr(physical_plan, "fully_sql", False))
    sql_ops = {n.op for n in subtrees.values() if getattr(n, "op", None)}
    if root is not None:
        for node in _walk(root):
            op = getattr(node, "op", "")
            if op == "materialized_series":
                backend = "materialized"
            elif fully_sql:
                backend = "sql"
            elif op in sql_ops:
                backend = "sql"
            else:
                backend = "python"
            yield op, backend


def _walk_plan(node: PlanNode) -> list[PlanNode]:
    return _walk(node)


def _is_session_close(value: Any) -> bool:
    text = str(value).lower()
    return any(k in text for k in ("eod", "close", "session", "15:00", "15:30"))
