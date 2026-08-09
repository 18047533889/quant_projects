"""逻辑计划子树的结构化哈希（与具体列数据无关），供 CSE 与执行期子树缓存键复用。"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from planner.logical_plan import PlanNode


class NonFiniteLiteralError(ValueError):
    """计划中出现了非有限 literal（NaN / +/-Inf），无法生成稳定缓存键。"""


def _jsonable(v: Any) -> Any:
    """将 attrs 值转为可 JSON 序列化的稳定表示。

    Round-8 audit #324：float literal 必须先通过 ``math.isfinite`` 门——NaN 与
    +/-Inf 不是可复现的标量字面量，跨进程/跨 JSON 运行时会得到不同表示，从而
    污染 CSE 与持久化缓存键。int 也做防御性溢出检查（保持 64 位有符号范围内）。
    """
    if isinstance(v, float):
        if not math.isfinite(v):
            raise NonFiniteLiteralError(
                f"plan literal {v!r} is not a finite float; NaN/Inf are forbidden in plan keys"
            )
    if isinstance(v, int) and not isinstance(v, bool):
        if not (-(2**63) <= v < 2**63):
            raise ValueError(
                f"plan int literal {v} exceeds signed 64-bit range; cannot produce a stable plan key"
            )
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(val) for k, val in sorted(v.items())}
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
    """
    if op in {"column", "literal", "plan_ref"}:
        return {"semantic_version": 1}
    try:
        from cleaned_operators.operator_policy import infer_operator_policy
        from cleaned_operators.registry import OperatorRegistry
        from backend.evidence_provenance import compute_payload_hash
        from backend.production_signature import signature_for

        canonical = OperatorRegistry.resolve_canonical(op)
        catalog = OperatorRegistry._catalog.get(canonical, {})
        signature = signature_for(canonical)
        signature_payload = None
        if signature is not None:
            signature_payload = {
                "default_status": signature.default_status,
                "params": [(p.name, p.constraint, p.status) for p in signature.params],
            }
        # R6-154: bind the SELECTED backend's implementation source hash directly,
        # not just the manually-bumped semantic_version.  A code change that
        # forgets to bump ``semantic_version`` now still invalidates every plan
        # and persistent-cache key that depends on the operator.
        impl_hash: dict[str, str] = {}
        from backend.evidence_provenance import implementation_hashes_for

        try:
            impl_hash = implementation_hashes_for(canonical)
        except (ImportError, AttributeError, KeyError, RuntimeError, ValueError, TypeError):
            impl_hash = {}
        return {
            "canonical": canonical,
            "semantic_version": str(catalog.get("semantic_version") or "1.0"),
            "policy_hash": compute_payload_hash(
                infer_operator_policy(canonical, canonical=canonical).to_dict()
            ),
            "signature_hash": compute_payload_hash(signature_payload),
            "implementation_hash": compute_payload_hash(impl_hash),
        }
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
    """Hash-side parameter canonicalization (R13 NEW-P0-17/18).

    Only the HASH payload canonicalizes values (float-noise rounding + declared
    ``equivalence="positive_scale"`` weight normalization).  Execution plans keep
    user values verbatim — see ``planner.canonicalize_params``.  Literals/columns
    are never touched: an explicit constant must hash by its true value.
    """
    if node.op in {"column", "literal", "plan_ref", "materialized_series"}:
        return dict(node.attrs)
    from planner.canonicalize_params import canonicalize_parameter_values

    try:
        from cleaned_operators.registry import OperatorRegistry

        canonical = OperatorRegistry._aliases.get(str(node.op), str(node.op))
    except Exception:  # pragma: no cover - bootstrap
        canonical = str(node.op)
    return canonicalize_parameter_values(node.attrs, canonical=canonical)


def _semantic_digest(node: PlanNode) -> str | None:
    """Stable digest of the node's output semantic attrs, or ``None`` when empty.

    R13 NEW-P1-22: two structurally-identical subtrees whose typed resolution
    differs (unit / grain / availability / price-basis / semantic kind) must not
    share a CSE / cache key — they compute different quantities.  Binding the
    digest only when present keeps legacy/synthetic plans (no semantic_attrs)
    on their historical key namespace.
    """
    if not node.semantic_attrs:
        return None

    def _norm(v: Any) -> Any:
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if isinstance(v, (set, frozenset)):
            return sorted((_norm(x) for x in v), key=repr)
        if isinstance(v, (list, tuple)):
            return [_norm(x) for x in v]
        if isinstance(v, dict):
            return {str(k): _norm(val) for k, val in sorted(v.items())}
        return repr(v)

    payload = sorted((k, _norm(v)) for k, v in node.semantic_attrs.items())
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr)
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
            from fields import compute_field_catalog_hash

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
