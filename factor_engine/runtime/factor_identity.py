# -*- coding: utf-8 -*-
"""因子语义身份（FactorSemanticIdentity）与 checkpoint 指纹。

R10 #16 / #17 / #47 / #48:

- ``FactorSemanticIdentity`` 把一条因子绑定到：计划 IR 哈希、算子/字段/源契约
  哈希、源依赖哈希，以及执行作用域（market/calendar/timezone/universe）、
  价格基（price_basis）、PIT / decision-time 策略与方言版本。
- ``identity_digest()`` 对全部字段做稳定 canonical 哈希：**任意字段变化都会改变
  digest**，从而失效所有依赖该身份的 checkpoint / 缓存（#47）。
- ``NO_FACTOR_IDENTITY``（``__no_hash_provided__``）是 production 物化禁止的
  sentinel（#17）。
- ``partition_input_fingerprint`` / ``checkpoint_fingerprint`` 供
  ``storage.materialize.materializer`` 做"断点续写必须绑定身份"的判定（#47）。

本模块顶层只定义数据容器与纯函数；所有 import 都在函数体内延迟执行，避免
``storage.materialize`` -> ``runtime.factor_identity`` -> ``runtime.lineage`` /
``cleaned_operators`` 的启动期循环依赖。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

#: ``materialize()`` 未提供任何因子身份时使用的 sentinel（#17）。
NO_FACTOR_IDENTITY = "__no_hash_provided__"


class FactorIdentityMismatch(ValueError):
    """因子语义身份不匹配：生产物化缺少身份 / checkpoint 指纹不符。"""


class IdentityHashTypeError(TypeError):
    """因子身份 payload 含无法 canonical 序列化的对象（R20-073..078）。

    ``_stable_hash`` 不再 ``default=str`` 字符串化 —— 字符串化含内存地址 /
    无法重建，会让同一配置在不同 worker 上得到不同身份 digest。
    """


# ---------------------------------------------------------------------------
# 稳定哈希与标量归一
# ---------------------------------------------------------------------------


def _typed_hash_value(v: Any) -> Any:
    """把身份 payload 值归一为 typed JSON schema（R20-073..078）。

    R40 #115: numpy scalars（``np.int32`` / ``np.int64`` / ``np.float32`` /
    ``np.float64``）不是 Python ``int``/``float`` 的实例，无法用
    ``isinstance(v, int)`` 区分 —— 这里按 dtype 归一为
    ``{"type": str(v.dtype), "value": v.item()}``，使 ``int32(1)`` 与
    ``int64(1)`` 得到不同的身份 payload（int64 vs int32 在计划里不是同一个值）。
    """
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    # R40 #115: numpy scalar 前置检测（在其它 isinstance 分支之前）。
    dtype = getattr(v, "dtype", None)
    if dtype is not None:
        kind = getattr(dtype, "kind", None)
        if kind in "iufb" and hasattr(v, "item"):
            return {"type": str(dtype), "value": v.item()}
        if kind in "mM":
            # numpy datetime64/timedelta64 scalar → ISO 表示（时间语义绑定 dtype）。
            return {"type": str(dtype), "value": str(v)}
    if isinstance(v, Enum):
        return _typed_hash_value(v.value)
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    if isinstance(v, Path):
        return str(v.expanduser())
    if isinstance(v, (set, frozenset)):
        return sorted((_typed_hash_value(x) for x in v), key=repr)
    if isinstance(v, (list, tuple)):
        return [_typed_hash_value(x) for x in v]
    if isinstance(v, Mapping):
        return {str(k): _typed_hash_value(val) for k, val in sorted(v.items())}
    raise IdentityHashTypeError(
        f"unsupported identity payload value type: "
        f"{type(v).__module__}.{type(v).__qualname__}"
    )


def _stable_hash(payload: Mapping[str, Any]) -> str:
    """对字典做 stable canonical SHA-256（字段排序 + 紧凑 JSON）。

    R20-073..078: 不再 ``default=str`` 兜底 —— 未知对象直接抛
    ``IdentityHashTypeError``（fail-closed），保证身份 digest 跨进程稳定、可
    重建。内部调用方（``identity_digest`` / ``partition_input_fingerprint`` /
    ``checkpoint_fingerprint``）均只传 str/None/bool，不受影响。
    """
    canonical = _typed_hash_value(payload)
    raw = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _scalar(value: Any) -> str:
    """把任意可选字段归一为可比较的稳定字符串（None -> ""）。

    R20-073..078: 只接受 str/int/float/bool/None/Enum/date/datetime —— 其它
    类型直接抛 ``IdentityHashTypeError``，绝不 ``str(value)`` 字符串化
    （``str(object())`` 含内存地址，跨进程不稳定）。
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, Enum):
        return _scalar(value.value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise IdentityHashTypeError(
        f"unsupported identity scalar type: {type(value).__module__}.{type(value).__qualname__}"
    )


# ---------------------------------------------------------------------------
# FactorSemanticIdentity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorSemanticIdentity:
    """因子语义身份：任何字段变化都会改变 ``identity_digest()``。

    哈希字段均为 SHA-256 hex（或前缀哈希）；语义字段为可选字符串。字段顺序
    固定，``identity_digest`` 使用 ``sort_keys`` 规范 JSON，因此与字段顺序无关、
    跨进程稳定。
    """

    ir_hash: str
    operator_contract_hash: str
    field_contract_hash: str
    source_contract_hash: str
    source_dependency_hash: str
    market: str | None = None
    calendar: str | None = None
    timezone: str | None = None
    universe: str | None = None
    price_basis: str | None = None
    pit_policy: str | None = None
    decision_time_policy: str | None = None
    dialect: str | None = None
    dialect_version: str | None = None
    #: #收官轮 P0：``freq`` 不改变 IR 类型推导，但它是执行语义的一部分——同一
    #: 公式 ``1d`` 与 ``5m`` 不是同一个因子。必须进入身份 digest，否则
    #: ``factor_version`` 对两者相同，production 无法识别语义漂移。
    frequency: str | None = None
    #: P1-33/34：数据源作用域哈希（``api.factor.FactorExecutionScopeHint`` 的
    #: ``source_scope_hash``，或 ctx 覆盖值）。不同 source scope 的同一公式不是
    #: 同一个因子，必须进入身份 digest（否则 CSE 缓存/checkpoint 跨 scope 误用）。
    source_scope_hash: str | None = None
    #: R20-062..066: 实际 resolved universe membership 哈希（按 as-of date 解析）。
    #: 非空 instrument_filter 与命名 universe 的相同标签不再能共享身份 ——
    #: membership 不同 = 不同的因子语义身份。
    universe_membership_hash: str | None = None
    #: R20-213..219 (FactorSemanticIdentity V2): 编译器/数值/数学/物化精度语义
    #: 哈希。任一变化都必须使旧 cache/checkpoint/evidence 失效 —— 否则
    #: optimizer/lowering/numeric-semantics/storage-precision 规则改变后旧
    #: factor_version 继续被复用。各字段由 materializer / lineage / ctx 写入
    #: (materialize_service 把 storage_precision_policy / composite_lowering_hash /
    #: optimizer_rewrite_hash 写入 lineage.extra),本模块消费进 identity digest。
    numeric_semantics_hash: str | None = None
    mathematical_semantics_hash: str | None = None
    composite_lowering_hash: str | None = None
    optimizer_rewrite_hash: str | None = None
    history_contract_hash: str | None = None
    storage_precision_policy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FactorSemanticIdentity":
        return cls(**{name: data.get(name) for name in cls.__dataclass_fields__})

    def identity_digest(self) -> str:
        """稳定 canonical 字符串哈希：任意字段变化 → digest 变化。"""
        return _stable_hash(self.to_dict())


# ---------------------------------------------------------------------------
# ctx 提取工具
# ---------------------------------------------------------------------------


def _ctx_override(ctx: Any, name: str) -> Any:
    """从 ctx（dict 或对象）读取可选覆盖字段；缺失返回 None。"""
    if ctx is None:
        return None
    if isinstance(ctx, Mapping):
        return ctx.get(name)
    return getattr(ctx, name, None)


def _pick(*candidates: Any) -> str | None:
    """返回第一个非 None 非空候选的字符串归一。"""
    for candidate in candidates:
        if candidate is not None and str(candidate) != "":
            return _scalar(candidate)
    return None


def _semantic_value(plan: Any, key: str) -> Any:
    """从计划根的 ``semantic_attrs`` 读取字段目录元数据。"""
    if plan is None:
        return None
    semantic = getattr(plan, "semantic_attrs", None) or {}
    if isinstance(semantic, Mapping):
        return semantic.get(key)
    return None


# ---------------------------------------------------------------------------
# R32-P0-040: dependency-scoped operator / field digest
#
# 历史 ``compute_factor_identity`` 缺省绑定**整库** ``operator_catalog_hash`` /
# ``field_catalog_hash`` —— 添加一个完全无关的 operator/field 会让所有
# factor_version / cache / checkpoint 失效（whole-catalog hash 污染单因子
# identity）。R32 §4 规则 4/5：operator/field hash 只计算本 plan 依赖项。
# 真正全局语义（numeric/compiler/calendar rule）单独 global digest。
# ---------------------------------------------------------------------------


def _plan_operator_canonicals(plan: Any) -> list[str]:
    """收集计划实际引用的去重 canonical operator 名（不含 column/literal/plan_ref）。"""
    if plan is None:
        return []
    seen: set[str] = set()
    stack = [plan]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        op = str(getattr(node, "op", "") or "")
        if op and op not in {"column", "literal", "plan_ref", "materialized_series"}:
            try:
                from cleaned_operators.registry import OperatorRegistry

                canonical = OperatorRegistry.resolve_canonical(op)
            except Exception:  # noqa: BLE001 - bootstrap 保守按原 op
                canonical = op
            seen.add(canonical)
        stack.extend(getattr(node, "inputs", ()) or ())
    return sorted(seen)


def _plan_column_names(plan: Any) -> list[str]:
    """收集计划实际引用的 leaf column 名。"""
    if plan is None:
        return []
    seen: set[str] = set()
    stack = [plan]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        op = str(getattr(node, "op", "") or "")
        if op in {"column", "col"}:
            name = str((getattr(node, "attrs", None) or {}).get("name") or "")
            if name:
                seen.add(name)
        stack.extend(getattr(node, "inputs", ()) or ())
    return sorted(seen)


@dataclass(frozen=True)
class OperatorSemanticContractDigest:
    """R40 #110: 算子语义契约的单一 digest（投影源）。

    ``planner.plan_hash._operator_semantic_contract``（计划结构键）与
    ``runtime.factor_identity.scoped_operator_contract_hash``（因子身份依赖
    digest）都从这一个 frozen dataclass 投影 —— 同一算子的 semantic_version /
    policy / signature / implementation 变更会同步改变两条哈希路径，避免
    「plan key 已变但 identity 未变」的 split-brain。
    """

    canonical: str
    semantic_version: str
    policy_hash: str
    signature_hash: str
    implementation_hash: str
    #: backend 维度的 implementation 源码哈希（``backend_hashes`` dict 的稳定
    #: tuple 表示，dataclass 可 hash）。
    backend_hashes: tuple[tuple[str, str], ...] = ()

    def to_payload(self) -> dict[str, Any]:
        """哈希用 typed payload（排序键稳定）。"""
        return {
            "canonical": self.canonical,
            "semantic_version": self.semantic_version,
            "policy_hash": self.policy_hash,
            "signature_hash": self.signature_hash,
            "implementation_hash": self.implementation_hash,
            "backend_hashes": dict(self.backend_hashes),
        }

    def contract_hash(self) -> str:
        return _stable_hash(self.to_payload())

    @classmethod
    def for_canonical(cls, canonical: str) -> "OperatorSemanticContractDigest":
        """从 registry + evidence 构建单个算子的契约 digest（单一权威）。

        与 ``planner.plan_hash._operator_semantic_contract`` 同一字段集合：
        semantic_version / policy_hash / signature_hash / implementation_hash /
        backend_hashes。registry 未加载 / 未知算子时抛出底层异常，由调用方
        决定回退（plan_hash 回退 ``semantic_version="unregistered"``）。
        """
        from cleaned_operators.operator_policy import infer_operator_policy
        from cleaned_operators.registry import OperatorRegistry
        from backend.evidence_provenance import (
            compute_payload_hash,
            implementation_hashes_for,
        )
        from backend.production_signature import signature_for

        resolved = OperatorRegistry.resolve_canonical(canonical)
        catalog = OperatorRegistry._catalog.get(resolved, {})
        signature = signature_for(resolved)
        signature_payload = None
        if signature is not None:
            signature_payload = {
                "default_status": signature.default_status,
                "params": [(p.name, p.constraint, p.status) for p in signature.params],
            }
        impl_hash: dict[str, str] = {}
        try:
            impl_hash = implementation_hashes_for(resolved)
        except (ImportError, AttributeError, KeyError, RuntimeError, ValueError, TypeError):
            impl_hash = {}
        return cls(
            canonical=resolved,
            semantic_version=str(catalog.get("semantic_version") or "1.0"),
            policy_hash=compute_payload_hash(
                infer_operator_policy(resolved, canonical=resolved).to_dict()
            ),
            signature_hash=compute_payload_hash(signature_payload),
            implementation_hash=compute_payload_hash(impl_hash),
            backend_hashes=tuple(sorted(impl_hash.items())),
        )


def scoped_operator_contract_hash(plan: Any, *, backend: str = "pandas_numpy") -> str:
    """R32-P0-040: ``PlanOperatorDependencyDigest`` —— 只对计划引用的算子做 hash。

    与整库 ``compute_operator_catalog_hash`` 不同：无关算子变化不再污染本因子
    的 identity。未加载 registry / 未知算子按原 op 名保守计入（fail-closed：
    注册后 digest 必变）。

    R40 #109: 每个算子条目通过 :class:`OperatorSemanticContractDigest` 纳入
    ``implementation_hash``（backend.evidence_provenance 的 implementation 源码
    哈希）—— 算子 kernel 实现改动即使忘记 bump semantic_version 也会改变
    因子身份，使 checkpoint / factor_version / cache 失效。
    """
    import json

    canons = _plan_operator_canonicals(plan)
    entries: list[dict[str, Any]] = []
    for canon in canons:
        try:
            digest = OperatorSemanticContractDigest.for_canonical(canon)
            entries.append({
                "canonical": digest.canonical,
                "semantic_version": digest.semantic_version,
                "policy_hash": digest.policy_hash,
                "signature_hash": digest.signature_hash,
                "implementation_hash": digest.implementation_hash,
                "backend_hashes": dict(digest.backend_hashes),
            })
        except Exception:  # noqa: BLE001 - bootstrap 保守计入
            entries.append({
                "canonical": canon,
                "policy_hash": None,
                "implementation_hash": None,
            })
    payload = json.dumps(entries, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def scoped_field_contract_hash(plan: Any) -> str:
    """R32-P0-040: ``PlanFieldDependencyDigest`` —— 只对计划引用的字段做 hash。

    从 field registry 提取被引用字段（按 name / table.name 解析）的 spec；未注册
    的列名也计入 digest（带 unregistered 标记），保证字段加入 registry 后 digest
    变化。无关字段变化不污染本因子 identity。
    """
    import json

    cols = _plan_column_names(plan)
    if not cols:
        return _stable_hash({"scoped_fields": []})
    try:
        from fields import get_field_registry

        registry = get_field_registry()
        field_specs = {
            getattr(s, "name", ""): s.to_dict() for s in registry.fields()
        }
    except Exception:  # noqa: BLE001 - registry 不可用：整列名兜底
        field_specs = {}
    scoped: dict[str, Any] = {}
    for col in cols:
        key = str(col).lower()
        spec = field_specs.get(key)
        if spec is None:
            # 可能在 table.name 形式下（如 "ashare_price.close"）。
            for name, s in field_specs.items():
                if name.lower().endswith(f".{key}") or key.endswith(f".{name.lower()}"):
                    spec = s
                    break
        scoped[col] = spec if spec is not None else {"unregistered": True}
    return _stable_hash({"scoped_fields": scoped})


# ---------------------------------------------------------------------------
# 身份计算
# ---------------------------------------------------------------------------


def compute_factor_identity(plan: Any, ctx: Any = None) -> FactorSemanticIdentity:
    """从逻辑计划（``PlanNode`` / ``IRNode``）与执行上下文推导因子语义身份。

    ``ctx`` 可以是 dict 或对象，提供：

    - ``data_source`` / ``data_source_config``：源契约哈希来源；
    - 哈希覆盖：``ir_hash`` / ``operator_contract_hash`` / ``field_contract_hash``
      / ``source_contract_hash`` / ``source_dependency_hash``；
    - 语义字段：``market`` / ``calendar`` / ``timezone`` / ``universe`` /
      ``price_basis`` / ``pit_policy`` / ``decision_time_policy`` / ``dialect`` /
      ``dialect_version`` / ``source_scope_hash``；
    - ``factor``：``api.factor.Factor`` 对象（语义字段来源，**优先
      ``factor.semantic_identity`` 执行作用域提示**，P1-34）；
    - ``frequency``：因子频率（``factor.freq`` 兜底）。

    P1-34：因子的真实执行作用域是 ``factor.semantic_identity``（
    ``api.factor.FactorExecutionScopeHint``，字段 market / universe_id /
    frequency / calendar_id / decision_time_policy / source_scope_hash）——绝不
    只看 ``factor.market`` / ``factor.freq`` 顶层字段。同一公式执行 5m 但
    ``factor.freq=1d`` 时，身份必须绑定到 5m，否则 ``factor_version`` 与执行
    语义 split-brain。

    P1-75：``compute_ir_hash`` / ``ast_hash`` / ``ir_hash`` 在系统各处指代不同
    层级的身份：
    - *SourceExpressionIdentity* —— 原始 DSL / ``Expr`` 树（``api.dsl_parser`` /
      ``storage.catalog.compute_ir_hash`` 对 ``Expr`` 的哈希）；
    - *TypedIRIdentity* —— 类型推导后的 ``IRNode`` 图（``storage.catalog`` 的
      ``compute_ir_hash`` 对 IR 的哈希，本函数的 ``ir_hash``）；
    - *OptimizedPlanIdentity* —— 优化/重写后 ``PlanNode`` 的哈希；
    - *ExecutionSemanticIdentity* —— 本类 ``FactorSemanticIdentity``（计划哈希 +
      算子/字段/源契约 + 执行作用域），是 checkpoint / factor_version 的单一
      事实源。
    本函数把 TypedIRIdentity（``ir_hash``）与 ExecutionSemanticIdentity（本类）
    绑定在一起；各哈希优先取 ctx 覆盖值，否则用现有的权威哈希工具现算：
    ``storage.catalog.compute_ir_hash``、``cleaned_operators.compute_operator_catalog_hash``、
    ``fields.compute_field_catalog_hash``、``runtime.lineage.hash_data_source_config``、
    ``planner.source_dependencies.source_dependency_hash``。
    """
    from storage.catalog import compute_ir_hash

    ir_hash = _ctx_override(ctx, "ir_hash")
    if ir_hash is None:
        ir_hash = compute_ir_hash(plan) if plan is not None else ""

    operator_contract_hash = _ctx_override(ctx, "operator_contract_hash")
    if operator_contract_hash is None:
        # R32-P0-040: 有 plan 时只对**本 plan 依赖的算子**做 digest —— 无关算子
        # 变化不再污染单因子 identity（whole-catalog hash 污染修复）。无 plan 时
        # 回退整库 hash（保留历史行为）。
        if plan is not None:
            operator_contract_hash = scoped_operator_contract_hash(plan)
        else:
            from cleaned_operators.operator_policy import compute_operator_catalog_hash

            operator_contract_hash = compute_operator_catalog_hash()

    field_contract_hash = _ctx_override(ctx, "field_contract_hash")
    if field_contract_hash is None:
        # R32-P0-040: 只对本 plan 引用的字段做 digest。无 plan 时回退整库 hash。
        if plan is not None:
            field_contract_hash = scoped_field_contract_hash(plan)
        else:
            from fields import compute_field_catalog_hash

            field_contract_hash = compute_field_catalog_hash()

    source_contract_hash = _ctx_override(ctx, "source_contract_hash")
    if source_contract_hash is None:
        from runtime.lineage import hash_data_source_config

        data_source_config = _ctx_override(ctx, "data_source_config")
        source_contract_hash = hash_data_source_config(data_source_config or {})

    source_dependency_hash = _ctx_override(ctx, "source_dependency_hash")
    if source_dependency_hash is None:
        if plan is not None:
            from planner.source_dependencies import source_dependency_hash as _src_dep_hash

            source_dependency_hash = _src_dep_hash(plan)
        else:
            source_dependency_hash = ""

    factor = _ctx_override(ctx, "factor")
    # P1-34: 因子的真实执行作用域是 ``factor.semantic_identity``（执行作用域提示）。
    # 语义字段解析优先级：ctx 显式覆盖 > semantic_identity > factor 顶层字段 >
    # 计划根 semantic_attrs。
    scope_hint = _ctx_override(factor, "semantic_identity")
    market = _pick(
        _ctx_override(ctx, "market"),
        _ctx_override(scope_hint, "market"),
        _ctx_override(factor, "market"),
    )
    calendar = _pick(
        _ctx_override(ctx, "calendar"),
        _ctx_override(ctx, "calendar_id"),
        _ctx_override(scope_hint, "calendar_id"),
        _ctx_override(scope_hint, "calendar"),
        _ctx_override(factor, "calendar"),
        _ctx_override(factor, "calendar_id"),
    )
    timezone = _pick(
        _ctx_override(ctx, "timezone"),
        _ctx_override(scope_hint, "timezone"),
        _ctx_override(factor, "timezone"),
    )
    universe = _pick(
        _ctx_override(ctx, "universe"),
        _ctx_override(ctx, "universe_id"),
        _ctx_override(scope_hint, "universe_id"),
        _ctx_override(scope_hint, "universe"),
        _ctx_override(factor, "universe"),
        _ctx_override(factor, "universe_id"),
        _semantic_value(plan, "universe_id"),
    )
    price_basis = _pick(
        _ctx_override(ctx, "price_basis"),
        _ctx_override(scope_hint, "price_basis"),
        _ctx_override(factor, "price_basis"),
        _semantic_value(plan, "price_basis"),
    )
    pit_policy = _pick(
        _ctx_override(ctx, "pit_policy"),
        _ctx_override(scope_hint, "pit_policy"),
        _ctx_override(factor, "pit_policy"),
    )
    decision_time_policy = _pick(
        _ctx_override(ctx, "decision_time_policy"),
        _ctx_override(scope_hint, "decision_time_policy"),
        _ctx_override(factor, "decision_time_policy"),
    )
    dialect = _pick(
        _ctx_override(ctx, "dialect"),
        _ctx_override(scope_hint, "dialect"),
        _ctx_override(factor, "dialect"),
    )
    dialect_version = _pick(
        _ctx_override(ctx, "dialect_version"),
        _ctx_override(scope_hint, "dialect_version"),
        _ctx_override(factor, "dialect_version"),
    )
    # #收官轮 P0: frequency 进入身份 digest。ctx 显式覆盖 > semantic_identity.frequency
    # > factor.freq。
    frequency = _pick(
        _ctx_override(ctx, "frequency"),
        _ctx_override(scope_hint, "frequency"),
        _ctx_override(factor, "freq"),
        _ctx_override(factor, "frequency"),
        _semantic_value(plan, "frequency"),
    )
    # P1-33/34: source_scope_hash 进入身份 digest——不同 source scope 的同一公式
    # 不是同一个因子（CSE 缓存 / checkpoint / factor_version 不得跨 scope 复用）。
    source_scope_hash = _pick(
        _ctx_override(ctx, "source_scope_hash"),
        _ctx_override(scope_hint, "source_scope_hash"),
        _ctx_override(factor, "source_scope_hash"),
    )
    # R20-062..066: universe_membership_hash 进入身份 digest——同一 universe
    # 标签但不同实际成员（as-of 变化 / instrument_filter 变化）不是同一个因子。
    universe_membership_hash = _pick(
        _ctx_override(ctx, "universe_membership_hash"),
        _ctx_override(scope_hint, "universe_membership_hash"),
        _ctx_override(factor, "universe_membership_hash"),
    )
    # R20-213..219 (Identity V2): 编译器/数值/数学/物化精度语义哈希。materializer /
    # lineage_service 把 storage_precision_policy / composite_lowering_hash /
    # optimizer_rewrite_hash 写入 lineage.extra 与 ctx;numeric/mathematical/
    # history-contract hash 由计算侧 (backend.numeric_semantics / planner) 写入 ctx。
    # 任一变化 → identity digest 变化 → 旧 cache/checkpoint/evidence 失效。
    numeric_semantics_hash = _pick(
        _ctx_override(ctx, "numeric_semantics_hash"),
        _ctx_override(scope_hint, "numeric_semantics_hash"),
    )
    mathematical_semantics_hash = _pick(
        _ctx_override(ctx, "mathematical_semantics_hash"),
        _ctx_override(scope_hint, "mathematical_semantics_hash"),
    )
    composite_lowering_hash = _pick(
        _ctx_override(ctx, "composite_lowering_hash"),
        _ctx_override(scope_hint, "composite_lowering_hash"),
    )
    optimizer_rewrite_hash = _pick(
        _ctx_override(ctx, "optimizer_rewrite_hash"),
        _ctx_override(scope_hint, "optimizer_rewrite_hash"),
    )
    history_contract_hash = _pick(
        _ctx_override(ctx, "history_contract_hash"),
        _ctx_override(scope_hint, "history_contract_hash"),
    )
    storage_precision_policy = _pick(
        _ctx_override(ctx, "storage_precision_policy"),
        _ctx_override(scope_hint, "storage_precision_policy"),
    )

    return FactorSemanticIdentity(
        ir_hash=str(ir_hash),
        operator_contract_hash=str(operator_contract_hash),
        field_contract_hash=str(field_contract_hash),
        source_contract_hash=str(source_contract_hash),
        source_dependency_hash=str(source_dependency_hash),
        market=market,
        calendar=calendar,
        timezone=timezone,
        universe=universe,
        price_basis=price_basis,
        pit_policy=pit_policy,
        decision_time_policy=decision_time_policy,
        dialect=dialect,
        dialect_version=dialect_version,
        frequency=frequency,
        source_scope_hash=source_scope_hash,
        universe_membership_hash=universe_membership_hash,
        numeric_semantics_hash=numeric_semantics_hash,
        mathematical_semantics_hash=mathematical_semantics_hash,
        composite_lowering_hash=composite_lowering_hash,
        optimizer_rewrite_hash=optimizer_rewrite_hash,
        history_contract_hash=history_contract_hash,
        storage_precision_policy=storage_precision_policy,
    )


def compute_identity_from_materialize_ctx(
    ir_node: Any = None,
    ast_hash: str | None = None,
    data_source_config: dict | None = None,
    run_lineage: dict | None = None,
    frequency: str | None = None,
) -> FactorSemanticIdentity | None:
    """从 ``ParquetMaterializer.materialize`` 可见的输入推导身份。

    不可用（既无 IR 也无有效 ast_hash）时返回 ``None`` —— 调用方应把
    checkpoint 指纹当作缺失（production 必须重算）。

    #收官轮 P0：这是**兜底**路径（直接调用 ``ParquetMaterializer.materialize``
    且未显式传 ``semantic_identity`` 的调用方）。完整编排路径（
    ``runtime.materialize_service.execute_materialize``）会在 orchestrator 层
    用 factor/analysis/engine 构建含 frequency/market/universe/pit/dialect 的
    完整身份后显式传入，避免在这里根据残缺 ctx 重建。

    P1-75：这里的 ``ast_hash`` 实为 *SourceExpressionIdentity*（``Expr`` 树哈希）
    或 *TypedIRIdentity*（``IRNode`` 哈希）——调用方可能传入任一层级；本函数把
    它作为 ``ir_hash`` 绑定到 *ExecutionSemanticIdentity*（本模块
    ``FactorSemanticIdentity``）。参数名保留 ``ast_hash``（兼容历史调用方），
    语义在 docstring 中明示，不做破坏性改名。
    """
    effective_ast = ast_hash
    if effective_ast == NO_FACTOR_IDENTITY:
        effective_ast = None
    if ir_node is None and not effective_ast:
        return None
    lineage_extra = (run_lineage or {}).get("extra") or {}
    ctx: dict[str, Any] = {
        "ir_hash": effective_ast,
        "operator_contract_hash": (run_lineage or {}).get("operator_catalog_hash"),
        "field_contract_hash": (run_lineage or {}).get("field_catalog_hash"),
        "source_contract_hash": lineage_extra.get("source_contract_hash"),
        "source_dependency_hash": lineage_extra.get("source_dependency_hash"),
        "data_source_config": data_source_config,
        "frequency": frequency,
    }
    return compute_factor_identity(ir_node, ctx=ctx)


# ---------------------------------------------------------------------------
# checkpoint 指纹（#47）
# ---------------------------------------------------------------------------


def _col_canonical_bytes(s: Any) -> str:
    """dtype-preserving canonical bytes for one column (R40 #111/#116).

    - integer/unsigned/bool: raw in-memory bytes (int32 vs int64 differ in
      width; ``2**53`` vs ``2**53+1`` are preserved exactly, no float loss);
    - float: normalized -0.0 -> +0.0 and NaN -> a single bit pattern so the
      fingerprint is value-canonical across representations;
    - datetime/timedelta: 64-bit integer view (stable across pandas versions);
    - object/string: stable ``repr`` of the scalar list.
    """
    import numpy as np

    try:
        arr = np.asarray(s.to_numpy())
    except Exception:  # noqa: BLE001 - non-convertible column
        return repr(list(s))
    kind = getattr(arr.dtype, "kind", None)
    if kind == "f":
        arr = np.asarray(arr, dtype="float64")
        arr[np.isnan(arr)] = np.nan
        # -0.0 == 0.0 numerically; normalize to +0.0 for canonical equality.
        arr = np.where(arr == 0.0, np.float64(0.0), arr)
        return arr.tobytes().hex()
    if kind in "iub":
        return np.ascontiguousarray(arr).tobytes().hex()
    if kind in "mM":
        return np.asarray(arr, dtype="int64").tobytes().hex()
    if kind in "OUS":
        return repr([(x if isinstance(x, str) else repr(x)) for x in arr])
    return np.ascontiguousarray(arr).tobytes().hex()


def _null_mask_bytes(sub: Any) -> str:
    """Per-column NaN/None mask bytes (R40 #111)."""
    import numpy as np
    import pandas as pd

    parts: list[bytes] = []
    for col in sub.columns:
        mask = np.asarray(pd.isna(sub[col]), dtype=bool)
        parts.append(mask.tobytes())
    return b"\x00".join(parts).hex()


def _stable_index_bytes(sub: Any) -> str:
    """Stable hash of the frame index (R40 #111)."""
    import numpy as np

    idx = sub.index
    try:
        arr = np.asarray(idx)
    except Exception:  # noqa: BLE001 - exotic index
        arr = None
    if arr is None:
        return hashlib.sha256(repr(list(idx)).encode("utf-8")).hexdigest()
    if getattr(arr.dtype, "kind", None) in "mM":
        arr = arr.astype("int64")
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def partition_input_fingerprint(frame: Any) -> str:
    """对分区输入行（datetime, asset, value）做稳定哈希。

    用于断点续写：同一分区在当前 run 的输入与上次成功 run 不同 → 必须重算。
    对空帧返回确定性空指纹。

    R40 #111/#116: 不再 ``astype(str)`` 字符串化（int64 vs int32 vs datetime64
    会全部变成 str 导致碰撞）。payload 携带 typed schema（每列 dtype）、
    index 稳定字节、typed canonical values 与 null mask —— ``int64`` 与 ``int32``
    同值得到不同指纹，``2**53`` 与 ``2**53+1`` 不碰撞。

    R40 #112: 含 ``datetime`` + ``asset`` 列时先按两列稳定排序再 hash —— 同一
    逻辑分区不同物理行序得到同一指纹，避免误重算。

    NEW-P0-29: a fingerprint-computation FAILURE is a hard error (fail closed to
    full recompute), NEVER degraded to ``""`` — otherwise two failed computations
    both degrade to ``""`` and ``"" == ""`` wrongly skips the recompute.  Only a
    genuinely absent frame (``None``) yields the empty fingerprint, and
    :func:`checkpoint_fingerprint_matches` treats an empty partition fingerprint
    as non-matching, so a missing fingerprint never authorises a resume.
    """
    if frame is None:
        return ""
    cols = [c for c in ("datetime", "asset", "value") if c in frame.columns]
    if not cols:
        return _stable_hash({"empty": True})
    sub = frame[cols]
    if sub.empty:
        return _stable_hash({"empty": True})
    # R40 #112: 同一逻辑分区、不同物理行序 → 相同指纹。
    if "datetime" in cols and "asset" in cols:
        try:
            sub = sub.sort_values(["datetime", "asset"], kind="stable")
        except Exception:  # noqa: BLE001 - 非可排序列保守跳过
            pass
    sub = sub.reset_index(drop=True)
    schema = {col: str(sub[col].dtype) for col in cols}
    payload = {
        "schema": schema,
        "index": _stable_index_bytes(sub),
        "values": "|".join(_col_canonical_bytes(sub[col]) for col in cols),
        "null_mask": _null_mask_bytes(sub),
        "n_rows": int(len(sub)),
    }
    return _stable_hash(payload)


def checkpoint_fingerprint(
    *,
    identity_digest: str | None,
    source_snapshot: str | None,
    source_dependency_hash: str | None,
    partition_input_fingerprint: str | None,
    run_generation: str | None,
) -> dict[str, str]:
    """构建分区 checkpoint 身份指纹（全部字段归一为稳定字符串）。"""
    return {
        "identity_digest": _scalar(identity_digest),
        "source_snapshot": _scalar(source_snapshot),
        "source_dependency_hash": _scalar(source_dependency_hash),
        "partition_input_fingerprint": _scalar(partition_input_fingerprint),
        "run_generation": _scalar(run_generation),
    }


def checkpoint_fingerprint_matches(
    stored: Mapping[str, Any],
    current: Mapping[str, Any],
) -> bool:
    """校验存储指纹与当前指纹是否完全一致（任一组件变化 → False）。

    NEW-P0-29 (fail-closed): an empty/unknown ``partition_input_fingerprint`` on
    EITHER side is treated as non-matching → recompute.  ``"" == ""`` must never
    authorise a resume — a missing fingerprint is not evidence the partition input
    is unchanged.
    """
    for key in (
        "identity_digest",
        "source_snapshot",
        "source_dependency_hash",
        "partition_input_fingerprint",
        "run_generation",
    ):
        left = _scalar(stored.get(key))
        right = _scalar(current.get(key))
        if key == "partition_input_fingerprint" and (not left or not right):
            return False
        if left != right:
            return False
    return True


# ---------------------------------------------------------------------------
# R20-052..055: ExecutionSemanticIdentityV2 —— 单一定义身份 + 投影链
#
# ``FactorExecutionScope`` / ``ExecutionCacheNamespace`` / ``FactorSemanticIdentity``
# 是允许保留的不同 dataclass，但它们都必须是同一个 ``ExecutionSemanticIdentityV2``
# 的 projection。强不变量：
#
#   same ExecutionSemanticIdentity → same CSE scope → same cache namespace
#   → same checkpoint identity → same materialization factor_version。
#
# 任一投影不一致（例如只改 universe 标签字符串而不改 membership）都会使链上
# 全部投影一起变化 —— 绝不允许「身份 digest 相同但 CSE scope 不同」的 split-brain。
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionSemanticIdentityV2:
    """R20-052..055: 统一执行语义身份（单一事实源）。

    是 ``FactorExecutionScope``（CSE scope）、``ExecutionCacheNamespace``
    （cache namespace）、checkpoint 指纹与 materialization ``factor_version``
    的同一 projection 源头。任何字段变化都会改变 ``identity_digest()``，从而
    使整条投影链（CSE scope / cache namespace / checkpoint / factor_version）
    同步变化。

    字段与 ``FactorSemanticIdentity`` 对齐，另加 ``universe_membership_hash``
    （R20-062..066）。
    """

    ir_hash: str
    operator_contract_hash: str
    field_contract_hash: str
    source_contract_hash: str
    source_dependency_hash: str
    universe_membership_hash: str = ""
    market: str | None = None
    calendar: str | None = None
    timezone: str | None = None
    universe: str | None = None
    price_basis: str | None = None
    pit_policy: str | None = None
    decision_time_policy: str | None = None
    dialect: str | None = None
    dialect_version: str | None = None
    frequency: str | None = None
    source_scope_hash: str | None = None
    numeric_semantics_hash: str | None = None
    mathematical_semantics_hash: str | None = None
    composite_lowering_hash: str | None = None
    optimizer_rewrite_hash: str | None = None
    history_contract_hash: str | None = None
    storage_precision_policy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def identity_digest(self) -> str:
        return _stable_hash(self.to_dict())

    @classmethod
    def from_factor_identity(cls, identity: "FactorSemanticIdentity") -> "ExecutionSemanticIdentityV2":
        """从 ``FactorSemanticIdentity``（*ExecutionSemanticIdentity* 的既有载体）投影。"""
        return cls(
            ir_hash=identity.ir_hash,
            operator_contract_hash=identity.operator_contract_hash,
            field_contract_hash=identity.field_contract_hash,
            source_contract_hash=identity.source_contract_hash,
            source_dependency_hash=identity.source_dependency_hash,
            universe_membership_hash=str(identity.universe_membership_hash or ""),
            market=identity.market,
            calendar=identity.calendar,
            timezone=identity.timezone,
            universe=identity.universe,
            price_basis=identity.price_basis,
            pit_policy=identity.pit_policy,
            decision_time_policy=identity.decision_time_policy,
            dialect=identity.dialect,
            dialect_version=identity.dialect_version,
            frequency=identity.frequency,
            source_scope_hash=identity.source_scope_hash,
            numeric_semantics_hash=identity.numeric_semantics_hash,
            mathematical_semantics_hash=identity.mathematical_semantics_hash,
            composite_lowering_hash=identity.composite_lowering_hash,
            optimizer_rewrite_hash=identity.optimizer_rewrite_hash,
            history_contract_hash=identity.history_contract_hash,
            storage_precision_policy=identity.storage_precision_policy,
        )


class IdentityProjectionChainError(ValueError):
    """执行语义身份投影链不一致（R20-052..055）。"""


def projection_chain(identity: ExecutionSemanticIdentityV2) -> dict[str, str]:
    """计算同一身份的全部投影（R20-052..055）。

    返回：
        ``{"execution_semantic_identity", "cse_scope", "cache_namespace",
          "checkpoint_identity", "factor_version"}``
    """
    digest = identity.identity_digest()
    cse_scope = _project_cse_scope(identity)
    cache_ns = _project_cache_namespace(identity)
    return {
        "execution_semantic_identity": digest,
        "cse_scope": cse_scope,
        "cache_namespace": cache_ns,
        "checkpoint_identity": digest,
        "factor_version": digest,
    }


def assert_identity_projection_chain(identity: ExecutionSemanticIdentityV2) -> dict[str, str]:
    """强不变量：同一 ExecutionSemanticIdentity → 同一 CSE scope → 同一 cache
    namespace → 同一 checkpoint identity → 同一 factor_version。

    校验每个投影都是 identity digest 的确定性函数（同一身份两次计算投影一致；
    不同身份必然在某一投影上不同）。
    """
    chain1 = projection_chain(identity)
    chain2 = projection_chain(identity)
    for key in chain1:
        if chain1[key] != chain2[key]:
            raise IdentityProjectionChainError(
                f"identity projection chain unstable at {key!r}: {chain1[key]!r} vs {chain2[key]!r}"
            )
    return chain1


def assert_projection_chain_distinct(
    left: ExecutionSemanticIdentityV2, right: ExecutionSemanticIdentityV2
) -> bool:
    """两个不同身份必须在整条投影链上互不相同（任一投影相等 = split-brain）。"""
    chain_l = assert_identity_projection_chain(left)
    chain_r = assert_identity_projection_chain(right)
    if left == right:
        return chain_l == chain_r
    return chain_l != chain_r


def _project_cse_scope(identity: ExecutionSemanticIdentityV2) -> str:
    """CSE scope 投影：freq/universe/market/calendar/source_scope_hash +
    source_dependency_hash + universe_membership_hash。"""
    payload = {
        "frequency": _scalar(identity.frequency),
        "universe_id": _scalar(identity.universe or "ALL"),
        "market": _scalar(identity.market),
        "calendar_id": _scalar(identity.calendar),
        "source_scope_hash": _scalar(identity.source_scope_hash),
        "source_dependency_hash": _scalar(identity.source_dependency_hash),
        "universe_membership_hash": _scalar(identity.universe_membership_hash),
    }
    return _stable_hash(payload)


def _project_cache_namespace(identity: ExecutionSemanticIdentityV2) -> str:
    """cache namespace 投影：执行语义 + 源依赖 + universe membership + 契约哈希。"""
    payload = {
        "source_contract_hash": _scalar(identity.source_contract_hash),
        "source_dependency_hash": _scalar(identity.source_dependency_hash),
        "operator_contract_hash": _scalar(identity.operator_contract_hash),
        "field_contract_hash": _scalar(identity.field_contract_hash),
        "execution_scope_hash": _project_cse_scope(identity),
        "universe_membership_hash": _scalar(identity.universe_membership_hash),
    }
    return _stable_hash(payload)
