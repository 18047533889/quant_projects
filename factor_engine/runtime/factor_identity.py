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
from typing import Any, Mapping

#: ``materialize()`` 未提供任何因子身份时使用的 sentinel（#17）。
NO_FACTOR_IDENTITY = "__no_hash_provided__"


class FactorIdentityMismatch(ValueError):
    """因子语义身份不匹配：生产物化缺少身份 / checkpoint 指纹不符。"""


# ---------------------------------------------------------------------------
# 稳定哈希与标量归一
# ---------------------------------------------------------------------------


def _stable_hash(payload: Mapping[str, Any]) -> str:
    """对字典做 stable canonical SHA-256（字段排序 + 紧凑 JSON）。"""
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _scalar(value: Any) -> str:
    """把任意可选字段归一为可比较的稳定字符串（None -> ""）。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


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
        from cleaned_operators.operator_policy import compute_operator_catalog_hash

        operator_contract_hash = compute_operator_catalog_hash()

    field_contract_hash = _ctx_override(ctx, "field_contract_hash")
    if field_contract_hash is None:
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


def partition_input_fingerprint(frame: Any) -> str:
    """对分区输入行（datetime, asset, value）做稳定哈希。

    用于断点续写：同一分区在当前 run 的输入与上次成功 run 不同 → 必须重算。
    对空帧返回确定性空指纹。

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
    # 统一转字符串，避免 dtype/float 表示跨进程不稳定。
    payload = sub.astype(str).to_dict(orient="list")
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
