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
      ``dialect_version``；
    - ``factor``：``api.factor.Factor`` 对象（语义字段兜底来源）。

    各哈希优先取 ctx 覆盖值，否则用现有的权威哈希工具现算：
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
    # 语义字段解析优先级：ctx 显式覆盖 > factor 对象 > 计划根 semantic_attrs。
    market = _pick(_ctx_override(ctx, "market"), _ctx_override(factor, "market"))
    calendar = _pick(
        _ctx_override(ctx, "calendar"),
        _ctx_override(ctx, "calendar_id"),
        _ctx_override(factor, "calendar"),
        _ctx_override(factor, "calendar_id"),
    )
    timezone = _pick(
        _ctx_override(ctx, "timezone"), _ctx_override(factor, "timezone")
    )
    universe = _pick(
        _ctx_override(ctx, "universe"),
        _ctx_override(ctx, "universe_id"),
        _ctx_override(factor, "universe"),
        _ctx_override(factor, "universe_id"),
        _semantic_value(plan, "universe_id"),
    )
    price_basis = _pick(
        _ctx_override(ctx, "price_basis"),
        _ctx_override(factor, "price_basis"),
        _semantic_value(plan, "price_basis"),
    )
    pit_policy = _pick(
        _ctx_override(ctx, "pit_policy"), _ctx_override(factor, "pit_policy")
    )
    decision_time_policy = _pick(
        _ctx_override(ctx, "decision_time_policy"),
        _ctx_override(factor, "decision_time_policy"),
    )
    dialect = _pick(
        _ctx_override(ctx, "dialect"), _ctx_override(factor, "dialect")
    )
    dialect_version = _pick(
        _ctx_override(ctx, "dialect_version"),
        _ctx_override(factor, "dialect_version"),
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
    )


def compute_identity_from_materialize_ctx(
    ir_node: Any = None,
    ast_hash: str | None = None,
    data_source_config: dict | None = None,
    run_lineage: dict | None = None,
) -> FactorSemanticIdentity | None:
    """从 ``ParquetMaterializer.materialize`` 可见的输入推导身份。

    不可用（既无 IR 也无有效 ast_hash）时返回 ``None`` —— 调用方应把
    checkpoint 指纹当作缺失（production 必须重算）。
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
    }
    return compute_factor_identity(ir_node, ctx=ctx)


# ---------------------------------------------------------------------------
# checkpoint 指纹（#47）
# ---------------------------------------------------------------------------


def partition_input_fingerprint(frame: Any) -> str:
    """对分区输入行（datetime, asset, value）做稳定哈希。

    用于断点续写：同一分区在当前 run 的输入与上次成功 run 不同 → 必须重算。
    对空帧返回确定性空指纹。
    """
    if frame is None:
        return ""
    try:
        cols = [c for c in ("datetime", "asset", "value") if c in frame.columns]
        if not cols:
            return _stable_hash({"empty": True})
        sub = frame[cols]
        if sub.empty:
            return _stable_hash({"empty": True})
        # 统一转字符串，避免 dtype/float 表示跨进程不稳定。
        payload = sub.astype(str).to_dict(orient="list")
        return _stable_hash(payload)
    except Exception:
        return ""


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
    """校验存储指纹与当前指纹是否完全一致（任一组件变化 → False）。"""
    for key in (
        "identity_digest",
        "source_snapshot",
        "source_dependency_hash",
        "partition_input_fingerprint",
        "run_generation",
    ):
        if _scalar(stored.get(key)) != _scalar(current.get(key)):
            return False
    return True
