# -*- coding: utf-8 -*-
"""自动生成缺失的 production typed signature。

R34 typed-signature 权威：每个 production canonical 都应有一个
:class:`~backend.production_signature.OperatorProductionSignature`，否则 canonical
ledger 判定 ``FAIL: no typed signature``。本模块为当前缺签名的 production canonical
保守地生成签名，**只**在以下两种确定情形生成（不做 name heuristic）：

a. 无标量参数（只有序列/panel 输入）→ ``OperatorProductionSignature(canonical, ())``
   （``default_status="production"``）；
b. 所有标量参数都带完整 :class:`~cleaned_operators.base.ParamSpec` → 逐个映射约束：

   - ``choices``          → ``"enum"``（``choices=tuple(...)``）
   - ``dtype=int, min>=1`` → ``"positive_integer"``
   - ``dtype=int, min>=0`` → ``"nonnegative_integer"``
   - ``dtype=int, 无 min`` → ``"nonnegative_integer"``
   - ``dtype=float, min>=0 且 max<=1`` → ``"probability"``
   - ``dtype=float``       → ``"finite_scalar"``
   - ``dtype=bool``        → ``"boolean"``

任何无 spec 的标量参数、或 dtype 不在上述映射域（str / int 负下界等）的标量参数
会使**整个算子跳过**（保持缺签名）——绝不伪造约束。

``input_index`` 标参数在 ``PlanNode.inputs`` 中的真实位置（与
``backend/production_signature._daily_signatures`` / fiscal v2 既有约定一致：
序列输入占用 input 槽位，标量参数从序列输入之后排起，计数含面板输入）。

Import 安全：本模块 import 时不触碰 registry；``generate_missing_production_signatures``
内部惰性调用 ``ensure_cleaned_loaded()`` 完成 ``load_all()`` 后才读取 registry。
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from factor_engine.backend.production_signature import (
    OperatorProductionSignature,
    ParamConstraint,
)

# 序列 / panel 输入名（绝不生成标量约束）。任务列出的权威集合 + 由既有
# fiscal v2 签名（conftest 强制 revision_policy input_index）证明的 ``period_id``
# + 跨算子一致的 feature/share/predictor panel 命名（f1/f2/f3, s1..s8, x1..x3）。
_PANEL_INPUT_NAMES: frozenset[str] = frozenset({
    "x", "y", "close", "open", "high", "low", "volume", "ret",
    "group", "group_id", "condition", "flag", "signal", "event",
    "trigger", "target", "predictor", "a", "b", "w",
    "period_id",              # 证据：fiscal v2 signatures（period_average 等）
    "f1", "f2", "f3",         # feature 列（knn/ridge/group-feature 家族）
    "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8",   # holder 份额 panel
    "x1", "x2", "x3",         # 多 predictor panel
})


def _constraint_kind(spec: Any) -> tuple[str, tuple[Any, ...]] | None:
    """把单个 ParamSpec 映射为 ``(constraint, choices)``；不可映射返回 None。

    映射顺序严格按任务要求：choices 优先；其次 int；其次 float；其次 bool。
    返回 None 表示该参数无法安全映射（无 spec / str / int 负下界等）——
    调用方应跳过整个算子。
    """
    if spec is None:
        return None
    choices = getattr(spec, "choices", None)
    if choices:
        return "enum", tuple(choices)
    dtype = getattr(spec, "dtype", None)
    if dtype is int:
        mn = getattr(spec, "min", None)
        if mn is not None and mn >= 1:
            return "positive_integer", ()
        if mn is None or mn >= 0:
            return "nonnegative_integer", ()
        # int 且 min<0：无对应约束词汇 -> 保守跳过。
        return None
    if dtype is float:
        mn = getattr(spec, "min", None)
        mx = getattr(spec, "max", None)
        if mn is not None and mn >= 0 and mx is not None and mx <= 1:
            return "probability", ()
        return "finite_scalar", ()
    if dtype is bool:
        return "boolean", ()
    # str / 其他 dtype：无对应约束词汇 -> 保守跳过。
    return None


def _build_signature(canonical: str) -> OperatorProductionSignature | None:
    """为一个缺签名 canonical 构建签名；无法安全生成时返回 None。

    读取 pandas_numpy 实例的 ``metadata``（``param_names`` / ``param_specs`` /
    ``panel_params``）。``panel_params`` 显式声明优先于 ``_PANEL_INPUT_NAMES``。
    """
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    operator = OperatorRegistry._operators.get(canonical, {}).get("pandas_numpy")
    if operator is None:
        return None
    metadata = getattr(operator, "metadata", None)
    if metadata is None:
        return None
    names = list(getattr(metadata, "param_names", None) or [])
    declared_panels = frozenset(getattr(metadata, "panel_params", None) or ())
    specs = getattr(metadata, "param_specs", None) or {}

    constraints: list[ParamConstraint] = []
    position = 0
    for name in names:
        if name in declared_panels or name in _PANEL_INPUT_NAMES:
            # 序列/panel 输入占用一个 input 槽位，但不产生标量约束。
            position += 1
            continue
        mapped = _constraint_kind(specs.get(name))
        if mapped is None:
            # 任一标量参数无法安全映射 -> 整个算子保持缺签名（保守）。
            return None
        kind, choices = mapped
        constraints.append(
            ParamConstraint(name, kind, input_index=position, choices=choices)
        )
        position += 1
    return OperatorProductionSignature(
        canonical,
        tuple(constraints),
        default_status="production",
    )


@lru_cache(maxsize=1)
def _generated_signatures() -> dict[str, OperatorProductionSignature]:
    """惰性构建 canonical -> signature 的 dict（模块级缓存，只建一次）。"""
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded

    # 生成器 import 时安全：只有在这里才确保 registry 已 load_all 完成。
    ensure_cleaned_loaded()

    from factor_engine.backend import production_signature as ps
    from factor_engine.cleaned_operators.production_hardening import factor_production_targets

    covered = set(ps.PRODUCTION_SIGNATURES) | set(ps._COMPATIBILITY_SIGNATURES)
    out: dict[str, OperatorProductionSignature] = {}
    for canonical in sorted(factor_production_targets()):
        if canonical in covered:
            continue
        signature = _build_signature(canonical)
        if signature is not None:
            out[canonical] = signature
    return out


def generate_missing_production_signatures() -> dict[str, OperatorProductionSignature]:
    """返回缺签名 canonical 的自动生成签名 dict（只含本模块生成的）。"""
    return dict(_generated_signatures())


def generated_signature_for(canonical: str) -> OperatorProductionSignature | None:
    """按 canonical 查自动生成的签名（无则 None）。"""
    return _generated_signatures().get(canonical)
