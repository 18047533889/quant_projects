"""报告侧算子与字段释义 API。

从 FactorEngine 源头（cleaned_operators/docs 释义体系 + fields.catalog 释义
注入）取权威释义，供报告渲染层（render_evoalpha14_pages 等）展示，避免各
页面各自硬编码、或出现"尚待注册表"的兜底空话。

释义来源层级（operator_doc_tier）：
    explicit > extended > batch > pattern:* > registry > fallback
本 API 只把前五层当作有效释义；fallback（纯推测兜底）返回 None，由调用方
决定是否降级展示。
"""
from __future__ import annotations

_TIER_LABEL: dict[str, str] = {
    "explicit": "显式文档",
    "extended": "扩展文档",
    "batch": "批量文档",
    "registry": "注册表描述",
}


def operator_tier(name: str) -> str:
    """算子释义来源层级；fallback 表示无有效释义。"""
    from factor_engine.cleaned_operators.docs.operator_doc_semantics import (
        operator_doc_tier as _tier,
    )

    return _tier(name)


def operator_explanation(name: str) -> dict | None:
    """算子释义 dict（name/meaning/compute/latex/tier）；未收录返回 None。

    别名自动归一（如 intra_* / ROLLING_BETA / safe_div_null 等都会路由到
    规范名），与评估引擎使用的释义完全同源。
    """
    from factor_engine.cleaned_operators.docs.operator_doc_semantics import (
        get_operator_doc,
    )

    tier = operator_tier(name)
    if tier == "fallback":
        return None
    doc = get_operator_doc(name)
    if doc is None:
        return None
    return {
        "name": name,
        "meaning": doc.meaning,
        "compute": doc.compute,
        "latex": doc.latex,
        "tier": tier,
        "tier_label": _TIER_LABEL.get(tier, tier),
    }


def operator_explanation_text(name: str) -> str | None:
    """一行中文算子释义（含义 + 计算口径）；未收录返回 None。"""
    e = operator_explanation(name)
    if e is None:
        return None
    text = (e["meaning"] or "").strip()
    compute = (e["compute"] or "").strip()
    if compute and compute not in text:
        text = f"{text} 计算口径：{compute}"
    return text or None


def explain_operators(names) -> dict[str, str | None]:
    """批量释义：{name: text or None}。"""
    return {n: operator_explanation_text(n) for n in names}


def field_explanation(field_ref: str, market: str = "ashare") -> dict | None:
    """字段释义 dict（label/table/source_name/unit/temporal_model）。

    field_ref 可为 catalog 内部名（如 ``close``、``valuation.free_cap``）。
    释义来自 fields.catalog 的 FieldSpec.description（由 glossary.py 注入，
    368/368 全覆盖），找不到字段返回 None。
    """
    from factor_engine.fields.market_registry import MULTI_MARKET_FIELD_REGISTRY

    spec = MULTI_MARKET_FIELD_REGISTRY.resolve_field(
        market, str(field_ref).removeprefix("valuation."), strict=False
    )
    if spec is None:
        return None
    return {
        "name": spec.name,
        "label": spec.description or spec.name,
        "table": spec.table,
        "source_name": spec.source_name,
        "unit": str(spec.unit),
        "temporal_model": spec.temporal_model,
    }
