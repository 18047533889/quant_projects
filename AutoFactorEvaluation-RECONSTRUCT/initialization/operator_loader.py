"""从仓库根最新 FactorEngine 加载算子。"""
from __future__ import annotations

from integrations.quant_platform import bootstrap_quant_platform


def ensure_factor_engine_path() -> None:
    """兼容旧调用；真实路径引导统一由 integrations 负责。"""
    bootstrap_quant_platform()


def load_operators(factor_engine_operators: str | None = None) -> int:
    bootstrap_quant_platform()
    from backend.cleaned_bridge import ensure_cleaned_loaded

    ensure_cleaned_loaded()
    if factor_engine_operators is not None:
        from backend.operator_loader import load_external_operators

        load_external_operators(factor_engine_operators)
    return len(get_registered_operators())


def get_registered_operators() -> list[str]:
    bootstrap_quant_platform()
    from cleaned_operators.registry import OperatorRegistry

    return sorted(OperatorRegistry.list_canonical())
