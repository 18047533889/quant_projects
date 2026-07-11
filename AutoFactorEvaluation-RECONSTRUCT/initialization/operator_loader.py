"""算子库加载。

负责将 factor_engine 的内建算子或外部算子库加载到 OperatorRegistry。
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_factor_engine_path() -> None:
    """确保 factor_engine 目录在 sys.path 中。"""
    fe_path = str(_PROJECT_ROOT / "factor_engine")
    if fe_path not in sys.path:
        sys.path.insert(0, fe_path)


def load_operators(factor_engine_operators: str | None = None) -> int:
    """加载算子库到 OperatorRegistry。

    Args:
        factor_engine_operators: 外部算子库绝对路径。None=使用内建。

    Returns:
        已注册的算子总数。
    """
    ensure_factor_engine_path()

    # 1. 加载内建算子
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
    ensure_cleaned_loaded()

    # 2. 加载外部算子
    if factor_engine_operators is not None:
        from factor_engine.backend.operator_loader import load_external_operators
        load_external_operators(factor_engine_operators)

    return len(get_registered_operators())


def get_registered_operators() -> list[str]:
    """从 OperatorRegistry 获取所有已注册算子名。"""
    ensure_factor_engine_path()
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    return sorted(OperatorRegistry.list_canonical())
