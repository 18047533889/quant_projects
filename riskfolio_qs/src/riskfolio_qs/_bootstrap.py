"""
引导模块（保留占位）：包初始化时的钩子入口。

当前 v0.2 直接使用 cvxpy，不再依赖 vendor/riskfolio。
保留此模块作为将来可能的扩展注入点。
"""

from __future__ import annotations

from pathlib import Path


def ensure_vendor_on_path() -> Path:
    """占位：返回项目根目录。

    历史：曾用于将 vendor/riskfolio 注入 sys.path。
    v0.2 起 cvxpy 直接调用，不再需要此逻辑。
    """
    return Path(__file__).resolve().parents[2]
