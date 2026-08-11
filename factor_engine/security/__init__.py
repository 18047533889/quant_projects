# -*- coding: utf-8 -*-
"""FactorEngine 安全薄层（R24 P0-S4 §6 派生数据权限继承）。"""
from __future__ import annotations

# R40 #81: 相对导入 —— 绝对导入 ``factor_engine.security.access`` 假设 wheel 顶层
# 名固定为 ``factor_engine``（与 pyproject name ``factor-engine`` 不同），换包名或
# 以 ``security`` 子包被安装时都会 ImportError。相对导入与包结构绑定，始终正确。
from .access import (
    access_tag_level,
    max_sensitivity,
    derive_derived_access_tags,
    DeclassificationApproval,
    require_declassification_approval,
)

__all__ = [
    "access_tag_level",
    "max_sensitivity",
    "derive_derived_access_tags",
    "DeclassificationApproval",
    "require_declassification_approval",
]
