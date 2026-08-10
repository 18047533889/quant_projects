# -*- coding: utf-8 -*-
"""FactorEngine 安全薄层（R24 P0-S4 §6 派生数据权限继承）。"""
from __future__ import annotations

from factor_engine.security.access import (
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
