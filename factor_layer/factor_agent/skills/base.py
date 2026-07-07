# 技能基类/接口（第3层）
# Layer 1：常驻 system 中的简短描述（~百 token/技能）
# Layer 2：按需 load_skill(name) 时注入的完整说明（~千级 token）
from __future__ import annotations

from typing import TypedDict


class SkillMeta(TypedDict, total=False):
    name: str
    brief: str
    token_estimate: int


def skill_full_tag(name: str, body: str) -> str:
    """包裹完整技能内容为 <skill name="...">...</skill> 供 Layer 2 注入。"""
    return f'<skill name="{name}">\n{body}\n</skill>'
