# 技能注册与按名称调用（Layer 1 列表 + Layer 2 按需加载）
from __future__ import annotations

# Layer 1：常驻 system 的简短列表（Skills available: ...）
SKILLS_LAYER1: list[tuple[str, str, int]] = [
    ("report_to_yaml", "研报/文本 → 首版 YAML 配置", 800),
    ("score_feedback_revise", "根据评分与反馈修订 YAML", 600),
    ("yaml_schema", "YAML 结构、达标标准与禁止项", 500),
    ("eval_usage", "评价脚本调用与结果解读", 400),
]


def format_layer1_for_system() -> str:
    """返回可拼进 system 的「Skills available:」段落。"""
    lines = ["Skills available:"]
    for name, brief, est in SKILLS_LAYER1:
        lines.append(f"  - {name}: {brief} (~{est} tokens)")
    return "\n".join(lines)


def get_skill_content(name: str) -> str:
    """Layer 2：按需返回该技能的完整说明（<skill>...</skill> 或纯文本）。"""
    from skills import eval_usage_skill
    from skills import report_to_yaml_skill
    from skills import score_feedback_revise_skill
    from skills import yaml_schema_skill

    lookup = {
        "report_to_yaml": report_to_yaml_skill.FULL,
        "score_feedback_revise": score_feedback_revise_skill.FULL,
        "yaml_schema": yaml_schema_skill.FULL,
        "eval_usage": eval_usage_skill.FULL,
    }
    return lookup.get(name, f"(unknown skill: {name})")
