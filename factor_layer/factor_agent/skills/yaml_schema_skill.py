# 技能：YAML 结构、达标标准与禁止项（Layer 1 简短 + Layer 2 完整占位）
from skills.base import skill_full_tag

BRIEF = "YAML 结构、达标标准与禁止项"
TOKEN_ESTIMATE = 500

FULL = skill_full_tag(
    "yaml_schema",
    """
Full reference: 配置文件 YAML 的合法结构、必填项、达标标准与禁止项。

- 结构约定见 config/yaml_schema.py 与 docs/CLAUDE.md。
- 达标标准：由 verifiers/evaluator 与 config 中的 SCORE_THRESHOLD 决定。
- 禁止项：不得缺失必填字段；不得超出约定取值范围。

（占位：后续从 yaml_schema 与 CLAUDE 同步生成或手写摘要。）
""".strip(),
)
