# 技能：研报 → 首版 YAML（Layer 1 简短 + Layer 2 完整占位）
from skills.base import skill_full_tag

BRIEF = "研报/文本 → 首版 YAML 配置"
TOKEN_ESTIMATE = 800

FULL = skill_full_tag(
    "report_to_yaml",
    """
Full workflow: 从研报或给定文本抽取信息，按 yaml_schema 生成首版配置文件。

Step 1: 读取研报内容（或使用已有文本）。
Step 2: 识别因子、参数、回测区间、风控等关键信息。
Step 3: 按 config/yaml_schema 约定填写 YAML，写至指定路径。
Step 4: （可选）调用 run_eval 做首版评分。

（占位：后续补全与 schema 的字段对应与示例。）
""".strip(),
)
