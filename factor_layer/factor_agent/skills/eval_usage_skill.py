# 技能：评价脚本调用与结果解读（Layer 1 简短 + Layer 2 完整占位）
from skills.base import skill_full_tag

BRIEF = "评价脚本调用与结果解读"
TOKEN_ESTIMATE = 400

FULL = skill_full_tag(
    "eval_usage",
    """
Full reference: 如何调用评价脚本并解读返回结果。

Step 1: 使用工具 run_eval(yaml_path) 对当前 YAML 运行 scripts/配置文件评价脚本.py。
Step 2: 返回为 JSON：score（数值）、passed（是否达标）、detail（说明/扣分原因）。
Step 3: 若 passed 为 true，流程结束，当前 YAML 为最终版；否则根据 detail 修订后重跑。

（占位：后续补全与 evaluator 实际返回字段的对应。）
""".strip(),
)
