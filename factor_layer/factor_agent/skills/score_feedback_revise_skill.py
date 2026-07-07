# 技能：根据评分反馈修订 YAML（Layer 1 简短 + Layer 2 完整占位）
from skills.base import skill_full_tag

BRIEF = "根据评分与反馈修订 YAML"
TOKEN_ESTIMATE = 600

FULL = skill_full_tag(
    "score_feedback_revise",
    """
Full workflow: 根据 run_eval 返回的 score、passed、detail 修订当前 YAML。

Step 1: 解析上一轮 run_eval 的 result（score, passed, detail）。
Step 2: 根据 detail 或扣分项定位要修改的块（因子/参数/风控等）。
Step 3: 在保持 yaml_schema 合规前提下修改 YAML 内容。
Step 4: 写回文件后再次调用 run_eval，直到 passed 或达到最大轮次。

（占位：后续补全典型扣分原因与修改策略。）
""".strip(),
)
