# 运行时状态：当前任务、上下文、YAML 版本等
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """单次 agent 运行的可见状态，供 Hooks / Verifiers 使用。"""
    messages: list[dict[str, Any]] = field(default_factory=list)
    round: int = 0
    rounds_since_todo: int = 0  # 自上次调用 todo 工具以来的轮数，用于 nag 提醒
    current_yaml_path: str | None = None
    passed: bool = False
    final_output: Any = None
    last_error: str | None = None
    last_yaml_path: str | None = None
    last_eval_raw: str | None = None
