# 待办列表：创建/完成/状态（TodoManager 存储带状态的项目，同一时间只允许一个 in_progress）
from __future__ import annotations

from typing import Any


class TodoManager:
    """存储带状态的待办项。同一时间只允许一个 in_progress。"""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def reset(self) -> None:
        """清空待办，用于新一次 agent 运行。"""
        self.items = []

    def update(self, items: list[dict[str, Any]]) -> str:
        """
        更新待办列表。items 每项需含 id、text，可选 status（pending|in_progress|completed）。
        若超过一个 status 为 in_progress 则抛 ValueError。
        返回 render() 供模型阅读。
        """
        validated: list[dict[str, Any]] = []
        in_progress_count = 0
        for item in items:
            status = item.get("status", "pending")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({
                "id": item["id"],
                "text": item["text"],
                "status": status,
            })
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress")
        self.items = validated
        return self.render()

    def render(self) -> str:
        """返回当前待办列表的文本表示，供模型或 tool 结果使用。"""
        if not self.items:
            return "（当前无待办）"
        lines = []
        for i, item in enumerate(self.items, 1):
            sid = item.get("id", "")
            text = item.get("text", "")
            status = item.get("status", "pending")
            lines.append(f"{i}. [{status}] {text} (id={sid})")
        return "\n".join(lines)


# 单例，供 todo 工具与 agent_loop 使用；每次 agent_loop 开始时可选 reset()
todo_manager = TodoManager()
