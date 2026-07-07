# 子 Agent 基类与运行循环（第5层）
# 父智能体拥有 task 工具；子智能体仅拥有除 task 外的基础工具，禁止递归。
from __future__ import annotations

import json
from typing import Any, Callable

from config.settings import (
    MODEL,
    MAX_TOKENS,
    SUBAGENT_MAX_ITERATIONS,
    SUBAGENT_SYSTEM,
)
from tools import registry as tools_registry

# 子 Agent 单次 tool 结果最大长度，避免上下文爆炸
SUBAGENT_TOOL_RESULT_MAX_CHARS = 50000


def _extract_final_text(content: list) -> str:
    """从 response.content 中抽取所有 text 块拼接为最终文本。"""
    parts = []
    for block in content:
        btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
        if btype == "text":
            text = getattr(block, "text", None) or block.get("text", "")
            if text:
                parts.append(text)
    return "\n".join(parts) if parts else ""


def run_subagent(
    prompt: str,
    create_message: Callable[..., Any],
    *,
    model: str = MODEL,
    max_tokens: int = MAX_TOKENS,
    system: str | None = None,
    max_iterations: int = SUBAGENT_MAX_ITERATIONS,
) -> str:
    """
    子智能体以 messages=[] 启动，运行自己的循环；仅将最终文本返回给父智能体。
    子智能体使用 CHILD_TOOLS（无 task），禁止递归生成子智能体。
    """
    sub_messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    system_text = system or SUBAGENT_SYSTEM
    child_tools = tools_registry.get_child_api_tools()

    for _ in range(max_iterations):
        response = create_message(
            model=model,
            system=system_text,
            messages=sub_messages,
            tools=child_tools,
            max_tokens=max_tokens,
        )
        stop_reason = getattr(response, "stop_reason", None) or response.get("stop_reason", "")
        content = getattr(response, "content", None) or response.get("content", [])

        sub_messages.append({"role": "assistant", "content": content})

        if stop_reason != "tool_use":
            return _extract_final_text(content) or "(子智能体无文本输出)"

        results: list[dict[str, Any]] = []
        for block in content:
            btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
            if btype != "tool_use":
                continue
            block_id = getattr(block, "id", None) or block.get("id", "")
            block_name = getattr(block, "name", None) or block.get("name", "")
            block_input = getattr(block, "input", None) or block.get("input", {})

            if block_name == "task":
                output = json.dumps({"error": "子智能体禁止调用 task，不可递归生成子智能体。"}, ensure_ascii=False)
            else:
                output = tools_registry.run(block_name, block_input)

            if len(output) > SUBAGENT_TOOL_RESULT_MAX_CHARS:
                output = output[:SUBAGENT_TOOL_RESULT_MAX_CHARS] + "\n...(结果已截断)"

            results.append({
                "type": "tool_result",
                "tool_use_id": block_id,
                "content": output,
            })

        sub_messages.append({"role": "user", "content": results})

    return "(子智能体达到安全迭代上限，未返回最终文本)"
