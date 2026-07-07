# Agent 循环：感知 → 推理 → 行动 → 观察 → 更新状态
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from config.settings import CLAUDE_MD_PATH, MAX_ROUNDS, MAX_TOKENS, MODEL
from core.state import AgentState
from hooks.base import run_hooks
from planning.todo import todo_manager
from hooks.pre_yaml_hooks import get_pre_yaml_hooks
from hooks.post_pass_hooks import get_post_pass_hooks
from planning.sub_agents.base import run_subagent
from tools import registry as tools_registry
from verifiers.audit import log_final_json
from verifiers.evaluator import is_passed


def load_system(claude_path: Path | None = None) -> str:
    """第1层：常驻上下文，从 CLAUDE.md 加载。"""
    path = claude_path or CLAUDE_MD_PATH
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "本 Agent：研报 PDF → YAML 配置 → 评价脚本评分 → 迭代至达标。"


def agent_loop(
    query: str,
    create_message: Callable[..., Any],
    *,
    system: str | None = None,
    max_rounds: int = MAX_ROUNDS,
    max_tokens: int = MAX_TOKENS,
    model: str = MODEL,
) -> tuple[Any, AgentState]:
    """
    主循环：query → 调用模型 → 若 tool_use 则执行工具并回填 → 直到非 tool_use 或达标或达到最大轮次。

    create_message: 签名 (model, system, messages, tools, max_tokens) -> response。
    response 需有 .content (list of blocks)、.stop_reason。
    block 需有 .type；若 .type == "tool_use" 则需 .id, .name, .input。

    返回 (final_response, state)。state.passed 表示是否因评分达标结束，state.final_output 可带最后结果。
    """
    system_text = system or load_system()
    tools = tools_registry.get_parent_api_tools()
    messages: list[dict[str, Any]] = [{"role": "user", "content": query}]
    state = AgentState(messages=messages)
    todo_manager.reset()

    for state.round in range(1, max_rounds + 1):
        # Nag reminder：连续 3 轮以上未调用 todo 时，在最后一条 user 消息前注入提醒
        if state.rounds_since_todo >= 3 and messages:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    content = messages[i].get("content")
                    if isinstance(content, list):
                        content.insert(0, {
                            "type": "text",
                            "text": "<reminder>Update your todos.</reminder>",
                        })
                    break

        response = create_message(
            model=model,
            system=system_text,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
        )
        stop_reason = getattr(response, "stop_reason", None) or response.get("stop_reason", "")
        content = getattr(response, "content", None) or response.get("content", [])

        # 追加 assistant 消息（含 text + tool_use blocks）
        messages.append({"role": "assistant", "content": content})

        if stop_reason != "tool_use":
            state.final_output = response
            state.rounds_since_todo += 1
            return response, state

        # 本轮有 tool_use：逐个执行并收集 tool_result
        results: list[dict[str, Any]] = []
        used_todo_this_round = False
        for block in content:
            block_type = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
            if block_type != "tool_use":
                continue
            block_id = getattr(block, "id", None) or block.get("id", "")
            block_name = getattr(block, "name", None) or block.get("name", "")
            block_input = getattr(block, "input", None) or block.get("input", {})

            # 父智能体专用：task 由 run_subagent 执行，不经过 registry
            if block_name == "task":
                prompt = block_input.get("prompt", "")
                output = run_subagent(
                    prompt,
                    create_message,
                    model=model,
                    max_tokens=max_tokens,
                )
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block_id,
                    "content": output,
                })
                continue

            # 第4层：写 YAML 前强制走 Hooks
            if block_name == "write_file":
                path = block_input.get("path", "")
                if path.endswith(".yaml") or path.endswith(".yml"):
                    state.last_yaml_path = path
                    try:
                        run_hooks(
                            get_pre_yaml_hooks(),
                            path,
                            block_input.get("content", ""),
                        )
                    except Exception as e:
                        state.last_error = f"pre_yaml_hook_failed: {e}"
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": block_id,
                            "content": json.dumps({"error": str(e)}, ensure_ascii=False),
                        })
                        continue

            output = tools_registry.run(block_name, block_input)
            if isinstance(output, str):
                try:
                    maybe = json.loads(output)
                    if isinstance(maybe, dict) and maybe.get("error"):
                        state.last_error = f"{block_name}_error: {maybe.get('error')}"
                except json.JSONDecodeError:
                    pass
            if block_name == "todo":
                used_todo_this_round = True
            results.append({
                "type": "tool_result",
                "tool_use_id": block_id,
                "content": output,
            })

            # 第6层：若为 run_eval 且达标，则 Hooks + 审计并结束
            if block_name == "run_eval":
                state.current_yaml_path = block_input.get("yaml_path")
                state.last_yaml_path = state.current_yaml_path
                state.last_eval_raw = output if isinstance(output, str) else str(output)
                try:
                    eval_result = json.loads(output)
                except json.JSONDecodeError:
                    state.last_error = "run_eval_output_not_json"
                    eval_result = {}
                if isinstance(eval_result, dict) and eval_result.get("detail"):
                    state.last_error = str(eval_result.get("detail"))
                if is_passed(eval_result):
                    state.passed = True
                    state.final_output = eval_result
                    run_hooks(get_post_pass_hooks(), state.current_yaml_path, state)
                    return response, state

        if used_todo_this_round:
            state.rounds_since_todo = 0
        else:
            state.rounds_since_todo += 1
        messages.append({"role": "user", "content": results})

    # 达到最大轮次仍未达标
    failure_result = state.final_output if isinstance(state.final_output, dict) else {}
    if not failure_result:
        failure_result = {
            "last_error": state.last_error,
            "last_yaml_path": state.last_yaml_path or state.current_yaml_path,
            "last_eval_raw": state.last_eval_raw,
            "reason": "max_rounds_reached_without_pass",
        }
    state.final_output = failure_result
    log_final_json(state.current_yaml_path or state.last_yaml_path, failure_result, False)
    return None, state
