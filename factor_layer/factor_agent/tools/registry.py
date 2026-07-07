# 工具注册与调用入口（第2层）
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

# 避免循环导入：在 run 时按名分发或延迟导入
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_file(path: str) -> str:
    p = _PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
    return p.read_text(encoding="utf-8")


def _write_file(path: str, content: str) -> str:
    p = _PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {p}"


def _run_eval(yaml_path: str) -> str:
    from verifiers.evaluator import run_eval_script
    result = run_eval_script(yaml_path)
    return json.dumps(result, ensure_ascii=False, indent=2)


def _todo_update(items: list) -> str:
    from planning.todo import todo_manager
    try:
        return todo_manager.update(items)
    except ValueError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# 内置工具定义：name -> (description, input_schema, callable)
_TOOLS: dict[str, tuple[str, dict, Callable[..., str]]] = {}


def register(name: str, description: str, input_schema: dict, fn: Callable[..., str]) -> None:
    _TOOLS[name] = (description, input_schema, fn)


def run(name: str, inp: dict[str, Any]) -> str:
    if name not in _TOOLS:
        return json.dumps({"error": f"unknown tool: {name}"}, ensure_ascii=False)
    _, _, fn = _TOOLS[name]
    try:
        return fn(**inp)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def get_api_tools() -> list[dict[str, Any]]:
    """返回 API 所需的 tools 列表（如 Anthropic messages.create(tools=...)）。"""
    return [
        {
            "name": name,
            "description": desc,
            "input_schema": schema,
        }
        for name, (desc, schema, _) in _TOOLS.items()
    ]


def get_child_api_tools() -> list[dict[str, Any]]:
    """子智能体可用工具：不含 task，禁止递归生成子智能体。"""
    return get_api_tools()


# 父智能体专用：task 工具定义（不放入 _TOOLS，由 agent_loop 直接执行）
TASK_TOOL_DEF = {
    "name": "task",
    "description": "Spawn a subagent with fresh context. Give it a clear prompt; only the subagent's final text reply is returned.",
    "input_schema": {
        "type": "object",
        "properties": {"prompt": {"type": "string"}},
        "required": ["prompt"],
    },
}


def get_parent_api_tools() -> list[dict[str, Any]]:
    """父智能体可用工具：子智能体全部工具 + task。"""
    return get_child_api_tools() + [TASK_TOOL_DEF]


# 默认注册：与研报→YAML→评分 流程相关
def _register_defaults() -> None:
    register(
        "read_file",
        "读取项目内或绝对路径文件内容。",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        lambda path: _read_file(path),
    )
    register(
        "write_file",
        "写入文本到指定路径（写 YAML 前会经 Hooks 校验）。",
        {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
        lambda path, content: _write_file(path, content),
    )
    register(
        "run_eval",
        "对指定 YAML 配置文件运行评价脚本，返回评分与是否达标。",
        {"type": "object", "properties": {"yaml_path": {"type": "string"}}, "required": ["yaml_path"]},
        _run_eval,
    )
    register(
        "todo",
        "更新待办列表。同一时间只允许一项为 in_progress。每项需 id、text，可选 status：pending|in_progress|completed。",
        {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "text": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                        },
                        "required": ["id", "text"],
                    },
                },
            },
            "required": ["items"],
        },
        lambda items: _todo_update(items),
    )


_register_defaults()
