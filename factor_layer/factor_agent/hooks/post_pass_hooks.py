# 达标后的落库、审计等（第4层，不进上下文）
from typing import Any


def post_pass_audit(yaml_path: str | None, state: Any) -> None:
    """达标后执行：落库、审计。具体由 verifiers.audit 完成。"""
    from verifiers.audit import log_final_json
    log_final_json(yaml_path, getattr(state, "final_output", None) or {}, getattr(state, "passed", False))


def get_post_pass_hooks() -> list:
    """返回达标后要执行的钩子列表。"""
    return [post_pass_audit]
