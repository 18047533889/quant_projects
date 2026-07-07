# 钩子基类（第4层）
from typing import Any, Callable


def run_hooks(hook_list: list[Callable[..., None]], *args: Any, **kwargs: Any) -> None:
    """顺序执行一组钩子，不把结果进上下文。"""
    for h in hook_list:
        h(*args, **kwargs)
