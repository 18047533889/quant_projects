"""代码模式（``calc_mode="code"``）执行器。

接收一个 Python 函数字符串（``"def ..."``），动态编译后
对数据源的 MultiIndex Series 列执行计算，返回结果 Series。
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from typing import Any

from logging_utils import get_logger

logger = get_logger("backend.code_runner")


class CodeExecutionError(RuntimeError):
    """代码执行异常。"""


def _parse_function_source(source: str) -> ast.FunctionDef:
    """解析函数源码，返回 AST FunctionDef 节点。"""
    tree = ast.parse(source, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return node
    raise CodeExecutionError("源码中未找到任何函数定义")


def compile_code_function(source: str) -> tuple[types.CodeType, str]:
    """编译 Python 函数源码字符串为可执行代码对象。

    Args:
        source: 函数源码字符串，形如 ``"def my_factor(close, volume, ...): ..."``。

    Returns:
        (compiled_code, function_name) 元组。

    Raises:
        CodeExecutionError: 编译失败。
    """
    source = source.strip()
    if not source:
        raise CodeExecutionError("源码为空")

    try:
        func_def = _parse_function_source(source)
        func_name = func_def.name
        code = compile(source, "<factor_code>", "exec")
        return code, func_name
    except SyntaxError as e:
        raise CodeExecutionError(f"函数源码语法错误: {e}") from e


def execute_code_function(
    source: str,
    data_columns: dict[str, Any],
    tiny_run: bool = False,
) -> Any:
    """执行代码模式因子。

    Args:
        source: 函数源码字符串。
        data_columns: {字段名: MultiIndex Series} 映射。
        tiny_run: 是否小体量运行。

    Returns:
        函数返回值（通常为 MultiIndex Series）。

    Raises:
        CodeExecutionError: 执行异常。
    """
    code, func_name = compile_code_function(source)

    # 在受限命名空间中执行
    namespace: dict[str, Any] = {
        "pd": __import__("pandas", fromlist=["pd"]),
        "np": __import__("numpy", fromlist=["np"]),
    }
    # 注入所有数据列作为可用变量
    namespace.update(data_columns)

    try:
        exec(code, namespace)
    except Exception as e:
        raise CodeExecutionError(f"函数定义执行失败: {e}") from e

    func = namespace.get(func_name)
    if func is None or not callable(func):
        raise CodeExecutionError(f"函数 {func_name!r} 未在命名空间中找到或不可调用")

    # 收集函数参数
    import inspect
    sig = inspect.signature(func)
    kwargs: dict[str, Any] = {}
    for param_name in sig.parameters:
        if param_name in data_columns:
            kwargs[param_name] = data_columns[param_name]
        elif param_name == "tiny_run":
            kwargs[param_name] = tiny_run
        else:
            raise CodeExecutionError(
                f"函数参数 {param_name!r} 无法从数据列或内置参数中获取。"
                f"可用列: {list(data_columns.keys())}"
            )

    logger.info("执行代码模式函数 '%s'(%s)", func_name, ", ".join(kwargs))
    try:
        result = func(**kwargs)
        return result
    except Exception as e:
        raise CodeExecutionError(f"函数执行失败 '{func_name}': {e}") from e
