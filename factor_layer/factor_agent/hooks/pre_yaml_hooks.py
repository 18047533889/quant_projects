# 写 YAML 前的校验等（第4层，不进上下文）
from __future__ import annotations

from pathlib import Path
from typing import Any

from config.yaml_schema import validate_yaml_dict


def pre_yaml_validate(path: str, content: str) -> None:
    """写 YAML 前执行：格式/必填项校验，不通过可抛异常以阻断。"""
    if not path or not path.strip():
        raise ValueError("YAML path 不能为空")
    if not content or not content.strip():
        raise ValueError("YAML content 不能为空")

    if not (path.endswith(".yaml") or path.endswith(".yml")):
        raise ValueError("仅允许写入 .yaml/.yml 文件")

    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ValueError("缺少 PyYAML 依赖，请先安装 pyyaml") from e

    try:
        data = yaml.safe_load(content)
    except Exception as e:
        raise ValueError(f"YAML 语法错误: {e}") from e

    errs = validate_yaml_dict(data if isinstance(data, dict) else {})
    if errs:
        raise ValueError("YAML schema 校验失败: " + " | ".join(errs))


def get_pre_yaml_hooks() -> list:
    """返回写 YAML 前要执行的钩子列表。"""
    return [pre_yaml_validate]
