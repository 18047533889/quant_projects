"""StrictYAMLLoader —— duplicate-key 保护的 YAML loader（#P0-46）。

datasets.yaml / semantic_fields.yaml 里的 ``market_cap:`` 重复定义默认是后覆盖前，
这会让生产配置静默丢失条目。所有平台配置统一用本 loader：duplicate mapping key
→ 硬错误。
"""
from __future__ import annotations

from typing import Any

import yaml


class StrictYAMLLoader(yaml.SafeLoader):
    """SafeLoader + duplicate mapping key → ConstructorError（fail-closed）。

    先 ``flatten_mapping`` 处理 ``<<: *anchor`` 合并键（显式 key 覆盖 merged key
    不会产生 duplicate）；真正重复的**显式** key 才报错。
    """

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[str, Any]:
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
        mapping: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"duplicate mapping key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def strict_yaml_load(text: str, *, context: str = "yaml") -> Any:
    """用 StrictYAMLLoader 解析 YAML 文本；duplicate key 抛 ValidationError。"""
    from data_access.core.exceptions import ValidationError

    try:
        return yaml.load(text, Loader=StrictYAMLLoader)
    except yaml.YAMLError as exc:
        raise ValidationError(f"{context} 解析失败（含 duplicate key 检查）：{exc}") from exc
