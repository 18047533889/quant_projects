"""IR 结果 schema：值类型、dtype、索引名（预留，供未来类型推断扩展）。"""

from dataclasses import dataclass

from .types import ValueType


@dataclass(frozen=True)
class Schema:
    """描述某个 IR 子树输出结果的静态 schema（当前引擎主路径未强依赖）。"""

    value_type: ValueType  # scalar / series / panel
    dtype: str             # 如 float64
    index: tuple[str, ...]  # MultiIndex 层名，如 ("timestamp", "instrument")
