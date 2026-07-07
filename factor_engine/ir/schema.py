from dataclasses import dataclass

from .types import ValueType


@dataclass(frozen=True)
class Schema:
    value_type: ValueType
    dtype: str
    index: tuple[str, ...]
