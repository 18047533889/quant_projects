from enum import Enum


class ValueType(str, Enum):
    SCALAR = "scalar"
    SERIES = "series"
    PANEL = "panel"
