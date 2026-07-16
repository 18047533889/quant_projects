"""IR 类型标注：因子值的逻辑类型（标量 / 序列 / 面板）。"""

from enum import Enum


class ValueType(str, Enum):
    """编译期/运行期对因子结果形态的粗分类（目前 schema 层使用较少）。"""

    SCALAR = "scalar"   # 单值
    SERIES = "series"   # 一维序列（通常指单标的时序）
    PANEL = "panel"     # 二维面板 date × instrument
