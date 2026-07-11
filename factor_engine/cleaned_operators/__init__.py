# -*- coding: utf-8 -*-
"""
可独立交付的统一算子库（**唯一 runtime 层**）。

目录结构
--------
- ``common/``          共通运算（元素数学、时序、截面、分组、清洗、统计）
- ``price_volume/``    量价衍生
- ``technical/``       技术指标与信号
- ``fundamental/``     财报衍生
- ``microstructure/``  日内微观结构
- ``docs/``            算子文档（与代码分离）
- 根目录               注册中心 ``registry``、基类 ``base``、别名 ``_aliases``

用法::

    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry
    load_all()
    OperatorRegistry.get("RSI").calculate(close_df, window=14)
"""
from cleaned_operators.registry import OperatorRegistry

from cleaned_operators import common  # noqa: F401
from cleaned_operators import price_volume  # noqa: F401
from cleaned_operators import technical  # noqa: F401
from cleaned_operators import fundamental  # noqa: F401
from cleaned_operators import microstructure  # noqa: F401
from cleaned_operators import _aliases  # noqa: F401  注册别名，必须最后加载

__all__ = [
    "OperatorRegistry",
    "common",
    "price_volume",
    "technical",
    "fundamental",
    "microstructure",
]

# 加载顺序：common 子模块 → GTJA 严格覆盖 → 领域模块 → dedupe。
# gtja_compat 必须位于 common.time_series 之后，以同 canonical 覆盖历史 pandas 实现。
_LOAD_MODULES = (
    "cleaned_operators.common.elementwise",
    "cleaned_operators.common.time_series",
    "cleaned_operators.common.gtja_compat",
    "cleaned_operators.common.shift_cum",
    "cleaned_operators.common.cross_sectional",
    "cleaned_operators.common.group",
    "cleaned_operators.common.data_cleaning",
    "cleaned_operators.common.statistics",
    "cleaned_operators.common.polars_ops",
    "cleaned_operators.common.group_polars",
    "cleaned_operators.common.shift_polars",
    "cleaned_operators.common.polars_extended",
    "cleaned_operators.common.polars_auto",
    "cleaned_operators.common.polars_data_cleaning",
    "cleaned_operators.common.polars_statistics",
    "cleaned_operators.common.polars_np_parity",
    "cleaned_operators.common.polars_math_extended",
    "cleaned_operators.common.polars_batch_mirror",
    "cleaned_operators.price_volume.ops",
    "cleaned_operators.price_volume.polars_price_volume",
    "cleaned_operators.technical.signal",
    "cleaned_operators.technical.polars_signal",
    "cleaned_operators.fundamental.ops",
    "cleaned_operators.microstructure.ops",
    "cleaned_operators.microstructure.polars_microstructure",
)


def load_all() -> None:
    """Import 全部算子子模块并完成注册与去重。"""
    for mod in _LOAD_MODULES:
        __import__(mod, fromlist=["*"])
    from cleaned_operators._dedupe import apply_operator_deduplication

    apply_operator_deduplication()
