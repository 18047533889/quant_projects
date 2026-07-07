# -*- coding: utf-8 -*-
"""
可独立交付的统一算子库（**唯一 runtime 层**）。

与 ``api/`` 的关系
-----------------
- 挖掘写公式 → ``api`` 造 ``CleanedCall`` 树（不算数）；
- 引擎执行 → ``cleaned_bridge`` 调本包 ``Operator.calculate(panel)``。

模块地图
--------
- elementwise_math      逐元素数学 / 比较 / 矩阵
- time_series           时序滚动（ts_mean, ts_corr, decay_linear …）
- shift_diff_cum        滞后 / 差分 / 累计
- cross_sectional       截面 rank / zscore / scale
- group_neutralization  分组中性化 / panel_* 变换
- data_cleaning         缺失值 / 截断 / protected_*
- statistics_regression 统计量 / 回归 / 相关
- price_volume          收益 / 波动 / beta 等量价衍生
- technical_signal      经典 TA（MACD/RSI/ADX…）+ 信号（trade_when/where…）
- fundamental           财报衍生（ttm / yoy / quarter / avg2）
- intraday_microstructure  微观结构（real_turnover_rate 等）
- _aliases              DSL 别名 → canonical（**必须**在 load_all 末尾 import）

用法::

    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry
    load_all()
    OperatorRegistry.get("RSI").calculate(close_df, window=14)
"""
from cleaned_operators.registry import OperatorRegistry

from cleaned_operators import elementwise_math
from cleaned_operators import time_series
from cleaned_operators import shift_diff_cum
from cleaned_operators import cross_sectional
from cleaned_operators import group_neutralization
from cleaned_operators import data_cleaning
from cleaned_operators import statistics_regression
from cleaned_operators import price_volume
from cleaned_operators import technical_signal  # MACD/RSI/ADX + trade_when/where
from cleaned_operators import fundamental
from cleaned_operators import intraday_microstructure
from cleaned_operators import _aliases  # noqa: F401  注册别名，必须最后加载

__all__ = [
    "OperatorRegistry",
    'elementwise_math',
    'time_series',
    'shift_diff_cum',
    'cross_sectional',
    'group_neutralization',
    'data_cleaning',
    'statistics_regression',
    'price_volume',
    'technical_signal',
    'fundamental',
    'intraday_microstructure',
]


def load_all() -> None:
    """Import 全部算子子模块并完成 ``_aliases`` 注册；``cleaned_bridge`` 首次执行前会自动调用。"""
    for mod in __all__[1:]:
        __import__(f"cleaned_operators.{mod}", fromlist=[mod])
