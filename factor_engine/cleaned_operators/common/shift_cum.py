# -*- coding: utf-8 -*-
"""
滞后、差分与累计算子（**时间维度的位移与累加**）。

语义
----
- **shift / delay**：``delay(x, d)``、``ts_delay`` — 沿时间向后看 d 期；
- **diff / delta**：``delta``、``ts_delta`` — 与 d 期前值之差；
- **cum**：``cumsum``、``cumprod``、``cummax`` 等 — 从序列起点累计到当前。

与 ``time_series.py`` 的分工：本文件侧重非 rolling 的整列位移/累计；滚动窗口统计在 ``time_series``。
别名示例：``delay`` ↔ ``ts_delay``（见 ``_aliases.py``）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from cleaned_operators._causal import causal_bfill, causal_lag
from cleaned_operators.base import (
    Operator,
    OperatorMetadata,
    SeriesOperator,
    ScalarOperator,
    TwoVarOperator,
    register_operator,
)

import numpy as np
import pandas as pd

# canonical=Lead backend=pandas_numpy selected=Lead source=time_series/shift_ops.py
@register_operator(name="Lead", category="time_series", business_category="shift_diff_cum", canonical="Lead", source="factor_dsl_np")
class Lead(SeriesOperator):
    """取未来n期的值"""

    metadata = OperatorMetadata(
        name="Lead",
        category="time_series",
        description="取未来n期的值（因子链路中禁前视，负滞后输出 NaN）",
        examples=["Lead(close, 1)"],
        param_names=["x", "n"],
        return_type="series",
        tags=["time_series", "shift", "future"]
    )

    def _calculate_series(self, x: pd.DataFrame, n: int = 1, **kwargs) -> pd.DataFrame:
        return causal_lag(x, -abs(int(n)))



# canonical=cum_avg backend=pandas_numpy selected=cum_avg source=time_series/cum_ops.py
@register_operator(name="cum_avg", category="time_series", business_category="shift_diff_cum", canonical="cum_avg", source="factor_dsl_np")
class CumAvg(SeriesOperator):
    metadata = OperatorMetadata(
        name="cum_avg", category="time_series",
        description="累积均值 (扩展平均)",
        examples=["cum_avg(close)"],
        param_names=["x"], return_type="series",
        tags=["time_series", "cumulative", "average"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.expanding(min_periods=1).mean()



# canonical=cum_count backend=pandas_numpy selected=cum_count source=time_series/cum_ops.py
@register_operator(name="cum_count", category="time_series", business_category="shift_diff_cum", canonical="cum_count", source="factor_dsl_np")
class CumCount(SeriesOperator):
    metadata = OperatorMetadata(
        name="cum_count", category="time_series",
        description="累积非空值计数",
        examples=["cum_count(close)"],
        param_names=["x"], return_type="series",
        tags=["time_series", "cumulative", "count"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.notna().cumsum()



# canonical=cum_delta backend=pandas_numpy selected=cum_delta source=time_series/cum_ops.py
@register_operator(name="cum_delta", category="time_series", business_category="shift_diff_cum", canonical="cum_delta", source="factor_dsl_np")
class CumDelta(SeriesOperator):
    metadata = OperatorMetadata(
        name="cum_delta", category="time_series",
        description="累积变化量 x - first(x)",
        examples=["cum_delta(close)"],
        param_names=["x"], return_type="series",
        tags=["time_series", "cumulative", "delta"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        result = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
        for col in x.columns:
            s = x[col]
            first_val = np.nan
            for i in range(len(s)):
                v = s.iloc[i]
                if pd.notna(v) and pd.isna(first_val):
                    first_val = v
                if pd.notna(v) and pd.notna(first_val):
                    result.iloc[i, result.columns.get_loc(col)] = v - first_val
        return result



# canonical=cum_first backend=pandas_numpy selected=cum_first source=time_series/cum_ops.py
@register_operator(name="cum_first", category="time_series", business_category="shift_diff_cum", canonical="cum_first", source="factor_dsl_np")
class CumFirst(SeriesOperator):
    metadata = OperatorMetadata(
        name="cum_first", category="time_series",
        description="截至目前的首个有效值",
        examples=["cum_first(close)"],
        param_names=["x"], return_type="series",
        tags=["time_series", "cumulative", "first"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        result = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
        for col in x.columns:
            first_idx = x[col].first_valid_index()
            if first_idx is not None:
                result.loc[first_idx:, col] = x[col].loc[first_idx]
        return result



# canonical=cum_last backend=pandas_numpy selected=cum_last source=time_series/cum_ops.py
@register_operator(name="cum_last", category="time_series", business_category="shift_diff_cum", canonical="cum_last", source="factor_dsl_np")
class CumLast(SeriesOperator):
    metadata = OperatorMetadata(
        name="cum_last", category="time_series",
        description="截至目前的最末有效值",
        examples=["cum_last(close)"],
        param_names=["x"], return_type="series",
        tags=["time_series", "cumulative", "last"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.ffill()



# canonical=cum_max backend=pandas_numpy selected=cum_max source=time_series/cum_ops.py
@register_operator(name="cum_max", category="time_series", business_category="shift_diff_cum", canonical="cum_max", source="factor_dsl_np")
class CumMax(SeriesOperator):
    metadata = OperatorMetadata(
        name="cum_max", category="time_series",
        description="累积最大值",
        examples=["cum_max(high)"],
        param_names=["x"], return_type="series",
        tags=["time_series", "cumulative", "max"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.cummax()



# canonical=cum_min backend=pandas_numpy selected=cum_min source=time_series/cum_ops.py
@register_operator(name="cum_min", category="time_series", business_category="shift_diff_cum", canonical="cum_min", source="factor_dsl_np")
class CumMin(SeriesOperator):
    metadata = OperatorMetadata(
        name="cum_min", category="time_series",
        description="累积最小值",
        examples=["cum_min(low)"],
        param_names=["x"], return_type="series",
        tags=["time_series", "cumulative", "min"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.cummin()



# canonical=cum_positive_streak backend=pandas_numpy selected=cum_positive_streak source=time_series/cum_ops.py
@register_operator(name="cum_positive_streak", category="time_series", business_category="shift_diff_cum", canonical="cum_positive_streak", source="factor_dsl_np")
class CumPositiveStreak(SeriesOperator):
    metadata = OperatorMetadata(
        name="cum_positive_streak", category="time_series",
        description="连续正值计数",
        examples=["cum_positive_streak(returns)"],
        param_names=["x"], return_type="series",
        tags=["time_series", "cumulative", "streak"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        def _streak(col):
            positive = col > 0
            groups = (~positive).cumsum()
            streak = positive.groupby(groups).cumsum()
            result = pd.Series(np.where(positive, streak, 0), index=col.index, dtype=float)
            result[col.isna()] = np.nan
            return result
        return x.apply(_streak)



# canonical=cum_prod backend=pandas_numpy selected=cum_prod source=time_series/cum_ops.py
@register_operator(name="cum_prod", category="time_series", business_category="shift_diff_cum", canonical="cum_prod", source="factor_dsl_np")
class CumProd(SeriesOperator):
    metadata = OperatorMetadata(
        name="cum_prod", category="time_series",
        description="累积乘积",
        examples=["cum_prod(1 + returns)"],
        param_names=["x"], return_type="series",
        tags=["time_series", "cumulative", "product"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.cumprod()



# canonical=cum_rank backend=pandas_numpy selected=cum_rank source=time_series/cum_ops.py
@register_operator(name="cum_rank", category="time_series", business_category="shift_diff_cum", canonical="cum_rank", source="factor_dsl_np")
class CumRank(SeriesOperator):
    metadata = OperatorMetadata(
        name="cum_rank", category="time_series",
        description="累积排名 (当前值在历史中的百分位排名)",
        examples=["cum_rank(close)"],
        param_names=["x"], return_type="series",
        tags=["time_series", "cumulative", "rank"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.expanding(min_periods=1).rank(pct=True)



# canonical=cum_standardize backend=pandas_numpy selected=cum_standardize source=time_series/cum_ops.py
@register_operator(name="cum_standardize", category="time_series", business_category="shift_diff_cum", canonical="expanding_zscore", source="factor_dsl_np")
class CumStandardize(SeriesOperator):
    metadata = OperatorMetadata(
        name="cum_standardize", category="time_series",
        description="累积Z-Score标准化",
        examples=["cum_standardize(close)"],
        param_names=["x"], return_type="series",
        tags=["time_series", "cumulative", "standardize"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        mean = x.expanding(min_periods=1).mean()
        std = x.expanding(min_periods=2).std().replace(0, 1)
        return (x - mean) / std



# canonical=cum_std backend=pandas_numpy selected=cum_std source=time_series/cum_ops.py
@register_operator(name="cum_std", category="time_series", business_category="shift_diff_cum", canonical="cum_std", source="factor_dsl_np")
class CumStd(SeriesOperator):
    metadata = OperatorMetadata(
        name="cum_std", category="time_series",
        description="累积标准差 (扩展标准差)",
        examples=["cum_std(returns)"],
        param_names=["x"], return_type="series",
        tags=["time_series", "cumulative", "std"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.expanding(min_periods=2).std()



# canonical=cum_sum backend=pandas_numpy selected=cum_sum source=time_series/cum_ops.py
@register_operator(name="cum_sum", category="time_series", business_category="shift_diff_cum", canonical="cum_sum", source="factor_dsl_np")
class CumSum(SeriesOperator):
    metadata = OperatorMetadata(
        name="cum_sum", category="time_series",
        description="累积求和",
        examples=["cum_sum(volume)"],
        param_names=["x"], return_type="series",
        tags=["time_series", "cumulative", "sum"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.cumsum()



# canonical=next backend=pandas_numpy selected=next source=time_series/shift_ops.py
@register_operator(name="next", category="time_series", business_category="shift_diff_cum", canonical="next", source="factor_dsl_np")
class Next(SeriesOperator):
    """后一个值"""

    metadata = OperatorMetadata(
        name="next",
        category="time_series",
        description="取后一个值 Lead(x, 1)",
        examples=["next(close)"],
        param_names=["x"],
        return_type="series",
        tags=["time_series", "shift"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return causal_lag(x, -1)



# canonical=prev backend=pandas_numpy selected=prev source=time_series/shift_ops.py
@register_operator(name="prev", category="time_series", business_category="shift_diff_cum", canonical="prev", source="factor_dsl_np")
class Prev(SeriesOperator):
    """前一个值"""

    metadata = OperatorMetadata(
        name="prev",
        category="time_series",
        description="取前一个值 Delay(x, 1)",
        examples=["prev(close)"],
        param_names=["x"],
        return_type="series",
        tags=["time_series", "shift", "lag"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return causal_lag(x, 1)

