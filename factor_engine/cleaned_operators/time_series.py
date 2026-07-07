# -*- coding: utf-8 -*-
"""
时序滚动算子（**沿时间轴、按单标的** 在 panel 上计算）。

语义
----
对每个标的列独立做 rolling / EWM / 滞后窗口统计；**不是**截面 rank（见 ``cross_sectional.py``）。
DSL 常用 ``ts_*`` 前缀：``ts_mean``、``ts_std``、``ts_corr``、``ts_rank``、``ts_regression`` 等。

本模块还包含
------------
- 移动平均族：``SMA`` / ``WMA`` / ``EMA``（别名见 ``_aliases.py``）；
- 衰减加权：``decay_linear``（``ts_decay_linear``）、``ts_decay``、``ts_sum_decay``；
- ``m_*`` / ``tm_*``：窗口内 top-N、分位数、beta 等扩展滚动统计；
- 与 ``shift_diff_cum.py`` 部分重叠的 ``ts_delay`` / ``ts_delta``（以 registry canonical 为准）。

输入：宽表 ``DataFrame``，index=交易日，columns=instrument。
输出：同形 panel。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from cleaned_operators._causal import causal_lag
from cleaned_operators._rolling_fast import (
    cum_top_n_mean,
    cum_top_n_sum,
    rolling_beta,
    rolling_bottom_n_mean,
    rolling_bottom_n_sum,
    rolling_linear_weighted,
    rolling_regression,
    rolling_top_n_mean,
    rolling_top_n_mean_window,
    rolling_top_n_std,
    rolling_top_n_sum,
    rolling_top_n_sum_window,
)
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
try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

# canonical=SMA backend=pandas_numpy selected=SMA source=time_series/m_ops.py
@register_operator(name="SMA", category="time_series", business_category="time_series", canonical="SMA", source="factor_dsl_np")
class SMA(SeriesOperator):
    """简单移动平均"""

    metadata = OperatorMetadata(
        name="SMA",
        category="time_series",
        description="计算n期简单移动平均（与m_avg相同）",
        examples=["SMA(close, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "sma", "simple"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).mean()



# canonical=WMA backend=pandas_numpy selected=WMA source=time_series/m_ops.py
@register_operator(name="WMA", category="time_series", business_category="time_series", canonical="WMA", source="factor_dsl_np")
class WMA(SeriesOperator):
    """加权移动平均"""

    metadata = OperatorMetadata(
        name="WMA",
        category="time_series",
        description="计算n期加权移动平均",
        examples=["WMA(close, 10)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "wma", "weighted"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 10, **kwargs) -> pd.DataFrame:
        return rolling_linear_weighted(x, window)



# canonical=aggr_top_n backend=pandas_numpy selected=aggr_top_n source=time_series/topn_ops.py
@register_operator(name="aggr_top_n", category="time_series", business_category="time_series", canonical="aggr_top_n", source="factor_dsl_np")
class AggrTopN(SeriesOperator):
    metadata = OperatorMetadata(
        name="aggr_top_n", category="time_series",
        description="自定义Top-N聚合",
        examples=["aggr_top_n('sum', close, volume, 10, True)"],
        param_names=["aggr_func", "x", "sort_col", "top", "asc"], return_type="series",
        tags=["time_series", "top_n", "aggregate"]
    )
    def _calculate_series(self, aggr_func: str = "sum", x: pd.DataFrame = None,
                          sort_col: pd.DataFrame = None, top: int = 10,
                          asc: bool = True, **kwargs) -> pd.DataFrame:
        if x is None:
            return pd.DataFrame()
        if sort_col is None:
            sort_col = x
        result = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
        for idx in x.index:
            row_x = x.loc[idx]
            row_sort = sort_col.loc[idx] if idx in sort_col.index else row_x
            valid_mask = row_x.notna() & row_sort.notna()
            if valid_mask.sum() == 0:
                continue
            valid_x = row_x[valid_mask]
            valid_sort = row_sort[valid_mask]
            sorted_cols = valid_sort.sort_values(ascending=asc).index[:top]
            selected = valid_x[sorted_cols]
            if aggr_func == "sum":
                result.loc[idx, sorted_cols] = selected.sum()
            elif aggr_func == "avg" or aggr_func == "mean":
                result.loc[idx, sorted_cols] = selected.mean()
            elif aggr_func == "max":
                result.loc[idx, sorted_cols] = selected.max()
            elif aggr_func == "min":
                result.loc[idx, sorted_cols] = selected.min()
            elif aggr_func == "std":
                result.loc[idx, sorted_cols] = selected.std()
            elif aggr_func == "count":
                result.loc[idx, sorted_cols] = selected.count()
            else:
                result.loc[idx, sorted_cols] = selected.sum()
        return result



# canonical=cum_top_n_avg backend=pandas_numpy selected=cum_top_n_avg source=time_series/topn_ops.py
@register_operator(name="cum_top_n_avg", category="time_series", business_category="time_series", canonical="cum_top_n_avg", source="factor_dsl_np")
class CumTopNAvg(SeriesOperator):
    metadata = OperatorMetadata(
        name="cum_top_n_avg", category="time_series",
        description="累积前N大值均值",
        examples=["cum_top_n_avg(volume, 5)"],
        param_names=["x", "n"], return_type="series",
        tags=["time_series", "cumulative", "top_n", "average"]
    )
    def _calculate_series(self, x: pd.DataFrame, n: int = 5, **kwargs) -> pd.DataFrame:
        return cum_top_n_mean(x, n)



# canonical=cum_top_n_sum backend=pandas_numpy selected=cum_top_n_sum source=time_series/topn_ops.py
@register_operator(name="cum_top_n_sum", category="time_series", business_category="time_series", canonical="cum_top_n_sum", source="factor_dsl_np")
class CumTopNSum(SeriesOperator):
    metadata = OperatorMetadata(
        name="cum_top_n_sum", category="time_series",
        description="累积前N大值求和",
        examples=["cum_top_n_sum(volume, 5)"],
        param_names=["x", "n"], return_type="series",
        tags=["time_series", "cumulative", "top_n", "sum"]
    )
    def _calculate_series(self, x: pd.DataFrame, n: int = 5, **kwargs) -> pd.DataFrame:
        return cum_top_n_sum(x, n)



# canonical=decay_linear backend=pandas_numpy selected=ts_decay_linear source=time_series/ts_ops.py
@register_operator(name="ts_decay_linear", category="time_series", business_category="time_series", canonical="decay_linear", source="factor_dsl_np")
class TSDecayLinear(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_decay_linear", category="time_series",
        description="线性加权滚动平均",
        examples=["ts_decay_linear(close, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "decay", "linear"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return rolling_linear_weighted(x, window)

# aliases: DECAY_LINEAR, TS_DECAY_LINEAR



# canonical=ema backend=pandas_numpy selected=EMA source=time_series/m_ops.py
@register_operator(name="EMA", category="time_series", business_category="time_series", canonical="ema", source="factor_dsl_np")
class EMA(SeriesOperator):
    """指数移动平均"""

    metadata = OperatorMetadata(
        name="EMA",
        category="time_series",
        description="计算n期指数移动平均",
        examples=["EMA(close, 12)", "EMA(close, 26)"],
        param_names=["x", "span"],
        return_type="series",
        tags=["time_series", "ema", "exponential"]
    )

    def _calculate_series(self, x: pd.DataFrame, span: int = 12, **kwargs) -> pd.DataFrame:
        return x.ewm(span=span, adjust=False).mean()



# canonical=m_argmax backend=pandas_numpy selected=m_argmax source=time_series/m_ops.py
@register_operator(name="m_argmax", category="time_series", business_category="time_series", canonical="m_argmax", source="factor_dsl_np")
class MovingArgmax(SeriesOperator):
    """移动窗口内最大值的位置"""

    metadata = OperatorMetadata(
        name="m_argmax",
        category="time_series",
        description="返回窗口内最大值的位置",
        examples=["m_argmax(close, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "argmax"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        from cleaned_operators._rolling_fast import rolling_argmax

        return rolling_argmax(x, window)



# canonical=m_argmin backend=pandas_numpy selected=m_argmin source=time_series/m_ops.py
@register_operator(name="m_argmin", category="time_series", business_category="time_series", canonical="m_argmin", source="factor_dsl_np")
class MovingArgmin(SeriesOperator):
    """移动窗口内最小值的位置"""

    metadata = OperatorMetadata(
        name="m_argmin",
        category="time_series",
        description="返回窗口内最小值的位置",
        examples=["m_argmin(close, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "argmin"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        from cleaned_operators._rolling_fast import rolling_argmin

        return rolling_argmin(x, window)



# canonical=m_beta backend=pandas_numpy selected=m_beta source=time_series/m_ops.py
@register_operator(name="m_beta", category="time_series", business_category="time_series", canonical="m_beta", source="factor_dsl_np")
class MovingBeta(SeriesOperator):
    """移动Beta"""

    metadata = OperatorMetadata(
        name="m_beta",
        category="time_series",
        description="计算两变量的n期移动Beta",
        examples=["m_beta(returns, market, 20)"],
        param_names=["y", "x", "window"],
        return_type="series",
        tags=["time_series", "moving", "beta"]
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return y.rolling(window=window, min_periods=1).cov(x) / x.rolling(window=window, min_periods=1).var()



# canonical=m_bottom_n_avg backend=pandas_numpy selected=m_bottom_n_avg source=time_series/topn_ops.py
@register_operator(name="m_bottom_n_avg", category="time_series", business_category="time_series", canonical="m_bottom_n_avg", source="factor_dsl_np")
class MovingBottomNAvg(SeriesOperator):
    metadata = OperatorMetadata(
        name="m_bottom_n_avg", category="time_series",
        description="滚动窗口内后N小值的均值",
        examples=["m_bottom_n_avg(volume, 5)"],
        param_names=["x", "n"], return_type="series",
        tags=["time_series", "bottom_n", "average"]
    )
    def _calculate_series(self, x: pd.DataFrame, n: int = 5, **kwargs) -> pd.DataFrame:
        return rolling_bottom_n_mean(x, n)



# canonical=m_bottom_n_sum backend=pandas_numpy selected=m_bottom_n_sum source=time_series/topn_ops.py
@register_operator(name="m_bottom_n_sum", category="time_series", business_category="time_series", canonical="m_bottom_n_sum", source="factor_dsl_np")
class MovingBottomNSum(SeriesOperator):
    metadata = OperatorMetadata(
        name="m_bottom_n_sum", category="time_series",
        description="滚动窗口内后N小值的求和",
        examples=["m_bottom_n_sum(volume, 5)"],
        param_names=["x", "n"], return_type="series",
        tags=["time_series", "bottom_n", "sum"]
    )
    def _calculate_series(self, x: pd.DataFrame, n: int = 5, **kwargs) -> pd.DataFrame:
        return rolling_bottom_n_sum(x, n)



# canonical=m_mad backend=pandas_numpy selected=m_mad source=time_series/m_ops.py
@register_operator(name="m_mad", category="time_series", business_category="time_series", canonical="m_mad", source="factor_dsl_np")
class MovingMAD(SeriesOperator):
    """移动平均绝对离差"""

    metadata = OperatorMetadata(
        name="m_mad",
        category="time_series",
        description="计算n期移动平均绝对离差",
        examples=["m_mad(close, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "mad"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        median = x.rolling(window=window, min_periods=1).median()
        return (x - median).abs().rolling(window=window, min_periods=1).mean()



# canonical=m_median backend=pandas_numpy selected=m_median source=time_series/m_ops.py
@register_operator(name="m_median", category="time_series", business_category="time_series", canonical="m_median", source="factor_dsl_np")
class MovingMedian(SeriesOperator):
    """移动中位数"""

    metadata = OperatorMetadata(
        name="m_median",
        category="time_series",
        description="计算n期移动中位数",
        examples=["m_median(close, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "moving", "median"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).median()



# canonical=m_pct_change backend=pandas_numpy selected=m_pct_change source=time_series/m_ops.py
@register_operator(name="m_pct_change", category="time_series", business_category="time_series", canonical="m_pct_change", source="factor_dsl_np")
class MovingPctChange(SeriesOperator):
    """移动百分比变化"""

    metadata = OperatorMetadata(
        name="m_pct_change",
        category="time_series",
        description="计算n期百分比变化",
        examples=["m_pct_change(close, 1)"],
        param_names=["x", "periods"],
        return_type="series",
        tags=["time_series", "pct_change"]
    )

    def _calculate_series(self, x: pd.DataFrame, periods: int = 1, **kwargs) -> pd.DataFrame:
        return x.pct_change(periods)



# canonical=m_percentile backend=pandas_numpy selected=m_percentile source=time_series/m_ops.py
@register_operator(name="m_percentile", category="time_series", business_category="time_series", canonical="m_percentile", source="factor_dsl_np")
class MovingPercentile(SeriesOperator):
    """移动分位数"""

    metadata = OperatorMetadata(
        name="m_percentile",
        category="time_series",
        description="计算n期移动分位数",
        examples=["m_percentile(returns, 20, 0.75)"],
        param_names=["x", "window", "p"],
        return_type="series",
        tags=["time_series", "moving", "percentile"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, p: float = 0.5, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).quantile(p)



# canonical=m_top_n_avg backend=pandas_numpy selected=m_top_n_avg source=time_series/topn_ops.py
@register_operator(name="m_top_n_avg", category="time_series", business_category="time_series", canonical="m_top_n_avg", source="factor_dsl_np")
class MovingTopNAvg(SeriesOperator):
    metadata = OperatorMetadata(
        name="m_top_n_avg", category="time_series",
        description="滚动窗口内前N大值的均值",
        examples=["m_top_n_avg(volume, 5)"],
        param_names=["x", "n"], return_type="series",
        tags=["time_series", "top_n", "average"]
    )
    def _calculate_series(self, x: pd.DataFrame, n: int = 5, **kwargs) -> pd.DataFrame:
        return rolling_top_n_mean(x, n)



# canonical=m_top_n_std backend=pandas_numpy selected=m_top_n_std source=time_series/topn_ops.py
@register_operator(name="m_top_n_std", category="time_series", business_category="time_series", canonical="m_top_n_std", source="factor_dsl_np")
class MovingTopNStd(SeriesOperator):
    metadata = OperatorMetadata(
        name="m_top_n_std", category="time_series",
        description="滚动窗口内前N大值的标准差",
        examples=["m_top_n_std(volume, 5)"],
        param_names=["x", "n"], return_type="series",
        tags=["time_series", "top_n", "std"]
    )
    def _calculate_series(self, x: pd.DataFrame, n: int = 5, **kwargs) -> pd.DataFrame:
        return rolling_top_n_std(x, n)



# canonical=m_top_n_sum backend=pandas_numpy selected=m_top_n_sum source=time_series/topn_ops.py
@register_operator(name="m_top_n_sum", category="time_series", business_category="time_series", canonical="m_top_n_sum", source="factor_dsl_np")
class MovingTopNSum(SeriesOperator):
    metadata = OperatorMetadata(
        name="m_top_n_sum", category="time_series",
        description="滚动窗口内前N大值的求和",
        examples=["m_top_n_sum(volume, 5)"],
        param_names=["x", "n"], return_type="series",
        tags=["time_series", "top_n", "sum"]
    )
    def _calculate_series(self, x: pd.DataFrame, n: int = 5, **kwargs) -> pd.DataFrame:
        return rolling_top_n_sum(x, n)



# canonical=m_var backend=pandas_numpy selected=m_var source=time_series/m_ops.py
@register_operator(name="m_var", category="time_series", business_category="time_series", canonical="m_var", source="factor_dsl_np")
class MovingVariance(SeriesOperator):
    """移动方差"""

    metadata = OperatorMetadata(
        name="m_var",
        category="time_series",
        description="计算n期移动方差",
        examples=["m_var(returns, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "moving", "variance"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).var()



# canonical=m_zscore backend=pandas_numpy selected=m_zscore source=time_series/m_ops.py
@register_operator(name="m_zscore", category="time_series", business_category="time_series", canonical="m_zscore", source="factor_dsl_np")
class MovingZscore(SeriesOperator):
    """移动窗口Z-Score标准化"""

    metadata = OperatorMetadata(
        name="m_zscore",
        category="time_series",
        description="移动窗口内的Z-Score标准化",
        examples=["m_zscore(close, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "zscore"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        mean = x.rolling(window=window, min_periods=1).mean()
        std = x.rolling(window=window, min_periods=1).std().replace(0, 1)
        return (x - mean) / std



# canonical=tm_top_n_avg backend=pandas_numpy selected=tm_top_n_avg source=time_series/topn_ops.py
@register_operator(name="tm_top_n_avg", category="time_series", business_category="time_series", canonical="tm_top_n_avg", source="factor_dsl_np")
class TimeWindowTopNAvg(SeriesOperator):
    metadata = OperatorMetadata(
        name="tm_top_n_avg", category="time_series",
        description="时间窗口内前N大值的均值",
        examples=["tm_top_n_avg(close, 20, 5)"],
        param_names=["x", "time_window", "n"], return_type="series",
        tags=["time_series", "top_n", "average", "time_window"]
    )
    def _calculate_series(self, x: pd.DataFrame, time_window: int = 20, n: int = 5, **kwargs) -> pd.DataFrame:
        return rolling_top_n_mean_window(x, time_window, n)



# canonical=tm_top_n_sum backend=pandas_numpy selected=tm_top_n_sum source=time_series/topn_ops.py
@register_operator(name="tm_top_n_sum", category="time_series", business_category="time_series", canonical="tm_top_n_sum", source="factor_dsl_np")
class TimeWindowTopNSum(SeriesOperator):
    metadata = OperatorMetadata(
        name="tm_top_n_sum", category="time_series",
        description="时间窗口内前N大值的求和",
        examples=["tm_top_n_sum(close, 20, 5)"],
        param_names=["x", "time_window", "n"], return_type="series",
        tags=["time_series", "top_n", "sum", "time_window"]
    )
    def _calculate_series(self, x: pd.DataFrame, time_window: int = 20, n: int = 5, **kwargs) -> pd.DataFrame:
        return rolling_top_n_sum_window(x, time_window, n)



# canonical=ts_corr backend=pandas_numpy selected=ts_corr source=time_series/ts_ops.py

# helper for ts_corr
class TSCorrelation(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_correlation", category="time_series",
        description="滚动相关系数 (与m_cor相同)",
        examples=["ts_correlation(close, volume, 20)"],
        param_names=["x", "y", "window"], return_type="series",
        tags=["time_series", "ts_", "correlation"]
    )
    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=2).corr(y)

@register_operator(name="ts_corr", category="time_series", business_category="time_series", canonical="ts_corr", source="factor_dsl_np")
class TSCorr(TSCorrelation):
    metadata = OperatorMetadata(
        name="ts_corr", category="time_series",
        description="滚动相关系数 (ts_correlation的别名)",
        examples=["ts_corr(close, volume, 20)"],
        param_names=["x", "y", "window"], return_type="series",
        tags=["time_series", "ts_", "corr"]
    )

# aliases: TS_CORR, correlation, m_cor, ts_correlation



# canonical=ts_cov backend=pandas_numpy selected=ts_cov source=time_series/ts_ops.py
@register_operator(name="ts_cov", category="time_series", business_category="time_series", canonical="ts_cov", source="factor_dsl_np")
class TSCov(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_cov", category="time_series",
        description="滚动协方差 (与m_cov相同)",
        examples=["ts_cov(returns, market, 20)"],
        param_names=["x", "y", "window"], return_type="series",
        tags=["time_series", "ts_", "cov"]
    )
    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=2).cov(y)

# aliases: TS_COV, m_cov, ts_covariance



# canonical=ts_decay backend=pandas_numpy selected=ts_decay source=time_series/ts_ops.py
@register_operator(name="ts_decay", category="time_series", business_category="time_series", canonical="ts_decay", source="factor_dsl_np")
class TSDecay(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_decay", category="time_series",
        description="衰减操作 (与ts_decay_linear相同)",
        examples=["ts_decay(close, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "decay"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return rolling_linear_weighted(x, window)



# canonical=ts_decay_exp_window backend=pandas_numpy selected=ts_decay_exp_window source=time_series/ts_ops.py
@register_operator(name="ts_decay_exp_window", category="time_series", business_category="time_series", canonical="ts_decay_exp_window", source="factor_dsl_np")
class TSDecayExpWindow(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_decay_exp_window", category="time_series",
        description="指数加权滚动",
        examples=["ts_decay_exp_window(volume, 10, 0.5)"],
        param_names=["x", "window", "alpha"], return_type="series",
        tags=["time_series", "ts_", "decay", "exp"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 10, alpha: float = 0.5, **kwargs) -> pd.DataFrame:
        weights = np.array([alpha ** i for i in range(window)][::-1], dtype=np.float64)
        weights = weights / weights.sum()
        return x.rolling(window=window, min_periods=1).apply(
            lambda s: np.dot(s, weights[-len(s):]) / weights[-len(s):].sum() if len(s) > 0 else np.nan,
            raw=True,
        )



# canonical=ts_delay backend=pandas_numpy selected=ts_delay source=time_series/ts_ops.py
@register_operator(name="ts_delay", category="time_series", business_category="time_series", canonical="ts_delay", source="factor_dsl_np")
class TSDelay(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_delay", category="time_series",
        description="n期滞后 (与Ref相同)",
        examples=["ts_delay(close, 5)"],
        param_names=["x", "n"], return_type="series",
        tags=["time_series", "ts_", "delay"]
    )
    def _calculate_series(self, x: pd.DataFrame, n: int = 1, **kwargs) -> pd.DataFrame:
        return causal_lag(x, n)

# aliases: DELAY, Delay, Ref, delay, m_delay, shift



# canonical=ts_delta backend=pandas_numpy selected=ts_delta source=time_series/ts_ops.py
@register_operator(name="ts_delta", category="time_series", business_category="time_series", canonical="ts_delta", source="factor_dsl_np")
class TSDelta(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_delta", category="time_series",
        description="n期差分 (与Delta相同)",
        examples=["ts_delta(close, 1)"],
        param_names=["x", "n"], return_type="series",
        tags=["time_series", "ts_", "delta"]
    )
    def _calculate_series(self, x: pd.DataFrame, n: int = 1, **kwargs) -> pd.DataFrame:
        return x - causal_lag(x, n)

# aliases: Delta, Diff, TS_DELTA, pct_change



# canonical=ts_kurt backend=pandas_numpy selected=m_kurt source=time_series/m_ops.py
@register_operator(name="m_kurt", category="time_series", business_category="time_series", canonical="ts_kurt", source="factor_dsl_np")
class MovingKurt(SeriesOperator):
    """移动峰度"""

    metadata = OperatorMetadata(
        name="m_kurt",
        category="time_series",
        description="计算n期移动峰度",
        examples=["m_kurt(returns, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "moving", "kurtosis"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).kurt()

# aliases: TS_KURT



# canonical=ts_max backend=pandas_numpy selected=ts_max source=time_series/ts_ops.py
@register_operator(name="ts_max", category="time_series", business_category="time_series", canonical="ts_max", source="factor_dsl_np")
class TSMax(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_max", category="time_series",
        description="滚动最大值 (与m_max相同)",
        examples=["ts_max(high, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "max"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).max()

# aliases: Max, TS_MAX, m_max, max



# canonical=ts_mean backend=pandas_numpy selected=ts_mean source=time_series/ts_ops.py
@register_operator(name="ts_mean", category="time_series", business_category="time_series", canonical="ts_mean", source="factor_dsl_np")
class TSMean(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_mean", category="time_series",
        description="滚动均值 (与m_avg相同)",
        examples=["ts_mean(close, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "mean"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).mean()

# aliases: Mean, TS_MEAN, m_avg, mean



# canonical=ts_min backend=pandas_numpy selected=ts_min source=time_series/ts_ops.py
@register_operator(name="ts_min", category="time_series", business_category="time_series", canonical="ts_min", source="factor_dsl_np")
class TSMin(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_min", category="time_series",
        description="滚动最小值 (与m_min相同)",
        examples=["ts_min(low, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "min"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).min()

# aliases: Min, TS_MIN, m_min, min



# canonical=ts_product backend=pandas_numpy selected=ts_product source=time_series/ts_ops.py
@register_operator(name="ts_product", category="time_series", business_category="time_series", canonical="ts_product", source="factor_dsl_np")
class TSProduct(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_product", category="time_series",
        description="滚动乘积",
        examples=["ts_product(volume + 1, 5)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "product"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 5, **kwargs) -> pd.DataFrame:
        log_x = np.log(x.replace(0, np.nan))
        return np.exp(log_x.rolling(window=window, min_periods=1).sum())



# canonical=ts_rank backend=pandas_numpy selected=ts_rank source=time_series/ts_ops.py
@register_operator(name="ts_rank", category="time_series", business_category="time_series", canonical="ts_rank", source="factor_dsl_np")
class TSRank(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_rank", category="time_series",
        description="滚动排名 (与m_rank相同)",
        examples=["ts_rank(volume, 10)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "rank"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).rank(pct=True)

# aliases: TS_RANK, m_rank



# canonical=ts_regression backend=pandas_numpy selected=ts_regression source=time_series/ts_ops.py
@register_operator(name="ts_regression", category="time_series", business_category="time_series", canonical="ts_regression", source="factor_dsl_np")
class TSRegression(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_regression", category="time_series",
        description="滚动回归",
        examples=["ts_regression(returns, market, 252, 1, 'slope')"],
        param_names=["y", "x", "window", "lag", "retval"], return_type="series",
        tags=["time_series", "ts_", "regression"]
    )
    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 252,
                          lag: int = 0, retval: str = 'slope', **kwargs) -> pd.DataFrame:
        return rolling_regression(
            y, x, window=window, min_periods=3, lag=lag, retval=retval
        )

# aliases: TS_REGRESSION_SLOPE, ts_regression_slope



# canonical=ts_skew backend=pandas_numpy selected=m_skew source=time_series/m_ops.py
@register_operator(name="m_skew", category="time_series", business_category="time_series", canonical="ts_skew", source="factor_dsl_np")
class MovingSkew(SeriesOperator):
    """移动偏度"""

    metadata = OperatorMetadata(
        name="m_skew",
        category="time_series",
        description="计算n期移动偏度",
        examples=["m_skew(returns, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "moving", "skewness"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).skew()

# aliases: TS_SKEW



# canonical=ts_std backend=pandas_numpy selected=ts_std source=time_series/ts_ops.py

# helper for ts_std
class TSStdDev(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_std_dev", category="time_series",
        description="滚动标准差 (与m_std相同)",
        examples=["ts_std_dev(returns, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "std"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).std()

@register_operator(name="ts_std", category="time_series", business_category="time_series", canonical="ts_std", source="factor_dsl_np")
class TSStd(TSStdDev):
    metadata = OperatorMetadata(
        name="ts_std", category="time_series",
        description="滚动标准差 (ts_std_dev的别名)",
        examples=["ts_std(returns, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "std"]
    )

# aliases: Std, TS_STD, m_std, std, ts_std_dev, ts_stddev



# canonical=ts_sum backend=pandas_numpy selected=ts_sum source=time_series/ts_ops.py
@register_operator(name="ts_sum", category="time_series", business_category="time_series", canonical="ts_sum", source="factor_dsl_np")
class TSSum(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_sum", category="time_series",
        description="滚动求和 (与m_sum相同)",
        examples=["ts_sum(volume, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "sum"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).sum()

# aliases: TS_SUM, m_sum



# canonical=ts_sum_decay backend=pandas_numpy selected=ts_sum_decay source=time_series/ts_ops.py
@register_operator(name="ts_sum_decay", category="time_series", business_category="time_series", canonical="ts_sum_decay", source="factor_dsl_np")
class TSSumDecay(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_sum_decay", category="time_series",
        description="衰减求和",
        examples=["ts_sum_decay(returns, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "decay", "sum"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        weights = np.array([2 ** (i / window) for i in range(window)])
        weights = weights / weights.sum()
        return x.rolling(window=window, min_periods=1).apply(
            lambda s: np.dot(s, weights[:len(s)]) / weights[:len(s)].sum() if len(s) > 0 else np.nan,
            raw=True
        )



# canonical=ts_zscore backend=pandas_numpy selected=ts_zscore source=time_series/ts_ops.py
@register_operator(name="ts_zscore", category="time_series", business_category="time_series", canonical="ts_zscore", source="factor_dsl_np")
class TSZScore(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_zscore", category="time_series",
        description="滚动Z-Score (与m_zscore相同)",
        examples=["ts_zscore(close, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "zscore"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        mean = x.rolling(window=window, min_periods=1).mean()
        std = x.rolling(window=window, min_periods=1).std().replace(0, 1)
        return (x - mean) / std


try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore
from cleaned_operators.base_polars import (
    Operator as PolarsOperator,
    OperatorMetadata as PolarsOperatorMetadata,
    SeriesOperator as PolarsSeriesOperator,
    register_operator as register_polars_operator,
    apply_numba_rolling,
    apply_numba_zscore,
    apply_numba_rank,
)
# polars blocks below reuse names Operator/SeriesOperator/register_operator via aliases
Operator = PolarsOperator
OperatorMetadata = PolarsOperatorMetadata
SeriesOperator = PolarsSeriesOperator
register_operator = register_polars_operator


# canonical=decay_linear backend=polars selected=ts_decay_linear source=time_series/ts_ops_polars.py
@register_operator(name="ts_decay_linear", category="time_series", business_category="time_series", canonical="decay_linear", source="factor_dsl_np")
class TSDecayLinearPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_decay_linear", category="time_series",
        description="线性加权滚动平均",
        examples=["ts_decay_linear(close, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "decay", "linear"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        weights = np.arange(1, window + 1, dtype=float)
        weights = weights / weights.sum()

        def linear_decay(s):
            if len(s) == 0:
                return None
            w = weights[:len(s)]
            return np.dot(s, w) / w.sum()

        return x.with_columns([
            pl.col(c).rolling_map(linear_decay, window_size=window).alias(c)
            for c in numeric_cols
        ])

# aliases: DECAY_LINEAR, TS_DECAY_LINEAR



# canonical=ts_corr backend=polars selected=ts_corr source=time_series/ts_ops_polars.py

# helper for ts_corr
class TSCorrelation(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_correlation", category="time_series",
        description="滚动相关系数 (与m_cor相同)",
        examples=["ts_correlation(close, volume, 20)"],
        param_names=["x", "y", "window"], return_type="series",
        tags=["time_series", "ts_", "correlation"]
    )
    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        result = x.select(['date'] if 'date' in x.columns else [])

        for col in numeric_cols:
            if col in y.columns:
                x_col = x[col]
                y_col = y[col]
                corr = x_col.rolling_corr(y_col, window_size=window, min_periods=2)
                result = result.with_columns([corr.alias(col)])

        return result

@register_operator(name="ts_corr", category="time_series", business_category="time_series", canonical="ts_corr", source="factor_dsl_np")
class TSCorrPolars(TSCorrelation):
    metadata = OperatorMetadata(
        name="ts_corr", category="time_series",
        description="滚动相关系数 (ts_correlation的别名)",
        examples=["ts_corr(close, volume, 20)"],
        param_names=["x", "y", "window"], return_type="series",
        tags=["time_series", "ts_", "corr"]
    )

# aliases: TS_CORR, correlation, m_cor, ts_correlation



# canonical=ts_cov backend=polars selected=ts_cov source=time_series/ts_ops_polars.py
@register_operator(name="ts_cov", category="time_series", business_category="time_series", canonical="ts_cov", source="factor_dsl_np")
class TSCovPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_cov", category="time_series",
        description="滚动协方差 (与m_cov相同)",
        examples=["ts_cov(returns, market, 20)"],
        param_names=["x", "y", "window"], return_type="series",
        tags=["time_series", "ts_", "cov"]
    )
    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        result = x.select(['date'] if 'date' in x.columns else [])

        for col in numeric_cols:
            if col in y.columns:
                x_col = x[col]
                y_col = y[col]
                cov = x_col.rolling_cov(y_col, window_size=window, min_periods=2)
                result = result.with_columns([cov.alias(col)])

        return result

# aliases: TS_COV, m_cov, ts_covariance



# canonical=ts_decay backend=polars selected=ts_decay source=time_series/ts_ops_polars.py
@register_operator(name="ts_decay", category="time_series", business_category="time_series", canonical="ts_decay", source="factor_dsl_np")
class TSDecayPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_decay", category="time_series",
        description="衰减操作 (与ts_decay_linear相同)",
        examples=["ts_decay(close, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "decay"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        return TSDecayLinearPolars()._calculate_series(x, window, **kwargs)



# canonical=ts_decay_exp_window backend=polars selected=ts_decay_exp_window source=time_series/ts_ops_polars.py
@register_operator(name="ts_decay_exp_window", category="time_series", business_category="time_series", canonical="ts_decay_exp_window", source="factor_dsl_np")
class TSDecayExpWindowPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_decay_exp_window", category="time_series",
        description="指数加权滚动",
        examples=["ts_decay_exp_window(volume, 10, 0.5)"],
        param_names=["x", "window", "alpha"], return_type="series",
        tags=["time_series", "ts_", "decay", "exp"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 10, alpha: float = 0.5, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        weights = np.array([alpha ** i for i in range(window)][::-1], dtype=np.float64)
        weights = weights / weights.sum()

        def exp_decay(s):
            if len(s) == 0:
                return None
            w = weights[-len(s):]
            return float(np.dot(s, w) / w.sum())

        return x.with_columns([
            pl.col(c).rolling_map(exp_decay, window_size=window, min_periods=1).alias(c)
            for c in numeric_cols
        ])



# canonical=ts_delay backend=polars selected=ts_delay source=time_series/ts_ops_polars.py
@register_operator(name="ts_delay", category="time_series", business_category="time_series", canonical="ts_delay", source="factor_dsl_np")
class TSDelayPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_delay", category="time_series",
        description="n期滞后 (与Ref相同)",
        examples=["ts_delay(close, 5)"],
        param_names=["x", "n"], return_type="series",
        tags=["time_series", "ts_", "delay"]
    )
    def _calculate_series(self, x: pl.DataFrame, n: int = 1, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        lag = int(n)
        if lag < 0:
            return x.with_columns([pl.lit(None).cast(pl.Float64).alias(c) for c in numeric_cols])
        return x.with_columns([
            pl.col(c).shift(lag).alias(c) for c in numeric_cols
        ])

# aliases: DELAY, Delay, Ref, delay, m_delay, shift



# canonical=ts_delta backend=polars selected=ts_delta source=time_series/ts_ops_polars.py
@register_operator(name="ts_delta", category="time_series", business_category="time_series", canonical="ts_delta", source="factor_dsl_np")
class TSDeltaPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_delta", category="time_series",
        description="n期差分 (与Delta相同)",
        examples=["ts_delta(close, 1)"],
        param_names=["x", "n"], return_type="series",
        tags=["time_series", "ts_", "delta"]
    )
    def _calculate_series(self, x: pl.DataFrame, n: int = 1, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        lag = int(n)
        if lag < 0:
            return x.with_columns([pl.lit(None).cast(pl.Float64).alias(c) for c in numeric_cols])
        return x.with_columns([
            (pl.col(c) - pl.col(c).shift(lag)).alias(c) for c in numeric_cols
        ])

# aliases: Delta, Diff, TS_DELTA, pct_change



# canonical=ts_kurt backend=polars selected=ts_kurtosis source=time_series/ts_ops_polars.py
@register_operator(name="ts_kurtosis", category="time_series", business_category="time_series", canonical="ts_kurt", source="factor_dsl_np")
class TSKurtosisPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_kurtosis", category="time_series",
        description="滚动峰度 (与m_kurt相同)",
        examples=["ts_kurtosis(returns, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "kurtosis"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        return x.with_columns([
            pl.col(c).rolling_map(lambda s: s.kurtosis() if len(s) > 3 else None, window_size=window).alias(c)
            for c in numeric_cols
        ])

# aliases: TS_KURT



# canonical=ts_max backend=polars selected=ts_max source=time_series/ts_ops_polars.py
@register_operator(name="ts_max", category="time_series", business_category="time_series", canonical="ts_max", source="factor_dsl_np")
class TSMaxPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_max", category="time_series",
        description="滚动最大值 (与m_max相同)",
        examples=["ts_max(high, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "max"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        # 使用 Numba 加速
        return apply_numba_rolling(x, window, 'max')

# aliases: Max, TS_MAX, m_max, max



# canonical=ts_mean backend=polars selected=ts_mean source=time_series/ts_ops_polars.py
@register_operator(name="ts_mean", category="time_series", business_category="time_series", canonical="ts_mean", source="factor_dsl_np")
class TSMeanPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_mean", category="time_series",
        description="滚动均值 (与m_avg相同)",
        examples=["ts_mean(close, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "mean"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        # 使用 Numba 加速
        return apply_numba_rolling(x, window, 'mean')

# aliases: Mean, TS_MEAN, m_avg, mean



# canonical=ts_min backend=polars selected=ts_min source=time_series/ts_ops_polars.py
@register_operator(name="ts_min", category="time_series", business_category="time_series", canonical="ts_min", source="factor_dsl_np")
class TSMinPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_min", category="time_series",
        description="滚动最小值 (与m_min相同)",
        examples=["ts_min(low, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "min"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        # 使用 Numba 加速
        return apply_numba_rolling(x, window, 'min')

# aliases: Min, TS_MIN, m_min, min



# canonical=ts_product backend=polars selected=ts_product source=time_series/ts_ops_polars.py
@register_operator(name="ts_product", category="time_series", business_category="time_series", canonical="ts_product", source="factor_dsl_np")
class TSProductPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_product", category="time_series",
        description="滚动乘积",
        examples=["ts_product(volume + 1, 5)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "product"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 5, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]

        def rolling_prod(s):
            if len(s) == 0:
                return None
            return np.prod(s)

        return x.with_columns([
            pl.col(c).rolling_map(rolling_prod, window_size=window).alias(c)
            for c in numeric_cols
        ])



# canonical=ts_rank backend=polars selected=ts_rank source=time_series/ts_ops_polars.py
@register_operator(name="ts_rank", category="time_series", business_category="time_series", canonical="ts_rank", source="factor_dsl_np")
class TSRankPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_rank", category="time_series",
        description="滚动排名 (与m_rank相同)",
        examples=["ts_rank(volume, 10)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "rank"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        # 使用 Polars 的 rolling_apply
        def _rolling_rank_pct(s):
            if len(s) == 0:
                return None
            ranks = s.rank(method="average")
            return float(ranks[-1] / len(s))

        return x.with_columns([
            pl.col(c).rolling_map(_rolling_rank_pct, window_size=window, min_periods=1).alias(c)
            for c in numeric_cols
        ])

# aliases: TS_RANK, m_rank



# canonical=ts_skew backend=polars selected=ts_skewness source=time_series/ts_ops_polars.py
@register_operator(name="ts_skewness", category="time_series", business_category="time_series", canonical="ts_skew", source="factor_dsl_np")
class TSSkewnessPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_skewness", category="time_series",
        description="滚动偏度 (与m_skew相同)",
        examples=["ts_skewness(returns, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "skewness"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        # 使用 Polars 的 rolling_skew（如果可用）或手动计算
        return x.with_columns([
            pl.col(c).rolling_map(lambda s: s.skew() if len(s) > 2 else None, window_size=window).alias(c)
            for c in numeric_cols
        ])

# aliases: TS_SKEW



# canonical=ts_std backend=polars selected=ts_std source=time_series/ts_ops_polars.py

# helper for ts_std
class TSStdDev(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_std_dev", category="time_series",
        description="滚动标准差 (与m_std相同)",
        examples=["ts_std_dev(returns, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "std"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        # 使用 Numba 加速
        return apply_numba_rolling(x, window, 'std')

@register_operator(name="ts_std", category="time_series", business_category="time_series", canonical="ts_std", source="factor_dsl_np")
class TSStdPolars(TSStdDev):
    metadata = OperatorMetadata(
        name="ts_std", category="time_series",
        description="滚动标准差 (ts_std_dev的别名)",
        examples=["ts_std(returns, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "std"]
    )

# aliases: Std, TS_STD, m_std, std, ts_std_dev, ts_stddev



# canonical=ts_sum backend=polars selected=ts_sum source=time_series/ts_ops_polars.py
@register_operator(name="ts_sum", category="time_series", business_category="time_series", canonical="ts_sum", source="factor_dsl_np")
class TSSumPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_sum", category="time_series",
        description="滚动求和 (与m_sum相同)",
        examples=["ts_sum(volume, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "sum"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        # 使用 Numba 加速
        return apply_numba_rolling(x, window, 'sum')

# aliases: TS_SUM, m_sum



# canonical=ts_sum_decay backend=polars selected=ts_sum_decay source=time_series/ts_ops_polars.py
@register_operator(name="ts_sum_decay", category="time_series", business_category="time_series", canonical="ts_sum_decay", source="factor_dsl_np")
class TSSumDecayPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_sum_decay", category="time_series",
        description="衰减求和",
        examples=["ts_sum_decay(returns, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "decay", "sum"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        weights = np.array([2 ** (i / window) for i in range(window)])
        weights = weights / weights.sum()

        def decay_sum(s):
            if len(s) == 0:
                return None
            w = weights[:len(s)]
            return np.dot(s, w) / w.sum()

        return x.with_columns([
            pl.col(c).rolling_map(decay_sum, window_size=window).alias(c)
            for c in numeric_cols
        ])



# canonical=ts_zscore backend=polars selected=ts_zscore source=time_series/ts_ops_polars.py
@register_operator(name="ts_zscore", category="time_series", business_category="time_series", canonical="ts_zscore", source="factor_dsl_np")
class TSZScorePolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_zscore", category="time_series",
        description="滚动Z-Score (与m_zscore相同)",
        examples=["ts_zscore(close, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "zscore"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        # 使用 Polars 的 rolling_mean 和 rolling_std
        return x.with_columns([
            ((pl.col(c) - pl.col(c).rolling_mean(window_size=window, min_periods=1)) /
             pl.when(pl.col(c).rolling_std(window_size=window, min_periods=1) == 0)
             .then(1)
             .otherwise(pl.col(c).rolling_std(window_size=window, min_periods=1))
            ).alias(c)
            for c in numeric_cols
        ])


@register_operator(name="ts_pct", category="time_series", business_category="time_series", canonical="ts_pct", source="lqtp_numpy")
class LqtpTspctOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_pct",
        category="time_series",
        description="LQTP numpy implementation",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._lqtp_numpy import ts_pct_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: ts_pct_(s.values, **kwargs) if kwargs else ts_pct_(s.values))
        return ts_pct_(*args, **kwargs)


@register_operator(name="price_spread_deviation", category="time_series", business_category="time_series", canonical="price_spread_deviation", source="lqtp_numpy")
class LqtpPricespreaddeviationOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="price_spread_deviation",
        category="time_series",
        description="LQTP numpy implementation",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._lqtp_numpy import price_spread_deviation_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: price_spread_deviation_(s.values, **kwargs) if kwargs else price_spread_deviation_(s.values))
        return price_spread_deviation_(*args, **kwargs)


@register_operator(name="ts_quantile", category="time_series", business_category="time_series", canonical="ts_quantile", source="lqtp_numpy")
class LqtpTsquantileOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_quantile",
        category="time_series",
        description="LQTP numpy implementation",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._lqtp_numpy import ts_quantile_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: ts_quantile_(s.values, **kwargs) if kwargs else ts_quantile_(s.values))
        return ts_quantile_(*args, **kwargs)


@register_operator(name="ts_moment", category="time_series", business_category="time_series", canonical="ts_moment", source="lqtp_numpy")
class LqtpTsmomentOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_moment",
        category="time_series",
        description="LQTP numpy implementation",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._lqtp_numpy import ts_moment_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: ts_moment_(s.values, **kwargs) if kwargs else ts_moment_(s.values))
        return ts_moment_(*args, **kwargs)


@register_operator(name="ts_topk_sum", category="time_series", business_category="time_series", canonical="ts_topk_sum", source="lqtp_numpy")
class LqtpTstopksumOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_topk_sum",
        category="time_series",
        description="LQTP numpy implementation",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._lqtp_numpy import ts_topk_sum_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: ts_topk_sum_(s.values, **kwargs) if kwargs else ts_topk_sum_(s.values))
        return ts_topk_sum_(*args, **kwargs)


@register_operator(name="rank_corr", category="time_series", business_category="time_series", canonical="rank_corr", source="lqtp_numpy")
class LqtpRankcorrOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="rank_corr",
        category="time_series",
        description="LQTP numpy implementation",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._lqtp_numpy import rank_corr_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: rank_corr_(s.values, **kwargs) if kwargs else rank_corr_(s.values))
        return rank_corr_(*args, **kwargs)


@register_operator(name="ts_poly2_coeff", category="time_series", business_category="time_series", canonical="ts_poly2_coeff", source="lqtp_numpy")
class LqtpTspoly2CoeffOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_poly2_coeff",
        category="time_series",
        description="LQTP numpy implementation",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._lqtp_numpy import ts_poly2_coeff_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: ts_poly2_coeff_(s.values, **kwargs) if kwargs else ts_poly2_coeff_(s.values))
        return ts_poly2_coeff_(*args, **kwargs)


@register_operator(name="ts_poly2_resid", category="time_series", business_category="time_series", canonical="ts_poly2_resid", source="lqtp_numpy")
class LqtpTspoly2ResidOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_poly2_resid",
        category="time_series",
        description="LQTP numpy implementation",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._lqtp_numpy import ts_poly2_resid_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: ts_poly2_resid_(s.values, **kwargs) if kwargs else ts_poly2_resid_(s.values))
        return ts_poly2_resid_(*args, **kwargs)


@register_operator(name="digital_count", category="time_series", business_category="time_series", canonical="digital_count", source="lqtp_numpy")
class LqtpDigitalcountOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="digital_count",
        category="time_series",
        description="LQTP numpy implementation",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._lqtp_numpy import digital_count_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: digital_count_(s.values, **kwargs) if kwargs else digital_count_(s.values))
        return digital_count_(*args, **kwargs)


@register_operator(name="ts_max_buildup", category="time_series", business_category="time_series", canonical="ts_max_buildup", source="lqtp_numpy")
class LqtpTsmaxbuildupOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_max_buildup",
        category="time_series",
        description="LQTP numpy implementation",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._lqtp_numpy import ts_max_buildup_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: ts_max_buildup_(s.values, **kwargs) if kwargs else ts_max_buildup_(s.values))
        return ts_max_buildup_(*args, **kwargs)


@register_operator(name="ts_argmax", category="time_series", business_category="time_series", canonical="ts_argmax", source="lqtp_numpy")
class LqtpTsargmaxOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_argmax",
        category="time_series",
        description="LQTP numpy implementation",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._lqtp_numpy import ts_argmax_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: ts_argmax_(s.values, **kwargs) if kwargs else ts_argmax_(s.values))
        return ts_argmax_(*args, **kwargs)


@register_operator(name="ts_argmin", category="time_series", business_category="time_series", canonical="ts_argmin", source="lqtp_numpy")
class LqtpTsargminOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_argmin",
        category="time_series",
        description="LQTP numpy implementation",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._lqtp_numpy import ts_argmin_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: ts_argmin_(s.values, **kwargs) if kwargs else ts_argmin_(s.values))
        return ts_argmin_(*args, **kwargs)
