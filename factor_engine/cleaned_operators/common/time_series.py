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

import os

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

# 重复实现：见 ts_mean；dedupe 注销
# @register_operator(name="SMA", category="time_series", business_category="time_series", canonical="SMA", source="factor_dsl_np")
class SMA(SeriesOperator):
    """简单移动平均（SMA）；dedupe 后别名指向 ``ts_mean``。"""

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
    """加权移动平均（线性衰减权重）。"""

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
    """自定义 Top-N 截面聚合算子（按排序列选取前 N 标的聚合）。"""

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
    """扩展窗口内前 N 大值的均值。"""

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
    """扩展窗口内前 N 大值的求和。"""

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
@register_operator(name="ts_decay_linear", category="time_series", business_category="time_series", canonical="ts_decay_linear", source="factor_dsl_np")
class TSDecayLinear(SeriesOperator):
    """线性衰减加权滚动平均（``ts_decay_linear`` canonical）。"""

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
@register_operator(name="EMA", category="time_series", business_category="time_series", canonical="ts_ema", source="factor_dsl_np")
class EMA(SeriesOperator):
    """指数移动平均（EWM，``adjust=False``）。"""

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



# canonical=ts_argmax backend=pandas_numpy selected=ts_argmax source=time_series/m_ops.py
@register_operator(name="ts_argmax", category="time_series", business_category="time_series", canonical="ts_argmax", source="factor_dsl_np")
class TSArgmax(SeriesOperator):
    """滚动窗口内最大值位置（0=窗口内最新 bar）。"""

    metadata = OperatorMetadata(
        name="ts_argmax",
        category="time_series",
        description="返回窗口内最大值的位置",
        examples=["ts_argmax(close, 20)"],
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "argmax"]
    )

    def _calculate_series(self, x: pd.DataFrame, d: int = 20, **kwargs) -> pd.DataFrame:
        from cleaned_operators._rolling_fast import rolling_argmax

        window = int(kwargs.get("window", d))
        return rolling_argmax(x, window)



# canonical=ts_argmin backend=pandas_numpy selected=ts_argmin source=time_series/m_ops.py
@register_operator(name="ts_argmin", category="time_series", business_category="time_series", canonical="ts_argmin", source="factor_dsl_np")
class TSArgmin(SeriesOperator):
    """滚动窗口内最小值位置（0=窗口内最新 bar）。"""

    metadata = OperatorMetadata(
        name="ts_argmin",
        category="time_series",
        description="返回窗口内最小值的位置",
        examples=["ts_argmin(close, 20)"],
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "argmin"]
    )

    def _calculate_series(self, x: pd.DataFrame, d: int = 20, **kwargs) -> pd.DataFrame:
        from cleaned_operators._rolling_fast import rolling_argmin

        window = int(kwargs.get("window", d))
        return rolling_argmin(x, window)



# canonical=m_beta backend=pandas_numpy selected=m_beta source=time_series/m_ops.py
@register_operator(name="m_beta", category="time_series", business_category="time_series", canonical="ts_beta", source="factor_dsl_np")
class MovingBeta(SeriesOperator):
    """两变量滚动 Beta：``Cov(y,x)/Var(x)``。"""

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
@register_operator(name="m_bottom_n_avg", category="time_series", business_category="time_series", canonical="ts_bottom_n_avg", source="factor_dsl_np")
class MovingBottomNAvg(SeriesOperator):
    """滚动窗口内后 N 小值的均值。"""

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
@register_operator(name="m_bottom_n_sum", category="time_series", business_category="time_series", canonical="ts_bottom_n_sum", source="factor_dsl_np")
class MovingBottomNSum(SeriesOperator):
    """滚动窗口内后 N 小值的求和。"""

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
@register_operator(name="m_mad", category="time_series", business_category="time_series", canonical="ts_mad", source="factor_dsl_np")
class MovingMAD(SeriesOperator):
    """滚动平均绝对离差（MAD）。"""

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
@register_operator(name="m_median", category="time_series", business_category="time_series", canonical="ts_median", source="factor_dsl_np")
class MovingMedian(SeriesOperator):
    """滚动中位数。"""

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



# canonical=ts_pct backend=pandas_numpy selected=ts_pct source=time_series/m_ops.py
@register_operator(name="ts_pct", category="time_series", business_category="time_series", canonical="ts_pct", source="factor_dsl_np")
class TSPctChange(SeriesOperator):
    """d 期变化率：``x_t / x_{t-d} - 1``。"""

    metadata = OperatorMetadata(
        name="ts_pct",
        category="time_series",
        description="d 期变化率",
        examples=["ts_pct(close, 1)"],
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "pct_change", "pit_safe"]
    )

    def _calculate_series(self, x: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
        periods = int(kwargs.get("periods", d))
        prev = x.shift(periods)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = x / prev - 1.0
        return out.where(prev.notna() & (prev != 0))



# canonical=ts_log_return backend=pandas_numpy selected=ts_log_return source=time_series/m_ops.py
@register_operator(name="ts_log_return", category="time_series", business_category="time_series", canonical="ts_log_return", source="factor_dsl_np")
class TSLogReturn(SeriesOperator):
    """d 期对数收益率：``ln(x_t / x_{t-d})``。"""

    metadata = OperatorMetadata(
        name="ts_log_return",
        category="time_series",
        description="d 期对数收益率 ln(x_t / x_{t-d})",
        examples=["ts_log_return(close, 1)"],
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "returns", "log", "pit_safe"],
    )

    def _calculate_series(self, x: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
        n = max(1, int(kwargs.get("periods", d)))
        prev = x.shift(n)
        ratio = x / prev.replace(0, np.nan)
        return np.log(ratio.replace([np.inf, -np.inf], np.nan))



# canonical=ts_sharpe backend=pandas_numpy selected=ts_sharpe source=time_series/m_ops.py
@register_operator(name="ts_sharpe", category="time_series", business_category="time_series", canonical="ts_sharpe", source="factor_dsl_np")
class TSSharpe(SeriesOperator):
    """滚动夏普比率（年化，可配置 ``ann_factor``）。"""

    metadata = OperatorMetadata(
        name="ts_sharpe",
        category="time_series",
        description="滚动夏普：mean/std * sqrt(ann_factor)",
        examples=["ts_sharpe(returns, 60)"],
        param_names=["x", "window", "ann_factor"],
        return_type="series",
        tags=["time_series", "sharpe", "pit_safe"],
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 60,
        ann_factor: float = 252.0,
        min_periods: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        w = max(2, int(window))
        mp = max(2, int(min_periods)) if min_periods is not None else max(2, w // 3)
        mean = x.rolling(window=w, min_periods=mp).mean()
        std = x.rolling(window=w, min_periods=mp).std(ddof=1)
        zero_vol = std.eq(0) | std.isna()
        sharpe = mean / std.replace(0, np.nan)
        sharpe = sharpe.mask(zero_vol & mean.gt(0), np.inf)
        sharpe = sharpe.mask(zero_vol & mean.le(0), 0.0)
        return sharpe * np.sqrt(float(ann_factor))


# canonical=ts_autocorr backend=pandas_numpy selected=ts_autocorr source=time_series/m_ops.py
@register_operator(
    name="ts_autocorr",
    category="time_series",
    business_category="time_series",
    canonical="ts_autocorr",
    source="factor_dsl_np",
)
class TSAutocorr(SeriesOperator):
    """滚动自相关系数：窗口内 ``corr(x, x.shift(lag))``。"""

    metadata = OperatorMetadata(
        name="ts_autocorr",
        category="time_series",
        description="滚动自相关 corr(x_t, x_{t-lag}) within window",
        examples=["ts_autocorr(returns, 20, 1)"],
        param_names=["x", "window", "lag"],
        return_type="series",
        tags=["time_series", "autocorrelation", "pit_safe"],
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 20,
        lag: int = 1,
        min_periods: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        k = max(1, int(lag))
        w = max(k + 2, int(window))
        mp = max(k + 2, int(min_periods)) if min_periods is not None else max(2, w // 3)
        if int(lag) < 1:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)
        y = x.shift(k)
        return x.rolling(window=w, min_periods=mp).corr(y)


# canonical=ts_quantile backend=pandas_numpy selected=ts_quantile source=time_series/m_ops.py
@register_operator(name="ts_quantile", category="time_series", business_category="time_series", canonical="ts_quantile", source="factor_dsl_np")
class TSQuantile(SeriesOperator):
    """滚动窗口分位数 ``Q_q``。"""

    metadata = OperatorMetadata(
        name="ts_quantile",
        category="time_series",
        description="滚动窗口分位数",
        examples=["ts_quantile(returns, 20, 0.75)"],
        param_names=["x", "d", "q"],
        return_type="series",
        tags=["time_series", "moving", "percentile"]
    )

    def _calculate_series(self, x: pd.DataFrame, d: int = 20, q: float = 0.5, **kwargs) -> pd.DataFrame:
        window = int(kwargs.get("window", d))
        quantile = float(kwargs.get("p", q))
        return x.rolling(window=window, min_periods=1).quantile(quantile)



# canonical=m_top_n_avg backend=pandas_numpy selected=m_top_n_avg source=time_series/topn_ops.py
@register_operator(name="m_top_n_avg", category="time_series", business_category="time_series", canonical="ts_top_n_avg", source="factor_dsl_np")
class MovingTopNAvg(SeriesOperator):
    """滚动窗口内前 N 大值的均值。"""

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
@register_operator(name="m_top_n_std", category="time_series", business_category="time_series", canonical="ts_top_n_std", source="factor_dsl_np")
class MovingTopNStd(SeriesOperator):
    """滚动窗口内前 N 大值的标准差。"""

    metadata = OperatorMetadata(
        name="m_top_n_std", category="time_series",
        description="滚动窗口内前N大值的标准差",
        examples=["m_top_n_std(volume, 5)"],
        param_names=["x", "n"], return_type="series",
        tags=["time_series", "top_n", "std"]
    )
    def _calculate_series(self, x: pd.DataFrame, n: int = 5, **kwargs) -> pd.DataFrame:
        return rolling_top_n_std(x, n)



# canonical=ts_topk_sum backend=pandas_numpy selected=ts_topk_sum source=time_series/topn_ops.py
@register_operator(name="ts_topk_sum", category="time_series", business_category="time_series", canonical="ts_topk_sum", source="factor_dsl_np")
class TSTopKSum(SeriesOperator):
    """滚动窗口内 Top-K 求和。"""

    metadata = OperatorMetadata(
        name="ts_topk_sum", category="time_series",
        description="滚动窗口内 Top-K 求和",
        examples=["ts_topk_sum(volume, 20, 5)"],
        param_names=["x", "d", "k"], return_type="series",
        tags=["time_series", "top_n", "sum"]
    )
    def _calculate_series(self, x: pd.DataFrame, d: int = 20, k: int | None = None, **kwargs) -> pd.DataFrame:
        from cleaned_operators._rolling_fast import rolling_top_n_sum_window

        window = int(kwargs.get("window", d))
        top_k = int(k if k is not None else kwargs.get("n", window))
        return rolling_top_n_sum_window(x, window, top_k)



# canonical=m_var backend=pandas_numpy selected=m_var source=time_series/m_ops.py
@register_operator(name="m_var", category="time_series", business_category="time_series", canonical="ts_var", source="factor_dsl_np")
class MovingVariance(SeriesOperator):
    """滚动方差。"""

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



# 重复实现：见 ts_zscore；dedupe 注销
# @register_operator(name="m_zscore", category="time_series", business_category="time_series", canonical="m_zscore", source="factor_dsl_np")
class MovingZscore(SeriesOperator):
    """滚动窗口 Z-Score 标准化（dedupe 后别名指向 ``ts_zscore``）。"""

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
    """指定时间窗口内前 N 大值的均值。"""

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
    """指定时间窗口内前 N 大值的求和。"""

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
    """滚动 Pearson 相关系数（``ts_corr`` 基类）。"""

    metadata = OperatorMetadata(
        name="ts_correlation", category="time_series",
        description="滚动相关系数 (与m_cor相同)",
        examples=["ts_correlation(close, volume, 20)"],
        param_names=["x", "y", "window"], return_type="series",
        tags=["time_series", "ts_", "correlation"]
    )
    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        use_numba = os.environ.get("FACTOR_ENGINE_USE_NUMBA", "").lower() in (
            "1",
            "true",
            "yes",
        )
        if use_numba:
            try:
                from backend.numba_kernels import rolling_corr_panel

                fast = rolling_corr_panel(
                    x.to_numpy(dtype=float),
                    y.to_numpy(dtype=float),
                    int(window),
                    min_count=2,
                )
                if fast is not None:
                    return pd.DataFrame(fast, index=x.index, columns=x.columns)
            except Exception:
                pass
        return x.rolling(window=window, min_periods=2).corr(y)

@register_operator(name="ts_corr", category="time_series", business_category="time_series", canonical="ts_corr", source="factor_dsl_np")
class TSCorr(TSCorrelation):
    """滚动相关系数（``ts_corr`` canonical）。"""

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
    """滚动协方差。"""

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



# 重复实现：见 ts_decay_linear；dedupe 注销
# @register_operator(name="ts_decay", ...)
class TSDecay(SeriesOperator):
    """线性衰减加权滚动（``ts_decay_linear`` 别名，dedupe 注销）。"""

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
    """指数加权滚动平均。"""

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
    """n 期因果滞后（PIT-safe，负滞后返回 NaN）。"""

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
    """n 期差分：``x - lag(x, n)``。"""

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
    """滚动峰度。"""

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
    """滚动最大值。"""

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
    """滚动均值（支持 Numba 加速路径）。"""

    metadata = OperatorMetadata(
        name="ts_mean", category="time_series",
        description="滚动均值 (与m_avg相同)",
        examples=["ts_mean(close, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "mean"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        from backend.routing import numba_enabled_for_op

        if numba_enabled_for_op("ts_mean", window=int(window)):
            try:
                from backend.numba_kernels import rolling_mean_panel

                fast = rolling_mean_panel(x.to_numpy(dtype=float), int(window), min_count=1)
                if fast is not None:
                    return pd.DataFrame(fast, index=x.index, columns=x.columns)
            except Exception:
                pass
        return x.rolling(window=window, min_periods=1).mean()

# aliases: Mean, TS_MEAN, m_avg, mean



# canonical=ts_min backend=pandas_numpy selected=ts_min source=time_series/ts_ops.py
@register_operator(name="ts_min", category="time_series", business_category="time_series", canonical="ts_min", source="factor_dsl_np")
class TSMin(SeriesOperator):
    """滚动最小值。"""

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
    """滚动乘积（对数域累加实现）。"""

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
    """滚动百分位排名（支持 Numba 加速路径）。"""

    metadata = OperatorMetadata(
        name="ts_rank", category="time_series",
        description="滚动排名 (与m_rank相同)",
        examples=["ts_rank(volume, 10)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "rank"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        from backend.routing import numba_enabled_for_op

        if numba_enabled_for_op("ts_rank", window=int(window)):
            try:
                from backend.numba_kernels import rolling_rank_pct_panel

                fast = rolling_rank_pct_panel(x.to_numpy(dtype=float), int(window), min_count=1)
                if fast is not None:
                    return pd.DataFrame(fast, index=x.index, columns=x.columns)
            except Exception:
                pass
        return x.rolling(window=window, min_periods=1).rank(pct=True)

# aliases: TS_RANK, m_rank



# canonical=ts_regression backend=pandas_numpy selected=ts_regression source=time_series/ts_ops.py
@register_operator(name="ts_regression", category="time_series", business_category="time_series", canonical="ts_regression", source="factor_dsl_np")
class TSRegression(SeriesOperator):
    """滚动 OLS 回归（slope/intercept/r²/residual）。"""

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
    """滚动偏度。"""

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
    """滚动标准差（``ts_std`` 基类）。"""

    metadata = OperatorMetadata(
        name="ts_std_dev", category="time_series",
        description="滚动标准差 (与m_std相同)",
        examples=["ts_std_dev(returns, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "std"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        from backend.routing import numba_enabled_for_op

        if numba_enabled_for_op("ts_std", window=int(window)):
            try:
                from backend.numba_kernels import rolling_std_panel

                fast = rolling_std_panel(x.to_numpy(dtype=float), int(window), min_count=1)
                if fast is not None:
                    return pd.DataFrame(fast, index=x.index, columns=x.columns)
            except Exception:
                pass
        return x.rolling(window=window, min_periods=1).std()

@register_operator(name="ts_std", category="time_series", business_category="time_series", canonical="ts_std", source="factor_dsl_np")
class TSStd(TSStdDev):
    """滚动标准差（``ts_std`` canonical）。"""

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
    """滚动求和。"""

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
    """指数衰减权重滚动求和。"""

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
    """滚动 Z-Score 标准化。"""

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


def _apply_colwise_kernel(x: pd.DataFrame, fn, **kwargs) -> pd.DataFrame:
    """对 panel 每列应用 numpy 一维内核函数。

    参数:
        x: 输入宽表 panel。
        fn: 接收一维 numpy 数组的核函数。
        **kwargs: 传给 ``fn`` 的额外关键字参数。

    返回:
        逐列计算后的 ``pd.DataFrame``。
    """
    return x.apply(lambda s: fn(s.values, **kwargs) if kwargs else fn(s.values))


@register_operator(name="price_spread_deviation", category="time_series", business_category="time_series", canonical="price_spread_deviation", source="factor_dsl_np")
class PriceSpreadDeviation(SeriesOperator):
    """相对窗口均值偏离度。"""

    metadata = OperatorMetadata(
        name="price_spread_deviation",
        category="time_series",
        description="相对窗口均值偏离: x / mean(x, d) - 1",
        examples=["price_spread_deviation(close, 20)"],
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "deviation"],
    )

    def _calculate_series(self, x: pd.DataFrame, d: int = 20, **kwargs) -> pd.DataFrame:
        from cleaned_operators._numpy_kernels import price_spread_deviation_

        window = int(kwargs.get("window", d))
        return _apply_colwise_kernel(x, price_spread_deviation_, d=window)


@register_operator(name="ts_moment", category="time_series", business_category="time_series", canonical="ts_moment", source="factor_dsl_np")
class TSMoment(SeriesOperator):
    """滚动 k 阶中心矩。"""

    metadata = OperatorMetadata(
        name="ts_moment",
        category="time_series",
        description="窗口 k 阶中心矩",
        examples=["ts_moment(close, 20, 3)"],
        param_names=["x", "d", "k"],
        return_type="series",
        tags=["time_series", "moment"],
    )

    def _calculate_series(self, x: pd.DataFrame, d: int = 20, k: int = 3, **kwargs) -> pd.DataFrame:
        from cleaned_operators._numpy_kernels import ts_moment_

        return _apply_colwise_kernel(x, ts_moment_, d=int(d), k=int(k))


@register_operator(name="rank_corr", category="time_series", business_category="time_series", canonical="rank_corr", source="factor_dsl_np")
class RankCorr(SeriesOperator):
    """秩相关系数（d=0 截面，d>0 时序窗口）。"""

    metadata = OperatorMetadata(
        name="rank_corr",
        category="time_series",
        description="秩相关系数（d=0 为截面，d>0 为时序窗口）",
        examples=["rank_corr(close, volume, 20)"],
        param_names=["x", "y", "d"],
        return_type="series",
        tags=["time_series", "correlation"],
    )

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, d: int = 0, **kwargs) -> pd.DataFrame:
        from cleaned_operators._numpy_kernels import rank_corr_

        result = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
        for col in x.columns:
            if col in y.columns:
                result[col] = rank_corr_(x[col].values, y[col].values, d=int(d))
        return result


@register_operator(name="ts_poly2_coeff", category="time_series", business_category="time_series", canonical="ts_poly2_coeff", source="factor_dsl_np")
class TSPoly2Coeff(SeriesOperator):
    """时间二次拟合二次项系数。"""

    metadata = OperatorMetadata(
        name="ts_poly2_coeff",
        category="time_series",
        description="时间二次拟合二次项系数",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "regression"],
    )

    def _calculate_series(self, x: pd.DataFrame, d: int = 20, **kwargs) -> pd.DataFrame:
        from cleaned_operators._numpy_kernels import ts_poly2_coeff_

        return _apply_colwise_kernel(x, ts_poly2_coeff_, d=int(d))


@register_operator(name="ts_poly2_resid", category="time_series", business_category="time_series", canonical="ts_poly2_resid", source="factor_dsl_np")
class TSPoly2Resid(SeriesOperator):
    """二次拟合窗口末残差。"""

    metadata = OperatorMetadata(
        name="ts_poly2_resid",
        category="time_series",
        description="二次拟合残差",
        param_names=["y", "x", "d"],
        return_type="series",
        tags=["time_series", "regression"],
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, d: int = 20, **kwargs) -> pd.DataFrame:
        from cleaned_operators._numpy_kernels import ts_poly2_resid_

        result = pd.DataFrame(np.nan, index=y.index, columns=y.columns, dtype=float)
        for col in y.columns:
            if col in x.columns:
                result[col] = ts_poly2_resid_(y[col].values, x[col].values, int(d))
        return result


@register_operator(name="digital_count", category="time_series", business_category="time_series", canonical="digital_count", source="factor_dsl_np")
class DigitalCount(SeriesOperator):
    """连续小波动片段计数。"""

    metadata = OperatorMetadata(
        name="digital_count",
        category="time_series",
        description="连续小波动片段计数",
        param_names=["x", "d", "threshold", "run"],
        return_type="series",
        tags=["time_series", "microstructure"],
    )

    def _calculate_series(
        self, x: pd.DataFrame, d: int = 20, threshold: float = 0.01, run: int = 3, **kwargs
    ) -> pd.DataFrame:
        from cleaned_operators._numpy_kernels import digital_count_

        return _apply_colwise_kernel(
            x, digital_count_, d=int(d), threshold=float(threshold), run=int(run)
        )


@register_operator(name="ts_max_buildup", category="time_series", business_category="time_series", canonical="ts_max_buildup", source="factor_dsl_np")
class TSMaxBuildup(SeriesOperator):
    """窗口内持续创新高次数。"""

    metadata = OperatorMetadata(
        name="ts_max_buildup",
        category="time_series",
        description="窗口内持续创新高次数",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "momentum"],
    )

    def _calculate_series(self, x: pd.DataFrame, d: int = 20, **kwargs) -> pd.DataFrame:
        from cleaned_operators._numpy_kernels import ts_max_buildup_

        return _apply_colwise_kernel(x, ts_max_buildup_, d=int(d))


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
@register_operator(
    name="ts_decay_linear",
    category="time_series",
    business_category="time_series",
    canonical="ts_decay_linear",
    source="factor_dsl_np",
    backend="polars",
)
class TSDecayLinearPolars(SeriesOperator):
    """Polars 线性衰减加权滚动平均。"""

    metadata = OperatorMetadata(
        name="ts_decay_linear", category="time_series",
        description="线性加权滚动平均",
        examples=["ts_decay_linear(close, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "decay", "linear"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        wlen = max(1, int(window))
        weights = np.arange(1, wlen + 1, dtype=float)

        def linear_decay(s):
            arr = np.asarray(s, dtype=float)
            valid = np.isfinite(arr)
            if not np.any(valid):
                return None
            seg = arr[valid]
            ww = weights[-len(seg):]
            return float(np.dot(seg, ww) / ww.sum())

        return x.with_columns([
            pl.col(c).rolling_map(linear_decay, window_size=wlen, min_samples=1).alias(c)
            for c in numeric_cols
        ])

# aliases: DECAY_LINEAR, TS_DECAY_LINEAR



# canonical=ts_corr backend=polars selected=ts_corr source=time_series/ts_ops_polars.py

# helper for ts_corr
class TSCorrelation(SeriesOperator):
    """滚动相关系数 (与m_cor相同)"""
    metadata = OperatorMetadata(
        name="ts_correlation", category="time_series",
        description="滚动相关系数 (与m_cor相同)",
        examples=["ts_correlation(close, volume, 20)"],
        param_names=["x", "y", "window"], return_type="series",
        tags=["time_series", "ts_", "correlation"]
    )
    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        w = max(int(window), 2)
        merged = x
        y_cols: list[str] = []
        for col in numeric_cols:
            if col in y.columns:
                yname = f"__y_{col}"
                y_cols.append(yname)
                merged = merged.with_columns(y[col].alias(yname))
        exprs = [
            pl.rolling_corr(pl.col(col), pl.col(f"__y_{col}"), window_size=w, min_samples=2).alias(col)
            for col in numeric_cols
            if col in y.columns
        ]
        if not exprs:
            return x
        return merged.with_columns(exprs).drop(y_cols)

@register_operator(name="ts_corr", category="time_series", business_category="time_series", canonical="ts_corr", source="factor_dsl_np")
class TSCorrPolars(TSCorrelation):
    """Polars 滚动相关系数。"""

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
    """Polars 滚动协方差。"""

    metadata = OperatorMetadata(
        name="ts_cov", category="time_series",
        description="滚动协方差 (与m_cov相同)",
        examples=["ts_cov(returns, market, 20)"],
        param_names=["x", "y", "window"], return_type="series",
        tags=["time_series", "ts_", "cov"]
    )
    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        w = max(int(window), 2)
        merged = x
        y_cols: list[str] = []
        for col in numeric_cols:
            if col in y.columns:
                yname = f"__y_{col}"
                y_cols.append(yname)
                merged = merged.with_columns(y[col].alias(yname))
        exprs = [
            pl.rolling_cov(pl.col(f"__y_{col}"), pl.col(col), window_size=w, min_samples=2).alias(col)
            for col in numeric_cols
            if col in y.columns
        ]
        if not exprs:
            return x
        return merged.with_columns(exprs).drop(y_cols)

# aliases: TS_COV, m_cov, ts_covariance



# canonical=ts_decay backend=polars selected=ts_decay source=time_series/ts_ops_polars.py
# 重复实现：见 ts_decay_linear；dedupe 注销
# @register_operator(name="ts_decay", ...)
class TSDecayPolars(SeriesOperator):
    """Polars 线性衰减滚动（``ts_decay_linear`` 别名）。"""

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
    """Polars 指数加权滚动。"""

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
    """Polars n 期滞后（负滞后返回 NaN）。"""

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
    """Polars n 期差分。"""

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
    """Polars 滚动峰度。"""

    metadata = OperatorMetadata(
        name="ts_kurtosis", category="time_series",
        description="滚动峰度 (与m_kurt相同)",
        examples=["ts_kurtosis(returns, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "kurtosis"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.base_polars import panel_pandas_bridge

        w = int(kwargs.get("d", window))
        return panel_pandas_bridge(x, lambda pdf: pdf.rolling(window=w, min_periods=1).kurt())

# aliases: TS_KURT



# canonical=ts_max backend=polars selected=ts_max source=time_series/ts_ops_polars.py
@register_operator(name="ts_max", category="time_series", business_category="time_series", canonical="ts_max", source="factor_dsl_np")
class TSMaxPolars(SeriesOperator):
    """Polars 滚动最大值。"""

    metadata = OperatorMetadata(
        name="ts_max", category="time_series",
        description="滚动最大值 (与m_max相同)",
        examples=["ts_max(high, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "max"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        return x.with_columns([
            pl.col(c).rolling_max(window_size=window, min_samples=1).alias(c) for c in cols
        ])

# aliases: Max, TS_MAX, m_max, max



# canonical=ts_mean backend=polars selected=ts_mean source=time_series/ts_ops_polars.py
@register_operator(name="ts_mean", category="time_series", business_category="time_series", canonical="ts_mean", source="factor_dsl_np")
class TSMeanPolars(SeriesOperator):
    """Polars 滚动均值。"""

    metadata = OperatorMetadata(
        name="ts_mean", category="time_series",
        description="滚动均值 (与m_avg相同)",
        examples=["ts_mean(close, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "mean"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        return x.with_columns([
            pl.col(c).rolling_mean(window_size=window, min_samples=1).alias(c) for c in cols
        ])

# aliases: Mean, TS_MEAN, m_avg, mean



# canonical=ts_min backend=polars selected=ts_min source=time_series/ts_ops_polars.py
@register_operator(name="ts_min", category="time_series", business_category="time_series", canonical="ts_min", source="factor_dsl_np")
class TSMinPolars(SeriesOperator):
    """Polars 滚动最小值。"""

    metadata = OperatorMetadata(
        name="ts_min", category="time_series",
        description="滚动最小值 (与m_min相同)",
        examples=["ts_min(low, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "min"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        return x.with_columns([
            pl.col(c).rolling_min(window_size=window, min_samples=1).alias(c) for c in cols
        ])

# aliases: Min, TS_MIN, m_min, min



# canonical=ts_product backend=polars selected=ts_product source=time_series/ts_ops_polars.py
@register_operator(name="ts_product", category="time_series", business_category="time_series", canonical="ts_product", source="factor_dsl_np")
class TSProductPolars(SeriesOperator):
    """Polars 滚动乘积。"""

    metadata = OperatorMetadata(
        name="ts_product", category="time_series",
        description="滚动乘积",
        examples=["ts_product(volume + 1, 5)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "product"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 5, **kwargs) -> pl.DataFrame:
        from cleaned_operators.base_polars import panel_pandas_bridge

        w = int(kwargs.get("d", window))
        return panel_pandas_bridge(
            x,
            lambda pdf: pdf.rolling(window=w, min_periods=1).apply(
                lambda arr: float(np.nanprod(arr)), raw=True
            ),
        )



# canonical=ts_rank backend=polars selected=ts_rank source=time_series/ts_ops_polars.py
@register_operator(name="ts_rank", category="time_series", business_category="time_series", canonical="ts_rank", source="factor_dsl_np")
class TSRankPolars(SeriesOperator):
    """Polars 滚动百分位排名。"""

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
    """Polars 滚动偏度。"""

    metadata = OperatorMetadata(
        name="ts_skewness", category="time_series",
        description="滚动偏度 (与m_skew相同)",
        examples=["ts_skewness(returns, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "skewness"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.base_polars import panel_pandas_bridge

        w = int(kwargs.get("d", window))
        return panel_pandas_bridge(x, lambda pdf: pdf.rolling(window=w, min_periods=1).skew())

# aliases: TS_SKEW



# canonical=ts_std backend=polars selected=ts_std source=time_series/ts_ops_polars.py

# helper for ts_std
class TSStdDev(SeriesOperator):
    """滚动标准差 (与m_std相同)"""
    metadata = OperatorMetadata(
        name="ts_std_dev", category="time_series",
        description="滚动标准差 (与m_std相同)",
        examples=["ts_std_dev(returns, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "std"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        return x.with_columns([
            pl.col(c).rolling_std(window_size=window, min_samples=1).alias(c) for c in cols
        ])

@register_operator(name="ts_std", category="time_series", business_category="time_series", canonical="ts_std", source="factor_dsl_np")
class TSStdPolars(TSStdDev):
    """Polars 滚动标准差。"""

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
    """Polars 滚动求和。"""

    metadata = OperatorMetadata(
        name="ts_sum", category="time_series",
        description="滚动求和 (与m_sum相同)",
        examples=["ts_sum(volume, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "sum"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        return x.with_columns([
            pl.col(c).rolling_sum(window_size=window, min_samples=1).alias(c) for c in cols
        ])

# aliases: TS_SUM, m_sum



# canonical=ts_sum_decay backend=polars selected=ts_sum_decay source=time_series/ts_ops_polars.py
@register_operator(name="ts_sum_decay", category="time_series", business_category="time_series", canonical="ts_sum_decay", source="factor_dsl_np")
class TSSumDecayPolars(SeriesOperator):
    """Polars 衰减权重滚动求和。"""

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
@register_operator(
    name="ts_zscore",
    category="time_series",
    business_category="time_series",
    canonical="ts_zscore",
    source="factor_dsl_np",
    backend="polars",
)
class TSZScorePolars(SeriesOperator):
    """Polars 滚动 Z-Score。"""

    metadata = OperatorMetadata(
        name="ts_zscore", category="time_series",
        description="滚动Z-Score (与m_zscore相同)",
        examples=["ts_zscore(close, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "zscore"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        w = max(1, int(window))
        return x.with_columns([
            ((pl.col(c) - pl.col(c).rolling_mean(window_size=w, min_samples=1)) /
             pl.when(pl.col(c).rolling_std(window_size=w, min_samples=1, ddof=1) == 0)
             .then(1)
             .otherwise(pl.col(c).rolling_std(window_size=w, min_samples=1, ddof=1))
            ).alias(c)
            for c in numeric_cols
        ])



# canonical=ts_sharpe backend=polars selected=ts_sharpe source=time_series/ts_ops_polars.py
@register_operator(
    name="ts_sharpe",
    category="time_series",
    business_category="time_series",
    canonical="ts_sharpe",
    source="factor_dsl_np",
    backend="polars",
)
class TSSharpePolars(SeriesOperator):
    """Polars 滚动夏普比率。"""

    metadata = OperatorMetadata(
        name="ts_sharpe",
        category="time_series",
        description="滚动夏普：mean/std * sqrt(ann_factor)",
        examples=["ts_sharpe(returns, 60)"],
        param_names=["x", "window", "ann_factor"],
        return_type="series",
        tags=["time_series", "sharpe", "pit_safe"],
    )

    def _calculate_series(
        self,
        x: pl.DataFrame,
        window: int = 60,
        ann_factor: float = 252.0,
        min_periods: int | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        w = max(2, int(window))
        mp = max(2, int(min_periods)) if min_periods is not None else max(2, w // 3)
        cols = [c for c in x.columns if c not in ["date", "stock_code"]]
        sqrt_af = float(np.sqrt(float(ann_factor)))
        return x.with_columns(
            [
                pl.when(
                    (pl.col(c).rolling_std(window_size=w, min_samples=mp, ddof=1) == 0)
                    & (pl.col(c).rolling_mean(window_size=w, min_samples=mp) > 0)
                )
                .then(float("inf"))
                .when(pl.col(c).rolling_std(window_size=w, min_samples=mp, ddof=1) == 0)
                .then(0.0)
                .otherwise(
                    pl.col(c).rolling_mean(window_size=w, min_samples=mp)
                    / pl.col(c).rolling_std(window_size=w, min_samples=mp, ddof=1)
                )
                .mul(sqrt_af)
                .alias(c)
                for c in cols
            ]
        )


# canonical=ts_autocorr backend=polars selected=ts_autocorr source=time_series/ts_ops_polars.py
@register_operator(
    name="ts_autocorr",
    category="time_series",
    business_category="time_series",
    canonical="ts_autocorr",
    source="factor_dsl_np",
    backend="polars",
)
class TSAutocorrPolars(SeriesOperator):
    """Polars 滚动自相关。"""

    metadata = OperatorMetadata(
        name="ts_autocorr",
        category="time_series",
        description="滚动自相关 corr(x_t, x_{t-lag}) within window",
        examples=["ts_autocorr(returns, 20, 1)"],
        param_names=["x", "window", "lag"],
        return_type="series",
        tags=["time_series", "autocorrelation", "pit_safe"],
    )

    def _calculate_series(
        self,
        x: pl.DataFrame,
        window: int = 20,
        lag: int = 1,
        min_periods: int | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        if int(lag) < 1:
            cols = [c for c in x.columns if c not in ["date", "stock_code"]]
            return x.with_columns([pl.lit(None).cast(pl.Float64).alias(c) for c in cols])
        k = max(1, int(lag))
        w = max(k + 2, int(window))
        mp = max(k + 2, int(min_periods)) if min_periods is not None else max(2, w // 3)
        cols = [c for c in x.columns if c not in ["date", "stock_code"]]
        return x.with_columns(
            [
                pl.rolling_corr(
                    pl.col(c),
                    pl.col(c).shift(k),
                    window_size=w,
                    min_samples=mp,
                ).alias(c)
                for c in cols
            ]
        )

