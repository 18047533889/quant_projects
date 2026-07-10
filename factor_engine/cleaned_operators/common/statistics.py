# -*- coding: utf-8 -*-
"""
统计量、分布检验与回归类算子。

语义
----
- **单序列统计**：``Mean``/``Var``/``Skew``/``Kurt``、``Mad``、``Median``、``Percentile``（多含大小写别名）；
- **双序列**：``Corr``、``Cov``、``rank_corr``、``Beta``/``beta``、``Slope``/``slope``；
- **时序扩展**：``ACF``、滚动回归系数、假设检验相关算子（见各类 ``metadata.description``）。

与 ``time_series.ts_regression`` 区别：本模块偏「全窗口/分布」统计与回归组件，时序 rolling 封装在 ``time_series``。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from cleaned_operators.base import (
    Operator,
    OperatorMetadata,
    SeriesOperator,
    ScalarOperator,
    TwoVarOperator,
    register_operator,
)
from cleaned_operators._rolling_fast import rolling_beta, rolling_regression
from cleaned_operators._causal import (
    expanding_argext,
    expanding_bivariate,
    expanding_first_not_null,
    expanding_panel_stat,
    expanding_two_sample,
    expanding_univariate,
)

import numpy as np
import pandas as pd
try:
    from scipy import stats
except ImportError:
    stats = None  # type: ignore


def _aligned_pair(x_col: pd.Series, y_col: pd.Series) -> tuple[pd.Series, pd.Series]:
    """按 index 对齐后 dropna，避免分别 dropna 再 iloc 配对错位。"""
    paired = pd.concat([x_col.rename("x"), y_col.rename("y")], axis=1).dropna()
    return paired["x"], paired["y"]


def _broadcast_scalar_row(x: pd.DataFrame, values: dict) -> pd.DataFrame:
    """将每列标量统计广播为与 panel 同形的 DataFrame。"""
    series = pd.Series(values, dtype=float).reindex(x.columns)
    return pd.DataFrame(
        np.broadcast_to(series.to_numpy(), x.shape),
        index=x.index,
        columns=x.columns,
    )


def _broadcast_column_stat(x: pd.DataFrame, stat: pd.Series) -> pd.DataFrame:
    """将按列聚合的统计量广播到每一行。"""
    stat = stat.reindex(x.columns)
    return pd.DataFrame(
        np.broadcast_to(stat.to_numpy(), x.shape),
        index=x.index,
        columns=x.columns,
    )

# canonical=ACF backend=pandas_numpy selected=ACF source=statistics/basic_stats.py
@register_operator(name="ACF", category="statistics", business_category="statistics_regression", canonical="ACF", source="factor_dsl_np")
class ACF(SeriesOperator):
    """自相关系数"""

    metadata = OperatorMetadata(
        name="ACF",
        category="statistics",
        description="自相关系数",
        examples=["ACF(returns, 20, 5)"],
        param_names=["x", "window", "lag"],
        return_type="series",
        tags=["statistics", "autocorrelation"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, lag: int = 1, **kwargs) -> pd.DataFrame:
        lag = int(lag)
        if lag < 0:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)

        def acf_func(s, lag):
            n = len(s)
            if n < max(lag, 2):
                return 0
            mean = s.mean()
            var = ((s - mean) ** 2).sum()
            if var == 0:
                return 0
            return ((s[:-lag] - mean) * (s[lag:] - mean)).sum() / var

        return x.rolling(window=window, min_periods=1).apply(acf_func, raw=True, args=(lag,))

# aliases: acf



# canonical=Beta backend=pandas_numpy selected=Beta source=statistics/regression.py
@register_operator(name="Beta", category="statistics", business_category="statistics_regression", canonical="Beta", source="factor_dsl_np")
class Beta(SeriesOperator):
    """回归Beta系数"""
    metadata = OperatorMetadata(
        name="Beta", category="statistics",
        description="回归Beta系数",
        examples=["Beta(returns, market, 20)"],
        param_names=["y", "x", "window"], return_type="series",
        tags=["statistics", "regression", "beta"]
    )
    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return rolling_beta(y, x, window=window, min_periods=2)



# canonical=Corr backend=pandas_numpy selected=Corr source=statistics/basic_stats.py
@register_operator(name="Corr", category="statistics", business_category="statistics_regression", canonical="Corr", source="factor_dsl_np")
class Corr(SeriesOperator):
    """相关系数"""

    metadata = OperatorMetadata(
        name="Corr",
        category="statistics",
        description="相关系数",
        examples=["Corr(close, volume, 20)"],
        param_names=["x", "y", "window"],
        return_type="series",
        tags=["statistics", "corr", "correlation"]
    )

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).corr(y)

# aliases: corr



# canonical=Cov backend=pandas_numpy selected=Cov source=statistics/basic_stats.py
@register_operator(name="Cov", category="statistics", business_category="statistics_regression", canonical="Cov", source="factor_dsl_np")
class Cov(SeriesOperator):
    """协方差"""

    metadata = OperatorMetadata(
        name="Cov",
        category="statistics",
        description="协方差",
        examples=["Cov(returns, market, 20)"],
        param_names=["x", "y", "window"],
        return_type="series",
        tags=["statistics", "cov", "covariance"]
    )

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).cov(y)

# aliases: cov



# canonical=Covariance backend=pandas_numpy selected=Covariance source=statistics/regression_ex.py
@register_operator(name="Covariance", category="statistics", business_category="statistics_regression", canonical="Covariance", source="factor_dsl_np")
class Covariance(SeriesOperator):
    """协方差全称"""
    metadata = OperatorMetadata(
        name="Covariance", category="statistics",
        description="协方差全称",
        examples=["Covariance(returns, market, 20)"],
        param_names=["x", "y", "window"], return_type="series",
        tags=["statistics", "covariance"]
    )
    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).cov(y)



# canonical=Intercept backend=pandas_numpy selected=Intercept source=statistics/regression.py
@register_operator(name="Intercept", category="statistics", business_category="statistics_regression", canonical="Intercept", source="factor_dsl_np")
class Intercept(SeriesOperator):
    """OLS回归截距"""
    metadata = OperatorMetadata(
        name="Intercept", category="statistics",
        description="OLS回归截距",
        examples=["Intercept(close, market, 20)"],
        param_names=["y", "x", "window"], return_type="series",
        tags=["statistics", "regression", "intercept"]
    )
    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return rolling_regression(y, x, window=window, min_periods=2, retval="intercept")



# canonical=Kurt backend=pandas_numpy selected=Kurt source=statistics/basic_stats.py
@register_operator(name="Kurt", category="statistics", business_category="statistics_regression", canonical="Kurt", source="factor_dsl_np")
class Kurt(SeriesOperator):
    """峰度"""

    metadata = OperatorMetadata(
        name="Kurt",
        category="statistics",
        description="峰度",
        examples=["Kurt(returns, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["statistics", "kurt", "kurtosis"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).kurt()

# aliases: kurt



# canonical=Mad backend=pandas_numpy selected=Mad source=statistics/basic_stats.py
@register_operator(name="Mad", category="statistics", business_category="statistics_regression", canonical="Mad", source="factor_dsl_np")
class Mad(SeriesOperator):
    """平均绝对离差"""

    metadata = OperatorMetadata(
        name="Mad",
        category="statistics",
        description="平均绝对离差",
        examples=["Mad(returns, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["statistics", "mad", "deviation"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        median = x.rolling(window=window, min_periods=1).median()
        return (x - median).abs().rolling(window=window, min_periods=1).mean()

# aliases: mad



# canonical=Median backend=pandas_numpy selected=Median source=statistics/basic_stats.py
@register_operator(name="Median", category="statistics", business_category="statistics_regression", canonical="Median", source="factor_dsl_np")
class Median(SeriesOperator):
    """中位数"""

    metadata = OperatorMetadata(
        name="Median",
        category="statistics",
        description="中位数",
        examples=["Median(close, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["statistics", "median"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).median()

# aliases: median



# canonical=Mode backend=pandas_numpy selected=Mode source=statistics/basic_stats.py
@register_operator(name="Mode", category="statistics", business_category="statistics_regression", canonical="Mode", source="factor_dsl_np")
class Mode(SeriesOperator):
    """众数"""

    metadata = OperatorMetadata(
        name="Mode",
        category="statistics",
        description="众数",
        examples=["Mode(close, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["statistics", "mode"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).apply(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else s.iloc[-1], raw=False)

# aliases: mode



# canonical=Percentile backend=pandas_numpy selected=Percentile source=statistics/basic_stats.py
@register_operator(name="Percentile", category="statistics", business_category="statistics_regression", canonical="Percentile", source="factor_dsl_np")
class Percentile(SeriesOperator):
    """分位数"""

    metadata = OperatorMetadata(
        name="Percentile",
        category="statistics",
        description="分位数（0-1之间）",
        examples=["Percentile(returns, 20, 0.75)"],
        param_names=["x", "window", "p"],
        return_type="series",
        tags=["statistics", "percentile", "quantile"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, p: float = 0.5, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).quantile(p)

# aliases: percentile



# canonical=R2 backend=pandas_numpy selected=R2 source=statistics/regression.py
@register_operator(name="R2", category="statistics", business_category="statistics_regression", canonical="R2", source="factor_dsl_np")
class R2(SeriesOperator):
    """R-squared决定系数"""
    metadata = OperatorMetadata(
        name="R2", category="statistics",
        description="R-squared决定系数",
        examples=["R2(close, market, 20)"],
        param_names=["y", "x", "window"], return_type="series",
        tags=["statistics", "regression", "r2"]
    )
    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return rolling_regression(y, x, window=window, min_periods=2, retval="r_squared")



# canonical=Residual backend=pandas_numpy selected=Residual source=statistics/regression.py
@register_operator(name="Residual", category="statistics", business_category="statistics_regression", canonical="Residual", source="factor_dsl_np")
class Residual(SeriesOperator):
    """回归残差均值"""
    metadata = OperatorMetadata(
        name="Residual", category="statistics",
        description="回归残差均值",
        examples=["Residual(close, market, 20)"],
        param_names=["y", "x", "window"], return_type="series",
        tags=["statistics", "regression", "residual"]
    )
    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return rolling_regression(y, x, window=window, min_periods=2, retval="residual")



# canonical=Skew backend=pandas_numpy selected=Skew source=statistics/basic_stats.py
@register_operator(name="Skew", category="statistics", business_category="statistics_regression", canonical="Skew", source="factor_dsl_np")
class Skew(SeriesOperator):
    """偏度"""

    metadata = OperatorMetadata(
        name="Skew",
        category="statistics",
        description="偏度",
        examples=["Skew(returns, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["statistics", "skew", "skewness"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).skew()

# aliases: skew



# canonical=Slope backend=pandas_numpy selected=Slope source=statistics/regression.py
@register_operator(name="Slope", category="statistics", business_category="statistics_regression", canonical="Slope", source="factor_dsl_np")
class Slope(SeriesOperator):
    """时间序列斜率 (对时间t的线性回归斜率)"""
    metadata = OperatorMetadata(
        name="Slope", category="statistics",
        description="时间序列斜率 (对时间t的线性回归斜率)",
        examples=["Slope(close, 20)", "Slope(volume, 5)"],
        param_names=["x", "window"], return_type="series",
        tags=["statistics", "regression", "slope", "time_series"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        from cleaned_operators._rolling_fast import rolling_time_slope

        return rolling_time_slope(x, window)



# canonical=Sum backend=pandas_numpy selected=Sum source=statistics/basic_stats.py
@register_operator(name="Sum", category="statistics", business_category="statistics_regression", canonical="Sum", source="factor_dsl_np")
class Sum(SeriesOperator):
    """求和"""

    metadata = OperatorMetadata(
        name="Sum",
        category="statistics",
        description="滚动求和",
        examples=["Sum(volume, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["statistics", "sum"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).sum()

# aliases: sum



# canonical=Var backend=pandas_numpy selected=Var source=statistics/basic_stats.py
@register_operator(name="Var", category="statistics", business_category="statistics_regression", canonical="Var", source="factor_dsl_np")
class Var(SeriesOperator):
    """方差"""

    metadata = OperatorMetadata(
        name="Var",
        category="statistics",
        description="方差",
        examples=["Var(returns, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["statistics", "var", "variance"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).var()

# aliases: var



# canonical=at_imax backend=pandas_numpy selected=at_imax source=statistics/aggregate_ops.py
@register_operator(name="at_imax", category="statistics", business_category="statistics_regression", canonical="at_imax", source="factor_dsl_np")
class at_imax(SeriesOperator):
    """列最大值索引"""
    metadata = OperatorMetadata(
        name="at_imax", category="statistics",
        description="列最大值索引",
        examples=["at_imax(close)"],
        param_names=["x"], return_type="series",
        tags=["statistics", "aggregate", "imax"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_argext(x, which="max")



# canonical=at_imin backend=pandas_numpy selected=at_imin source=statistics/aggregate_ops.py
@register_operator(name="at_imin", category="statistics", business_category="statistics_regression", canonical="at_imin", source="factor_dsl_np")
class at_imin(SeriesOperator):
    """列最小值索引"""
    metadata = OperatorMetadata(
        name="at_imin", category="statistics",
        description="列最小值索引",
        examples=["at_imin(close)"],
        param_names=["x"], return_type="series",
        tags=["statistics", "aggregate", "imin"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_argext(x, which="min")



# canonical=autocorr backend=pandas_numpy selected=autocorr source=statistics/aggregate_ops.py
@register_operator(name="autocorr", category="statistics", business_category="statistics_regression", canonical="autocorr", source="factor_dsl_np")
class autocorr(SeriesOperator):
    """自相关系数"""
    metadata = OperatorMetadata(
        name="autocorr", category="statistics",
        description="自相关系数",
        examples=["autocorr(returns, 5)"],
        param_names=["x", "lag"], return_type="series",
        tags=["statistics", "aggregate", "autocorrelation"]
    )
    def _calculate_series(self, x: pd.DataFrame, lag: int = 1, **kwargs) -> pd.DataFrame:
        lag = int(lag)
        if lag < 0:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)
        return expanding_univariate(
            x,
            lambda v: pd.Series(v).autocorr(lag=lag),
            min_periods=max(lag + 2, 2),
        )



# canonical=avg backend=pandas_numpy selected=avg source=statistics/aggregate_ops.py
@register_operator(name="avg", category="statistics", business_category="statistics_regression", canonical="avg", source="factor_dsl_np")
class avg(SeriesOperator):
    """列平均值"""
    metadata = OperatorMetadata(
        name="avg", category="statistics",
        description="列平均值",
        examples=["avg(close)"],
        param_names=["x"], return_type="series",
        tags=["statistics", "aggregate", "average"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_panel_stat(x, "mean")



# canonical=bartlett_test backend=pandas_numpy selected=bartlett_test source=statistics/hypothesis_ops.py
@register_operator(name="bartlett_test", category="statistics", business_category="statistics_regression", canonical="bartlett_test", source="factor_dsl_np")
class bartlett_test(SeriesOperator):
    """Bartlett方差齐性检验"""
    metadata = OperatorMetadata(
        name="bartlett_test", category="statistics",
        description="Bartlett方差齐性检验",
        examples=["bartlett_test(x, y)"],
        param_names=["x", "y"], return_type="series",
        tags=["statistics", "hypothesis", "bartlett"]
    )
    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_two_sample(
            x,
            y,
            lambda a, b: stats.bartlett(a, b)[1],
            min_periods=2,
        )



# canonical=beta backend=pandas_numpy selected=beta source=statistics/regression_ex.py
@register_operator(name="beta", category="statistics", business_category="statistics_regression", canonical="beta", source="factor_dsl_np")
class beta(SeriesOperator):
    """小写beta回归系数"""
    metadata = OperatorMetadata(
        name="beta", category="statistics",
        description="小写beta回归系数",
        examples=["beta(returns, market, 20)"],
        param_names=["y", "x", "window"], return_type="series",
        tags=["statistics", "regression", "beta"]
    )
    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return rolling_beta(y, x, window=window, min_periods=2)



# canonical=cdf_chi2 backend=pandas_numpy selected=cdf_chi2 source=statistics/probability_ops.py
@register_operator(name="cdf_chi2", category="statistics", business_category="statistics_regression", canonical="cdf_chi2", source="factor_dsl_np")
class cdf_chi2(SeriesOperator):
    """卡方分布累积分布函数"""
    metadata = OperatorMetadata(
        name="cdf_chi2", category="statistics",
        description="卡方分布累积分布函数",
        examples=["cdf_chi2(x, 5)"],
        param_names=["x", "df"], return_type="series",
        tags=["statistics", "probability", "chi2", "cdf"]
    )
    def _calculate_series(self, x: pd.DataFrame, df: float = 5.0, **kwargs) -> pd.DataFrame:
        return x.apply(lambda col: stats.chi2.cdf(col, df=df))



# canonical=cdf_f backend=pandas_numpy selected=cdf_f source=statistics/probability_ops.py
@register_operator(name="cdf_f", category="statistics", business_category="statistics_regression", canonical="cdf_f", source="factor_dsl_np")
class cdf_f(SeriesOperator):
    """F分布累积分布函数"""
    metadata = OperatorMetadata(
        name="cdf_f", category="statistics",
        description="F分布累积分布函数",
        examples=["cdf_f(x, 5, 10)"],
        param_names=["x", "d1", "d2"], return_type="series",
        tags=["statistics", "probability", "f", "cdf"]
    )
    def _calculate_series(self, x: pd.DataFrame, d1: float = 5.0, d2: float = 10.0, **kwargs) -> pd.DataFrame:
        return x.apply(lambda col: stats.f.cdf(col, dfn=d1, dfd=d2))



# canonical=cdf_normal backend=pandas_numpy selected=cdf_normal source=statistics/probability_ops.py
@register_operator(name="cdf_normal", category="statistics", business_category="statistics_regression", canonical="cdf_normal", source="factor_dsl_np")
class cdf_normal(SeriesOperator):
    """正态分布累积分布函数"""
    metadata = OperatorMetadata(
        name="cdf_normal", category="statistics",
        description="正态分布累积分布函数",
        examples=["cdf_normal(x, 0, 1)"],
        param_names=["x", "mu", "sigma"], return_type="series",
        tags=["statistics", "probability", "normal", "cdf"]
    )
    def _calculate_series(self, x: pd.DataFrame, mu: float = 0.0, sigma: float = 1.0, **kwargs) -> pd.DataFrame:
        return x.apply(lambda col: stats.norm.cdf(col, loc=mu, scale=sigma))



# canonical=cdf_t backend=pandas_numpy selected=cdf_t source=statistics/probability_ops.py
@register_operator(name="cdf_t", category="statistics", business_category="statistics_regression", canonical="cdf_t", source="factor_dsl_np")
class cdf_t(SeriesOperator):
    """t分布累积分布函数"""
    metadata = OperatorMetadata(
        name="cdf_t", category="statistics",
        description="t分布累积分布函数",
        examples=["cdf_t(x, 10)"],
        param_names=["x", "df"], return_type="series",
        tags=["statistics", "probability", "t", "cdf"]
    )
    def _calculate_series(self, x: pd.DataFrame, df: float = 10.0, **kwargs) -> pd.DataFrame:
        return x.apply(lambda col: stats.t.cdf(col, df=df))



# canonical=chi_square_test backend=pandas_numpy selected=chi_square_test source=statistics/hypothesis_ops.py
@register_operator(name="chi_square_test", category="statistics", business_category="statistics_regression", canonical="chi_square_test", source="factor_dsl_np")
class chi_square_test(SeriesOperator):
    """卡方检验"""
    metadata = OperatorMetadata(
        name="chi_square_test", category="statistics",
        description="卡方检验",
        examples=["chi_square_test(observed, expected)"],
        param_names=["x", "y"], return_type="series",
        tags=["statistics", "hypothesis", "chi_square"]
    )
    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_bivariate(
            x,
            y,
            lambda a, b: stats.chisquare(a, b)[1],
            min_periods=2,
        )



# canonical=corr_test backend=pandas_numpy selected=corr_test source=statistics/hypothesis_ops.py
@register_operator(name="corr_test", category="statistics", business_category="statistics_regression", canonical="corr_test", source="factor_dsl_np")
class corr_test(SeriesOperator):
    """Pearson相关性检验"""
    metadata = OperatorMetadata(
        name="corr_test", category="statistics",
        description="Pearson相关性检验",
        examples=["corr_test(x, y)"],
        param_names=["x", "y"], return_type="series",
        tags=["statistics", "hypothesis", "correlation"]
    )
    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_bivariate(
            x,
            y,
            lambda a, b: stats.pearsonr(a, b)[1],
            min_periods=3,
        )



# canonical=count backend=pandas_numpy selected=count source=statistics/aggregate_ops.py
@register_operator(name="count", category="statistics", business_category="statistics_regression", canonical="count", source="factor_dsl_np")
class count(SeriesOperator):
    """非空值计数"""
    metadata = OperatorMetadata(
        name="count", category="statistics",
        description="非空值计数",
        examples=["count(close)"],
        param_names=["x"], return_type="series",
        tags=["statistics", "aggregate", "count"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_panel_stat(x, "count")



# canonical=durbin_watson_test backend=pandas_numpy selected=durbin_watson_test source=statistics/hypothesis_ops.py
@register_operator(name="durbin_watson_test", category="statistics", business_category="statistics_regression", canonical="durbin_watson_test", source="factor_dsl_np")
class durbin_watson_test(SeriesOperator):
    """Durbin-Watson自相关检验"""
    metadata = OperatorMetadata(
        name="durbin_watson_test", category="statistics",
        description="Durbin-Watson自相关检验",
        examples=["durbin_watson_test(residuals)"],
        param_names=["residuals"], return_type="series",
        tags=["statistics", "hypothesis", "durbin_watson"]
    )
    def _calculate_series(self, residuals: pd.DataFrame, **kwargs) -> pd.DataFrame:
        from statsmodels.stats.stattools import durbin_watson

        return expanding_univariate(residuals, durbin_watson, min_periods=2)



# canonical=first backend=pandas_numpy selected=first source=statistics/aggregate_ops.py
@register_operator(name="first", category="statistics", business_category="statistics_regression", canonical="first", source="factor_dsl_np")
class first(SeriesOperator):
    """列首值"""
    metadata = OperatorMetadata(
        name="first", category="statistics",
        description="列首值",
        examples=["first(close)"],
        param_names=["x"], return_type="series",
        tags=["statistics", "aggregate", "first"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_first_not_null(x)



# canonical=first_not_null backend=pandas_numpy selected=first_not_null source=statistics/aggregate_ops.py
@register_operator(name="first_not_null", category="statistics", business_category="statistics_regression", canonical="first_not_null", source="factor_dsl_np")
class first_not_null(SeriesOperator):
    """列首个非空值"""
    metadata = OperatorMetadata(
        name="first_not_null", category="statistics",
        description="列首个非空值",
        examples=["first_not_null(close)"],
        param_names=["x"], return_type="series",
        tags=["statistics", "aggregate", "first_not_null"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_first_not_null(x)



# canonical=granger_causality backend=pandas_numpy selected=granger_causality source=statistics/hypothesis_ops.py
@register_operator(name="granger_causality", category="statistics", business_category="statistics_regression", canonical="granger_causality", source="factor_dsl_np")
class granger_causality(SeriesOperator):
    """Granger因果检验"""
    metadata = OperatorMetadata(
        name="granger_causality", category="statistics",
        description="Granger因果检验",
        examples=["granger_causality(x, y, 5)"],
        param_names=["x", "y", "lag"], return_type="series",
        tags=["statistics", "hypothesis", "granger", "causality"]
    )
    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, lag: int = 5, **kwargs) -> pd.DataFrame:
        lag = int(lag)
        if lag < 1:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)

        def _granger_pval(a, b):
            from statsmodels.tsa.stattools import grangercausalitytests

            data = pd.DataFrame({"x": a, "y": b})
            test_result = grangercausalitytests(data[["y", "x"]], maxlag=lag, verbose=False)
            return test_result[lag][0]["ssr_ftest"][1]

        return expanding_bivariate(x, y, _granger_pval, min_periods=lag + 2)



# canonical=intercept backend=pandas_numpy selected=intercept source=statistics/regression_ex.py
@register_operator(name="intercept", category="statistics", business_category="statistics_regression", canonical="intercept", source="factor_dsl_np")
class intercept(SeriesOperator):
    """小写intercept回归截距"""
    metadata = OperatorMetadata(
        name="intercept", category="statistics",
        description="小写intercept回归截距",
        examples=["intercept(close, market, 20)"],
        param_names=["y", "x", "window"], return_type="series",
        tags=["statistics", "regression", "intercept"]
    )
    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return rolling_regression(y, x, window=window, min_periods=2, retval="intercept")



# canonical=jarque_bera_test backend=pandas_numpy selected=jarque_bera_test source=statistics/hypothesis_ops.py
@register_operator(name="jarque_bera_test", category="statistics", business_category="statistics_regression", canonical="jarque_bera_test", source="factor_dsl_np")
class jarque_bera_test(SeriesOperator):
    """Jarque-Bera正态性检验"""
    metadata = OperatorMetadata(
        name="jarque_bera_test", category="statistics",
        description="Jarque-Bera正态性检验",
        examples=["jarque_bera_test(returns)"],
        param_names=["x"], return_type="series",
        tags=["statistics", "hypothesis", "jarque_bera", "normality"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_univariate(
            x,
            lambda v: stats.jarque_bera(v)[1],
            min_periods=3,
        )



# canonical=kendall_corr_test backend=pandas_numpy selected=kendall_corr_test source=statistics/hypothesis_ops.py
@register_operator(name="kendall_corr_test", category="statistics", business_category="statistics_regression", canonical="kendall_corr_test", source="factor_dsl_np")
class kendall_corr_test(SeriesOperator):
    """Kendall相关性检验"""
    metadata = OperatorMetadata(
        name="kendall_corr_test", category="statistics",
        description="Kendall相关性检验",
        examples=["kendall_corr_test(x, y)"],
        param_names=["x", "y"], return_type="series",
        tags=["statistics", "hypothesis", "kendall", "correlation"]
    )
    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_bivariate(
            x,
            y,
            lambda a, b: stats.kendalltau(a, b)[1],
            min_periods=3,
        )



# canonical=kpss_test backend=pandas_numpy selected=kpss_test source=statistics/hypothesis_ops.py
@register_operator(name="kpss_test", category="statistics", business_category="statistics_regression", canonical="kpss_test", source="factor_dsl_np")
class kpss_test(SeriesOperator):
    """KPSS平稳性检验"""
    metadata = OperatorMetadata(
        name="kpss_test", category="statistics",
        description="KPSS平稳性检验",
        examples=["kpss_test(prices)"],
        param_names=["x"], return_type="series",
        tags=["statistics", "hypothesis", "kpss", "stationarity"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        def _kpss_pval(v):
            from statsmodels.tsa.stattools import kpss

            return kpss(v, regression="c", nlags="auto")[1]

        return expanding_univariate(x, _kpss_pval, min_periods=10)



# canonical=ks_test backend=pandas_numpy selected=ks_test source=statistics/hypothesis_ops.py
@register_operator(name="ks_test", category="statistics", business_category="statistics_regression", canonical="ks_test", source="factor_dsl_np")
class ks_test(SeriesOperator):
    """KS检验"""
    metadata = OperatorMetadata(
        name="ks_test", category="statistics",
        description="KS检验",
        examples=["ks_test(returns, 'norm')"],
        param_names=["x", "dist"], return_type="series",
        tags=["statistics", "hypothesis", "ks"]
    )
    def _calculate_series(self, x: pd.DataFrame, dist: str = "norm", **kwargs) -> pd.DataFrame:
        return expanding_univariate(
            x,
            lambda v: stats.kstest(v, dist)[1],
            min_periods=2,
        )



# canonical=lasso backend=pandas_numpy selected=lasso source=statistics/regression_ex.py
@register_operator(name="lasso", category="statistics", business_category="statistics_regression", canonical="lasso", source="factor_dsl_np")
class Lasso(SeriesOperator):
    """Lasso回归(L1正则化)，单参数时返回序列的L1正则化趋势，双参数时执行Lasso回归"""
    metadata = OperatorMetadata(
        name="lasso", category="statistics",
        description="Lasso回归(L1正则化)，单参数时返回序列的L1正则化趋势，双参数时执行Lasso回归",
        examples=["lasso(close, 20)", "lasso(close, market, 20, 0.1)"],
        param_names=["x", "window_or_y", "window", "alpha"], return_type="series",
        tags=["statistics", "regression", "lasso", "regularization"]
    )
    def _calculate_series(self, x: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        window = 20
        alpha = 0.1
        y = None

        if len(args) >= 1:
            if isinstance(args[0], pd.DataFrame):
                y = args[0]
                if len(args) >= 2:
                    window = int(args[1]) if not isinstance(args[1], pd.DataFrame) else window
                if len(args) >= 3:
                    alpha = float(args[3]) if len(args) > 3 else alpha
            else:
                try:
                    window = int(args[0])
                except (TypeError, ValueError):
                    pass
                if len(args) >= 2:
                    try:
                        alpha = float(args[1])
                    except (TypeError, ValueError):
                        pass

        def _lasso_trend(x_vals):
            n = len(x_vals)
            valid = ~np.isnan(x_vals)
            if valid.sum() < 3:
                return np.nan
            xv = x_vals[valid]
            idx = np.arange(n)[valid]
            if y is None:
                X = np.column_stack([np.ones(len(idx)), (idx - idx.mean()) / (idx.std() + 1e-10)])
                try:
                    from sklearn.linear_model import Lasso as LassoCV
                    model = LassoCV(alpha=alpha, max_iter=1000)
                    model.fit(X, xv)
                    return model.coef_[1] if len(model.coef_) > 1 else model.coef_[0]
                except ImportError:
                    coeffs = np.linalg.lstsq(X, xv, rcond=None)[0]
                    l1_penalty = alpha * np.sum(np.abs(coeffs))
                    return coeffs[-1] / (1 + l1_penalty)
            return np.nan

        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
        for col in x.columns:
            try:
                result[col] = x[col].rolling(window=window, min_periods=3).apply(
                    _lasso_trend, raw=True
                )
            except Exception:
                result[col] = np.nan
        return result



# canonical=last backend=pandas_numpy selected=last source=statistics/aggregate_ops.py
@register_operator(name="last", category="statistics", business_category="statistics_regression", canonical="last", source="factor_dsl_np")
class last(SeriesOperator):
    """列末值"""
    metadata = OperatorMetadata(
        name="last", category="statistics",
        description="列末值",
        examples=["last(close)"],
        param_names=["x"], return_type="series",
        tags=["statistics", "aggregate", "last"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.ffill()



# canonical=last_not_null backend=pandas_numpy selected=last_not_null source=statistics/aggregate_ops.py
@register_operator(name="last_not_null", category="statistics", business_category="statistics_regression", canonical="last_not_null", source="factor_dsl_np")
class last_not_null(SeriesOperator):
    """列最后非空值"""
    metadata = OperatorMetadata(
        name="last_not_null", category="statistics",
        description="列最后非空值",
        examples=["last_not_null(close)"],
        param_names=["x"], return_type="series",
        tags=["statistics", "aggregate", "last_not_null"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.ffill()



# canonical=levene_test backend=pandas_numpy selected=levene_test source=statistics/hypothesis_ops.py
@register_operator(name="levene_test", category="statistics", business_category="statistics_regression", canonical="levene_test", source="factor_dsl_np")
class levene_test(SeriesOperator):
    """Levene方差齐性检验"""
    metadata = OperatorMetadata(
        name="levene_test", category="statistics",
        description="Levene方差齐性检验",
        examples=["levene_test(x, y)"],
        param_names=["x", "y"], return_type="series",
        tags=["statistics", "hypothesis", "levene"]
    )
    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_two_sample(
            x,
            y,
            lambda a, b: stats.levene(a, b)[1],
            min_periods=2,
        )



# canonical=lilliefors_test backend=pandas_numpy selected=lilliefors_test source=statistics/hypothesis_ops.py
@register_operator(name="lilliefors_test", category="statistics", business_category="statistics_regression", canonical="lilliefors_test", source="factor_dsl_np")
class lilliefors_test(SeriesOperator):
    """Lilliefors正态性检验"""
    metadata = OperatorMetadata(
        name="lilliefors_test", category="statistics",
        description="Lilliefors正态性检验",
        examples=["lilliefors_test(returns)"],
        param_names=["x"], return_type="series",
        tags=["statistics", "hypothesis", "lilliefors", "normality"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_univariate(
            x,
            lambda v: stats.lilliefors(v)[1],
            min_periods=5,
        )



# canonical=mean_agg backend=pandas_numpy selected=mean_agg source=statistics/aggregate_ops.py
@register_operator(name="mean_agg", category="statistics", business_category="statistics_regression", canonical="mean_agg", source="factor_dsl_np")
class mean_agg(SeriesOperator):
    """列平均值(同avg)"""
    metadata = OperatorMetadata(
        name="mean_agg", category="statistics",
        description="列平均值(同avg)",
        examples=["mean_agg(close)"],
        param_names=["x"], return_type="series",
        tags=["statistics", "aggregate", "mean"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_panel_stat(x, "mean")



# canonical=pacf backend=pandas_numpy selected=pacf source=statistics/aggregate_ops.py
@register_operator(name="pacf", category="statistics", business_category="statistics_regression", canonical="pacf", source="factor_dsl_np")
class pacf(SeriesOperator):
    """偏自相关系数"""
    metadata = OperatorMetadata(
        name="pacf", category="statistics",
        description="偏自相关系数",
        examples=["pacf(returns, 5)"],
        param_names=["x", "lag"], return_type="series",
        tags=["statistics", "aggregate", "pacf"]
    )
    def _calculate_series(self, x: pd.DataFrame, lag: int = 1, **kwargs) -> pd.DataFrame:
        from statsmodels.tsa.stattools import pacf as sm_pacf

        lag = int(lag)
        if lag < 0:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)

        def _pacf_at_lag(v):
            pacf_vals = sm_pacf(v, nlags=lag)
            return pacf_vals[lag] if lag < len(pacf_vals) else np.nan

        return expanding_univariate(x, _pacf_at_lag, min_periods=max(lag + 1, 3))



# canonical=pdf_chi2 backend=pandas_numpy selected=pdf_chi2 source=statistics/probability_ops.py
@register_operator(name="pdf_chi2", category="statistics", business_category="statistics_regression", canonical="pdf_chi2", source="factor_dsl_np")
class pdf_chi2(SeriesOperator):
    """卡方分布概率密度函数"""
    metadata = OperatorMetadata(
        name="pdf_chi2", category="statistics",
        description="卡方分布概率密度函数",
        examples=["pdf_chi2(x, 5)"],
        param_names=["x", "df"], return_type="series",
        tags=["statistics", "probability", "chi2", "pdf"]
    )
    def _calculate_series(self, x: pd.DataFrame, df: float = 5.0, **kwargs) -> pd.DataFrame:
        return x.apply(lambda col: stats.chi2.pdf(col, df=df))



# canonical=pdf_f backend=pandas_numpy selected=pdf_f source=statistics/probability_ops.py
@register_operator(name="pdf_f", category="statistics", business_category="statistics_regression", canonical="pdf_f", source="factor_dsl_np")
class pdf_f(SeriesOperator):
    """F分布概率密度函数"""
    metadata = OperatorMetadata(
        name="pdf_f", category="statistics",
        description="F分布概率密度函数",
        examples=["pdf_f(x, 5, 10)"],
        param_names=["x", "d1", "d2"], return_type="series",
        tags=["statistics", "probability", "f", "pdf"]
    )
    def _calculate_series(self, x: pd.DataFrame, d1: float = 5.0, d2: float = 10.0, **kwargs) -> pd.DataFrame:
        return x.apply(lambda col: stats.f.pdf(col, dfn=d1, dfd=d2))



# canonical=pdf_normal backend=pandas_numpy selected=pdf_normal source=statistics/probability_ops.py
@register_operator(name="pdf_normal", category="statistics", business_category="statistics_regression", canonical="pdf_normal", source="factor_dsl_np")
class pdf_normal(SeriesOperator):
    """正态分布概率密度函数"""
    metadata = OperatorMetadata(
        name="pdf_normal", category="statistics",
        description="正态分布概率密度函数",
        examples=["pdf_normal(x, 0, 1)"],
        param_names=["x", "mu", "sigma"], return_type="series",
        tags=["statistics", "probability", "normal", "pdf"]
    )
    def _calculate_series(self, x: pd.DataFrame, mu: float = 0.0, sigma: float = 1.0, **kwargs) -> pd.DataFrame:
        return x.apply(lambda col: stats.norm.pdf(col, loc=mu, scale=sigma))



# canonical=pdf_t backend=pandas_numpy selected=pdf_t source=statistics/probability_ops.py
@register_operator(name="pdf_t", category="statistics", business_category="statistics_regression", canonical="pdf_t", source="factor_dsl_np")
class pdf_t(SeriesOperator):
    """t分布概率密度函数"""
    metadata = OperatorMetadata(
        name="pdf_t", category="statistics",
        description="t分布概率密度函数",
        examples=["pdf_t(x, 10)"],
        param_names=["x", "df"], return_type="series",
        tags=["statistics", "probability", "t", "pdf"]
    )
    def _calculate_series(self, x: pd.DataFrame, df: float = 10.0, **kwargs) -> pd.DataFrame:
        return x.apply(lambda col: stats.t.pdf(col, df=df))



# canonical=product backend=pandas_numpy selected=product source=statistics/aggregate_ops.py
@register_operator(name="product", category="statistics", business_category="statistics_regression", canonical="product", source="factor_dsl_np")
class product(SeriesOperator):
    """列连乘积"""
    metadata = OperatorMetadata(
        name="product", category="statistics",
        description="列连乘积",
        examples=["product(returns)"],
        param_names=["x"], return_type="series",
        tags=["statistics", "aggregate", "product"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_panel_stat(x, "product")



# canonical=quantile backend=pandas_numpy selected=quantile source=statistics/regression_ex.py
@register_operator(name="quantile", category="statistics", business_category="statistics_regression", canonical="quantile", source="factor_dsl_np")
class quantile(SeriesOperator):
    """分箱/离散化"""
    metadata = OperatorMetadata(
        name="quantile", category="statistics",
        description="分箱/离散化",
        examples=["quantile(close, 10)"],
        param_names=["x", "bins"], return_type="series",
        tags=["statistics", "quantile", "discretization"]
    )
    def _calculate_series(self, x: pd.DataFrame, bins: int = 10, **kwargs) -> pd.DataFrame:
        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
        for idx in x.index:
            row = x.loc[idx]
            valid = row.dropna()
            if len(valid) < 2:
                continue
            try:
                labels = pd.qcut(valid, q=bins, labels=False, duplicates="drop")
                result.loc[idx, labels.index] = labels.astype(float)
            except (ValueError, TypeError):
                continue
        return result



# canonical=quantile_normal backend=pandas_numpy selected=quantile_normal source=statistics/probability_ops.py
@register_operator(name="quantile_normal", category="statistics", business_category="statistics_regression", canonical="quantile_normal", source="factor_dsl_np")
class quantile_normal(SeriesOperator):
    """正态分布分位数函数"""
    metadata = OperatorMetadata(
        name="quantile_normal", category="statistics",
        description="正态分布分位数函数",
        examples=["quantile_normal(0.95, 0, 1)"],
        param_names=["p", "mu", "sigma"], return_type="series",
        tags=["statistics", "probability", "normal", "quantile"]
    )
    def _calculate_series(self, p: pd.DataFrame, mu: float = 0.0, sigma: float = 1.0, **kwargs) -> pd.DataFrame:
        return p.apply(lambda col: stats.norm.ppf(col, loc=mu, scale=sigma))



# canonical=quantile_t backend=pandas_numpy selected=quantile_t source=statistics/probability_ops.py
@register_operator(name="quantile_t", category="statistics", business_category="statistics_regression", canonical="quantile_t", source="factor_dsl_np")
class quantile_t(SeriesOperator):
    """t分布分位数函数"""
    metadata = OperatorMetadata(
        name="quantile_t", category="statistics",
        description="t分布分位数函数",
        examples=["quantile_t(0.95, 10)"],
        param_names=["p", "df"], return_type="series",
        tags=["statistics", "probability", "t", "quantile"]
    )
    def _calculate_series(self, p: pd.DataFrame, df: float = 10.0, **kwargs) -> pd.DataFrame:
        return p.apply(lambda col: stats.t.ppf(col, df=df))



# canonical=r_squared backend=pandas_numpy selected=r_squared source=statistics/regression_ex.py
@register_operator(name="r_squared", category="statistics", business_category="statistics_regression", canonical="r_squared", source="factor_dsl_np")
class r_squared(SeriesOperator):
    """R-squared决定系数"""
    metadata = OperatorMetadata(
        name="r_squared", category="statistics",
        description="R-squared决定系数",
        examples=["r_squared(close, market, 20)"],
        param_names=["y", "x", "window"], return_type="series",
        tags=["statistics", "regression", "r_squared"]
    )
    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return rolling_regression(y, x, window=window, min_periods=2, retval="r_squared")



# canonical=rand_exp backend=pandas_numpy selected=rand_exp source=statistics/probability_ops.py
@register_operator(name="rand_exp", category="statistics", business_category="statistics_regression", canonical="rand_exp", source="factor_dsl_np")
class rand_exp(SeriesOperator):
    """指数分布随机数"""
    metadata = OperatorMetadata(
        name="rand_exp", category="statistics",
        description="指数分布随机数",
        examples=["rand_exp(1.0)"],
        param_names=["lam"], return_type="series",
        tags=["statistics", "probability", "random", "exponential"]
    )
    def _calculate_series(self, lam: float = 1.0, **kwargs) -> pd.DataFrame:
        rng = np.random.default_rng()
        scale = 1.0 / lam if lam != 0 else 1.0
        return pd.DataFrame(rng.exponential(scale=scale, size=(1, 1)), columns=["value"], dtype=float)



# canonical=rand_lognormal backend=pandas_numpy selected=rand_lognormal source=statistics/probability_ops.py
@register_operator(name="rand_lognormal", category="statistics", business_category="statistics_regression", canonical="rand_lognormal", source="factor_dsl_np")
class rand_lognormal(SeriesOperator):
    """对数正态分布随机数"""
    metadata = OperatorMetadata(
        name="rand_lognormal", category="statistics",
        description="对数正态分布随机数",
        examples=["rand_lognormal(0, 1)"],
        param_names=["mu", "sigma"], return_type="series",
        tags=["statistics", "probability", "random", "lognormal"]
    )
    def _calculate_series(self, mu: float = 0.0, sigma: float = 1.0, **kwargs) -> pd.DataFrame:
        rng = np.random.default_rng()
        return pd.DataFrame(rng.lognormal(mean=mu, sigma=sigma, size=(1, 1)), columns=["value"], dtype=float)



# canonical=rand_normal backend=pandas_numpy selected=rand_normal source=statistics/probability_ops.py
@register_operator(name="rand_normal", category="statistics", business_category="statistics_regression", canonical="rand_normal", source="factor_dsl_np")
class rand_normal(SeriesOperator):
    """正态分布随机数"""
    metadata = OperatorMetadata(
        name="rand_normal", category="statistics",
        description="正态分布随机数",
        examples=["rand_normal(0, 1)"],
        param_names=["mu", "sigma"], return_type="series",
        tags=["statistics", "probability", "random", "normal"]
    )
    def _calculate_series(self, mu: float = 0.0, sigma: float = 1.0, **kwargs) -> pd.DataFrame:
        rng = np.random.default_rng()
        return pd.DataFrame(rng.normal(loc=mu, scale=sigma, size=(1, 1)), columns=["value"], dtype=float)



# canonical=rand_poisson backend=pandas_numpy selected=rand_poisson source=statistics/probability_ops.py
@register_operator(name="rand_poisson", category="statistics", business_category="statistics_regression", canonical="rand_poisson", source="factor_dsl_np")
class rand_poisson(SeriesOperator):
    """泊松分布随机数"""
    metadata = OperatorMetadata(
        name="rand_poisson", category="statistics",
        description="泊松分布随机数",
        examples=["rand_poisson(5.0)"],
        param_names=["lam"], return_type="series",
        tags=["statistics", "probability", "random", "poisson"]
    )
    def _calculate_series(self, lam: float = 5.0, **kwargs) -> pd.DataFrame:
        rng = np.random.default_rng()
        return pd.DataFrame(rng.poisson(lam=lam, size=(1, 1)), columns=["value"], dtype=float)



# canonical=rand_uniform backend=pandas_numpy selected=rand_uniform source=statistics/probability_ops.py
@register_operator(name="rand_uniform", category="statistics", business_category="statistics_regression", canonical="rand_uniform", source="factor_dsl_np")
class rand_uniform(SeriesOperator):
    """均匀分布随机数"""
    metadata = OperatorMetadata(
        name="rand_uniform", category="statistics",
        description="均匀分布随机数",
        examples=["rand_uniform(0, 1)"],
        param_names=["low", "high"], return_type="series",
        tags=["statistics", "probability", "random", "uniform"]
    )
    def _calculate_series(self, low: float = 0.0, high: float = 1.0, **kwargs) -> pd.DataFrame:
        rng = np.random.default_rng()
        return pd.DataFrame(rng.uniform(low=low, high=high, size=(1, 1)), columns=["value"], dtype=float)



# canonical=regress backend=pandas_numpy selected=regress source=statistics/regression_ex.py
@register_operator(name="regress", category="statistics", business_category="statistics_regression", canonical="regress", source="factor_dsl_np")
class regress(SeriesOperator):
    """通用回归接口，支持指定返回值类型(slope/intercept/r_squared/residual)"""
    metadata = OperatorMetadata(
        name="regress", category="statistics",
        description="通用回归接口，支持指定返回值类型(slope/intercept/r_squared/residual)",
        examples=["regress(close, market, 252, 'slope')"],
        param_names=["y", "x", "window", "retval"], return_type="series",
        tags=["statistics", "regression"]
    )
    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 252,
                          retval: str = 'slope', **kwargs) -> pd.DataFrame:
        rv = retval if retval in ("slope", "intercept", "r_squared", "residual") else "slope"
        return rolling_regression(y, x, window=window, min_periods=3, retval=rv)



# canonical=residual backend=pandas_numpy selected=residual source=statistics/regression_ex.py
@register_operator(name="residual", category="statistics", business_category="statistics_regression", canonical="residual", source="factor_dsl_np")
class residual(SeriesOperator):
    """小写residual回归残差均值"""
    metadata = OperatorMetadata(
        name="residual", category="statistics",
        description="小写residual回归残差均值",
        examples=["residual(close, market, 20)"],
        param_names=["y", "x", "window"], return_type="series",
        tags=["statistics", "regression", "residual"]
    )
    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return rolling_regression(y, x, window=window, min_periods=2, retval="residual")



# canonical=ridge backend=pandas_numpy selected=ridge source=statistics/regression_ex.py
@register_operator(name="ridge", category="statistics", business_category="statistics_regression", canonical="ridge", source="factor_dsl_np")
class Ridge(SeriesOperator):
    """Ridge回归(L2正则化)，单参数时返回序列的L2正则化趋势，双参数时执行Ridge回归"""
    metadata = OperatorMetadata(
        name="ridge", category="statistics",
        description="Ridge回归(L2正则化)，单参数时返回序列的L2正则化趋势，双参数时执行Ridge回归",
        examples=["ridge(close, 20)", "ridge(close, market, 20, 0.1)"],
        param_names=["x", "window_or_y", "window", "alpha"], return_type="series",
        tags=["statistics", "regression", "ridge", "regularization"]
    )
    def _calculate_series(self, x: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        window = 20
        alpha = 0.1
        y = None

        if len(args) >= 1:
            if isinstance(args[0], pd.DataFrame):
                y = args[0]
                if len(args) >= 2:
                    window = int(args[1]) if not isinstance(args[1], pd.DataFrame) else window
                if len(args) >= 3:
                    alpha = float(args[3]) if len(args) > 3 else alpha
            else:
                try:
                    window = int(args[0])
                except (TypeError, ValueError):
                    pass
                if len(args) >= 2:
                    try:
                        alpha = float(args[1])
                    except (TypeError, ValueError):
                        pass

        def _ridge_trend(x_vals):
            n = len(x_vals)
            valid = ~np.isnan(x_vals)
            if valid.sum() < 3:
                return np.nan
            xv = x_vals[valid]
            idx = np.arange(n)[valid]
            if y is None:
                X = np.column_stack([np.ones(len(idx)), (idx - idx.mean()) / (idx.std() + 1e-10)])
                try:
                    XtX = X.T @ X + alpha * np.eye(X.shape[1])
                    XtX_inv = np.linalg.inv(XtX)
                    coeffs = XtX_inv @ (X.T @ xv)
                    return coeffs[-1]
                except np.linalg.LinAlgError:
                    return np.nan
            return np.nan

        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
        for col in x.columns:
            try:
                result[col] = x[col].rolling(window=window, min_periods=3).apply(
                    _ridge_trend, raw=True
                )
            except Exception:
                result[col] = np.nan
        return result



# canonical=sample backend=pandas_numpy selected=sample source=statistics/probability_ops.py
@register_operator(name="sample", category="statistics", business_category="statistics_regression", canonical="sample", source="factor_dsl_np")
class sample(SeriesOperator):
    """随机抽样"""
    metadata = OperatorMetadata(
        name="sample", category="statistics",
        description="随机抽样",
        examples=["sample(close, 10, False)"],
        param_names=["x", "n", "replace"], return_type="series",
        tags=["statistics", "probability", "random", "sample"]
    )
    def _calculate_series(self, x: pd.DataFrame, n: int = 10, replace: bool = False, **kwargs) -> pd.DataFrame:
        rng = np.random.default_rng()
        result = pd.DataFrame(index=range(n), columns=x.columns, dtype=float)
        for col in x.columns:
            col_data = x[col].dropna()
            if len(col_data) == 0:
                result[col] = np.nan
                continue
            sample_size = min(n, len(col_data)) if not replace else n
            sampled = rng.choice(col_data.values, size=sample_size, replace=replace)
            result[col] = pd.Series(sampled)
        return result



# canonical=sem backend=pandas_numpy selected=sem source=statistics/aggregate_ops.py
@register_operator(name="sem", category="statistics", business_category="statistics_regression", canonical="sem", source="factor_dsl_np")
class sem(SeriesOperator):
    """均值标准误"""
    metadata = OperatorMetadata(
        name="sem", category="statistics",
        description="均值标准误",
        examples=["sem(returns)"],
        param_names=["x"], return_type="series",
        tags=["statistics", "aggregate", "sem"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_panel_stat(x, "sem")



# canonical=shuffle backend=pandas_numpy selected=shuffle source=statistics/probability_ops.py
@register_operator(name="shuffle", category="statistics", business_category="statistics_regression", canonical="shuffle", source="factor_dsl_np")
class shuffle(SeriesOperator):
    """随机打乱"""
    metadata = OperatorMetadata(
        name="shuffle", category="statistics",
        description="随机打乱",
        examples=["shuffle(close)"],
        param_names=["x"], return_type="series",
        tags=["statistics", "probability", "random", "shuffle"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)



# canonical=slope backend=pandas_numpy selected=slope source=statistics/regression_ex.py
@register_operator(name="slope", category="statistics", business_category="statistics_regression", canonical="slope", source="factor_dsl_np")
class slope(SeriesOperator):
    """小写slope回归斜率"""
    metadata = OperatorMetadata(
        name="slope", category="statistics",
        description="小写slope回归斜率",
        examples=["slope(close, market, 20)"],
        param_names=["y", "x", "window"], return_type="series",
        tags=["statistics", "regression", "slope"]
    )
    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return rolling_regression(y, x, window=window, min_periods=2, retval="slope")



# canonical=spearman_corr_test backend=pandas_numpy selected=spearman_corr_test source=statistics/hypothesis_ops.py
@register_operator(name="spearman_corr_test", category="statistics", business_category="statistics_regression", canonical="spearman_corr_test", source="factor_dsl_np")
class spearman_corr_test(SeriesOperator):
    """Spearman相关性检验"""
    metadata = OperatorMetadata(
        name="spearman_corr_test", category="statistics",
        description="Spearman相关性检验",
        examples=["spearman_corr_test(x, y)"],
        param_names=["x", "y"], return_type="series",
        tags=["statistics", "hypothesis", "spearman", "correlation"]
    )
    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_bivariate(
            x,
            y,
            lambda a, b: stats.spearmanr(a, b)[1],
            min_periods=3,
        )



# canonical=stationarity_test backend=pandas_numpy selected=stationarity_test source=statistics/hypothesis_ops.py
@register_operator(name="stationarity_test", category="statistics", business_category="statistics_regression", canonical="stationarity_test", source="factor_dsl_np")
class stationarity_test(SeriesOperator):
    """ADF平稳性检验"""
    metadata = OperatorMetadata(
        name="stationarity_test", category="statistics",
        description="ADF平稳性检验",
        examples=["stationarity_test(prices)"],
        param_names=["x"], return_type="series",
        tags=["statistics", "hypothesis", "adf", "stationarity"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        def _adf_pval(v):
            from statsmodels.tsa.stattools import adfuller

            return adfuller(v, autolag="AIC")[1]

        return expanding_univariate(x, _adf_pval, min_periods=10)



# canonical=std_agg backend=pandas_numpy selected=std_agg source=statistics/aggregate_ops.py
@register_operator(name="std_agg", category="statistics", business_category="statistics_regression", canonical="std_agg", source="factor_dsl_np")
class std_agg(SeriesOperator):
    """样本标准差"""
    metadata = OperatorMetadata(
        name="std_agg", category="statistics",
        description="样本标准差",
        examples=["std_agg(returns)"],
        param_names=["x"], return_type="series",
        tags=["statistics", "aggregate", "std"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_panel_stat(x, "std", ddof=1)



# canonical=stdp backend=pandas_numpy selected=stdp source=statistics/aggregate_ops.py
@register_operator(name="stdp", category="statistics", business_category="statistics_regression", canonical="stdp", source="factor_dsl_np")
class stdp(SeriesOperator):
    """总体标准差"""
    metadata = OperatorMetadata(
        name="stdp", category="statistics",
        description="总体标准差",
        examples=["stdp(returns)"],
        param_names=["x"], return_type="series",
        tags=["statistics", "aggregate", "std", "population"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_panel_stat(x, "std", ddof=0)



# canonical=sum_agg backend=pandas_numpy selected=sum_agg source=statistics/aggregate_ops.py
@register_operator(name="sum_agg", category="statistics", business_category="statistics_regression", canonical="sum_agg", source="factor_dsl_np")
class sum_agg(SeriesOperator):
    """列求和"""
    metadata = OperatorMetadata(
        name="sum_agg", category="statistics",
        description="列求和",
        examples=["sum_agg(close)"],
        param_names=["x"], return_type="series",
        tags=["statistics", "aggregate", "sum"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_panel_stat(x, "sum")



# canonical=ttest_one_sample backend=pandas_numpy selected=ttest_one_sample source=statistics/hypothesis_ops.py
@register_operator(name="ttest_one_sample", category="statistics", business_category="statistics_regression", canonical="ttest_one_sample", source="factor_dsl_np")
class ttest_one_sample(SeriesOperator):
    """单样本t检验"""
    metadata = OperatorMetadata(
        name="ttest_one_sample", category="statistics",
        description="单样本t检验",
        examples=["ttest_one_sample(returns, 0)"],
        param_names=["x", "mu"], return_type="series",
        tags=["statistics", "hypothesis", "ttest"]
    )
    def _calculate_series(self, x: pd.DataFrame, mu: float = 0.0, **kwargs) -> pd.DataFrame:
        return expanding_univariate(
            x,
            lambda v: stats.ttest_1samp(v, mu)[1],
            min_periods=2,
        )



# canonical=ttest_paired backend=pandas_numpy selected=ttest_paired source=statistics/hypothesis_ops.py
@register_operator(name="ttest_paired", category="statistics", business_category="statistics_regression", canonical="ttest_paired", source="factor_dsl_np")
class ttest_paired(SeriesOperator):
    """配对t检验"""
    metadata = OperatorMetadata(
        name="ttest_paired", category="statistics",
        description="配对t检验",
        examples=["ttest_paired(before, after)"],
        param_names=["x", "y"], return_type="series",
        tags=["statistics", "hypothesis", "ttest", "paired"]
    )
    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_bivariate(
            x,
            y,
            lambda a, b: stats.ttest_rel(a, b)[1],
            min_periods=2,
        )



# canonical=ttest_two_samples backend=pandas_numpy selected=ttest_two_samples source=statistics/hypothesis_ops.py
@register_operator(name="ttest_two_samples", category="statistics", business_category="statistics_regression", canonical="ttest_two_samples", source="factor_dsl_np")
class ttest_two_samples(SeriesOperator):
    """双样本t检验"""
    metadata = OperatorMetadata(
        name="ttest_two_samples", category="statistics",
        description="双样本t检验",
        examples=["ttest_two_samples(returns_a, returns_b)"],
        param_names=["x", "y"], return_type="series",
        tags=["statistics", "hypothesis", "ttest"]
    )
    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_two_sample(
            x,
            y,
            lambda a, b: stats.ttest_ind(a, b)[1],
            min_periods=2,
        )



# canonical=varp backend=pandas_numpy selected=varp source=statistics/aggregate_ops.py
@register_operator(name="varp", category="statistics", business_category="statistics_regression", canonical="varp", source="factor_dsl_np")
class varp(SeriesOperator):
    """总体方差"""
    metadata = OperatorMetadata(
        name="varp", category="statistics",
        description="总体方差",
        examples=["varp(returns)"],
        param_names=["x"], return_type="series",
        tags=["statistics", "aggregate", "variance", "population"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_panel_stat(x, "var", ddof=0)



# canonical=wavg backend=pandas_numpy selected=wavg source=statistics/regression_ex.py
@register_operator(name="wavg", category="statistics", business_category="statistics_regression", canonical="wavg", source="factor_dsl_np")
class wavg(SeriesOperator):
    """加权平均"""
    metadata = OperatorMetadata(
        name="wavg", category="statistics",
        description="加权平均",
        examples=["wavg(close, volume, 20)"],
        param_names=["x", "w", "window"], return_type="series",
        tags=["statistics", "weighted", "average"]
    )
    def _calculate_series(self, x: pd.DataFrame, w: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        def _wavg(x_vals, w_vals):
            valid = ~(np.isnan(x_vals) | np.isnan(w_vals))
            x_v = x_vals[valid]
            w_v = w_vals[valid]
            if len(x_v) == 0 or np.nansum(w_v) == 0:
                return np.nan
            return np.nansum(x_v * w_v) / np.nansum(w_v)

        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
        for col in x.columns:
            x_col = x[col]
            w_col = w[col] if col in w.columns else w.iloc[:, 0]
            result[col] = x_col.rolling(window=window, min_periods=1).apply(
                lambda x_w: _wavg(x_w, w_col.loc[x_w.index].values), raw=False
            )
        return result



# canonical=wsum backend=pandas_numpy selected=wsum source=statistics/regression_ex.py
@register_operator(name="wsum", category="statistics", business_category="statistics_regression", canonical="wsum", source="factor_dsl_np")
class wsum(SeriesOperator):
    """加权求和"""
    metadata = OperatorMetadata(
        name="wsum", category="statistics",
        description="加权求和",
        examples=["wsum(close, volume, 20)"],
        param_names=["x", "w", "window"], return_type="series",
        tags=["statistics", "weighted", "sum"]
    )
    def _calculate_series(self, x: pd.DataFrame, w: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        def _wsum(x_vals, w_vals):
            valid = ~(np.isnan(x_vals) | np.isnan(w_vals))
            x_v = x_vals[valid]
            w_v = w_vals[valid]
            if len(x_v) == 0:
                return np.nan
            return np.nansum(x_v * w_v)

        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
        for col in x.columns:
            x_col = x[col]
            w_col = w[col] if col in w.columns else w.iloc[:, 0]
            result[col] = x_col.rolling(window=window, min_periods=1).apply(
                lambda x_w: _wsum(x_w, w_col.loc[x_w.index].values), raw=False
            )
        return result

