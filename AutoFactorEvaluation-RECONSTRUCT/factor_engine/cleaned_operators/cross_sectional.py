# -*- coding: utf-8 -*-
"""
截面算子（**同一交易日内、跨标的** 变换）。

语义
----
每个时间行（一行 = 一个 timestamp）在所有 instrument 列上做统计或排序；
例如 ``rank(x)`` 把当日各股票值转为百分位排名，``zscore(x)`` 做截面标准化。

主要算子
--------
- **单序列截面**：``rank``、``zscore``、``scale``、``cs_demean``（``c_demean``）；
- **截面聚合**：``c_mean``、``c_std``、``c_sum``、``c_count``、``c_percentile``；
- **行向量统计**（``row_*``）：对每行多列做 corr/beta/sum 等（面板内多字段组合）。

与 ``group_neutralization.neutralize`` 区别：本模块不做行业/市值分组回归，只做全截面或行内运算。
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

import numpy as np
import pandas as pd
try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None  # type: ignore
try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

# canonical=c_count backend=pandas_numpy selected=c_count source=cross_sectional/c_ops.py
@register_operator(name="c_count", category="cross_sectional", business_category="cross_sectional", canonical="c_count", source="factor_dsl_np")
class CrossSectionalCount(SeriesOperator):
    """截面计数"""

    metadata = OperatorMetadata(
        name="c_count",
        category="cross_sectional",
        description="计算截面非空计数",
        examples=["c_count(close)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "count"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.count(axis=1).to_frame().reindex(columns=x.columns, fill_value=0)



# canonical=c_mean backend=pandas_numpy selected=c_mean source=cross_sectional/c_ops.py
@register_operator(name="c_mean", category="cross_sectional", business_category="cross_sectional", canonical="c_mean", source="factor_dsl_np")
class CrossSectionalMean(SeriesOperator):
    """截面均值"""

    metadata = OperatorMetadata(
        name="c_mean",
        category="cross_sectional",
        description="计算截面均值",
        examples=["c_mean(volume)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "mean"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.mean(axis=1).to_frame().reindex(columns=x.columns, fill_value=0)



# canonical=c_percentile backend=pandas_numpy selected=c_percentile source=cross_sectional/c_ops.py
@register_operator(name="c_percentile", category="cross_sectional", business_category="cross_sectional", canonical="c_percentile", source="factor_dsl_np")
class CrossSectionalPercentile(SeriesOperator):
    """截面分位数"""

    metadata = OperatorMetadata(
        name="c_percentile",
        category="cross_sectional",
        description="返回截面分位数(0-1)",
        examples=["c_percentile(PE, 0.5)"],
        param_names=["x", "p"],
        return_type="series",
        tags=["cross_sectional", "percentile"]
    )

    def _calculate_series(self, x: pd.DataFrame, p: float = 0.5, **kwargs) -> pd.DataFrame:
        return x.rank(pct=True, axis=1) >= p



# canonical=c_std backend=pandas_numpy selected=c_std source=cross_sectional/c_ops.py
@register_operator(name="c_std", category="cross_sectional", business_category="cross_sectional", canonical="c_std", source="factor_dsl_np")
class CrossSectionalStd(SeriesOperator):
    """截面标准差"""

    metadata = OperatorMetadata(
        name="c_std",
        category="cross_sectional",
        description="计算截面标准差",
        examples=["c_std(returns)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "std"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.std(axis=1).to_frame().reindex(columns=x.columns, fill_value=0)



# canonical=c_sum backend=pandas_numpy selected=c_sum source=cross_sectional/c_ops.py
@register_operator(name="c_sum", category="cross_sectional", business_category="cross_sectional", canonical="c_sum", source="factor_dsl_np")
class CrossSectionalSum(SeriesOperator):
    """截面求和"""

    metadata = OperatorMetadata(
        name="c_sum",
        category="cross_sectional",
        description="计算截面求和",
        examples=["c_sum(volume)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "sum"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.sum(axis=1).to_frame().reindex(columns=x.columns, fill_value=0)



# canonical=cs_demean backend=pandas_numpy selected=c_demean source=cross_sectional/c_ops.py
@register_operator(name="c_demean", category="cross_sectional", business_category="cross_sectional", canonical="cs_demean", source="factor_dsl_np")
class CrossSectionalDemean(SeriesOperator):
    """截面中心化"""

    metadata = OperatorMetadata(
        name="c_demean",
        category="cross_sectional",
        description="截面数据减去均值",
        examples=["c_demean(returns)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "demean", "center"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.sub(x.mean(axis=1), axis=0)

# aliases: CS_DEMEAN



# canonical=neutralize backend=pandas_numpy selected=c_neutralize source=cross_sectional/c_ops.py
@register_operator(name="c_neutralize", category="cross_sectional", business_category="cross_sectional", canonical="neutralize", source="factor_dsl_np")
class CrossSectionalNeutralize(SeriesOperator):
    """行业中性化"""

    metadata = OperatorMetadata(
        name="c_neutralize",
        category="cross_sectional",
        description="对因子进行行业中性化处理",
        examples=["c_neutralize(ROE, industry)"],
        param_names=["x", "group"],
        return_type="series",
        tags=["cross_sectional", "neutralize", "industry"]
    )

    def _calculate_series(self, x: pd.DataFrame, group: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        if group is None:
            return x.sub(x.mean(axis=1), axis=0)
        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
        for date in x.index:
            x_slice = x.loc[date]
            if date in group.index:
                group_slice = group.loc[date]
            else:
                group_slice = None
            if group_slice is None or group_slice.isna().all():
                result.loc[date] = x_slice - x_slice.mean()
                continue
            for group_val in group_slice.dropna().unique():
                mask = (group_slice == group_val) & x_slice.notna()
                if mask.sum() > 0:
                    group_data = x_slice[mask]
                    result.loc[date, group_data.index] = group_data - group_data.mean()
        return result

# aliases: NEUTRALIZE, group_neutralize, industry_size_neutralize, size_industry_neutralize



# canonical=rank backend=pandas_numpy selected=rank source=cross_sectional/c_ops.py

# helper for rank
class CrossSectionalRank(SeriesOperator):
    """截面排名"""

    metadata = OperatorMetadata(
        name="c_rank",
        category="cross_sectional",
        description="在同一时间点对所有股票进行排名，返回归一化排名(0-1)",
        examples=["c_rank(PE)", "c_rank(volume)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "rank"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.rank(pct=True, axis=1)

@register_operator(name="rank", category="cross_sectional", business_category="cross_sectional", canonical="rank", source="factor_dsl_np")
class Rank(CrossSectionalRank):
    """截面排名（rank的别名）"""

    metadata = OperatorMetadata(
        name="rank",
        category="cross_sectional",
        description="在同一时间点对所有股票进行排名（与c_rank相同）",
        examples=["rank(close)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "rank"]
    )

# aliases: CS_RANK, RANK, c_rank, cs_rank



# canonical=row_avg backend=pandas_numpy selected=row_avg source=cross_sectional/row_ops.py
@register_operator(name="row_avg", category="cross_sectional", business_category="cross_sectional", canonical="row_avg", source="factor_dsl_np")
class RowAvg(SeriesOperator):

    metadata = OperatorMetadata(
        name="row_avg",
        category="cross_sectional",
        description="对每行（每个日期截面）求均值",
        examples=["row_avg(factor)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "row", "avg"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        row_means = x.mean(axis=1)
        return pd.DataFrame(
            np.tile(row_means.values[:, None], (1, len(x.columns))),
            index=x.index, columns=x.columns, dtype=float
        )



# canonical=row_beta backend=pandas_numpy selected=row_beta source=cross_sectional/row_ops.py
@register_operator(name="row_beta", category="cross_sectional", business_category="cross_sectional", canonical="row_beta", source="factor_dsl_np")
class RowBeta(SeriesOperator):

    metadata = OperatorMetadata(
        name="row_beta",
        category="cross_sectional",
        description="对每行（每个日期截面）计算y对x的beta系数",
        examples=["row_beta(factor_y, factor_x)"],
        param_names=["y", "x"],
        return_type="series",
        tags=["cross_sectional", "row", "beta"]
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        result = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)
        for date in y.index:
            y_slice = y.loc[date]
            x_slice = x.loc[date]
            valid = y_slice.notna() & x_slice.notna()
            if valid.sum() > 1:
                yv = y_slice[valid].values.astype(float)
                xv = x_slice[valid].values.astype(float)
                xv_mean = xv.mean()
                yv_mean = yv.mean()
                xv_centered = xv - xv_mean
                yv_centered = yv - yv_mean
                denom = (xv_centered ** 2).sum()
                if denom != 0:
                    beta_val = (xv_centered * yv_centered).sum() / denom
                else:
                    beta_val = np.nan
            else:
                beta_val = np.nan
            result.loc[date] = beta_val
        return result



# canonical=row_corr backend=pandas_numpy selected=row_corr source=cross_sectional/row_ops.py
@register_operator(name="row_corr", category="cross_sectional", business_category="cross_sectional", canonical="row_corr", source="factor_dsl_np")
class RowCorr(SeriesOperator):

    metadata = OperatorMetadata(
        name="row_corr",
        category="cross_sectional",
        description="对每行（每个日期截面）计算y和x的相关系数",
        examples=["row_corr(factor_y, factor_x)"],
        param_names=["y", "x"],
        return_type="series",
        tags=["cross_sectional", "row", "corr"]
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        result = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)
        for date in y.index:
            y_slice = y.loc[date]
            x_slice = x.loc[date]
            valid = y_slice.notna() & x_slice.notna()
            if valid.sum() > 1:
                yv = y_slice[valid].values.astype(float)
                xv = x_slice[valid].values.astype(float)
                corr_val = np.corrcoef(xv, yv)[0, 1]
                if np.isnan(corr_val):
                    corr_val = np.nan
            else:
                corr_val = np.nan
            result.loc[date] = corr_val
        return result



# canonical=row_count backend=pandas_numpy selected=row_count source=cross_sectional/row_ops.py
@register_operator(name="row_count", category="cross_sectional", business_category="cross_sectional", canonical="row_count", source="factor_dsl_np")
class RowCount(SeriesOperator):

    metadata = OperatorMetadata(
        name="row_count",
        category="cross_sectional",
        description="对每行（每个日期截面）计算非空值数量",
        examples=["row_count(factor)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "row", "count"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        row_counts = x.count(axis=1)
        return pd.DataFrame(
            np.tile(row_counts.values[:, None], (1, len(x.columns))),
            index=x.index, columns=x.columns, dtype=float
        )



# canonical=row_kurt backend=pandas_numpy selected=row_kurt source=cross_sectional/row_ops.py
@register_operator(name="row_kurt", category="cross_sectional", business_category="cross_sectional", canonical="row_kurt", source="factor_dsl_np")
class RowKurt(SeriesOperator):

    metadata = OperatorMetadata(
        name="row_kurt",
        category="cross_sectional",
        description="对每行（每个日期截面）求峰度",
        examples=["row_kurt(factor)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "row", "kurt"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
        for date in x.index:
            row_vals = x.loc[date].dropna().values.astype(float)
            if len(row_vals) > 3:
                kurt_val = float(scipy_stats.kurtosis(row_vals, bias=False))
            else:
                kurt_val = np.nan
            result.loc[date] = kurt_val
        return result



# canonical=row_max backend=pandas_numpy selected=row_max source=cross_sectional/row_ops.py
@register_operator(name="row_max", category="cross_sectional", business_category="cross_sectional", canonical="row_max", source="factor_dsl_np")
class RowMax(SeriesOperator):

    metadata = OperatorMetadata(
        name="row_max",
        category="cross_sectional",
        description="对每行（每个日期截面）求最大值",
        examples=["row_max(factor)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "row", "max"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        row_maxs = x.max(axis=1)
        return pd.DataFrame(
            np.tile(row_maxs.values[:, None], (1, len(x.columns))),
            index=x.index, columns=x.columns, dtype=float
        )



# canonical=row_median backend=pandas_numpy selected=row_median source=cross_sectional/row_ops.py
@register_operator(name="row_median", category="cross_sectional", business_category="cross_sectional", canonical="row_median", source="factor_dsl_np")
class RowMedian(SeriesOperator):

    metadata = OperatorMetadata(
        name="row_median",
        category="cross_sectional",
        description="对每行（每个日期截面）求中位数",
        examples=["row_median(factor)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "row", "median"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        row_medians = x.median(axis=1)
        return pd.DataFrame(
            np.tile(row_medians.values[:, None], (1, len(x.columns))),
            index=x.index, columns=x.columns, dtype=float
        )



# canonical=row_min backend=pandas_numpy selected=row_min source=cross_sectional/row_ops.py
@register_operator(name="row_min", category="cross_sectional", business_category="cross_sectional", canonical="row_min", source="factor_dsl_np")
class RowMin(SeriesOperator):

    metadata = OperatorMetadata(
        name="row_min",
        category="cross_sectional",
        description="对每行（每个日期截面）求最小值",
        examples=["row_min(factor)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "row", "min"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        row_mins = x.min(axis=1)
        return pd.DataFrame(
            np.tile(row_mins.values[:, None], (1, len(x.columns))),
            index=x.index, columns=x.columns, dtype=float
        )



# canonical=row_prod backend=pandas_numpy selected=row_prod source=cross_sectional/row_ops.py
@register_operator(name="row_prod", category="cross_sectional", business_category="cross_sectional", canonical="row_prod", source="factor_dsl_np")
class RowProd(SeriesOperator):

    metadata = OperatorMetadata(
        name="row_prod",
        category="cross_sectional",
        description="对每行（每个日期截面）求乘积",
        examples=["row_prod(factor)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "row", "prod"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        row_prods = x.prod(axis=1)
        return pd.DataFrame(
            np.tile(row_prods.values[:, None], (1, len(x.columns))),
            index=x.index, columns=x.columns, dtype=float
        )



# canonical=row_skew backend=pandas_numpy selected=row_skew source=cross_sectional/row_ops.py
@register_operator(name="row_skew", category="cross_sectional", business_category="cross_sectional", canonical="row_skew", source="factor_dsl_np")
class RowSkew(SeriesOperator):

    metadata = OperatorMetadata(
        name="row_skew",
        category="cross_sectional",
        description="对每行（每个日期截面）求偏度",
        examples=["row_skew(factor)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "row", "skew"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
        for date in x.index:
            row_vals = x.loc[date].dropna().values.astype(float)
            if len(row_vals) > 2:
                skew_val = float(scipy_stats.skew(row_vals, bias=False))
            else:
                skew_val = np.nan
            result.loc[date] = skew_val
        return result



# canonical=row_std backend=pandas_numpy selected=row_std source=cross_sectional/row_ops.py
@register_operator(name="row_std", category="cross_sectional", business_category="cross_sectional", canonical="row_std", source="factor_dsl_np")
class RowStd(SeriesOperator):

    metadata = OperatorMetadata(
        name="row_std",
        category="cross_sectional",
        description="对每行（每个日期截面）求标准差",
        examples=["row_std(factor)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "row", "std"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        row_stds = x.std(axis=1)
        return pd.DataFrame(
            np.tile(row_stds.values[:, None], (1, len(x.columns))),
            index=x.index, columns=x.columns, dtype=float
        )



# canonical=row_sum backend=pandas_numpy selected=row_sum source=cross_sectional/row_ops.py
@register_operator(name="row_sum", category="cross_sectional", business_category="cross_sectional", canonical="row_sum", source="factor_dsl_np")
class RowSum(SeriesOperator):

    metadata = OperatorMetadata(
        name="row_sum",
        category="cross_sectional",
        description="对每行（每个日期截面）求和",
        examples=["row_sum(factor)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "row", "sum"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        row_sums = x.sum(axis=1)
        return pd.DataFrame(
            np.tile(row_sums.values[:, None], (1, len(x.columns))),
            index=x.index, columns=x.columns, dtype=float
        )



# canonical=row_var backend=pandas_numpy selected=row_var source=cross_sectional/row_ops.py
@register_operator(name="row_var", category="cross_sectional", business_category="cross_sectional", canonical="row_var", source="factor_dsl_np")
class RowVar(SeriesOperator):

    metadata = OperatorMetadata(
        name="row_var",
        category="cross_sectional",
        description="对每行（每个日期截面）求方差",
        examples=["row_var(factor)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "row", "var"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        row_vars = x.var(axis=1)
        return pd.DataFrame(
            np.tile(row_vars.values[:, None], (1, len(x.columns))),
            index=x.index, columns=x.columns, dtype=float
        )



# canonical=scale backend=pandas_numpy selected=scale source=cross_sectional/c_ops.py

# helper for scale
class CrossSectionalScale(SeriesOperator):
    """截面缩放"""

    metadata = OperatorMetadata(
        name="c_scale",
        category="cross_sectional",
        description="缩放数据使sum(abs(x))=指定值",
        examples=["c_scale(Return, 1)", "c_scale(factor, 100)"],
        param_names=["x", "to"],
        return_type="series",
        tags=["cross_sectional", "scale"]
    )

    def _calculate_series(self, x: pd.DataFrame, to: float = 1, **kwargs) -> pd.DataFrame:
        abs_sum = x.abs().sum(axis=1).replace(0, 1)
        return x.mul(to / abs_sum, axis=0)

@register_operator(name="scale", category="cross_sectional", business_category="cross_sectional", canonical="scale", source="factor_dsl_np")
class Scale(CrossSectionalScale):

    metadata = OperatorMetadata(
        name="scale",
        category="cross_sectional",
        description="缩放数据使sum(abs(x))=指定值（与c_scale相同）",
        examples=["scale(Return, 1)", "scale(factor, 100)"],
        param_names=["x", "to"],
        return_type="series",
        tags=["cross_sectional", "scale"]
    )

# aliases: SCALE, c_scale



# canonical=zscore backend=pandas_numpy selected=zscore source=cross_sectional/c_ops.py

# helper for zscore
class CrossSectionalZscore(SeriesOperator):
    """截面Z-Score标准化"""

    metadata = OperatorMetadata(
        name="c_zscore",
        category="cross_sectional",
        description="对截面数据进行Z-Score标准化 (x-mean)/std",
        examples=["c_zscore(Return)", "c_zscore(ROE)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "standardize", "zscore"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        mean = x.mean(axis=1)
        std = x.std(axis=1)
        std = std.replace(0, 1)
        return (x.sub(mean, axis=0)).div(std, axis=0)

@register_operator(name="zscore", category="cross_sectional", business_category="cross_sectional", canonical="zscore", source="factor_dsl_np")
class Zscore(CrossSectionalZscore):
    """截面Z-Score（zscore的别名）"""

    metadata = OperatorMetadata(
        name="zscore",
        category="cross_sectional",
        description="对截面数据进行Z-Score标准化（与c_zscore相同）",
        examples=["zscore(PE)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "standardize", "zscore"]
    )

# aliases: CS_ZSCORE, ZSCORE, c_zscore, cs_zscore


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


# canonical=c_count backend=polars selected=c_count source=cross_sectional/c_ops_polars.py
@register_operator(name="c_count", category="cross_sectional", business_category="cross_sectional", canonical="c_count", source="factor_dsl_np")
class CrossSectionalCountPolars(SeriesOperator):
    """截面计数"""

    metadata = OperatorMetadata(
        name="c_count",
        category="cross_sectional",
        description="计算截面非空计数",
        examples=["c_count(close)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "count"]
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]

        # 计算每行非空计数
        count_expr = sum(pl.col(c).is_not_null().cast(pl.Int32) for c in numeric_cols)

        return x.with_columns([
            count_expr.alias(c) for c in numeric_cols
        ])



# canonical=c_mean backend=polars selected=c_mean source=cross_sectional/c_ops_polars.py
@register_operator(name="c_mean", category="cross_sectional", business_category="cross_sectional", canonical="c_mean", source="factor_dsl_np")
class CrossSectionalMeanPolars(SeriesOperator):
    """截面均值"""

    metadata = OperatorMetadata(
        name="c_mean",
        category="cross_sectional",
        description="计算截面均值",
        examples=["c_mean(volume)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "mean"]
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]

        # 计算每行均值
        mean_expr = sum(pl.col(c) for c in numeric_cols) / len(numeric_cols)

        return x.with_columns([
            mean_expr.alias(c) for c in numeric_cols
        ])



# canonical=c_percentile backend=polars selected=c_percentile source=cross_sectional/c_ops_polars.py
@register_operator(name="c_percentile", category="cross_sectional", business_category="cross_sectional", canonical="c_percentile", source="factor_dsl_np")
class CrossSectionalPercentilePolars(SeriesOperator):
    """截面分位数"""

    metadata = OperatorMetadata(
        name="c_percentile",
        category="cross_sectional",
        description="返回截面分位数(0-1)",
        examples=["c_percentile(PE, 0.5)"],
        param_names=["x", "p"],
        return_type="series",
        tags=["cross_sectional", "percentile"]
    )

    def _calculate_series(self, x: pl.DataFrame, p: float = 0.5, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]

        # 计算每行的分位数
        arr = x.select(numeric_cols).to_numpy()
        percentiles = np.nanpercentile(arr, p * 100, axis=1, keepdims=True)

        # 比较
        result_arr = (arr >= percentiles).astype(float)

        # 转换回 Polars
        result_df = pl.DataFrame(result_arr, schema=numeric_cols)

        if 'date' in x.columns:
            result_df = result_df.with_columns([x['date']])

        return result_df



# canonical=c_std backend=polars selected=c_std source=cross_sectional/c_ops_polars.py
@register_operator(name="c_std", category="cross_sectional", business_category="cross_sectional", canonical="c_std", source="factor_dsl_np")
class CrossSectionalStdPolars(SeriesOperator):
    """截面标准差"""

    metadata = OperatorMetadata(
        name="c_std",
        category="cross_sectional",
        description="计算截面标准差",
        examples=["c_std(returns)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "std"]
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]

        # 计算每行标准差
        arr = x.select(numeric_cols).to_numpy()
        stds = np.nanstd(arr, axis=1, keepdims=True)

        # 转换回 Polars
        result_df = pl.DataFrame(np.repeat(stds, len(numeric_cols), axis=1), schema=numeric_cols)

        if 'date' in x.columns:
            result_df = result_df.with_columns([x['date']])

        return result_df



# canonical=c_sum backend=polars selected=c_sum source=cross_sectional/c_ops_polars.py
@register_operator(name="c_sum", category="cross_sectional", business_category="cross_sectional", canonical="c_sum", source="factor_dsl_np")
class CrossSectionalSumPolars(SeriesOperator):
    """截面求和"""

    metadata = OperatorMetadata(
        name="c_sum",
        category="cross_sectional",
        description="计算截面求和",
        examples=["c_sum(volume)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "sum"]
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]

        # 计算每行求和
        sum_expr = sum(pl.col(c) for c in numeric_cols)

        return x.with_columns([
            sum_expr.alias(c) for c in numeric_cols
        ])



# canonical=cs_demean backend=polars selected=c_demean source=cross_sectional/c_ops_polars.py
@register_operator(name="c_demean", category="cross_sectional", business_category="cross_sectional", canonical="cs_demean", source="factor_dsl_np")
class CrossSectionalDemeanPolars(SeriesOperator):
    """截面中心化"""

    metadata = OperatorMetadata(
        name="c_demean",
        category="cross_sectional",
        description="截面数据减去均值",
        examples=["c_demean(returns)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "demean", "center"]
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]

        # 计算每行均值
        mean_expr = sum(pl.col(c) for c in numeric_cols) / len(numeric_cols)

        return x.with_columns([
            (pl.col(c) - mean_expr).alias(c) for c in numeric_cols
        ])

# aliases: CS_DEMEAN



# canonical=neutralize backend=polars selected=c_neutralize source=cross_sectional/c_ops_polars.py
@register_operator(name="c_neutralize", category="cross_sectional", business_category="cross_sectional", canonical="neutralize", source="factor_dsl_np")
class CrossSectionalNeutralizePolars(SeriesOperator):
    """行业中性化"""

    metadata = OperatorMetadata(
        name="c_neutralize",
        category="cross_sectional",
        description="对因子进行行业中性化处理",
        examples=["c_neutralize(ROE, industry)"],
        param_names=["x", "group"],
        return_type="series",
        tags=["cross_sectional", "neutralize", "industry"]
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame = None, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]

        if group is None:
            # 如果没有分组信息，使用简单的截面中心化
            return self._calculate_series(x, **kwargs)

        # 将数据转换为 numpy 进行处理
        x_arr = x.select(numeric_cols).to_numpy()
        group_arr = group.select(numeric_cols).to_numpy()

        result_arr = np.full_like(x_arr, np.nan)

        for row_idx in range(len(x_arr)):
            row_data = x_arr[row_idx]
            row_group = group_arr[row_idx]

            # 获取唯一的组值
            unique_groups = np.unique(row_group[~np.isnan(row_group)])

            if len(unique_groups) == 0:
                # 没有分组信息，使用整行均值
                row_mean = np.nanmean(row_data)
                result_arr[row_idx] = row_data - row_mean
            else:
                # 按组计算均值并去中心化
                for g in unique_groups:
                    mask = row_group == g
                    if np.any(mask):
                        group_mean = np.nanmean(row_data[mask])
                        result_arr[row_idx, mask] = row_data[mask] - group_mean

        # 转换回 Polars
        result_df = pl.DataFrame(result_arr, schema=numeric_cols)

        if 'date' in x.columns:
            result_df = result_df.with_columns([x['date']])

        return result_df

# aliases: NEUTRALIZE, group_neutralize, industry_size_neutralize, size_industry_neutralize



# canonical=rank backend=polars selected=rank source=cross_sectional/c_ops_polars.py

# helper for rank
class CrossSectionalRank(SeriesOperator):
    """截面排名"""

    metadata = OperatorMetadata(
        name="c_rank",
        category="cross_sectional",
        description="在同一时间点对所有股票进行排名，返回归一化排名(0-1)",
        examples=["c_rank(PE)", "c_rank(volume)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "rank"]
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        # 使用 Numba 加速的截面排名（按行）
        return apply_numba_rank(x, axis=1)

@register_operator(name="rank", category="cross_sectional", business_category="cross_sectional", canonical="rank", source="factor_dsl_np")
class RankPolars(CrossSectionalRank):
    """截面排名（rank的别名）"""

    metadata = OperatorMetadata(
        name="rank",
        category="cross_sectional",
        description="在同一时间点对所有股票进行排名（与c_rank相同）",
        examples=["rank(close)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "rank"]
    )

# aliases: CS_RANK, RANK, c_rank, cs_rank



# canonical=scale backend=polars selected=scale source=cross_sectional/c_ops_polars.py

# helper for scale
class CrossSectionalScale(SeriesOperator):
    """截面缩放"""

    metadata = OperatorMetadata(
        name="c_scale",
        category="cross_sectional",
        description="缩放数据使sum(abs(x))=指定值",
        examples=["c_scale(Return, 1)", "c_scale(factor, 100)"],
        param_names=["x", "to"],
        return_type="series",
        tags=["cross_sectional", "scale"]
    )

    def _calculate_series(self, x: pl.DataFrame, to: float = 1, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]

        # 计算每行的绝对值之和
        abs_sum_expr = sum(pl.col(c).abs() for c in numeric_cols)
        scale_factor = to / pl.when(abs_sum_expr == 0).then(1).otherwise(abs_sum_expr)

        return x.with_columns([
            (pl.col(c) * scale_factor).alias(c) for c in numeric_cols
        ])

@register_operator(name="scale", category="cross_sectional", business_category="cross_sectional", canonical="scale", source="factor_dsl_np")
class ScalePolars(CrossSectionalScale):

    metadata = OperatorMetadata(
        name="scale",
        category="cross_sectional",
        description="缩放数据使sum(abs(x))=指定值（与c_scale相同）",
        examples=["scale(Return, 1)", "scale(factor, 100)"],
        param_names=["x", "to"],
        return_type="series",
        tags=["cross_sectional", "scale"]
    )

# aliases: SCALE, c_scale



# canonical=winsorize backend=polars selected=c_winsorize source=cross_sectional/c_ops_polars.py
@register_operator(name="c_winsorize", category="cross_sectional", business_category="cross_sectional", canonical="winsorize", source="factor_dsl_np")
class CrossSectionalWinsorizePolars(SeriesOperator):
    """截面去极值"""

    metadata = OperatorMetadata(
        name="c_winsorize",
        category="cross_sectional",
        description="对截面数据进行缩尾处理",
        examples=["c_winsorize(ROE, 0.05, 0.95)"],
        param_names=["x", "min_pct", "max_pct"],
        return_type="series",
        tags=["cross_sectional", "winsorize", "outlier"]
    )

    def _calculate_series(self, x: pl.DataFrame, min_pct: float = 0.05, max_pct: float = 0.95, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]

        # 使用 Numba 计算分位数并裁剪
        arr = x.select(numeric_cols).to_numpy()

        # 计算每行的分位数
        lower = np.nanpercentile(arr, min_pct * 100, axis=1, keepdims=True)
        upper = np.nanpercentile(arr, max_pct * 100, axis=1, keepdims=True)

        # 裁剪
        clipped = np.clip(arr, lower, upper)

        # 转换回 Polars
        result_df = pl.DataFrame(clipped, schema=numeric_cols)

        if 'date' in x.columns:
            result_df = result_df.with_columns([x['date']])

        return result_df

# aliases: WINSORIZE



# canonical=zscore backend=polars selected=zscore source=cross_sectional/c_ops_polars.py

# helper for zscore
class CrossSectionalZscore(SeriesOperator):
    """截面Z-Score标准化"""

    metadata = OperatorMetadata(
        name="c_zscore",
        category="cross_sectional",
        description="对截面数据进行Z-Score标准化 (x-mean)/std",
        examples=["c_zscore(Return)", "c_zscore(ROE)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "standardize", "zscore"]
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        # 使用 Numba 加速的截面 Z-Score（按行）
        return apply_numba_zscore(x, axis=1)

@register_operator(name="zscore", category="cross_sectional", business_category="cross_sectional", canonical="zscore", source="factor_dsl_np")
class ZscorePolars(CrossSectionalZscore):
    """截面Z-Score（zscore的别名）"""

    metadata = OperatorMetadata(
        name="zscore",
        category="cross_sectional",
        description="对截面数据进行Z-Score标准化（与c_zscore相同）",
        examples=["zscore(PE)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "standardize", "zscore"]
    )

# aliases: CS_ZSCORE, ZSCORE, c_zscore, cs_zscore


@register_operator(name="cs_resid", category="cross_sectional", business_category="cross_sectional", canonical="cs_resid", source="lqtp_numpy")
class LqtpCsresidOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="cs_resid",
        category="cross_sectional",
        description="LQTP numpy implementation",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._lqtp_numpy import cs_resid_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: cs_resid_(s.values, **kwargs) if kwargs else cs_resid_(s.values))
        return cs_resid_(*args, **kwargs)


@register_operator(name="cs_regression", category="cross_sectional", business_category="cross_sectional", canonical="cs_regression", source="lqtp_numpy")
class LqtpCsregressionOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="cs_regression",
        category="cross_sectional",
        description="LQTP numpy implementation",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._lqtp_numpy import cs_regression_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: cs_regression_(s.values, **kwargs) if kwargs else cs_regression_(s.values))
        return cs_regression_(*args, **kwargs)
