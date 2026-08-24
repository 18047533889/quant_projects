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

from factor_engine.cleaned_operators.common.cs_broadcast import (
    broadcast_row_stat,
    broadcast_row_stat_all_null_null,
    cs_rank_01,
    cs_rank_pct,
)
from factor_engine.cleaned_operators.base import (
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

# ---------------------------------------------------------------------------
# Rank tie-method policy (R40 #197).
#
# ``rank(method="first")`` assigns ranks by row/column *iteration* order — the
# result depends on the physical ordering of instruments and is not a stable
# execution identity.  Production rejects ``method="first"`` (a deterministic
# ``method="average"``/``"min"``/``"max"``/``"dense"`` must be declared).  If a
# rank must order ties by instrument, the caller sorts the cross-section by the
# canonical InstrumentKey before ranking — this gate is the single authority.
# ---------------------------------------------------------------------------
_RANK_DETERMINISTIC_METHODS = frozenset({"average", "min", "max", "dense"})


def check_rank_method(method: str | None, *, production: bool = True, canonical: str = "") -> str:
    """Production gate for the rank tie method (R40 #197).

    ``method="first"`` is rejected in production (iteration-order dependent).
    ``None`` resolves to the deterministic default ``"average"``.  An unknown
    method raises rather than silently degrading.  This policy is NOT a mining
    continuous-search dimension — the gate always pins a deterministic method.
    """
    eff = (method or "average").lower()
    if production and eff == "first":
        raise ValueError(
            f"{canonical or 'rank'}: method='first' is rejected in production "
            "(tie-break by iteration order is not a stable execution identity; "
            "use a deterministic method or sort by canonical InstrumentKey first — "
            "R40 #197)"
        )
    if eff not in _RANK_DETERMINISTIC_METHODS and eff != "first":
        raise ValueError(
            f"{canonical or 'rank'}: unknown rank method {method!r}; expected one "
            f"of {sorted(_RANK_DETERMINISTIC_METHODS)} (research may use 'first' only "
            "explicitly)"
        )
    return eff


def _finite_stats_input(arr: np.ndarray) -> np.ndarray:
    """Map ±Inf → NaN so ``np.nan*`` statistics ignore Inf as a valid sample.

    R40 #188: ``np.nanmean/np.nanstd/np.nanmedian/np.nanpercentile`` treat ±Inf
    as a real observation, which manufactures garbage cross-sectional stats (a
    single Inf column makes the whole row's mean Inf).  Masking non-finite to
    NaN first makes every fallback use the same finite-sample policy as the
    ``np.isfinite``-masked paths.
    """
    out = np.array(arr, dtype=np.float64, copy=True)
    out[~np.isfinite(out)] = np.nan
    return out
try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore


class CrossSectionSampleMask:
    """统一截面统计样本有效性（R19-027..029）。

    生产因子数值样本的统计有效性统一为 ``np.isfinite``：±Inf 与 NaN 一样
    不可用，不得计入截面统计的分母/分子（``count``/``mean``/``std``/``sum``/
    ``percentile``/``mad`` 等共享）。``mask`` 把非有限单元替换为 NaN，使 pandas
    行统计在 ``skipna=True`` 语义下天然排除 Inf。
    """

    sample_validity = "finite"

    @staticmethod
    def mask(x: pd.DataFrame) -> pd.DataFrame:
        arr = x.to_numpy(dtype=float, copy=False)
        return pd.DataFrame(
            np.where(np.isfinite(arr), arr, np.nan),
            index=x.index,
            columns=x.columns,
            dtype=float,
        )


# canonical=c_count backend=pandas_numpy selected=c_count source=cross_sectional/c_ops.py
@register_operator(name="c_count", category="cross_sectional", business_category="cross_sectional", canonical="c_count", source="factor_dsl_np")
class CrossSectionalCount(SeriesOperator):
    """截面计数"""

    metadata = OperatorMetadata(
        name="c_count",
        category="cross_sectional",
        description="计算截面有限值计数（sample_validity=finite：±Inf 不计入）",
        examples=["c_count(close)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "count"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        xm = CrossSectionSampleMask.mask(x)
        return broadcast_row_stat(x, xm.count(axis=1))



# canonical=c_mean backend=pandas_numpy selected=c_mean source=cross_sectional/c_ops.py
@register_operator(name="c_mean", category="cross_sectional", business_category="cross_sectional", canonical="c_mean", source="factor_dsl_np")
class CrossSectionalMean(SeriesOperator):
    """截面均值"""

    metadata = OperatorMetadata(
        name="c_mean",
        category="cross_sectional",
        description="计算截面均值（sample_validity=finite：±Inf 不计入）",
        examples=["c_mean(volume)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "mean"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        xm = CrossSectionSampleMask.mask(x)
        return broadcast_row_stat_all_null_null(xm, xm.mean(axis=1))



# canonical=c_percentile backend=pandas_numpy selected=c_percentile source=cross_sectional/c_ops.py
@register_operator(name="c_percentile", category="cross_sectional", business_category="cross_sectional", canonical="c_percentile", source="factor_dsl_np")
class CrossSectionalPercentile(SeriesOperator):
    """截面分位数"""

    metadata = OperatorMetadata(
        name="c_percentile",
        category="cross_sectional",
        description="截面 p 分位数值（广播到各列，非百分位排名；sample_validity=finite）",
        examples=["c_percentile(PE, 0.5)"],
        param_names=["x", "p"],
        return_type="series",
        tags=["cross_sectional", "percentile", "pit_safe"],
    )

    def _calculate_series(self, x: pd.DataFrame, p: float = 0.5, **kwargs) -> pd.DataFrame:
        xm = CrossSectionSampleMask.mask(x)
        q = xm.quantile(float(p), axis=1)
        return broadcast_row_stat(x, q)



# canonical=c_std backend=pandas_numpy selected=c_std source=cross_sectional/c_ops.py
@register_operator(name="c_std", category="cross_sectional", business_category="cross_sectional", canonical="c_std", source="factor_dsl_np")
class CrossSectionalStd(SeriesOperator):
    """截面标准差"""

    metadata = OperatorMetadata(
        name="c_std",
        category="cross_sectional",
        description="计算截面标准差（sample_validity=finite：±Inf 不计入）",
        examples=["c_std(returns)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "std"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        xm = CrossSectionSampleMask.mask(x)
        return broadcast_row_stat_all_null_null(xm, xm.std(axis=1))


@register_operator(name="cs_mad", category="cross_sectional", business_category="cross_sectional", canonical="cs_mad", source="factor_dsl_np")
class CrossSectionalMad(SeriesOperator):
    """截面中位绝对偏差（MAD，广播到各列）"""
    metadata = OperatorMetadata(
        name="cs_mad",
        category="cross_sectional",
        description="截面中位绝对偏差（MAD，广播到各列；sample_validity=finite）",
        examples=["cs_mad(PE)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "mad", "pit_safe"],
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        xm = CrossSectionSampleMask.mask(x)
        med = xm.median(axis=1)
        mad = xm.sub(med, axis=0).abs().median(axis=1)
        return broadcast_row_stat(x, mad)


@register_operator(name="cs_mad_zscore", category="cross_sectional", business_category="cross_sectional", canonical="cs_mad_zscore", source="factor_dsl_np")
class CrossSectionalMadZscore(SeriesOperator):
    """raw-MAD 标准化得分：(x - median) / MAD（不乘 0.6745）"""
    metadata = OperatorMetadata(
        name="cs_mad_zscore",
        category="cross_sectional",
        description="raw-MAD standardized score：(x - median) / MAD；不做 0.6745 缩放，"
                    "MAD 不乘 1.4826。要 scaled robust zscore 请显式构造：0.6745*cs_mad_zscore(x)。",
        examples=["cs_mad_zscore(PE)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "mad", "zscore", "pit_safe"],
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        xm = CrossSectionSampleMask.mask(x)
        med = xm.median(axis=1)
        mad = xm.sub(med, axis=0).abs().median(axis=1).replace(0, np.nan)
        return xm.sub(med, axis=0).div(mad, axis=0)


# canonical=c_sum backend=pandas_numpy selected=c_sum source=cross_sectional/c_ops.py
@register_operator(name="c_sum", category="cross_sectional", business_category="cross_sectional", canonical="c_sum", source="factor_dsl_np")
class CrossSectionalSum(SeriesOperator):
    """截面求和"""

    metadata = OperatorMetadata(
        name="c_sum",
        category="cross_sectional",
        description="计算截面求和（sample_validity=finite：±Inf 不计入）",
        examples=["c_sum(volume)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "sum"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        xm = CrossSectionSampleMask.mask(x)
        return broadcast_row_stat_all_null_null(xm, xm.sum(axis=1))



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
        # R19-063: ``group is None`` 不再退化为全局 demean（那是 cs_demean 的职责，
        # 是另一个因子）。缺 group → fail-closed：输出全 NaN。全局 demean 请显式调用
        # ``cs_demean``/``c_demean``。
        if group is None:
            return pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
        for date in x.index:
            x_slice = x.loc[date]
            if date in group.index:
                group_slice = group.loc[date]
            else:
                group_slice = None
            # R19-064: 该日 group 全 missing 不能 silent global demean -> 保持 NaN。
            if group_slice is None or group_slice.isna().all():
                result.loc[date] = np.nan
                continue
            # R19-065: group membership 部分 missing 时，只有 group known 的股票参与
            # 组中性化；unknown group 的 cell 保持 NaN（result 初始为 NaN）。
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
    """截面 pandas 百分位排名（rank/count）。

    R19-088: 该 semantic 与 ``cs_rank_01`` 不同 —— 它是 pandas ``rank(pct=True)``
    （最小值 1/n，非 0-1 归一化，singleton=1.0），而 ``cs_rank_01`` 是 0-1 归一化
    （singleton=0.5）。两者是两种 rank semantic，不是 duplicate，保持分开。
    """

    metadata = OperatorMetadata(
        name="c_rank",
        category="cross_sectional",
        description="截面 pandas 百分位排名 rank/count（最小值 1/n，非 0-1）",
        examples=["c_rank(PE)", "c_rank(volume)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "rank", "pit_safe"],
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.rank(pct=True, axis=1)

# R19-066: ``rank`` 与 ``cs_rank_01`` 是 exact duplicate（同一实现 cs_rank_01）——
# 数值等价已由 tests/operators/test_r19_truthiness_domain.py 证明。合并为一个
# canonical + alias 需要 registry/surface 协调（unregister + register_alias），
# 由中央协调处理；此处两个都保留并明示同一实现。
@register_operator(name="rank", category="cross_sectional", business_category="cross_sectional", canonical="rank", source="factor_dsl_np")
class Rank(SeriesOperator):
    """截面 0-1 排名（与 cs_rank_01 exact duplicate）。"""

    metadata = OperatorMetadata(
        name="rank",
        category="cross_sectional",
        description="截面 0-1 排名（单有效值 → 0.5；与 cs_rank_01 同实现）；pandas 百分位排名见 rank_pct",
        examples=["rank(close)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "rank", "pit_safe"],
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return cs_rank_01(x)


@register_operator(
    name="rank_pct",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="rank_pct",
    source="factor_dsl_np",
)
# R19-067: ``rank_pct`` 与 ``cs_pct_rank`` 是 exact duplicate（同一实现
# CrossSectionalRank._calculate_series = x.rank(pct=True)）。
class RankPct(CrossSectionalRank):
    """截面 pandas 百分位排名 rank/count（与 cs_pct_rank exact duplicate）"""
    metadata = OperatorMetadata(
        name="rank_pct",
        category="cross_sectional",
        description="截面 pandas 百分位排名 rank/count（与 cs_pct_rank 同实现）",
        examples=["rank_pct(close)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "rank", "pit_safe"],
    )


@register_operator(
    name="cs_quantile",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_quantile",
    source="factor_dsl_np",
)
# R19-068: ``cs_quantile`` 与 ``c_percentile`` 是 exact duplicate（同一实现
# CrossSectionalPercentile._calculate_series = x.quantile(p, axis=1) 广播）。
class CsQuantile(CrossSectionalPercentile):
    """截面 p 分位数值（广播到各列；与 c_percentile exact duplicate）"""
    metadata = OperatorMetadata(
        name="cs_quantile",
        category="cross_sectional",
        description="截面 p 分位数值（广播到各列；与 c_percentile 同实现；百分位排名见 cs_pct_rank）",
        examples=["cs_quantile(PE, 0.5)"],
        param_names=["x", "p"],
        return_type="series",
        tags=["cross_sectional", "quantile", "pit_safe"],
    )


@register_operator(
    name="cs_pct_rank",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_pct_rank",
    source="factor_dsl_np",
)
class CsPctRank(CrossSectionalRank):
    """截面百分位排名（与 rank_pct exact duplicate）"""
    metadata = OperatorMetadata(
        name="cs_pct_rank",
        category="cross_sectional",
        description="截面百分位排名（与 rank_pct 同实现）",
        examples=["cs_pct_rank(close)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "rank", "pit_safe"],
    )


@register_operator(
    name="cs_rank_01",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_rank_01",
    source="factor_dsl_np",
)
class CsRank01(SeriesOperator):
    """截面 0-1 排名（与 rank exact duplicate，canonical 实现）"""
    metadata = OperatorMetadata(
        name="cs_rank_01",
        category="cross_sectional",
        description="截面 0-1 排名（单有效值 → 0.5；与 rank 同实现）",
        examples=["cs_rank_01(close)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "rank", "pit_safe"],
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return cs_rank_01(x)

# aliases: CS_RANK, RANK, c_rank, cs_rank



# canonical=row_avg backend=pandas_numpy selected=row_avg source=cross_sectional/row_ops.py
@register_operator(name="row_avg", category="cross_sectional", business_category="cross_sectional", canonical="row_avg", source="factor_dsl_np")
class RowAvg(SeriesOperator):
    """对每行（每个日期截面）求均值"""

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
    """对每行（每个日期截面）计算y对x的beta系数"""

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
            valid = np.isfinite(y_slice.to_numpy(dtype=float)) & np.isfinite(x_slice.to_numpy(dtype=float))
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
    """对每行（每个日期截面）计算y和x的相关系数"""

    metadata = OperatorMetadata(
        name="row_corr",
        category="cross_sectional",
        description="对每行（每个日期截面）计算y和x的相关系数",
        examples=["row_corr(factor_y, factor_x)"],
        param_names=["x", "y"],
        return_type="series",
        tags=["cross_sectional", "row", "corr"]
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        result = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)
        for date in y.index:
            y_slice = y.loc[date]
            x_slice = x.loc[date]
            valid = np.isfinite(y_slice.to_numpy(dtype=float)) & np.isfinite(x_slice.to_numpy(dtype=float))
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
    """对每行（每个日期截面）计算非空值数量"""

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
        xm = CrossSectionSampleMask.mask(x)
        row_counts = xm.count(axis=1)
        return pd.DataFrame(
            np.tile(row_counts.values[:, None], (1, len(x.columns))),
            index=x.index, columns=x.columns, dtype=float
        )



# canonical=row_kurt backend=pandas_numpy selected=row_kurt source=cross_sectional/row_ops.py
@register_operator(name="row_kurt", category="cross_sectional", business_category="cross_sectional", canonical="row_kurt", source="factor_dsl_np")
class RowKurt(SeriesOperator):
    """对每行（每个日期截面）求峰度"""

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
            row_vals = x.loc[date].to_numpy(dtype=float)
            row_vals = row_vals[np.isfinite(row_vals)]
            if len(row_vals) > 3:
                kurt_val = float(scipy_stats.kurtosis(row_vals, bias=False))
            else:
                kurt_val = np.nan
            result.loc[date] = kurt_val
        return result



# canonical=row_max backend=pandas_numpy selected=row_max source=cross_sectional/row_ops.py
@register_operator(name="row_max", category="cross_sectional", business_category="cross_sectional", canonical="row_max", source="factor_dsl_np")
class RowMax(SeriesOperator):
    """对每行（每个日期截面）求最大值"""

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
    """对每行（每个日期截面）求中位数"""

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
    """对每行（每个日期截面）求最小值"""

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
    """对每行（每个日期截面）求乘积"""

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
    """对每行（每个日期截面）求偏度"""

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
            row_vals = x.loc[date].to_numpy(dtype=float)
            row_vals = row_vals[np.isfinite(row_vals)]
            if len(row_vals) > 2:
                skew_val = float(scipy_stats.skew(row_vals, bias=False))
            else:
                skew_val = np.nan
            result.loc[date] = skew_val
        return result



# canonical=row_std backend=pandas_numpy selected=row_std source=cross_sectional/row_ops.py
@register_operator(name="row_std", category="cross_sectional", business_category="cross_sectional", canonical="row_std", source="factor_dsl_np")
class RowStd(SeriesOperator):
    """对每行（每个日期截面）求标准差"""

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
    """对每行（每个日期截面）求和"""

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
    """对每行（每个日期截面）求方差"""

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
    """缩放数据使sum(abs(x))=指定值（与c_scale相同）"""

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


@register_operator(name="cs_resid", category="cross_sectional", business_category="cross_sectional", canonical="cs_resid", source="factor_dsl_np")
class CSResid(SeriesOperator):
    """截面线性回归残差 y - (α + βx)"""
    metadata = OperatorMetadata(
        name="cs_resid",
        category="cross_sectional",
        description="截面线性回归残差 y - (α + βx)",
        examples=["cs_resid(factor, size)"],
        param_names=["y", "x"],
        return_type="series",
        tags=["cross_sectional", "regression", "residual"],
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        from factor_engine.cleaned_operators._numpy_kernels import cs_resid_

        result = pd.DataFrame(np.nan, index=y.index, columns=y.columns, dtype=float)
        for idx in y.index:
            result.loc[idx] = cs_resid_(y.loc[idx].values, x.loc[idx].values)
        return result


@register_operator(name="cs_regression", category="cross_sectional", business_category="cross_sectional", canonical="cs_regression", source="factor_dsl_np")
class CSRegression(SeriesOperator):
    """截面回归: mode=0 残差, 1 beta, 2 拟合值"""
    metadata = OperatorMetadata(
        name="cs_regression",
        category="cross_sectional",
        description="截面回归: mode=0 残差, 1 beta, 2 拟合值",
        examples=["cs_regression(factor, size, 0)"],
        param_names=["y", "x", "mode"],
        return_type="series",
        tags=["cross_sectional", "regression"],
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, mode: int = 0, **kwargs) -> pd.DataFrame:
        from factor_engine.cleaned_operators._numpy_kernels import cs_regression_

        result = pd.DataFrame(np.nan, index=y.index, columns=y.columns, dtype=float)
        for idx in y.index:
            result.loc[idx] = cs_regression_(y.loc[idx].values, x.loc[idx].values, mode=int(mode))
        return result


try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore
from factor_engine.cleaned_operators.base_polars import (
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
        count_expr = sum(
            (pl.col(c).is_not_null() & ~pl.col(c).is_nan()).cast(pl.Int32)
            for c in numeric_cols
        )

        return x.with_columns([
            count_expr.cast(pl.Float64).alias(c) for c in numeric_cols
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
        values = [pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)) for c in numeric_cols]
        mean_expr = pl.mean_horizontal(*values)
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
        pdf = x.select(numeric_cols).to_pandas()
        q = pdf.quantile(float(p), axis=1)
        broadcast = broadcast_row_stat(pdf, q)
        return x.with_columns([
            pl.Series(name=c, values=broadcast[c].to_numpy()) for c in numeric_cols
        ])



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
        arr = _finite_stats_input(x.select(numeric_cols).to_numpy())
        stds = np.nanstd(arr, axis=1, ddof=1, keepdims=True)

        # 转换回 Polars
        result_df = pl.DataFrame(np.repeat(stds, len(numeric_cols), axis=1), schema=numeric_cols)

        if 'date' in x.columns:
            result_df = result_df.with_columns([x['date']])

        return result_df



# canonical=cs_mad backend=polars selected=cs_mad source=cross_sectional/c_ops_polars.py
@register_operator(name="cs_mad", category="cross_sectional", business_category="cross_sectional", canonical="cs_mad", source="factor_dsl_np", backend="polars")
class CrossSectionalMadPolars(SeriesOperator):
    """Polars 截面中位绝对偏差（MAD，广播到各列）"""
    metadata = OperatorMetadata(
        name="cs_mad",
        category="cross_sectional",
        description="截面中位绝对偏差（MAD，广播到各列）",
        examples=["cs_mad(PE)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "mad", "pit_safe"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ["date", "stock_code"]]
        arr = _finite_stats_input(x.select(numeric_cols).to_numpy())
        med = np.nanmedian(arr, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(arr - med), axis=1, keepdims=True)
        out = np.repeat(mad, len(numeric_cols), axis=1)
        result = pl.DataFrame(out, schema=numeric_cols)
        if "date" in x.columns:
            result = result.with_columns([x["date"]])
        return result


# canonical=cs_mad_zscore backend=polars selected=cs_mad_zscore source=cross_sectional/c_ops_polars.py
@register_operator(name="cs_mad_zscore", category="cross_sectional", business_category="cross_sectional", canonical="cs_mad_zscore", source="factor_dsl_np", backend="polars")
class CrossSectionalMadZscorePolars(SeriesOperator):
    """Polars raw-MAD 标准化得分：(x - median) / MAD（不乘 0.6745）"""
    metadata = OperatorMetadata(
        name="cs_mad_zscore",
        category="cross_sectional",
        description="raw-MAD standardized score：(x - median) / MAD；不做 0.6745 缩放，"
                    "MAD 不乘 1.4826。要 scaled robust zscore 请显式构造：0.6745*cs_mad_zscore(x)。",
        examples=["cs_mad_zscore(PE)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "mad", "zscore", "pit_safe"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ["date", "stock_code"]]
        arr = _finite_stats_input(x.select(numeric_cols).to_numpy())
        med = np.nanmedian(arr, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(arr - med), axis=1, keepdims=True)
        mad = np.where(mad == 0, np.nan, mad)
        z = (arr - med) / mad
        result = pl.DataFrame(z, schema=numeric_cols)
        if "date" in x.columns:
            result = result.with_columns([x["date"]])
        return result



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

        values = [pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)) for c in numeric_cols]
        valid_count = pl.sum_horizontal([v.is_not_null().cast(pl.Int64) for v in values])
        raw_sum = pl.sum_horizontal(values, ignore_nulls=True)
        sum_expr = pl.when(valid_count > 0).then(raw_sum).otherwise(None)

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
        mean_expr = pl.mean_horizontal(*[pl.col(c) for c in numeric_cols])
        return x.with_columns([
            (pl.col(c) - mean_expr).alias(c) for c in numeric_cols
        ])

# aliases: CS_DEMEAN



# canonical=group_neutralize backend=polars selected=c_neutralize source=cross_sectional/c_ops_polars.py
@register_operator(name="group_neutralize", category="cross_sectional", business_category="cross_sectional", canonical="group_neutralize", source="factor_dsl_np")
class CrossSectionalNeutralizePolars(SeriesOperator):
    """行业中性化"""

    metadata = OperatorMetadata(
        name="group_neutralize",
        category="cross_sectional",
        description="对因子进行行业中性化处理",
        examples=["c_neutralize(ROE, industry)"],
        param_names=["x", "group", "fallback_policy"],
        return_type="series",
        tags=["cross_sectional", "neutralize", "industry"]
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame = None, fallback_policy: str = "nan", **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]

        # R19-063: ``group is None`` 不再退化为全局 demean（那是 cs_demean 的职责）。
        # 缺 group → fail-closed：输出全 NaN。
        if group is None:
            result_df = pl.DataFrame(
                np.full((x.height, len(numeric_cols)), np.nan), schema=numeric_cols
            )
            if 'date' in x.columns:
                result_df = result_df.with_columns([x['date']])
            return result_df

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
                # R19-064: 该日 group 全 missing 不能 silent global demean -> 保持 NaN。
                result_arr[row_idx] = np.nan
                continue
            # R19-065: 只有 group known 的股票参与组中性化；unknown group 的 cell
            # 保持 NaN（result_arr 初始化为 NaN）。
            for g in unique_groups:
                mask = row_group == g
                if np.any(mask):
                    # R40 #188: ±Inf is not a valid sample — mask non-finite to
                    # NaN so np.nanmean ignores it (a lone Inf in the group would
                    # otherwise make the whole group mean Inf).
                    group_mean = np.nanmean(_finite_stats_input(row_data[mask]))
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
        numeric_cols = [c for c in x.columns if c not in {"date", "stock_code"}]
        values = pl.concat_list([
            pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c))
            for c in numeric_cols
        ])
        ranks = values.list.eval(pl.element().rank(method="average"))
        count = values.list.drop_nulls().list.len()
        return x.with_columns([
            pl.when(values.list.get(i).is_null()).then(None)
            .when(count <= 1).then(0.5)
            .otherwise((ranks.list.get(i) - 1.0) / (count - 1.0))
            .alias(c)
            for i, c in enumerate(numeric_cols)
        ])

@register_operator(name="rank", category="cross_sectional", business_category="cross_sectional", canonical="rank", source="factor_dsl_np")
class RankPolars(SeriesOperator):
    """截面 0-1 排名（Polars）。"""

    metadata = OperatorMetadata(
        name="rank",
        category="cross_sectional",
        description="截面 0-1 排名；pandas 百分位排名见 rank_pct",
        examples=["rank(close)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "rank", "pit_safe"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return CrossSectionalRank()._calculate_series(x, **kwargs)


@register_operator(
    name="rank_pct",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="rank_pct",
    source="factor_dsl_polars",
)
class RankPctPolars(SeriesOperator):
    """Polars 截面 pandas 百分位排名 rank/count"""
    metadata = OperatorMetadata(
        name="rank_pct",
        category="cross_sectional",
        description="截面 pandas 百分位排名 rank/count",
        examples=["rank_pct(close)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "rank", "pit_safe", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in {"date", "stock_code"}]
        values = pl.concat_list([
            pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c))
            for c in numeric_cols
        ])
        ranks = values.list.eval(pl.element().rank(method="average"))
        count = values.list.drop_nulls().list.len()
        return x.with_columns([
            pl.when(values.list.get(i).is_null()).then(None)
            .otherwise(ranks.list.get(i) / count)
            .alias(c)
            for i, c in enumerate(numeric_cols)
        ])


@register_operator(
    name="cs_pct_rank",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_pct_rank",
    source="factor_dsl_polars",
)
class CsPctRankPolars(RankPctPolars):
    """Polars 截面百分位排名（同 rank_pct）"""
    metadata = OperatorMetadata(
        name="cs_pct_rank",
        category="cross_sectional",
        description="截面百分位排名（同 rank_pct）",
        examples=["cs_pct_rank(close)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "rank", "pit_safe", "polars"],
    )


@register_operator(
    name="cs_quantile",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_quantile",
    source="factor_dsl_polars",
)
class CsQuantilePolars(SeriesOperator):
    """Polars 截面 p 分位数值（广播到各列）"""
    metadata = OperatorMetadata(
        name="cs_quantile",
        category="cross_sectional",
        description="截面 p 分位数值（广播到各列）",
        examples=["cs_quantile(PE, 0.5)"],
        param_names=["x", "p"],
        return_type="series",
        tags=["cross_sectional", "quantile", "pit_safe", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, p: float = 0.5, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in {"date", "stock_code"}]
        pdf = x.select(numeric_cols).to_pandas()
        q = pdf.quantile(float(p), axis=1)
        broadcast = broadcast_row_stat(pdf, q)
        return x.with_columns([
            pl.Series(name=c, values=broadcast[c].to_numpy()) for c in numeric_cols
        ])

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
    """Polars 缩放数据使sum(abs(x))=指定值（与c_scale相同）"""

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
        param_names=["x", "lower", "upper"],
        return_type="series",
        tags=["cross_sectional", "winsorize", "outlier"]
    )

    def _calculate_series(self, x: pl.DataFrame, lower: float = 0.05, upper: float = 0.95, **kwargs) -> pl.DataFrame:
        min_pct, max_pct = lower, upper

        # 使用 Numba 计算分位数并裁剪
        arr = _finite_stats_input(x.select(numeric_cols).to_numpy())

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
        numeric_cols = [c for c in x.columns if c not in {"date", "stock_code"}]
        pdf = x.select(numeric_cols).to_pandas()
        mean = pdf.mean(axis=1)
        std = pdf.std(axis=1).replace(0, 1)
        z = pdf.sub(mean, axis=0).div(std, axis=0)
        return x.with_columns([
            pl.Series(name=c, values=z[c].to_numpy()) for c in numeric_cols
        ])

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


# ---------------------------------------------------------------------------
# 截面回归 Polars：按 **行**（每个 timestamp）做 OLS，复用 _numpy_kernels 保证数值一致
# mode: 0=残差, 1=beta, 2=拟合值（见 cs_regression_）
# ---------------------------------------------------------------------------


def _cs_regression_rowwise(y_arr: np.ndarray, x_arr: np.ndarray, mode: int) -> np.ndarray:
    """对每个交易日截面行调用 ``cs_regression_``。"""
    from factor_engine.cleaned_operators._numpy_kernels import cs_regression_

    out = np.full_like(y_arr, np.nan)
    for i in range(len(y_arr)):
        out[i] = cs_regression_(y_arr[i], x_arr[i], mode=int(mode))
    return out


@register_operator(
    name="cs_resid",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_resid",
    source="factor_dsl_polars",
    backend="polars",
)
class CSResidPolars(SeriesOperator):
    """Polars 截面线性回归残差 y - (α + βx)"""
    metadata = OperatorMetadata(
        name="cs_resid",
        category="cross_sectional",
        description="截面线性回归残差 y - (α + βx)",
        examples=["cs_resid(factor, size)"],
        param_names=["y", "x"],
        return_type="series",
        tags=["cross_sectional", "regression", "residual"],
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators._numpy_kernels import cs_resid_

        numeric_cols = [c for c in y.columns if c not in ("date", "stock_code")]
        y_arr = y.select(numeric_cols).to_numpy()
        x_arr = x.select(numeric_cols).to_numpy()
        out = np.full_like(y_arr, np.nan)
        for i in range(len(y_arr)):
            out[i] = cs_resid_(y_arr[i], x_arr[i])
        result_df = pl.DataFrame(out, schema=numeric_cols)
        if "date" in y.columns:
            result_df = result_df.with_columns([y["date"]])
        return result_df


@register_operator(
    name="cs_regression",
    category="cross_sectional",
    business_category="cross_sectional",
    canonical="cs_regression",
    source="factor_dsl_polars",
    backend="polars",
)
class CSRegressionPolars(SeriesOperator):
    """Polars 截面回归: mode=0 残差, 1 beta, 2 拟合值"""
    metadata = OperatorMetadata(
        name="cs_regression",
        category="cross_sectional",
        description="截面回归: mode=0 残差, 1 beta, 2 拟合值",
        examples=["cs_regression(factor, size, 0)"],
        param_names=["y", "x", "mode"],
        return_type="series",
        tags=["cross_sectional", "regression"],
    )

    def _calculate_series(
        self, y: pl.DataFrame, x: pl.DataFrame, mode: int = 0, **kwargs
    ) -> pl.DataFrame:
        numeric_cols = [c for c in y.columns if c not in ("date", "stock_code")]
        y_arr = y.select(numeric_cols).to_numpy()
        x_arr = x.select(numeric_cols).to_numpy()
        out = _cs_regression_rowwise(y_arr, x_arr, int(mode))
        result_df = pl.DataFrame(out, schema=numeric_cols)
        if "date" in y.columns:
            result_df = result_df.with_columns([y["date"]])
        return result_df
