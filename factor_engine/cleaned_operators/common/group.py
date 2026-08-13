# -*- coding: utf-8 -*-
"""
分组与中性化算子（**按行业/市值等分组去均值或回归残差**）。

语义
----
- ``neutralize(x, group)`` / ``industry_neutralize``：对分组标签做 demean 或回归中性化；
- ``panel_*``：面板维度的分组变换（与截面 ``c_*`` 不同，需显式 group 列）。

典型用途：剔除行业暴露后再做 rank；投递公式中 ``NEUTRALIZE``、``INDUSTRY_NEUTRAL`` 等别名指向本模块。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from cleaned_operators._causal import causal_lag
from cleaned_operators.base import (
    Operator,
    OperatorMetadata,
    ParamRole,
    ParamSpec,
    SeriesOperator,
    ScalarOperator,
    TwoVarOperator,
    register_operator,
)

import numpy as np
import pandas as pd

# R24-095/096: strict fallback policy enum.  An unknown string must RAISE —
# never fall through to a default branch that silently changes the economic
# definition (e.g. "nna" becoming global demean).
_GROUP_FALLBACK_POLICIES = frozenset({"nan", "error", "global", "keep_original"})
_FALLBACK_POLICY_SPEC = ParamSpec(
    dtype=str,
    choices=("nan", "error", "global", "keep_original"),
    default="nan",
    searchable=False,
)


def validate_fallback_policy(policy: str, *, operator: str = "") -> None:
    if policy not in _GROUP_FALLBACK_POLICIES:
        raise ValueError(
            f"{operator} fallback_policy must be one of "
            f"{sorted(_GROUP_FALLBACK_POLICIES)!r}, got {policy!r} "
            "(R24-096 — unknown policy never falls to a default branch)"
        )


def strict_group_align(x: pd.DataFrame, group: pd.DataFrame | None):
    """R24-097: strict axes gate for group panels — no silent ``reindex``.

    A misaligned group panel would re-pair an instrument to a different group
    and fabricate a spurious group statistic.  Raise instead of reindexing.
    """
    if group is None:
        return x, None
    from cleaned_operators.relation.ops import strict_relation_align

    x, group = strict_relation_align(x, group)
    return x, group


def _strict_group_panel(group: pd.DataFrame | None, x: pd.DataFrame) -> pd.DataFrame | None:
    """Strict-check a group panel against x; raise on any axis mismatch.

    Replaces the legacy ``_strict_group_panel(group, x)`` —
    a misaligned group is a semantic corruption, never silently repaired.
    """
    _x, _g = strict_group_align(x, group)
    return _g


# canonical=deltas backend=pandas_numpy selected=deltas source=time_series/panel_ops.py
@register_operator(name="deltas", category="time_series", business_category="group_neutralization", canonical="deltas", source="factor_dsl_np")
class Deltas(SeriesOperator):
    """n期差分 (与Delta相同) x - Ref(x, n)"""
    metadata = OperatorMetadata(
        name="deltas", category="time_series",
        description="n期差分 (与Delta相同) x - Ref(x, n)",
        examples=["deltas(close, 5)"],
        param_names=["x", "n"], return_type="series",
        tags=["time_series", "delta", "diff"]
    )
    def _calculate_series(self, x: pd.DataFrame, n: int = 1, **kwargs) -> pd.DataFrame:
        return x - causal_lag(x, n)



def _group_rank_weighted_value_panel(
    x: pd.DataFrame,
    group: pd.DataFrame | None,
    fallback_policy: str = "nan",
) -> pd.DataFrame:
    """组内按**平均排名**线性加权（CS rank-weighted value）。

    每个交易日按 ``group`` 分组，对组内 ``x`` 取平均排名（并列同权重，等价
    ``rank(method='average')``），权重 ∝ 平均排名并在组内归一化，输出
    ``x * weight``。时间维度不参与；真实时间衰减见 ``group_ts_decay_linear``。
    整列分组缺失时按 ``fallback_policy``（PIT 生产默认 nan）。``window``
    不存在于此签名 —— 兼容算子 ``group_decay_linear`` 的 ``window`` 不参与计算。
    """
    result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)

    for date in x.index:
        x_slice = x.loc[date]

        if group is not None and date in group.index:
            group_slice = group.loc[date]
        else:
            group_slice = None

        if group_slice is None or group_slice.isna().all():
            # 整列 group 缺失：按 fallback_policy 决定行为。PIT 生产默认 nan
            # （与 SQL/Polars 路径一致，未知分组不参与计算）。
            if fallback_policy == "nan":
                continue
            if fallback_policy == "keep_original":
                result.loc[date] = x_slice
                continue
            valid_mask = x_slice.notna()
            if valid_mask.sum() > 0:
                data = x_slice[valid_mask]
                ranks = data.rank(method="average")
                w = ranks / ranks.sum()
                result.loc[date, valid_mask] = data * w
            continue

        for group_val in group_slice.dropna().unique():
            mask = (group_slice == group_val) & x_slice.notna()
            if mask.sum() > 0:
                group_data = x_slice[mask]
                ranks = group_data.rank(method="average")
                w = ranks / ranks.sum()
                result.loc[date, group_data.index] = group_data * w

    return result


# canonical=group_rank_weighted_value backend=pandas_numpy selected=group_rank_weighted_value source=cross_sectional/group_ops.py
@register_operator(name="group_rank_weighted_value", category="cross_sectional", business_category="group_neutralization", canonical="group_rank_weighted_value", source="factor_dsl_np")
class GroupRankWeightedValue(SeriesOperator):
    """组内按**平均排名**线性加权（跨截面 rank weighting，非时间衰减）。

    每个交易日按 ``group`` 分组，对组内 ``x`` 取平均排名（并列同权重），权重
    ∝ 平均排名并在组内归一化，输出 ``x * weight``。时间维度不参与；真实时间
    衰减见 ``group_ts_decay_linear``。兼容旧名 ``group_decay_linear``（其
    ``window`` 参数不参与计算）。"""

    metadata = OperatorMetadata(
        name="group_rank_weighted_value",
        category="cross_sectional",
        description="组内按平均排名线性加权（CS rank-weighted value；非时间衰减）",
        examples=[
            "group_rank_weighted_value(ROE, industry_code)",
            "group_rank_weighted_value(returns, get('industry_sw'))"
        ],
        param_names=["x", "group", "fallback_policy"],
        param_specs={"fallback_policy": _FALLBACK_POLICY_SPEC},
        return_type="series",
        tags=["cross_sectional", "rank_weighted", "group", "linear"]
    )

    def _calculate_series(self, x: pd.DataFrame, group: pd.DataFrame = None, fallback_policy: str = "nan", **kwargs) -> pd.DataFrame:
        
        validate_fallback_policy(fallback_policy, operator=type(self).__name__)
        return _group_rank_weighted_value_panel(x, group, fallback_policy=fallback_policy)


# canonical=group_decay_linear backend=pandas_numpy selected=group_decay_linear source=cross_sectional/group_ops.py
@register_operator(name="group_decay_linear", category="cross_sectional", business_category="group_neutralization", canonical="group_decay_linear", source="factor_dsl_np")
class GroupDecayLinear(SeriesOperator):
    """兼容名：组内按排名线性加权（诚实名称 ``group_rank_weighted_value``）。

    ``window`` 参数仅为兼容保留、**不参与计算**：
    ``group_decay_linear(x,g,5)==group_decay_linear(x,g,10)==
    group_decay_linear(x,g,60)``。声明为 ``searchable=False / ParamRole.POLICY``
    （校验但不作搜索维度）。真实时间衰减见 ``group_ts_decay_linear``。"""

    metadata = OperatorMetadata(
        name="group_decay_linear",
        category="cross_sectional",
        description="组内按排名线性加权（兼容名；window 不参与计算，诚实名称 group_rank_weighted_value）",
        examples=[
            "group_decay_linear(ROE, industry_code)",
            "group_decay_linear(returns, get('industry_sw'), 5)"
        ],
        param_names=["x", "group", "window", "fallback_policy"],
        return_type="series",
        tags=["cross_sectional", "decay", "linear", "group", "rank_weighted", "compat_alias"],
        param_specs={"window": ParamSpec(dtype=int, searchable=False, param_role=ParamRole.POLICY)},
    )

    def _calculate_series(self, x: pd.DataFrame, group: pd.DataFrame = None, window: int = 5, fallback_policy: str = "nan", **kwargs) -> pd.DataFrame:
        
        validate_fallback_policy(fallback_policy, operator=type(self).__name__)
        return _group_rank_weighted_value_panel(x, group, fallback_policy=fallback_policy)



# canonical=group_ts_decay_linear backend=pandas_numpy selected=group_ts_decay_linear source=cross_sectional/group_ops.py
@register_operator(name="group_ts_decay_linear", category="cross_sectional", business_category="group_neutralization", canonical="group_ts_decay_linear", source="factor_dsl_np")
class GroupTSDecayLinear(SeriesOperator):
    """组内**时间位置**线性衰减加权（真实时间衰减，窗口参数参与计算）。

    每只股票取截至当前行的 trailing ``window`` bar，权重按时间线性衰减
    （最近 bar 权重最高，w_j = 1..window，缺失值剔除后归一化）。``group``
    用于 gate:股票当日分组未知 → NaN；``normalize=True`` 时在当日组内做
    横截面 z-score。整列分组缺失时按 ``fallback_policy``（PIT 默认 nan）。"""

    metadata = OperatorMetadata(
        name="group_ts_decay_linear",
        category="cross_sectional",
        description="组内按时间位置线性衰减加权（真实时间衰减；窗口参与计算）",
        examples=[
            "group_ts_decay_linear(close, industry_code, 20)",
            "group_ts_decay_linear(returns, get('industry_sw'), 10, normalize=True)"
        ],
        param_names=["x", "group", "window", "normalize", "fallback_policy"],
        return_type="series",
        tags=["cross_sectional", "decay", "linear", "group", "time_decay"]
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        group: pd.DataFrame = None,
        window: int = 20,
        normalize: bool = False,
        fallback_policy: str = "nan",
        **kwargs,
    ) -> pd.DataFrame:
        from cleaned_operators._rolling_fast import rolling_linear_weighted

        w = max(1, int(window))
        decay = rolling_linear_weighted(x, w).reindex(index=x.index, columns=x.columns)
        if group is None:
            # 无分组输入：纯时间衰减，不做任何横截面 gate。
            return decay

        g = _strict_group_panel(group, x)
        result = decay.copy()
        for date in x.index:
            group_slice = g.loc[date]
            if group_slice.isna().all():
                if fallback_policy == "global":
                    continue  # 保留纯时间衰减
                if fallback_policy == "keep_original":
                    result.loc[date] = x.loc[date]
                else:  # "nan"（默认）
                    result.loc[date] = np.nan
                continue
            # 单股分组未知 → NaN（unknown_group_policy=nan）
            result.loc[date, group_slice.isna()] = np.nan
            if normalize:
                for group_val in group_slice.dropna().unique():
                    mask = (group_slice == group_val)
                    values = result.loc[date, mask]
                    mu = float(values.mean())
                    sd = float(values.std(ddof=0))
                    if sd > 0:
                        result.loc[date, mask] = (values - mu) / sd
                    else:
                        result.loc[date, mask] = np.nan
        return result


# canonical=group_demean backend=pandas_numpy selected=group_demean source=cross_sectional/group_ops.py
@register_operator(name="group_demean", category="cross_sectional", business_category="group_neutralization", canonical="group_neutralize", source="factor_dsl_np")
class GroupDemean(SeriesOperator):
    """在指定分组内进行去均值处理 (x - group_mean)"""

    metadata = OperatorMetadata(
        name="group_demean",
        category="cross_sectional",
        description="在指定分组内进行去均值处理 (x - group_mean)",
        examples=[
            "group_demean(ROE, industry_code)",
            "group_demean(returns, get('industry_sw'))"
        ],
        param_names=["x", "group", "fallback_policy"],
        param_specs={"fallback_policy": _FALLBACK_POLICY_SPEC},
        return_type="series",
        tags=["cross_sectional", "demean", "group", "industry"]
    )

    def _calculate_series(self, x: pd.DataFrame, group: pd.DataFrame = None, fallback_policy: str = "nan", **kwargs) -> pd.DataFrame:
        
        validate_fallback_policy(fallback_policy, operator=type(self).__name__)
        from cleaned_operators._numpy_kernels import group_demean_panel_

        # R19-063..065: a missing ``group`` must NOT silently degrade to a
        # global cross-sectional demean — that is a different factor, not a
        # fallback.  Route the no-group call through the kernel's declared
        # ``fallback`` policy: the default ``"nan"`` leaves the output NaN
        # (fail-closed); only the explicit ``fallback_policy="global"`` opts
        # into row-wise demean, and ``"keep_original"`` preserves the input.
        g = _strict_group_panel(group, x).to_numpy() if group is not None else None
        out = group_demean_panel_(
            x.to_numpy(dtype=float, copy=False), g, fallback=fallback_policy
        )
        return pd.DataFrame(out, index=x.index, columns=x.columns)



# canonical=group_mean backend=pandas_numpy selected=group_mean source=cross_sectional/group_ops.py
@register_operator(name="group_mean", category="cross_sectional", business_category="group_neutralization", canonical="group_mean", source="factor_dsl_np")
class GroupMean(SeriesOperator):
    """组内均值"""

    metadata = OperatorMetadata(
        name="group_mean",
        category="cross_sectional",
        description="计算组内均值",
        examples=["group_mean(ROE, industry_code)"],
        param_names=["x", "group", "fallback_policy"],
        param_specs={"fallback_policy": _FALLBACK_POLICY_SPEC},
        return_type="series",
        tags=["cross_sectional", "mean", "group", "aggregate"]
    )

    def _calculate_series(self, x: pd.DataFrame, group: pd.DataFrame = None, fallback_policy: str = "nan", **kwargs) -> pd.DataFrame:
        
        validate_fallback_policy(fallback_policy, operator=type(self).__name__)
        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)

        for date in x.index:
            x_slice = x.loc[date]
            
            if group is not None and date in group.index:
                group_slice = group.loc[date]
            else:
                group_slice = None

            if group_slice is None or group_slice.isna().all():
                if fallback_policy == "nan":
                    continue
                if fallback_policy == "keep_original":
                    result.loc[date] = x_slice
                    continue

                mean_val = x_slice.mean()
                result.loc[date] = mean_val
                continue

            for group_val in group_slice.dropna().unique():
                mask = (group_slice == group_val) & x_slice.notna()
                if mask.sum() > 0:
                    mean_val = x_slice[mask].mean()
                    result.loc[date, x_slice[mask].index] = mean_val

        return result



# canonical=group_sum backend=pandas_numpy selected=group_sum source=cross_sectional/group_ops.py
@register_operator(name="group_sum", category="cross_sectional", business_category="group_neutralization", canonical="group_sum", source="factor_dsl_np")
class GroupSum(SeriesOperator):
    """组内求和"""

    metadata = OperatorMetadata(
        name="group_sum",
        category="cross_sectional",
        description="计算组内求和",
        examples=["group_sum(volume, industry_code)"],
        param_names=["x", "group", "fallback_policy"],
        param_specs={"fallback_policy": _FALLBACK_POLICY_SPEC},
        return_type="series",
        tags=["cross_sectional", "sum", "group", "aggregate"],
    )

    def _calculate_series(self, x: pd.DataFrame, group: pd.DataFrame = None, fallback_policy: str = "nan", **kwargs) -> pd.DataFrame:
        
        validate_fallback_policy(fallback_policy, operator=type(self).__name__)
        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
        for date in x.index:
            x_slice = x.loc[date]
            group_slice = group.loc[date] if group is not None and date in group.index else None
            if group_slice is None or group_slice.isna().all():
                if fallback_policy == "nan":
                    continue
                if fallback_policy == "keep_original":
                    result.loc[date] = x_slice
                    continue

                result.loc[date] = x_slice.sum()
                continue
            for group_val in group_slice.dropna().unique():
                mask = (group_slice == group_val) & x_slice.notna()
                if mask.sum() > 0:
                    result.loc[date, x_slice[mask].index] = x_slice[mask].sum()
        return result



# canonical=group_min backend=pandas_numpy selected=group_min source=cross_sectional/group_ops.py
@register_operator(name="group_min", category="cross_sectional", business_category="group_neutralization", canonical="group_min", source="factor_dsl_np")
class GroupMin(SeriesOperator):
    """组内最小值"""

    metadata = OperatorMetadata(
        name="group_min",
        category="cross_sectional",
        description="计算组内最小值",
        examples=["group_min(close, industry_code)"],
        param_names=["x", "group", "fallback_policy"],
        param_specs={"fallback_policy": _FALLBACK_POLICY_SPEC},
        return_type="series",
        tags=["cross_sectional", "min", "group", "aggregate"],
    )

    def _calculate_series(self, x: pd.DataFrame, group: pd.DataFrame = None, fallback_policy: str = "nan", **kwargs) -> pd.DataFrame:
        
        validate_fallback_policy(fallback_policy, operator=type(self).__name__)
        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
        for date in x.index:
            x_slice = x.loc[date]
            group_slice = group.loc[date] if group is not None and date in group.index else None
            if group_slice is None or group_slice.isna().all():
                if fallback_policy == "nan":
                    continue
                if fallback_policy == "keep_original":
                    result.loc[date] = x_slice
                    continue

                result.loc[date] = x_slice.min()
                continue
            for group_val in group_slice.dropna().unique():
                mask = (group_slice == group_val) & x_slice.notna()
                if mask.sum() > 0:
                    result.loc[date, x_slice[mask].index] = x_slice[mask].min()
        return result



# canonical=group_max backend=pandas_numpy selected=group_max source=cross_sectional/group_ops.py
@register_operator(name="group_max", category="cross_sectional", business_category="group_neutralization", canonical="group_max", source="factor_dsl_np")
class GroupMax(SeriesOperator):
    """组内最大值"""

    metadata = OperatorMetadata(
        name="group_max",
        category="cross_sectional",
        description="计算组内最大值",
        examples=["group_max(close, industry_code)"],
        param_names=["x", "group", "fallback_policy"],
        param_specs={"fallback_policy": _FALLBACK_POLICY_SPEC},
        return_type="series",
        tags=["cross_sectional", "max", "group", "aggregate"],
    )

    def _calculate_series(self, x: pd.DataFrame, group: pd.DataFrame = None, fallback_policy: str = "nan", **kwargs) -> pd.DataFrame:
        
        validate_fallback_policy(fallback_policy, operator=type(self).__name__)
        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
        for date in x.index:
            x_slice = x.loc[date]
            group_slice = group.loc[date] if group is not None and date in group.index else None
            if group_slice is None or group_slice.isna().all():
                if fallback_policy == "nan":
                    continue
                if fallback_policy == "keep_original":
                    result.loc[date] = x_slice
                    continue

                result.loc[date] = x_slice.max()
                continue
            for group_val in group_slice.dropna().unique():
                mask = (group_slice == group_val) & x_slice.notna()
                if mask.sum() > 0:
                    result.loc[date, x_slice[mask].index] = x_slice[mask].max()
        return result



# canonical=group_count backend=pandas_numpy selected=group_count source=cross_sectional/group_ops.py
@register_operator(name="group_count", category="cross_sectional", business_category="group_neutralization", canonical="group_count", source="factor_dsl_np")
class GroupCount(SeriesOperator):
    """组内非空计数"""

    metadata = OperatorMetadata(
        name="group_count",
        category="cross_sectional",
        description="计算组内非空样本数",
        examples=["group_count(close, industry_code)"],
        param_names=["x", "group", "fallback_policy"],
        param_specs={"fallback_policy": _FALLBACK_POLICY_SPEC},
        return_type="series",
        tags=["cross_sectional", "count", "group", "aggregate"],
    )

    def _calculate_series(self, x: pd.DataFrame, group: pd.DataFrame = None, fallback_policy: str = "nan", **kwargs) -> pd.DataFrame:
        
        validate_fallback_policy(fallback_policy, operator=type(self).__name__)
        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
        for date in x.index:
            x_slice = x.loc[date]
            group_slice = group.loc[date] if group is not None and date in group.index else None
            if group_slice is None or group_slice.isna().all():
                if fallback_policy == "nan":
                    continue
                if fallback_policy == "keep_original":
                    result.loc[date] = x_slice
                    continue

                result.loc[date] = np.isfinite(x_slice.to_numpy(dtype=float)).sum()
                continue
            for group_val in group_slice.dropna().unique():
                mask = (group_slice == group_val).to_numpy() & np.isfinite(x_slice.to_numpy(dtype=float))
                result.loc[date, x_slice[mask].index] = float(mask.sum())
        return result



# canonical=group_normalize backend=pandas_numpy selected=group_normalize source=cross_sectional/group_ops.py
@register_operator(name="group_normalize", category="cross_sectional", business_category="group_neutralization", canonical="group_normalize", source="factor_dsl_np")
class GroupNormalize(SeriesOperator):
    """在指定分组内对股票进行归一化到[0,1]"""

    metadata = OperatorMetadata(
        name="group_normalize",
        category="cross_sectional",
        description="在指定分组内对股票进行归一化到[0,1]",
        examples=[
            "group_normalize(ROE, industry_code)",
            "group_normalize(PE, get('industry_sw'))"
        ],
        param_names=["x", "group", "fallback_policy"],
        param_specs={"fallback_policy": _FALLBACK_POLICY_SPEC},
        return_type="series",
        tags=["cross_sectional", "normalize", "group", "industry"]
    )

    def _calculate_series(self, x: pd.DataFrame, group: pd.DataFrame = None, fallback_policy: str = "nan", **kwargs) -> pd.DataFrame:
        
        validate_fallback_policy(fallback_policy, operator=type(self).__name__)
        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)

        for date in x.index:
            x_slice = x.loc[date]

            if group is not None and date in group.index:
                group_slice = group.loc[date]
            else:
                group_slice = None

            if group_slice is None or group_slice.isna().all():
                if fallback_policy == "nan":
                    continue
                if fallback_policy == "keep_original":
                    result.loc[date] = x_slice
                    continue

                valid_mask = x_slice.notna()
                if valid_mask.sum() > 0:
                    data = x_slice[valid_mask]
                    min_val = data.min()
                    max_val = data.max()
                    rng = max_val - min_val
                    if rng != 0 and not pd.isna(rng):
                        result.loc[date, valid_mask] = (data - min_val) / rng
                    else:
                        result.loc[date, valid_mask] = 0.5
                continue

            for group_val in group_slice.dropna().unique():
                mask = (group_slice == group_val) & x_slice.notna()
                if mask.sum() > 0:
                    group_data = x_slice[mask]
                    min_val = group_data.min()
                    max_val = group_data.max()
                    rng = max_val - min_val
                    if rng != 0 and not pd.isna(rng):
                        result.loc[date, group_data.index] = (group_data - min_val) / rng
                    else:
                        result.loc[date, group_data.index] = 0.5

        return result



# canonical=group_percentile backend=pandas_numpy selected=group_percentile source=cross_sectional/group_ops.py
@register_operator(name="group_percentile", category="cross_sectional", business_category="group_neutralization", canonical="group_percentile", source="factor_dsl_np")
class GroupPercentile(SeriesOperator):
    """组内分位数判断"""

    metadata = OperatorMetadata(
        name="group_percentile",
        category="cross_sectional",
        description="判断股票在组内的分位位置，返回0或1",
        examples=[
            "group_percentile(ROE, industry_code, 0.5)",
            "group_percentile(PE, get('industry_sw'), 0.3)"
        ],
        param_names=["x", "group", "p"],
        return_type="series",
        tags=["cross_sectional", "percentile", "group", "filter"]
    )

    def _calculate_series(self, x: pd.DataFrame, group: pd.DataFrame = None, p: float = 0.5, **kwargs) -> pd.DataFrame:
        """
        参数:
            x: 待判断的因子值DataFrame
            group: 分组标签DataFrame
            p: 分位数阈值(0-1)，默认0.5表示前50%
        返回:
            布尔值DataFrame，表示是否在指定分位内
        """
        result = pd.DataFrame(np.nan, index=x.index, columns=x.columns)

        for date in x.index:
            x_slice = x.loc[date]
            
            if group is not None and date in group.index:
                group_slice = group.loc[date]
            else:
                group_slice = None

            if group_slice is None or group_slice.isna().all():
                if fallback_policy == "nan":
                    continue
                if fallback_policy == "keep_original":
                    result.loc[date] = x_slice
                    continue

                # 全截面判断
                valid_mask = x_slice.notna()
                if valid_mask.sum() > 0:
                    is_in = x_slice[valid_mask].rank(pct=True) <= p
                    result.loc[date, valid_mask] = is_in.astype(float)
                continue

            # 按组判断
            for group_val in group_slice.dropna().unique():
                mask = (group_slice == group_val) & x_slice.notna()
                if mask.sum() > 0:
                    group_data = x_slice[mask]
                    if len(group_data) > 0:
                        is_in = (group_data.rank(pct=True) <= p).astype(float)
                        result.loc[date, is_in.index] = is_in

        return result



# canonical=group_rank backend=pandas_numpy selected=group_rank source=cross_sectional/group_ops.py
@register_operator(name="group_rank", category="cross_sectional", business_category="group_neutralization", canonical="group_rank", source="factor_dsl_np")
class GroupRank(SeriesOperator):
    """组内排名 - 按分组进行截面排名"""

    metadata = OperatorMetadata(
        name="group_rank",
        category="cross_sectional",
        description="在指定分组内对股票进行排名，返回归一化排名(0-1)",
        examples=[
            "group_rank(ROE, industry_code)",
            "group_rank(PE, get('industry_sw'))"
        ],
        param_names=["x", "group", "fallback_policy"],
        param_specs={"fallback_policy": _FALLBACK_POLICY_SPEC},
        return_type="series",
        tags=["cross_sectional", "rank", "group", "industry"]
    )

    def _calculate_series(self, x: pd.DataFrame, group: pd.DataFrame = None, fallback_policy: str = "nan", **kwargs) -> pd.DataFrame:
        
        validate_fallback_policy(fallback_policy, operator=type(self).__name__)
        """
        参数:
            x: 待排名的因子值DataFrame (dates x stocks)
            group: 分组标签DataFrame (dates x stocks)，如行业代码。若为None则全截面排名
        返回:
            组内归一化排名DataFrame (0-1)，值越小表示排名越靠前
        """
        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)

        for date in x.index:
            x_slice = x.loc[date]

            # 获取分组信息
            if group is not None and date in group.index:
                group_slice = group.loc[date]
            else:
                group_slice = None

            if group_slice is None or group_slice.isna().all():
                if fallback_policy == "nan":
                    continue
                if fallback_policy == "keep_original":
                    result.loc[date] = x_slice
                    continue

                # 如果没有分组信息，进行全截面排名
                valid_mask = x_slice.notna()
                if valid_mask.sum() > 0:
                    result.loc[date, valid_mask] = x_slice[valid_mask].rank(pct=True)
                continue

            # 按组进行排名
            for group_val in group_slice.dropna().unique():
                mask = (group_slice == group_val) & x_slice.notna()
                if mask.sum() > 0:
                    group_data = x_slice[mask]
                    if len(group_data) > 0:
                        ranks = group_data.rank(pct=True)
                        result.loc[date, ranks.index] = ranks

        return result



# canonical=group_std backend=pandas_numpy selected=group_std source=cross_sectional/group_ops.py
@register_operator(name="group_std", category="cross_sectional", business_category="group_neutralization", canonical="group_std", source="factor_dsl_np")
class GroupStd(SeriesOperator):
    """组内标准差"""

    metadata = OperatorMetadata(
        name="group_std",
        category="cross_sectional",
        description="计算组内标准差",
        examples=["group_std(ROE, industry_code)"],
        param_names=["x", "group", "fallback_policy"],
        param_specs={"fallback_policy": _FALLBACK_POLICY_SPEC},
        return_type="series",
        tags=["cross_sectional", "std", "group", "aggregate"]
    )

    def _calculate_series(self, x: pd.DataFrame, group: pd.DataFrame = None, fallback_policy: str = "nan", **kwargs) -> pd.DataFrame:
        
        validate_fallback_policy(fallback_policy, operator=type(self).__name__)
        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)

        for date in x.index:
            x_slice = x.loc[date]
            
            if group is not None and date in group.index:
                group_slice = group.loc[date]
            else:
                group_slice = None

            if group_slice is None or group_slice.isna().all():
                if fallback_policy == "nan":
                    continue
                if fallback_policy == "keep_original":
                    result.loc[date] = x_slice
                    continue

                std_val = x_slice.std()
                result.loc[date] = std_val if not pd.isna(std_val) else 0
                continue

            for group_val in group_slice.dropna().unique():
                mask = (group_slice == group_val) & x_slice.notna()
                if mask.sum() > 0:
                    std_val = x_slice[mask].std()
                    if pd.isna(std_val):
                        std_val = 0
                    result.loc[date, x_slice[mask].index] = std_val

        return result



# canonical=group_winsorize backend=pandas_numpy selected=group_winsorize source=cross_sectional/group_ops.py
@register_operator(name="group_winsorize", category="cross_sectional", business_category="group_neutralization", canonical="group_winsorize", source="factor_dsl_np")
class GroupWinsorize(SeriesOperator):
    """在指定分组内进行缩尾处理，将超出分位数a的值截断"""

    metadata = OperatorMetadata(
        name="group_winsorize",
        category="cross_sectional",
        description="在指定分组内进行缩尾处理，将超出分位数a的值截断",
        examples=[
            "group_winsorize(ROE, industry_code, 0.05)",
            "group_winsorize(PE, get('industry_sw'), 0.01)"
        ],
        param_names=["x", "group", "a"],
        return_type="series",
        tags=["cross_sectional", "winsorize", "group", "outlier"]
    )

    def _calculate_series(self, x: pd.DataFrame, group: pd.DataFrame = None, a: float = 0.05, **kwargs) -> pd.DataFrame:
        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)

        for date in x.index:
            x_slice = x.loc[date]

            if group is not None and date in group.index:
                group_slice = group.loc[date]
            else:
                group_slice = None

            if group_slice is None or group_slice.isna().all():
                if fallback_policy == "nan":
                    continue
                if fallback_policy == "keep_original":
                    result.loc[date] = x_slice
                    continue

                valid_mask = x_slice.notna()
                if valid_mask.sum() > 0:
                    data = x_slice[valid_mask]
                    lower = data.quantile(a)
                    upper = data.quantile(1 - a)
                    result.loc[date, valid_mask] = data.clip(lower=lower, upper=upper)
                continue

            for group_val in group_slice.dropna().unique():
                mask = (group_slice == group_val) & x_slice.notna()
                if mask.sum() > 0:
                    group_data = x_slice[mask]
                    lower = group_data.quantile(a)
                    upper = group_data.quantile(1 - a)
                    result.loc[date, group_data.index] = group_data.clip(lower=lower, upper=upper)

        return result



# canonical=group_zscore backend=pandas_numpy selected=group_zscore source=cross_sectional/group_ops.py
@register_operator(name="group_zscore", category="cross_sectional", business_category="group_neutralization", canonical="group_zscore", source="factor_dsl_np")
class GroupZScore(SeriesOperator):
    """组内Z-Score标准化"""

    metadata = OperatorMetadata(
        name="group_zscore",
        category="cross_sectional",
        description="在指定分组内进行Z-Score标准化 (x-mean)/std",
        examples=[
            "group_zscore(ROE, industry_code)",
            "group_zscore(returns, get('industry_sw'))"
        ],
        param_names=["x", "group", "fallback_policy"],
        param_specs={"fallback_policy": _FALLBACK_POLICY_SPEC},
        return_type="series",
        tags=["cross_sectional", "zscore", "group", "industry", "standardize"]
    )

    def _calculate_series(self, x: pd.DataFrame, group: pd.DataFrame = None, fallback_policy: str = "nan", **kwargs) -> pd.DataFrame:
        
        validate_fallback_policy(fallback_policy, operator=type(self).__name__)
        """
        参数:
            x: 待标准化的因子值DataFrame (dates x stocks)
            group: 分组标签DataFrame (dates x stocks)。若为None则全截面标准化
        返回:
            组内Z-Score标准化后的DataFrame
        """
        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)

        for date in x.index:
            x_slice = x.loc[date]

            if group is not None and date in group.index:
                group_slice = group.loc[date]
            else:
                group_slice = None

            if group_slice is None or group_slice.isna().all():
                if fallback_policy == "nan":
                    continue
                if fallback_policy == "keep_original":
                    result.loc[date] = x_slice
                    continue

                # 全截面标准化
                valid_mask = x_slice.notna()
                if valid_mask.sum() > 0:
                    data = x_slice[valid_mask]
                    mean = data.mean()
                    std = data.std()
                    if std != 0 and not pd.isna(std):
                        result.loc[date, valid_mask] = (data - mean) / std
                    else:
                        result.loc[date, valid_mask] = 0
                continue

            # 按组标准化
            for group_val in group_slice.dropna().unique():
                mask = (group_slice == group_val) & x_slice.notna()
                if mask.sum() > 0:
                    group_data = x_slice[mask]
                    mean = group_data.mean()
                    std = group_data.std()
                    if std != 0 and not pd.isna(std):
                        result.loc[date, group_data.index] = (group_data - mean) / std
                    else:
                        result.loc[date, group_data.index] = 0

        return result



# canonical=move backend=pandas_numpy selected=move source=time_series/panel_ops.py
@register_operator(name="move", category="time_series", business_category="group_neutralization", canonical="move", source="factor_dsl_np")
class Move(SeriesOperator):
    """滑动窗口均值 (与m_avg相同)"""
    metadata = OperatorMetadata(
        name="move", category="time_series",
        description="滑动窗口均值 (与m_avg相同)",
        examples=["move(close, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "moving", "average"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).mean()



# canonical=panel_neutralize backend=pandas_numpy selected=panel_neutralize source=time_series/panel_ops.py
@register_operator(name="panel_neutralize", category="time_series", business_category="group_neutralization", canonical="panel_neutralize", source="factor_dsl_np")
class PanelNeutralize(SeriesOperator):
    """多维中性化 (按组去均值)"""
    metadata = OperatorMetadata(
        name="panel_neutralize", category="time_series",
        description="多维中性化 (按组去均值)",
        examples=["panel_neutralize(close, industry)"],
        param_names=["x", "groups"], return_type="series",
        tags=["time_series", "panel", "neutralize"]
    )
    def _calculate_series(self, x: pd.DataFrame, groups: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        if groups is None:
            return x.sub(x.mean(axis=1), axis=0)
        result = x.copy()
        for idx in x.index:
            row_x = x.loc[idx]
            row_g = groups.loc[idx] if idx in groups.index else None
            if row_g is not None:
                for g_val in row_g.unique():
                    mask = row_g == g_val
                    if mask.sum() > 0:
                        group_mean = row_x[mask].mean()
                        result.loc[idx, mask] = row_x[mask] - group_mean
            else:
                result.loc[idx] = row_x - row_x.mean()
        return result



# canonical=panel_rank backend=pandas_numpy selected=panel_rank source=time_series/panel_ops.py
@register_operator(name="panel_rank", category="time_series", business_category="group_neutralization", canonical="panel_rank", source="factor_dsl_np")
class PanelRank(SeriesOperator):
    """面板排名 (按行百分位排名)"""
    metadata = OperatorMetadata(
        name="panel_rank", category="time_series",
        description="面板排名 (按行百分位排名)",
        examples=["panel_rank(close)"],
        param_names=["x"], return_type="series",
        tags=["time_series", "panel", "rank"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.rank(axis=1, pct=True)



# canonical=panel_standardize backend=pandas_numpy selected=panel_standardize source=time_series/panel_ops.py
@register_operator(name="panel_standardize", category="time_series", business_category="group_neutralization", canonical="panel_standardize", source="factor_dsl_np")
class PanelStandardize(SeriesOperator):
    """面板Z-Score标准化 (按行标准化)"""
    metadata = OperatorMetadata(
        name="panel_standardize", category="time_series",
        description="面板Z-Score标准化 (按行标准化)",
        examples=["panel_standardize(close)"],
        param_names=["x"], return_type="series",
        tags=["time_series", "panel", "standardize"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        mean = x.mean(axis=1)
        std = x.std(axis=1).replace(0, 1)
        return x.sub(mean, axis=0).div(std, axis=0)



# canonical=panel_zscore backend=pandas_numpy selected=panel_zscore source=time_series/panel_ops.py
@register_operator(name="panel_zscore", category="time_series", business_category="group_neutralization", canonical="panel_zscore", source="factor_dsl_np")
class PanelZscore(SeriesOperator):
    """面板Z-Score (按行Z-Score标准化)"""
    metadata = OperatorMetadata(
        name="panel_zscore", category="time_series",
        description="面板Z-Score (按行Z-Score标准化)",
        examples=["panel_zscore(close)"],
        param_names=["x"], return_type="series",
        tags=["time_series", "panel", "zscore"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        mean = x.mean(axis=1)
        std = x.std(axis=1).replace(0, 1)
        return x.sub(mean, axis=0).div(std, axis=0)



# canonical=ratios backend=pandas_numpy selected=ratios source=time_series/panel_ops.py
@register_operator(name="ratios", category="time_series", business_category="group_neutralization", canonical="ts_ratio", source="factor_dsl_np")
class Ratios(SeriesOperator):
    """当前值与前一期之比 x / Ref(x, 1)"""
    metadata = OperatorMetadata(
        name="ratios", category="time_series",
        description="当前值与前一期之比 x / Ref(x, 1)",
        examples=["ratios(close)"],
        param_names=["x"], return_type="series",
        tags=["time_series", "ratio"]
    )
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        prev = x.shift(1)
        return x / prev.replace(0, np.nan)


# 重复实现：见 group_neutralize；dedupe 注销
# @register_operator(name="industry_neutralize", ...)
class IndustryNeutralize(SeriesOperator):
    """行业组内去均值（需传入 industry 分组列）"""
    metadata = OperatorMetadata(
        name="industry_neutralize",
        category="group_neutralization",
        description="行业组内去均值（需传入 industry 分组列）",
        examples=["industry_neutralize(factor, industry)"],
        param_names=["x", "group"],
        return_type="series",
        tags=["group", "industry", "neutralize"],
    )

    def _calculate_series(self, x: pd.DataFrame, group: pd.DataFrame | None = None, **kwargs) -> pd.DataFrame:
        if group is None:
            return x.sub(x.mean(axis=1), axis=0)
        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
        for date in x.index:
            x_slice = x.loc[date]
            group_slice = group.loc[date] if date in group.index else None
            if group_slice is None or group_slice.isna().all():
                if fallback_policy == "nan":
                    continue
                if fallback_policy == "keep_original":
                    result.loc[date] = x_slice
                    continue

                result.loc[date] = x_slice - x_slice.mean()
                continue
            for group_val in group_slice.dropna().unique():
                mask = (group_slice == group_val) & x_slice.notna()
                if mask.sum() > 0:
                    group_data = x_slice[mask]
                    result.loc[date, group_data.index] = group_data - group_data.mean()
        return result


@register_operator(name="size_neutralize", category="group_neutralization", business_category="group_neutralization", canonical="size_neutralize", source="factor_dsl_np")
class SizeNeutralize(SeriesOperator):
    """市值中性化：每日对 log(max(MarketCap, 1)) 做截面 OLS，取残差。"""
    metadata = OperatorMetadata(
        name="size_neutralize",
        category="group_neutralization",
        description="市值中性化（对 log(max(market_cap, 1)) 做截面回归残差）",
        examples=["size_neutralize(factor, market_cap)"],
        param_names=["x", "market_cap"],
        return_type="series",
        tags=["group", "size", "neutralize"],
    )

    def _calculate_series(self, x: pd.DataFrame, market_cap: pd.DataFrame | None = None, **kwargs) -> pd.DataFrame:
        from cleaned_operators._numpy_kernels import size_resid_panel_

        if market_cap is None:
            raise ValueError("size_neutralize requires market_cap; use size_neutralize(x, market_cap)")
        cap = market_cap.reindex(index=x.index, columns=x.columns)
        out = size_resid_panel_(
            x.to_numpy(dtype=float, copy=False),
            cap.to_numpy(dtype=float, copy=False),
        )
        return pd.DataFrame(out, index=x.index, columns=x.columns)


@register_operator(
    name="industry_size_neutralize",
    category="group_neutralization",
    business_category="group_neutralization",
    canonical="industry_size_neutralize",
    source="factor_dsl_np",
)
class IndustrySizeNeutralize(SeriesOperator):
    """行业+市值双中性：先行业组内 demean，再对 log(max(MarketCap, 1)) 取残差。"""
    metadata = OperatorMetadata(
        name="industry_size_neutralize",
        category="group_neutralization",
        description="行业中性后再做市值中性（与评估双中性语义一致）",
        examples=["industry_size_neutralize(factor, industry, market_cap)"],
        param_names=["x", "industry", "market_cap"],
        return_type="series",
        tags=["group", "industry", "size", "neutralize"],
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        industry: pd.DataFrame | None = None,
        market_cap: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        from cleaned_operators._numpy_kernels import industry_size_resid_panel_

        if industry is None or market_cap is None:
            raise ValueError(
                "industry_size_neutralize requires (x, industry, market_cap)"
            )
        ind = industry.reindex(index=x.index, columns=x.columns)
        cap = market_cap.reindex(index=x.index, columns=x.columns)
        out = industry_size_resid_panel_(
            x.to_numpy(dtype=float, copy=False),
            ind.to_numpy(),
            cap.to_numpy(dtype=float, copy=False),
        )
        return pd.DataFrame(out, index=x.index, columns=x.columns)


# ``group_rank_linear_weighted_value`` 是 ``group_decay_linear``（组内按排名线性
# 加权）的诚实名称；``group_ts_decay_linear`` 才是真实时间衰减。
from cleaned_operators.registry import OperatorRegistry as _group_registry  # noqa: E402
_group_registry.register_alias("group_rank_linear_weighted_value", "group_decay_linear")

# 本轮 audit item 6/8：``group_rank_weighted_value`` 是诚实 canonical（无 window
# 参数），登记到 extended 表面使 layer_governance 的静态分区检查通过。
from cleaned_operators.operator_surface import extend_extended_only  # noqa: E402
extend_extended_only(["group_rank_weighted_value"])
