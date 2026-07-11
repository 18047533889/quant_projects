# -*- coding: utf-8 -*-
"""
缺失值处理、截断与保护性窗口算子。

语义
----
- **填充**：``fillna``、``ffill``、``bfill``、``coalesce`` — 处理 NaN/空值；
- **检测**：``is_nan``、``is_finite`` — 布尔或掩码；
- **截断/保护**：``clip``、``protected_div``、``protected_log`` 等 — 避免除零或对数非法域；
- **窗口裁剪**：与 rolling 配合的 ``protected_*`` 变体。

用于数据质量与数值稳定；不改变时间/截面维度，只替换或标记元素值。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from cleaned_operators._causal import causal_bfill, causal_interpolate_panel
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

# canonical=causal_bfill backend=pandas_numpy selected=causal_bfill source=data_handling/missing_values.py
@register_operator(
    name="causal_bfill",
    category="data_handling",
    business_category="data_cleaning",
    canonical="causal_bfill",
    source="factor_dsl_np",
    status="research",
)
class CausalBFill(SeriesOperator):
    """因果后向占位：不引用未来值，NaN 保持 NaN。"""

    metadata = OperatorMetadata(
        name="causal_bfill",
        category="data_handling",
        description="因果后向占位（非传统 bfill；不引用未来值，NaN 保持 NaN）",
        examples=["causal_bfill(close)"],
        param_names=["x"],
        return_type="series",
        tags=["data_handling", "missing", "causal", "pit_safe"],
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return causal_bfill(x)


# canonical=bfill backend=pandas_numpy selected=bfill source=data_handling/missing_values.py
@register_operator(
    name="bfill",
    category="data_handling",
    business_category="data_cleaning",
    canonical="bfill",
    source="factor_dsl_np",
    status="deprecated",
)
class FillBackward(SeriesOperator):
    """已弃用：请使用 ``causal_bfill``（名称易误解为传统 backward fill）。"""

    metadata = OperatorMetadata(
        name="bfill",
        category="data_handling",
        description="[deprecated] 请改用 causal_bfill；因子链路禁前视，不引用未来值",
        examples=["causal_bfill(close)"],
        param_names=["x"],
        return_type="series",
        tags=["data_handling", "missing", "backward", "deprecated"],
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return causal_bfill(x)

# aliases: FillBackward, fillna_backward



# canonical=dropna backend=pandas_numpy selected=dropna source=data_handling/missing_values.py
@register_operator(name="dropna", category="data_handling", business_category="data_cleaning", canonical="dropna", source="factor_dsl_np", status="research")
class DropNA(SeriesOperator):
    """删除缺失值"""

    metadata = OperatorMetadata(
        name="dropna",
        category="data_handling",
        description="删除缺失值（返回非NaN索引）",
        examples=["dropna(factor)"],
        param_names=["x"],
        return_type="series",
        tags=["data_handling", "missing", "drop"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.dropna()



# canonical=ewm backend=pandas_numpy selected=ewm source=data_handling/window_ops.py
@register_operator(name="ewm", category="data_handling", business_category="data_cleaning", canonical="ewm", source="factor_dsl_np")
class EWM(SeriesOperator):
    """指数加权移动"""

    metadata = OperatorMetadata(
        name="ewm",
        category="data_handling",
        description="指数加权移动",
        examples=["ewm(close, 0.1)"],
        param_names=["x", "alpha"],
        return_type="series",
        tags=["data_handling", "ewm", "exponential"]
    )

    def _calculate_series(self, x: pd.DataFrame, alpha: float = 0.1, **kwargs) -> pd.DataFrame:
        return x.ewm(alpha=alpha, adjust=False).mean()



# canonical=ewm_corr backend=pandas_numpy selected=ewm_corr source=data_handling/window_ops.py
@register_operator(name="ewm_corr", category="data_handling", business_category="data_cleaning", canonical="ewm_corr", source="factor_dsl_np")
class EWMCorr(SeriesOperator):
    """指数加权相关系数"""
    metadata = OperatorMetadata(
        name="ewm_corr",
        category="data_handling",
        description="指数加权相关系数",
        examples=["ewm_corr(close, volume, 20)"],
        param_names=["x", "y", "span"],
        return_type="series",
        tags=["data_handling", "ewm", "corr"]
    )

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, span: int = 20, **kwargs) -> pd.DataFrame:
        return x.ewm(span=span, adjust=False).corr(y)



# canonical=ewm_cov backend=pandas_numpy selected=ewm_cov source=data_handling/window_ops.py
@register_operator(name="ewm_cov", category="data_handling", business_category="data_cleaning", canonical="ewm_cov", source="factor_dsl_np")
class EWMCov(SeriesOperator):
    """指数加权协方差"""
    metadata = OperatorMetadata(
        name="ewm_cov",
        category="data_handling",
        description="指数加权协方差",
        examples=["ewm_cov(close, volume, 20)"],
        param_names=["x", "y", "span"],
        return_type="series",
        tags=["data_handling", "ewm", "cov"]
    )

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, span: int = 20, **kwargs) -> pd.DataFrame:
        return x.ewm(span=span, adjust=False).cov(y)



# canonical=ewm_mean backend=pandas_numpy selected=ewm_mean source=data_handling/window_ops.py
@register_operator(name="ewm_mean", category="data_handling", business_category="data_cleaning", canonical="ewm_mean", source="factor_dsl_np")
class EWMMean(SeriesOperator):
    """指数移动平均"""

    metadata = OperatorMetadata(
        name="ewm_mean",
        category="data_handling",
        description="EMA 指数移动平均",
        examples=["ewm_mean(close, 20)"],
        param_names=["x", "span"],
        return_type="series",
        tags=["data_handling", "ewm", "ema"]
    )

    def _calculate_series(self, x: pd.DataFrame, span: int = 20, **kwargs) -> pd.DataFrame:
        return x.ewm(span=span, adjust=False).mean()



# canonical=ewm_std backend=pandas_numpy selected=ewm_std source=data_handling/window_ops.py
@register_operator(name="ewm_std", category="data_handling", business_category="data_cleaning", canonical="ewm_std", source="factor_dsl_np")
class EWMStd(SeriesOperator):
    """EW标准差"""

    metadata = OperatorMetadata(
        name="ewm_std",
        category="data_handling",
        description="EW标准差",
        examples=["ewm_std(returns, 20)"],
        param_names=["x", "span"],
        return_type="series",
        tags=["data_handling", "ewm", "std"]
    )

    def _calculate_series(self, x: pd.DataFrame, span: int = 20, **kwargs) -> pd.DataFrame:
        return x.ewm(span=span, adjust=False).std()



# canonical=ewm_var backend=pandas_numpy selected=ewm_var source=data_handling/window_ops.py
@register_operator(name="ewm_var", category="data_handling", business_category="data_cleaning", canonical="ewm_var", source="factor_dsl_np")
class EWMVar(SeriesOperator):
    """指数加权方差"""
    metadata = OperatorMetadata(
        name="ewm_var",
        category="data_handling",
        description="指数加权方差",
        examples=["ewm_var(returns, 20)"],
        param_names=["x", "span"],
        return_type="series",
        tags=["data_handling", "ewm", "var"]
    )

    def _calculate_series(self, x: pd.DataFrame, span: int = 20, **kwargs) -> pd.DataFrame:
        return x.ewm(span=span, adjust=False).var()



# canonical=expanding_max backend=pandas_numpy selected=expanding_max source=data_handling/window_ops.py
@register_operator(name="expanding_max", category="data_handling", business_category="data_cleaning", canonical="expanding_max", source="factor_dsl_np")
class ExpandingMax(SeriesOperator):
    """扩展窗口最大值"""
    metadata = OperatorMetadata(
        name="expanding_max",
        category="data_handling",
        description="扩展窗口最大值",
        examples=["expanding_max(high)"],
        param_names=["x"],
        return_type="series",
        tags=["data_handling", "expanding", "max"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.expanding(min_periods=1).max()



# canonical=expanding_mean backend=pandas_numpy selected=expanding_mean source=data_handling/window_ops.py
@register_operator(name="expanding_mean", category="data_handling", business_category="data_cleaning", canonical="expanding_mean", source="factor_dsl_np")
class ExpandingMean(SeriesOperator):
    """扩展窗口均值"""
    metadata = OperatorMetadata(
        name="expanding_mean",
        category="data_handling",
        description="扩展窗口均值",
        examples=["expanding_mean(close)"],
        param_names=["x"],
        return_type="series",
        tags=["data_handling", "expanding", "mean"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.expanding(min_periods=1).mean()



# canonical=expanding_min backend=pandas_numpy selected=expanding_min source=data_handling/window_ops.py
@register_operator(name="expanding_min", category="data_handling", business_category="data_cleaning", canonical="expanding_min", source="factor_dsl_np")
class ExpandingMin(SeriesOperator):
    """扩展窗口最小值"""
    metadata = OperatorMetadata(
        name="expanding_min",
        category="data_handling",
        description="扩展窗口最小值",
        examples=["expanding_min(low)"],
        param_names=["x"],
        return_type="series",
        tags=["data_handling", "expanding", "min"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.expanding(min_periods=1).min()



# canonical=expanding_rank backend=pandas_numpy selected=expanding_rank source=data_handling/window_ops.py
@register_operator(name="expanding_rank", category="data_handling", business_category="data_cleaning", canonical="expanding_rank", source="factor_dsl_np")
class ExpandingRank(SeriesOperator):
    """扩展窗口排名（当前值在历史中的百分位排名）"""
    metadata = OperatorMetadata(
        name="expanding_rank",
        category="data_handling",
        description="扩展窗口排名（当前值在历史中的百分位排名）",
        examples=["expanding_rank(close)"],
        param_names=["x"],
        return_type="series",
        tags=["data_handling", "expanding", "rank"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.expanding(min_periods=1).rank(pct=True)



# canonical=expanding_std backend=pandas_numpy selected=expanding_std source=data_handling/window_ops.py
@register_operator(name="expanding_std", category="data_handling", business_category="data_cleaning", canonical="expanding_std", source="factor_dsl_np")
class ExpandingStd(SeriesOperator):
    """扩展窗口标准差"""
    metadata = OperatorMetadata(
        name="expanding_std",
        category="data_handling",
        description="扩展窗口标准差",
        examples=["expanding_std(returns)"],
        param_names=["x"],
        return_type="series",
        tags=["data_handling", "expanding", "std"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.expanding(min_periods=2).std()



# canonical=expanding_sum backend=pandas_numpy selected=expanding_sum source=data_handling/window_ops.py
@register_operator(name="expanding_sum", category="data_handling", business_category="data_cleaning", canonical="expanding_sum", source="factor_dsl_np")
class ExpandingSum(SeriesOperator):
    """扩展窗口求和"""
    metadata = OperatorMetadata(
        name="expanding_sum",
        category="data_handling",
        description="扩展窗口求和",
        examples=["expanding_sum(volume)"],
        param_names=["x"],
        return_type="series",
        tags=["data_handling", "expanding", "sum"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.expanding(min_periods=1).sum()



# canonical=ffill backend=pandas_numpy selected=ffill source=data_handling/missing_values.py
@register_operator(name="ffill", category="data_handling", business_category="data_cleaning", canonical="ffill", source="factor_dsl_np")
class FillForward(SeriesOperator):
    """前向填充"""

    metadata = OperatorMetadata(
        name="ffill",
        category="data_handling",
        description="前向填充（用前值填充NaN）",
        examples=["ffill(close)"],
        param_names=["x"],
        return_type="series",
        tags=["data_handling", "missing", "forward"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.ffill()

# aliases: FillForward, fillna_forward



# canonical=fillna backend=pandas_numpy selected=fillna source=data_handling/missing_values.py
@register_operator(name="fillna", category="data_handling", business_category="data_cleaning", canonical="fillna", source="factor_dsl_np")
class FillNA(SeriesOperator):
    """缺失值填充"""

    metadata = OperatorMetadata(
        name="fillna",
        category="data_handling",
        description="缺失值填充（method: 'mean', 'median', 'zero', 'ffill', 'bfill'）",
        examples=["fillna(close, 'mean')", "fillna(close, 0)"],
        param_names=["x", "method"],
        return_type="series",
        tags=["data_handling", "missing", "fill"]
    )

    def _calculate_series(self, x: pd.DataFrame, method='zero', **kwargs) -> pd.DataFrame:
        if isinstance(method, (int, float)) and not isinstance(method, bool):
            return x.fillna(method)
        if method == 'mean':
            return x.fillna(x.mean(axis=1), axis=0)
        elif method == 'median':
            return x.fillna(x.median(axis=1), axis=0)
        elif method == 'zero':
            return x.fillna(0)
        elif method == 'ffill':
            return x.fillna(method='ffill')
        elif method == 'bfill':
            return causal_bfill(x)
        else:
            return x.fillna(method='zero')

# aliases: FillNA



# canonical=fillna_const backend=pandas_numpy selected=fillna_const source=data_handling/missing_values.py
@register_operator(name="fillna_const", category="data_handling", business_category="data_cleaning", canonical="fillna_const", source="factor_dsl_np")
class FillNAConst(SeriesOperator):
    """常量填充"""

    metadata = OperatorMetadata(
        name="fillna_const",
        category="data_handling",
        description="常量填充",
        examples=["fillna_const(close, 0)"],
        param_names=["x", "value"],
        return_type="series",
        tags=["data_handling", "missing", "fill"]
    )

    def _calculate_series(self, x: pd.DataFrame, value: float = 0, **kwargs) -> pd.DataFrame:
        return x.fillna(value)



# canonical=fillna_interpolate backend=pandas_numpy selected=fillna_interpolate source=data_handling/missing_values.py
@register_operator(name="fillna_interpolate", category="data_handling", business_category="data_cleaning", canonical="fillna_interpolate", source="factor_dsl_np")
class FillnaInterpolate(SeriesOperator):
    """插值填充缺失值（method: linear, quadratic, cubic）"""
    metadata = OperatorMetadata(
        name="fillna_interpolate",
        category="data_handling",
        description="插值填充缺失值（method: linear, quadratic, cubic）",
        examples=["fillna_interpolate(close, 'linear')", "fillna_interpolate(close, 'cubic')"],
        param_names=["x", "method"],
        return_type="series",
        tags=["data_handling", "missing", "interpolate"]
    )

    def _calculate_series(self, x: pd.DataFrame, method: str = 'linear', **kwargs) -> pd.DataFrame:
        return causal_interpolate_panel(x, method=method)



# canonical=is_inf backend=pandas_numpy selected=is_inf source=data_handling/missing_values.py
@register_operator(name="is_inf", category="data_handling", business_category="data_cleaning", canonical="is_inf", source="factor_dsl_np")
class IsInf(SeriesOperator):
    """判断是否为无穷"""

    metadata = OperatorMetadata(
        name="is_inf",
        category="data_handling",
        description="判断是否为无穷",
        examples=["is_inf(value)"],
        param_names=["x"],
        return_type="series",
        tags=["data_handling", "infinite", "check"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        from backend.elementwise_semantics import is_infinite_pandas

        return is_infinite_pandas(x)



# canonical=is_nan backend=pandas_numpy selected=is_nan source=data_handling/missing_values.py
@register_operator(name="is_nan", category="data_handling", business_category="data_cleaning", canonical="is_nan", source="factor_dsl_np")
class IsNaN(SeriesOperator):
    """判断是否为NaN"""

    metadata = OperatorMetadata(
        name="is_nan",
        category="data_handling",
        description="判断是否为NaN",
        examples=["is_nan(close)"],
        param_names=["x"],
        return_type="series",
        tags=["data_handling", "missing", "check"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        import numpy as np

        arr = x.to_numpy(dtype=np.float64, copy=False)
        out = np.isnan(arr).astype(np.float64)
        return pd.DataFrame(out, index=x.index, columns=x.columns)


# canonical=is_null backend=pandas_numpy
@register_operator(name="is_null", category="data_handling", business_category="data_cleaning", canonical="is_null", source="factor_dsl_np")
class IsNull(SeriesOperator):
    """判断是否为 NULL/缺失。"""

    metadata = OperatorMetadata(
        name="is_null",
        category="data_handling",
        description="判断是否为 NULL/缺失",
        examples=["is_null(close)"],
        param_names=["x"],
        return_type="series",
        tags=["data_handling", "missing", "check"],
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.isna().astype(float)


# canonical=is_not_null backend=pandas_numpy
@register_operator(name="is_not_null", category="data_handling", business_category="data_cleaning", canonical="is_not_null", source="factor_dsl_np")
class IsNotNull(SeriesOperator):
    """判断是否非 NULL。"""

    metadata = OperatorMetadata(
        name="is_not_null",
        category="data_handling",
        description="判断是否非 NULL",
        examples=["is_not_null(close)"],
        param_names=["x"],
        return_type="series",
        tags=["data_handling", "missing", "check"],
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.notna().astype(float)


# canonical=is_infinite backend=pandas_numpy
@register_operator(name="is_infinite", category="data_handling", business_category="data_cleaning", canonical="is_infinite", source="factor_dsl_np")
class IsInfinite(SeriesOperator):
    """判断是否为 ±Inf（NULL/NaN/有限 → 0）。"""

    metadata = OperatorMetadata(
        name="is_infinite",
        category="data_handling",
        description="判断是否为 ±Inf",
        examples=["is_infinite(value)"],
        param_names=["x"],
        return_type="series",
        tags=["data_handling", "check"],
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        from backend.elementwise_semantics import is_infinite_pandas

        return is_infinite_pandas(x)



# canonical=nan_to_num backend=pandas_numpy selected=nan_to_num source=data_handling/missing_values.py
@register_operator(name="nan_to_num", category="data_handling", business_category="data_cleaning", canonical="nan_to_num", source="factor_dsl_np")
class NaNToNum(SeriesOperator):
    """NaN转数值"""

    metadata = OperatorMetadata(
        name="nan_to_num",
        category="data_handling",
        description="NaN转数值",
        examples=["nan_to_num(close, 0)"],
        param_names=["x", "num"],
        return_type="series",
        tags=["data_handling", "missing", "fill"]
    )

    def _calculate_series(self, x: pd.DataFrame, num: float = 0, **kwargs) -> pd.DataFrame:
        arr = x.to_numpy(dtype=float, copy=True)
        filled = np.nan_to_num(arr, nan=num, posinf=num, neginf=num)
        return pd.DataFrame(filled, index=x.index, columns=x.columns)

# aliases: NAN_TO_NUM



# canonical=window_max backend=pandas_numpy selected=window_max source=data_handling/window_ops.py
@register_operator(name="window_max", category="data_handling", business_category="data_cleaning", canonical="window_max", source="factor_dsl_np")
class WindowMax(SeriesOperator):
    """窗口最大值"""

    metadata = OperatorMetadata(
        name="window_max",
        category="data_handling",
        description="窗口最大值",
        examples=["window_max(high, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["data_handling", "window", "max"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).max()



# canonical=window_mean backend=pandas_numpy selected=window_mean source=data_handling/window_ops.py
@register_operator(name="window_mean", category="data_handling", business_category="data_cleaning", canonical="window_mean", source="factor_dsl_np")
class WindowMean(SeriesOperator):
    """窗口均值"""

    metadata = OperatorMetadata(
        name="window_mean",
        category="data_handling",
        description="窗口均值",
        examples=["window_mean(close, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["data_handling", "window", "mean"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).mean()



# canonical=window_min backend=pandas_numpy selected=window_min source=data_handling/window_ops.py
@register_operator(name="window_min", category="data_handling", business_category="data_cleaning", canonical="window_min", source="factor_dsl_np")
class WindowMin(SeriesOperator):
    """窗口最小值"""

    metadata = OperatorMetadata(
        name="window_min",
        category="data_handling",
        description="窗口最小值",
        examples=["window_min(low, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["data_handling", "window", "min"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).min()



# canonical=window_std backend=pandas_numpy selected=window_std source=data_handling/window_ops.py
@register_operator(name="window_std", category="data_handling", business_category="data_cleaning", canonical="window_std", source="factor_dsl_np")
class WindowStd(SeriesOperator):
    """窗口标准差"""

    metadata = OperatorMetadata(
        name="window_std",
        category="data_handling",
        description="窗口标准差",
        examples=["window_std(returns, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["data_handling", "window", "std"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).std()



# canonical=window_sum backend=pandas_numpy selected=window_sum source=data_handling/window_ops.py
@register_operator(name="window_sum", category="data_handling", business_category="data_cleaning", canonical="window_sum", source="factor_dsl_np")
class WindowSum(SeriesOperator):
    """窗口求和"""

    metadata = OperatorMetadata(
        name="window_sum",
        category="data_handling",
        description="窗口求和",
        examples=["window_sum(volume, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["data_handling", "window", "sum"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).sum()


# canonical=protected_div backend=pandas_numpy selected=protected_div source=basic_runtime
@register_operator(name="protected_div", category="data_cleaning", business_category="data_cleaning", canonical="protected_div", source="basic_runtime")
class ProtectedDivOp(TwoVarOperator):
    """安全除法：|y|<=epsilon 时返回 default"""
    metadata = OperatorMetadata(
        name="protected_div",
        category="data_cleaning",
        description="安全除法：|y|<=epsilon 时返回 default",
        param_names=["x", "y", "epsilon", "default"],
        return_type="series",
        tags=["data_cleaning", "pit_safe"],
    )

    def _calculate_series(self, x, y, epsilon=1e-12, default=0.0, **kwargs):
        from backend.elementwise_semantics import protected_div_pandas

        return protected_div_pandas(x, y, epsilon=float(epsilon), default=float(default))


# canonical=safe_div_null backend=pandas_numpy selected=safe_div_null source=basic_runtime
@register_operator(name="safe_div_null", category="data_cleaning", business_category="data_cleaning", canonical="safe_div_null", source="basic_runtime")
class SafeDivNullOp(TwoVarOperator):
    """比率除法：零/NULL 分母 → NULL（不填充 default）。"""
    metadata = OperatorMetadata(
        name="safe_div_null",
        category="data_cleaning",
        description="比率除法：numerator/denominator；NULL 或 |denom|<=epsilon → NULL",
        param_names=["x", "y", "epsilon"],
        return_type="series",
        tags=["data_cleaning", "pit_safe", "ratio"],
    )

    def _calculate_series(self, x, y, epsilon=1e-12, **kwargs):
        denom = y.where(y.abs() > epsilon)
        out = x / denom
        return out.replace([np.inf, -np.inf], np.nan)


# canonical=protected_log backend=pandas_numpy selected=protected_log source=basic_runtime
@register_operator(name="protected_log", category="data_cleaning", business_category="data_cleaning", canonical="protected_log", source="basic_runtime")
class ProtectedLogOp(SeriesOperator):
    """安全对数：log(max(x, epsilon))"""
    metadata = OperatorMetadata(
        name="protected_log",
        category="data_cleaning",
        description="安全对数：log(max(x, epsilon))",
        param_names=["x", "epsilon"],
        return_type="series",
        tags=["data_cleaning", "pit_safe"],
    )

    def _calculate_series(self, x, epsilon=1e-12, **kwargs):
        from backend.elementwise_semantics import protected_log_pandas

        return protected_log_pandas(x, epsilon=float(epsilon))


# canonical=div_or_null backend=pandas_numpy
@register_operator(name="div_or_null", category="data_cleaning", business_category="data_cleaning", canonical="div_or_null", source="basic_runtime")
class DivOrNullOp(TwoVarOperator):
    """比率除法：NULL 保持 NULL（同 protected_div 新语义）。"""
    metadata = OperatorMetadata(
        name="div_or_null",
        category="data_cleaning",
        description="NULL 保持 NULL 的安全除法",
        param_names=["x", "y", "epsilon", "default"],
        return_type="series",
        tags=["data_cleaning", "pit_safe"],
    )

    def _calculate_series(self, x, y, epsilon=1e-12, default=0.0, **kwargs):
        from backend.elementwise_semantics import protected_div_pandas

        return protected_div_pandas(x, y, epsilon=float(epsilon), default=float(default))


# canonical=div_or_default backend=pandas_numpy
@register_operator(name="div_or_default", category="data_cleaning", business_category="data_cleaning", canonical="div_or_default", source="basic_runtime")
class DivOrDefaultOp(TwoVarOperator):
    """除法：NULL/小分母 → default。"""
    metadata = OperatorMetadata(
        name="div_or_default",
        category="data_cleaning",
        description="NULL 或 |denom|<=epsilon 时返回 default",
        param_names=["x", "y", "epsilon", "default"],
        return_type="series",
        tags=["data_cleaning"],
    )

    def _calculate_series(self, x, y, epsilon=1e-12, default=0.0, **kwargs):
        from backend.elementwise_semantics import div_or_default_pandas

        return div_or_default_pandas(x, y, epsilon=float(epsilon), default=float(default))


# canonical=log_fill_invalid backend=pandas_numpy
@register_operator(name="log_fill_invalid", category="data_cleaning", business_category="data_cleaning", canonical="log_fill_invalid", source="basic_runtime")
class LogFillInvalidOp(SeriesOperator):
    """对数：NULL/非法域 → log(epsilon)。"""
    metadata = OperatorMetadata(
        name="log_fill_invalid",
        category="data_cleaning",
        description="NULL/非法域填充 log(epsilon)",
        param_names=["x", "epsilon"],
        return_type="series",
        tags=["data_cleaning"],
    )

    def _calculate_series(self, x, epsilon=1e-12, **kwargs):
        from backend.elementwise_semantics import log_fill_invalid_pandas

        return log_fill_invalid_pandas(x, epsilon=float(epsilon))


# canonical=protected_sqrt backend=pandas_numpy selected=protected_sqrt source=basic_runtime
@register_operator(name="protected_sqrt", category="data_cleaning", business_category="data_cleaning", canonical="protected_sqrt", source="basic_runtime")
class ProtectedSqrtOp(SeriesOperator):
    """安全平方根：sqrt(max(x, 0))"""
    metadata = OperatorMetadata(
        name="protected_sqrt",
        category="data_cleaning",
        description="安全平方根：sqrt(max(x, 0))",
        param_names=["x"],
        return_type="series",
        tags=["data_cleaning", "pit_safe"],
    )

    def _calculate_series(self, x, **kwargs):
        return np.sqrt(x.clip(lower=0))
