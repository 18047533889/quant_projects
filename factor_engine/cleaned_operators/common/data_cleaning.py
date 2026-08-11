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

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from cleaned_operators._causal import causal_linear_extrapolate_panel
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

# R13 P1-12: fillna method spellings that are canonical forward-fill aliases.
# ``fillna(x, method=...)`` must be rewritten onto the single gated ffill
# implementation so the ``forward_fill_allowed`` / ``max_ffill_gap`` contract
# cannot be bypassed by spelling the same operation through ``fillna``.
_FFILL_METHOD_ALIASES = frozenset({"ffill", "pad", "forward_fill"})

# Single source of truth for the forward-fill allowance gate.  A field provider
# declares non-ffillable semantics (returns / events / revisions) by passing
# ``forward_fill_allowed=False`` (missing stays missing, fail-closed).  The
# defaults live here so ``ffill`` and ``fillna(method="ffill")`` share one
# contract.  TODO(r13): derive from ``FieldSpec.missing_policy`` once that field
# exists; today the gate is keyword-driven at the operator boundary.
_FORWARD_FILL_ALLOWED_DEFAULT = True
_MAX_FFILL_GAP_DEFAULT = 0


# ---------------------------------------------------------------------------
# CarryForwardPolicy (R40 #190).  Forward-fill is a FIELD-SEMANTIC decision, not
# a free user choice.  A non-ffillable field (returns / events / revisions)
# must NEVER be forward-filled.  ``CarryForwardPolicy`` carries the allowed /
# max-gap / reset contract; the compile-time gate binds it from the input
# semantic contract and the user can only TIGHTEN (never loosen a provider
# FORBID).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CarryForwardPolicy:
    """Declared carry-forward contract (R40 #190)."""

    allowed: bool = True
    max_gap_sessions: int = 0
    reset_on_session_boundary: bool = True
    reset_on_event: bool = False

    def tighten(self, other: "CarryForwardPolicy") -> "CarryForwardPolicy":
        """Combine a user policy with a provider policy — user can only tighten.

        ``allowed`` is AND (provider FORBID wins); max gap is the MINIMUM of the
        two (the tighter limit wins); reset flags OR (any reset requirement wins).
        """
        return CarryForwardPolicy(
            allowed=bool(self.allowed and other.allowed),
            max_gap_sessions=min(self.max_gap_sessions, other.max_gap_sessions),
            reset_on_session_boundary=bool(self.reset_on_session_boundary or other.reset_on_session_boundary),
            reset_on_event=bool(self.reset_on_event or other.reset_on_event),
        )

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "forward_fill_allowed": self.allowed,
            "max_ffill_gap": self.max_gap_sessions,
        }


def check_carry_forward_policy(policy: CarryForwardPolicy | None, *, canonical: str) -> CarryForwardPolicy:
    """Production gate: forward-fill requires an explicit CarryForwardPolicy.

    An undeclared policy is rejected in production (the old
    ``_FORWARD_FILL_ALLOWED_DEFAULT=True`` silently allowed filling any field,
    including returns/events/revisions that a provider FORBIDS).
    """
    if policy is None:
        raise ValueError(
            f"{canonical}: forward-fill has no declared CarryForwardPolicy; "
            "production requires the allowed/max_gap/reset contract bound from "
            "the field's semantic contract (R40 #190)"
        )
    return policy


def _forward_fill_panel(
    x: pd.DataFrame,
    *,
    forward_fill_allowed: bool = _FORWARD_FILL_ALLOWED_DEFAULT,
    max_ffill_gap: int = _MAX_FFILL_GAP_DEFAULT,
) -> pd.DataFrame:
    """Single gated forward-fill implementation (R13 P1-12).

    ``ffill`` and ``fillna(method="ffill"/"pad"/"forward_fill")`` both land here
    so the gate has exactly one path and cannot be bypassed.
    """
    if not bool(forward_fill_allowed):
        return x.copy()
    gap = int(max_ffill_gap)
    if gap > 0:
        return x.ffill(limit=gap)
    return x.ffill()


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
        import numpy as np

        def _per_column(s: pd.Series) -> pd.Series:
            run = 0.0
            cnt = 0
            out: list[float] = []
            for v in s:
                if pd.isna(v):
                    out.append(np.nan)
                else:
                    run += float(v)
                    cnt += 1
                    out.append(run / cnt)
            return pd.Series(out, index=s.index, dtype=float)

        return x.apply(_per_column)



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
        import numpy as np

        def _per_column(s: pd.Series) -> pd.Series:
            run = 0.0
            out: list[float] = []
            for v in s:
                if pd.isna(v):
                    out.append(np.nan)
                else:
                    run += float(v)
                    out.append(run)
            return pd.Series(out, index=s.index, dtype=float)

        return x.apply(_per_column)



# canonical=ffill backend=pandas_numpy selected=ffill source=data_handling/missing_values.py
@register_operator(name="ffill", category="data_handling", business_category="data_cleaning", canonical="ffill", source="factor_dsl_np")
class FillForward(SeriesOperator):
    """前向填充（受字段前向填充许可门控）"""

    metadata = OperatorMetadata(
        name="ffill",
        category="data_handling",
        description="前向填充（用前值填充NaN）；forward_fill_allowed=False 的字段（returns/events/revisions）拒绝填充，max_ffill_gap 限制最大跨 bar 数",
        examples=["ffill(close)"],
        param_names=["x", "forward_fill_allowed", "max_ffill_gap"],
        param_types={"forward_fill_allowed": bool, "max_ffill_gap": int},
        return_type="series",
        tags=["data_handling", "missing", "forward", "gated"]
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        forward_fill_allowed: bool = _FORWARD_FILL_ALLOWED_DEFAULT,
        max_ffill_gap: int = _MAX_FFILL_GAP_DEFAULT,
        **kwargs,
    ) -> pd.DataFrame:
        # R11 #179: the forward-fill allowance is part of the operator contract,
        # so a field provider can declare returns / events / revisions as
        # non-ffillable.  ``forward_fill_allowed=False`` fails closed (missing
        # stays missing) instead of manufacturing a stale value.
        # R13 P1-12: single implementation path — ``fillna(method="ffill")``
        # delegates here so the gate cannot be bypassed.
        return _forward_fill_panel(
            x,
            forward_fill_allowed=forward_fill_allowed,
            max_ffill_gap=max_ffill_gap,
        )

# aliases: FillForward, fillna_forward



# canonical=fillna backend=pandas_numpy selected=fillna source=data_handling/missing_values.py
@register_operator(name="fillna", category="data_handling", business_category="data_cleaning", canonical="fillna", source="factor_dsl_np")
class FillNA(SeriesOperator):
    """缺失值填充"""

    metadata = OperatorMetadata(
        name="fillna",
        category="data_handling",
        description="缺失值填充（method: 'mean', 'median', 'zero', 'ffill'）",
        examples=["fillna(close, 'mean')", "fillna(close, 0)"],
        param_names=["x", "method"],
        return_type="series",
        tags=["data_handling", "missing", "fill"]
    )

    def _calculate_series(self, x: pd.DataFrame, method='zero', **kwargs) -> pd.DataFrame:
        if isinstance(method, (int, float)) and not isinstance(method, bool):
            return x.fillna(method)
        if isinstance(method, str) and method.strip().lower() in _FFILL_METHOD_ALIASES:
            # R13 P1-12: canonical rewrite — forward fill must go through the
            # single gated ffill implementation so ``forward_fill_allowed`` /
            # ``max_ffill_gap`` apply.  Previously this branch called
            # ``x.fillna(method='ffill')`` directly and bypassed the gate.
            return _forward_fill_panel(
                x,
                forward_fill_allowed=bool(
                    kwargs.get("forward_fill_allowed", _FORWARD_FILL_ALLOWED_DEFAULT)
                ),
                max_ffill_gap=int(
                    kwargs.get("max_ffill_gap", _MAX_FFILL_GAP_DEFAULT)
                ),
            )
        if method == 'mean':
            return x.fillna(x.mean(axis=1), axis=0)
        elif method == 'median':
            return x.fillna(x.median(axis=1), axis=0)
        elif method == 'zero':
            return x.fillna(0)
        elif method == 'bfill':
            raise ValueError("fillna(method='bfill') was removed because it is not point-in-time safe")
        else:
            # A misspelled or unsupported method must fail loudly, never silently
            # collapse to a zero-fill (that turns config errors into a constant
            # factor and masks the mistake).
            raise ValueError(
                f"unknown fillna method: {method!r} "
                "(supported: 'mean', 'median', 'zero', 'ffill' or a constant value)"
            )

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



# canonical=causal_linear_extrapolate backend=pandas_numpy source=data_handling/missing_values.py
@register_operator(name="causal_linear_extrapolate", category="data_handling", business_category="data_cleaning", canonical="causal_linear_extrapolate", source="factor_dsl_np", status="research")
class CausalLinearExtrapolate(SeriesOperator):
    """仅使用历史已知点做线性外推。"""
    metadata = OperatorMetadata(
        name="causal_linear_extrapolate",
        category="data_handling",
        description="仅使用最近两个历史有效点线性外推缺失值",
        examples=["causal_linear_extrapolate(close)"],
        param_names=["x"],
        return_type="series",
        tags=["data_handling", "missing", "causal", "extrapolate", "pit_safe"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return causal_linear_extrapolate_panel(x)



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



def _nan_only_to_num(arr: np.ndarray, num: float) -> np.ndarray:
    """R40 #192: replace ONLY NaN with ``num``; ±Inf passes through unchanged.

    ``np.nan_to_num(arr, nan=num, posinf=num, neginf=num)`` silently collapsed
    ±Inf into ``num`` — a naming/semantic trap for a "nan_to_num" operator.  The
    canonical semantics handle only NaN; callers who want ±Inf mapped too use
    the explicitly-named :func:`_nonfinite_to_num` / ``nonfinite_to_num``.
    """
    out = np.array(arr, dtype=float, copy=True)
    out[np.isnan(out)] = num
    return out


def _nonfinite_to_num(arr: np.ndarray, num: float) -> np.ndarray:
    """Replace NaN AND ±Inf with ``num`` (legacy nan_to_num behavior)."""
    out = np.array(arr, dtype=float, copy=True)
    out[~np.isfinite(out)] = num
    return out


# canonical=nan_to_num backend=pandas_numpy selected=nan_to_num source=data_handling/missing_values.py
@register_operator(name="nan_to_num", category="data_handling", business_category="data_cleaning", canonical="nan_to_num", source="factor_dsl_np")
class NaNToNum(SeriesOperator):
    """NaN转数值（仅 NaN；±Inf 透传）"""

    metadata = OperatorMetadata(
        name="nan_to_num",
        category="data_handling",
        description="仅把 NaN 转为 num；±Inf 保持不变（R40 #192）。需要同时映射 ±Inf 请用 nonfinite_to_num",
        examples=["nan_to_num(close, 0)"],
        param_names=["x", "num"],
        return_type="series",
        tags=["data_handling", "missing", "fill"]
    )

    def _calculate_series(self, x: pd.DataFrame, num: float = 0, **kwargs) -> pd.DataFrame:
        arr = x.to_numpy(dtype=float, copy=True)
        filled = _nan_only_to_num(arr, num)
        return pd.DataFrame(filled, index=x.index, columns=x.columns)

# aliases: NAN_TO_NUM


# canonical=nonfinite_to_num backend=pandas_numpy
@register_operator(name="nonfinite_to_num", category="data_handling", business_category="data_cleaning", canonical="nonfinite_to_num", source="factor_dsl_np", status="research")
class NonFiniteToNum(SeriesOperator):
    """NaN 与 ±Inf 全部转为 num（旧 nan_to_num 行为，显式命名）。"""

    metadata = OperatorMetadata(
        name="nonfinite_to_num",
        category="data_handling",
        description="把 NaN 与 ±Inf 全部转为 num（旧 nan_to_num 行为；R40 #192 语义澄清）",
        examples=["nonfinite_to_num(close, 0)"],
        param_names=["x", "num"],
        return_type="series",
        tags=["data_handling", "missing", "fill"]
    )

    def _calculate_series(self, x: pd.DataFrame, num: float = 0, **kwargs) -> pd.DataFrame:
        arr = x.to_numpy(dtype=float, copy=True)
        filled = _nonfinite_to_num(arr, num)
        return pd.DataFrame(filled, index=x.index, columns=x.columns)



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
