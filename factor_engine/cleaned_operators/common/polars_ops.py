# -*- coding: utf-8 -*-
"""高频元素级与时序算子的 Polars 实现（auto / hybrid 路径优先选用）。

数据约定
--------
- 输入/输出均为 **宽表 panel**：行 = 时间，列 = 标的（另有 ``date`` / ``stock_code`` 元数据列时跳过）。
- 二元算子要求 x/y 列名对齐，逐列独立计算。

实现策略
--------
- **四则运算**（``add/subtract/multiply/divide``）：与 SQL 下推节点对称，hybrid 在 SQL 不可编译时可回退 Polars。
- **Top-N / 秩相关**：复用 ``_rolling_fast`` / ``_numpy_kernels``，保证与 pandas backend 数值一致。
- Polars 无原生 API 或语义复杂时，走 **pandas 桥接**（转宽表 → 计算 → 写回 Polars）。
"""
from __future__ import annotations

import numpy as np

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

from factor_engine.cleaned_operators.base import ParamRole, ParamSpec
from factor_engine.cleaned_operators.base_polars import (
    OperatorMetadata,
    SeriesOperator,
    panel_pandas_bridge,
    register_operator,
)

_SKIP = frozenset({"date", "stock_code"})  # 宽表元数据列，不参与因子计算


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    """返回 panel 中参与计算的标的列名。

参数:
    df: Polars 宽表 panel。

返回:
    排除 ``date``/``stock_code`` 后的列名列表。
"""
    return [c for c in df.columns if c not in _SKIP]


def _align_cols(*dfs: pl.DataFrame) -> list[str]:
    """多输入宽表严格对齐（review #5 R5-03）。

    生产语义：Panel×Panel 二元/多元算子要求 index 完全相同、column 完全相同、
    column 顺序完全相同。历史实现按列名取交集——当 x=[A,B,C]、y=[A,B] 时只算
    A/B 而把 C 留在结果 frame，制造"列对不上"的静默漂移。现在列集不一致直接
    raise，只有显式 Join/Broadcast 算子（tag）才允许例外。

参数:
    *dfs: 一个或多个 Polars 宽表。

返回:
    第一个输入的全部数值列名列表。

异常:
    ValueError: 任一输入的数值列集与第一个输入不一致时。
"""
    base = _numeric_cols(dfs[0])
    for df in dfs[1:]:
        other = _numeric_cols(df)
        if other != base:
            raise ValueError(
                f"polars panel column mismatch: base {base} vs input {other}; "
                "Panel×Panel operators require identical column sets in the "
                "same order (R5-03)"
            )
    return base


def _binary_colwise(combine):
    """工厂函数：生成逐列二元运算的 ``_calculate_series`` 方法。

参数:
    combine: 接收两个 Polars 表达式返回组合结果的函数。

返回:
    可绑定为算子类 ``_calculate_series`` 的方法。
"""

    def calc(self, x, y, **kwargs) -> pl.DataFrame:
        if isinstance(x, (int, float)) and isinstance(y, pl.DataFrame):
            cols = _numeric_cols(y)
            lit = float(x)
            return y.with_columns([combine(lit, pl.col(c)).alias(c) for c in cols])
        if isinstance(y, (int, float)):
            if not isinstance(x, pl.DataFrame):
                raise TypeError(f"expected polars DataFrame or scalar, got {type(x)!r}")
            cols = _numeric_cols(x)
            lit = float(y)
            return x.with_columns([combine(pl.col(c), lit).alias(c) for c in cols])
        if not isinstance(x, pl.DataFrame):
            raise TypeError(f"expected polars DataFrame or scalar, got {type(x)!r}")
        if not isinstance(y, pl.DataFrame):
            raise TypeError(f"expected polars DataFrame or scalar, got {type(y)!r}")
        cols = _align_cols(x, y)
        return x.with_columns([combine(pl.col(c), y[c]).alias(c) for c in cols])

    return calc


# ---------------------------------------------------------------------------
# SQL 下推四则：与 ``backend/sql_pushdown`` 白名单对称，供 hybrid 降级使用
# ---------------------------------------------------------------------------


@register_operator(name="add", category="elementwise_math", business_category="elementwise_math", canonical="add", source="factor_dsl_polars")
class AddPolars(SeriesOperator):
    """Polars 逐元素加法算子。"""

    metadata = OperatorMetadata(
        name="add", category="elementwise_math", description="逐元素加法",
        param_names=["x", "y"], return_type="series", tags=["elementwise", "polars"],
    )
    _calculate_series = _binary_colwise(lambda a, b: a + b)


@register_operator(name="subtract", category="elementwise_math", business_category="elementwise_math", canonical="subtract", source="factor_dsl_polars")
class SubtractPolars(SeriesOperator):
    """Polars 逐元素减法算子。"""

    metadata = OperatorMetadata(
        name="subtract", category="elementwise_math", description="逐元素减法",
        param_names=["x", "y"], return_type="series", tags=["elementwise", "polars"],
    )
    _calculate_series = _binary_colwise(lambda a, b: a - b)


@register_operator(name="multiply", category="elementwise_math", business_category="elementwise_math", canonical="multiply", source="factor_dsl_polars")
class MultiplyPolars(SeriesOperator):
    """Polars 逐元素乘法算子。"""

    metadata = OperatorMetadata(
        name="multiply", category="elementwise_math", description="逐元素乘法",
        param_names=["x", "y"], return_type="series", tags=["elementwise", "polars"],
    )
    _calculate_series = _binary_colwise(lambda a, b: a * b)


@register_operator(name="divide", category="elementwise_math", business_category="elementwise_math", canonical="divide", source="factor_dsl_polars")
class DividePolars(SeriesOperator):
    """Polars 逐元素除法算子。"""

    metadata = OperatorMetadata(
        name="divide", category="elementwise_math", description="逐元素除法",
        param_names=["x", "y"], return_type="series", tags=["elementwise", "polars"],
    )
    _calculate_series = _binary_colwise(lambda a, b: a / b)


def _unary(expr_fn):
    """工厂函数：生成逐列一元运算的 ``_calculate_series`` 方法。

参数:
    expr_fn: 接收列表达式返回变换结果的函数。

返回:
    可绑定为算子类 ``_calculate_series`` 的方法。
"""
    def calc(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([expr_fn(pl.col(c)).alias(c) for c in cols])

    return calc


@register_operator(name="abs", category="math", business_category="elementwise_math", canonical="abs", source="factor_dsl_polars")
class AbsPolars(SeriesOperator):
    """Polars 绝对值一元算子。"""

    metadata = OperatorMetadata(
        name="abs", category="math", description="绝对值",
        examples=["abs(x)"], param_names=["x"], return_type="series", tags=["math", "polars"],
    )
    _calculate_series = _unary(lambda c: c.abs())


@register_operator(name="neg", category="math", business_category="elementwise_math", canonical="neg", source="factor_dsl_polars")
class NegPolars(SeriesOperator):
    """Polars 取负一元算子。"""

    metadata = OperatorMetadata(
        name="neg", category="math", description="取负",
        examples=["neg(x)"], param_names=["x"], return_type="series", tags=["math", "polars"],
    )
    _calculate_series = _unary(lambda c: -c)


@register_operator(name="log", category="math", business_category="elementwise_math", canonical="log", source="factor_dsl_polars")
class LogPolars(SeriesOperator):
    """Polars 自然对数一元算子。"""

    metadata = OperatorMetadata(
        name="log", category="math", description="自然对数",
        examples=["log(x)"], param_names=["x"], return_type="series", tags=["math", "polars"],
    )
    _calculate_series = _unary(lambda c: c.log())


@register_operator(name="exp", category="math", business_category="elementwise_math", canonical="exp", source="factor_dsl_polars")
class ExpPolars(SeriesOperator):
    """Polars 指数一元算子。"""

    metadata = OperatorMetadata(
        name="exp", category="math", description="指数",
        examples=["exp(x)"], param_names=["x"], return_type="series", tags=["math", "polars"],
    )
    _calculate_series = _unary(lambda c: c.exp())


@register_operator(name="sqrt", category="math", business_category="elementwise_math", canonical="sqrt", source="factor_dsl_polars")
class SqrtPolars(SeriesOperator):
    """Polars 平方根一元算子。"""

    metadata = OperatorMetadata(
        name="sqrt", category="math", description="平方根",
        examples=["sqrt(x)"], param_names=["x"], return_type="series", tags=["math", "polars"],
    )
    _calculate_series = _unary(lambda c: c.sqrt())


@register_operator(name="sin", category="math", business_category="elementwise_math", canonical="sin", source="factor_dsl_polars")
class SinPolars(SeriesOperator):
    """Polars 正弦一元算子。"""

    metadata = OperatorMetadata(
        name="sin", category="math", description="正弦",
        examples=["sin(x)"], param_names=["x"], return_type="series", tags=["math", "polars"],
    )
    _calculate_series = _unary(lambda c: c.sin())


@register_operator(name="cos", category="math", business_category="elementwise_math", canonical="cos", source="factor_dsl_polars")
class CosPolars(SeriesOperator):
    """Polars 余弦一元算子。"""

    metadata = OperatorMetadata(
        name="cos", category="math", description="余弦",
        examples=["cos(x)"], param_names=["x"], return_type="series", tags=["math", "polars"],
    )
    _calculate_series = _unary(lambda c: c.cos())


@register_operator(name="sign", category="math", business_category="elementwise_math", canonical="sign", source="factor_dsl_polars")
class SignPolars(SeriesOperator):
    """Polars 符号函数一元算子。"""

    metadata = OperatorMetadata(
        name="sign", category="math", description="符号函数",
        examples=["sign(x)"], param_names=["x"], return_type="series", tags=["math", "polars"],
    )
    _calculate_series = _unary(lambda c: c.sign())


@register_operator(name="square", category="math", business_category="elementwise_math", canonical="square", source="factor_dsl_polars")
class SquarePolars(SeriesOperator):
    """Polars 平方一元算子。"""

    metadata = OperatorMetadata(
        name="square", category="math", description="平方",
        examples=["square(x)"], param_names=["x"], return_type="series", tags=["math", "polars"],
    )
    _calculate_series = _unary(lambda c: c.pow(2))


@register_operator(name="clip", category="math", business_category="elementwise_math", canonical="clip", source="factor_dsl_polars")
class ClipPolars(SeriesOperator):
    """Polars 区间裁剪一元算子。"""

    metadata = OperatorMetadata(
        name="clip", category="math", description="裁剪到 [lo, hi]",
        examples=["clip(x, -3, 3)"], param_names=["x", "lo", "hi"], return_type="series", tags=["math", "polars"],
        # R19-050: legacy ``min``/``max`` alias spellings declared explicitly.
        param_aliases={"min": "lo", "max": "hi"},
        param_specs={
            "lo": ParamSpec(dtype=float, min=None, max=None, default=-3.0, searchable=True,
                            param_role=ParamRole.ECONOMIC),
            "hi": ParamSpec(dtype=float, min=None, max=None, default=3.0, searchable=True,
                            param_role=ParamRole.ECONOMIC),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, lo: float = -3.0, hi: float = 3.0, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.common.strict_params import strict_float

        lo_v = strict_float(kwargs.get("min", lo), "lo")
        hi_v = strict_float(kwargs.get("max", hi), "hi")
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).clip(lo_v, hi_v).alias(c) for c in cols])


@register_operator(name="where", category="signal", business_category="technical_signal", canonical="where", source="factor_dsl_polars")
class WherePolars(SeriesOperator):
    """Polars 条件选择（if-else）算子。"""

    metadata = OperatorMetadata(
        name="where", category="signal", description="条件选择",
        examples=["where(cond, x, y)"], param_names=["condition", "x", "y"], return_type="series",
        tags=["signal", "polars"],
    )

    def _calculate_series(
        self,
        condition: pl.DataFrame,
        x: pl.DataFrame,
        y: pl.DataFrame,
        **kwargs,
    ) -> pl.DataFrame:
        # R5-04: unknown condition -> null output (never the false branch).  The
        # old ``cast(pl.Boolean, strict=False)`` mapped NaN/null to False, so a
        # missing condition silently returned ``b`` — the exact drift pandas'
        # ``where`` fixed.  Mirror the pandas contract: condition NaN/null -> NaN
        # output, condition != 0 -> x, condition == 0 -> y.
        cols = _align_cols(condition, x, y)
        return condition.select([
            pl.when(
                pl.col(c).cast(pl.Float64, strict=False).is_not_null()
                & pl.col(c).cast(pl.Float64, strict=False).is_finite()
            )
            .then(
                pl.when(pl.col(c).cast(pl.Float64, strict=False) != 0.0)
                .then(x[c])
                .otherwise(y[c])
            )
            .otherwise(None)
            .alias(c)
            for c in cols
        ])


@register_operator(name="ts_pct", category="time_series", business_category="time_series", canonical="ts_pct", source="factor_dsl_polars")
class TSPctPolars(SeriesOperator):
    """Polars d 期变化率算子。"""

    metadata = OperatorMetadata(
        name="ts_pct", category="time_series", description="d 期变化率",
        examples=["ts_pct(close, 1)"], param_names=["x", "d"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 1, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        periods = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            previous = pl.col(c).shift(periods)
            result = pl.col(c) / previous - 1.0
            exprs.append(pl.when(previous.is_not_null() & (previous != 0))
                         .then(result).otherwise(None).alias(c))
        return x.with_columns(exprs)


@register_operator(name="log_returns", category="financial", business_category="price_volume", canonical="log_returns", source="factor_dsl_polars")
class LogReturnsPolars(SeriesOperator):
    """Polars 对数收益率算子。"""

    metadata = OperatorMetadata(
        name="log_returns", category="financial", description="对数收益率",
        examples=["log_returns(close)"], param_names=["x"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([
            (pl.col(c) / pl.col(c).shift(1)).log().alias(c) for c in cols
        ])


@register_operator(name="ts_argmax", category="time_series", business_category="time_series", canonical="ts_argmax", source="factor_dsl_polars")
class TSArgmaxPolars(SeriesOperator):
    """Polars 滚动窗口最大值距当前 bar 的 bar 数（age，0=当前/最新 bar，并列取最新）。

    R19-039..042/049: ``ts_argmax`` canonical 语义为 **age**（0=当前），与 pandas
    backend / 共享 kernel ``rolling_days_since_extreme`` 一致。canonical 参数
    ``window``，``d`` 为别名。"""

    metadata = OperatorMetadata(
        name="ts_argmax", category="time_series", description="窗口最大值距当前 bar 的 bar 数（0=当前，并列取最近）",
        examples=["ts_argmax(close, 20)"], param_names=["x", "window", "min_periods"], return_type="series", tags=["time_series", "polars"],
        param_aliases={"d": "window"},
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True,
                                param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, min_periods: int = 1, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.overhaul.daily import pd_argmax
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        w = strict_int(kwargs.get("d", window), "window", minimum=1)
        return panel_pandas_bridge(
            x, lambda pdf: pd_argmax(pdf, w, min_periods=min_periods)
        )


@register_operator(name="ts_argmin", category="time_series", business_category="time_series", canonical="ts_argmin", source="factor_dsl_polars")
class TSArgminPolars(SeriesOperator):
    """Polars 滚动窗口最小值距当前 bar 的 bar 数（age，0=当前/最新 bar，并列取最新）。

    R19-039..042/049: ``ts_argmin`` canonical 语义为 **age**（0=当前），并列取
    最新 occurrence；需要 "0=窗口最旧 bar" 的 index 语义请用
    ``ts_argmin_index_from_oldest``。canonical 参数 ``window``，``d`` 为别名。"""

    metadata = OperatorMetadata(
        name="ts_argmin", category="time_series", description="窗口最小值距当前 bar 的 bar 数（0=当前，并列取最近）",
        examples=["ts_argmin(close, 20)"], param_names=["x", "window", "min_periods"], return_type="series", tags=["time_series", "polars"],
        param_aliases={"d": "window"},
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True,
                                param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self,
        x: pl.DataFrame,
        window: int = 20,
        min_periods: int = 1,
        **kwargs,
    ) -> pl.DataFrame:
        from factor_engine.cleaned_operators.overhaul.daily import pd_argmin
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        # ``d`` is a declared parser-level alias for ``window`` (param_aliases).
        w = strict_int(kwargs.get("d", window), "window", minimum=1)
        return panel_pandas_bridge(
            x, lambda pdf: pd_argmin(pdf, w, min_periods=min_periods)
        )


@register_operator(name="ts_quantile", category="time_series", business_category="time_series", canonical="ts_quantile", source="factor_dsl_polars")
class TSQuantilePolars(SeriesOperator):
    """Polars 滚动分位数算子。"""

    metadata = OperatorMetadata(
        name="ts_quantile", category="time_series", description="滚动分位数",
        examples=["ts_quantile(returns, 20, 0.75)"], param_names=["x", "d", "q"], return_type="series",
        tags=["time_series", "polars"],
        # R19-050..052: hidden window/p aliases declared explicitly; q in [0,1].
        param_aliases={"window": "d", "p": "q"},
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True,
                           param_role=ParamRole.HORIZON),
            "q": ParamSpec(dtype=float, min=0, max=1, default=0.5, searchable=True,
                           param_role=ParamRole.STATE_THRESHOLD),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 20, q: float = 0.5, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.common.strict_params import strict_int, strict_probability

        w = strict_int(kwargs.get("window", d), "d", minimum=1)
        quantile = strict_probability(kwargs.get("p", q), "q")
        cols = _numeric_cols(x)
        # pandas rolling().quantile() 跳过 NaN；polars rolling_quantile 会把 NaN
        # 当作最大参与排序。先 fill_nan(None) 对齐 pandas 缺失语义。
        return x.with_columns([
            pl.when(pl.col(c).is_nan() | pl.col(c).is_infinite()).then(None).otherwise(pl.col(c))
            .rolling_quantile(
                quantile=quantile,
                interpolation="linear",
                window_size=w,
                min_samples=1,
            ).alias(c)
            for c in cols
        ])


# ---------------------------------------------------------------------------
# 时序 Top-N / 秩相关（pandas 内核桥接，保证与 pandas_numpy backend 一致）
# ---------------------------------------------------------------------------


@register_operator(name="rank_corr", category="time_series", business_category="time_series", canonical="rank_corr", source="factor_dsl_polars")
class RankCorrPolars(SeriesOperator):
    """Polars 秩相关系数算子（时序窗口，d>0）。"""

    metadata = OperatorMetadata(
        name="rank_corr", category="time_series", description="滚动窗口秩相关系数（d>0 时序窗口）",
        examples=["rank_corr(close, volume, 20)"], param_names=["x", "y", "d"],
        return_type="series", tags=["time_series", "polars"],
        # R19-016..018: d is the trailing window (>= 1); the old d=0 full-sample
        # cross-section broadcast was removed (use cs_rank_corr for that).
        param_aliases={"window": "d"},
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True,
                           param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators._numpy_kernels import ts_rank_corr_
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        w = strict_int(kwargs.get("window", d), "d", minimum=1)
        cols = _align_cols(x, y)
        out: dict[str, np.ndarray] = {}
        for c in cols:
            out[c] = ts_rank_corr_(x[c].to_numpy(), y[c].to_numpy(), w)
        result = pl.DataFrame(out)
        if "date" in x.columns:
            result = result.with_columns(x["date"])
        return result


@register_operator(name="m_top_n_avg", category="time_series", business_category="time_series", canonical="ts_top_n_avg", source="factor_dsl_polars")
class TSTopNAvgPolars(SeriesOperator):
    """Polars 滚动前 N 大均值算子。"""

    metadata = OperatorMetadata(
        name="m_top_n_avg", category="time_series", description="滚动前 N 大均值",
        param_names=["x", "n"], return_type="series", tags=["time_series", "polars"],
        # R19-050: hidden ``d`` alias declared explicitly.
        param_aliases={"d": "n"},
        param_specs={
            "n": ParamSpec(dtype=int, min=1, default=5, searchable=True,
                           param_role=ParamRole.ECONOMIC),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, n: int = 5, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators._rolling_fast import rolling_top_n_mean
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        top_n = strict_int(kwargs.get("d", n), "n", minimum=1)
        cols = _numeric_cols(x)
        pdf = x.select(cols).to_pandas()
        out = rolling_top_n_mean(pdf, top_n)
        return x.with_columns([pl.Series(name=c, values=out[c].to_numpy()) for c in cols])


@register_operator(name="m_top_n_std", category="time_series", business_category="time_series", canonical="ts_top_n_std", source="factor_dsl_polars")
class TSTopNStdPolars(SeriesOperator):
    """Polars 滚动前 N 大标准差算子。"""

    metadata = OperatorMetadata(
        name="m_top_n_std", category="time_series", description="滚动前 N 大标准差",
        param_names=["x", "n"], return_type="series", tags=["time_series", "polars"],
        param_aliases={"d": "n"},
        param_specs={
            "n": ParamSpec(dtype=int, min=2, default=5, searchable=True,
                           param_role=ParamRole.ECONOMIC),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, n: int = 5, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators._rolling_fast import rolling_top_n_std
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        top_n = strict_int(kwargs.get("d", n), "n", minimum=2)
        cols = _numeric_cols(x)
        pdf = x.select(cols).to_pandas()
        out = rolling_top_n_std(pdf, top_n)
        return x.with_columns([pl.Series(name=c, values=out[c].to_numpy()) for c in cols])


@register_operator(name="ts_topk_sum", category="time_series", business_category="time_series", canonical="ts_topk_sum", source="factor_dsl_polars")
class TSTopKSumPolars(SeriesOperator):
    """Polars 滚动 Top-K 求和算子。"""

    metadata = OperatorMetadata(
        name="ts_topk_sum", category="time_series", description="滚动 Top-K 求和",
        param_names=["x", "d", "k", "min_periods"], return_type="series", tags=["time_series", "polars"],
        # R19-050: hidden window/n aliases declared explicitly.
        param_aliases={"window": "d", "n": "k"},
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True,
                           param_role=ParamRole.HORIZON),
            "k": ParamSpec(dtype=int, min=1, default=None, searchable=True,
                           param_role=ParamRole.ECONOMIC),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 20, k: int | None = None, min_periods: int | None = None, **kwargs) -> pl.DataFrame:
        from factor_engine.backend.operator_errors import OperatorParameterError
        from factor_engine.cleaned_operators._rolling_fast import rolling_top_n_sum_window
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        w = strict_int(kwargs.get("window", d), "d", minimum=1)
        k_eff = k if k is not None else kwargs.get("n", d)
        top_k = strict_int(k_eff, "k", minimum=1)
        if top_k > w:
            raise OperatorParameterError("k must be <= window")
        cols = _numeric_cols(x)
        pdf = x.select(cols).to_pandas()
        mp = w if min_periods is None else int(min_periods)
        out = rolling_top_n_sum_window(pdf, w, top_k)
        # min_periods gate: values with fewer than mp finite rows stay NaN.
        out = out.mask(pdf.notna().sum(axis=1) < mp) if mp < w else out
        return x.with_columns([pl.Series(name=c, values=out[c].to_numpy()) for c in cols])


@register_operator(name="cum_top_n_avg", category="time_series", business_category="time_series", canonical="cum_top_n_avg", source="factor_dsl_polars")
class CumTopNAvgPolars(SeriesOperator):
    """Polars 累积前 N 大均值算子。"""

    metadata = OperatorMetadata(
        name="cum_top_n_avg", category="time_series", description="累积前 N 大均值",
        param_names=["x", "n"], return_type="series", tags=["time_series", "polars"],
        param_aliases={"d": "n"},
        param_specs={
            "n": ParamSpec(dtype=int, min=1, default=5, searchable=True,
                           param_role=ParamRole.ECONOMIC),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, n: int = 5, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators._rolling_fast import cum_top_n_mean
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        top_n = strict_int(kwargs.get("d", n), "n", minimum=1)
        cols = _numeric_cols(x)
        pdf = x.select(cols).to_pandas()
        out = cum_top_n_mean(pdf, top_n)
        return x.with_columns([pl.Series(name=c, values=out[c].to_numpy()) for c in cols])


@register_operator(name="cum_top_n_sum", category="time_series", business_category="time_series", canonical="cum_top_n_sum", source="factor_dsl_polars")
class CumTopNSumPolars(SeriesOperator):
    """Polars 累积前 N 大求和算子。"""

    metadata = OperatorMetadata(
        name="cum_top_n_sum", category="time_series", description="累积前 N 大求和",
        param_names=["x", "n"], return_type="series", tags=["time_series", "polars"],
        param_aliases={"d": "n"},
        param_specs={
            "n": ParamSpec(dtype=int, min=1, default=5, searchable=True,
                           param_role=ParamRole.ECONOMIC),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, n: int = 5, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators._rolling_fast import cum_top_n_sum
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        top_n = strict_int(kwargs.get("d", n), "n", minimum=1)
        cols = _numeric_cols(x)
        pdf = x.select(cols).to_pandas()
        out = cum_top_n_sum(pdf, top_n)
        return x.with_columns([pl.Series(name=c, values=out[c].to_numpy()) for c in cols])


def _flex_compare(
    x: pl.DataFrame,
    y_or_window,
    *,
    compare_fn,
    rolling_fn,
) -> pl.DataFrame:
    """flex_max/flex_min 共用：逐点比较或滚动窗口极值。

参数:
    x: 第一个输入 Polars 宽表。
    y_or_window: 第二个宽表、标量或滚动窗口长度。
    compare_fn: 逐点比较函数（如 ``pl.max_horizontal``）。
    rolling_fn: 滚动极值函数工厂。

返回:
    比较或滚动极值后的 Polars DataFrame。

异常:
    TypeError: ``y_or_window`` 类型不合法时。
"""
    cols = _numeric_cols(x)
    if isinstance(y_or_window, pl.DataFrame):
        aligned = _align_cols(x, y_or_window)
        return x.with_columns(
            [compare_fn(pl.col(c), y_or_window[c]).alias(c) for c in aligned]
        )
    if isinstance(y_or_window, (int, float)) and not isinstance(y_or_window, bool):
        scalar = float(y_or_window)
        if isinstance(y_or_window, float) and scalar != int(scalar):
            return x.with_columns(
                [compare_fn(pl.col(c), pl.lit(scalar)).alias(c) for c in cols]
            )
        window = int(y_or_window)
        if window <= 0:
            return x.with_columns(
                [compare_fn(pl.col(c), pl.lit(scalar)).alias(c) for c in cols]
            )
        return x.with_columns(
            [rolling_fn(pl.col(c), window).alias(c) for c in cols]
        )
    raise TypeError(f"flex op expects series or scalar/window, got {type(y_or_window)}")


@register_operator(name="flex_max", category="math", business_category="elementwise_math", canonical="flex_max", source="factor_dsl_polars")
class FlexMaxPolars(SeriesOperator):
    """Polars 灵活最大值算子（逐点或滚动）。"""

    metadata = OperatorMetadata(
        name="flex_max", category="math", description="max(x,y) 或 rolling max",
        param_names=["x", "y_or_window"], return_type="series", tags=["math", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, y_or_window, **kwargs) -> pl.DataFrame:
        return _flex_compare(
            x,
            y_or_window,
            compare_fn=lambda a, b: pl.max_horizontal(a, b),
            rolling_fn=lambda c, w: c.rolling_max(window_size=w, min_samples=1),
        )


@register_operator(name="flex_min", category="math", business_category="elementwise_math", canonical="flex_min", source="factor_dsl_polars")
class FlexMinPolars(SeriesOperator):
    """Polars 灵活最小值算子（逐点或滚动）。"""

    metadata = OperatorMetadata(
        name="flex_min", category="math", description="min(x,y) 或 rolling min",
        param_names=["x", "y_or_window"], return_type="series", tags=["math", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, y_or_window, **kwargs) -> pl.DataFrame:
        return _flex_compare(
            x,
            y_or_window,
            compare_fn=lambda a, b: pl.min_horizontal(a, b),
            rolling_fn=lambda c, w: c.rolling_min(window_size=w, min_samples=1),
        )


def _expanding_pandas_bridge(x: pl.DataFrame, fn) -> pl.DataFrame:
    """扩展窗口 pandas 内核桥接：转 pandas 计算后写回 Polars。

参数:
    x: 输入 Polars 宽表 panel。
    fn: 接收 pandas 宽表返回同形结果的扩展统计函数。

返回:
    数值列替换为计算结果后的 Polars DataFrame。
"""
    cols = _numeric_cols(x)
    pdf = x.select(cols).to_pandas()
    out = fn(pdf)
    return x.with_columns([pl.Series(name=c, values=out[c].to_numpy()) for c in cols])


@register_operator(name="geometric_mean", category="math", business_category="elementwise_math", canonical="geometric_mean", source="factor_dsl_polars")
class GeometricMeanPolars(SeriesOperator):
    """Polars 扩展几何平均算子。"""

    metadata = OperatorMetadata(
        name="geometric_mean", category="math", description="扩展几何平均",
        param_names=["x"], return_type="series", tags=["math", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators._causal import expanding_geometric_mean

        return _expanding_pandas_bridge(x, expanding_geometric_mean)


@register_operator(name="harmonic_mean", category="math", business_category="elementwise_math", canonical="harmonic_mean", source="factor_dsl_polars")
class HarmonicMeanPolars(SeriesOperator):
    """Polars 扩展调和平均算子。"""

    metadata = OperatorMetadata(
        name="harmonic_mean", category="math", description="扩展调和平均",
        param_names=["x"], return_type="series", tags=["math", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators._causal import expanding_harmonic_mean

        return _expanding_pandas_bridge(x, expanding_harmonic_mean)


@register_operator(name="first_not_null", category="statistics", business_category="statistics_regression", canonical="first_not_null", source="factor_dsl_polars")
class FirstNotNullPolars(SeriesOperator):
    """Polars 扩展首个非空值算子。"""

    metadata = OperatorMetadata(
        name="first_not_null", category="statistics", description="扩展首个非空",
        param_names=["x"], return_type="series", tags=["statistics", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators._causal import expanding_first_not_null

        return _expanding_pandas_bridge(x, expanding_first_not_null)
