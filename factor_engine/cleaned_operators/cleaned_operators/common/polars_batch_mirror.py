# -*- coding: utf-8 -*-
"""批量 pandas→Polars 桥接算子（复用 pandas_numpy 语义，保证 auto 路径可跑）。"""
from __future__ import annotations

from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.common._polars_bridge import bridge_registry, colwise_numpy_kernel
from factor_engine.cleaned_operators.registry import OperatorRegistry

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore


def _has_polars(canonical: str) -> bool:
    return "polars" in OperatorRegistry.backends_for(canonical)


def _register_bridge(
    canonical: str,
    *,
    name: str | None = None,
    category: str = "math",
    business_category: str = "elementwise_math",
    param_names: list[str] | None = None,
) -> None:
    if _has_polars(canonical):
        return
    op_name = name or canonical
    params = param_names or ["x"]

    class _BridgeOp(SeriesOperator):
        metadata = OperatorMetadata(
            name=op_name,
            category=category,
            description=f"{canonical}（Polars 桥接）",
            param_names=params,
            return_type="series",
            tags=["polars", "bridge"],
        )

        def _calculate_series(self, x: pl.DataFrame, *args, **kwargs) -> pl.DataFrame:
            return bridge_registry(canonical, x, *args, **kwargs)

    _BridgeOp.__doc__ = f"{canonical}（Polars 桥接）"

    _BridgeOp.__name__ = f"{canonical.title().replace('_', '')}PolarsBridge"
    register_operator(
        name=op_name,
        category=category,
        business_category=business_category,
        canonical=canonical,
        source="factor_dsl_polars_bridge",
    )(_BridgeOp)


def _register_colwise(
    canonical: str,
    kernel_name: str,
    *,
    category: str = "time_series",
    business_category: str = "time_series",
    param_names: list[str] | None = None,
) -> None:
    if _has_polars(canonical):
        return
    params = param_names or ["x", "d"]

    class _ColwiseOp(SeriesOperator):
        metadata = OperatorMetadata(
            name=canonical,
            category=category,
            description=f"{canonical}（Polars colwise 核）",
            param_names=params,
            return_type="series",
            tags=["polars", "colwise"],
        )

        def _calculate_series(self, x: pl.DataFrame, d: int = 20, k: int = 3, **kwargs) -> pl.DataFrame:
            import factor_engine.cleaned_operators._numpy_kernels as kernels

            kernel = getattr(kernels, kernel_name)
            window = int(kwargs.get("d", kwargs.get("window", d)))
            k_val = int(kwargs.get("k", kwargs.get("order", k)))
            if canonical == "ts_moment":
                return colwise_numpy_kernel(x, kernel, d=window, k=k_val)
            extra = {key: val for key, val in kwargs.items() if key not in {"d", "window", "k", "order"}}
            return colwise_numpy_kernel(x, kernel, d=window, **extra)

    _ColwiseOp.__doc__ = f"{canonical}（Polars colwise 核）"

    _ColwiseOp.__name__ = f"{canonical.title().replace('_', '')}PolarsColwise"
    register_operator(
        name=canonical,
        category=category,
        business_category=business_category,
        canonical=canonical,
        source="factor_dsl_polars_bridge",
    )(_ColwiseOp)


# ---- 一元 / 简单桥接 ---------------------------------------------------------
for _canon in (
    "conj",
    "phase",
    "polar",
    "rank_transform",
    "rankavg_transform",
    "product",
    "nan_to_num",
    "protected_log",
    "signed_sqrt",
    "sqr",
    "sqrt_abs",
    "unitize",
    "tukey_transform",
    "van_der_waerden_transform",
    "std_agg",
    "sum_agg",
    "stdp",
    "varp",
    "sem",
    "weighted_mean",
    "winsorize_mean",
    "wsum",
    "ts_ratio",
):
    _register_bridge(_canon, param_names=["real", "imag"] if _canon == "complex" else None)

_register_bridge(
    "size_neutralize",
    category="group_neutralization",
    business_category="group_neutralization",
    param_names=["x", "market_cap"],
)
_register_bridge(
    "industry_size_neutralize",
    category="group_neutralization",
    business_category="group_neutralization",
    param_names=["x", "industry", "market_cap"],
)

_register_bridge(
    "protected_div",
    category="data_cleaning",
    business_category="data_cleaning",
    param_names=["x", "y"],
)
_register_bridge(
    "safe_div_null",
    category="data_cleaning",
    business_category="data_cleaning",
    param_names=["x", "y"],
)
_register_bridge(
    "wavg",
    category="statistics",
    business_category="statistics_regression",
    param_names=["x", "w"],
)
for _f in ("quarter", "ttm", "yoy"):
    _register_bridge(
        _f,
        category="fundamental",
        business_category="fundamental",
        param_names=["x", "fiscal_quarter"],
    )

# ---- colwise 单序列核 --------------------------------------------------------
_register_colwise("price_spread_deviation", "price_spread_deviation_")
_register_colwise("ts_max_buildup", "ts_max_buildup_")
_register_colwise("ts_moment", "ts_moment_", param_names=["x", "d", "k"])
_register_colwise("ts_poly2_coeff", "ts_poly2_coeff_")

# 多参或截面 Top-N：registry 桥接
for _canon in (
    "ts_poly2_resid",
    "tm_top_n_avg",
    "tm_top_n_sum",
    "ts_bottom_n_avg",
    "ts_bottom_n_sum",
):
    _register_bridge(_canon, category="time_series", business_category="time_series")

# aggr_top_n 是跨截面路由算子（非 time_series）：按 sort_col 取当日全截面前 N
# 标的再聚合（routing 状态为全局 per-day）。通用 ``_register_bridge`` 的
# ``_calculate_series(x, *args)`` 契约与 aggr_top_n 的字符串首参（aggr_func）
# 不兼容，故用专用 polars 后端（复用 pandas_numpy 内核，无效 aggr_func 由
# pandas 内核抛 ValueError —— 两条后端路径一致）。
@register_operator(
    name="aggr_top_n",
    category="cross_sectional",
    business_category="cross_sectional_routing",
    canonical="aggr_top_n",
    source="factor_dsl_polars_bridge",
    backend="polars",
)
class AggrTopNPolars(SeriesOperator):
    """Polars 跨截面 Top-N 路由聚合（pandas 内核 parity）。"""

    metadata = OperatorMetadata(
        name="aggr_top_n",
        category="cross_sectional",
        description="跨截面 Top-N 路由聚合（pandas 内核 parity）",
        param_names=["aggr_func", "x", "sort_col", "top", "asc"],
        return_type="series",
        tags=["cross_sectional", "routing", "top_n", "polars"],
    )

    def _calculate_series(
        self,
        aggr_func: str = "sum",
        x: pl.DataFrame | None = None,
        sort_col: pl.DataFrame | None = None,
        top: int = 10,
        asc: bool = True,
        **kwargs,
    ) -> pl.DataFrame:
        from factor_engine.cleaned_operators.common._polars_bridge import bridge_pandas

        if x is None:
            raise ValueError("aggr_top_n requires an x panel")
        # Use the pandas class directly (not the registry lookup): aggr_top_n is
        # governance-blocked and unregistered from the final registry after a
        # full load_all, but the owned implementation must still be invokable.
        from factor_engine.cleaned_operators.common.time_series import AggrTopN as _AggrTopN

        # bridge_pandas calls compute(pdf, *pargs); pargs = (aggr_func, sort_col,
        # top, asc) and pdf is the x panel.
        def _compute(pdf, aggr_func, sort_col, top, asc, **kwargs):
            return _AggrTopN().calculate(aggr_func, pdf, sort_col, top, asc, **kwargs)

        return bridge_pandas(x, _compute, aggr_func, sort_col, top, asc, **kwargs)

# 统计 / 回归 / 假设检验（桥接，生产仍推荐 Tier-1 算子）
for _canon in (
    "slope",
    "regress",
    "ridge",
    "lasso",
    "quantile_normal",
    "quantile_t",
    "unwrap",
    "correlate",
    "decimate",
    "ACF",
    "pacf",
    "stationarity_test",
    "spearman_corr_test",
    "kendall_corr_test",
    "ks_test",
    "jarque_bera_test",
    "granger_causality",
    "levene_test",
    "bartlett_test",
    "chi_square_test",
    "kpss_test",
    "lilliefors_test",
    "ttest_one_sample",
    "ttest_paired",
    "ttest_two_samples",
):
    _register_bridge(_canon, category="statistics", business_category="statistics_regression")


# ts_moment：覆盖 Colwise 注册，支持第三参数 k
@register_operator(
    name="ts_moment",
    category="time_series",
    business_category="time_series",
    canonical="ts_moment",
    source="factor_dsl_polars_bridge",
    backend="polars",
)
class TSMomentPolarsBridge(SeriesOperator):
    """窗口 k 阶中心矩（pandas 核 parity）"""
    metadata = OperatorMetadata(
        name="ts_moment",
        category="time_series",
        description="窗口 k 阶中心矩（pandas 核 parity）",
        param_names=["x", "d", "k"],
        return_type="series",
        tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 20, k: int = 3, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators._numpy_kernels import ts_moment_

        window = int(kwargs.get("d", kwargs.get("window", d)))
        k_val = int(kwargs.get("k", kwargs.get("order", k)))
        return colwise_numpy_kernel(x, ts_moment_, d=window, k=k_val)
