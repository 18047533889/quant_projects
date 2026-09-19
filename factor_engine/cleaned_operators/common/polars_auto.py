# -*- coding: utf-8 -*-
"""批量注册简单元素级算子的 Polars 实现（跳过已有 polars backend 的 canonical）。"""
from __future__ import annotations

from typing import Callable

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore


def _pl():
    if pl is None:
        raise RuntimeError(
            "polars is required for this operator; install factor-engine[polars]"
        )
    return pl


from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.base import Any, ParamSpec, ParamRole
from factor_engine.backend.contracts import (
    ExecutionKind, PhysicalImplementationSpec,
)

DYN_NOTE__register_compare = 'Genuine polars expression kernel (pl.col comparison + with_columns); runtime probe shows 0 pl.DataFrame.to_pandas calls. Declared to connect the polars_long channel (execution_kind was previously absent).'


_SKIP = frozenset({"date", "stock_code"})


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _has_polars(canonical: str) -> bool:
    return "polars" in OperatorRegistry.backends_for(canonical)


def _unary_calc(expr_fn: Callable):
    def calc(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([expr_fn(pl.col(c)).alias(c) for c in cols])

    return calc


def _finite_result(expr: pl.Expr) -> pl.Expr:
    """Match reference operators that replace non-finite results with NaN."""
    return pl.when(expr.is_finite()).then(expr).otherwise(None)


# ---------------------------------------------------------------------------
# R57 backend-coverage batch 3 — explicit execution-kind declarations.
# These kernels are genuine polars expressions (pl.col / with_columns /
# group_by over pl.Expr).  Previously they had no _physical_spec, so
# canonical_polars_kind(production_mode=True) failed closed to UNSUPPORTED.
# ---------------------------------------------------------------------------
_BATCH3_NOTE = (
    "Genuine polars expression kernel (pl.Expr over columns, no pandas round-trip); "
    "runtime marshal probe on real daily data records 0 pl.DataFrame.to_pandas "
    "calls. Eager panel API only: no lazy/streaming or production-parity claim."
)


def _batch3_native_spec(canonical: str, kernel: str) -> PhysicalImplementationSpec:
    """Explicit execution-kind contract for a genuine polars expression kernel.

    Built from the batch-3 evidence: the kernel body is ``pl.Expr`` construction
    (no pandas round-trip) and the runtime marshal probe records zero
    ``pl.DataFrame.to_pandas`` calls on real daily data.  Eager panel API, so
    ``supports_lazy`` / ``supports_streaming`` stay False.
    """
    return PhysicalImplementationSpec(
        canonical=canonical,
        backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
        materializes_full_panel=True,
        supports_nulls=True,
        supports_nan=True,
        supports_inf=True,
        implementation_source_hash=f"common.polars_auto:{kernel}:v1",
        emitter_identity=f"polars_expr:{canonical}",
        kernel_identity=f"common.polars_auto:{kernel}",
        parameter_domain_hash=f"{canonical}:declared:v1",
        semantic_contract_hash=f"{canonical}:polars_native_expr:v1",
        notes=_BATCH3_NOTE,
    )


_NATIVE_UNARY_SPEC_CANONICALS = frozenset(['log_abs', 'sigmoid'])


def _register_unary(
    canonical: str,
    *,
    name: str | None = None,
    description: str = "",
    expr_fn: Callable,
    category: str = "math",
    business_category: str = "elementwise_math",
) -> None:
    if _has_polars(canonical):
        return
    op_name = name or canonical
    desc = description or f"Polars {canonical}"

    class _UnaryPolars(SeriesOperator):
        metadata = OperatorMetadata(
            name=op_name,
            category=category,
            description=desc,
            examples=[f"{op_name}(x)"],
            param_names=(["x", "decimals"] if canonical == "round" else ["x"]),
            return_type="series",
            tags=[category, "polars", "auto"],
        )
        if canonical in _NATIVE_UNARY_SPEC_CANONICALS:
            _physical_spec = _batch3_native_spec(
                canonical, f"{canonical.title().replace('_', '')}PolarsAuto"
            )
        _calculate_series = _unary_calc(expr_fn)

    _UnaryPolars.__doc__ = desc

    _UnaryPolars.__name__ = f"{canonical.title().replace('_', '')}PolarsAuto"
    register_operator(
        name=op_name,
        category=category,
        business_category=business_category,
        canonical=canonical,
        source="factor_dsl_polars_auto",
    )(_UnaryPolars)


def _register_binary_horizontal(
    canonical: str,
    *,
    name: str | None = None,
    description: str = "",
    combine,
    category: str = "math",
    business_category: str = "elementwise_math",
) -> None:
    if _has_polars(canonical):
        return
    op_name = name or canonical

    class _BinaryPolars(SeriesOperator):
        metadata = OperatorMetadata(
            name=op_name,
            category=category,
            description=description or f"Polars {canonical}",
            examples=[f"{op_name}(x, y)"],
            param_names=["x", "y"],
            return_type="series",
            tags=[category, "polars", "auto"],
        )

        def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, **kwargs) -> pl.DataFrame:
            cols = [c for c in _numeric_cols(x) if c in y.columns]
            return x.with_columns([combine(pl.col(c), y[c]).alias(c) for c in cols])

    _BinaryPolars.__doc__ = description or f"Polars {canonical}"

    _BinaryPolars.__name__ = f"{canonical.title().replace('_', '')}PolarsAuto"
    register_operator(
        name=op_name,
        category=category,
        business_category=business_category,
        canonical=canonical,
        source="factor_dsl_polars_auto",
    )(_BinaryPolars)


# --- 一元映射（canonical → pl.Expr 变换）---
_UNARY: list[tuple[str, str, Callable]] = [
    ("acos", "acos", lambda c: c.arccos()),
    ("asin", "asin", lambda c: c.arcsin()),
    ("atan", "atan", lambda c: c.arctan()),
    ("cbrt", "cbrt", lambda c: c.pow(1.0 / 3.0)),
    ("ceil", "ceil", lambda c: c.ceil()),
    ("floor", "floor", lambda c: c.floor()),
    ("round", "round", lambda c: c.round()),
    ("fix", "fix", lambda c: c.truncate()),
    ("truncate", "truncate", lambda c: c.truncate()),
    ("cosh", "cosh", lambda c: c.cosh()),
    ("sinh", "sinh", lambda c: c.sinh()),
    ("tan", "tan", lambda c: c.tan()),
    ("log2", "log2", lambda c: c.log(2.0)),
    ("log_abs", "log_abs", lambda c: _finite_result(c.abs().log())),
    ("exp_neg", "exp_neg", lambda c: _finite_result((-c).exp())),
    ("reciprocal", "reciprocal", lambda c: 1.0 / c),
    ("cube", "cube", lambda c: c.pow(3)),
    ("identity", "identity", lambda c: c),
    ("negate", "negate", lambda c: -c),
    ("sec", "sec", lambda c: 1.0 / c.cos()),
    ("csc", "csc", lambda c: 1.0 / c.sin()),
    ("cot", "cot", lambda c: 1.0 / c.tan()),
    ("cumulative_max", "cumulative_max", lambda c: c.cum_max()),
    ("cumulative_min", "cumulative_min", lambda c: c.cum_min()),
    ("running_sum", "running_sum", lambda c: c.cum_sum()),
]

for _canon, _name, _fn in _UNARY:
    _register_unary(_canon, name=_name, expr_fn=_fn)


def _expanding_mean_col(c: pl.Expr) -> pl.Expr:
    cnt = c.is_not_null().cast(pl.Float64).cum_sum()
    return c.cum_sum() / cnt


def _expanding_std_col(c: pl.Expr) -> pl.Expr:
    cnt = c.is_not_null().cast(pl.Float64).cum_sum()
    mu = c.cum_sum() / cnt
    mean_sq = c.pow(2).cum_sum() / cnt
    return (mean_sq - mu.pow(2)).sqrt()


for _canon, _name in (
    ("cumulative_mean", "cumulative_mean"),
    ("running_mean", "running_mean"),
):
    if not _has_polars(_canon):

        class _ExpandingMeanPolars(SeriesOperator):
            metadata = OperatorMetadata(
                name=_name,
                category="math",
                description=f"Polars {_canon}",
                param_names=["x"],
                return_type="series",
                tags=["math", "polars", "auto"],
            )

            def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
                cols = _numeric_cols(x)
                return x.with_columns([_expanding_mean_col(pl.col(c)).alias(c) for c in cols])

        _ExpandingMeanPolars.__doc__ = f"Polars {_canon}"

        _ExpandingMeanPolars.__name__ = f"{_canon.title()}PolarsAuto"
        register_operator(
            name=_name,
            category="math",
            business_category="elementwise_math",
            canonical=_canon,
            source="factor_dsl_polars_auto",
        )(_ExpandingMeanPolars)

if not _has_polars("running_std"):

    class RunningStdPolarsAuto(SeriesOperator):
        """Polars running_std"""
        metadata = OperatorMetadata(
            name="running_std",
            category="math",
            description="Polars running_std",
            param_names=["x"],
            return_type="series",
            tags=["math", "polars", "auto"],
        )

        def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
            cols = _numeric_cols(x)
            return x.with_columns([_expanding_std_col(pl.col(c)).alias(c) for c in cols])

    register_operator(
        name="running_std",
        category="math",
        business_category="elementwise_math",
        canonical="running_std",
        source="factor_dsl_polars_auto",
    )(RunningStdPolarsAuto)

_register_binary_horizontal(
    "fmax",
    name="fmax",
    description="逐元素 fmax",
    combine=lambda *args, **kwargs: _pl().max_horizontal(*args, **kwargs),
)
_register_binary_horizontal(
    "fmin",
    name="fmin",
    description="逐元素 fmin",
    combine=lambda *args, **kwargs: _pl().min_horizontal(*args, **kwargs),
)


def _register_cleaning(canonical: str, name: str, calc_fn) -> None:
    # R30: tombstoned operators (dropna/bfill/…) have no runtime implementation
    # and no Polars surface either.  ``_has_polars`` would call ``backends_for``
    # which asserts the name is still callable and raise RemovedOperatorError.
    from factor_engine.cleaned_operators.tombstones import is_tombstoned

    if is_tombstoned(canonical):
        return
    if _has_polars(canonical):
        return

    class _CleaningPolars(SeriesOperator):
        metadata = OperatorMetadata(
            name=name,
            category="data_handling",
            description=f"Polars {canonical}",
            param_names=["x"],
            return_type="series",
            tags=["data_handling", "polars", "auto"],
        )
        _calculate_series = calc_fn

    _CleaningPolars.__doc__ = f"Polars {canonical}"

    _CleaningPolars.__name__ = f"{canonical.title()}PolarsAuto"
    register_operator(
        name=name,
        category="data_handling",
        business_category="data_cleaning",
        canonical=canonical,
        source="factor_dsl_polars_auto",
    )(_CleaningPolars)


# R13 P1-12: fillna method spellings that are canonical forward-fill aliases.
# Both the polars ``_ffill`` and ``FillNaPolarsAuto(method="ffill")`` land on the
# same gated polars forward-fill path below.
_FFILL_METHOD_ALIASES = frozenset({"ffill", "pad", "forward_fill"})


def _ffill(self, x, **kwargs) -> pl.DataFrame:
    # Mirror the pandas ``ffill`` gate (data_cleaning.FillForward): a field that
    # does not allow forward fill stays missing (fail-closed) and ``max_ffill_gap``
    # bounds the carry, so the polars path cannot bypass the pandas gate.
    if not bool(kwargs.get("forward_fill_allowed", True)):
        return x.clone()
    gap = int(kwargs.get("max_ffill_gap", 0))
    cols = _numeric_cols(x)
    if gap > 0:
        return x.with_columns([pl.col(c).forward_fill(limit=gap).alias(c) for c in cols])
    return x.with_columns([pl.col(c).forward_fill().alias(c) for c in cols])


def _dropna(self, x, **kwargs) -> pl.DataFrame:
    cols = _numeric_cols(x)
    return x.with_columns([pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c) for c in cols])


_register_cleaning("ffill", "ffill", _ffill)
_register_cleaning("dropna", "dropna", _dropna)

if not _has_polars("coalesce"):

    class CoalescePolarsAuto(SeriesOperator):
        """Polars coalesce"""
        metadata = OperatorMetadata(
            name="coalesce",
            category="elementwise_math",
            description="Polars coalesce",
            param_names=["x", "y"],
            return_type="series",
            tags=["elementwise", "polars", "auto"],
        )

        def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, **kwargs) -> pl.DataFrame:
            cols = [c for c in _numeric_cols(x) if c in y.columns]
            return x.with_columns([
                pl.coalesce([pl.col(c), y[c]]).alias(c) for c in cols
            ])

    register_operator(
        name="coalesce",
        category="elementwise_math",
        business_category="elementwise_math",
        canonical="coalesce",
        source="factor_dsl_polars_auto",
    )(CoalescePolarsAuto)

_register_unary("sigmoid", name="sigmoid", description="Sigmoid", expr_fn=lambda c: 1.0 / (1.0 + (-c).exp()))


def _truthy_series(s: pl.Series) -> pl.Series:
    """与 pandas ``_truthy_series`` 一致：非空、非 NaN、非零为真。

    不能直接 ``cast(Boolean)``——polars 把 NaN cast 成 True，而 pandas
    ``_truthy_series`` 把 NaN 当 False（缺失/NaN/0 → 假）。
    """
    return s.is_not_null() & (s != 0) & s.cast(pl.Float64, strict=False).is_not_nan()


if not _has_polars("and_"):

    class AndPolarsAuto(SeriesOperator):
        """逻辑与"""
        metadata = OperatorMetadata(
            name="and_", category="elementwise_math", description="逻辑与",
            param_names=["x", "y"], return_type="series", tags=["elementwise", "polars"],
        )

        def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, **kwargs) -> pl.DataFrame:
            cols = [c for c in _numeric_cols(x) if c in y.columns]
            replacements = []
            for c in cols:
                xb = _truthy_series(x[c])
                yb = _truthy_series(y[c])
                replacements.append((xb & yb).cast(pl.Float64).alias(c))
            return x.with_columns(replacements)
        _physical_spec = PhysicalImplementationSpec(
            canonical="and_", backend="polars",
            execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
            materializes_full_panel=True,
            supports_nulls=True, supports_nan=True, supports_inf=True,
            implementation_source_hash="common.polars_auto:AndPolarsAuto:v1",
            emitter_identity="polars_expr:and_",
            kernel_identity="common.polars_auto:AndPolarsAuto",
            parameter_domain_hash="and_:declared:v1",
            semantic_contract_hash="and_:polars_native_expr:v1",
            notes=('Genuine polars expression kernel (pl.col/with_columns); runtime probe shows 0 pl.DataFrame.to_pandas calls and the kernel body has no pandas/NumPy term. Declared to connect the polars_long channel: execution_kind was previously absent, so canonical_polars_kind(production_mode=True) reported UNSUPPORTED.'),
        )

    register_operator(
        name="and_", category="elementwise_math", business_category="elementwise_math",
        canonical="and_", source="factor_dsl_polars_auto",
    )(AndPolarsAuto)

if not _has_polars("or_"):

    class OrPolarsAuto(SeriesOperator):
        """逻辑或"""
        metadata = OperatorMetadata(
            name="or_", category="elementwise_math", description="逻辑或",
            param_names=["x", "y"], return_type="series", tags=["elementwise", "polars"],
        )

        def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, **kwargs) -> pl.DataFrame:
            cols = [c for c in _numeric_cols(x) if c in y.columns]
            replacements = []
            for c in cols:
                xb = _truthy_series(x[c])
                yb = _truthy_series(y[c])
                replacements.append((xb | yb).cast(pl.Float64).alias(c))
            return x.with_columns(replacements)

    register_operator(
        name="or_", category="elementwise_math", business_category="elementwise_math",
        canonical="or_", source="factor_dsl_polars_auto",
    )(OrPolarsAuto)

def _not_expr(c: pl.Expr) -> pl.Expr:
    """逻辑非：非真即真（缺失/NaN/0 → True），与 pandas ``~_truthy_series`` 一致。"""
    truthy = c.is_not_null() & (c != 0) & c.cast(pl.Float64, strict=False).is_not_nan()
    return (~truthy).cast(pl.Float64)


_register_unary("not_", name="not_", description="逻辑非", expr_fn=_not_expr)

if not _has_polars("fillna"):

    class FillNaPolarsAuto(SeriesOperator):
        """NaN 填充"""
        metadata = OperatorMetadata(
            name="fillna", category="data_handling", description="NaN 填充",
            param_names=["x", "method", "forward_fill_allowed", "max_ffill_gap"], return_type="series",
            param_specs={
                "method": ParamSpec(
                    alternatives=(ParamSpec(dtype=str), ParamSpec(dtype=float)),
                    default="zero", param_role=ParamRole.THRESHOLD,
                ),
                "forward_fill_allowed": ParamSpec(dtype=bool, default=True, searchable=False, param_role=ParamRole.MISSING_POLICY),
                "max_ffill_gap": ParamSpec(dtype=int, min=0, default=0, searchable=False, param_role=ParamRole.MISSING_POLICY),
            },
            tags=["data_handling", "polars"],
        )

        def _calculate_series(self, x: pl.DataFrame, method="zero", forward_fill_allowed=True, max_ffill_gap=0, **kwargs) -> pl.DataFrame:
            if isinstance(method, str) and method.strip().lower() in _FFILL_METHOD_ALIASES:
                # R13 P1-12: canonical rewrite onto the gated polars ffill so
                # ``fillna(x, method="ffill")`` cannot bypass the gate.
                return _ffill(self, x, forward_fill_allowed=forward_fill_allowed, max_ffill_gap=max_ffill_gap)
            cols = _numeric_cols(x)
            if isinstance(method, (int, float)) and not isinstance(method, bool):
                fill = float(method)
                return x.with_columns([
                    pl.col(c).fill_nan(fill).fill_null(fill).alias(c) for c in cols
                ])
            kind = method.strip().lower() if isinstance(method, str) else method
            if kind == "zero":
                return x.with_columns([
                    pl.col(c).fill_nan(0.0).fill_null(0.0).alias(c) for c in cols
                ])
            finite = [pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None) for c in cols]
            if kind == "mean":
                row_fill = pl.mean_horizontal(finite)
            elif kind == "median":
                row_fill = pl.concat_list(finite).list.drop_nulls().list.median()
            elif kind == "bfill":
                raise ValueError("fillna(method='bfill') was removed because it is not point-in-time safe")
            else:
                raise ValueError(
                    f"unknown fillna method: {method!r} "
                    "(supported: 'mean', 'median', 'zero', 'ffill' or a constant value)"
                )
            return x.with_columns([
                pl.when(pl.col(c).is_null() | pl.col(c).is_nan())
                .then(row_fill).otherwise(pl.col(c)).alias(c) for c in cols
            ])

    register_operator(
        name="fillna", category="data_handling", business_category="data_cleaning",
        canonical="fillna", source="factor_dsl_polars_auto",
    )(FillNaPolarsAuto)

def _register_compare(canon: str, name: str, op_fn) -> None:
    """注册比较算子；始终覆盖以修复旧版 for-loop 闭包 late-binding。"""

    class _ComparePolars(SeriesOperator):
        metadata = OperatorMetadata(
            name=name,
            category="elementwise_math",
            description=f"Polars {canon}",
            param_names=["x", "y"],
            return_type="series",
            tags=["elementwise", "polars", "auto"],
        )

        def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, **kwargs) -> pl.DataFrame:
            cols = [c for c in _numeric_cols(x) if c in y.columns]
            return x.with_columns([op_fn(pl.col(c), y[c]).alias(c) for c in cols])
        _physical_spec = PhysicalImplementationSpec(
            canonical=canon, backend="polars",
            execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
            materializes_full_panel=True,
            supports_nulls=True, supports_nan=True, supports_inf=True,
            implementation_source_hash=f"common.polars_auto:{canon}:v1",
            emitter_identity=f"polars_expr:{canon}",
            kernel_identity=f"common.polars_auto:{canon}",
            parameter_domain_hash=f"{canon}:declared:v1",
            semantic_contract_hash=f"{canon}:polars_native_expr:v1",
            notes=(DYN_NOTE__register_compare),
        )

    _ComparePolars.__doc__ = f"Polars {canon}"

    _ComparePolars.__name__ = f"{canon.upper()}PolarsAuto"
    register_operator(
        name=name,
        category="elementwise_math",
        business_category="elementwise_math",
        canonical=canon,
        source="factor_dsl_polars_auto",
    )(_ComparePolars)


for _canon, _name, _op in (
    ("eq", "eq", lambda a, b: (a == b).cast(pl.Float64)),
    ("ne", "ne", lambda a, b: (a != b).cast(pl.Float64)),
    ("gt", "gt", lambda a, b: (a > b).cast(pl.Float64)),
    ("ge", "ge", lambda a, b: (a >= b).cast(pl.Float64)),
    ("lt", "lt", lambda a, b: (a < b).cast(pl.Float64)),
    ("le", "le", lambda a, b: (a <= b).cast(pl.Float64)),
):
    _register_compare(_canon, _name, _op)

if not _has_polars("atan2"):

    class Atan2PolarsAuto(SeriesOperator):
        """atan2"""
        metadata = OperatorMetadata(
            name="atan2", category="math", description="atan2",
            param_names=["y", "x"], return_type="series", tags=["math", "polars"],
        )

        def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
            cols = [c for c in _numeric_cols(y) if c in x.columns]
            return y.with_columns([pl.arctan2(y[c], x[c]).alias(c) for c in cols])

    register_operator(
        name="atan2", category="math", business_category="elementwise_math",
        canonical="atan2", source="factor_dsl_polars_auto",
    )(Atan2PolarsAuto)
