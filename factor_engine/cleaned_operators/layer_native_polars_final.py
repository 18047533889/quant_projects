# -*- coding: utf-8 -*-
"""Expression-native Polars replacements for audited rolling operators.

A registered Polars backend is not considered native when it converts the
whole panel to NumPy or calls a Python ``rolling_map`` callback.  This layer
replaces the high-usage cases with Polars expressions and removes misleading
registrations where a stable expression implementation is not yet available.
"""
from __future__ import annotations

from typing import Callable

from cleaned_operators.overhaul.base import (
    PolarsFunctionOperator,
    pl,
    pl_base_with,
    pl_cols,
    positive_int,
    window_params,
)
from cleaned_operators.registry import OperatorRegistry

_APPLIED = False
_SOURCE = "final_expression_native_polars"


def _finite(name: str):
    value = pl.col(name).cast(pl.Float64, strict=False)
    return pl.when(value.is_not_null() & value.is_finite()).then(value).otherwise(None)


def _rolling_list(expr, window: int):
    # Current observation is first, so list index is exactly distance from now.
    return pl.concat_list([expr.shift(lag) for lag in range(window)])


def _replace_columns(frame, builder: Callable[[str], object]):
    replacements = {}
    for column in pl_cols(frame):
        temp = pl.DataFrame({"v": frame[column]})
        replacements[column] = temp.select(builder("v").alias("out"))["out"]
    return pl_base_with(frame, replacements)


def pl_last_if(x, condition, window, **_):
    w = positive_int(window, "window")
    replacements = {}
    for column in [name for name in pl_cols(x) if name in condition.columns]:
        temp = pl.DataFrame({"x": x[column], "c": condition[column]})
        xv = pl.col("x").cast(pl.Float64, strict=False)
        cv = pl.col("c").cast(pl.Float64, strict=False)
        selected = pl.when(
            xv.is_not_null() & xv.is_finite() & cv.is_not_null() & cv.is_finite() & (cv != 0)
        ).then(xv).otherwise(None)
        values = _rolling_list(selected, w).list.drop_nulls()
        replacements[column] = temp.select(values.list.first().alias("out"))["out"]
    return pl_base_with(x, replacements)


def pl_true_streak(condition, **_):
    replacements = {}
    for column in pl_cols(condition):
        temp = pl.DataFrame({"c": condition[column]}).with_row_index("__row")
        value = pl.col("c").cast(pl.Float64, strict=False)
        truth = value.is_not_null() & value.is_finite() & (value != 0)
        row = pl.col("__row").cast(pl.Int64)
        last_false = pl.when(~truth).then(row).otherwise(None).forward_fill().fill_null(-1)
        result = pl.when(truth).then((row - last_false).cast(pl.Float64)).otherwise(0.0)
        replacements[column] = temp.select(result.alias("out"))["out"]
    return pl_base_with(condition, replacements)


def _argext(frame, window, min_periods, *, largest: bool):
    w, mp = window_params(window, min_periods, default_mp=1)

    def build(name):
        values = _rolling_list(_finite(name), w)
        count = values.list.drop_nulls().list.len()
        position = values.list.arg_max() if largest else values.list.arg_min()
        return pl.when(count >= mp).then(position.cast(pl.Float64)).otherwise(None)

    return _replace_columns(frame, build)


def pl_argmax(x, window, min_periods=1, **_):
    return _argext(x, window, min_periods, largest=True)


def pl_argmin(x, window, min_periods=1, **_):
    return _argext(x, window, min_periods, largest=False)


def _top_bottom(frame, window, k=None, min_periods=None, *, top: bool, statistic: str):
    w = positive_int(window, "window")
    count_k = w if k is None else positive_int(k, "k")
    if count_k > w:
        raise ValueError("k must not exceed window")
    mp = count_k if min_periods is None else positive_int(min_periods, "min_periods")
    if mp > w:
        raise ValueError("min_periods must not exceed window")
    required = max(count_k, mp)

    def build(name):
        values = _rolling_list(_finite(name), w).list.drop_nulls()
        ordered = values.list.sort()
        selected = ordered.list.tail(count_k) if top else ordered.list.head(count_k)
        if statistic == "mean":
            result = selected.list.mean()
        elif statistic == "sum":
            result = selected.list.sum()
        else:
            result = selected.list.std(ddof=1)
        enough = values.list.len() >= required
        if statistic == "std":
            enough = enough & (pl.lit(count_k) >= 2)
        return pl.when(enough).then(result.cast(pl.Float64)).otherwise(None)

    return _replace_columns(frame, build)


def _top_spec(top: bool, statistic: str):
    return lambda x, window, k=None, min_periods=None, **_: _top_bottom(
        x, window, k, min_periods, top=top, statistic=statistic
    )


def _register_or_replace(name: str, params: list[str], description: str, function) -> None:
    exists = "polars" in OperatorRegistry.backends_for(name)
    OperatorRegistry.register(
        PolarsFunctionOperator(name, "time_series", params, description, function),
        canonical=name,
        backend="polars",
        source=_SOURCE,
        status="production",
        backend_explicit=True,
        replace=exists,
        replacement_reason=(
            "replace NumPy/Python rolling callback with expression-native Polars"
            if exists else "add expression-native Polars implementation"
        ),
        semantic_version="3.0",
    )


def _remove_polars(name: str, reason: str) -> None:
    implementations = OperatorRegistry._operators.get(name)
    if not implementations or "polars" not in implementations:
        return
    implementations.pop("polars", None)
    catalog = OperatorRegistry._catalog.get(name)
    if catalog is not None:
        metadata = dict(catalog.get("backend_meta") or {})
        metadata.pop("polars", None)
        catalog["backend_meta"] = metadata
        catalog["backends"] = sorted(implementations)
        catalog["polars_downgrade_reason"] = reason


def install_final_native_polars() -> None:
    global _APPLIED
    if _APPLIED or pl is None:
        return

    _register_or_replace(
        "ts_last_if", ["x", "condition", "window"],
        "most recent finite value satisfying a finite condition", pl_last_if,
    )
    _register_or_replace(
        "ts_true_streak", ["condition"],
        "consecutive finite non-zero condition length", pl_true_streak,
    )
    _register_or_replace(
        "ts_argmax", ["x", "window", "min_periods"],
        "distance to most recent rolling maximum", pl_argmax,
    )
    _register_or_replace(
        "ts_argmin", ["x", "window", "min_periods"],
        "distance to most recent rolling minimum", pl_argmin,
    )

    for name, top, statistic in (
        ("ts_topk_mean", True, "mean"),
        ("ts_topk_sum", True, "sum"),
        ("ts_topk_std", True, "std"),
        ("ts_bottomk_mean", False, "mean"),
        ("ts_bottomk_sum", False, "sum"),
        ("ts_bottomk_std", False, "std"),
    ):
        _register_or_replace(
            name, ["x", "window", "k", "min_periods"],
            f"expression-native rolling {statistic} of {'largest' if top else 'smallest'} K values",
            _top_spec(top, statistic),
        )

    # These historical registrations execute Python/NumPy over the whole panel
    # or per rolling window.  A truthful Pandas fallback is preferable to a
    # misleading Polars production claim.
    for name, reason in {
        "period_lag": "revision-aware fiscal as-of state is not expression-native",
        "cs_rank_gaussian": "normal inverse CDF implementation converts the full panel to NumPy",
        "ts_tail_mean": "quantile-tail implementation uses a Python rolling callback",
        "ts_time_slope": "missing-aware time coordinates use a Python rolling callback",
    }.items():
        _remove_polars(name, reason)

    from cleaned_operators import operator_policy
    operator_policy.POLARS_PARITY_VERIFIED = frozenset(
        name for name in operator_policy.POLARS_PARITY_VERIFIED
        if "polars" in OperatorRegistry.backends_for(name)
    )
    operator_policy.POLARS_PRODUCTION_SAFE = frozenset(
        name for name in operator_policy.POLARS_PRODUCTION_SAFE
        if "polars" in OperatorRegistry.backends_for(name)
    )
    _APPLIED = True


__all__ = ["install_final_native_polars"]
