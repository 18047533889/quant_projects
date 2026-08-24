# -*- coding: utf-8 -*-
"""Final operator fixes from the 2026-07 full registry audit.

This module is intentionally loaded after ``layer_composite_fixes`` and before
the final cleanup/governance passes.  It replaces only proven-bad active
backend slots while preserving the reviewed canonical names and surfaces.
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any, Sequence

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.overhaul.base import (
    EPS,
    PandasFunctionOperator,
    PolarsFunctionOperator,
    aligned_pd,
    frame_pd,
    pl,
    pl_base_with,
    pl_cols,
    positive_int,
)
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _register(
    canonical: str,
    *,
    pandas_fn=None,
    polars_fn=None,
    pandas_source: str = "operator_overhaul_audited",
    polars_source: str = "operator_overhaul_native_polars",
) -> None:
    """Replace an active backend slot without changing canonical metadata."""
    catalog = OperatorRegistry._catalog.get(canonical, {})
    params = list(catalog.get("param_names") or [])
    description = str(catalog.get("description") or canonical)
    category = str(catalog.get("business_category") or catalog.get("scope") or "audited")
    if pandas_fn is not None:
        OperatorRegistry.register(
            PandasFunctionOperator(canonical, category, params, description, pandas_fn),
            canonical=canonical,
            backend="pandas_numpy",
            source=pandas_source,
            status=str(catalog.get("status") or "production"),
            backend_explicit=True,
        )
    if pl is not None and polars_fn is not None:
        OperatorRegistry.register(
            PolarsFunctionOperator(canonical, category, params, description, polars_fn),
            canonical=canonical,
            backend="polars",
            source=polars_source,
            status=str(catalog.get("status") or "production"),
            backend_explicit=True,
        )


# ---------------------------------------------------------------------------
# Fiscal-period parsing and lagging
# ---------------------------------------------------------------------------

_QUARTER_PATTERNS = (
    re.compile(r"^(\d{4})\s*Q\s*([1-4])$", re.IGNORECASE),
    re.compile(r"^(\d{4})0([1-4])$"),
    re.compile(r"^(\d{4})([1-4])$"),
)
_DATE_COMPACT = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
_DATE_SEPARATED = re.compile(r"^(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?")


def _quarter_ordinal(year: int, quarter: int) -> int | None:
    if year < 1000 or not 1 <= quarter <= 4:
        return None
    return year * 4 + quarter - 1


def _month_ordinal(year: int, month: int) -> int | None:
    if year < 1000 or not 1 <= month <= 12:
        return None
    return year * 4 + (month - 1) // 3


def period_ordinal_strict(value: Any) -> int | None:
    """Parse common fiscal-quarter/date encodings without numeric ambiguity."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64, datetime, date)):
        ts = pd.Timestamp(value)
        return None if pd.isna(ts) else _month_ordinal(int(ts.year), int(ts.month))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            return None
        value = int(value)
    if isinstance(value, (int, np.integer)):
        text = str(int(value))
    else:
        text = str(value).strip().upper()
        if not text or text in {"NAN", "NAT", "NONE", "<NA>"}:
            return None

    for pattern in _QUARTER_PATTERNS:
        match = pattern.fullmatch(text)
        if match:
            return _quarter_ordinal(int(match.group(1)), int(match.group(2)))

    match = _DATE_COMPACT.fullmatch(text)
    if match:
        return _month_ordinal(int(match.group(1)), int(match.group(2)))

    match = _DATE_SEPARATED.match(text)
    if match:
        return _month_ordinal(int(match.group(1)), int(match.group(2)))

    # Explicit ordinal IDs remain supported, but only after known calendar
    # encodings have been ruled out.
    try:
        number = int(text)
    except (TypeError, ValueError):
        return None
    return number if abs(number) < 100000 else None


def _ordinal_frame(period_id: pd.DataFrame) -> pd.DataFrame:
    return period_id.apply(lambda col: col.map(period_ordinal_strict)).astype("Float64")


def _pd_period_lag(
    x: pd.DataFrame,
    period_id: pd.DataFrame,
    periods: int = 1,
    revision_policy: str = "latest_available",
    **_: Any,
) -> pd.DataFrame:
    x, period_id = aligned_pd(x, period_id)
    lag = int(periods)
    if lag < 0 or float(periods) != float(lag):
        raise ValueError("periods must be a non-negative integer")
    policy = str(revision_policy).lower()
    if policy not in {"latest_available", "first_available"}:
        raise ValueError("revision_policy must be latest_available or first_available")

    values = x.to_numpy(dtype=float)
    periods_raw = period_id.to_numpy(dtype=object)
    out = np.full(values.shape, np.nan, dtype=float)
    for col in range(values.shape[1]):
        visible: dict[int, float] = {}
        for row in range(values.shape[0]):
            ordinal = period_ordinal_strict(periods_raw[row, col])
            if ordinal is None:
                continue
            value = values[row, col]
            if np.isfinite(value) and (
                policy == "latest_available" or ordinal not in visible
            ):
                visible[ordinal] = float(value)
            out[row, col] = visible.get(ordinal - lag, np.nan)
    return frame_pd(x, out)


def _pd_period_inputs(x, period_id, periods, revision_policy="latest_available"):
    x, period_id = aligned_pd(x, period_id)
    lag = positive_int(periods, "periods")
    ordinal = _ordinal_frame(period_id).astype(float)
    previous = _pd_period_lag(x, period_id, lag, revision_policy)
    previous_ordinal = _pd_period_lag(ordinal, period_id, lag, revision_policy)
    consecutive = ordinal.sub(previous_ordinal).eq(float(lag))
    return x, period_id, ordinal, previous, consecutive


def _pd_period_change(
    x,
    period_id,
    periods=1,
    mode="absolute",
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
):
    x, _, _, previous, consecutive = _pd_period_inputs(
        x, period_id, periods, revision_policy
    )
    valid = np.isfinite(x) & np.isfinite(previous)
    if bool(require_consecutive):
        valid &= consecutive
    mode = str(mode).lower()
    if mode == "absolute":
        result = x - previous
    elif mode == "ratio":
        result = x / previous.where(previous.abs() > EPS) - 1.0
    elif mode == "log":
        ratio = x / previous.where(previous.abs() > EPS)
        result = np.log(ratio.where(ratio > 0))
    else:
        raise ValueError("mode must be absolute, ratio, or log")
    return result.where(valid).replace([np.inf, -np.inf], np.nan)


def _pd_period_average(
    x,
    period_id,
    periods=2,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
):
    x, period_id = aligned_pd(x, period_id)
    count = positive_int(periods, "periods")
    ordinal = _ordinal_frame(period_id).astype(float)
    pieces = [x]
    valid = np.isfinite(x)
    for lag in range(1, count):
        piece = _pd_period_lag(x, period_id, lag, revision_policy)
        prior_ordinal = _pd_period_lag(ordinal, period_id, lag, revision_policy)
        pieces.append(piece)
        valid &= np.isfinite(piece)
        if bool(require_consecutive):
            valid &= ordinal.sub(prior_ordinal).eq(float(lag))
    total = pieces[0].copy()
    for piece in pieces[1:]:
        total = total + piece
    return (total / float(count)).where(valid)


def _pd_period_cagr(
    x,
    period_id,
    periods=12,
    periods_per_year=4,
    sign_policy="strict",
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
):
    x, _, _, previous, consecutive = _pd_period_inputs(
        x, period_id, periods, revision_policy
    )
    ppy = positive_int(periods_per_year, "periods_per_year")
    lag = positive_int(periods, "periods")
    valid = np.isfinite(x) & np.isfinite(previous)
    if bool(require_consecutive):
        valid &= consecutive
    policy = str(sign_policy).lower()
    if policy == "strict":
        valid &= (x > 0) & (previous > 0)
        ratio = x / previous
    elif policy == "absolute":
        valid &= previous.abs().gt(EPS)
        ratio = x.abs() / previous.abs()
    else:
        raise ValueError("sign_policy must be strict or absolute")
    result = ratio.pow(float(ppy) / float(lag)) - 1.0
    return result.where(valid).replace([np.inf, -np.inf], np.nan)


def _pd_quarter_from_cumulative(
    x,
    period_id,
    fiscal_quarter=None,
    revision_policy="latest_available",
    **_,
):
    x, period_id = aligned_pd(x, period_id)
    ordinal = _ordinal_frame(period_id).astype(float)
    previous = _pd_period_lag(x, period_id, 1, revision_policy)
    previous_ordinal = _pd_period_lag(ordinal, period_id, 1, revision_policy)
    if fiscal_quarter is None:
        quarter = ordinal.mod(4).add(1.0)
    else:
        _, fiscal_quarter = aligned_pd(x, fiscal_quarter)
        quarter = fiscal_quarter.astype(float)
    consecutive = ordinal.sub(previous_ordinal).eq(1.0)
    result = x.where(quarter.eq(1.0), x - previous.where(consecutive))
    valid = np.isfinite(x) & np.isfinite(quarter) & (
        quarter.eq(1.0) | consecutive
    )
    return result.where(valid).replace([np.inf, -np.inf], np.nan)


def _pd_ttm_from_quarterly(
    x,
    period_id,
    periods=4,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
):
    x, period_id = aligned_pd(x, period_id)
    count = positive_int(periods, "periods")
    ordinal = _ordinal_frame(period_id).astype(float)
    pieces = [x]
    valid = np.isfinite(x)
    for lag in range(1, count):
        piece = _pd_period_lag(x, period_id, lag, revision_policy)
        prior_ordinal = _pd_period_lag(ordinal, period_id, lag, revision_policy)
        pieces.append(piece)
        valid &= np.isfinite(piece)
        if bool(require_consecutive):
            valid &= ordinal.sub(prior_ordinal).eq(float(lag))
    total = pieces[0].copy()
    for piece in pieces[1:]:
        total = total + piece
    return total.where(valid).replace([np.inf, -np.inf], np.nan)


def _pd_ttm_from_cumulative(
    x,
    period_id,
    fiscal_quarter=None,
    revision_policy="latest_available",
    **_,
):
    quarterly = _pd_quarter_from_cumulative(
        x,
        period_id,
        fiscal_quarter,
        revision_policy=revision_policy,
    )
    return _pd_ttm_from_quarterly(
        quarterly,
        period_id,
        periods=4,
        require_consecutive=True,
        revision_policy=revision_policy,
    )


def _pd_yoy_by_period(
    x,
    period_id,
    periods=4,
    denominator="signed",
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
):
    x, period_id = aligned_pd(x, period_id)
    lag = positive_int(periods, "periods")
    previous = _pd_period_lag(x, period_id, lag, revision_policy)
    mode = str(denominator).lower()
    if mode == "signed":
        denom = previous
    elif mode == "absolute":
        denom = previous.abs()
    else:
        raise ValueError("denominator must be signed or absolute")
    valid = np.isfinite(x) & np.isfinite(denom) & denom.abs().gt(EPS)
    if bool(require_consecutive):
        ordinal = _ordinal_frame(period_id).astype(float)
        prior_ordinal = _pd_period_lag(ordinal, period_id, lag, revision_policy)
        valid &= ordinal.sub(prior_ordinal).eq(float(lag))
    return ((x - previous) / denom.where(valid)).where(valid).replace(
        [np.inf, -np.inf], np.nan
    )


if pl is not None:

    def _pl_period_ordinal_expr(column: str):
        text = (
            pl.col(column)
            .cast(pl.Utf8, strict=False)
            .str.strip_chars()
            .str.to_uppercase()
        )
        q_year = text.str.extract(r"^([0-9]{4})\s*Q\s*([1-4])$", 1).cast(
            pl.Int64, strict=False
        )
        q_num = text.str.extract(r"^([0-9]{4})\s*Q\s*([1-4])$", 2).cast(
            pl.Int64, strict=False
        )
        y0q_year = text.str.extract(r"^([0-9]{4})0([1-4])$", 1).cast(
            pl.Int64, strict=False
        )
        y0q_num = text.str.extract(r"^([0-9]{4})0([1-4])$", 2).cast(
            pl.Int64, strict=False
        )
        yq_year = text.str.extract(r"^([0-9]{4})([1-4])$", 1).cast(
            pl.Int64, strict=False
        )
        yq_num = text.str.extract(r"^([0-9]{4})([1-4])$", 2).cast(
            pl.Int64, strict=False
        )
        compact_year = text.str.extract(r"^([0-9]{4})([0-9]{2})([0-9]{2})$", 1).cast(
            pl.Int64, strict=False
        )
        compact_month = text.str.extract(
            r"^([0-9]{4})([0-9]{2})([0-9]{2})$", 2
        ).cast(pl.Int64, strict=False)
        sep_year = text.str.extract(
            r"^([0-9]{4})[-/]([0-9]{1,2})(?:[-/][0-9]{1,2})?", 1
        ).cast(pl.Int64, strict=False)
        sep_month = text.str.extract(
            r"^([0-9]{4})[-/]([0-9]{1,2})(?:[-/][0-9]{1,2})?", 2
        ).cast(pl.Int64, strict=False)

        quarter_ordinal = q_year * 4 + q_num - 1
        y0q_ordinal = y0q_year * 4 + y0q_num - 1
        yq_ordinal = yq_year * 4 + yq_num - 1
        compact_ordinal = (
            compact_year * 4 + ((compact_month - 1) / 3).floor().cast(pl.Int64)
        )
        sep_ordinal = sep_year * 4 + ((sep_month - 1) / 3).floor().cast(pl.Int64)
        raw = pl.col(column).cast(pl.Int64, strict=False)
        raw_valid = raw.abs() < 100000

        return (
            pl.when(q_year.is_not_null())
            .then(quarter_ordinal)
            .when(y0q_year.is_not_null())
            .then(y0q_ordinal)
            .when(yq_year.is_not_null())
            .then(yq_ordinal)
            .when(
                compact_year.is_not_null()
                & compact_month.is_between(1, 12, closed="both")
            )
            .then(compact_ordinal)
            .when(
                sep_year.is_not_null()
                & sep_month.is_between(1, 12, closed="both")
            )
            .then(sep_ordinal)
            .when(raw_valid)
            .then(raw)
            .otherwise(None)
            .cast(pl.Int64)
        )

    def _pl_lag_one(values, periods, lag: int, policy: str, name: str):
        temp = pl.DataFrame(
            {
                "_row": pl.int_range(0, len(values), eager=True),
                "_value": values.cast(pl.Float64, strict=False),
                "_period": periods,
            }
        ).with_columns(_pl_period_ordinal_expr("_period").alias("_ordinal"))
        targets = temp.select(
            "_row",
            (pl.col("_ordinal") - lag).alias("_target"),
        )
        history = temp.select(
            pl.col("_row").alias("_history_row"),
            pl.col("_ordinal").alias("_history_ordinal"),
            pl.col("_value").alias("_history_value"),
        )
        candidates = (
            targets.join(
                history,
                left_on="_target",
                right_on="_history_ordinal",
                how="inner",
            )
            .filter(
                (pl.col("_history_row") <= pl.col("_row"))
                & pl.col("_history_value").is_not_null()
                & pl.col("_history_value").is_finite()
            )
            .sort(["_row", "_history_row"])
        )
        aggregate = (
            pl.col("_history_value").first()
            if policy == "first_available"
            else pl.col("_history_value").last()
        )
        selected = candidates.group_by("_row", maintain_order=True).agg(
            aggregate.alias("_out")
        )
        return (
            temp.select("_row")
            .join(selected, on="_row", how="left", maintain_order="left")
            .select(pl.col("_out").alias(name))[name]
        )

    def _pl_period_lag(
        x,
        period_id,
        periods=1,
        revision_policy="latest_available",
        **_,
    ):
        lag = int(periods)
        if lag < 0 or float(periods) != float(lag):
            raise ValueError("periods must be a non-negative integer")
        policy = str(revision_policy).lower()
        if policy not in {"latest_available", "first_available"}:
            raise ValueError("revision_policy must be latest_available or first_available")
        replacements = {}
        for col in [c for c in pl_cols(x) if c in period_id.columns]:
            replacements[col] = _pl_lag_one(
                x[col], period_id[col], lag, policy, col
            )
        return pl_base_with(x, replacements)

    def _pl_ordinal_frame(x, period_id):
        replacements = {}
        for col in [c for c in pl_cols(x) if c in period_id.columns]:
            replacements[col] = (
                pl.DataFrame({"_period": period_id[col]})
                .select(_pl_period_ordinal_expr("_period").alias(col))[col]
            )
        return pl_base_with(x, replacements)

    def _pl_binary_valid(x, y, fn):
        replacements = {}
        for col in [c for c in pl_cols(x) if c in y.columns]:
            temp = pl.DataFrame(
                {
                    "_x": x[col].cast(pl.Float64, strict=False),
                    "_y": y[col].cast(pl.Float64, strict=False),
                }
            )
            valid = (
                pl.col("_x").is_not_null()
                & pl.col("_x").is_finite()
                & pl.col("_y").is_not_null()
                & pl.col("_y").is_finite()
            )
            replacements[col] = temp.select(
                pl.when(valid).then(fn(pl.col("_x"), pl.col("_y"))).otherwise(None).alias(col)
            )[col]
        return pl_base_with(x, replacements)

    def _pl_period_change(
        x,
        period_id,
        periods=1,
        mode="absolute",
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ):
        previous = _pl_period_lag(x, period_id, periods, revision_policy)
        mode = str(mode).lower()
        if mode == "absolute":
            return _pl_binary_valid(x, previous, lambda a, b: a - b)
        if mode == "ratio":
            return _pl_binary_valid(
                x,
                previous,
                lambda a, b: pl.when(b.abs() > EPS).then(a / b - 1.0),
            )
        if mode == "log":
            return _pl_binary_valid(
                x,
                previous,
                lambda a, b: pl.when((b.abs() > EPS) & (a / b > 0)).then(
                    (a / b).log()
                ),
            )
        raise ValueError("mode must be absolute, ratio, or log")

    def _pl_period_average(
        x,
        period_id,
        periods=2,
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ):
        count = positive_int(periods, "periods")
        pieces = [x] + [
            _pl_period_lag(x, period_id, lag, revision_policy)
            for lag in range(1, count)
        ]
        replacements = {}
        for col in pl_cols(x):
            temp = pl.DataFrame(
                {f"_v{i}": piece[col].cast(pl.Float64, strict=False) for i, piece in enumerate(pieces)}
            )
            valid = pl.all_horizontal(
                *[
                    pl.col(f"_v{i}").is_not_null()
                    & pl.col(f"_v{i}").is_finite()
                    for i in range(count)
                ]
            )
            total = pl.sum_horizontal(*[pl.col(f"_v{i}") for i in range(count)])
            replacements[col] = temp.select(
                pl.when(valid).then(total / float(count)).otherwise(None).alias(col)
            )[col]
        return pl_base_with(x, replacements)

    def _pl_period_cagr(
        x,
        period_id,
        periods=12,
        periods_per_year=4,
        sign_policy="strict",
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ):
        lag = positive_int(periods, "periods")
        exponent = positive_int(periods_per_year, "periods_per_year") / float(lag)
        previous = _pl_period_lag(x, period_id, lag, revision_policy)
        policy = str(sign_policy).lower()
        if policy == "strict":
            return _pl_binary_valid(
                x,
                previous,
                lambda a, b: pl.when((a > 0) & (b > 0)).then(
                    (a / b).pow(exponent) - 1.0
                ),
            )
        if policy == "absolute":
            return _pl_binary_valid(
                x,
                previous,
                lambda a, b: pl.when(b.abs() > EPS).then(
                    (a.abs() / b.abs()).pow(exponent) - 1.0
                ),
            )
        raise ValueError("sign_policy must be strict or absolute")

    def _pl_quarter_from_cumulative(
        x,
        period_id,
        fiscal_quarter=None,
        revision_policy="latest_available",
        **_,
    ):
        previous = _pl_period_lag(x, period_id, 1, revision_policy)
        ordinal = _pl_ordinal_frame(x, period_id)
        previous_ordinal = _pl_period_lag(
            ordinal, period_id, 1, revision_policy
        )
        replacements = {}
        for col in [c for c in pl_cols(x) if c in period_id.columns]:
            if fiscal_quarter is None:
                quarter = ordinal[col].cast(pl.Float64, strict=False) % 4 + 1
            else:
                quarter = fiscal_quarter[col].cast(pl.Float64, strict=False)
            temp = pl.DataFrame(
                {
                    "_x": x[col].cast(pl.Float64, strict=False),
                    "_previous": previous[col].cast(pl.Float64, strict=False),
                    "_ordinal": ordinal[col].cast(pl.Float64, strict=False),
                    "_previous_ordinal": previous_ordinal[col].cast(
                        pl.Float64, strict=False
                    ),
                    "_quarter": quarter,
                }
            )
            consecutive = pl.col("_ordinal") - pl.col("_previous_ordinal") == 1
            valid = (
                pl.col("_x").is_not_null()
                & pl.col("_x").is_finite()
                & pl.col("_quarter").is_not_null()
                & ((pl.col("_quarter") == 1) | consecutive)
            )
            value = (
                pl.when(pl.col("_quarter") == 1)
                .then(pl.col("_x"))
                .otherwise(pl.col("_x") - pl.col("_previous"))
            )
            replacements[col] = temp.select(
                pl.when(valid).then(value).otherwise(None).alias(col)
            )[col]
        return pl_base_with(x, replacements)

    def _pl_ttm_from_quarterly(
        x,
        period_id,
        periods=4,
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ):
        count = positive_int(periods, "periods")
        pieces = [x] + [
            _pl_period_lag(x, period_id, lag, revision_policy)
            for lag in range(1, count)
        ]
        replacements = {}
        for col in pl_cols(x):
            temp = pl.DataFrame(
                {f"_v{i}": piece[col].cast(pl.Float64, strict=False) for i, piece in enumerate(pieces)}
            )
            valid = pl.all_horizontal(
                *[
                    pl.col(f"_v{i}").is_not_null()
                    & pl.col(f"_v{i}").is_finite()
                    for i in range(count)
                ]
            )
            total = pl.sum_horizontal(*[pl.col(f"_v{i}") for i in range(count)])
            replacements[col] = temp.select(
                pl.when(valid).then(total).otherwise(None).alias(col)
            )[col]
        return pl_base_with(x, replacements)

    def _pl_ttm_from_cumulative(
        x,
        period_id,
        fiscal_quarter=None,
        revision_policy="latest_available",
        **_,
    ):
        quarterly = _pl_quarter_from_cumulative(
            x,
            period_id,
            fiscal_quarter,
            revision_policy=revision_policy,
        )
        return _pl_ttm_from_quarterly(
            quarterly,
            period_id,
            4,
            True,
            revision_policy,
        )

    def _pl_yoy_by_period(
        x,
        period_id,
        periods=4,
        denominator="signed",
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ):
        previous = _pl_period_lag(
            x, period_id, positive_int(periods, "periods"), revision_policy
        )
        mode = str(denominator).lower()
        if mode == "signed":
            return _pl_binary_valid(
                x,
                previous,
                lambda a, b: pl.when(b.abs() > EPS).then((a - b) / b),
            )
        if mode == "absolute":
            return _pl_binary_valid(
                x,
                previous,
                lambda a, b: pl.when(b.abs() > EPS).then((a - b) / b.abs()),
            )
        raise ValueError("denominator must be signed or absolute")


# ---------------------------------------------------------------------------
# Regression semantics
# ---------------------------------------------------------------------------

def _pd_regression_r2(
    y,
    x,
    window,
    min_periods=None,
    add_intercept=True,
    **_,
):
    y, x = aligned_pd(y, x)
    w = positive_int(window, "window")
    mp = 3 if min_periods is None else positive_int(min_periods, "min_periods")
    if mp > w:
        raise ValueError("min_periods must not exceed window")
    ya, xa = y.to_numpy(dtype=float), x.to_numpy(dtype=float)
    out = np.full(y.shape, np.nan, dtype=float)
    use_intercept = bool(add_intercept)
    for col in range(y.shape[1]):
        for row in range(y.shape[0]):
            start = max(0, row - w + 1)
            yv, xv = ya[start : row + 1, col], xa[start : row + 1, col]
            mask = np.isfinite(yv) & np.isfinite(xv)
            yy, xx = yv[mask], xv[mask]
            width = 2 if use_intercept else 1
            if yy.size < mp or yy.size <= width or np.var(xx) <= 0:
                continue
            design = xx[:, None]
            if use_intercept:
                design = np.column_stack((np.ones(xx.size), xx))
            if np.linalg.matrix_rank(design) < design.shape[1]:
                continue
            beta, *_ = np.linalg.lstsq(design, yy, rcond=None)
            resid = yy - design @ beta
            sse = float(resid @ resid)
            if use_intercept:
                centered = yy - yy.mean()
                denominator = float(centered @ centered)
            else:
                denominator = float(yy @ yy)
            if denominator > 0:
                out[row, col] = 1.0 - sse / denominator
    return frame_pd(y, out)


def _ols_residual_strict(
    y: np.ndarray,
    features: Sequence[np.ndarray],
    *,
    weights: np.ndarray | None,
    add_intercept: bool,
    min_obs: int | None,
) -> np.ndarray:
    out = np.full(y.shape, np.nan, dtype=float)
    mask = np.isfinite(y)
    for feature in features:
        mask &= np.isfinite(feature)
    if weights is not None:
        mask &= np.isfinite(weights) & (weights > 0)
    coefficients = len(features) + (1 if add_intercept else 0)
    if coefficients == 0:
        raise ValueError("neutralization requires an intercept or an exposure")
    # R22: default minimum observations = parameters + 1 (exactly 1 DOF).
    # The old hard ``max(..., 3)`` floor wrongly rejected 2-observation
    # no-intercept fits.
    required = coefficients + 1 if min_obs is None else int(min_obs)
    if required <= coefficients:
        raise ValueError("min_obs must exceed fitted coefficient count")
    if mask.sum() < required:
        return out
    design = (
        np.column_stack([feature[mask] for feature in features])
        if features
        else np.empty((int(mask.sum()), 0), dtype=float)
    )
    if add_intercept:
        design = np.column_stack((np.ones(design.shape[0]), design))
    if (
        design.shape[0] <= design.shape[1]
        or np.linalg.matrix_rank(design) < design.shape[1]
        or np.linalg.cond(design) > 1e12
    ):
        return out
    target = y[mask]
    fit_design, fit_target = design, target
    if weights is not None:
        root = np.sqrt(weights[mask])
        fit_design = design * root[:, None]
        fit_target = target * root
    beta, *_ = np.linalg.lstsq(fit_design, fit_target, rcond=None)
    out[mask] = target - design @ beta
    return out


def _pd_cs_neutralize(
    y,
    *exposures,
    group=None,
    weight=None,
    add_intercept=True,
    min_obs=None,
    **_,
):
    if not exposures and group is None:
        raise ValueError("cs_neutralize requires exposures and/or group")
    frames = [y, *exposures]
    if group is not None:
        frames.append(group)
    if weight is not None:
        frames.append(weight)
    aligned = aligned_pd(*frames)
    target = aligned[0]
    n_exposures = len(exposures)
    xs = list(aligned[1 : 1 + n_exposures])
    pos = 1 + n_exposures
    groups = aligned[pos] if group is not None else None
    pos += 1 if group is not None else 0
    weights = aligned[pos] if weight is not None else None
    out = np.full(target.shape, np.nan, dtype=float)

    for row in range(target.shape[0]):
        target_values = target.iloc[row].to_numpy(dtype=float)
        features = [x.iloc[row].to_numpy(dtype=float) for x in xs]
        if groups is not None:
            group_row = groups.iloc[row]
            group_valid = group_row.notna().to_numpy()
            target_values = target_values.copy()
            target_values[~group_valid] = np.nan
            dummies = pd.get_dummies(
                group_row.where(group_row.notna()),
                dummy_na=False,
                dtype=float,
            )
            dummies = dummies.loc[:, dummies.sum(axis=0) > 0]
            if bool(add_intercept) and dummies.shape[1] > 0:
                dummies = dummies.iloc[:, 1:]
            features.extend(
                dummies.iloc[:, index].to_numpy(dtype=float)
                for index in range(dummies.shape[1])
            )
        out[row] = _ols_residual_strict(
            target_values,
            features,
            weights=None
            if weights is None
            else weights.iloc[row].to_numpy(dtype=float),
            add_intercept=bool(add_intercept),
            min_obs=min_obs,
        )
    return frame_pd(target, out)


# ---------------------------------------------------------------------------
# Other cross-backend edge semantics
# ---------------------------------------------------------------------------

if pl is not None:

    def _pl_fundamental_staleness(available_at, decision_time, **_):
        replacements = {}
        for col in [
            c for c in pl_cols(available_at) if c in decision_time.columns
        ]:
            temp = pl.DataFrame(
                {
                    "_available": available_at[col].cast(pl.Datetime, strict=False),
                    "_decision": decision_time[col].cast(pl.Datetime, strict=False),
                }
            )
            days = (
                (pl.col("_decision") - pl.col("_available")).dt.total_seconds()
                / 86400.0
            )
            replacements[col] = temp.select(
                pl.when(days.is_not_null() & (days >= 0))
                .then(days.cast(pl.Float64))
                .otherwise(None)
                .alias(col)
            )[col]
        return pl_base_with(available_at, replacements)


def _wrap_exact_window(canonical: str):
    old = OperatorRegistry.get(canonical, backend="pandas_numpy")
    if old is None:
        return None

    def calculate(x, window=20, min_periods=None, **kwargs):
        exact_window = positive_int(window, "window")
        exact_min = (
            None
            if min_periods is None
            else positive_int(min_periods, "min_periods")
        )
        if exact_min is not None and exact_min > exact_window:
            raise ValueError("min_periods must not exceed window")
        if "scale" in kwargs:
            scale = float(kwargs["scale"])
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("scale must be finite and positive")
        return old.calculate(
            x,
            exact_window,
            min_periods=exact_min,
            **kwargs,
        )

    return calculate


def apply_operator_audit_fixes() -> None:
    _register(
        "period_lag",
        pandas_fn=_pd_period_lag,
        polars_fn=_pl_period_lag if pl is not None else None,
    )
    _register(
        "period_change",
        pandas_fn=_pd_period_change,
        polars_fn=_pl_period_change if pl is not None else None,
        pandas_source="layer_governance_primitives",
    )
    _register(
        "period_average",
        pandas_fn=_pd_period_average,
        polars_fn=_pl_period_average if pl is not None else None,
        pandas_source="layer_governance_primitives",
    )
    _register(
        "period_cagr",
        pandas_fn=_pd_period_cagr,
        polars_fn=_pl_period_cagr if pl is not None else None,
        pandas_source="layer_governance_primitives",
    )
    _register(
        "quarter_from_cumulative",
        pandas_fn=_pd_quarter_from_cumulative,
        polars_fn=_pl_quarter_from_cumulative if pl is not None else None,
    )
    _register(
        "ttm_from_quarterly",
        pandas_fn=_pd_ttm_from_quarterly,
        polars_fn=_pl_ttm_from_quarterly if pl is not None else None,
    )
    _register(
        "ttm_from_cumulative",
        pandas_fn=_pd_ttm_from_cumulative,
        polars_fn=_pl_ttm_from_cumulative if pl is not None else None,
    )
    _register(
        "yoy_by_period",
        pandas_fn=_pd_yoy_by_period,
        polars_fn=_pl_yoy_by_period if pl is not None else None,
    )
    _register("ts_regression_r2", pandas_fn=_pd_regression_r2)
    _register("cs_neutralize", pandas_fn=_pd_cs_neutralize)
    if pl is not None:
        _register(
            "fundamental_staleness",
            polars_fn=_pl_fundamental_staleness,
        )
    for canonical in ("ts_product", "ts_mad"):
        wrapped = _wrap_exact_window(canonical)
        if wrapped is not None:
            _register(
                canonical,
                pandas_fn=wrapped,
                pandas_source="semantic_hardening",
            )


apply_operator_audit_fixes()
