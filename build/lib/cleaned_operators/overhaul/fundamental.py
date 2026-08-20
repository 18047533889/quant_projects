# -*- coding: utf-8 -*-
"""Strict PIT-safe fiscal-period operators."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.overhaul.base import (
    EPS,
    Spec,
    aligned_pd,
    finite_pd,
    frame_pd,
    nonnegative_int,
    pl,
    pl_base_with,
    pl_cols,
    positive_int,
    register_specs,
)


def period_key(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return value.item()
    except AttributeError:
        return value


def pd_period_lag(
    x: pd.DataFrame,
    period_id: pd.DataFrame,
    periods: int = 1,
    revision_policy: str = "latest_available",
    **_,
) -> pd.DataFrame:
    x, period_id = aligned_pd(x, period_id)
    lag = nonnegative_int(periods, "periods")
    policy = str(revision_policy).lower()
    if policy not in {"latest_available", "first_available"}:
        raise ValueError("revision_policy must be 'latest_available' or 'first_available'")
    xv, pv = x.to_numpy(), period_id.to_numpy()
    out = np.full(x.shape, np.nan)
    for col in range(x.shape[1]):
        order, positions, values = [], {}, {}
        for row in range(x.shape[0]):
            key = period_key(pv[row, col])
            if key is None:
                continue
            if key not in positions:
                positions[key] = len(order)
                order.append(key)
            value = xv[row, col]
            if np.isfinite(value) and (policy == "latest_available" or key not in values):
                values[key] = float(value)
            target = positions[key] - lag
            if target >= 0:
                out[row, col] = values.get(order[target], np.nan)
    return frame_pd(x, out)


def pl_period_lag(
    x,
    period_id,
    periods=1,
    revision_policy="latest_available",
    **_,
):
    lag = nonnegative_int(periods, "periods")
    policy = str(revision_policy).lower()
    if policy not in {"latest_available", "first_available"}:
        raise ValueError("revision_policy must be 'latest_available' or 'first_available'")
    replacements = {}
    for col in [c for c in pl_cols(x) if c in period_id.columns]:
        pid_text = pl.col("_pid").cast(pl.Utf8, strict=False)
        year = pid_text.str.extract(r"(\d{4})", 1).cast(pl.Int64, strict=False)
        quarter = pid_text.str.extract(r"([1-4])$", 1).cast(pl.Int64, strict=False)
        numeric = pl.col("_pid").cast(pl.Float64, strict=False)
        ordinal = (
            pl.when(year.is_not_null() & quarter.is_not_null())
            .then(year * 4 + quarter - 1)
            .otherwise(numeric)
            .cast(pl.Float64)
        )
        temp = pl.DataFrame({
            "_row": pl.int_range(0, x.height, eager=True),
            "_x": x[col].cast(pl.Float64, strict=False),
            "_pid": period_id[col],
        }).with_columns(ordinal.alias("_ordinal"))
        targets = temp.with_columns((pl.col("_ordinal") - lag).alias("_target"))
        history = temp.select(
            pl.col("_row").alias("_hrow"),
            pl.col("_ordinal").alias("_hordinal"),
            pl.col("_x").alias("_hx"),
        )
        candidates = (
            targets.select("_row", "_target")
            .join(history, left_on="_target", right_on="_hordinal", how="inner")
            .filter((pl.col("_hrow") <= pl.col("_row")) & pl.col("_hx").is_not_null() & pl.col("_hx").is_finite())
            .sort(["_row", "_hrow"])
        )
        aggregate = pl.col("_hx").first() if policy == "first_available" else pl.col("_hx").last()
        selected = candidates.group_by("_row", maintain_order=True).agg(aggregate.alias("_out"))
        result = (
            temp.select("_row", "_pid")
            .join(selected, on="_row", how="left", maintain_order="left")
            .select(pl.when(pl.col("_pid").is_null()).then(None).otherwise(pl.col("_out")).alias("_out"))
        )["_out"]
        replacements[col] = result.rename(col)
    return pl_base_with(x, replacements)


def _pl_binary(x, y, fn):
    replacements = {}
    for col in [c for c in pl_cols(x) if c in y.columns]:
        replacements[col] = pl.DataFrame({"x": x[col], "y": y[col]}).select(fn(pl.col("x"), pl.col("y")).alias(col))[col]
    return pl_base_with(x, replacements)


def pl_period_change(x, period_id, periods=1, mode="absolute", require_consecutive=True, **_):
    lag = positive_int(periods, "periods")
    previous = pl_period_lag(x, period_id, lag)
    mode = str(mode).lower()
    if mode == "absolute":
        return _pl_binary(x, previous, lambda a, b: a - b)
    if mode == "ratio":
        return _pl_binary(x, previous, lambda a, b: pl.when(b.abs() > EPS).then(a / b - 1.0))
    if mode == "log":
        return _pl_binary(x, previous, lambda a, b: pl.when((b.abs() > EPS) & (a / b > 0)).then((a / b).log()))
    raise ValueError("mode must be 'absolute', 'ratio', or 'log'")


def pl_period_average(x, period_id, periods=2, require_consecutive=True, **_):
    count = positive_int(periods, "periods")
    pieces = [x] + [pl_period_lag(x, period_id, lag) for lag in range(1, count)]
    replacements = {}
    for col in pl_cols(x):
        temp = pl.DataFrame({f"v{i}": piece[col] for i, piece in enumerate(pieces)})
        valid = pl.all_horizontal(*[pl.col(f"v{i}").is_not_null() for i in range(count)])
        total = pl.sum_horizontal(*[pl.col(f"v{i}") for i in range(count)])
        replacements[col] = temp.select(pl.when(valid).then(total / count).alias(col))[col]
    return pl_base_with(x, replacements)


def pl_period_cagr(x, period_id, periods=12, periods_per_year=4, sign_policy="strict", require_consecutive=True, **_):
    lag = positive_int(periods, "periods")
    exponent = positive_int(periods_per_year, "periods_per_year") / float(lag)
    previous = pl_period_lag(x, period_id, lag)
    policy = str(sign_policy).lower()
    if policy == "strict":
        return _pl_binary(x, previous, lambda a, b: pl.when((a > 0) & (b > 0)).then((a / b).pow(exponent) - 1.0))
    if policy == "absolute":
        return _pl_binary(x, previous, lambda a, b: pl.when(b.abs() > EPS).then((a.abs() / b.abs()).pow(exponent) - 1.0))
    raise ValueError("sign_policy must be 'strict' or 'absolute'")


def pl_quarter_from_cumulative(x, period_id, fiscal_quarter, **_):
    previous = pl_period_lag(x, period_id, 1)
    previous_q = pl_period_lag(fiscal_quarter, period_id, 1)
    replacements = {}
    for col in pl_cols(x):
        temp = pl.DataFrame({"x": x[col], "p": previous[col], "q": fiscal_quarter[col], "pq": previous_q[col]})
        consecutive = ((pl.col("pq") == 4) & (pl.col("q") == 1)) | (pl.col("q") == pl.col("pq") + 1)
        expr = pl.when(pl.col("q") == 1).then(pl.col("x")).when(consecutive).then(pl.col("x") - pl.col("p"))
        replacements[col] = temp.select(expr.alias(col))[col]
    return pl_base_with(x, replacements)


def pl_ttm_from_quarterly(x, period_id, periods=4, require_consecutive=True, **_):
    count = positive_int(periods, "periods")
    pieces = [x] + [pl_period_lag(x, period_id, lag) for lag in range(1, count)]
    replacements = {}
    for col in pl_cols(x):
        temp = pl.DataFrame({f"v{i}": piece[col] for i, piece in enumerate(pieces)})
        valid = pl.all_horizontal(*[pl.col(f"v{i}").is_not_null() for i in range(count)])
        replacements[col] = temp.select(pl.when(valid).then(pl.sum_horizontal(*[pl.col(f"v{i}") for i in range(count)])).alias(col))[col]
    return pl_base_with(x, replacements)


def pl_ttm_from_cumulative(x, period_id, fiscal_quarter, **_):
    return pl_ttm_from_quarterly(pl_quarter_from_cumulative(x, period_id, fiscal_quarter), period_id)


def pl_yoy_by_period(x, period_id, periods=4, denominator="signed", **_):
    previous = pl_period_lag(x, period_id, positive_int(periods, "periods"))
    mode = str(denominator).lower()
    if mode == "signed":
        return _pl_binary(x, previous, lambda a, b: pl.when(b.abs() > EPS).then((a - b) / b))
    if mode == "absolute":
        return _pl_binary(x, previous, lambda a, b: pl.when(b.abs() > EPS).then((a - b) / b.abs()))
    raise ValueError("denominator must be 'signed' or 'absolute'")


def pd_quarter_from_cumulative(x, period_id, fiscal_quarter, **_):
    x, period_id, fiscal_quarter = aligned_pd(x, period_id, fiscal_quarter)
    previous = pd_period_lag(x, period_id, 1)
    previous_q = pd_period_lag(fiscal_quarter, period_id, 1)
    q = fiscal_quarter.astype(float)
    consecutive = ((previous_q == 4) & (q == 1)) | (q == previous_q + 1)
    result = x.where(q == 1, x - previous.where(consecutive))
    return result.where(finite_pd(x) & finite_pd(q))


def pd_ttm_from_quarterly(
    x,
    period_id,
    periods=4,
    require_consecutive=True,
    **_,
):
    count = positive_int(periods, "periods")
    pieces = [x] + [pd_period_lag(x, period_id, lag) for lag in range(1, count)]
    total = pieces[0].copy()
    valid = pieces[0].notna()
    for piece in pieces[1:]:
        total = total + piece
        valid &= piece.notna()
    if require_consecutive:
        return total.where(valid)
    return total.where(pieces[0].notna())


def pd_ttm_from_cumulative(x, period_id, fiscal_quarter, **_):
    quarterly = pd_quarter_from_cumulative(x, period_id, fiscal_quarter)
    return pd_ttm_from_quarterly(quarterly, period_id, periods=4, require_consecutive=True)


def pd_yoy_by_period(
    x,
    period_id,
    periods=4,
    denominator="signed",
    **_,
):
    previous = pd_period_lag(x, period_id, positive_int(periods, "periods"))
    mode = str(denominator).lower()
    if mode == "signed":
        denom = previous
    elif mode == "absolute":
        denom = previous.abs()
    else:
        raise ValueError("denominator must be 'signed' or 'absolute'")
    valid = finite_pd(x) & finite_pd(denom) & denom.abs().gt(EPS)
    return ((x - previous) / denom.where(valid)).where(valid).replace([np.inf, -np.inf], np.nan)


def register() -> None:
    register_specs({
        "period_lag": Spec(
            "fundamental_period",
            ["x", "period_id", "periods", "revision_policy"],
            "按真实可见的不同财务报告期滞后",
            pd_period_lag,
            pl_period_lag,
        ),
        "period_change": Spec("fundamental_period", ["x", "period_id", "periods", "mode", "require_consecutive"], "按报告期计算变化", pd_period_lag, pl_period_change),
        "period_average": Spec("fundamental_period", ["x", "period_id", "periods", "require_consecutive"], "按报告期计算平均", pd_period_lag, pl_period_average),
        "period_cagr": Spec("fundamental_period", ["x", "period_id", "periods", "periods_per_year", "sign_policy", "require_consecutive"], "按报告期计算复合增长", pd_period_lag, pl_period_cagr),
        "quarter_from_cumulative": Spec(
            "fundamental_period",
            ["x", "period_id", "fiscal_quarter"],
            "累计流量值按报告期转单季度",
            pd_quarter_from_cumulative,
            pl_quarter_from_cumulative,
        ),
        "ttm_from_quarterly": Spec(
            "fundamental_period",
            ["x", "period_id", "periods", "require_consecutive"],
            "按不同报告期累计 TTM，避免日频 forward-fill 重复计数",
            pd_ttm_from_quarterly,
            pl_ttm_from_quarterly,
        ),
        "ttm_from_cumulative": Spec(
            "fundamental_period",
            ["x", "period_id", "fiscal_quarter"],
            "累计值转单季后按报告期计算 TTM",
            pd_ttm_from_cumulative,
            pl_ttm_from_cumulative,
        ),
        "yoy_by_period": Spec(
            "fundamental_period",
            ["x", "period_id", "periods", "denominator"],
            "按不同报告期计算同比变化",
            pd_yoy_by_period,
            pl_yoy_by_period,
        ),
    })
