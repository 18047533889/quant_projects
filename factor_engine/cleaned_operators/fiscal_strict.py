# -*- coding: utf-8 -*-
"""Final strict PIT-safe fiscal-period operators.

The implementation is intentionally backend-symmetric.  Fiscal lags use exact
quarter ordinals, revisions are selected only from information visible at the
current row, and every semantic parameter is honoured by both Pandas and
Polars implementations.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.overhaul.base import (
    EPS,
    Spec,
    finite_pd,
    frame_pd,
    pl,
    pl_base_with,
    pl_cols,
    register_specs,
)

_REVISION_POLICIES = frozenset({"latest_available", "first_available"})


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if number < 1 or float(value) != float(number):
        raise ValueError(f"{name} must be a positive integer")
    return number


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a non-negative integer") from exc
    if number < 0 or float(value) != float(number):
        raise ValueError(f"{name} must be a non-negative integer")
    return number


def _revision_policy(value: Any) -> str:
    policy = str(value).lower()
    if policy not in _REVISION_POLICIES:
        raise ValueError("revision_policy must be 'latest_available' or 'first_available'")
    return policy


def _strict_align(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    if not frames:
        return ()
    base = frames[0]
    if not isinstance(base, pd.DataFrame):
        raise TypeError("fiscal operators require pandas DataFrame inputs")
    if not base.index.is_unique or not base.columns.is_unique:
        raise ValueError("fiscal operator input axes must be unique")
    for position, frame in enumerate(frames[1:], start=1):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"fiscal input {position} must be a pandas DataFrame")
        if not frame.index.is_unique or not frame.columns.is_unique:
            raise ValueError(f"fiscal input {position} axes must be unique")
        if not frame.index.equals(base.index):
            raise ValueError(f"fiscal input {position} index does not match the primary input")
        if not frame.columns.equals(base.columns):
            raise ValueError(f"fiscal input {position} columns do not match the primary input")
    return tuple(frames)


def period_ordinal(value: Any) -> int | None:
    """Parse an explicit fiscal-quarter identifier into a monotonic ordinal."""
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(value)
        return None if pd.isna(ts) else int(ts.year * 4 + (ts.month - 1) // 3)
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        number = int(value)
        # An 8-digit int is a YYYYMMDD date (e.g. 20240101), not a YYYYQ fiscal
        # encoding: do not silently map it through the year*10+quarter branch
        # (review §5.6).  Parse it as a calendar date quarter instead.
        if 10_000_000 <= number <= 99_999_999:
            try:
                ts = pd.Timestamp(f"{number // 10_000:04d}-{(number // 100) % 100:02d}-{number % 100:02d}")
            except Exception:
                ts = None
            if ts is not None and not pd.isna(ts):
                return int(ts.year * 4 + (ts.month - 1) // 3)
        year, quarter = divmod(number, 10)
        return year * 4 + quarter - 1 if year >= 1000 and 1 <= quarter <= 4 else number
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return period_ordinal(int(value))
    text = str(value).strip().upper()
    match = re.fullmatch(r"(\d{4})\D*Q?([1-4])", text)
    if match:
        return int(match.group(1)) * 4 + int(match.group(2)) - 1
    try:
        ts = pd.Timestamp(text)
    except Exception:
        return None
    return None if pd.isna(ts) else int(ts.year * 4 + (ts.month - 1) // 3)


def ordinal_frame(period_id: pd.DataFrame) -> pd.DataFrame:
    return period_id.apply(lambda col: col.map(period_ordinal)).astype("Float64")


def _lag_array(values: np.ndarray, period_values: np.ndarray, lag: int, policy: str) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    for col in range(values.shape[1]):
        visible: dict[int, float] = {}
        for row in range(values.shape[0]):
            ordinal = period_ordinal(period_values[row, col])
            if ordinal is None:
                continue
            value = values[row, col]
            if np.isfinite(value) and (policy == "latest_available" or ordinal not in visible):
                visible[int(ordinal)] = float(value)
            out[row, col] = visible.get(int(ordinal) - lag, np.nan)
    return out


def pd_period_lag(x, period_id, periods=1, revision_policy="latest_available", **_):
    x, period_id = _strict_align(x, period_id)
    lag = _nonnegative_int(periods, "periods")
    policy = _revision_policy(revision_policy)
    return frame_pd(
        x,
        _lag_array(
            x.to_numpy(dtype=float),
            period_id.to_numpy(dtype=object),
            lag,
            policy,
        ),
    )


def _ordinal_lag(period_id, lag, policy):
    ordinal = ordinal_frame(period_id).astype(float)
    return pd_period_lag(ordinal, period_id, lag, policy)


def _period_validity(x, period_id, lag, policy, require_consecutive):
    previous = pd_period_lag(x, period_id, lag, policy)
    valid = finite_pd(x) & finite_pd(previous)
    if bool(require_consecutive):
        current_ordinal = ordinal_frame(period_id).astype(float)
        prior_ordinal = _ordinal_lag(period_id, lag, policy)
        valid &= current_ordinal.sub(prior_ordinal).eq(float(lag))
    return previous, valid


def pd_period_change(
    x,
    period_id,
    periods=1,
    mode="absolute",
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
):
    x, period_id = _strict_align(x, period_id)
    lag = _positive_int(periods, "periods")
    policy = _revision_policy(revision_policy)
    previous, valid = _period_validity(x, period_id, lag, policy, require_consecutive)
    mode = str(mode).lower()
    if mode == "absolute":
        result = x - previous
    elif mode == "ratio":
        valid &= previous.abs().gt(EPS)
        result = x / previous.where(valid) - 1.0
    elif mode == "log":
        valid &= previous.abs().gt(EPS)
        ratio = x / previous.where(valid)
        valid &= ratio.gt(0)
        result = np.log(ratio.where(valid))
    else:
        raise ValueError("mode must be 'absolute', 'ratio', or 'log'")
    return result.where(valid).replace([np.inf, -np.inf], np.nan)


def _pieces(x, period_id, count, policy):
    ordinal = ordinal_frame(period_id).astype(float)
    pieces = [x]
    strict = finite_pd(x)
    available = finite_pd(x)
    for lag in range(1, count):
        piece = pd_period_lag(x, period_id, lag, policy)
        prior_ordinal = _ordinal_lag(period_id, lag, policy)
        pieces.append(piece)
        available &= finite_pd(piece)
        strict &= finite_pd(piece) & ordinal.sub(prior_ordinal).eq(float(lag))
    return pieces, strict, available


def pd_period_average(
    x,
    period_id,
    periods=2,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
):
    x, period_id = _strict_align(x, period_id)
    count = _positive_int(periods, "periods")
    policy = _revision_policy(revision_policy)
    pieces, strict, available = _pieces(x, period_id, count, policy)
    total = pieces[0].copy()
    for piece in pieces[1:]:
        total = total + piece
    return (total / float(count)).where(strict if bool(require_consecutive) else available)


def pd_period_cagr(
    x,
    period_id,
    periods=12,
    periods_per_year=4,
    sign_policy="strict",
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
):
    x, period_id = _strict_align(x, period_id)
    lag = _positive_int(periods, "periods")
    ppy = _positive_int(periods_per_year, "periods_per_year")
    policy = _revision_policy(revision_policy)
    previous, valid = _period_validity(x, period_id, lag, policy, require_consecutive)
    sign_policy = str(sign_policy).lower()
    if sign_policy == "strict":
        valid &= x.gt(0) & previous.gt(0)
        ratio = x / previous.where(valid)
    elif sign_policy == "absolute":
        valid &= previous.abs().gt(EPS)
        ratio = x.abs() / previous.abs().where(valid)
    else:
        raise ValueError("sign_policy must be 'strict' or 'absolute'")
    result = ratio.pow(float(ppy) / float(lag)) - 1.0
    return result.where(valid).replace([np.inf, -np.inf], np.nan)


def pd_quarter_from_cumulative(
    x,
    period_id,
    fiscal_quarter=None,
    revision_policy="latest_available",
    **_,
):
    x, period_id = _strict_align(x, period_id)
    policy = _revision_policy(revision_policy)
    ordinal = ordinal_frame(period_id).astype(float)
    previous = pd_period_lag(x, period_id, 1, policy)
    previous_ordinal = _ordinal_lag(period_id, 1, policy)
    if fiscal_quarter is None:
        quarter = ordinal.mod(4).add(1)
    else:
        _, _, fiscal_quarter = _strict_align(x, period_id, fiscal_quarter)
        quarter = fiscal_quarter.astype(float)
    consecutive = ordinal.sub(previous_ordinal).eq(1.0)
    result = x.where(quarter.eq(1.0), x - previous.where(consecutive))
    valid = finite_pd(x) & finite_pd(quarter) & (quarter.eq(1.0) | consecutive)
    return result.where(valid).replace([np.inf, -np.inf], np.nan)


def pd_ttm_from_quarterly(
    x,
    period_id,
    periods=4,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
):
    x, period_id = _strict_align(x, period_id)
    count = _positive_int(periods, "periods")
    policy = _revision_policy(revision_policy)
    pieces, strict, available = _pieces(x, period_id, count, policy)
    total = pieces[0].copy()
    for piece in pieces[1:]:
        total = total + piece
    return total.where(strict if bool(require_consecutive) else available).replace([np.inf, -np.inf], np.nan)


def pd_ttm_from_cumulative(
    x,
    period_id,
    fiscal_quarter=None,
    revision_policy="latest_available",
    **_,
):
    policy = _revision_policy(revision_policy)
    quarterly = pd_quarter_from_cumulative(
        x,
        period_id,
        fiscal_quarter,
        revision_policy=policy,
    )
    return pd_ttm_from_quarterly(
        quarterly,
        period_id,
        periods=4,
        require_consecutive=True,
        revision_policy=policy,
    )


def pd_yoy_by_period(
    x,
    period_id,
    periods=4,
    denominator="signed",
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
):
    x, period_id = _strict_align(x, period_id)
    lag = _positive_int(periods, "periods")
    policy = _revision_policy(revision_policy)
    previous, valid = _period_validity(x, period_id, lag, policy, require_consecutive)
    mode = str(denominator).lower()
    if mode == "signed":
        denom = previous
    elif mode == "absolute":
        denom = previous.abs()
    else:
        raise ValueError("denominator must be 'signed' or 'absolute'")
    valid &= denom.abs().gt(EPS)
    return ((x - previous) / denom.where(valid)).where(valid).replace([np.inf, -np.inf], np.nan)


if pl is not None:

    def _pl_utf8_of(frame, source_col: str, target_col: str | None = None):
        """Build a Utf8 expression for ``target_col`` (defaults to ``source_col``).

        Pandas object panels (report-period strings interleaved with leading
        NaN) reach Polars as ``pl.Object`` dtype, which cannot be ``.cast`` to
        Utf8.  Map Object columns element-wise; fast-cast String columns.
        """
        target_col = target_col or source_col
        col = pl.col(target_col)
        if frame.schema.get(source_col) == pl.Object:
            return col.map_elements(
                lambda v: None if v is None else str(v),
                return_dtype=pl.Utf8,
            )
        return col.cast(pl.Utf8, strict=False)

    def _pl_numeric_col(frame, source_col: str, target_col: str | None = None):
        """Expression usable for integer-encoded period ids.

        Object columns hold strings/dates (never encoded ints), so the numeric
        fallback is a null literal for them to avoid a hard ``cast`` failure.
        """
        target_col = target_col or source_col
        if frame.schema.get(source_col) == pl.Object:
            return pl.lit(None)
        return pl.col(target_col)

    def _pl_float_of(frame, source_col: str, target_col: str | None = None):
        """Float64 expression; Object columns (e.g. a quarter panel) map element-wise."""
        target_col = target_col or source_col
        col = pl.col(target_col)
        if frame.schema.get(source_col) == pl.Object:
            return col.map_elements(
                lambda v: None if v is None else float(v),
                return_dtype=pl.Float64,
            )
        return col.cast(pl.Float64, strict=False)

    def _pl_period_ordinal_expr(column: str, raw_utf8=None, raw_numeric=None):
        """Return the exact Polars equivalent of :func:`period_ordinal`.

        Polars' format-inferred ``str.to_date`` can fail the whole expression on
        mixed fiscal identifiers such as ``2023Q1`` and ISO dates.  Parse each
        supported representation with anchored expressions instead, and keep
        unknown values null rather than guessing.
        """
        raw = pl.col(column)
        text = (
            raw_utf8
            if raw_utf8 is not None
            else raw.cast(pl.Utf8, strict=False)
        ).str.strip_chars().str.to_uppercase()

        quarter_year = text.str.extract(
            r"^(\d{4})(?:\D*Q?)([1-4])$", 1
        ).cast(pl.Int64, strict=False)
        quarter_number = text.str.extract(
            r"^(\d{4})(?:\D*Q?)([1-4])$", 2
        ).cast(pl.Int64, strict=False)

        iso_year = text.str.extract(
            r"^(\d{4})[-/](\d{1,2})[-/]\d{1,2}(?:[ T].*)?$", 1
        ).cast(pl.Int64, strict=False)
        iso_month = text.str.extract(
            r"^(\d{4})[-/](\d{1,2})[-/]\d{1,2}(?:[ T].*)?$", 2
        ).cast(pl.Int64, strict=False)
        compact_year = text.str.extract(
            r"^(\d{4})(\d{2})\d{2}(?:[ T].*)?$", 1
        ).cast(pl.Int64, strict=False)
        compact_month = text.str.extract(
            r"^(\d{4})(\d{2})\d{2}(?:[ T].*)?$", 2
        ).cast(pl.Int64, strict=False)
        date_year = pl.coalesce([iso_year, compact_year])
        date_month = pl.coalesce([iso_month, compact_month])
        valid_date = date_year.is_not_null() & date_month.is_between(1, 12)

        raw_num = raw_numeric if raw_numeric is not None else raw
        numeric_int = raw_num.cast(pl.Int64, strict=False)
        numeric_float = raw_num.cast(pl.Float64, strict=False)
        numeric_year = numeric_int // 10
        numeric_quarter = numeric_int % 10
        encoded_quarter = (
            numeric_int.is_not_null()
            & (numeric_year >= 1000)
            & numeric_quarter.is_between(1, 4)
        )

        return (
            pl.when(quarter_year.is_not_null() & quarter_number.is_not_null())
            .then(quarter_year * 4 + quarter_number - 1)
            .when(valid_date)
            .then(date_year * 4 + ((date_month - 1) // 3))
            .when(encoded_quarter)
            .then(numeric_year * 4 + numeric_quarter - 1)
            .otherwise(numeric_float)
            .cast(pl.Float64)
        )


    def _pl_ordinal_frame(period_id):
        return period_id.with_columns(
            [
                _pl_period_ordinal_expr(
                    col,
                    _pl_utf8_of(period_id, col),
                    _pl_numeric_col(period_id, col),
                ).alias(col)
                for col in pl_cols(period_id)
            ]
        )


    def pl_period_lag(
        x,
        period_id,
        periods=1,
        revision_policy="latest_available",
        **_,
    ):
        lag = _nonnegative_int(periods, "periods")
        policy = _revision_policy(revision_policy)
        replacements = {}
        for col in [c for c in pl_cols(x) if c in period_id.columns]:
            temp = pl.DataFrame(
                {
                    "_row": pl.int_range(0, x.height, eager=True),
                    "_x": x[col].cast(pl.Float64, strict=False),
                    "_pid": period_id[col],
                }
            ).with_columns(
                _pl_period_ordinal_expr(
                    "_pid",
                    _pl_utf8_of(period_id, col, "_pid"),
                    _pl_numeric_col(period_id, col, "_pid"),
                ).alias("_ordinal")
            )
            targets = temp.select(
                "_row",
                "_pid",
                "_ordinal",
                (pl.col("_ordinal") - lag).alias("_target"),
            )
            history = temp.select(
                pl.col("_row").alias("_hrow"),
                pl.col("_ordinal").alias("_hordinal"),
                pl.col("_x").alias("_hx"),
            )
            candidates = (
                targets.select("_row", "_target")
                .join(history, left_on="_target", right_on="_hordinal", how="inner")
                .filter(
                    (pl.col("_hrow") <= pl.col("_row"))
                    & pl.col("_hx").is_not_null()
                    & pl.col("_hx").is_finite()
                )
                .sort(["_row", "_hrow"])
            )
            aggregate = (
                pl.col("_hx").first()
                if policy == "first_available"
                else pl.col("_hx").last()
            )
            selected = candidates.group_by("_row", maintain_order=True).agg(
                aggregate.alias("_out")
            )
            result = (
                targets.select("_row", "_pid", "_ordinal")
                .join(selected, on="_row", how="left", maintain_order="left")
                .select(
                    pl.when(pl.col("_ordinal").is_null())
                    .then(None)
                    .otherwise(pl.col("_out"))
                    .alias(col)
                )
            )[col]
            replacements[col] = result
        return pl_base_with(x, replacements)


    def _pl_valid(expr):
        return expr.is_not_null() & expr.is_finite()


    def _pl_binary_frames(x, y, fn):
        replacements = {}
        for col in [c for c in pl_cols(x) if c in y.columns]:
            temp = pl.DataFrame(
                {
                    "a": x[col].cast(pl.Float64, strict=False),
                    "b": y[col].cast(pl.Float64, strict=False),
                }
            )
            replacements[col] = temp.select(fn(pl.col("a"), pl.col("b")).alias(col))[col]
        return pl_base_with(x, replacements)


    def _pl_period_validity(x, period_id, lag, policy, require_consecutive):
        previous = pl_period_lag(x, period_id, lag, policy)
        if not bool(require_consecutive):
            return previous, None
        current_ordinal = _pl_ordinal_frame(period_id)
        prior_ordinal = pl_period_lag(current_ordinal, period_id, lag, policy)
        return previous, (current_ordinal, prior_ordinal)


    def pl_period_change(
        x,
        period_id,
        periods=1,
        mode="absolute",
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ):
        lag = _positive_int(periods, "periods")
        policy = _revision_policy(revision_policy)
        previous, ordinal_pair = _pl_period_validity(
            x, period_id, lag, policy, require_consecutive
        )
        mode = str(mode).lower()
        replacements = {}
        for col in [c for c in pl_cols(x) if c in previous.columns]:
            data = {
                "a": x[col].cast(pl.Float64, strict=False),
                "b": previous[col].cast(pl.Float64, strict=False),
            }
            if ordinal_pair is not None:
                data["o"] = ordinal_pair[0][col].cast(pl.Float64, strict=False)
                data["po"] = ordinal_pair[1][col].cast(pl.Float64, strict=False)
            temp = pl.DataFrame(data)
            a, b = pl.col("a"), pl.col("b")
            valid = _pl_valid(a) & _pl_valid(b)
            if ordinal_pair is not None:
                valid &= (pl.col("o") - pl.col("po") == float(lag))
            if mode == "absolute":
                expr = a - b
            elif mode == "ratio":
                valid &= b.abs() > EPS
                expr = a / b - 1.0
            elif mode == "log":
                valid &= (b.abs() > EPS) & (a / b > 0)
                expr = (a / b).log()
            else:
                raise ValueError("mode must be 'absolute', 'ratio', or 'log'")
            replacements[col] = temp.select(
                pl.when(valid).then(expr).otherwise(None).alias(col)
            )[col]
        return pl_base_with(x, replacements)


    def _pl_pieces(x, period_id, count, policy):
        ordinals = _pl_ordinal_frame(period_id)
        pieces = [x]
        prior_ordinals = [ordinals]
        for lag in range(1, count):
            pieces.append(pl_period_lag(x, period_id, lag, policy))
            prior_ordinals.append(pl_period_lag(ordinals, period_id, lag, policy))
        return pieces, prior_ordinals


    def pl_period_average(
        x,
        period_id,
        periods=2,
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ):
        count = _positive_int(periods, "periods")
        policy = _revision_policy(revision_policy)
        pieces, ordinal_pieces = _pl_pieces(x, period_id, count, policy)
        replacements = {}
        for col in pl_cols(x):
            data = {f"v{i}": piece[col].cast(pl.Float64, strict=False) for i, piece in enumerate(pieces)}
            if bool(require_consecutive):
                data.update(
                    {f"o{i}": ordinal_pieces[i][col].cast(pl.Float64, strict=False) for i in range(count)}
                )
            temp = pl.DataFrame(data)
            valid = pl.all_horizontal(*[_pl_valid(pl.col(f"v{i}")) for i in range(count)])
            if bool(require_consecutive):
                valid &= pl.all_horizontal(
                    *[(pl.col("o0") - pl.col(f"o{i}") == float(i)) for i in range(1, count)]
                )
            total = pl.sum_horizontal(*[pl.col(f"v{i}") for i in range(count)])
            replacements[col] = temp.select(
                pl.when(valid).then(total / float(count)).otherwise(None).alias(col)
            )[col]
        return pl_base_with(x, replacements)


    def pl_period_cagr(
        x,
        period_id,
        periods=12,
        periods_per_year=4,
        sign_policy="strict",
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ):
        lag = _positive_int(periods, "periods")
        ppy = _positive_int(periods_per_year, "periods_per_year")
        policy = _revision_policy(revision_policy)
        previous, ordinal_pair = _pl_period_validity(
            x, period_id, lag, policy, require_consecutive
        )
        sign_policy = str(sign_policy).lower()
        replacements = {}
        for col in [c for c in pl_cols(x) if c in previous.columns]:
            data = {
                "a": x[col].cast(pl.Float64, strict=False),
                "b": previous[col].cast(pl.Float64, strict=False),
            }
            if ordinal_pair is not None:
                data["o"] = ordinal_pair[0][col].cast(pl.Float64, strict=False)
                data["po"] = ordinal_pair[1][col].cast(pl.Float64, strict=False)
            temp = pl.DataFrame(data)
            a, b = pl.col("a"), pl.col("b")
            valid = _pl_valid(a) & _pl_valid(b)
            if ordinal_pair is not None:
                valid &= (pl.col("o") - pl.col("po") == float(lag))
            if sign_policy == "strict":
                valid &= (a > 0) & (b > 0)
                ratio = a / b
            elif sign_policy == "absolute":
                valid &= b.abs() > EPS
                ratio = a.abs() / b.abs()
            else:
                raise ValueError("sign_policy must be 'strict' or 'absolute'")
            expr = ratio.pow(float(ppy) / float(lag)) - 1.0
            replacements[col] = temp.select(
                pl.when(valid).then(expr).otherwise(None).alias(col)
            )[col]
        return pl_base_with(x, replacements)


    def pl_quarter_from_cumulative(
        x,
        period_id,
        fiscal_quarter=None,
        revision_policy="latest_available",
        **_,
    ):
        policy = _revision_policy(revision_policy)
        ordinal = _pl_ordinal_frame(period_id)
        previous = pl_period_lag(x, period_id, 1, policy)
        previous_ordinal = pl_period_lag(ordinal, period_id, 1, policy)
        quarter_frame = (
            ordinal.with_columns([((pl.col(c) % 4) + 1).alias(c) for c in pl_cols(ordinal)])
            if fiscal_quarter is None
            else fiscal_quarter.with_columns(
                [_pl_float_of(fiscal_quarter, c).alias(c) for c in pl_cols(fiscal_quarter)]
            )
        )
        replacements = {}
        for col in [c for c in pl_cols(x) if c in quarter_frame.columns]:
            temp = pl.DataFrame(
                {
                    "x": x[col].cast(pl.Float64, strict=False),
                    "p": previous[col].cast(pl.Float64, strict=False),
                    "o": ordinal[col].cast(pl.Float64, strict=False),
                    "po": previous_ordinal[col].cast(pl.Float64, strict=False),
                    "q": quarter_frame[col].cast(pl.Float64, strict=False),
                }
            )
            consecutive = pl.col("o") - pl.col("po") == 1.0
            valid = _pl_valid(pl.col("x")) & _pl_valid(pl.col("q")) & (
                (pl.col("q") == 1.0) | consecutive
            )
            expr = (
                pl.when(pl.col("q") == 1.0)
                .then(pl.col("x"))
                .otherwise(pl.col("x") - pl.col("p"))
            )
            replacements[col] = temp.select(
                pl.when(valid).then(expr).otherwise(None).alias(col)
            )[col]
        return pl_base_with(x, replacements)


    def pl_ttm_from_quarterly(
        x,
        period_id,
        periods=4,
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ):
        count = _positive_int(periods, "periods")
        policy = _revision_policy(revision_policy)
        pieces, ordinal_pieces = _pl_pieces(x, period_id, count, policy)
        replacements = {}
        for col in pl_cols(x):
            data = {f"v{i}": piece[col].cast(pl.Float64, strict=False) for i, piece in enumerate(pieces)}
            if bool(require_consecutive):
                data.update(
                    {f"o{i}": ordinal_pieces[i][col].cast(pl.Float64, strict=False) for i in range(count)}
                )
            temp = pl.DataFrame(data)
            valid = pl.all_horizontal(*[_pl_valid(pl.col(f"v{i}")) for i in range(count)])
            if bool(require_consecutive):
                valid &= pl.all_horizontal(
                    *[(pl.col("o0") - pl.col(f"o{i}") == float(i)) for i in range(1, count)]
                )
            total = pl.sum_horizontal(*[pl.col(f"v{i}") for i in range(count)])
            replacements[col] = temp.select(
                pl.when(valid).then(total).otherwise(None).alias(col)
            )[col]
        return pl_base_with(x, replacements)


    def pl_ttm_from_cumulative(
        x,
        period_id,
        fiscal_quarter=None,
        revision_policy="latest_available",
        **_,
    ):
        policy = _revision_policy(revision_policy)
        quarterly = pl_quarter_from_cumulative(
            x,
            period_id,
            fiscal_quarter,
            revision_policy=policy,
        )
        return pl_ttm_from_quarterly(
            quarterly,
            period_id,
            periods=4,
            require_consecutive=True,
            revision_policy=policy,
        )


    def pl_yoy_by_period(
        x,
        period_id,
        periods=4,
        denominator="signed",
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ):
        lag = _positive_int(periods, "periods")
        policy = _revision_policy(revision_policy)
        previous, ordinal_pair = _pl_period_validity(
            x, period_id, lag, policy, require_consecutive
        )
        denominator = str(denominator).lower()
        replacements = {}
        for col in [c for c in pl_cols(x) if c in previous.columns]:
            data = {
                "a": x[col].cast(pl.Float64, strict=False),
                "b": previous[col].cast(pl.Float64, strict=False),
            }
            if ordinal_pair is not None:
                data["o"] = ordinal_pair[0][col].cast(pl.Float64, strict=False)
                data["po"] = ordinal_pair[1][col].cast(pl.Float64, strict=False)
            temp = pl.DataFrame(data)
            a, b = pl.col("a"), pl.col("b")
            valid = _pl_valid(a) & _pl_valid(b)
            if ordinal_pair is not None:
                valid &= (pl.col("o") - pl.col("po") == float(lag))
            if denominator == "signed":
                denom = b
            elif denominator == "absolute":
                denom = b.abs()
            else:
                raise ValueError("denominator must be 'signed' or 'absolute'")
            valid &= denom.abs() > EPS
            expr = (a - b) / denom
            replacements[col] = temp.select(
                pl.when(valid).then(expr).otherwise(None).alias(col)
            )[col]
        return pl_base_with(x, replacements)

else:  # pragma: no cover - optional dependency
    pl_period_lag = None
    pl_period_change = None
    pl_period_average = None
    pl_period_cagr = None
    pl_quarter_from_cumulative = None
    pl_ttm_from_quarterly = None
    pl_ttm_from_cumulative = None
    pl_yoy_by_period = None


def register() -> None:
    register_specs(
        {
            "period_lag": Spec(
                "fundamental_period",
                ["x", "period_id", "periods", "revision_policy"],
                "exact fiscal ordinal lag with visible revision policy",
                pd_period_lag,
                pl_period_lag,
            ),
            "period_change": Spec(
                "fundamental_period",
                [
                    "x",
                    "period_id",
                    "periods",
                    "mode",
                    "require_consecutive",
                    "revision_policy",
                ],
                "strict fiscal-period change",
                pd_period_change,
                pl_period_change,
            ),
            "period_average": Spec(
                "fundamental_period",
                [
                    "x",
                    "period_id",
                    "periods",
                    "require_consecutive",
                    "revision_policy",
                ],
                "strict fiscal-period average",
                pd_period_average,
                pl_period_average,
            ),
            "period_cagr": Spec(
                "fundamental_period",
                [
                    "x",
                    "period_id",
                    "periods",
                    "periods_per_year",
                    "sign_policy",
                    "require_consecutive",
                    "revision_policy",
                ],
                "strict fiscal-period compound growth",
                pd_period_cagr,
                pl_period_cagr,
            ),
            "quarter_from_cumulative": Spec(
                "fundamental_period",
                ["x", "period_id", "fiscal_quarter", "revision_policy"],
                "strict cumulative-to-quarter conversion",
                pd_quarter_from_cumulative,
                pl_quarter_from_cumulative,
            ),
            "ttm_from_quarterly": Spec(
                "fundamental_period",
                [
                    "x",
                    "period_id",
                    "periods",
                    "require_consecutive",
                    "revision_policy",
                ],
                "strict consecutive-period TTM",
                pd_ttm_from_quarterly,
                pl_ttm_from_quarterly,
            ),
            "ttm_from_cumulative": Spec(
                "fundamental_period",
                ["x", "period_id", "fiscal_quarter", "revision_policy"],
                "strict cumulative-to-TTM",
                pd_ttm_from_cumulative,
                pl_ttm_from_cumulative,
            ),
            "yoy_by_period": Spec(
                "fundamental_period",
                [
                    "x",
                    "period_id",
                    "periods",
                    "denominator",
                    "require_consecutive",
                    "revision_policy",
                ],
                "strict fiscal-period growth",
                pd_yoy_by_period,
                pl_yoy_by_period,
            ),
        }
    )


register()
