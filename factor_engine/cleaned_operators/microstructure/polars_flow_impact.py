# -*- coding: utf-8 -*-
"""Genuine Polars backends for the order-flow microstructure family (minute → daily).

A minute wide panel is melted to long, aggregated per (date, instrument), and
pivoted back to a daily wide panel.  ``intraday_bvc_imbalance`` /
``intraday_impact_beta`` / ``intraday_impact_asymmetry`` are pure ``pl.Expr``
implementations.  ``intraday_return_wasserstein_shift`` and ``micro_bvc_vpin``
need cross-day / sequential-bucket state that plain window expressions cannot
express, so those two delegate the per-instrument daily computation to the
shared numpy kernels from ``flow_impact`` via ``map_batches`` (polars-native
plumbing, no pandas panel round-trip), which guarantees exact pandas parity by
construction.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.microstructure.flow_impact import (
    _bvc_flow,
    _normal_cdf,
)

_TIME_COLS = frozenset({"date", "timestamp", "time", "QuoteTime", "TradeDate"})
_EPS = 1e-12


def _time_col(df: pl.DataFrame) -> str:
    for c in df.columns:
        if c in _TIME_COLS:
            return c
    return "date"


def _melt(df: pl.DataFrame, value_name: str) -> pl.DataFrame:
    tc = _time_col(df)
    cols = [c for c in df.columns if c != tc]
    long = df.unpivot(index=[tc], on=cols, variable_name="instrument", value_name=value_name)
    # Keep only the normalised ``ts`` column; dropping the raw time column
    # avoids duplicate ``date`` columns when two melted frames are joined.
    return long.with_columns(pl.col(tc).cast(pl.Datetime).alias("ts")).drop(tc)


def _with_date(long: pl.DataFrame) -> pl.DataFrame:
    return long.with_columns(pl.col("ts").dt.date().alias("date"))


def _pivot(df: pl.DataFrame, value: str) -> pl.DataFrame:
    piv = df.pivot(index="date", on="instrument", values=value, aggregate_function="first")
    return piv.fill_null(float("nan"))


def _mk(canonical: str, description: str, params: list[str], fn):
    metadata = OperatorMetadata(
        name=canonical,
        category="intraday_microstructure",
        description=description,
        param_names=params,
        return_type="series",
        tags=["polars", "intraday", "minute", "native", "typed_v2"],
    )

    def _calculate_series(self, *args, **kwargs):
        return fn(*args, **kwargs)

    cls = type(
        f"PolarsFlowImpact_{canonical}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=canonical,
        category="intraday_microstructure",
        business_category="intraday_microstructure",
        canonical=canonical,
        source="microstructure.polars_flow_impact",
    )(cls)
    return cls


# ---------------------------------------------------------------------------
# BV-C signed flow imbalance
# ---------------------------------------------------------------------------

def _bvc_imbalance_polars(
    close: pl.DataFrame, volume: pl.DataFrame, scale_window: int, locked: Any
) -> pl.DataFrame:
    sw = max(2, int(scale_window))
    mp = max(2, sw // 2)
    lc = _melt(close, "close")
    lv = _melt(volume, "volume")
    long = _with_date(lc.join(lv, on=["ts", "instrument"], how="inner"))
    long = long.sort(["date", "instrument", "ts"])
    long = long.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over(["date", "instrument"])).log().alias("r")
    )
    long = long.with_columns(
        # NaN -> null so the rolling window (which skips nulls, not NaNs)
        # matches the numpy reference that drops non-finite returns.
        pl.when(pl.col("r").is_finite()).then(pl.col("r")).otherwise(None).alias("r_clean")
    )
    long = long.with_columns(
        pl.col("r_clean")
        .rolling_std(window_size=sw, min_periods=mp, ddof=0)
        .over(["date", "instrument"])
        .alias("scale")
    )
    z = pl.when(pl.col("r").is_finite() & pl.col("scale").is_finite()).then(
        pl.col("r") / (pl.col("scale") + _EPS)
    ).otherwise(0.0)
    p = z.map_batches(lambda s: _normal_cdf(s.to_numpy()), return_dtype=pl.Float64)
    # NaN volume -> 0 (mirror the numpy reference, which replaces non-finite
    # volume with zero before the daily sums).
    clean_v = pl.when(pl.col("volume").is_finite()).then(pl.col("volume")).otherwise(0.0)
    of = clean_v * (2.0 * p - 1.0)
    if locked is not None:
        lk = _melt(locked, "locked").select(["ts", "instrument", "locked"])
        long = long.join(lk, on=["ts", "instrument"], how="left")
        of = pl.when(
            pl.col("locked").fill_null(0.0).neq(0.0) | (~pl.col("volume").is_finite())
        ).then(0.0).otherwise(clean_v * (2.0 * p - 1.0))
    long = long.with_columns(
        of.alias("of"),
        clean_v.alias("vol_clean"),
    )
    daily = (
        long.group_by(["date", "instrument"])
        .agg(
            pl.col("of").sum().alias("num"),
            pl.col("vol_clean").sum().alias("den"),
        )
        .with_columns(
            pl.when(pl.col("den") > _EPS).then(pl.col("num") / pl.col("den")).otherwise(None).alias("v")
        )
    )
    return _pivot(daily, "v")


_mk(
    "intraday_bvc_imbalance",
    "BV-C 成交量分类买卖流不平衡 (sum V(2Phi(z)-1)/sum V)（Polars）。",
    ["close", "volume", "scale_window", "locked"],
    lambda close, volume, scale_window=20, locked=None: _bvc_imbalance_polars(
        close, volume, scale_window, locked
    ),
)


# ---------------------------------------------------------------------------
# Price-impact regression slope (and buy/sell asymmetry)
# ---------------------------------------------------------------------------

def _impact_slopes_polars(
    returns: pl.DataFrame, flow: pl.DataFrame, min_periods: int, side_min: int
) -> pl.DataFrame:
    """Return a daily frame with per-(date,instrument) impact slopes.

    Columns: ``lambda_all``, ``lambda_pos``, ``lambda_neg`` (each daily wide).
    """
    lr = _melt(returns, "r")
    lq = _melt(flow, "q")
    long = _with_date(lr.join(lq, on=["ts", "instrument"], how="inner")).drop_nulls(
        ["r", "q"]
    )
    long = long.filter(pl.col("r").is_finite() & pl.col("q").is_finite())

    allg = long.group_by(["date", "instrument"]).agg(
        pl.len().alias("n"),
        pl.cov(pl.col("r"), pl.col("q"), ddof=0).alias("cov"),
        pl.col("q").var(ddof=0).alias("var"),
    )
    allg = allg.with_columns(
        pl.when((pl.col("n") >= min_periods) & (pl.col("var") > _EPS))
        .then(pl.col("cov") / pl.col("var"))
        .otherwise(None)
        .alias("lambda_all")
    )
    pos = long.filter(pl.col("q") > 0.0).group_by(["date", "instrument"]).agg(
        pl.len().alias("np"),
        pl.cov(pl.col("r"), pl.col("q"), ddof=0).alias("covp"),
        pl.col("q").var(ddof=0).alias("varp"),
    )
    pos = pos.with_columns(
        pl.when((pl.col("np") >= side_min) & (pl.col("varp") > _EPS))
        .then(pl.col("covp") / pl.col("varp"))
        .otherwise(None)
        .alias("lambda_pos")
    )
    neg = long.filter(pl.col("q") < 0.0).group_by(["date", "instrument"]).agg(
        pl.len().alias("nn"),
        pl.cov(pl.col("r"), pl.col("q"), ddof=0).alias("covn"),
        pl.col("q").var(ddof=0).alias("varn"),
    )
    neg = neg.with_columns(
        pl.when((pl.col("nn") >= side_min) & (pl.col("varn") > _EPS))
        .then(pl.col("covn") / pl.col("varn"))
        .otherwise(None)
        .alias("lambda_neg")
    )
    joined = (
        allg.select(["date", "instrument", "lambda_all"])
        .join(
            pos.select(["date", "instrument", "lambda_pos"]),
            on=["date", "instrument"], how="full", coalesce=True,
        )
        .join(
            neg.select(["date", "instrument", "lambda_neg"]),
            on=["date", "instrument"], how="full", coalesce=True,
        )
    )
    return joined


_mk(
    "intraday_impact_beta",
    "日内价格冲击回归斜率 lambda (r = alpha + lambda*q)（Polars）。",
    ["returns", "flow", "min_periods"],
    lambda returns, flow, min_periods=20: _pivot(
        _impact_slopes_polars(returns, flow, max(5, int(min_periods)), 1),
        "lambda_all",
    ),
)


_mk(
    "intraday_impact_asymmetry",
    "买卖冲击不对称 (lambda_+ - |lambda_-|)/(|lambda_+|+|lambda_-|)（Polars）。",
    ["returns", "flow", "min_periods"],
    lambda returns, flow, min_periods=20: _pivot(
        _impact_slopes_polars(returns, flow, max(5, int(min_periods)), max(3, max(5, int(min_periods)) // 3)).with_columns(
            pl.when(
                pl.col("lambda_pos").is_not_null()
                & pl.col("lambda_neg").is_not_null()
            )
            .then(
                (pl.col("lambda_pos") - pl.col("lambda_neg").abs())
                / (pl.col("lambda_pos").abs() + pl.col("lambda_neg").abs() + _EPS)
            )
            .otherwise(None)
            .alias("v")
        ),
        "v",
    ),
)


# ---------------------------------------------------------------------------
# Return-distribution Wasserstein shift (cross-day history): map_batches kernel.
# ---------------------------------------------------------------------------

def _wasserstein_shift_polars(returns: pl.DataFrame, lookback_days: int) -> pl.DataFrame:
    from cleaned_operators.microstructure.flow_impact import _wasserstein_shift_series

    lb = max(2, int(lookback_days))
    lr = _melt(returns, "r")
    long = _with_date(lr).filter(pl.col("r").is_finite())
    daily = (
        long.group_by(["date", "instrument"])
        .agg(pl.col("r").drop_nulls().alias("rets"))
        .sort(["instrument", "date"])
    )
    wide = daily.pivot(index="date", on="instrument", values="rets", aggregate_function="first")
    out = {}
    for inst in wide.columns:
        if inst == "date":
            continue
        series = wide.select(pl.col(inst)).to_series()
        vals = _wasserstein_shift_series(series.to_list(), lb)
        out[inst] = vals
    dates = wide["date"].to_list()
    return pl.DataFrame({**{"date": dates}, **out}).fill_null(float("nan"))


_mk(
    "intraday_return_wasserstein_shift",
    "当日分钟收益分布相对过去 N 日的 W1 距离(MAD 标准化)（Polars）。",
    ["returns", "lookback_days"],
    lambda returns, lookback_days=10: _wasserstein_shift_polars(returns, lookback_days),
)


# ---------------------------------------------------------------------------
# VPIN over equal-volume buckets: map_batches kernel.
# ---------------------------------------------------------------------------

def _bvc_vpin_polars(
    close: pl.DataFrame, volume: pl.DataFrame, scale_window: int, bucket_count: int
) -> pl.DataFrame:
    from cleaned_operators.microstructure.flow_impact import _vpin_series

    sw = max(2, int(scale_window))
    buckets = max(2, int(bucket_count))
    lc = _melt(close, "close")
    lv = _melt(volume, "volume")
    # Keep every bar aligned (NaN close / NaN volume stay in place) so the
    # shared numpy kernel sees exactly the arrays the pandas reference sees.
    long = _with_date(lc.join(lv, on=["ts", "instrument"], how="inner")).sort(
        ["date", "instrument", "ts"]
    )
    daily = long.group_by(["date", "instrument"]).agg(
        pl.col("close").alias("close_l"),
        pl.col("volume").alias("volume_l"),
    )
    wide_c = daily.pivot(index="date", on="instrument", values="close_l", aggregate_function="first")
    wide_v = daily.pivot(index="date", on="instrument", values="volume_l", aggregate_function="first")
    out = {}
    for inst in wide_c.columns:
        if inst == "date":
            continue
        closes = wide_c.select(pl.col(inst)).to_series().to_list()
        volumes = wide_v.select(pl.col(inst)).to_series().to_list()
        out[inst] = _vpin_series(closes, volumes, sw, buckets)
    dates = wide_c["date"].to_list()
    return pl.DataFrame({**{"date": dates}, **out}).fill_null(float("nan"))


_mk(
    "micro_bvc_vpin",
    "BV-C 等量桶 VPIN (sum|OF_b|/total_volume), 边界分钟按量切分（Polars）。",
    ["close", "volume", "scale_window", "bucket_count"],
    lambda close, volume, scale_window=40, bucket_count=20: _bvc_vpin_polars(
        close, volume, scale_window, bucket_count
    ),
)


# Ensure the polars slots stay registered even when the module is re-imported
# after the pandas module already registered its surface.
def _register_surface() -> None:
    pass


_register_surface()
