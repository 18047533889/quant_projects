# -*- coding: utf-8 -*-
"""Polars backends for next-stage intraday operators (minute -> daily).

Genuine ``pl.Expr`` implementations: a minute wide panel is melted to long
format, aggregated per (date, instrument), and pivoted back to a daily wide
panel.  No pandas delegation.
"""
from __future__ import annotations

import math
from typing import Any

import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

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
    return long.with_columns(pl.col(tc).cast(pl.Datetime).alias("ts"))


def _returns(long: pl.DataFrame, close_col: str = "close") -> pl.DataFrame:
    return long.with_columns(
        (pl.col(close_col) / pl.col(close_col).shift(1).over(["date", "instrument"]))
        .log()
        .alias("r")
    )


def _pivot(df: pl.DataFrame, value: str) -> pl.DataFrame:
    piv = df.pivot(index="date", on="instrument", values=value)
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
        f"IntradayPolars_{canonical}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=canonical,
        category="intraday_microstructure",
        business_category="intraday_microstructure",
        canonical=canonical,
        source="intraday.polars_next_stage",
    )(cls)
    return cls


# ---------------------------------------------------------------------------
# § Realized moments
# ---------------------------------------------------------------------------

def _realized_moment(close: pl.DataFrame, kind: str) -> pl.DataFrame:
    long = _melt(close, "close")
    long = long.with_columns(pl.col("ts").dt.date().alias("date"))
    long = _returns(long)
    r = pl.col("r")
    agg = long.group_by(["date", "instrument"]).agg(
        pl.col("r").count().alias("n"),
        (r * r).sum().alias("r2"),
        (r * r * r).sum().alias("r3"),
        (r * r * r * r).sum().alias("r4"),
    )
    if kind == "skewness":
        out = agg.with_columns(((pl.col("r3") * pl.col("n").sqrt()) / pl.col("r2").pow(1.5)).alias("v"))
    elif kind == "kurtosis":
        out = agg.with_columns(((pl.col("r4") * pl.col("n")) / (pl.col("r2") * pl.col("r2"))).alias("v"))
    else:  # quarticity
        out = agg.with_columns(((pl.col("r4") * pl.col("n")) / 3.0).alias("v"))
    return _pivot(out, "v")


_mk("intra_realized_skewness", "日内已实现偏度（Polars）。", ["close"],
   lambda close: _realized_moment(close, "skewness"))
_mk("intra_realized_kurtosis", "日内已实现峰度（Polars）。", ["close"],
   lambda close: _realized_moment(close, "kurtosis"))
_mk("intra_realized_quarticity", "日内已实现四次变差（Polars）。", ["close"],
   lambda close: _realized_moment(close, "quarticity"))


# ---------------------------------------------------------------------------
# § Continuous / jump variance
# ---------------------------------------------------------------------------

def _cont_jump(close: pl.DataFrame, side: str) -> pl.DataFrame:
    long = _melt(close, "close")
    long = long.with_columns(pl.col("ts").dt.date().alias("date"))
    long = _returns(long)
    r = pl.col("r")
    long = long.with_columns(
        (r.abs() * r.abs().shift(1).over(["date", "instrument"])).alias("absprod")
    )
    agg = long.group_by(["date", "instrument"]).agg(
        (r * r).sum().alias("rv"),
        ((pl.col("absprod") * (math.pi / 2.0)).sum()).alias("bv"),
    )
    if side == "continuous":
        out = agg.with_columns(pl.min_horizontal("rv", "bv").alias("v"))
    else:
        out = agg.with_columns(((pl.col("rv") - pl.col("bv")).clip(lower_bound=0.0)).alias("v"))
    return _pivot(out, "v")


_mk("intra_continuous_variance", "日内连续方差 min(RV,BV)（Polars）。", ["close"],
   lambda close: _cont_jump(close, "continuous"))
_mk("intra_jump_variation", "日内跳跃方差 max(RV-BV,0)（Polars）。", ["close"],
   lambda close: _cont_jump(close, "jump"))


def _jump_stats(close: pl.DataFrame, stat: str, threshold_scale: float) -> pl.DataFrame:
    long = _melt(close, "close")
    long = long.with_columns(pl.col("ts").dt.date().alias("date"))
    long = _returns(long)
    r = pl.col("r")
    rv = long.group_by(["date", "instrument"]).agg(
        pl.col("r").count().alias("n"),
        (r * r).sum().alias("rv"),
    ).with_columns(
        ((pl.col("rv") / pl.col("n")).sqrt() * float(threshold_scale)).alias("thresh")
    )
    long = long.join(rv, on=["date", "instrument"])
    long = long.with_columns((r.abs() > pl.col("thresh")).alias("jump"))
    jr = pl.col("r")
    stats = long.group_by(["date", "instrument"]).agg(
        pl.col("jump").sum().alias("count"),
        ((jr * jr) * pl.col("jump") * (jr > 0)).sum().alias("pos"),
        ((jr * jr) * pl.col("jump") * (jr < 0)).sum().alias("neg"),
        ((jr * jr) * pl.col("jump")).sum().alias("total"),
        ((jr * jr) * pl.col("jump")).pow(2).sum().alias("conc_num"),
    )
    has_jump = pl.col("count") > 0
    if stat == "count":
        out = stats.select(["date", "instrument", pl.col("count").alias("v")])
    elif stat == "pos":
        out = stats.with_columns(
            pl.when(has_jump).then(pl.col("pos")).otherwise(None).alias("v")
        ).select(["date", "instrument", "v"])
    elif stat == "neg":
        out = stats.with_columns(
            pl.when(has_jump).then(pl.col("neg")).otherwise(None).alias("v")
        ).select(["date", "instrument", "v"])
    elif stat == "signed_ratio":
        out = stats.with_columns(
            pl.when(has_jump)
            .then((pl.col("pos") - pl.col("neg")) / (pl.col("pos") + pl.col("neg") + _EPS))
            .otherwise(None)
            .alias("v")
        ).select(["date", "instrument", "v"])
    else:  # concentration
        out = stats.with_columns(
            pl.when(has_jump)
            .then(pl.col("conc_num") / (pl.col("total") * pl.col("total")))
            .otherwise(None)
            .alias("v")
        ).select(["date", "instrument", "v"])
    return _pivot(out, "v")


_mk("intra_positive_jump_variation", "日内正跳跃方差（Polars）。", ["close", "threshold_scale"],
   lambda close, threshold_scale=3.0: _jump_stats(close, "pos", float(threshold_scale)))
_mk("intra_negative_jump_variation", "日内负跳跃方差（Polars）。", ["close", "threshold_scale"],
   lambda close, threshold_scale=3.0: _jump_stats(close, "neg", float(threshold_scale)))
_mk("intra_signed_jump_ratio", "有符号跳跃比（Polars）。", ["close", "threshold_scale"],
   lambda close, threshold_scale=3.0: _jump_stats(close, "signed_ratio", float(threshold_scale)))
_mk("intra_jump_count", "日内跳跃次数（Polars）。", ["close", "threshold_scale"],
   lambda close, threshold_scale=3.0: _jump_stats(close, "count", float(threshold_scale)))
_mk("intra_jump_concentration", "跳跃集中度（Polars）。", ["close", "threshold_scale"],
   lambda close, threshold_scale=3.0: _jump_stats(close, "concentration", float(threshold_scale)))


# ---------------------------------------------------------------------------
# § Interval return / volume share
# ---------------------------------------------------------------------------

def _interval_ret(close: pl.DataFrame, start: int, end: int) -> pl.DataFrame:
    long = _melt(close, "close")
    long = long.with_columns(
        pl.col("ts").dt.date().alias("date"),
        (pl.col("ts").dt.hour() * 60 + pl.col("ts").dt.minute()).alias("mod"),
    )
    mask = (pl.col("mod") >= int(start)) & (pl.col("mod") <= int(end))
    seg = long.filter(mask).group_by(["date", "instrument"]).agg(
        pl.col("close").first().alias("open"),
        pl.col("close").last().alias("last"),
    )
    out = seg.with_columns(((pl.col("last") / pl.col("open")) - 1.0).alias("v"))
    return _pivot(out, "v")


_mk("intra_interval_return", "指定分钟区间收益（Polars）。", ["close", "start_minute", "end_minute"],
   lambda close, start_minute=570, end_minute=900: _interval_ret(close, int(start_minute), int(end_minute)))


def _interval_share(value: pl.DataFrame, start: int, end: int) -> pl.DataFrame:
    long = _melt(value, "value")
    long = long.with_columns(
        pl.col("ts").dt.date().alias("date"),
        (pl.col("ts").dt.hour() * 60 + pl.col("ts").dt.minute()).alias("mod"),
    )
    total = long.group_by(["date", "instrument"]).agg(pl.col("value").sum().alias("total"))
    mask = (pl.col("mod") >= int(start)) & (pl.col("mod") <= int(end))
    seg = long.filter(mask).group_by(["date", "instrument"]).agg(pl.col("value").sum().alias("seg"))
    merged = seg.join(total, on=["date", "instrument"])
    out = merged.with_columns((pl.col("seg") / pl.col("total")).alias("v"))
    return _pivot(out, "v")


_mk("intra_interval_volume_share", "区间成交量占比（Polars）。", ["volume", "start_minute", "end_minute"],
   lambda volume, start_minute=570, end_minute=900: _interval_share(volume, int(start_minute), int(end_minute)))
_mk("intra_interval_amount_share", "区间成交额占比（Polars）。", ["amount", "start_minute", "end_minute"],
   lambda amount, start_minute=570, end_minute=900: _interval_share(amount, int(start_minute), int(end_minute)))


# ---------------------------------------------------------------------------
# § Drawdown / drawup
# ---------------------------------------------------------------------------

def _max_path(close: pl.DataFrame, side: str) -> pl.DataFrame:
    long = _melt(close, "close")
    long = long.with_columns(pl.col("ts").dt.date().alias("date"))
    long = long.with_columns(
        pl.col("close").cum_max().over(["date", "instrument"]).alias("run_max"),
        pl.col("close").cum_min().over(["date", "instrument"]).alias("run_min"),
    )
    if side == "down":
        out = long.group_by(["date", "instrument"]).agg(
            ((pl.col("close") / pl.col("run_max")) - 1.0).min().alias("v")
        )
    else:
        out = long.group_by(["date", "instrument"]).agg(
            ((pl.col("close") / pl.col("run_min")) - 1.0).max().alias("v")
        )
    return _pivot(out, "v")


_mk("intra_max_drawdown", "日内最大回撤（Polars）。", ["close"],
   lambda close: _max_path(close, "down"))
_mk("intra_max_drawup", "日内最大上涨段（Polars）。", ["close"],
   lambda close: _max_path(close, "up"))
