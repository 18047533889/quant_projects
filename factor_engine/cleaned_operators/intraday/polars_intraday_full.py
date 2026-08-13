# -*- coding: utf-8 -*-
"""Genuine Polars backends for the remaining ``intra_*`` minute→daily operators.

Completes the second wave of intraday Polars coverage on top of
``polars_next_stage.py`` (realized moments, jump variance, interval shares,
max drawdown/drawup).  Every operator here is a real ``pl.Expr`` implementation:
a minute wide panel is melted to long format, aggregated per (date, instrument),
and pivoted back to a daily wide panel.  No pandas delegation.

Families covered:

* segment aggregation (return / volume / amount share / vwap deviation / realized vol)
* realized variance / semivariance / bipower / jump-ratio
* price path (efficiency, high/low time, VWAP excursions, streaks, reversion, path slope/curvature)
* volume/amount distribution (concentration, entropy, signed imbalance)
* liquidity (Amihud, Kyle lambda), extreme bar, lunch gap
* daily-limit behaviour (first-hit time, duration, reopen count)
* cross-day slot memory and profile distances (same-slot, cosine / JSD / EMD)
* realized market beta family (market return built from the value-weighted cross-section)
"""
from __future__ import annotations

import math

import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.intraday._core import tripower_scale as _tripower_scale

_TIME_COLS = frozenset({"date", "timestamp", "time", "QuoteTime", "TradeDate"})
_EPS = 1e-12
_SESSION_TZ = "Asia/Shanghai"
_SEGMENT_RANGES = {"morning": (570, 690), "afternoon": (780, 900)}


def _time_col(df: pl.DataFrame) -> str:
    for c in df.columns:
        if c in _TIME_COLS:
            return c
    return "date"


def _ts_expr(tc: str, df: pl.DataFrame) -> pl.Expr:
    """Datetime column cast; tz-aware inputs are converted to session wall-clock."""
    dtype = df.schema[tc]
    if getattr(dtype, "time_zone", None):
        return (
            pl.col(tc)
            .cast(pl.Datetime)
            .dt.convert_time_zone(_SESSION_TZ)
            .dt.replace_time_zone(None)
        )
    return pl.col(tc).cast(pl.Datetime)


def _melt(df: pl.DataFrame, value_name: str) -> pl.DataFrame:
    tc = _time_col(df)
    cols = [c for c in df.columns if c != tc]
    long = df.unpivot(index=[tc], on=cols, variable_name="instrument", value_name=value_name)
    long = long.with_columns(_ts_expr(tc, df).alias("ts"))
    return long.drop(tc)


def _melt_daily(df: pl.DataFrame, value_name: str) -> pl.DataFrame:
    tc = _time_col(df)
    cols = [c for c in df.columns if c != tc]
    long = df.unpivot(index=[tc], on=cols, variable_name="instrument", value_name=value_name)
    return long.with_columns(pl.col(tc).cast(pl.Date).alias("date"))


def _with_date(long: pl.DataFrame) -> pl.DataFrame:
    return long.with_columns(pl.col("ts").dt.date().alias("date"))


def _with_mod(long: pl.DataFrame) -> pl.DataFrame:
    return long.with_columns(
        (pl.col("ts").dt.hour().cast(pl.Int64) * 60 + pl.col("ts").dt.minute().cast(pl.Int64)).alias("mod")
    )


def _pivot(df: pl.DataFrame, value: str) -> pl.DataFrame:
    piv = df.pivot(index="date", on="instrument", values=value)
    return piv.fill_null(float("nan"))


def _mk(canonical: str, description: str, params: list[str], fn, extra_tags=None):
    tags = ["polars", "intraday", "minute", "native", "typed_v2"]
    if extra_tags:
        tags.extend(extra_tags)
    metadata = OperatorMetadata(
        name=canonical,
        category="intraday_microstructure",
        description=description,
        param_names=params,
        return_type="series",
        tags=tags,
    )

    def _calculate_series(self, *args, **kwargs):
        return fn(*args, **kwargs)

    cls = type(
        f"IntradayPolarsFull_{canonical}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=canonical,
        category="intraday_microstructure",
        business_category="intraday_microstructure",
        canonical=canonical,
        source="intraday.polars_intraday_full",
    )(cls)
    return cls


# ---------------------------------------------------------------------------
# § Shared log-return helper (within day)
# ---------------------------------------------------------------------------

def _log_returns_long(long: pl.DataFrame) -> pl.DataFrame:
    """r_t = log(close_t / close_{t-1}) within (date, instrument), order by ts."""
    long = long.sort(["date", "instrument", "ts"]).with_columns(
        (pl.col("close") / pl.col("close").shift(1).over(["date", "instrument"])).log().alias("r")
    )
    return long


def _cum_vwap_long(long: pl.DataFrame) -> pl.DataFrame:
    """Cumulative VWAP over finite-close bars within the day (matches pandas)."""
    long = long.filter(pl.col("close").is_finite())
    long = long.sort(["date", "instrument", "ts"]).with_columns(
        pl.col("volume").fill_nan(0.0).cum_sum().over(["date", "instrument"]).alias("cum_v"),
        pl.col("amount").fill_nan(0.0).cum_sum().over(["date", "instrument"]).alias("cum_a"),
    )
    return long.with_columns(
        pl.when(pl.col("cum_v") > _EPS).then(pl.col("cum_a") / pl.col("cum_v")).otherwise(None).alias("cum_vwap")
    )


# ---------------------------------------------------------------------------
# § Segment aggregation
# ---------------------------------------------------------------------------

def _segment_mask(mod_expr: pl.Expr, segment: str) -> pl.Expr:
    lo, hi = _SEGMENT_RANGES[str(segment)]
    return (mod_expr >= lo) & (mod_expr <= hi)


def _seg_return(close: pl.DataFrame, segment: str) -> pl.DataFrame:
    long = _with_mod(_with_date(_melt(close, "close"))).filter(pl.col("close").is_finite())
    seg = long.filter(_segment_mask(pl.col("mod"), segment))
    seg = seg.sort(["date", "instrument", "ts"]).group_by(["date", "instrument"]).agg(
        pl.col("close").first().alias("open"),
        pl.col("close").last().alias("last"),
    )
    out = pl.when(pl.col("open")) - 1.0).alias("v")) != 0).then((seg.with_columns(((pl.col("last")) / (pl.col("open")) - 1.0).alias("v")))).otherwise(None)
    out = out.with_columns(pl.when(pl.col("last").is_not_null() & pl.col("open").is_not_null()).then(pl.col("v")).otherwise(None))
    return _pivot(out, "v")


_mk("intra_segment_return", "指定时段（morning/afternoon）收盘/开盘收益 - 1（Polars）。", ["close", "segment", "session_tz", "endpoint_policy"],
   lambda close, segment="morning", session_tz=None, endpoint_policy="exact": _seg_return(close, segment))


def _seg_share(value: pl.DataFrame, segment: str) -> pl.DataFrame:
    long = _with_mod(_with_date(_melt(value, "value"))).with_columns(pl.col("value").fill_nan(0.0))
    total = long.group_by(["date", "instrument"]).agg(pl.col("value").sum().alias("total"))
    seg = long.filter(_segment_mask(pl.col("mod"), segment)).group_by(["date", "instrument"]).agg(
        pl.col("value").sum().alias("seg")
    )
    merged = seg.join(total, on=["date", "instrument"])
    out = merged.with_columns(
        pl.when(pl.col("total") > _EPS).then(pl.col("seg") / pl.col("total")).otherwise(None).alias("v")
    )
    return _pivot(out, "v")


_mk("intra_segment_volume_share", "指定时段成交量占全天比例（Polars）。", ["volume", "segment", "session_tz"],
   lambda volume, segment="morning": _seg_share(volume, segment))
_mk("intra_segment_amount_share", "指定时段成交额占全天比例（Polars）。", ["amount", "segment", "session_tz"],
   lambda amount, segment="morning": _seg_share(amount, segment))


def _seg_vwap_dev(close: pl.DataFrame, amount: pl.DataFrame, volume: pl.DataFrame, segment: str) -> pl.DataFrame:
    long = _with_mod(_with_date(
        _melt(close, "close")
        .join(_melt(amount, "amount"), on=["ts", "instrument"], how="left")
        .join(_melt(volume, "volume"), on=["ts", "instrument"], how="left")
    ))
    seg = long.filter(_segment_mask(pl.col("mod"), segment))
    seg = seg.sort(["date", "instrument", "ts"]).group_by(["date", "instrument"]).agg(
        pl.col("amount").fill_nan(0.0).sum().alias("a"),
        pl.col("volume").fill_nan(0.0).sum().alias("v"),
        pl.col("close").filter(pl.col("close").is_finite()).last().alias("clast"),
    )
    out = seg.with_columns(
        pl.when(pl.col("v") > _EPS)
        .then(pl.col("clast") / (pl.col("a") / pl.col("v")) - 1.0)
        .otherwise(None)
        .alias("vout")
    )
    return _pivot(out.select(["date", "instrument", pl.col("vout").alias("v")]), "v")


_mk("intra_segment_vwap_deviation", "指定时段末价相对该时段累计 VWAP 的偏差（Polars）。",
   ["close", "amount", "volume", "segment", "session_tz"],
   lambda close, amount, volume, segment="morning", session_tz=None: _seg_vwap_dev(close, amount, volume, segment))


def _seg_realized_vol(close: pl.DataFrame, segment: str) -> pl.DataFrame:
    long = _with_mod(_with_date(_melt(close, "close"))).filter(pl.col("close").is_finite())
    long = long.filter(_segment_mask(pl.col("mod"), segment))
    long = _log_returns_long(long)
    out = long.group_by(["date", "instrument"]).agg(
        (pl.col("r") * pl.col("r")).fill_nan(0.0).sum().sqrt().alias("v")
    )
    return _pivot(out, "v")


_mk("intra_segment_realized_vol", "指定时段已实现波动率 sqrt(sum(r_t^2))（Polars）。", ["close", "segment", "session_tz"],
   lambda close, segment="morning", session_tz=None: _seg_realized_vol(close, segment))


# ---------------------------------------------------------------------------
# § Realized variance / semivariance / bipower / jump-ratio
# ---------------------------------------------------------------------------

def _rv(close: pl.DataFrame) -> pl.DataFrame:
    long = _with_date(_melt(close, "close")).filter(pl.col("close").is_finite())
    long = _log_returns_long(long)
    out = long.group_by(["date", "instrument"]).agg(
        (pl.col("r") * pl.col("r")).fill_nan(0.0).sum().alias("v")
    )
    return _pivot(out, "v")


_mk("intra_realized_variance", "日内已实现方差 sum(r_t^2)（Polars）。", ["close"],
   lambda close: _rv(close))


def _semivariance(close: pl.DataFrame, side: str) -> pl.DataFrame:
    long = _with_date(_melt(close, "close")).filter(pl.col("close").is_finite())
    long = _log_returns_long(long)
    r = pl.col("r")
    if side == "down":
        term = pl.when(r < 0).then(r * r).otherwise(None)
    elif side == "up":
        term = pl.when(r > 0).then(r * r).otherwise(None)
    else:
        term = r * r
    out = long.group_by(["date", "instrument"]).agg(term.sum().alias("v"))
    return _pivot(out, "v")


_mk("intra_realized_semivariance", "日内上/下半方差 sum(r_t^2 * 1(sign))（Polars）。", ["close", "side"],
   lambda close, side="down": _semivariance(close, side))


def _bipower(close: pl.DataFrame) -> pl.DataFrame:
    long = _with_date(_melt(close, "close")).filter(pl.col("close").is_finite())
    long = _log_returns_long(long).filter(pl.col("r").is_finite())
    long = long.sort(["date", "instrument", "ts"]).with_columns(
        (pl.col("r").abs() * pl.col("r").abs().shift(1).over(["date", "instrument"])).alias("absprod")
    )
    out = long.group_by(["date", "instrument"]).agg(
        ((pl.col("absprod") * (math.pi / 2.0)).fill_nan(0.0).sum()).alias("v")
    )
    return _pivot(out, "v")


_mk("intra_bipower_variation", "日内双幂变差 (pi/2)*sum(|r_t||r_{t-1}|)（Polars）。", ["close"],
   lambda close: _bipower(close))


def _jump_ratio(close: pl.DataFrame) -> pl.DataFrame:
    long = _with_date(_melt(close, "close")).filter(pl.col("close").is_finite())
    long = _log_returns_long(long).filter(pl.col("r").is_finite())
    long = long.sort(["date", "instrument", "ts"]).with_columns(
        (pl.col("r").abs() * pl.col("r").abs().shift(1).over(["date", "instrument"])).alias("absprod")
    )
    out = long.group_by(["date", "instrument"]).agg(
        (pl.col("r") * pl.col("r")).fill_nan(0.0).sum().alias("rv"),
        ((pl.col("absprod") * (math.pi / 2.0)).fill_nan(0.0).sum()).alias("bv"),
    ).with_columns(
        pl.when((pl.col("rv") > _EPS) & pl.col("rv").is_finite() & pl.col("bv").is_finite())
        .then(((pl.col("rv") - pl.col("bv")).clip(lower_bound=0.0)) / pl.col("rv"))
        .otherwise(None)
        .alias("v")
    )
    return _pivot(out, "v")


_mk("intra_jump_ratio", "日内跳跃占比 max(RV-BV,0)/RV（Polars）。", ["close"],
   lambda close: _jump_ratio(close))


# ---------------------------------------------------------------------------
# § Price path: efficiency, high/low time, excursions, streaks, reversion
# ---------------------------------------------------------------------------

def _path_efficiency(close: pl.DataFrame) -> pl.DataFrame:
    long = _with_date(_melt(close, "close")).filter(pl.col("close").is_finite())
    long = long.sort(["date", "instrument", "ts"]).with_columns(
        (pl.col("close") - pl.col("close").shift(1).over(["date", "instrument"])).alias("chg")
    )
    out = long.group_by(["date", "instrument"]).agg(
        pl.col("close").first().alias("first"),
        pl.col("close").last().alias("last"),
        pl.col("chg").fill_nan(0.0).abs().sum().alias("length"),
        pl.col("close").count().alias("n"),
    ).with_columns(
        pl.when((pl.col("n") >= 2) & (pl.col("length") > _EPS))
        .then((pl.col("last") - pl.col("first")).abs() / pl.col("length"))
        .otherwise(None)
        .alias("v")
    )
    return _pivot(out, "v")


_mk("intra_path_efficiency", "日内路径效率 |净位移|/路径长度（Polars）。", ["close"],
   lambda close: _path_efficiency(close))


def _position_of(series: pl.DataFrame, low: bool) -> pl.DataFrame:
    long = _with_date(_melt(series, "value")).filter(pl.col("value").is_finite())
    long = long.sort(["date", "instrument", "ts"])
    if low:
        pos = pl.col("value").arg_min()
    else:
        pos = pl.col("value").arg_max()
    out = long.group_by(["date", "instrument"]).agg(
        pos.alias("pos"),
        pl.col("value").count().alias("n"),
    ).with_columns(pl.when(pl.col("n") > 0).then(pl.col("pos") / pl.col("n")).otherwise(None).alias("v"))
    return _pivot(out, "v")


_mk("intra_high_time", "全天最高价首次出现位置 / 有效分钟数（Polars）。", ["high"],
   lambda high: _position_of(high, low=False))
_mk("intra_low_time", "全天最低价首次出现位置 / 有效分钟数（Polars）。", ["low"],
   lambda low: _position_of(low, low=True))


def _vwap_excursion(close: pl.DataFrame, amount: pl.DataFrame, volume: pl.DataFrame, side: str) -> pl.DataFrame:
    long = _with_date(
        _melt(close, "close")
        .join(_melt(amount, "amount"), on=["ts", "instrument"], how="left")
        .join(_melt(volume, "volume"), on=["ts", "instrument"], how="left")
    )
    long = _cum_vwap_long(long)
    long = long.with_columns(
        (pl.col("close") / pl.col("cum_vwap") - 1.0).alias("dev"),
        (pl.col("cum_vwap") > _EPS).fill_null(False).alias("ok"),
    )
    long = long.filter(pl.col("ok"))
    agg = long.group_by(["date", "instrument"]).agg(
        pl.col("dev").max().alias("vmax") if side == "max" else pl.col("dev").min().alias("vmin"),
    )
    value = "vmax" if side == "max" else "vmin"
    return _pivot(agg, value)


_mk("intra_price_vwap_max_positive_excursion", "相对累计 VWAP 最大正偏离（Polars）。", ["close", "amount", "volume"],
   lambda close, amount, volume: _vwap_excursion(close, amount, volume, "max"))
_mk("intra_price_vwap_max_negative_excursion", "相对累计 VWAP 最大负偏离（Polars）。", ["close", "amount", "volume"],
   lambda close, amount, volume: _vwap_excursion(close, amount, volume, "min"))


def _time_above_vwap(close: pl.DataFrame, amount: pl.DataFrame, volume: pl.DataFrame) -> pl.DataFrame:
    long = _with_date(
        _melt(close, "close")
        .join(_melt(amount, "amount"), on=["ts", "instrument"], how="left")
        .join(_melt(volume, "volume"), on=["ts", "instrument"], how="left")
    )
    long = _cum_vwap_long(long)
    long = long.with_columns(
        (pl.col("close") > pl.col("cum_vwap")).fill_null(False).alias("above")
    )
    out = long.group_by(["date", "instrument"]).agg(
        pl.when(pl.col("cum_vwap").is_not_null()).then(pl.col("above")).otherwise(None).mean().alias("v")
    )
    return _pivot(out, "v")


_mk("intra_time_above_vwap", "高于累计 VWAP 的分钟比例（Polars）。", ["close", "amount", "volume"],
   lambda close, amount, volume: _time_above_vwap(close, amount, volume))


def _longest_streak(close: pl.DataFrame, amount: pl.DataFrame, volume: pl.DataFrame, side: str) -> pl.DataFrame:
    long = _with_date(
        _melt(close, "close")
        .join(_melt(amount, "amount"), on=["ts", "instrument"], how="left")
        .join(_melt(volume, "volume"), on=["ts", "instrument"], how="left")
    )
    long = _cum_vwap_long(long)
    long = long.filter(pl.col("cum_vwap").is_not_null())
    above = pl.col("close") > pl.col("cum_vwap")
    flag = (~above) if side == "below" else above
    long = long.with_columns(flag.fill_null(False).alias("flag"))
    long = long.sort(["date", "instrument", "ts"]).with_columns(
        (pl.col("flag") != pl.col("flag").shift(1).fill_null(False)).cum_sum().over(["date", "instrument"]).alias("run_id")
    )
    # Only True segments count (pandas resets the counter on a False flag).
    runs = long.filter(pl.col("flag")).group_by(["date", "instrument", "run_id"]).agg(pl.len().alias("cnt"))
    out = runs.group_by(["date", "instrument"]).agg(pl.col("cnt").max().alias("v"))
    return _pivot(out, "v")


_mk("intra_longest_above_vwap_streak", "高于 VWAP 最长连续分钟数（Polars）。", ["close", "amount", "volume"],
   lambda close, amount, volume: _longest_streak(close, amount, volume, "above"))
_mk("intra_longest_below_vwap_streak", "低于 VWAP 最长连续分钟数（Polars）。", ["close", "amount", "volume"],
   lambda close, amount, volume: _longest_streak(close, amount, volume, "below"))


def _vwap_reversion_speed(close: pl.DataFrame, amount: pl.DataFrame, volume: pl.DataFrame) -> pl.DataFrame:
    long = _with_date(
        _melt(close, "close")
        .join(_melt(amount, "amount"), on=["ts", "instrument"], how="left")
        .join(_melt(volume, "volume"), on=["ts", "instrument"], how="left")
    )
    long = _cum_vwap_long(long).filter(pl.col("cum_vwap").is_not_null())
    long = long.sort(["date", "instrument", "ts"]).with_columns(
        (pl.col("close") / pl.col("cum_vwap") - 1.0).alias("d")
    ).with_columns(
        pl.col("d").shift(1).over(["date", "instrument"]).alias("dprev")
    ).filter(pl.col("dprev").is_not_null() & pl.col("d").is_not_null())
    g = long.group_by(["date", "instrument"]).agg(
        pl.col("d").count().alias("n"),
        (pl.col("d")).sum().alias("sd"),
        (pl.col("dprev")).sum().alias("sdp"),
        (pl.col("d") * pl.col("dprev")).sum().alias("sdpd"),
        (pl.col("dprev") * pl.col("dprev")).sum().alias("sdp2"),
    ).with_columns(
        pl.col("n").cast(pl.Float64).alias("nf")
    ).with_columns(
        ((pl.col("nf") * (pl.col("sdpd") - pl.col("sd") * pl.col("sdp") / pl.col("nf")))
         / ((pl.col("nf") - 1.0) * (pl.col("sdp2") - pl.col("sdp") * pl.col("sdp") / pl.col("nf")))).alias("beta")
    )
    out = g.with_columns(
        pl.when(
            (pl.col("n") >= 3)
            & ((pl.col("sdp2") - pl.col("sdp") * pl.col("sdp") / pl.col("nf")) / pl.col("nf") > _EPS)
        ).then(pl.col("beta")).otherwise(None).alias("v")
    )
    return _pivot(out, "v")


_mk("intra_vwap_reversion_speed", "VWAP 偏离 AR(1) 系数（Polars）。", ["close", "amount", "volume"],
   lambda close, amount, volume: _vwap_reversion_speed(close, amount, volume))


def _vwap_path_common(close: pl.DataFrame, amount: pl.DataFrame, volume: pl.DataFrame, degree: int, coeff_idx: int) -> pl.DataFrame:
    long = _with_date(
        _melt(close, "close")
        .join(_melt(amount, "amount"), on=["ts", "instrument"], how="left")
        .join(_melt(volume, "volume"), on=["ts", "instrument"], how="left")
    )
    long = _cum_vwap_long(long).filter(pl.col("cum_vwap").is_finite())
    long = long.sort(["date", "instrument", "ts"]).with_columns(
        pl.col("cum_vwap").cum_count().over(["date", "instrument"]).cast(pl.Float64).alias("idx1")
    )
    g = long.group_by(["date", "instrument"]).agg(
        pl.col("cum_vwap").count().alias("n"),
        pl.col("idx1").first().alias("one"),
    ).with_columns(pl.col("n").cast(pl.Float64).alias("nf"))
    # t = (idx - 1) / (n - 1); sums are computed from idx via t = (idx1-1)/(n-1)
    # Use sums of t, t^2..t^4 and t^k*y for k=0..2.
    g2 = long.join(g, on=["date", "instrument"]).with_columns(
        ((pl.col("idx1") - 1.0) / (pl.col("nf") - 1.0)).alias("t"),
        pl.col("cum_vwap").alias("y"),
    ).group_by(["date", "instrument"]).agg(
        pl.col("y").count().alias("n"),
        (pl.col("t")).sum().alias("S1"),
        (pl.col("t") * pl.col("t")).sum().alias("S2"),
        (pl.col("t").pow(3)).sum().alias("S3"),
        (pl.col("t").pow(4)).sum().alias("S4"),
        (pl.col("y")).sum().alias("S0y"),
        (pl.col("t") * pl.col("y")).sum().alias("S1y"),
        (pl.col("t").pow(2) * pl.col("y")).sum().alias("S2y"),
    )
    if degree == 1:
        out = g2.with_columns(
            ((pl.col("S1y") - 0.5 * pl.col("S0y"))
             / (pl.col("n").cast(pl.Float64) * (pl.col("n").cast(pl.Float64) + 1.0) / (12.0 * (pl.col("n").cast(pl.Float64) - 1.0))))
            .alias("v")
        )
        out = out.with_columns(
            pl.when(pl.col("n") >= 3).then(pl.col("v")).otherwise(None)
        )
    else:
        # beta2 = det(A)/det(X'X), A = X'X with last column replaced by X'y.
        nf = pl.col("n").cast(pl.Float64)
        detXX = (
            nf * (pl.col("S2") * pl.col("S4") - pl.col("S3") * pl.col("S3"))
            - pl.col("S1") * (pl.col("S1") * pl.col("S4") - pl.col("S3") * pl.col("S2"))
            + pl.col("S2") * (pl.col("S1") * pl.col("S3") - pl.col("S2") * pl.col("S2"))
        )
        detA = (
            nf * (pl.col("S2") * pl.col("S2y") - pl.col("S1y") * pl.col("S3"))
            - pl.col("S1") * (pl.col("S1") * pl.col("S2y") - pl.col("S1y") * pl.col("S2"))
            + pl.col("S0y") * (pl.col("S1") * pl.col("S3") - pl.col("S2") * pl.col("S2"))
        )
        out = np.where(detXX).alias("v")) != 0, g2.with_columns((detA / detXX).alias("v")), np.nan)
        out = out.with_columns(pl.when(pl.col("n") >= 4).then(pl.col("v")).otherwise(None))
    return _pivot(out, "v")


_mk("intra_vwap_path_slope", "累计 VWAP 路径斜率（Polars）。", ["close", "amount", "volume"],
   lambda close, amount, volume: _vwap_path_common(close, amount, volume, 1, 1))
_mk("intra_vwap_path_curvature", "累计 VWAP 路径曲率（Polars）。", ["close", "amount", "volume"],
   lambda close, amount, volume: _vwap_path_common(close, amount, volume, 2, 2))


# ---------------------------------------------------------------------------
# § Volume / amount distribution: concentration, entropy, signed imbalance
# ---------------------------------------------------------------------------

def _concentration(value: pl.DataFrame) -> pl.DataFrame:
    long = _with_date(_melt(value, "value")).with_columns(pl.col("value").fill_nan(0.0).abs())
    total = long.group_by(["date", "instrument"]).agg(pl.col("value").sum().alias("total"))
    long = long.join(total, on=["date", "instrument"]).with_columns(
        (pl.col("value") / pl.col("total")).alias("w")
    )
    out = long.group_by(["date", "instrument"]).agg(
        (pl.col("w") * pl.col("w")).sum().alias("v"),
        pl.col("total").first().alias("total"),
    )
    out = out.with_columns(pl.when(pl.col("total") > _EPS).then(pl.col("v")).otherwise(None))
    return _pivot(out, "v")


_mk("intra_concentration", "日内成交集中度 sum((value/sum)^2)（Polars）。", ["value"],
   lambda value: _concentration(value))


def _entropy(value: pl.DataFrame, normalize: bool) -> pl.DataFrame:
    long = _with_date(_melt(value, "value")).with_columns(pl.col("value").fill_nan(0.0).abs())
    long = long.filter(pl.col("value") > 0.0)
    total = long.group_by(["date", "instrument"]).agg(
        pl.col("value").sum().alias("total"),
        pl.col("value").count().alias("n"),
    )
    long = long.join(total, on=["date", "instrument"]).with_columns(
        (pl.col("value") / pl.col("total")).alias("w")
    )
    e = long.group_by(["date", "instrument"]).agg(
        (-(pl.col("w") * pl.col("w").log())).sum().alias("H"),
        pl.col("total").first().alias("total"),
        pl.col("n").first().alias("n"),
    )
    if normalize:
        e = e.with_columns(
            pl.when((pl.col("n") >= 2) & (pl.col("total") > _EPS))
            .then(pl.col("H") / pl.col("n").cast(pl.Float64).log())
            .otherwise(None)
            .alias("v")
        )
    else:
        e = e.with_columns(
            pl.when((pl.col("n") >= 2) & (pl.col("total") > _EPS))
            .then(pl.col("H"))
            .otherwise(None)
            .alias("v")
        )
    return _pivot(e, "v")


_mk("intra_entropy", "日内成交分布熵（归一化）（Polars）。", ["value", "normalize"],
   lambda value, normalize=True: _entropy(value, bool(normalize)))


def _signed_imbalance_proxy(close: pl.DataFrame, value: pl.DataFrame) -> pl.DataFrame:
    long = _with_date(
        _melt(close, "close").join(_melt(value, "value"), on=["ts", "instrument"], how="left")
    ).filter(pl.col("close").is_finite())
    long = _log_returns_long(long).with_columns(pl.col("value").fill_nan(0.0))
    out = long.group_by(["date", "instrument"]).agg(
        pl.when(pl.col("r").is_finite()).then(pl.col("r").sign() * pl.col("value")).otherwise(None).sum().alias("num"),
        pl.col("value").sum().alias("den"),
    ).with_columns(
        pl.when(pl.col("den") > _EPS).then(pl.col("num") / pl.col("den")).otherwise(None).alias("v")
    )
    return _pivot(out, "v")


_mk("intra_signed_imbalance_proxy", "基于分钟价格方向的成交不平衡代理（Polars）。", ["close", "value"],
   lambda close, value: _signed_imbalance_proxy(close, value))


# ---------------------------------------------------------------------------
# § Liquidity / extreme bar / lunch gap
# ---------------------------------------------------------------------------

def _intra_amihud(close: pl.DataFrame, amount: pl.DataFrame, scale: float) -> pl.DataFrame:
    long = _with_date(
        _melt(close, "close").join(_melt(amount, "amount"), on=["ts", "instrument"], how="left")
    ).filter(pl.col("close").is_finite())
    long = _log_returns_long(long).with_columns(pl.col("amount").fill_nan(0.0))
    long = long.with_columns(
        pl.when(pl.col("r").is_finite())
        .then(pl.col("r").abs() / pl.max_horizontal(pl.col("amount"), pl.lit(_EPS)))
        .otherwise(None)
        .alias("ratio")
    )
    out = long.group_by(["date", "instrument"]).agg(
        pl.col("ratio").mean().alias("m"),
        pl.col("ratio").count().alias("n"),
    ).with_columns(
        pl.when(pl.col("n") > 0).then(pl.col("m") * float(scale)).otherwise(None).alias("v")
    )
    return _pivot(out, "v")


_mk("intra_amihud", "日内 Amihud 非流动性 mean(|r|/max(amount,eps))*scale（Polars）。", ["close", "amount", "scale"],
   lambda close, amount, scale=1e8: _intra_amihud(close, amount, float(scale)))


def _kyle_lambda_proxy(close: pl.DataFrame, amount: pl.DataFrame) -> pl.DataFrame:
    long = _with_date(
        _melt(close, "close").join(_melt(amount, "amount"), on=["ts", "instrument"], how="left")
    ).filter(pl.col("close").is_finite())
    long = _log_returns_long(long).with_columns(pl.col("amount").fill_nan(0.0))
    total = long.group_by(["date", "instrument"]).agg(pl.col("amount").sum().alias("total"))
    long = long.join(total, on=["date", "instrument"]).with_columns(
        (pl.col("r").sign() * pl.col("amount") / pl.col("total")).alias("s")
    ).filter(pl.col("r").is_finite() & pl.col("s").is_finite())
    g = long.group_by(["date", "instrument"]).agg(
        pl.col("r").count().alias("n"),
        (pl.col("r")).sum().alias("sr"),
        (pl.col("s")).sum().alias("ss"),
        (pl.col("r") * pl.col("s")).sum().alias("srs"),
        (pl.col("s") * pl.col("s")).sum().alias("ss2"),
    ).with_columns(pl.col("n").cast(pl.Float64).alias("nf"))
    g = g.with_columns(
        # pandas np.cov (ddof=1) / np.var (ddof=0) ratio carries a n/(n-1) factor
        ((pl.col("srs") - pl.col("sr") * pl.col("ss") / pl.col("nf"))
         / (pl.col("ss2") - pl.col("ss") * pl.col("ss") / pl.col("nf"))
         * pl.col("nf") / (pl.col("nf") - 1.0)).alias("beta")
    )
    out = g.with_columns(
        pl.when((pl.col("n") >= 3) & ((pl.col("ss2") - pl.col("ss") * pl.col("ss") / pl.col("nf")) > _EPS * pl.col("nf")))
        .then(pl.col("beta"))
        .otherwise(None)
        .alias("v")
    )
    return _pivot(out, "v")


_mk("intra_kyle_lambda_proxy", "日内 Kyle Lambda 代理（Polars）。", ["close", "amount"],
   lambda close, amount: _kyle_lambda_proxy(close, amount))


def _extreme_bar_return(close: pl.DataFrame, side: str) -> pl.DataFrame:
    long = _with_date(_melt(close, "close")).filter(pl.col("close").is_finite())
    long = _log_returns_long(long).filter(pl.col("r").is_finite())
    agg = long.group_by(["date", "instrument"]).agg(
        pl.col("r").max().alias("vmax") if side == "max" else pl.col("r").min().alias("vmin")
    )
    value = "vmax" if side == "max" else "vmin"
    return _pivot(agg, value)


_mk("intra_extreme_bar_return", "日内单分钟最大/最小收益（Polars）。", ["close", "side"],
   lambda close, side="max": _extreme_bar_return(close, side))


def _lunch_gap_return(close: pl.DataFrame, open_px: pl.DataFrame, morning_cutoff: str, afternoon_start: str) -> pl.DataFrame:
    def _hm(text: str) -> int:
        hh, mm = str(text).split(":")
        return int(hh) * 60 + int(mm)

    cutoff, start = _hm(morning_cutoff), _hm(afternoon_start)
    long = _with_mod(_with_date(
        _melt(close, "close").join(_melt(open_px, "open"), on=["ts", "instrument"], how="left")
    ))
    morning = long.filter((pl.col("mod") <= cutoff) & pl.col("close").is_finite())
    afternoon = long.filter((pl.col("mod") >= start) & pl.col("open").is_finite())
    m = morning.sort(["date", "instrument", "ts"]).group_by(["date", "instrument"]).agg(
        pl.col("close").last().alias("mc")
    )
    a = afternoon.sort(["date", "instrument", "ts"]).group_by(["date", "instrument"]).agg(
        pl.col("open").first().alias("ao")
    )
    merged = a.join(m, on=["date", "instrument"])
    out = merged.with_columns(
        pl.when(pl.col("mc") > _EPS)
        .then(pl.col("ao") / pl.col("mc") - 1.0)
        .otherwise(None)
        .alias("v")
    )
    return _pivot(out, "v")


_mk("intra_lunch_gap_return", "午间跳空：下午首根 Open/上午末根 Close - 1（Polars）。",
   ["close", "open", "morning_cutoff", "afternoon_start", "session_tz", "endpoint_policy"],
   lambda close, open_px, morning_cutoff="11:30", afternoon_start="13:00", session_tz=None,
       endpoint_policy="exact":
       _lunch_gap_return(close, open_px, morning_cutoff, afternoon_start))


# ---------------------------------------------------------------------------
# § Daily-limit behaviour (daily limit panel broadcast to minute bars)
# ---------------------------------------------------------------------------

def _limit_long(close: pl.DataFrame, limit: pl.DataFrame, side: str) -> pl.DataFrame:
    long = _with_date(_melt(close, "close"))
    lim = _melt_daily(limit, "limit")
    long = long.join(lim, on=["date", "instrument"], how="left")
    if side == "up":
        cond = pl.col("close") >= pl.col("limit") - _EPS
    elif side == "down":
        cond = pl.col("close") <= pl.col("limit") + _EPS
    else:
        raise ValueError(f"unknown limit side: {side!r}")
    return long.with_columns(
        pl.when(pl.col("close").is_finite() & pl.col("limit").is_finite())
        .then(cond)
        .otherwise(False)
        .alias("mask")
    )


def _limit_side_limit(close: pl.DataFrame, high_limit, low_limit, side: str) -> pl.DataFrame:
    limit = high_limit if side == "up" else low_limit
    if limit is None:
        raise ValueError(f"missing {side} limit panel")
    return limit


def _limit_first_hit_time(close: pl.DataFrame, high_limit, low_limit, side: str) -> pl.DataFrame:
    limit = _limit_side_limit(close, high_limit, low_limit, side)
    long = _limit_long(close, limit, side)
    long = long.sort(["date", "instrument", "ts"])
    out = long.group_by(["date", "instrument"]).agg(
        pl.col("mask").sum().alias("cnt"),
        pl.col("mask").arg_max().alias("pos"),
        pl.len().alias("n"),
    ).with_columns(
        pl.when((pl.col("n") > 0) & (pl.col("cnt") > 0)).then(pl.col("pos") / pl.col("n")).otherwise(None).alias("v")
    )
    return _pivot(out, "v")


_mk("intra_limit_first_hit_time", "首次触及涨/跌停的分钟位置（Polars）。",
   ["close", "high", "low", "high_limit", "low_limit", "side"],
   lambda close, high=None, low=None, high_limit=None, low_limit=None, side="up":
       _limit_first_hit_time(close, high_limit, low_limit, side),
   extra_tags=["allow_panel_broadcast"])


def _limit_duration(close: pl.DataFrame, high_limit, low_limit, side: str) -> pl.DataFrame:
    limit = _limit_side_limit(close, high_limit, low_limit, side)
    long = _limit_long(close, limit, side)
    out = long.group_by(["date", "instrument"]).agg(
        pl.col("mask").sum().alias("cnt"),
        pl.len().alias("n"),
    ).with_columns(
        pl.when(pl.col("n") > 0).then(pl.col("cnt") / pl.col("n")).otherwise(None).alias("v")
    )
    return _pivot(out, "v")


_mk("intra_limit_duration", "收盘价处于涨/跌停价附近的分钟占比（Polars）。",
   ["close", "high_limit", "low_limit", "side"],
   lambda close, high_limit=None, low_limit=None, side="up":
       _limit_duration(close, high_limit, low_limit, side),
   extra_tags=["allow_panel_broadcast"])


def _limit_reopen_count(close: pl.DataFrame, high_limit, low_limit, side: str, transition: str) -> pl.DataFrame:
    limit = _limit_side_limit(close, high_limit, low_limit, side)
    long = _limit_long(close, limit, side)
    long = long.sort(["date", "instrument", "ts"]).with_columns(
        pl.col("mask").shift(1).over(["date", "instrument"]).alias("prev")
    )
    # pandas compares mask[:-1] vs mask[1:]; the first bar of a day is never a target.
    has_prev = pl.col("prev").is_not_null()
    if transition == "open":
        trans = has_prev & pl.col("prev") & (~pl.col("mask"))
    elif transition == "reseal":
        trans = has_prev & (~pl.col("prev")) & pl.col("mask")
    else:
        raise ValueError(f"unknown transition: {transition!r}")
    out = long.group_by(["date", "instrument"]).agg(trans.sum().alias("v"))
    return _pivot(out, "v")


_mk("intra_limit_reopen_count", "封板打开（open）/ 重新封板（reseal）次数（Polars）。",
   ["close", "high_limit", "low_limit", "side", "transition"],
   lambda close, high_limit=None, low_limit=None, side="up", transition="open":
       _limit_reopen_count(close, high_limit, low_limit, side, transition),
   extra_tags=["allow_panel_broadcast"])


# ---------------------------------------------------------------------------
# § Cross-day slot memory and profile distances
# ---------------------------------------------------------------------------

def _slot_frame(close: pl.DataFrame) -> pl.DataFrame:
    """Log-returns of the full per-instrument series, slotted by minute-of-day.

    ``r`` is computed across day boundaries (matches the pandas reference), then
    duplicate (date, slot, instrument) cells are averaged.
    """
    long = _melt(close, "close")
    long = _with_mod(_with_date(long)).rename({"mod": "slot"})
    long = long.sort(["instrument", "ts"]).with_columns(
        (pl.col("close") / pl.col("close").shift(1).over(["instrument"])).log().alias("r")
    )
    return long.filter(pl.col("r").is_finite()).group_by(["date", "slot", "instrument"]).agg(
        pl.col("r").mean().alias("r")
    )


def _hist_slot(slot: pl.DataFrame, window: int, value_col: str = "r") -> pl.DataFrame:
    w = max(2, int(window))
    mp = max(2, w // 2)
    out = slot.sort(["instrument", "slot", "date"]).with_columns(
        pl.col(value_col)
        .shift(1)
        .over(["instrument", "slot"])
        .rolling_mean(window_size=w, min_samples=mp)
        .over(["instrument", "slot"])
        .alias("hist")
    )
    return out


def _same_slot_score(close: pl.DataFrame, window: int, reverse: bool) -> pl.DataFrame:
    slot = _slot_frame(close)
    hist = _hist_slot(slot, window)
    joined = slot.join(hist.select(["date", "slot", "instrument", "hist"]), on=["date", "slot", "instrument"], how="left")
    if reverse:
        prod = (
            pl.when((pl.col("r").sign() != pl.col("hist").sign()) & pl.col("hist").is_not_null())
            .then(pl.col("r") * pl.col("hist"))
            .otherwise(None)
        )
        out = joined.group_by(["date", "instrument"]).agg(
            (-(prod.sum())).alias("s"),
            prod.count().alias("cnt"),
        ).with_columns(
            pl.when(pl.col("cnt") > 0).then(pl.col("s")).otherwise(None).alias("v")
        )
    else:
        out = joined.with_columns(
            (pl.col("r") * pl.col("hist")).alias("prod")
        ).group_by(["date", "instrument"]).agg(
            pl.col("prod").sum().alias("s"),
            pl.col("prod").count().alias("cnt"),
        ).with_columns(
            pl.when(pl.col("cnt") > 0).then(pl.col("s")).otherwise(None).alias("v")
        )
    return _pivot(out, "v")


_mk("intra_same_slot_momentum", "同日段跨日延续得分（Polars）。", ["close", "window"],
   lambda close, window=20: _same_slot_score(close, int(window), reverse=False))
_mk("intra_same_slot_reversal", "同日段反向得分（Polars）。", ["close", "window"],
   lambda close, window=20: _same_slot_score(close, int(window), reverse=True))


def _profile_frame(values: pl.DataFrame) -> pl.DataFrame:
    """Raw values slotted by minute-of-day (no return transform)."""
    long = _with_mod(_with_date(_melt(values, "value"))).rename({"mod": "slot"})
    return long.filter(pl.col("value").is_finite()).group_by(["date", "slot", "instrument"]).agg(
        pl.col("value").mean().alias("value")
    )


def _profile_cosine(values: pl.DataFrame, window: int) -> pl.DataFrame:
    slot = _profile_frame(values)
    hist = _hist_slot(slot, window, value_col="value")
    joined = slot.join(hist.select(["date", "slot", "instrument", "hist"]), on=["date", "slot", "instrument"], how="left")
    joined = joined.filter(pl.col("hist").is_not_null())
    out = joined.group_by(["date", "instrument"]).agg(
        pl.col("value").count().alias("n"),
        (pl.col("value") * pl.col("hist")).sum().alias("dot"),
        (pl.col("value") * pl.col("value")).sum().alias("srr"),
        (pl.col("hist") * pl.col("hist")).sum().alias("shh"),
    ).with_columns(
        pl.when((pl.col("n") >= 2) & (pl.col("srr") > _EPS) & (pl.col("shh") > _EPS))
        .then(pl.col("dot") / (pl.col("srr").sqrt() * pl.col("shh").sqrt()))
        .otherwise(None)
        .alias("v")
    )
    return _pivot(out, "v")


def _profile_jsd(values: pl.DataFrame, window: int) -> pl.DataFrame:
    slot = _profile_frame(values)
    hist = _hist_slot(slot, window, value_col="value")
    joined = slot.join(hist.select(["date", "slot", "instrument", "hist"]), on=["date", "slot", "instrument"], how="left")
    joined = joined.filter(pl.col("hist").is_not_null()).with_columns(
        (pl.col("value").clip(lower_bound=0.0) + 1e-12).alias("pa"),
        (pl.col("hist").clip(lower_bound=0.0) + 1e-12).alias("pb"),
    )
    sums = joined.group_by(["date", "instrument"]).agg(
        pl.col("pa").sum().alias("sum_pa"),
        pl.col("pb").sum().alias("sum_pb"),
        pl.col("pa").count().alias("n"),
    )
    joined = joined.join(sums, on=["date", "instrument"]).with_columns(
        (pl.col("pa") / pl.col("sum_pa")).alias("pa"),
        (pl.col("pb") / pl.col("sum_pb")).alias("pb"),
    ).with_columns((0.5 * (pl.col("pa") + pl.col("pb"))).alias("m"))
    out = joined.group_by(["date", "instrument"]).agg(
        pl.col("n").first().alias("n"),
        (pl.col("pa") * (pl.col("pa").log() - pl.col("m").log())).sum().alias("kl_a"),
        (pl.col("pb") * (pl.col("pb").log() - pl.col("m").log())).sum().alias("kl_b"),
    ).with_columns(
        pl.when(pl.col("n") >= 2).then(0.5 * (pl.col("kl_a") + pl.col("kl_b"))).otherwise(None).alias("v")
    )
    return _pivot(out, "v")


def _profile_emd(values: pl.DataFrame, window: int) -> pl.DataFrame:
    slot = _profile_frame(values)
    hist = _hist_slot(slot, window, value_col="value")
    joined = slot.join(hist.select(["date", "slot", "instrument", "hist"]), on=["date", "slot", "instrument"], how="left")
    joined = joined.filter(pl.col("hist").is_not_null()).with_columns(
        (pl.col("value").clip(lower_bound=0.0) + 1e-12).alias("pa"),
        (pl.col("hist").clip(lower_bound=0.0) + 1e-12).alias("pb"),
    )
    sums = joined.group_by(["date", "instrument"]).agg(
        pl.col("pa").sum().alias("sum_pa"),
        pl.col("pb").sum().alias("sum_pb"),
        pl.col("pa").count().alias("n"),
    )
    joined = joined.join(sums, on=["date", "instrument"]).with_columns(
        (pl.col("pa") / pl.col("sum_pa")).alias("pa"),
        (pl.col("pb") / pl.col("sum_pb")).alias("pb"),
    )
    joined = joined.sort(["date", "instrument", "slot"]).with_columns(
        pl.col("pa").cum_sum().over(["date", "instrument"]).alias("cdf_a"),
        pl.col("pb").cum_sum().over(["date", "instrument"]).alias("cdf_b"),
    )
    out = joined.group_by(["date", "instrument"]).agg(
        pl.col("n").first().alias("n"),
        ((pl.col("cdf_a") - pl.col("cdf_b")).abs()).sum().alias("v"),
    ).with_columns(pl.when(pl.col("n") >= 2).then(pl.col("v")).otherwise(None))
    return _pivot(out, "v")


for _name, _desc, _fn in (
    ("intra_return_profile_cosine", "分钟收益曲线与历史均值曲线余弦相似度（Polars）。", _profile_cosine),
    ("intra_volume_profile_cosine", "分钟成交量占比曲线余弦相似度（Polars）。", _profile_cosine),
    ("intra_amount_profile_cosine", "分钟成交额占比曲线余弦相似度（Polars）。", _profile_cosine),
    ("intra_volume_profile_jsd", "成交量分布与历史基准 JSD（Polars）。", _profile_jsd),
    ("intra_amount_profile_jsd", "成交额分布与历史基准 JSD（Polars）。", _profile_jsd),
    ("intra_profile_earth_mover_distance", "分布与历史基准 Wasserstein 距离（Polars）。", _profile_emd),
):
    _mk(_name, _desc, ["x", "window"], lambda x, window=20, _fn=_fn: _fn(x, int(window)))


# ---------------------------------------------------------------------------
# § Jump timing (higher-moment helpers)
# ---------------------------------------------------------------------------

def _jump_mask_long(close: pl.DataFrame, threshold_scale: float) -> pl.DataFrame:
    """Per-bar significant-jump mask |r| > scale * sqrt(RV/N); NaN r -> no jump."""
    ts = float(threshold_scale)
    long = _with_date(_melt(close, "close")).filter(pl.col("close").is_finite())
    long = _log_returns_long(long)
    rv = long.group_by(["date", "instrument"]).agg(
        (pl.col("r") * pl.col("r")).fill_nan(0.0).sum().alias("rv"),
        pl.col("r").count().alias("n"),
    )
    long = long.join(rv, on=["date", "instrument"]).with_columns(
        pl.when((pl.col("rv") > _EPS) & (pl.col("n") >= 2))
        .then((pl.col("rv") / pl.col("n")).sqrt() * ts)
        .otherwise(None)
        .alias("thresh")
    ).with_columns(
        pl.when((pl.col("r").abs() > pl.col("thresh")) & pl.col("r").is_finite() & pl.col("thresh").is_not_null())
        .then(True)
        .otherwise(False)
        .alias("jump")
    )
    return long


def _tripower_quarticity(close: pl.DataFrame) -> pl.DataFrame:
    long = _with_date(_melt(close, "close")).filter(pl.col("close").is_finite())
    long = _log_returns_long(long).filter(pl.col("r").is_finite())
    long = long.sort(["date", "instrument", "ts"]).with_columns(
        pl.col("r").abs().pow(4.0 / 3.0).alias("rp"),
    ).with_columns(
        (pl.col("rp") * pl.col("rp").shift(1).over(["date", "instrument"]) * pl.col("rp").shift(2).over(["date", "instrument"])).alias("prod")
    )
    scale = _tripower_scale()
    out = long.group_by(["date", "instrument"]).agg(
        pl.col("prod").fill_nan(0.0).sum().alias("s"),
        pl.col("r").count().alias("n"),
    ).with_columns(
        (pl.col("s") * pl.col("n") * scale).alias("v")
    )
    out = out.with_columns(pl.when(pl.col("n") >= 4).then(pl.col("v")).otherwise(None))
    return _pivot(out, "v")


def _jump_timing(close: pl.DataFrame, threshold_scale: float, which: str) -> pl.DataFrame:
    long = _jump_mask_long(close, threshold_scale)
    long = long.sort(["date", "instrument", "ts"]).with_columns(
        pl.col("ts").rank("ordinal").over(["date", "instrument"]).cast(pl.Float64).alias("pos")
    )
    stats = long.group_by(["date", "instrument"]).agg(
        pl.col("r").is_finite().sum().alias("n_finite"),
        pl.when(pl.col("jump")).then(pl.col("pos")).otherwise(None).min().alias("first_pos"),
        pl.when(pl.col("jump")).then(pl.col("pos")).otherwise(None).max().alias("last_pos"),
    )
    if which == "first":
        out = stats.with_columns(
            pl.when((pl.col("n_finite") > 1) & pl.col("first_pos").is_not_null())
            .then((pl.col("first_pos") - 1.0) / (pl.col("n_finite") - 1.0))
            .otherwise(None)
            .alias("v")
        )
    else:
        out = stats.with_columns(
            pl.when((pl.col("n_finite") > 1) & pl.col("last_pos").is_not_null())
            .then((pl.col("last_pos") - 1.0) / (pl.col("n_finite") - 1.0))
            .otherwise(None)
            .alias("v")
        )
    return _pivot(out, "v")


def _jump_clustering(close: pl.DataFrame, threshold_scale: float) -> pl.DataFrame:
    long = _jump_mask_long(close, threshold_scale)
    long = long.sort(["date", "instrument", "ts"]).with_columns(
        pl.col("ts").rank("ordinal").over(["date", "instrument"]).cast(pl.Float64).alias("pos")
    )
    jumps = long.filter(pl.col("jump")).group_by(["date", "instrument"]).agg(
        pl.col("pos").alias("poss")
    )
    out = jumps.with_columns(
        pl.col("poss").list.diff(1).alias("gaps")
    ).with_columns(
        pl.when(pl.col("gaps").list.len() >= 2)
        .then(pl.col("gaps").list.std(ddof=0) / pl.col("gaps").list.mean())
        .otherwise(None)
        .alias("v")
    )
    return _pivot(out, "v")


_mk("intra_tripower_quarticity", "Tripower quarticity 估计（Polars）。", ["close"],
   lambda close: _tripower_quarticity(close))
_mk("intra_jump_first_time", "首次跳跃位置/有效分钟数（Polars）。", ["close", "threshold_scale"],
   lambda close, threshold_scale=3.0: _jump_timing(close, float(threshold_scale), "first"))
_mk("intra_jump_last_time", "末次跳跃位置/有效分钟数（Polars）。", ["close", "threshold_scale"],
   lambda close, threshold_scale=3.0: _jump_timing(close, float(threshold_scale), "last"))
_mk("intra_jump_clustering", "跳跃间隔 CV（Polars）。", ["close", "threshold_scale"],
   lambda close, threshold_scale=3.0: _jump_clustering(close, float(threshold_scale)))


# ---------------------------------------------------------------------------
# § Realized market beta family
# ---------------------------------------------------------------------------

def _beta_frame(close: pl.DataFrame, free_market_cap: pl.DataFrame) -> pl.DataFrame:
    long = _with_date(_melt(close, "close"))
    # Round-2 review §intraday/realized_beta: each trading session is an
    # independent return process — the FIRST minute of every session is NaN
    # (its log-return would otherwise pair yesterday's last minute with
    # today's first minute = an overnight jump).  The pandas reference
    # (_aligned_market) breaks at the calendar-day boundary the same way.
    long = long.sort(["instrument", "ts"]).with_columns(
        pl.when(pl.col("date") == pl.col("date").shift(1).over(["instrument"]))
        .then((pl.col("close") / pl.col("close").shift(1).over(["instrument"])).log())
        .otherwise(None)
        .alias("r")
    )
    w = _melt_daily(free_market_cap, "w")
    long = long.join(w, on=["date", "instrument"], how="left")
    # weight must be finite AND strictly positive (a NaN/0/negative market cap is
    # excluded from the market return, matching the pandas validated-weight panel)
    wv = pl.col("w").is_finite() & (pl.col("w") > 0)
    mkt = long.group_by(["date", "ts"]).agg(
        pl.when(pl.col("r").is_finite() & wv).then(pl.col("r") * pl.col("w")).otherwise(None).sum().alias("num"),
        pl.when(pl.col("r").is_finite() & wv).then(pl.col("w")).otherwise(None).sum().alias("den"),
    ).with_columns(
        pl.when(pl.col("den").abs() > _EPS).then(pl.col("num") / pl.col("den")).otherwise(None).alias("m")
    )
    return long.join(mkt.select(["date", "ts", "m"]), on=["date", "ts"], how="left").filter(
        pl.col("m").is_finite()
    )


def _beta_pairs(long: pl.DataFrame, r_expr: pl.Expr, m_expr: pl.Expr, cond_r: pl.Expr | None, cond_m: pl.Expr | None) -> pl.DataFrame:
    rf = r_expr.is_finite() & m_expr.is_finite()
    if cond_r is not None:
        num_mask = rf & cond_r & cond_m
    else:
        num_mask = None
    den_mask = rf & (cond_m if cond_m is not None else pl.lit(True))
    return long.group_by(["date", "instrument"]).agg(
        pl.len().alias("n"),
        (pl.col("r")).sum().alias("sr"),
        (pl.col("m")).sum().alias("sm"),
        (pl.col("m") * pl.col("m")).sum().alias("sm2"),
        (pl.col("r") * pl.col("m")).sum().alias("srm"),
        ((pl.when(num_mask).then(pl.col("r") * pl.col("m")).otherwise(None)).sum() if num_mask is not None else pl.lit(None)).alias("qnum"),
        ((pl.when(den_mask).then(pl.col("m") * pl.col("m")).otherwise(None)).sum()).alias("qden"),
    )


def _beta_stat(close: pl.DataFrame, free_market_cap: pl.DataFrame, kind: str) -> pl.DataFrame:
    long = _beta_frame(close, free_market_cap)
    if kind in ("down_down", "up_up", "down_up", "up_down"):
        conds = {
            "down_down": (pl.col("r") < 0, pl.col("m") < 0),
            "up_up": (pl.col("r") > 0, pl.col("m") > 0),
            "down_up": (pl.col("r") < 0, pl.col("m") > 0),
            "up_down": (pl.col("r") > 0, pl.col("m") < 0),
        }
        cr, cm = conds[kind]
        g = long.group_by(["date", "instrument"]).agg(
            pl.when(pl.col("r").is_finite() & pl.col("m").is_finite() & cr & cm)
            .then(pl.col("r") * pl.col("m")).otherwise(None).sum().alias("num"),
            pl.when(pl.col("r").is_finite() & pl.col("m").is_finite() & cm)
            .then(pl.col("m") * pl.col("m")).otherwise(None).sum().alias("den"),
        )
        out = g.with_columns(
            pl.when(pl.col("den") > _EPS).then(pl.col("num") / pl.col("den")).otherwise(None).alias("v")
        )
        return _pivot(out, "v")
    if kind == "corr":
        g = long.filter(pl.col("r").is_finite()).group_by(["date", "instrument"]).agg(
            pl.len().alias("n"),
            pl.corr(pl.col("r"), pl.col("m")).alias("c"),
        )
        out = g.with_columns(
            pl.when((pl.col("n") >= 3) & pl.col("c").is_finite()).then(pl.col("c")).otherwise(None).alias("v")
        )
        return _pivot(out, "v")
    # realized_beta / idio family / r2
    g = long.filter(pl.col("r").is_finite() & pl.col("m").is_finite()).group_by(["date", "instrument"]).agg(
        pl.len().alias("n"),
        (pl.col("r")).sum().alias("sr"),
        (pl.col("m")).sum().alias("sm"),
        (pl.col("m") * pl.col("m")).sum().alias("sm2"),
        (pl.col("r") * pl.col("m")).sum().alias("srm"),
    ).with_columns(pl.col("n").cast(pl.Float64).alias("nf")).with_columns(
        (pl.col("srm") - pl.col("sr") * pl.col("sm") / pl.col("nf")).alias("cov_num"),
        (pl.col("sm2") - pl.col("sm") * pl.col("sm") / pl.col("nf")).alias("var_m_num"),
    )
    if kind == "realized_beta":
        out = g.with_columns(
            pl.when((pl.col("n") >= 3) & (pl.col("var_m_num") / pl.col("nf") > _EPS))
            .then(pl.col("cov_num") / pl.col("var_m_num"))
            .otherwise(None)
            .alias("v")
        )
        return _pivot(out, "v")
    # market model fit b — population moments (round-2 review: the pandas
    # reference now uses cov = mean((r-rbar)(m-mbar)) / var = mean((m-mbar)^2),
    # i.e. NO np.cov-ddof-1 / np.var-ddof-0 factor; the old nf/(nf-1) term is gone)
    g = g.with_columns(
        pl.when(
            (pl.col("n") >= 3) & (pl.col("var_m_num") / pl.col("nf") > _EPS)
        ).then(pl.col("cov_num") / pl.col("var_m_num")).otherwise(None).alias("b")
    ).with_columns(
        pl.when(pl.col("b").is_not_null())
        .then((pl.col("sr") / pl.col("nf")) - pl.col("b") * (pl.col("sm") / pl.col("nf")))
        .otherwise(None)
        .alias("a")
    )
    joined = long.filter(pl.col("r").is_finite() & pl.col("m").is_finite()).join(
        g.select(["date", "instrument", "a", "b", "nf"]), on=["date", "instrument"], how="left"
    ).with_columns(
        (pl.col("r") - (pl.col("a") + pl.col("b") * pl.col("m"))).alias("e")
    )
    if kind == "idio_variance":
        out = joined.group_by(["date", "instrument"]).agg((pl.col("e") * pl.col("e")).mean().alias("v"))
        return _pivot(out, "v")
    g2 = joined.group_by(["date", "instrument"]).agg(
        (pl.col("e")).sum().alias("se"),
        (pl.col("e") * pl.col("e")).sum().alias("se2"),
        (pl.col("e") * pl.col("e") * pl.col("e")).sum().alias("se3"),
        (pl.col("e") * pl.col("e") * pl.col("e") * pl.col("e")).sum().alias("se4"),
        (pl.col("r")).sum().alias("sr"),
        (pl.col("r") * pl.col("r")).sum().alias("sr2"),
        pl.len().alias("n"),
    ).with_columns(pl.col("n").cast(pl.Float64).alias("nf"))
    if kind == "idio_skew":
        mean_e = pl.when(pl.col("nf") != 0).then((pl.col("se")) / (pl.col("nf"))).otherwise(None)
        var_e = pl.when(pl.col("nf") - mean_e.pow(2) != 0).then((pl.col("se2")) / (pl.col("nf") - mean_e.pow(2))).otherwise(None)
        m3 = pl.when(pl.col("nf") - 3.0 * mean_e * pl.col("se2") / pl.col("nf") + 2.0 * mean_e.pow(3) != 0).then((pl.col("se3")) / (pl.col("nf") - 3.0 * mean_e * pl.col("se2") / pl.col("nf") + 2.0 * mean_e.pow(3))).otherwise(None)
        out = g2.with_columns(
            pl.when(var_e > _EPS).then(m3 / var_e.pow(1.5)).otherwise(None).alias("v")
        )
        return _pivot(out, "v")
    if kind == "idio_kurt":
        # E[((e-e_mean)/sd)^4]
        mean_e = pl.when(pl.col("nf") != 0).then((pl.col("se")) / (pl.col("nf"))).otherwise(None)
        var_e = pl.when(pl.col("nf") - mean_e.pow(2) != 0).then((pl.col("se2")) / (pl.col("nf") - mean_e.pow(2))).otherwise(None)
        m4 = pl.when(pl.col("nf") - 4 * mean_e * pl.col("se3") / pl.col("nf") + 6 * mean_e.pow(2) * pl.col("se2") / pl.col("nf") - 3 * mean_e.pow(4) != 0).then((pl.col("se4")) / (pl.col("nf") - 4 * mean_e * pl.col("se3") / pl.col("nf") + 6 * mean_e.pow(2) * pl.col("se2") / pl.col("nf") - 3 * mean_e.pow(4))).otherwise(None)
        out = g2.with_columns(
            pl.when(var_e > _EPS).then(m4 / var_e.pow(2)).otherwise(None).alias("v")
        )
        return _pivot(out, "v")
    # r2 = max(0, 1 - ss_res/ss_tot); ss_tot from the raw return series.
    out = g2.with_columns(
        ((pl.col("sr2") - pl.col("sr").pow(2) / pl.col("nf"))).alias("ss_tot"),
    ).with_columns(
        pl.when((pl.col("ss_tot") > _EPS))
        .then((1.0 - pl.col("se2") / pl.col("ss_tot")).clip(lower_bound=0.0))
        .otherwise(None)
        .alias("v")
    )
    return _pivot(out, "v")


def _beta_asymmetry(close: pl.DataFrame, free_market_cap: pl.DataFrame) -> pl.DataFrame:
    dd = _beta_stat(close, free_market_cap, "down_down")
    uu = _beta_stat(close, free_market_cap, "up_up")
    dd = dd.sort("date")
    uu = uu.sort("date")
    cols = [c for c in dd.columns if c != "date"]
    out = dd
    for c in cols:
        out = out.with_columns((pl.col(c) - uu[c]).alias(c))
    return out


for _name, _desc, _kind, _fn in (
    ("intra_realized_beta", "日内已实现 Beta（Polars）。", "realized_beta", _beta_stat),
    ("intra_realized_correlation", "日内已实现相关系数（Polars）。", "corr", _beta_stat),
    ("intra_down_down_semibeta", "下-下半 Beta（Polars）。", "down_down", _beta_stat),
    ("intra_up_up_semibeta", "上-上半 Beta（Polars）。", "up_up", _beta_stat),
    ("intra_down_up_semibeta", "下-上半 Beta（Polars）。", "down_up", _beta_stat),
    ("intra_up_down_semibeta", "上-下半 Beta（Polars）。", "up_down", _beta_stat),
    ("intra_beta_asymmetry", "下半 Beta 减上半 Beta（Polars）。", None, _beta_asymmetry),
    ("intra_idiosyncratic_variance", "分钟市场模型残差方差（Polars）。", "idio_variance", _beta_stat),
    ("intra_idiosyncratic_skewness", "分钟市场模型残差偏度（Polars）。", "idio_skew", _beta_stat),
    ("intra_idiosyncratic_kurtosis", "分钟市场模型残差峰度（Polars）。", "idio_kurt", _beta_stat),
    ("intra_market_model_r2", "分钟市场模型 R²（Polars）。", "r2", _beta_stat),
):
    if _kind is None:
        _mk(_name, _desc, ["close", "free_market_cap"],
            lambda close, free_market_cap, _fn=_fn: _fn(close, free_market_cap),
            extra_tags=["allow_panel_broadcast"])
    else:
        _mk(_name, _desc, ["close", "free_market_cap"],
            lambda close, free_market_cap, _kind=_kind, _fn=_fn: _fn(close, free_market_cap, _kind),
            extra_tags=["allow_panel_broadcast"])


# ---------------------------------------------------------------------------
# § Interval family (already part-covered in polars_next_stage; remainder here)
# ---------------------------------------------------------------------------

def _interval_rv(close: pl.DataFrame, start: int, end: int) -> pl.DataFrame:
    long = _with_mod(_with_date(_melt(close, "close"))).filter(pl.col("close").is_finite())
    long = long.filter((pl.col("mod") >= int(start)) & (pl.col("mod") <= int(end)))
    long = _log_returns_long(long)
    out = long.group_by(["date", "instrument"]).agg(
        (pl.col("r") * pl.col("r")).fill_nan(0.0).sum().alias("v")
    )
    return _pivot(out, "v")


_mk("intra_interval_realized_variance", "区间已实现方差（Polars）。", ["close", "start_minute", "end_minute"],
   lambda close, start_minute=570, end_minute=900: _interval_rv(close, int(start_minute), int(end_minute)))


def _interval_vwap_dev(close: pl.DataFrame, amount: pl.DataFrame, volume: pl.DataFrame, start: int, end: int) -> pl.DataFrame:
    long = _with_mod(_with_date(
        _melt(close, "close")
        .join(_melt(amount, "amount"), on=["ts", "instrument"], how="left")
        .join(_melt(volume, "volume"), on=["ts", "instrument"], how="left")
    ))
    seg = long.filter((pl.col("mod") >= int(start)) & (pl.col("mod") <= int(end)))
    seg = seg.sort(["date", "instrument", "ts"]).group_by(["date", "instrument"]).agg(
        pl.col("amount").fill_nan(0.0).sum().alias("a"),
        pl.col("volume").fill_nan(0.0).sum().alias("v"),
        pl.col("close").filter(pl.col("close").is_finite()).last().alias("clast"),
    )
    out = seg.with_columns(
        pl.when(pl.col("v") > _EPS)
        .then(pl.col("clast") / (pl.col("a") / pl.col("v")) - 1.0)
        .otherwise(None)
        .alias("vout")
    )
    return _pivot(out.select(["date", "instrument", pl.col("vout").alias("v")]), "v")


_mk("intra_interval_vwap_deviation", "区间 VWAP 偏离（Polars）。",
   ["close", "amount", "volume", "start_minute", "end_minute"],
   lambda close, amount, volume, start_minute=570, end_minute=900:
       _interval_vwap_dev(close, amount, volume, int(start_minute), int(end_minute)))


def _interval_illiq(close: pl.DataFrame, amount: pl.DataFrame, start: int, end: int) -> pl.DataFrame:
    long = _with_mod(_with_date(
        _melt(close, "close").join(_melt(amount, "amount"), on=["ts", "instrument"], how="left")
    )).filter(pl.col("close").is_finite())
    long = long.filter((pl.col("mod") >= int(start)) & (pl.col("mod") <= int(end)))
    long = _log_returns_long(long).with_columns(pl.col("amount").fill_nan(0.0))
    long = long.with_columns(
        pl.when(pl.col("r").is_finite())
        .then(pl.col("r").abs() / pl.max_horizontal(pl.col("amount"), pl.lit(_EPS)))
        .otherwise(None)
        .alias("ratio")
    )
    out = long.group_by(["date", "instrument"]).agg(
        pl.col("ratio").mean().alias("v"),
        pl.col("ratio").count().alias("n"),
    ).with_columns(pl.when(pl.col("n") > 0).then(pl.col("v")).otherwise(None))
    return _pivot(out, "v")


_mk("intra_interval_illiquidity", "区间 Amihud 非流动性（Polars）。",
   ["close", "amount", "start_minute", "end_minute"],
   lambda close, amount, start_minute=570, end_minute=900:
       _interval_illiq(close, amount, int(start_minute), int(end_minute)))


# ---------------------------------------------------------------------------
# § Drawdown / recovery
# ---------------------------------------------------------------------------

def _drawdown_stats(close: pl.DataFrame, kind: str) -> pl.DataFrame:
    """Drawdown statistics via the running peak (matches the pandas reference).

    ``dd = close / running_peak - 1``; the trough is ``argmin(dd)`` (first
    occurrence) and the peak is the running peak *at* the trough, i.e. the
    maximum price seen up to and including the trough.  This is the true max
    drawdown definition and is robust to a later global high.
    """
    long = _with_date(_melt(close, "close")).filter(pl.col("close").is_finite())
    long = long.sort(["date", "instrument", "ts"]).with_columns(
        pl.col("close").cum_count().over(["date", "instrument"]).cast(pl.Float64).alias("pos"),
        pl.col("close").cum_max().over(["date", "instrument"]).alias("run_peak"),
    ).with_columns((pl.col("close") / pl.col("run_peak") - 1.0).alias("dd"))
    g = long.group_by(["date", "instrument"]).agg(
        pl.len().alias("n"),
        pl.col("dd").arg_min().alias("trough_pos"),  # 0-based first minimum
        pl.col("dd").min().alias("min_dd"),
    ).with_columns(pl.col("trough_pos").cast(pl.Float64))
    j = long.join(g.select(["date", "instrument", "trough_pos", "n", "min_dd"]), on=["date", "instrument"], how="left")
    # trough value = close at the trough position (pos == trough_pos + 1)
    j = j.with_columns((pl.col("pos") == pl.col("trough_pos") + 1.0).alias("is_trough"))
    trough_val = j.filter(pl.col("is_trough")).select(["date", "instrument", "close"]).rename({"close": "trough"})
    j = j.join(trough_val, on=["date", "instrument"], how="left")
    # peak value = running peak at the trough; peak position = first ``pos`` within
    # [1, trough_pos+1] where close equals that peak (the first time the peak was hit).
    peak_v = j.filter(pl.col("is_trough")).select(["date", "instrument", "run_peak"]).rename({"run_peak": "peak"})
    j = j.join(peak_v, on=["date", "instrument"], how="left").with_columns(
        (pl.col("close") == pl.col("peak")).alias("is_peak")
    )
    peak_pos = (
        j.filter(pl.col("is_peak") & (pl.col("pos") <= pl.col("trough_pos") + 1.0))
        .group_by(["date", "instrument"])
        .agg(pl.col("pos").min().alias("peak_pos"))
    )
    j = j.join(peak_pos, on=["date", "instrument"], how="left")
    if kind == "depth":
        out = j.group_by(["date", "instrument"]).agg(
            pl.col("n").first().alias("n"),
            pl.col("peak").first().alias("peak"),
            pl.col("min_dd").first().alias("min_dd"),
        ).with_columns(
            pl.when((pl.col("n") >= 2) & pl.col("peak").is_not_null() & (pl.col("peak") > _EPS))
            .then(pl.col("min_dd")).otherwise(None).alias("v")
        )
        return _pivot(out, "v")
    if kind == "duration":
        out = j.group_by(["date", "instrument"]).agg(
            pl.col("n").first().alias("n"),
            pl.col("trough_pos").first().alias("trough_pos"),
            pl.col("peak_pos").first().alias("peak_pos"),
        ).with_columns(
            pl.when(pl.col("n") >= 2)
            .then(pl.when(pl.col("peak_pos").is_not_null()).then(pl.col("trough_pos") - pl.col("peak_pos") + 1.0).otherwise(0.0))
            .otherwise(None).alias("v")
        )
        return _pivot(out, "v")
    # recovery half-life: first pos >= trough_pos + 1 whose close reaches halfway,
    # expressed as the offset from the trough (matches ``finite[trough_idx:]``).
    j = j.with_columns((pl.col("trough") + 0.5 * (pl.col("peak") - pl.col("trough"))).alias("halfway"))
    out = j.group_by(["date", "instrument"]).agg(
        pl.col("n").first().alias("n"),
        pl.col("trough_pos").first().alias("trough_pos"),
        pl.col("trough").first().alias("trough"),
        pl.col("peak").first().alias("peak"),
        pl.when((pl.col("pos") >= pl.col("trough_pos") + 1.0) & (pl.col("close") >= pl.col("halfway")))
        .then(pl.col("pos")).otherwise(None).min().alias("recovered"),
    ).with_columns(
        pl.when(
            (pl.col("n") >= 2) & (pl.col("trough") > _EPS) & (pl.col("peak") > pl.col("trough")) & pl.col("recovered").is_not_null()
        ).then(pl.col("recovered") - 1.0 - pl.col("trough_pos")).otherwise(None).alias("v")
    )
    return _pivot(out, "v")


def _vwap_above_ratio(close: pl.DataFrame, amount: pl.DataFrame, volume: pl.DataFrame) -> pl.DataFrame:
    long = _with_date(
        _melt(close, "close")
        .join(_melt(amount, "amount"), on=["ts", "instrument"], how="left")
        .join(_melt(volume, "volume"), on=["ts", "instrument"], how="left")
    ).filter(pl.col("close").is_finite())
    long = long.with_columns(
        (pl.col("amount").is_finite() & pl.col("volume").is_finite() & (pl.col("volume") > 0)).alias("vv")
    )
    tot = long.group_by(["date", "instrument"]).agg(
        pl.when(pl.col("vv")).then(pl.col("volume")).otherwise(None).sum().alias("total_v"),
        pl.when(pl.col("vv")).then(pl.col("amount")).otherwise(None).sum().alias("total_a"),
    )
    long = long.join(tot, on=["date", "instrument"]).with_columns(
        pl.when(pl.col("total_v") > _EPS).then(pl.col("total_a") / pl.col("total_v")).otherwise(None).alias("day_vwap")
    )
    out = long.filter(pl.col("vv")).group_by(["date", "instrument"]).agg(
        (pl.col("close") > pl.col("day_vwap")).mean().alias("v")
    )
    return _pivot(out, "v")


_mk("intra_vwap_above_ratio", "收盘价高于当日 VWAP 的分钟占比（Polars）。", ["close", "amount", "volume"],
   lambda close, amount, volume: _vwap_above_ratio(close, amount, volume))


def _vwap_cross_count(close: pl.DataFrame, amount: pl.DataFrame, volume: pl.DataFrame) -> pl.DataFrame:
    long = _with_date(
        _melt(close, "close")
        .join(_melt(amount, "amount"), on=["ts", "instrument"], how="left")
        .join(_melt(volume, "volume"), on=["ts", "instrument"], how="left")
    )
    long = _cum_vwap_long(long)
    long = long.with_columns(
        (pl.col("close") - pl.col("cum_vwap")).sign().alias("sign")
    )
    nsig = long.group_by(["date", "instrument"]).agg(pl.col("sign").is_not_null().sum().alias("n"))
    long = long.filter(pl.col("sign").is_not_null()).with_columns(
        (pl.col("sign") != pl.col("sign").shift(1).over(["date", "instrument"])).alias("chg")
    )
    out = long.group_by(["date", "instrument"]).agg(pl.col("chg").sum().alias("chg"))
    out = out.join(nsig, on=["date", "instrument"]).with_columns(
        pl.when(pl.col("n") >= 2).then(pl.col("chg")).otherwise(0.0).alias("v")
    )
    return _pivot(out, "v")


_mk("intra_vwap_cross_count", "收盘价相对累计 VWAP 的方向变化次数（Polars）。", ["close", "amount", "volume"],
   lambda close, amount, volume: _vwap_cross_count(close, amount, volume))


def _return_activity_corr(close: pl.DataFrame, activity: pl.DataFrame, absolute_return: bool) -> pl.DataFrame:
    long = _with_date(
        _melt(close, "close").join(_melt(activity, "activity"), on=["ts", "instrument"], how="left")
    ).filter(pl.col("close").is_finite())
    long = _log_returns_long(long)
    rr = pl.col("r").abs() if absolute_return else pl.col("r")
    long = long.filter(pl.col("r").is_finite() & pl.col("activity").is_finite()).with_columns(rr.alias("rr"))
    g = long.group_by(["date", "instrument"]).agg(
        pl.len().alias("n"),
        pl.corr(pl.col("rr"), pl.col("activity")).alias("c"),
    )
    out = g.with_columns(pl.when(pl.col("n") >= 2).then(pl.col("c")).otherwise(None).alias("v"))
    return _pivot(out, "v")


_mk("intra_return_activity_corr", "分钟收益与成交活动相关性（可选绝对值）（Polars）。",
   ["close", "activity", "absolute_return"],
   lambda close, activity, absolute_return=False: _return_activity_corr(close, activity, bool(absolute_return)))


_mk("intra_drawdown_depth", "最大回撤深度（Polars）。", ["close"],
   lambda close: _drawdown_stats(close, "depth"))
_mk("intra_drawdown_duration", "最大回撤持续分钟数（Polars）。", ["close"],
   lambda close: _drawdown_stats(close, "duration"))
_mk("intra_drawdown_recovery_half_life", "回撤半恢复期（Polars）。", ["close"],
   lambda close: _drawdown_stats(close, "recovery"))


__all__ = [
    "intra_segment_return", "intra_segment_volume_share", "intra_segment_amount_share",
    "intra_segment_vwap_deviation", "intra_segment_realized_vol",
    "intra_realized_variance", "intra_realized_semivariance", "intra_bipower_variation",
    "intra_jump_ratio", "intra_path_efficiency", "intra_high_time", "intra_low_time",
    "intra_vwap_above_ratio", "intra_vwap_cross_count", "intra_concentration",
    "intra_entropy", "intra_signed_imbalance_proxy", "intra_return_activity_corr",
    "intra_amihud", "intra_kyle_lambda_proxy", "intra_extreme_bar_return",
    "intra_lunch_gap_return", "intra_limit_first_hit_time", "intra_limit_duration",
    "intra_limit_reopen_count", "intra_interval_realized_variance",
    "intra_interval_vwap_deviation", "intra_interval_illiquidity",
    "intra_same_slot_momentum", "intra_same_slot_reversal",
    "intra_return_profile_cosine", "intra_volume_profile_cosine", "intra_amount_profile_cosine",
    "intra_volume_profile_jsd", "intra_amount_profile_jsd", "intra_profile_earth_mover_distance",
    "intra_tripower_quarticity", "intra_jump_first_time", "intra_jump_last_time", "intra_jump_clustering",
    "intra_realized_beta", "intra_realized_correlation",
    "intra_down_down_semibeta", "intra_up_up_semibeta", "intra_down_up_semibeta", "intra_up_down_semibeta",
    "intra_beta_asymmetry", "intra_idiosyncratic_variance", "intra_idiosyncratic_skewness",
    "intra_idiosyncratic_kurtosis", "intra_market_model_r2",
    "intra_drawdown_depth", "intra_drawdown_duration", "intra_drawdown_recovery_half_life",
    "intra_vwap_path_slope", "intra_vwap_path_curvature",
    "intra_price_vwap_max_positive_excursion", "intra_price_vwap_max_negative_excursion",
    "intra_time_above_vwap", "intra_longest_above_vwap_streak", "intra_longest_below_vwap_streak",
    "intra_vwap_reversion_speed",
]
