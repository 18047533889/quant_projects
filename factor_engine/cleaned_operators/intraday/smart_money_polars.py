# -*- coding: utf-8 -*-
"""Polars backend for smart money intraday operators.

Native Polars implementations for:
- intra_dynamic_stock_graph_features
- intra_common_trading_intensity
- intra_local_conditional_entropy
- intra_smart_money_vwap_ratio
"""
from __future__ import annotations

import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_TIME_COLS = frozenset({"date", "timestamp", "time", "QuoteTime", "TradeDate"})
_EPS = 1e-12
_SESSION_TZ = "Asia/Shanghai"


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


def _with_date(long: pl.DataFrame) -> pl.DataFrame:
    return long.with_columns(pl.col("ts").dt.date().alias("date"))


def _pivot(df: pl.DataFrame, value: str) -> pl.DataFrame:
    piv = df.pivot(index="date", on="instrument", values=value)
    return piv.fill_null(float("nan"))


def _mk(canonical: str, description: str, params: list[str], fn, extra_tags=None):
    tags = ["polars", "intraday", "minute", "native", "typed_v2", "smart_money"]
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
        canonical,
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series},
    )
    register_operator(
        name=canonical,
        category="intraday_microstructure",
        business_category="intraday_smart_money",
        canonical=canonical,
        source="intraday.smart_money_polars",
        backend="polars",
        status="experimental",
    )(cls)


# ---------------------------------------------------------------------------
# § Graph features (autocorrelation proxy)
# ---------------------------------------------------------------------------

def _stock_graph_features(close: pl.DataFrame) -> pl.DataFrame:
    """First-order return autocorrelation as a proxy for co-movement."""
    long = _with_date(_melt(close, "close")).filter(pl.col("close").is_finite())
    long = long.sort(["instrument", "date", "ts"])
    long = long.with_columns(
        pl.col("close").log().diff().over(["instrument", "date"]).alias("r")
    )
    long = long.with_columns(
        pl.col("r").shift(1).over(["instrument", "date"]).alias("r_lag")
    ).filter(pl.col("r").is_finite() & pl.col("r_lag").is_finite())

    out = long.group_by(["date", "instrument"]).agg(
        pl.col("r").count().alias("n"),
        pl.col("r").mean().alias("r_mean"),
        pl.col("r_lag").mean().alias("rlag_mean"),
        (pl.col("r") * pl.col("r_lag")).mean().alias("rr_mean"),
        (pl.col("r") * pl.col("r")).mean().alias("r2"),
        (pl.col("r_lag") * pl.col("r_lag")).mean().alias("rlag2"),
    )

    out = out.with_columns(
        (pl.col("rr_mean") - pl.col("r_mean") * pl.col("rlag_mean")).alias("cov"),
        ((pl.col("r2") - pl.col("r_mean") ** 2) * (pl.col("rlag2") - pl.col("rlag_mean") ** 2)).sqrt().alias("std_prod"),
    )

    out = out.with_columns(
        pl.when((pl.col("n") >= 5) & (pl.col("std_prod") > _EPS))
        .then(pl.col("cov") / pl.col("std_prod"))
        .otherwise(None)
        .alias("v")
    )

    return _pivot(out, "v")


_mk("intra_dynamic_stock_graph_features", "日内股票关联图动态特征（Polars）。", ["close"],
   lambda close: _stock_graph_features(close))


# ---------------------------------------------------------------------------
# § Common trading intensity
# ---------------------------------------------------------------------------

def _common_trading_intensity(volume: pl.DataFrame, amount: pl.DataFrame) -> pl.DataFrame:
    """Peak/mean volume ratio * concentration."""
    vol_long = _with_date(_melt(volume, "volume")).filter(pl.col("volume").is_finite() & (pl.col("volume") > 0))
    amt_long = _with_date(_melt(amount, "amount")).filter(pl.col("amount").is_finite())

    long = vol_long.join(amt_long, on=["ts", "instrument", "date"], how="inner")

    agg = long.group_by(["date", "instrument"]).agg(
        pl.col("volume").count().alias("n"),
        pl.col("volume").mean().alias("mean_vol"),
        pl.col("volume").max().alias("max_vol"),
        pl.col("volume").sum().alias("total_vol"),
    )

    # Concentration: need per-bar shares squared
    long = long.join(agg.select(["date", "instrument", "total_vol"]), on=["date", "instrument"])
    long = long.with_columns((pl.col("volume") / pl.col("total_vol")).alias("share")) if pl.col("total_vol")).alias("share")) > 1e-10 else np.nan

    conc = long.group_by(["date", "instrument"]).agg(
        (pl.col("share") ** 2).sum().alias("concentration")
    )

    out = agg.join(conc, on=["date", "instrument"])

    out = out.with_columns(
        pl.when((pl.col("n") >= 3) & (pl.col("mean_vol") > _EPS))
        .then((pl.col("max_vol") / pl.col("mean_vol")) * pl.col("concentration"))
        .otherwise(None)
        .alias("v")
    )

    return _pivot(out, "v")


_mk("intra_common_trading_intensity", "日内共同交易强度（Polars）。", ["volume", "amount"],
   lambda volume, amount: _common_trading_intensity(volume, amount))


# ---------------------------------------------------------------------------
# § Local conditional entropy
# ---------------------------------------------------------------------------

def _local_conditional_entropy(close: pl.DataFrame, volume: pl.DataFrame) -> pl.DataFrame:
    """Conditional entropy H(Direction | VolumeState)."""
    close_long = _with_date(_melt(close, "close")).filter(pl.col("close").is_finite())
    vol_long = _with_date(_melt(volume, "volume")).filter(pl.col("volume").is_finite() & (pl.col("volume") > 0))

    long = close_long.join(vol_long, on=["ts", "instrument", "date"], how="inner")
    long = long.sort(["instrument", "date", "ts"])

    # Returns and direction
    long = long.with_columns(
        pl.col("close").log().diff().over(["instrument", "date"]).alias("r")
    )
    long = long.with_columns(pl.col("r").sign().alias("direction"))

    # Align volume with returns
    long = long.filter(pl.col("r").is_finite())

    # Volume terciles per (date, instrument)
    long = long.with_columns(
        pl.col("volume").quantile(0.3333).over(["date", "instrument"]).alias("vol_q33"),
        pl.col("volume").quantile(0.6667).over(["date", "instrument"]).alias("vol_q67"),
    )

    long = long.with_columns(
        pl.when(pl.col("volume") < pl.col("vol_q33"))
        .then(pl.lit(0))
        .when(pl.col("volume") < pl.col("vol_q67"))
        .then(pl.lit(1))
        .otherwise(pl.lit(2))
        .alias("vol_state")
    )

    # Count occurrences per (date, instrument, vol_state, direction)
    counts = long.group_by(["date", "instrument", "vol_state", "direction"]).agg(
        pl.col("direction").count().alias("count")
    )

    # Total per (date, instrument)
    totals = counts.group_by(["date", "instrument"]).agg(pl.col("count").sum().alias("total"))

    # Total per (date, instrument, vol_state)
    state_totals = counts.group_by(["date", "instrument", "vol_state"]).agg(
        pl.col("count").sum().alias("state_count")
    )

    # Join and compute probabilities
    counts = counts.join(totals, on=["date", "instrument"])
    counts = counts.join(state_totals, on=["date", "instrument", "vol_state"])

    counts = counts.with_columns(
        pl.when(pl.col("total") != 0).then(pl.col("state_count") / pl.col("total")).otherwise(None).alias("p_vs"),
        pl.when(pl.col("state_count") != 0).then(pl.col("count") / pl.col("state_count")).otherwise(None).alias("p_d_given_vs"),
    )

    # Entropy: -sum(p_vs * p_d_given_vs * log(p_d_given_vs))
    counts = counts.with_columns(
        pl.when(pl.col("p_d_given_vs") > _EPS)
        .then(-pl.col("p_vs") * pl.col("p_d_given_vs") * pl.col("p_d_given_vs").log())
        .otherwise(pl.lit(0.0))
        .alias("entropy_term")
    )

    out = counts.group_by(["date", "instrument"]).agg(
        pl.col("entropy_term").sum().alias("H"),
        pl.col("total").first().alias("n"),
    )

    out = out.with_columns(
        pl.when(pl.col("n") >= 10)
        .then(pl.col("H"))
        .otherwise(None)
        .alias("v")
    )

    return _pivot(out, "v")


_mk("intra_local_conditional_entropy", "日内局部条件熵（Polars）。", ["close", "volume"],
   lambda close, volume: _local_conditional_entropy(close, volume))


# ---------------------------------------------------------------------------
# § Smart money VWAP ratio
# ---------------------------------------------------------------------------

def _smart_money_vwap_ratio(close: pl.DataFrame, amount: pl.DataFrame, volume: pl.DataFrame) -> pl.DataFrame:
    """Large-trade VWAP / session VWAP."""
    close_long = _with_date(_melt(close, "close")).filter(pl.col("close").is_finite())
    amt_long = _with_date(_melt(amount, "amount")).filter(pl.col("amount").is_finite())
    vol_long = _with_date(_melt(volume, "volume")).filter(pl.col("volume").is_finite() & (pl.col("volume") > 0))

    long = close_long.join(amt_long, on=["ts", "instrument", "date"], how="inner")
    long = long.join(vol_long, on=["ts", "instrument", "date"], how="inner")

    # Session VWAP
    session = long.group_by(["date", "instrument"]).agg(
        pl.col("amount").sum().alias("total_amt"),
        pl.col("volume").sum().alias("total_vol"),
        pl.col("amount").quantile(0.75).alias("amt_q75"),
        pl.col("amount").count().alias("n"),
    )

    session = session.with_columns(
        pl.when(pl.col("total_vol") != 0).then(pl.col("total_amt") / pl.col("total_vol")).otherwise(None).alias("session_vwap")
    )

    # Join threshold back
    long = long.join(session.select(["date", "instrument", "amt_q75", "session_vwap"]), on=["date", "instrument"])

    # Large trades
    large = long.filter(pl.col("amount") >= pl.col("amt_q75"))

    large_agg = large.group_by(["date", "instrument"]).agg(
        pl.col("amount").sum().alias("large_amt"),
        pl.col("volume").sum().alias("large_vol"),
        pl.col("amount").count().alias("large_n"),
        pl.col("session_vwap").first().alias("session_vwap"),
    )

    large_agg = large_agg.with_columns(
        pl.when(pl.col("large_vol") != 0).then(pl.col("large_amt") / pl.col("large_vol")).otherwise(None).alias("large_vwap")
    )

    out = large_agg.with_columns(
        pl.when(
            (pl.col("large_n") >= 2)
            & (pl.col("large_vol") > _EPS)
            & (pl.col("session_vwap") > _EPS)
        )
        .then(pl.col("large_vwap") / pl.col("session_vwap"))
        .otherwise(None)
        .alias("v")
    )

    # Need to join back to full (date, instrument) grid
    all_pairs = session.select(["date", "instrument"])
    out = all_pairs.join(out.select(["date", "instrument", "v"]), on=["date", "instrument"], how="left")

    return _pivot(out, "v")


_mk("intra_smart_money_vwap_ratio", "智能资金 VWAP 比率（Polars）。", ["close", "amount", "volume"],
   lambda close, amount, volume: _smart_money_vwap_ratio(close, amount, volume))
