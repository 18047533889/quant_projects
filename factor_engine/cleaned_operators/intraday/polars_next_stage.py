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

from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.base import ParamRole, ParamSpec

_TIME_COLS = frozenset({"date", "timestamp", "time", "QuoteTime", "TradeDate"})
_EPS = 1e-12
_SESSION_TZ = "Asia/Shanghai"
_TZ_SPEC = ParamSpec(dtype=str, default=None, searchable=False, param_role=ParamRole.POLICY)


def _time_col(df: pl.DataFrame) -> str:
    for c in df.columns:
        if c in _TIME_COLS:
            return c
    return "date"


def _session_tz_panel(df: pl.DataFrame, session_tz: str | None) -> pl.DataFrame:
    tc = _time_col(df)
    if getattr(df.schema[tc], "time_zone", None):
        return df.with_columns(
            pl.col(tc).dt.convert_time_zone(session_tz or _SESSION_TZ).dt.replace_time_zone(None).alias(tc)
        )
    return df


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


def _daily_like(df: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    """Keep source date/symbol identities when estimates are unavailable."""
    tc = _time_col(source)
    columns = [c for c in source.columns if c != tc]
    dates = source.select(pl.col(tc).cast(pl.Datetime).dt.date().alias("date")).unique()
    grid = dates.join(pl.DataFrame({"instrument": columns}, schema={"instrument": pl.String}), how="cross")
    aligned = grid.join(df.select("date", "instrument", "v"), on=["date", "instrument"], how="left")
    result = _pivot(aligned, "v")
    missing = [c for c in columns if c not in result.columns]
    if missing:
        result = result.with_columns([pl.lit(float("nan"), dtype=pl.Float64).alias(c) for c in missing])
    return result.select("date", *columns).sort("date")


# ---------------------------------------------------------------------------
# R57 backend-coverage batch 3 — explicit execution-kind declarations.
# These kernels are genuine polars expressions (pl.col / with_columns /
# group_by over pl.Expr).  Previously they had no _physical_spec, so
# canonical_polars_kind(production_mode=True) failed closed to UNSUPPORTED.
# ---------------------------------------------------------------------------
from factor_engine.backend.contracts import (
    ExecutionKind,
    PhysicalImplementationSpec,
)

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
        implementation_source_hash=f"intraday.polars_next_stage:{kernel}:v1",
        emitter_identity=f"polars_expr:{canonical}",
        kernel_identity=f"intraday.polars_next_stage:{kernel}",
        parameter_domain_hash=f"{canonical}:declared:v1",
        semantic_contract_hash=f"{canonical}:polars_native_expr:v1",
        notes=_BATCH3_NOTE,
    )


_NATIVE_SPECS: dict[str, PhysicalImplementationSpec] = {
    _c: _batch3_native_spec(_c, _k)
    for _c, _k in (
        ("intra_realized_skewness", "IntradayPolars_intra_realized_skewness"),
        ("intra_realized_kurtosis", "IntradayPolars_intra_realized_kurtosis"),        ("intra_signed_jump_ratio", "IntradayPolars_intra_signed_jump_ratio"),
        ("intra_jump_variation", "IntradayPolars_intra_jump_variation"),

    )
}


def _mk(canonical: str, description: str, params: list[str], fn, panel_params=None, scalar_params=None, param_specs=None):
    metadata = OperatorMetadata(
        name=canonical,
        category="intraday_microstructure",
        description=description,
        param_names=params,
        return_type="series",
        tags=["polars", "intraday", "minute", "native", "typed_v2"],
        panel_params=tuple(panel_params or ()),
        panel_arity=len(panel_params) if panel_params else None,
        scalar_params=tuple(scalar_params or ()),
        param_specs=dict(param_specs or {}),
    )

    def _calculate_series(self, *args, **kwargs):
        return fn(*args, **kwargs)

    cls = type(
        f"IntradayPolars_{canonical}",
        (SeriesOperator,),
        {
            "metadata": metadata,
            "_calculate_series": _calculate_series,
            "__module__": __name__,
            **({"_physical_spec": _NATIVE_SPECS[canonical]}
               if canonical in _NATIVE_SPECS else {}),
        },
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
    if not math.isfinite(threshold_scale) or threshold_scale <= 0.0:
        raise ValueError("threshold_scale must be a finite number > 0")
    long = _melt(close, "close")
    long = long.with_columns(pl.col("ts").dt.date().alias("date"))
    long = _returns(long)
    r = pl.col("r")
    finite_r = r.is_finite().fill_null(False)
    rv = long.group_by(["date", "instrument"]).agg(
        finite_r.cast(pl.UInt32).sum().alias("n"),
        pl.when(finite_r).then(r * r).otherwise(None).sum().alias("rv"),
    ).with_columns(
        pl.when((pl.col("n") >= 2) & pl.col("rv").is_finite() & (pl.col("rv") > _EPS))
        .then((pl.col("rv") / pl.col("n")).sqrt() * threshold_scale)
        .otherwise(None)
        .alias("thresh")
    )
    long = long.join(rv, on=["date", "instrument"])
    long = long.with_columns(
        (finite_r & pl.col("thresh").is_not_null() & (r.abs() > pl.col("thresh")))
        .fill_null(False)
        .alias("jump")
    )
    jr = pl.col("r")
    jump = pl.col("jump")
    jump_sq = pl.when(jump).then(jr * jr).otherwise(0.0)
    stats = long.group_by(["date", "instrument"]).agg(
        jump.sum().alias("count"),
        pl.when(jump & (jr > 0)).then(jr * jr).otherwise(0.0).sum().alias("pos"),
        pl.when(jump & (jr < 0)).then(jr * jr).otherwise(0.0).sum().alias("neg"),
        jump_sq.sum().alias("total"),
        jump_sq.pow(2).sum().alias("conc_num"),
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
    return _daily_like(out, close)


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
        (pl.col("ts").dt.hour().cast(pl.Int64) * 60 + pl.col("ts").dt.minute().cast(pl.Int64)).alias("mod"),
    )
    mask = (pl.col("mod") >= int(start)) & (pl.col("mod") <= int(end))
    seg = long.filter(mask).group_by(["date", "instrument"]).agg(
        pl.col("close").first().alias("open"),
        pl.col("close").last().alias("last"),
    )
    out = seg.with_columns(((pl.col("last") / pl.col("open")) - 1.0).alias("v"))
    return _pivot(out, "v")


_mk("intra_interval_return", "指定分钟区间收益（Polars）。", ["close", "start_minute", "end_minute", "session_tz"],
   lambda close, start_minute=570, end_minute=900, session_tz=None: _interval_ret(_session_tz_panel(close, session_tz), int(start_minute), int(end_minute)), panel_params=("close",), scalar_params=("start_minute", "end_minute", "session_tz"), param_specs={"start_minute": ParamSpec(dtype=int, min=0, max=1440, default=570, param_role=ParamRole.STATE_THRESHOLD), "end_minute": ParamSpec(dtype=int, min=0, max=1440, default=900, param_role=ParamRole.STATE_THRESHOLD), "session_tz": _TZ_SPEC})


def _interval_share(value: pl.DataFrame, start: int, end: int) -> pl.DataFrame:
    long = _melt(value, "value")
    long = long.with_columns(
        pl.col("ts").dt.date().alias("date"),
        (pl.col("ts").dt.hour().cast(pl.Int64) * 60 + pl.col("ts").dt.minute().cast(pl.Int64)).alias("mod"),
    )
    total = long.group_by(["date", "instrument"]).agg(pl.col("value").sum().alias("total"))
    mask = (pl.col("mod") >= int(start)) & (pl.col("mod") <= int(end))
    seg = long.filter(mask).group_by(["date", "instrument"]).agg(pl.col("value").sum().alias("seg"))
    merged = seg.join(total, on=["date", "instrument"])
    out = merged.with_columns((pl.col("seg") / pl.col("total")).alias("v"))
    return _pivot(out, "v")


_mk("intra_interval_volume_share", "区间成交量占比（Polars）。", ["volume", "start_minute", "end_minute", "session_tz"],
   lambda volume, start_minute=570, end_minute=900, session_tz=None: _interval_share(_session_tz_panel(volume, session_tz), int(start_minute), int(end_minute)), panel_params=("volume",), scalar_params=("start_minute", "end_minute", "session_tz"), param_specs={"start_minute": ParamSpec(dtype=int, min=0, max=1440, default=570, param_role=ParamRole.STATE_THRESHOLD), "end_minute": ParamSpec(dtype=int, min=0, max=1440, default=900, param_role=ParamRole.STATE_THRESHOLD), "session_tz": _TZ_SPEC})
_mk("intra_interval_amount_share", "区间成交额占比（Polars）。", ["amount", "start_minute", "end_minute", "session_tz"],
   lambda amount, start_minute=570, end_minute=900, session_tz=None: _interval_share(_session_tz_panel(amount, session_tz), int(start_minute), int(end_minute)), panel_params=("amount",), scalar_params=("start_minute", "end_minute", "session_tz"), param_specs={"start_minute": ParamSpec(dtype=int, min=0, max=1440, default=570, param_role=ParamRole.STATE_THRESHOLD), "end_minute": ParamSpec(dtype=int, min=0, max=1440, default=900, param_role=ParamRole.STATE_THRESHOLD), "session_tz": _TZ_SPEC})


# ---------------------------------------------------------------------------
# § Drawdown / drawup
# ---------------------------------------------------------------------------

def _max_path(close: pl.DataFrame, side: str) -> pl.DataFrame:
    long = _melt(close, "close")
    long = long.with_columns(pl.col("ts").dt.date().alias("date"))
    finite = long.filter(pl.col("close").is_finite().fill_null(False)).sort(
        ["date", "instrument", "ts"]
    )
    finite = finite.with_columns(
        pl.col("close").cum_max().over(["date", "instrument"]).alias("run_max"),
        pl.col("close").cum_min().over(["date", "instrument"]).alias("run_min"),
    )
    if side == "down":
        out = finite.group_by(["date", "instrument"]).agg(
            pl.len().alias("n"),
            (pl.col("run_max") == 0).any().alias("undefined_reference"),
            pl.when(pl.col("run_max") != 0)
            .then(pl.col("close") / pl.col("run_max") - 1.0)
            .otherwise(None)
            .min()
            .alias("path_v"),
        )
    else:
        out = finite.group_by(["date", "instrument"]).agg(
            pl.len().alias("n"),
            (pl.col("run_min") == 0).any().alias("undefined_reference"),
            pl.when(pl.col("run_min") != 0)
            .then(pl.col("close") / pl.col("run_min") - 1.0)
            .otherwise(None)
            .max()
            .alias("path_v"),
        )
    out = out.with_columns(
        pl.when((pl.col("n") >= 2) & ~pl.col("undefined_reference")).then(pl.col("path_v")).otherwise(None).alias("v")
    )
    return _daily_like(out, close)


_mk("intra_max_drawdown", "日内最大回撤（Polars）。", ["close"],
   lambda close: _max_path(close, "down"))
_mk("intra_max_drawup", "日内最大上涨段（Polars）。", ["close"],
   lambda close: _max_path(close, "up"))
