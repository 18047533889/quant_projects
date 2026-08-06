# -*- coding: utf-8
"""Minute → daily parity harness for the intraday aggregation operators.

为什么需要独立的分钟 parity 管线
-------------------------------
The 25 ``intra_*`` operators consume a **minute** panel (one value per minute
bar, one column per instrument) and emit a **daily** panel.  They cannot be
certified through the daily-primitive six-way parity fixture: a daily fixture
contains one row per instrument-day, so any minute kernel evaluated on it
degenerates to a single-bar day and the three backends trivially agree on all-
NaN output — that is *fake* parity.  Real evidence must compare the three
backends on genuinely minute-shaped data, which is what this module does.

Convention
----------
* The pandas implementation from ``cleaned_operators.microstructure.intraday_agg``
  is the semantic reference (it is the production backend).
* For each operator we implement an *independent* native Polars expression and
  (where SQL-natural) a DuckDB SQL query over the same minute-long frame, and
  assert per-(date, instrument) value parity.
* Operators whose kernel math is order/tie/population-variance sensitive
  (``intra_kyle_lambda_proxy``) or needs daily limit broadcasting are either
  certified on a reduced backend set or left reference-only; the report records
  exactly which backends parity passed on — never fabricated.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

_EPS = 1e-12
_MORNING = (570, 690)   # 09:30 .. 11:30 in minute-of-day
_AFTERNOON = (780, 900)  # 13:00 .. 15:00
_SEGMENTS = {"morning": _MORNING, "afternoon": _AFTERNOON}
_SESSION_TZ = "Asia/Shanghai"
_SCALE = 1e8


def _pl():
    import polars as pl

    return pl


def _col(name: str):
    return _pl().col(name)


# ---------------------------------------------------------------------------
# Long-frame construction
# ---------------------------------------------------------------------------

def _session_local_index(index: pd.DatetimeIndex, tz: str | None = None) -> pd.DatetimeIndex:
    """Mirror intraday_agg._session_local: tz-aware UTC → Asia/Shanghai naive."""
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        idx = index.tz_convert(tz or _SESSION_TZ)
        return pd.DatetimeIndex(idx.tz_localize(None))
    return pd.DatetimeIndex(index)


def build_long(panels: dict[str, pd.DataFrame], *, session_tz: str | None = None):
    """Wide minute panels -> Polars long frame.

    Columns: ts (session-local naive datetime), date, minute, inst, plus one
    value column per input panel name.
    """
    pl = _pl()
    first_name = next(iter(panels))
    first = panels[first_name]
    idx = _session_local_index(first.index, session_tz)
    dates = idx.normalize()
    minutes = idx.hour * 60 + idx.minute
    rows = []
    for inst in first.columns:
        for k, ts in enumerate(idx):
            row = {
                "ts": ts,
                "date": dates[k].date(),
                "minute": int(minutes[k]),
                "inst": inst,
            }
            for name, panel in panels.items():
                v = panel.iloc[k][inst]
                row[name] = v if pd.notna(v) else None
            rows.append(row)
    return pl.DataFrame(rows).sort(["date", "inst", "ts"])


def _to_wide(long_out) -> pd.DataFrame:
    """Long (date, inst, value) -> wide daily panel matching the pandas shape."""
    pdf = long_out.select(["date", "inst", "value"]).to_pandas()
    wide = pdf.pivot_table(index="date", columns="inst", values="value").sort_index()
    wide.index = pd.DatetimeIndex(wide.index)
    wide.columns.name = None
    return wide


def _with_r(df):
    """Attach log returns per (date, inst): r_t = ln(close_t / close_{t-1})."""
    pl = _pl()
    return df.with_columns(
        (_col("close") / _col("close").shift(1)).log().over(["date", "inst"]).alias("r")
    )


def _segment_mask_expr(segment: str):
    lo, hi = _SEGMENTS[segment]
    return (_col("minute") >= lo) & (_col("minute") <= hi)


def _hit_condition(side: str):
    """pandas _limit_mask: finite close & finite limit, within EPS of the limit."""
    pl = _pl()
    cond = (
        _col("_c") >= _col("_lim") - _EPS
        if side == "up"
        else _col("_c") <= _col("_lim") + _EPS
    )
    return (
        pl.when(_col("_c").is_not_null() & _col("_lim").is_not_null())
        .then(cond)
        .otherwise(False)
    )


def _with_limit(df, limit_panel, side: str):
    pl = _pl()
    lim = _to_limit_long(limit_panel).rename({"high_limit": "_lim"})
    return df.join(lim, on=["date", "inst"], how="left").with_columns(
        _col("close").alias("_c")
    )


def _to_limit_long(limit_panel: pd.DataFrame):
    pl = _pl()
    rows = []
    for date, row in limit_panel.iterrows():
        for inst, v in row.items():
            rows.append({"date": pd.Timestamp(date).date(), "inst": inst, "v": v})
    return pl.DataFrame(rows).rename({"v": "high_limit"})


# ---------------------------------------------------------------------------
# Realized variance / semivariance / bipower / jump
# ---------------------------------------------------------------------------

def pl_realized_variance(df):
    pl = _pl()
    d = _with_r(df)
    out = (
        d.group_by(["date", "inst"])
        .agg(
            pl.when(_col("close").is_not_null().any())
            .then(
                pl.when(_col("r").is_not_null())
                .then(_col("r") * _col("r"))
                .otherwise(0.0)
                .sum()
            )
            .otherwise(None)
            .alias("value")
        )
        .sort(["date", "inst"])
    )
    return _to_wide(out)


def pl_realized_semivariance(df, side: str = "down"):
    pl = _pl()
    d = _with_r(df)
    cond = _col("r") < 0 if side == "down" else _col("r") > 0
    out = (
        d.group_by(["date", "inst"])
        .agg(
            pl.when(_col("close").is_not_null().any())
            .then(
                pl.when(cond)
                .then(_col("r") * _col("r"))
                .otherwise(0.0)
                .sum()
            )
            .otherwise(None)
            .alias("value")
        )
        .sort(["date", "inst"])
    )
    return _to_wide(out)


def pl_bipower_variation(df):
    """(pi/2)*sum(|r_i|*|r_{i-1}|) over consecutive *finite* returns (compacted)."""
    pl = _pl()
    finite = _with_r(df).filter(_col("r").is_not_null()).with_columns(
        _col("r").abs().alias("ra")
    )
    out = (
        finite.group_by(["date", "inst"])
        .agg(_col("ra").alias("rf"))
        .with_columns(_col("rf").list.len().alias("_n"))
        .with_columns(
            pl.when(_col("_n") >= 2)
            .then(
                (
                    _col("rf").list.tail(_col("_n") - 1)
                    * _col("rf").list.head(_col("_n") - 1)
                ).list.sum()
                * (math.pi / 2.0)
            )
            .otherwise(None)
            .alias("value")
        )
        .select(["date", "inst", "value"])
        .sort(["date", "inst"])
    )
    return _to_wide(out)


def _jump_from_rv_bv(rv: pd.DataFrame, bv: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=rv.index, columns=rv.columns, dtype=float)
    for col in out.columns:
        res = []
        for x, y in zip(rv[col].to_numpy(dtype=float), bv[col].to_numpy(dtype=float)):
            if not (np.isfinite(x) and np.isfinite(y)) or x <= _EPS:
                res.append(np.nan)
            else:
                res.append(max(x - y, 0.0) / x)
        out[col] = res
    return out


def pl_jump_ratio(df):
    return _jump_from_rv_bv(pl_realized_variance(df), pl_bipower_variation(df))


# ---------------------------------------------------------------------------
# Price path
# ---------------------------------------------------------------------------

def pl_path_efficiency(df):
    pl = _pl()
    d = df.filter(_col("close").is_not_null())
    out = (
        d.group_by(["date", "inst"])
        .agg(_col("close").alias("cf"))
        .with_columns(_col("cf").list.len().alias("_n"))
        .with_columns(
            _col("cf").list.diff().alias("_changes"),
            (_col("cf").list.get(-1) - _col("cf").list.get(0)).abs().alias("_net"),
        )
        .with_columns(
            _col("_changes").list.eval(pl.element().abs()).list.sum().alias("_length")
        )
        .with_columns(
            pl.when(_col("_n") < 2)
            .then(None)
            .when(_col("_length") <= _EPS)
            .then(0.0)
            .otherwise(_col("_net") / _col("_length"))
            .alias("value")
        )
        .select(["date", "inst", "value"])
        .sort(["date", "inst"])
    )
    return _to_wide(out)


def pl_high_time(df, *, low: bool):
    pl = _pl()
    col_name = "low" if low else "high"
    d = df.sort(["date", "inst", "ts"]).with_columns(
        pl.int_range(0, pl.len()).over(["date", "inst"]).alias("_pos")
    )
    finite = d.filter(_col(col_name).is_not_null())
    extreme = (
        _col(col_name) == _col(col_name).min().over(["date", "inst"])
        if low
        else _col(col_name) == _col(col_name).max().over(["date", "inst"])
    )
    first = (
        finite.with_columns(extreme.alias("_m"))
        .filter(_col("_m"))
        .group_by(["date", "inst"])
        .agg(_col("_pos").min().alias("_p"))
    )
    counts = finite.group_by(["date", "inst"]).agg(pl.len().alias("_n"))
    out = (
        first.join(counts, on=["date", "inst"], how="left")
        .with_columns((_col("_p") / _col("_n")).alias("value"))
        .select(["date", "inst", "value"])
        .sort(["date", "inst"])
    )
    return _to_wide(out)


# ---------------------------------------------------------------------------
# Segment aggregation
# ---------------------------------------------------------------------------

def pl_segment_return(df, segment: str):
    pl = _pl()
    d = df.filter(_segment_mask_expr(segment)).filter(_col("close").is_not_null())
    out = (
        d.group_by(["date", "inst"])
        .agg(
            pl.len().alias("_n"),
            (_col("close").last() / _col("close").first() - 1.0).alias("value"),
        )
        .with_columns(
            pl.when(_col("_n") >= 2).then(_col("value")).otherwise(None).alias("value")
        )
        .select(["date", "inst", "value"])
        .sort(["date", "inst"])
    )
    return _to_wide(out)


def _segment_share(df, segment: str, name: str):
    pl = _pl()
    d = df.with_columns(_segment_mask_expr(segment).alias("_seg"))
    out = (
        d.group_by(["date", "inst"])
        .agg(
            _col(name).filter(_col("_seg")).sum().alias("_seg_sum"),
            _col(name).sum().alias("_tot"),
        )
        .with_columns(
            pl.when(_col("_tot") > _EPS)
            .then(_col("_seg_sum") / _col("_tot"))
            .otherwise(None)
            .alias("value")
        )
        .select(["date", "inst", "value"])
        .sort(["date", "inst"])
    )
    return _to_wide(out)


def pl_segment_volume_share(df, segment: str):
    return _segment_share(df, segment, "volume")


def pl_segment_amount_share(df, segment: str):
    return _segment_share(df, segment, "amount")


def pl_segment_realized_vol(df, segment: str):
    pl = _pl()
    d = (
        df.filter(_segment_mask_expr(segment))
        .sort(["date", "inst", "ts"])
        .with_columns(
            (_col("close") / _col("close").shift(1)).log().over(["date", "inst"]).alias("r")
        )
    )
    out = (
        d.group_by(["date", "inst"])
        .agg(
            pl.when(_col("close").is_not_null().any())
            .then(
                pl.when(_col("r").is_not_null())
                .then(_col("r") * _col("r"))
                .otherwise(0.0)
                .sum()
                .sqrt()
            )
            .otherwise(None)
            .alias("value")
        )
        .sort(["date", "inst"])
    )
    return _to_wide(out)


def pl_segment_vwap_deviation(df, segment: str):
    pl = _pl()
    d = df.filter(_segment_mask_expr(segment)).filter(_col("close").is_not_null())
    out = (
        d.group_by(["date", "inst"])
        .agg(
            _col("amount").sum().alias("_a"),
            _col("volume").sum().alias("_v"),
            _col("close").last().alias("_last"),
        )
        .with_columns(
            pl.when((_col("_v") <= _EPS) | _col("_last").is_null())
            .then(None)
            .otherwise(_col("_last") / (_col("_a") / _col("_v")) - 1.0)
            .alias("value")
        )
        .select(["date", "inst", "value"])
        .sort(["date", "inst"])
    )
    return _to_wide(out)


# ---------------------------------------------------------------------------
# VWAP behaviour
# ---------------------------------------------------------------------------

def pl_vwap_above_ratio(df):
    pl = _pl()
    valid_v = (
        _col("amount").is_not_null() & _col("volume").is_not_null() & (_col("volume") > 0)
    )
    d = df.filter(_col("close").is_not_null()).with_columns(
        pl.when(valid_v).then(_col("volume")).otherwise(0.0).alias("_v"),
        pl.when(valid_v).then(_col("amount")).otherwise(0.0).alias("_a"),
    )
    daily = (
        d.group_by(["date", "inst"])
        .agg(_col("_v").sum().alias("_tv"), _col("_a").sum().alias("_ta"))
        .with_columns(
            pl.when(_col("_tv") <= _EPS)
            .then(None)
            .otherwise(_col("_ta") / _col("_tv"))
            .alias("_vwap")
        )
        .select(["date", "inst", "_tv", "_vwap"])
    )
    joined = d.join(daily, on=["date", "inst"])
    out = (
        joined.filter(_col("_vwap").is_not_null() & _col("close").is_not_null())
        .with_columns((_col("close") > _col("_vwap")).alias("_above"))
        .group_by(["date", "inst"])
        .agg(_col("_above").mean().alias("value"))
        .sort(["date", "inst"])
    )
    return _to_wide(out)


def pl_vwap_cross_count(df):
    pl = _pl()
    d = (
        df.filter(_col("close").is_not_null())
        .with_columns(_col("volume").fill_null(0.0).alias("_vol"), _col("amount").fill_null(0.0).alias("_amt"))
        .sort(["date", "inst", "ts"])
        .with_columns(
            _col("_vol").cum_sum().over(["date", "inst"]).alias("_cv"),
            _col("_amt").cum_sum().over(["date", "inst"]).alias("_ca"),
        )
        .with_columns(
            pl.when(_col("_cv") > _EPS)
            .then(_col("_ca") / _col("_cv"))
            .otherwise(None)
            .alias("_cvwap")
        )
        .with_columns(
            pl.when(_col("_cvwap").is_not_null())
            .then((_col("close") - _col("_cvwap")).sign())
            .otherwise(None)
            .alias("_sign")
        )
    )
    nclose = d.group_by(["date", "inst"]).agg(pl.len().alias("_n_close"))
    sig = (
        d.filter(_col("_sign").is_not_null())
        .with_columns(_col("_sign").shift(1).over(["date", "inst"]).alias("_prev"))
        .group_by(["date", "inst"])
        .agg(
            pl.len().alias("_n_sign"),
            (_col("_sign") != _col("_prev")).sum().alias("_changes"),
        )
    )
    joined = nclose.join(sig, on=["date", "inst"], how="left")
    out = joined.with_columns(
        pl.when(_col("_n_close") < 2)
        .then(None)
        .when(_col("_n_sign").is_null() | (_col("_n_sign") < 2))
        .then(0.0)
        .otherwise(_col("_changes"))
        .alias("value")
    ).select(["date", "inst", "value"])
    return _to_wide(out)


# ---------------------------------------------------------------------------
# Volume / amount distribution
# ---------------------------------------------------------------------------

def pl_concentration(df, name: str = "volume"):
    pl = _pl()
    d = df.with_columns(_col(name).fill_null(0.0).abs().alias("_abs"))
    daily = d.group_by(["date", "inst"]).agg(
        _col("_abs").sum().alias("_tot"),
        _col(name).is_not_null().any().alias("_any"),
    )
    joined = d.join(daily.select(["date", "inst", "_tot"]), on=["date", "inst"])
    res = (
        joined.with_columns((_col("_abs") / _col("_tot")).alias("_w"))
        .group_by(["date", "inst"])
        .agg(
            pl.when(_col("_tot").first() <= _EPS)
            .then(None)
            .otherwise((_col("_w") * _col("_w")).sum())
            .alias("value")
        )
        .join(daily.select(["date", "inst", "_any"]), on=["date", "inst"], how="left")
        .with_columns(
            pl.when(_col("_any")).then(_col("value")).otherwise(None).alias("value")
        )
        .select(["date", "inst", "value"])
        .sort(["date", "inst"])
    )
    return _to_wide(res)


def pl_entropy(df, name: str = "volume", normalize: bool = True):
    pl = _pl()
    d = df.with_columns(_col(name).fill_null(0.0).abs().alias("_abs"))
    daily = d.group_by(["date", "inst"]).agg(
        _col("_abs").sum().alias("_tot"),
        (_col("_abs") > 0).sum().alias("_n"),
        _col(name).is_not_null().any().alias("_any"),
    )
    joined = d.join(daily.select(["date", "inst", "_tot"]), on=["date", "inst"])
    res = (
        joined.with_columns((_col("_abs") / _col("_tot")).alias("_w"))
        .filter(_col("_w") > 0)
        .group_by(["date", "inst"])
        .agg((_col("_w") * _col("_w").log()).sum().neg().alias("_entropy"))
        .join(
            daily.select(["date", "inst", "_tot", "_n", "_any"]),
            on=["date", "inst"],
            how="left",
        )
        .with_columns(
            pl.when((_col("_tot") <= _EPS) | (_col("_n") < 2))
            .then(None)
            .when(normalize)
            .then(_col("_entropy") / _col("_n").log())
            .otherwise(_col("_entropy"))
            .alias("value")
        )
        .with_columns(
            pl.when(_col("_any")).then(_col("value")).otherwise(None).alias("value")
        )
        .select(["date", "inst", "value"])
        .sort(["date", "inst"])
    )
    return _to_wide(res)


def pl_signed_imbalance_proxy(df, name: str = "volume"):
    pl = _pl()
    d = _with_r(df).filter(_col("close").is_not_null()).with_columns(
        _col(name).fill_null(0.0).alias("_val")
    )
    out = (
        d.group_by(["date", "inst"])
        .agg(
            _col("_val").sum().alias("_tot"),
            pl.when(_col("r").is_not_null())
            .then(_col("r").sign() * _col("_val"))
            .otherwise(0.0)
            .sum()
            .alias("_signed"),
        )
        .with_columns(
            pl.when(_col("_tot") <= _EPS)
            .then(None)
            .otherwise(_col("_signed") / _col("_tot"))
            .alias("value")
        )
        .select(["date", "inst", "value"])
        .sort(["date", "inst"])
    )
    return _to_wide(out)


def pl_return_activity_corr(df, name: str = "activity", absolute_return: bool = False):
    pl = _pl()
    d = _with_r(df).filter(_col("close").is_not_null())
    if absolute_return:
        d = d.with_columns(_col("r").abs().alias("r"))
    out = (
        d.filter(_col("r").is_not_null() & _col(name).is_not_null())
        .group_by(["date", "inst"])
        .agg(
            pl.len().alias("_n"),
            _col("r").std().alias("_rs"),
            _col(name).std().alias("_as"),
            pl.corr(_col("r"), _col(name)).alias("value"),
        )
        .with_columns(
            pl.when((_col("_n") < 2) | (_col("_rs") <= _EPS) | (_col("_as") <= _EPS))
            .then(None)
            .otherwise(_col("value"))
            .alias("value")
        )
        .select(["date", "inst", "value"])
        .sort(["date", "inst"])
    )
    return _to_wide(out)


# ---------------------------------------------------------------------------
# Liquidity
# ---------------------------------------------------------------------------

def pl_amihud(df, scale: float = _SCALE):
    pl = _pl()
    d = (
        _with_r(df)
        .filter(_col("close").is_not_null())
        .with_columns(_col("amount").fill_null(0.0).alias("_amt"))
        .with_columns(pl.max_horizontal(_col("_amt"), pl.lit(_EPS)).alias("_denom"))
        .with_columns(
            pl.when(_col("r").is_not_null())
            .then(_col("r").abs() / _col("_denom"))
            .otherwise(None)
            .alias("_ratio")
        )
    )
    out = (
        d.filter(_col("_ratio").is_not_null())
        .group_by(["date", "inst"])
        .agg(_col("_ratio").mean().alias("value"))
        .sort(["date", "inst"])
    )
    wide = _to_wide(out)
    return wide * scale if wide is not None else None


def pl_kyle_lambda_proxy(df):
    """Kyle lambda proxy: beta = cov_sample(r, ss) / var_pop(ss).

    Matches the pandas reference: signed_share = sign(r) * amount / total where
    total is the day's amount sum over *all* close-finite bars; r/ss are then
    compacted to finite rows; ``np.cov`` is sample (ddof=1), ``np.var`` is
    population (ddof=0), so the SQL/polars expression must mix the two.
    """
    pl = _pl()
    d = (
        df.filter(_col("close").is_not_null())
        .sort(["date", "inst", "ts"])
        .with_columns(
            (_col("close") / _col("close").shift(1)).log().over(["date", "inst"]).alias("r")
        )
        .with_columns(_col("amount").fill_null(0.0).alias("_amt"))
        .with_columns(_col("_amt").sum().over(["date", "inst"]).alias("_tot"))
        .with_columns(
            pl.when(_col("r").is_not_null())
            .then(_col("r").sign() * _col("_amt") / _col("_tot"))
            .otherwise(None)
            .alias("_ss")
        )
    )
    out = (
        d.filter(_col("r").is_not_null())
        .group_by(["date", "inst"])
        .agg(
            pl.len().alias("_n"),
            pl.cov(_col("r"), _col("_ss")).alias("_cov"),
            _col("_ss").var().alias("_sv"),
            _col("_tot").first().alias("_tot"),
        )
        .with_columns(
            pl.when(
                (_col("_n") < 3)
                | (_col("_tot") <= _EPS)
                | (_col("_sv") * (_col("_n") - 1) / _col("_n") <= _EPS * _EPS)
            )
            .then(None)
            .otherwise(_col("_cov") / (_col("_sv") * (_col("_n") - 1) / _col("_n")))
            .alias("value")
        )
        .select(["date", "inst", "value"])
        .sort(["date", "inst"])
    )
    return _to_wide(out)


def pl_extreme_bar_return(df, side: str = "max"):
    pl = _pl()
    d = _with_r(df).filter(_col("r").is_not_null())
    out = (
        d.group_by(["date", "inst"])
        .agg(
            (_col("r").max() if side == "max" else _col("r").min()).alias("value")
        )
        .sort(["date", "inst"])
    )
    return _to_wide(out)


def pl_lunch_gap_return(df):
    pl = _pl()
    morning = df.filter(_col("minute") <= 690).filter(_col("close").is_not_null())
    afternoon = df.filter(_col("minute") >= 780).filter(_col("open").is_not_null())
    mc = morning.group_by(["date", "inst"]).agg(_col("close").last().alias("_mc"))
    ao = afternoon.group_by(["date", "inst"]).agg(_col("open").first().alias("_ao"))
    joined = mc.join(ao, on=["date", "inst"], how="left")
    out = (
        joined.with_columns(
            pl.when(_col("_ao").is_null() | (_col("_mc") <= _EPS))
            .then(None)
            .otherwise(_col("_ao") / _col("_mc") - 1.0)
            .alias("value")
        )
        .select(["date", "inst", "value"])
        .sort(["date", "inst"])
    )
    return _to_wide(out)


# ---------------------------------------------------------------------------
# Limit behaviour (daily limit broadcast onto minutes)
# ---------------------------------------------------------------------------

def pl_limit_duration(df, limit_panel, side: str = "up"):
    pl = _pl()
    d = _with_limit(df, limit_panel, side).filter(_col("close").is_not_null())
    out = (
        d.group_by(["date", "inst"])
        .agg(pl.len().alias("_n"), _hit_condition(side).sum().alias("_hit"))
        .with_columns(
            pl.when(_col("_n") > 0)
            .then(_col("_hit") / _col("_n"))
            .otherwise(None)
            .alias("value")
        )
        .select(["date", "inst", "value"])
        .sort(["date", "inst"])
    )
    return _to_wide(out)


def pl_limit_first_hit_time(df, limit_panel, side: str = "up"):
    pl = _pl()
    d = (
        _with_limit(df, limit_panel, side)
        .filter(_col("close").is_not_null())
        .sort(["date", "inst", "ts"])
        .with_columns(pl.int_range(0, pl.len()).over(["date", "inst"]).alias("_pos"))
        .with_columns(
            pl.when(_hit_condition(side)).then(_col("_pos")).otherwise(None).alias("_hit_pos"),
        )
    )
    out = (
        d.group_by(["date", "inst"])
        .agg(pl.len().alias("_n"), _col("_hit_pos").min().alias("_first"))
        .with_columns(
            pl.when(_col("_first").is_null())
            .then(None)
            .otherwise(_col("_first") / _col("_n"))
            .alias("value")
        )
        .select(["date", "inst", "value"])
        .sort(["date", "inst"])
    )
    return _to_wide(out)


def pl_limit_reopen_count(df, limit_panel, side: str = "up", transition: str = "open"):
    pl = _pl()
    d = (
        _with_limit(df, limit_panel, side)
        .filter(_col("close").is_not_null())
        .sort(["date", "inst", "ts"])
        .with_columns(_hit_condition(side).cast(int).alias("_m"))
        .with_columns(_col("_m").shift(1).over(["date", "inst"]).alias("_pm"))
    )
    if transition == "open":
        trans = (_col("_pm") == 1) & (_col("_m") == 0)
    else:
        trans = (_col("_pm") == 0) & (_col("_m") == 1)
    out = (
        d.group_by(["date", "inst"])
        .agg(pl.len().alias("_n"), trans.sum().alias("_trans"))
        .with_columns(
            pl.when(_col("_n") > 1)
            .then(_col("_trans").cast(float))
            .otherwise(0.0)
            .alias("value")
        )
        .select(["date", "inst", "value"])
        .sort(["date", "inst"])
    )
    return _to_wide(out)


# ---------------------------------------------------------------------------
# DuckDB SQL implementations (SQL-natural subset)
# ---------------------------------------------------------------------------

def _sql_wide(sql_result: pd.DataFrame) -> pd.DataFrame:
    out = sql_result
    if "ts" in out.columns:
        out = out.rename(columns={"ts": "date"})
    wide = out.pivot_table(index="date", columns="inst", values="value").sort_index()
    wide.index = pd.DatetimeIndex(wide.index)
    wide.columns.name = None
    return wide


def _sql(con, table, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def sql_realized_variance(con, table: str):
    q = f"""
    WITH t AS (
      SELECT date, inst, close,
             LN(close / LAG(close) OVER (PARTITION BY date, inst ORDER BY ts)) AS r
      FROM {table}
    )
    SELECT date, inst,
           CASE WHEN bool_or(close IS NOT NULL) THEN SUM(CASE WHEN r IS NOT NULL THEN r*r ELSE 0.0 END)
                ELSE NULL END AS value
    FROM t GROUP BY date, inst
    """
    return _sql_wide(_sql(con, table, q))


def sql_realized_semivariance(con, table: str, side: str = "down"):
    cond = "r < 0" if side == "down" else "r > 0"
    q = f"""
    WITH t AS (
      SELECT date, inst, close,
             LN(close / LAG(close) OVER (PARTITION BY date, inst ORDER BY ts)) AS r
      FROM {table}
    )
    SELECT date, inst,
           CASE WHEN bool_or(close IS NOT NULL) THEN SUM(CASE WHEN {cond} THEN r*r ELSE 0.0 END)
                ELSE NULL END AS value
    FROM t GROUP BY date, inst
    """
    return _sql_wide(_sql(con, table, q))


def sql_bipower_variation(con, table: str):
    q = f"""
    SELECT date, inst,
           CASE WHEN COUNT(*) >= 2 THEN {math.pi/2.0} * SUM(_p) ELSE NULL END AS value
    FROM (
      SELECT date, inst,
             ABS(r) * LAG(ABS(r)) OVER (PARTITION BY date, inst ORDER BY ts) AS _p
      FROM (
        SELECT date, inst, ts,
               LN(close / LAG(close) OVER (PARTITION BY date, inst ORDER BY ts)) AS r
        FROM {table}
      ) base
      WHERE r IS NOT NULL
    ) t
    GROUP BY date, inst
    """
    return _sql_wide(_sql(con, table, q))


def sql_jump_ratio(con, table: str):
    return _jump_from_rv_bv(sql_realized_variance(con, table), sql_bipower_variation(con, table))


def sql_path_efficiency(con, table: str):
    q = f"""
    SELECT date, inst,
           CASE WHEN COUNT(close) < 2 THEN NULL
                WHEN SUM(ABS(d)) <= 1e-12 THEN 0.0
                ELSE ABS(MAX(close) - MIN(close)) / SUM(ABS(d)) END AS value
    FROM (
      SELECT date, inst, close,
             close - LAG(close) OVER (PARTITION BY date, inst ORDER BY ts) AS d
      FROM (SELECT date, inst, ts, close FROM {table} WHERE close IS NOT NULL) f
    ) t
    GROUP BY date, inst
    """
    # path efficiency uses first/last *finite* bar, not max/min.
    q = f"""
    SELECT date, inst,
           CASE WHEN COUNT(*) < 2 THEN NULL
                WHEN SUM(ABS(d)) <= 1e-12 THEN 0.0
                ELSE ABS(LAST(close) - FIRST(close)) / SUM(ABS(d)) END AS value
    FROM (
      SELECT date, inst, ts, close,
             close - LAG(close) OVER (PARTITION BY date, inst ORDER BY ts) AS d
      FROM (SELECT date, inst, ts, close FROM {table} WHERE close IS NOT NULL) f
    ) t
    GROUP BY date, inst
    """
    return _sql_wide(_sql(con, table, q))


def sql_segment_return(con, table: str, segment: str):
    lo, hi = _SEGMENTS[segment]
    q = f"""
    SELECT date, inst,
           CASE WHEN COUNT(*) >= 2 THEN LAST(close) / FIRST(close) - 1.0 ELSE NULL END AS value
    FROM {table}
    WHERE minute BETWEEN {lo} AND {hi} AND close IS NOT NULL
    GROUP BY date, inst
    """
    return _sql_wide(_sql(con, table, q))


def sql_segment_volume_share(con, table: str, segment: str):
    lo, hi = _SEGMENTS[segment]
    q = f"""
    SELECT date, inst,
           CASE WHEN SUM(volume) > 1e-12 THEN
             SUM(CASE WHEN minute BETWEEN {lo} AND {hi} THEN volume ELSE 0.0 END) / SUM(volume)
           ELSE NULL END AS value
    FROM {table}
    GROUP BY date, inst
    """
    return _sql_wide(_sql(con, table, q))


def sql_concentration(con, table: str, name: str = "volume"):
    q = f"""
    SELECT date, inst,
           CASE WHEN SUM(ab) <= 1e-12 THEN NULL
                ELSE SUM(w*w) END AS value
    FROM (
      SELECT date, inst,
             ABS(COALESCE({name},0)) AS ab,
             ABS(COALESCE({name},0)) / NULLIF(SUM(ABS(COALESCE({name},0))) OVER (PARTITION BY date, inst), 0) AS w
      FROM {table}
    ) t
    GROUP BY date, inst
    """
    return _sql_wide(_sql(con, table, q))


def sql_entropy(con, table: str, name: str = "volume", normalize: bool = True):
    q = f"""
    SELECT date, inst,
           CASE WHEN COUNT(*) = 0 THEN NULL
                WHEN SUM(ab) <= 1e-12 THEN NULL
                ELSE -SUM(w * LN(w)) / CASE WHEN {1 if normalize else 0} = 1
                                            THEN LN(COUNT(*)) ELSE 1.0 END
           END AS value
    FROM (
      SELECT date, inst,
             ABS(COALESCE({name},0)) AS ab,
             ABS(COALESCE({name},0)) / NULLIF(SUM(ABS(COALESCE({name},0))) OVER (PARTITION BY date, inst), 0) AS w
      FROM {table}
    ) t
    WHERE w > 0
    GROUP BY date, inst
    """
    return _sql_wide(_sql(con, table, q))


def sql_signed_imbalance_proxy(con, table: str, name: str = "volume"):
    q = f"""
    WITH t AS (
      SELECT date, inst, {name},
             LN(close / LAG(close) OVER (PARTITION BY date, inst ORDER BY ts)) AS r
      FROM {table} WHERE close IS NOT NULL
    )
    SELECT date, inst,
           CASE WHEN SUM(COALESCE({name},0)) <= 1e-12 THEN NULL
                ELSE SUM(CASE WHEN r IS NOT NULL THEN SIGN(r)*COALESCE({name},0) ELSE 0.0 END)
                     / SUM(COALESCE({name},0))
           END AS value
    FROM t GROUP BY date, inst
    """
    return _sql_wide(_sql(con, table, q))


def sql_amihud(con, table: str, scale: float = _SCALE):
    q = f"""
    SELECT date, inst,
           CASE WHEN COUNT(r) = 0 THEN NULL
                ELSE AVG(ABS(r) / GREATEST(COALESCE(amount,0), 1e-12)) * {scale} END AS value
    FROM (
      SELECT date, inst, amount,
             LN(close / LAG(close) OVER (PARTITION BY date, inst ORDER BY ts)) AS r
      FROM {table} WHERE close IS NOT NULL
    ) t
    WHERE r IS NOT NULL
    GROUP BY date, inst
    """
    return _sql_wide(_sql(con, table, q))


def sql_kyle_lambda_proxy(con, table: str):
    """DuckDB: COVAR_SAMP(r, ss) / VAR_POP(ss); ss = sign(r)*amount/total."""
    q = f"""
    WITH t AS (
      SELECT date, inst,
             LN(close / LAG(close) OVER (PARTITION BY date, inst ORDER BY ts)) AS r,
             COALESCE(amount,0) AS amt
      FROM {table} WHERE close IS NOT NULL
    ), d AS (
      SELECT date, inst, SUM(amt) AS tot FROM t GROUP BY date, inst
    )
    SELECT q.date, q.inst,
           CASE WHEN COUNT(q.r) < 3 THEN NULL
                WHEN q.tot <= 1e-12 THEN NULL
                WHEN VAR_POP(q.ss) <= 1e-24 THEN NULL
                ELSE COVAR_SAMP(q.r, q.ss) / VAR_POP(q.ss) END AS value
    FROM (
      SELECT t.date, t.inst, t.r, d.tot, SIGN(t.r) * t.amt / d.tot AS ss
      FROM t JOIN d ON (t.date = d.date AND t.inst = d.inst)
      WHERE t.r IS NOT NULL
    ) q
    GROUP BY q.date, q.inst, q.tot
    """
    return _sql_wide(_sql(con, table, q))


def sql_extreme_bar_return(con, table: str, side: str = "max"):
    agg = "MAX" if side == "max" else "MIN"
    q = f"""
    SELECT date, inst, {agg}(r) AS value
    FROM (
      SELECT date, inst,
             LN(close / LAG(close) OVER (PARTITION BY date, inst ORDER BY ts)) AS r
      FROM {table}
    ) t
    WHERE r IS NOT NULL
    GROUP BY date, inst
    """
    return _sql_wide(_sql(con, table, q))


def sql_vwap_above_ratio(con, table: str):
    q = f"""
    WITH t AS (
      SELECT date, inst, close, amount, volume,
             CASE WHEN amount IS NOT NULL AND volume IS NOT NULL AND volume > 0 THEN volume ELSE 0.0 END AS v,
             CASE WHEN amount IS NOT NULL AND volume IS NOT NULL AND volume > 0 THEN amount ELSE 0.0 END AS a
      FROM {table} WHERE close IS NOT NULL
    ), d AS (
      SELECT date, inst, SUM(v) AS tv, SUM(a) AS ta,
             CASE WHEN SUM(v) > 1e-12 THEN SUM(a)/SUM(v) ELSE NULL END AS vwap
      FROM t GROUP BY date, inst
    )
    SELECT t.date, t.inst,
           CASE WHEN d.tv <= 1e-12 THEN NULL
                WHEN COUNT(CASE WHEN t.close IS NOT NULL AND t.v > 0 THEN 1 END) = 0 THEN NULL
                ELSE AVG(CASE WHEN t.close > d.vwap THEN 1.0 ELSE 0.0 END) END AS value
    FROM t JOIN d ON (t.date = d.date AND t.inst = d.inst)
    GROUP BY t.date, t.inst, d.tv
    """
    return _sql_wide(_sql(con, table, q))


# ---------------------------------------------------------------------------
# Certification driver
# ---------------------------------------------------------------------------

_POLARS_OPS = {
    "intra_realized_variance": pl_realized_variance,
    "intra_realized_semivariance": lambda df: pl_realized_semivariance(df, "down"),
    "intra_bipower_variation": pl_bipower_variation,
    "intra_jump_ratio": pl_jump_ratio,
    "intra_path_efficiency": pl_path_efficiency,
    "intra_high_time": lambda df: pl_high_time(df, low=False),
    "intra_low_time": lambda df: pl_high_time(df, low=True),
    "intra_segment_return": lambda df: pl_segment_return(df, "morning"),
    "intra_segment_volume_share": lambda df: pl_segment_volume_share(df, "morning"),
    "intra_segment_amount_share": lambda df: pl_segment_amount_share(df, "morning"),
    "intra_segment_realized_vol": lambda df: pl_segment_realized_vol(df, "morning"),
    "intra_segment_vwap_deviation": lambda df: pl_segment_vwap_deviation(df, "morning"),
    "intra_vwap_above_ratio": pl_vwap_above_ratio,
    "intra_vwap_cross_count": pl_vwap_cross_count,
    "intra_concentration": pl_concentration,
    "intra_entropy": pl_entropy,
    "intra_signed_imbalance_proxy": pl_signed_imbalance_proxy,
    "intra_return_activity_corr": pl_return_activity_corr,
    "intra_amihud": pl_amihud,
    "intra_kyle_lambda_proxy": pl_kyle_lambda_proxy,
    "intra_extreme_bar_return": pl_extreme_bar_return,
    "intra_lunch_gap_return": pl_lunch_gap_return,
}

_SQL_OPS = {
    "intra_realized_variance": lambda con, t: sql_realized_variance(con, t),
    "intra_realized_semivariance": lambda con, t: sql_realized_semivariance(con, t, "down"),
    "intra_bipower_variation": lambda con, t: sql_bipower_variation(con, t),
    "intra_jump_ratio": lambda con, t: sql_jump_ratio(con, t),
    "intra_path_efficiency": lambda con, t: sql_path_efficiency(con, t),
    "intra_segment_return": lambda con, t: sql_segment_return(con, t, "morning"),
    "intra_segment_volume_share": lambda con, t: sql_segment_volume_share(con, t, "morning"),
    "intra_concentration": lambda con, t: sql_concentration(con, t, "volume"),
    "intra_entropy": lambda con, t: sql_entropy(con, t, "volume", True),
    "intra_signed_imbalance_proxy": lambda con, t: sql_signed_imbalance_proxy(con, t, "volume"),
    "intra_amihud": lambda con, t: sql_amihud(con, t, _SCALE),
    "intra_kyle_lambda_proxy": lambda con, t: sql_kyle_lambda_proxy(con, t),
    "intra_extreme_bar_return": lambda con, t: sql_extreme_bar_return(con, t, "max"),
    "intra_vwap_above_ratio": lambda con, t: sql_vwap_above_ratio(con, t),
}

# Operators whose kernel is order/tie/population-variance sensitive or requires
# daily limit broadcasting; certified on a reduced backend set or reference-only.
_POLARS_LIMIT_OPS = {
    "intra_limit_duration": lambda df, lp: pl_limit_duration(df, lp, "up"),
    "intra_limit_first_hit_time": lambda df, lp: pl_limit_first_hit_time(df, lp, "up"),
    "intra_limit_reopen_count": lambda df, lp: pl_limit_reopen_count(df, lp, "up", "open"),
}

# All 25 minute aggregators now have at least pandas+polars parity; none remain
# reference-only.  The set is kept for the artifact schema's completeness check.
_REFERENCE_ONLY: set[str] = set()
