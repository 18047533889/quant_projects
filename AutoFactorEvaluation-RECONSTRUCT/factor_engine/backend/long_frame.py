# -*- coding: utf-8
"""Long-table 中间结果：Polars / SQL 统一的 ts / inst / value 语义。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .pandas_compat import pd

_TS = "ts"
_INST = "inst"
_VAL = "_v"


@dataclass(frozen=True)
class LongFrameResult:
    """PlanNode 在 long-table Polars LazyFrame 中的结果列。"""

    frame: Any  # pl.LazyFrame | pl.DataFrame
    value_col: str
    ts_col: str = "ts"
    inst_col: str = "inst"
    is_lazy: bool = True


def long_table_to_polars_lazy(
    frame: pd.DataFrame,
    *,
    float_cols: Iterable[str],
) -> Any:
    """Pandas 长表 → Polars LazyFrame；float 列保留 IEEE NaN（避免 ``from_pandas`` 的 nan→NULL）。"""
    import polars as pl

    fset = set(float_cols)
    cols: dict[str, Any] = {}
    for name in frame.columns:
        if name in fset:
            vals = frame[name].to_numpy(dtype="float64", na_value=float("nan"))
            cols[name] = pl.Series(name, vals, dtype=pl.Float64)
        else:
            cols[name] = frame[name]
    return pl.DataFrame(cols).lazy()


def series_to_polars_long_lazy(
    series: pd.Series,
    *,
    ts_col: str = "ts",
    inst_col: str = "inst",
    value_col: str = "_v",
) -> Any:
    """MultiIndex Series → long-table LazyFrame（``ts, inst, _v``）。"""
    import polars as pl

    from storage.factor_format import series_to_long_table

    if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels < 2:
        raise TypeError("series_to_polars_long_lazy expects MultiIndex (timestamp, instrument) Series")
    tcol = str(series.index.names[0])
    icol = str(series.index.names[1])
    pdf = series_to_long_table(
        series.astype("float64", copy=False),
        timestamp_col=tcol,
        asset_col=icol,
        value_col=value_col,
    )
    renamed = pdf.rename(columns={tcol: ts_col, icol: inst_col, value_col: "_v"})
    # 保留 IEEE NaN（pl.from_pandas 默认 nan_to_null=True 会把 NaN 变成 NULL）
    vals = renamed["_v"].to_numpy(dtype="float64", na_value=float("nan"))
    return pl.DataFrame(
        {
            ts_col: renamed[ts_col],
            inst_col: renamed[inst_col],
            "_v": pl.Series("_v", vals, dtype=pl.Float64),
        }
    ).lazy()


def is_polars_long_lazy(val: Any) -> bool:
    """是否为 long-table LazyFrame（含 ts/inst 与 value 列）。"""
    if val is None:
        return False
    try:
        import polars as pl
    except ImportError:
        return False
    if not isinstance(val, pl.LazyFrame):
        return False
    names = set(val.collect_schema().names())
    value_cols = {_VAL, "value", "_v"}
    return _TS in names and _INST in names and bool(names & value_cols)


def normalize_polars_long_lazy(
    lf: Any,
    *,
    ts_col: str = "ts",
    inst_col: str = "inst",
    value_col: str = "_v",
) -> Any:
    """统一 long LazyFrame 列名为 ``ts / inst / _v``。"""
    import polars as pl

    if not isinstance(lf, pl.LazyFrame):
        raise TypeError(f"expected LazyFrame, got {type(lf)!r}")
    rename: dict[str, str] = {}
    schema = lf.collect_schema().names()
    if ts_col != _TS and ts_col in schema:
        rename[ts_col] = _TS
    if inst_col != _INST and inst_col in schema:
        rename[inst_col] = _INST
    for cand in (value_col, "value", _VAL):
        if cand in schema and cand != _VAL:
            rename[cand] = _VAL
            break
    if rename:
        lf = lf.rename(rename)
    return lf.select(pl.col(_TS), pl.col(_INST), pl.col(_VAL))


def value_to_polars_long_lazy(
    val: Any,
    *,
    ts_col: str = "ts",
    inst_col: str = "inst",
    value_col: str = "_v",
) -> Any:
    """Series 或 long-table LazyFrame → ``ts / inst / _v`` LazyFrame（无 pandas 往返）。"""
    if is_polars_long_lazy(val):
        return normalize_polars_long_lazy(val, ts_col=ts_col, inst_col=inst_col, value_col=value_col)
    if isinstance(val, pd.Series):
        return series_to_polars_long_lazy(val, ts_col=ts_col, inst_col=inst_col, value_col=value_col)
    raise TypeError(f"value_to_polars_long_lazy expects Series or long LazyFrame, got {type(val)!r}")


def polars_long_to_multiindex_series(
    frame: Any,
    *,
    timestamp_col: str,
    instrument_col: str,
    value_col: str,
    template_index: pd.Index | None = None,
) -> pd.Series:
    """Long table → MultiIndex Series（仅在最终输出调用一次）。"""
    from storage.factor_format import long_table_to_series

    pdf = frame.to_pandas() if hasattr(frame, "to_pandas") else frame
    rename = {}
    if timestamp_col not in pdf.columns and "ts" in pdf.columns:
        rename["ts"] = timestamp_col
    if instrument_col not in pdf.columns and "inst" in pdf.columns:
        rename["inst"] = instrument_col
    if rename:
        pdf = pdf.rename(columns=rename)
    out = long_table_to_series(
        pdf,
        timestamp_col=timestamp_col,
        asset_col=instrument_col,
        value_col=value_col,
    )
    if template_index is not None:
        return out.reindex(template_index)
    return out


def optional_universe_index(ctx: Any) -> pd.Index | None:
    """``FACTOR_ENGINE_POLARS_LONG_ALIGN_UNIVERSE=1`` 时用 ``scan_index_long`` 构造对齐索引。"""
    import os

    if os.environ.get("FACTOR_ENGINE_POLARS_LONG_ALIGN_UNIVERSE", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    scan_idx = getattr(ctx.data_source, "scan_index_long", None) if ctx is not None else None
    if not callable(scan_idx):
        return None
    frame = scan_idx().collect()
    if hasattr(frame, "to_pandas"):
        pdf = frame.to_pandas()
    else:
        pdf = frame
    ts_col_name = getattr(ctx, "timestamp_col", "timestamp")
    icol_name = getattr(ctx, "instrument_col", "instrument")
    ts_vals = pd.to_datetime(pdf["ts"]).astype("datetime64[ns]")
    inst_vals = pdf["inst"].astype(str).astype(object)
    return pd.MultiIndex.from_arrays([ts_vals, inst_vals], names=[ts_col_name, icol_name])
