# -*- coding: utf-8 -*-
"""宽表 panel 与 Polars DataFrame 互转（列=标的，行=时间顺序）。

WS-B #245: ``panel_to_polars`` 保留 pandas ``DatetimeIndex`` —— 转换为显式
``__fe_time__`` 列（Option A），并由 ``PanelIdentity`` 记录时间轴/标的轴；
``strip_panel_metadata`` 在内核调用前剥离该元数据列，避免数值列提取把时间轴
当作特征。``polars_to_panel`` 始终用 template 的时间轴重建结果，不丢失时间轴。

R39-PERF-028: 批量转换默认走 Arrow C Data Interface 单块传输
（``pa.Table.from_pandas`` + ``pl.from_arrow``），避免逐列构造 Python dict 的
copy-heavy 路径。仅当整块是 Arrow 安全 dtype（纯数值/bool/datetime64[ns]）才走
批量；混合 dtype / object / nullable extension dtype 回退到原逐列路径。
批量/逐列计数器（``rep_conversion_bulk_count`` / ``rep_conversion_column_loop_count``）
与表示转换 KPI（``backend.rep_transition``）在此消费点真实记录。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .pandas_compat import pd

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

_SKIP_COLS = frozenset({"date", "stock_code"})

# Re-export the WS-B PanelSchema/PanelIdentity machinery (single source of truth
# lives in cleaned_operators.common._polars_bridge).
from cleaned_operators.common._polars_bridge import (  # noqa: E402
    FE_TIME_COL,
    PanelIdentity,
    PanelSchema,
    panel_schema,
    strip_panel_metadata,
    verify_frames_share_identity,
)

# 表示转换 KPI（R39-PERF-031）：转换发生在这里（fast + fallback），而非 cleaned_bridge。
from .rep_transition import (  # noqa: E402
    estimate_panel_bytes,
    record_transition,
    rep_transition_snapshot,
)

# 批量 fast path / 逐列 fallback 的观测计数器。
rep_conversion_bulk_count = 0
rep_conversion_column_loop_count = 0


# ---------------------------------------------------------------------------
# Arrow 安全判定
# ---------------------------------------------------------------------------
def _dtype_arrow_safe(dtype: Any) -> bool:
    """该 pandas 列 dtype 是否能在 Arrow 单块往返后与 ``to_numpy`` 字节一致。

    允许：标准 numpy bool/int/uint/float（除 float16，Arrow 无此类型）与
    ``datetime64[ns]``。排除：object / category / nullable extension dtype
    （Int64 等，``to_numpy`` 会产生 object 数组）、timedelta、tz-aware datetime。
    """
    if not isinstance(dtype, np.dtype):
        return False
    kind = dtype.kind
    if kind in "biuf":
        # float16 无 Arrow 类型，排除。
        if kind == "f" and dtype.itemsize == 2:
            return False
        return True
    if kind == "M":
        return str(dtype) == "datetime64[ns]"
    return False


def _arrow_safe_panel(panel: pd.DataFrame) -> bool:
    """整块是否可安全走 Arrow 单块传输（无重复列名且每列 Arrow 安全）。"""
    if not isinstance(panel, pd.DataFrame):
        return False
    cols = panel.columns
    if len(set(map(str, cols))) != len(cols):
        return False
    try:
        return all(_dtype_arrow_safe(panel[c].dtype) for c in cols)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# panel → polars
# ---------------------------------------------------------------------------
def convert_panel_to_polars_bulk(panel: pd.DataFrame) -> Any:
    """Arrow C Data Interface 单块传输：pandas → polars（PERF-028 fast path）。

    在保证字节一致的前提下：
    - ``pa.Table.from_pandas(panel, preserve_index=False)`` 传输数值 2D 块；
    - ``DatetimeIndex`` 显式转为 ``__fe_time__`` 列（与逐列路径一致）；
    - Arrow 把 pandas 的 NaN 表达为 null，float 列回填 NaN 以与参考路径一致。
    """
    global rep_conversion_bulk_count
    if pl is None:
        raise ImportError("polars is required for polars operator backend")
    import pyarrow as pa

    if panel.columns.empty:
        # 空 panel：直接构造（避免给 0 行 table 追加时间列报错）。
        if isinstance(panel.index, pd.DatetimeIndex):
            frame = pl.DataFrame({FE_TIME_COL: panel.index.to_numpy()})
        else:
            frame = pl.DataFrame()
    else:
        table = pa.Table.from_pandas(panel, preserve_index=False)
        if isinstance(panel.index, pd.DatetimeIndex):
            table = table.append_column(FE_TIME_COL, pa.array(panel.index.to_numpy()))
        frame = pl.from_arrow(table)
        float_cols = [
            c
            for c, dt in frame.schema.items()
            if dt.base_type() in (pl.Float32, pl.Float64)
        ]
        if float_cols:
            frame = frame.with_columns(
                [pl.col(c).fill_null(float("nan")).alias(c) for c in float_cols]
            )
    rep_conversion_bulk_count += 1
    record_transition("pandas", "polars", estimate_panel_bytes(panel))
    return frame


def _panel_to_polars_column_loop(panel: pd.DataFrame) -> Any:
    """参考（逐列）路径：保留原始语义，作为 fallback。"""
    global rep_conversion_column_loop_count
    data: dict[str, Any] = {str(c): panel[c].to_numpy() for c in panel.columns}
    if isinstance(panel.index, pd.DatetimeIndex):
        data[FE_TIME_COL] = panel.index.to_numpy()
    rep_conversion_column_loop_count += 1
    record_transition("pandas", "polars", estimate_panel_bytes(panel))
    return pl.DataFrame(data)


def panel_to_polars(panel: pd.DataFrame, *, use_arrow: bool = True) -> Any:
    """pandas 宽表 → polars（保留列名，不依赖 date/stock_code 列）。

    WS-B #245: 当宽表携带 ``DatetimeIndex`` 时，将其转换为显式 ``__fe_time__``
    列，使时间轴在转换后存活。``__fe_time__`` 是保留元数据列名，由
    ``strip_panel_metadata`` 在内核调用前剥离。

    R39-PERF-028: 默认 ``use_arrow=True``，Arrow 安全块走单块传输；否则回退
    逐列路径（语义不变）。
    """
    if pl is None:
        raise ImportError("polars is required for polars operator backend")
    if is_polars_frame(panel):
        return panel
    if not isinstance(panel, pd.DataFrame):
        raise TypeError(f"expected DataFrame panel, got {type(panel)!r}")
    if use_arrow and _arrow_safe_panel(panel):
        return convert_panel_to_polars_bulk(panel)
    return _panel_to_polars_column_loop(panel)


# ---------------------------------------------------------------------------
# polars → panel
# ---------------------------------------------------------------------------
def _polars_output_cols(result: Any, template: pd.DataFrame) -> list:
    out_cols = [c for c in template.columns if c in result.columns]
    if not out_cols:
        out_cols = [c for c in result.columns if c not in _SKIP_COLS]
    # The reserved __fe_time__ metadata column is never an output column.
    out_cols = [c for c in out_cols if c != FE_TIME_COL]
    return list(out_cols)


def _polars_to_panel_can_bulk(result: Any, out_cols: list) -> bool:
    """结果是否能用 ``to_pandas`` 批量重建且与参考 ``to_numpy`` 路径字节一致。

    仅数值/bool 单 dtype 块才安全（factor 结果几乎全为 Float64，命中 float 分支）：
    - float32/float64：``to_numpy`` 与 ``to_pandas`` 都保留 NaN，一致；
    - int/uint/bool 单 dtype：仅当选中列无 null（否则 ``to_numpy`` 会上转 float64，
      ``to_pandas`` 保持原 dtype，两者不一致）；
    - 其它 dtype（Categorical/Decimal/String/Datetime 等）保守回退逐列路径。
    """
    if not out_cols:
        return True
    frame = result.select([str(c) for c in out_cols])
    base_types = {frame.schema[str(c)].base_type() for c in out_cols}
    if len(base_types) != 1:
        return False
    bt = next(iter(base_types))
    if bt in (pl.Float32, pl.Float64):
        return True
    if not (bt == pl.Boolean or bt.is_integer()):
        return False
    try:
        total_nulls = int(frame.null_count().to_numpy().sum())
    except Exception:
        return False
    return total_nulls == 0


def convert_polars_to_panel_bulk(result: Any, template: pd.DataFrame) -> pd.DataFrame:
    """Arrow/pandas 批量重建：polars → 与 template 对齐的 pandas 宽表。

    经 ``to_pandas`` 单次物化（避免 ``select().to_numpy()`` 的 Python 往返），
    再对齐 template 时间轴。仅应在 ``_polars_to_panel_can_bulk`` 返回 True 时调用。
    """
    global rep_conversion_bulk_count
    if pl is None:
        raise ImportError("polars is required for polars operator backend")
    out_cols = _polars_output_cols(result, template)
    if not out_cols:
        pdf = pd.DataFrame(index=template.index, columns=template.columns, dtype=float)
    else:
        pdf = result.select([str(c) for c in out_cols]).to_pandas()
        pdf.index = template.index
        pdf.columns = out_cols
    rep_conversion_bulk_count += 1
    bytes_ = int(result.estimated_size()) if hasattr(result, "estimated_size") else 0
    record_transition("polars", "pandas", bytes_)
    return pdf


def _polars_to_panel_column_loop(result: Any, template: pd.DataFrame) -> pd.DataFrame:
    """参考（逐列）路径：保留原始语义，作为 fallback。"""
    global rep_conversion_column_loop_count
    out_cols = _polars_output_cols(result, template)
    if not out_cols:
        return pd.DataFrame(index=template.index, columns=template.columns, dtype=float)
    data = result.select([str(c) for c in out_cols]).to_numpy()
    rep_conversion_column_loop_count += 1
    bytes_ = int(result.estimated_size()) if hasattr(result, "estimated_size") else 0
    record_transition("polars", "pandas", bytes_)
    return pd.DataFrame(data, index=template.index, columns=out_cols)


def polars_to_panel(
    result: Any,
    *,
    template: pd.DataFrame,
    use_arrow: bool = True,
) -> pd.DataFrame:
    """polars 结果 → 与 template 对齐的 pandas 宽表。

    R39-PERF-028: 默认 ``use_arrow=True``，单 dtype 块走 ``to_pandas`` 批量；
    否则回退逐列 ``to_numpy`` 路径（语义不变）。
    """
    if pl is None:
        raise ImportError("polars is required for polars operator backend")
    if not isinstance(result, pl.DataFrame):
        raise TypeError(f"expected polars DataFrame, got {type(result)!r}")
    out_cols = _polars_output_cols(result, template)
    if use_arrow and _polars_to_panel_can_bulk(result, out_cols):
        return convert_polars_to_panel_bulk(result, template)
    return _polars_to_panel_column_loop(result, template)


def is_polars_frame(obj: Any) -> bool:
    """判断对象是否为 Polars DataFrame。

    参数:
        obj: 待检测对象。

    返回:
        polars 已安装且对象为 ``pl.DataFrame`` 时为 ``True``。
    """
    return pl is not None and isinstance(obj, pl.DataFrame)
