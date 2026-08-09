# -*- coding: utf-8 -*-
"""Polars panel ↔ pandas 桥接工具（复杂算子复用 pandas_numpy 实现）。

WS-B PanelSchema / PanelIdentity
-------------------------------
统一宽表 panel 的时间轴 / 标的轴 / 元数据列 / 值列语义：

* 元数据列（时间轴列 + 标的身份列）不参与因子数值计算；
* ``PanelIdentity`` 把一个 panel 的 (time_index_hash, instrument_axis_hash,
  grain, frequency) 固化为可比较值对象；多输入算子必须在调用内核前验证所有
  输入共享同一 PanelIdentity（#243/#244）；
* ``panel_to_polars`` 把 pandas ``DatetimeIndex`` 保留为显式 ``__fe_time__``
  列（#245），``strip_panel_metadata`` 在内核调用前把它剥离，避免数值列提取
  把时间轴当作特征；
* ``MissingNumeric := null OR NaN``，``polars_nan_to_null`` 使 Polars 的
  缺失语义与 pandas/DuckDB 一致（#241）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

import pandas as pd

# WS-B #235-#238: 统一的宽表元数据列。时间轴列优先 ``__fe_time__``（bridge 注入，
# 权威时间轴），其次常见显式时间列；身份列不参与因子数值计算。
_TIME_AXIS_COLUMNS = ("__fe_time__", "date", "timestamp", "trade_date", "datetime")
_IDENTITY_COLUMNS = ("stock_code", "instrument", "symbol", "session", "inst", "ts")
SKIP = frozenset((*_TIME_AXIS_COLUMNS, *_IDENTITY_COLUMNS))

FE_TIME_COL = "__fe_time__"


def _hash_time_values(values: Iterable[Any]) -> int:
    """Deterministic hash of an ordered time axis.

    Normalises ``pd.Timestamp`` / ``datetime.datetime`` / ``datetime.date`` /
    ``numpy.datetime64`` to a canonical ``isoformat`` string so the same
    timeline hashes identically whether it came from a pandas ``DatetimeIndex``
    or a polars ``Datetime``/``Date`` column.
    """
    out = []
    for v in values:
        if isinstance(v, pd.Timestamp):
            v = v.to_pydatetime()
        if isinstance(v, (np.datetime64,)):
            v = v.astype("datetime64[ns]").item()
        if hasattr(v, "isoformat"):
            out.append(v.isoformat())
        else:
            out.append(str(v))
    return hash(tuple(out))


@dataclass(frozen=True)
class PanelSchema:
    """描述一个宽表 panel 的轴/列结构（#235-#238 元数据/索引语义统一）。

    ``time_column`` 是 polars 宽表显式携带的时间轴列名（无则 ``None``）；
    pandas 宽表的时间轴在 ``.index``，此处 ``time_column`` 为 ``None`` 且由
    ``PanelIdentity.from_frame`` 直接读 index。``metadata_columns`` 是元数据列
    （时间轴 + 标的身份），``value_columns`` 是参与因子计算的标的轴（有序）。
    """

    time_column: str | None
    metadata_columns: tuple[str, ...]
    value_columns: tuple[str, ...]
    grain: str
    frequency: str


@dataclass(frozen=True)
class PanelIdentity:
    """宽表 panel 的可比较身份值对象（#243/#244）。

    ``time_index_hash`` 是时间轴内容哈希（pandas 取 index；polars 取时间列），
    无时间轴时为 ``None``；``instrument_axis_hash`` 是有序标的（值）列的哈希；
    ``grain``/``frequency`` 来自元数据或推导（``"unknown"`` 表示未声明）。
    多输入算子必须在调用内核前验证所有输入共享同一 PanelIdentity —— 日期轴
    移动一天、标的列置换都必须 loud-fail。
    """

    time_index_hash: int | None
    instrument_axis_hash: int
    grain: str
    frequency: str

    def describe(self) -> str:
        return (
            "PanelIdentity("
            f"time_hash={self.time_index_hash}, "
            f"instrument_hash={self.instrument_axis_hash}, "
            f"grain={self.grain!r}, frequency={self.frequency!r})"
        )

    # WS-B #243/#244: identity equality is defined by the time axis and the
    # ordered instrument axis (the two axes the review demands be verified:
    # a shifted date or a permuted column set must fail loudly).  ``grain`` and
    # ``frequency`` are descriptive metadata (exposed for #235-#238 schema
    # introspection); a pandas frame can infer a frequency while the same
    # timeline converted through ``panel_to_polars`` cannot, so an UNKNOWN side
    # must not make an otherwise-identical timeline "different".
    #
    # R9-P0-014: but when BOTH sides declare a KNOWN grain/frequency and they
    # disagree, the two panels are NOT the same execution identity — a daily
    # panel and a minute panel sharing an index-by-accident must loud-fail.
    # Unknown metadata enters explicit compatibility resolution (ignored); known
    # mismatch is a hard reject.  The hash covers only the two axes so that
    # ``eq`` never conflicts with ``hash`` (unequal objects may share a hash,
    # but equal objects must not).
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PanelIdentity):
            return NotImplemented
        if (
            self.time_index_hash != other.time_index_hash
            or self.instrument_axis_hash != other.instrument_axis_hash
        ):
            return False
        if (
            self.grain != "unknown"
            and other.grain != "unknown"
            and self.grain != other.grain
        ):
            return False
        if (
            self.frequency != "unknown"
            and other.frequency != "unknown"
            and self.frequency != other.frequency
        ):
            return False
        return True

    def __hash__(self) -> int:
        # R9-P0-014: deterministic hash over the execution-identity axes only —
        # consistent with __eq__ regardless of known/unknown metadata state.
        return hash((self.time_index_hash, self.instrument_axis_hash))

    def __hash__(self) -> int:
        return hash((self.time_index_hash, self.instrument_axis_hash))

    @classmethod
    def from_frame(cls, frame: Any) -> "PanelIdentity":
        """Compute a PanelIdentity from a pandas wide panel or polars wide frame."""
        if isinstance(frame, pd.DataFrame):
            if isinstance(frame.index, pd.MultiIndex):
                time_hash: int | None = None
            else:
                time_hash = _hash_time_values(frame.index)
            columns = tuple(str(c) for c in frame.columns)
            value_cols = tuple(c for c in columns if c not in SKIP)
            grain, frequency = _pandas_grain_frequency(frame)
            return cls(
                time_index_hash=time_hash,
                instrument_axis_hash=hash(value_cols),
                grain=grain,
                frequency=frequency,
            )
        if pl is not None and isinstance(frame, pl.DataFrame):
            columns = tuple(str(c) for c in frame.columns)
            time_col = next((c for c in _TIME_AXIS_COLUMNS if c in columns), None)
            if time_col is not None:
                time_hash = _hash_time_values(frame[time_col].to_list())
            else:
                time_hash = None
            value_cols = tuple(c for c in columns if c not in SKIP)
            return cls(
                time_index_hash=time_hash,
                instrument_axis_hash=hash(value_cols),
                grain="unknown",
                frequency="unknown",
            )
        raise TypeError(f"cannot compute PanelIdentity from {type(frame)!r}")


def _pandas_grain_frequency(frame: pd.DataFrame) -> tuple[str, str]:
    """Infer grain/frequency from a pandas index (best-effort)."""
    index = getattr(frame, "index", None)
    if index is None:
        return "unknown", "unknown"
    freq = getattr(index, "freq", None)
    freq_str = str(freq) if freq is not None else "unknown"
    if isinstance(index, pd.DatetimeIndex):
        grain = "daily" if index.freqstr and index.freqstr.endswith("D") else "unknown"
    else:
        grain = "unknown"
    return grain, freq_str


def panel_schema(frame: Any) -> PanelSchema:
    """Expose the panel's time axis / metadata columns / value columns / grain."""
    if isinstance(frame, pd.DataFrame):
        columns = tuple(str(c) for c in frame.columns)
        metadata_cols = tuple(c for c in columns if c in SKIP)
        value_cols = tuple(c for c in columns if c not in SKIP)
        grain, frequency = _pandas_grain_frequency(frame)
        return PanelSchema(
            time_column=None, metadata_columns=metadata_cols,
            value_columns=value_cols, grain=grain, frequency=frequency,
        )
    if pl is not None and isinstance(frame, pl.DataFrame):
        columns = tuple(str(c) for c in frame.columns)
        time_col = next((c for c in _TIME_AXIS_COLUMNS if c in columns), None)
        metadata_cols = tuple(c for c in columns if c in SKIP)
        value_cols = tuple(c for c in columns if c not in SKIP)
        return PanelSchema(
            time_column=time_col, metadata_columns=metadata_cols,
            value_columns=value_cols, grain="unknown", frequency="unknown",
        )
    raise TypeError(f"cannot infer panel schema from {type(frame)!r}")


def _is_panel_like(value: Any) -> bool:
    if isinstance(value, (int, float, str, bool, np.number)) or value is None:
        return False
    columns = getattr(value, "columns", None)
    if columns is None:
        return False
    try:
        return len(list(columns)) > 0
    except TypeError:
        return False


def verify_frames_share_identity(
    context: str, *args: Any, allow_broadcast: bool = False
) -> None:
    """Require every panel among ``args`` to share the same PanelIdentity.

    Multi-input operators must fail loudly before the kernel runs when one
    input's date axis is shifted or its stock columns are permuted (WS-B
    #243/#244).  Only a formal ``BroadcastSpec`` may relax this — the caller
    passes ``allow_broadcast=True`` when the operator declares a typed broadcast
    tag (``daily_to_minute_broadcast`` / ``scalar_to_cross_section_broadcast`` /
    ``same_trading_date_broadcast`` / ``session_boundary_broadcast``), because a
    broadcast input legitimately carries a different date axis.
    """
    if allow_broadcast:
        return
    frames = [a for a in args if _is_panel_like(a)]
    if len(frames) < 2:
        return
    base = PanelIdentity.from_frame(frames[0])
    for i, frame in enumerate(frames[1:], start=1):
        other = PanelIdentity.from_frame(frame)
        if other != base:
            raise ValueError(
                f"{context}: input panel {i} has a different PanelIdentity than "
                f"input 0.\n  input 0: {base.describe()}\n  input {i}: {other.describe()}"
            )


def strip_panel_metadata(frame: Any) -> Any:
    """Remove the reserved ``__fe_time__`` metadata column before the kernel.

    Identity verification runs on the frame *with* ``__fe_time__`` (the bridge
    preserves the DatetimeIndex there, #245); stripping afterwards ensures value
    column extraction (``pl_cols`` / ``numeric_cols`` / ``SKIP``) never treats
    the time axis as a factor feature.
    """
    if FE_TIME_COL not in getattr(frame, "columns", ()):
        return frame
    if pl is not None and isinstance(frame, pl.DataFrame):
        return frame.drop(FE_TIME_COL)
    if isinstance(frame, pd.DataFrame):
        return frame.drop(columns=[FE_TIME_COL])
    return frame


# ---------------------------------------------------------------------------
# MissingNumeric unification (#241): MissingNumeric := null OR NaN
# ---------------------------------------------------------------------------
def missing_numeric_polars(frame: "pl.DataFrame") -> "pl.DataFrame":
    """Boolean mask frame where ``MissingNumeric = is_null OR is_nan``."""
    if pl is None or not isinstance(frame, pl.DataFrame):
        raise TypeError("missing_numeric_polars requires a polars DataFrame")
    exprs = []
    for c in frame.columns:
        dtype = frame[c].dtype
        if dtype in (pl.Float32, pl.Float64):
            exprs.append((pl.col(c).is_null() | pl.col(c).is_nan()).alias(c))
        else:
            exprs.append(pl.col(c).is_null().alias(c))
    return frame.select(exprs)


def missing_numeric_pandas(panel: pd.DataFrame) -> pd.DataFrame:
    """Boolean mask frame where pandas considers the value missing (NaN or None)."""
    return panel.isna()


def polars_nan_to_null(frame: "pl.DataFrame") -> "pl.DataFrame":
    """Map NaN → null in every float column (pandas null semantics parity, #241)."""
    if pl is None or not isinstance(frame, pl.DataFrame):
        return frame
    exprs = []
    for c in frame.columns:
        if frame[c].dtype in (pl.Float32, pl.Float64):
            exprs.append(
                pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
            )
    if not exprs:
        return frame
    return frame.with_columns(exprs)


def normalize_missing_numeric(frame: Any, *, backend: str) -> Any:
    """Normalize NaN↔null so pandas/Polars agree on ``MissingNumeric``.

    ``backend="polars"`` maps NaN → null (Polars stores NaN distinct from null);
    ``backend="pandas"`` is a no-op because numeric null already renders as NaN.
    """
    backend = str(backend).lower()
    if backend == "polars":
        return polars_nan_to_null(frame)
    if backend == "pandas":
        return frame
    raise ValueError(f"unknown backend for normalize_missing_numeric: {backend!r}")


# ---------------------------------------------------------------------------
# #247: optional-polars ImportError policy
# ---------------------------------------------------------------------------
def should_skip_optional_import_error(exc: BaseException, mod: str) -> bool:
    """Whether an ImportError while importing ``mod`` may be swallowed.

    Only two cases are safe to skip:
      1. the module itself (or a parent package) does not exist — planned /
         next-stage surface that carries no operators yet;
      2. the only missing dependency is the optional ``polars`` package
         (``ModuleNotFoundError`` with ``exc.name == "polars"`` or a polars
         submodule).
    Any other ``ImportError`` (missing non-optional dependency, code error
    during import, ...) is fatal and must be re-raised.
    """
    if not isinstance(exc, ModuleNotFoundError):
        return False
    name = exc.name or ""
    if not name:
        return False
    if mod == name or mod.startswith(name + "."):
        return True  # module / parent package genuinely absent (planned surface)
    if name == "polars" or name.startswith("polars."):
        return True  # optional dependency absent
    return False


def numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in SKIP]


def align_cols(*dfs: pl.DataFrame) -> list[str]:
    """Strict multi-panel alignment (WS-B #242): never take the column intersection.

    Every input must share the same PanelIdentity (same time axis, same ordered
    instrument columns).  A frame that silently drops a stock — columns differing
    by name or order — raises ``ValueError`` instead of computing on the
    intersection.
    """
    if not dfs:
        return []
    verify_frames_share_identity("align_cols", *dfs)
    base_cols = numeric_cols(dfs[0])
    for i, df in enumerate(dfs[1:], start=1):
        other_cols = numeric_cols(df)
        if other_cols != base_cols:
            raise ValueError(
                f"align_cols: panel input {i} instrument columns {other_cols} "
                f"!= base {base_cols} (a silent intersection would drop stocks)"
            )
    return base_cols


def to_pandas_panel(df: pl.DataFrame) -> pd.DataFrame:
    return df.select(numeric_cols(df)).to_pandas()


def from_pandas_panel(base: pl.DataFrame, out: pd.DataFrame) -> pl.DataFrame:
    """Write a pandas result back onto the base polars frame (strict).

    A pandas result that is *missing* a base instrument column drops that stock
    silently — rejected (WS-B #242).  Extra columns in ``out`` that are not part
    of the base value set are ignored.
    """
    cols = numeric_cols(base)
    missing = [c for c in cols if c not in out.columns]
    if missing:
        raise ValueError(
            f"from_pandas_panel: pandas result is missing instrument columns {missing}; "
            "a silent drop would lose stocks"
        )
    return base.with_columns([pl.Series(name=c, values=out[c].to_numpy()) for c in cols])


def bridge_pandas(
    x: pl.DataFrame,
    compute: Callable[..., pd.DataFrame],
    *args: Any,
    **kwargs: Any,
) -> pl.DataFrame:
    """宽表 Polars → pandas 计算 → 写回 Polars。"""
    pdf = to_pandas_panel(x)
    pargs = []
    for arg in args:
        if isinstance(arg, pl.DataFrame):
            pargs.append(to_pandas_panel(arg))
        else:
            pargs.append(arg)
    out = compute(pdf, *pargs, **kwargs)
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f"bridge_pandas expects DataFrame result, got {type(out)}")
    return from_pandas_panel(x, out)


def bridge_registry(canonical: str, x: pl.DataFrame, *args: Any, **kwargs: Any) -> pl.DataFrame:
    from cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get(canonical, backend="pandas_numpy")
    pargs = []
    for arg in args:
        pargs.append(to_pandas_panel(arg) if isinstance(arg, pl.DataFrame) else arg)
    return bridge_pandas(x, op.calculate, *pargs, **kwargs)


def colwise_numpy_kernel(
    x: pl.DataFrame,
    kernel: Callable[..., np.ndarray],
    *args: Any,
    **kwargs: Any,
) -> pl.DataFrame:
    """逐列应用 numpy 向量核（与 time_series._apply_colwise_kernel 对齐）。"""
    cols = numeric_cols(x)
    pdf = to_pandas_panel(x)
    out = pd.DataFrame(index=pdf.index, columns=cols, dtype=float)
    for col in cols:
        out[col] = kernel(pdf[col].to_numpy(), *args, **kwargs)
    return from_pandas_panel(x, out)
