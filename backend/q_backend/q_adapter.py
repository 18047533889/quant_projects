"""q/K Type Adapter with Versioned Semantics.

处理 Python/Pandas/Arrow ↔ q 的类型转换（文档 §26, §P2-008）。

Hard Gates (文档 §85):
- Q_BACKEND_NULL_TIME_SEMANTICS_CERTIFIED: null/time 语义必须认证
- Q_BACKEND_PIT_PARITY: PIT 语义保持

显式定义（文档 §26）:
- q null
- float NaN/inf
- int null
- symbol
- string
- timestamp
- date
- timespan
- boolean null
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QResidentTableHandle:
    """Handle representing a table already resident in Q process memory.

    Allows consecutive Q regions to reuse intermediate results without re-uploading.
    Eliminates region-level data pingpong (文档 §85 Q_BACKEND_OPERATOR_LEVEL_PINGPONG_ZERO).
    """
    table_name: str  # Logical name exposed to the region planner
    q_table_ref: Any  # Reference to Q table object
    row_count: int
    byte_size: int  # Estimated memory footprint
    region_id: str  # Origin region ID
    connection_id: int | None = None  # Identity of the owning q connection
    workspace_id: str | None = None  # Owning execution workspace
    generation_id: str | None = None  # Lease generation within the process
    q_symbol: str | None = None  # Physical symbol in the owning workspace
    _release_callback: Any = field(default=None, repr=False, compare=False)
    _released: bool = field(default=False, init=False, repr=False, compare=False)

    def release(self) -> None:
        """Release the owning q workspace lease exactly once."""
        if self._released:
            return
        callback = self._release_callback
        object.__setattr__(self, "_released", True)
        if callback is not None:
            callback(self)

    close = release

    def __enter__(self) -> "QResidentTableHandle":
        if self._released:
            raise RuntimeError("Q-resident handle has already been released")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()
        return None

    def __repr__(self) -> str:
        return (
            f"QResidentTableHandle(table={self.table_name}, "
            f"rows={self.row_count}, bytes={self.byte_size}, "
            f"region={self.region_id}, workspace={self.workspace_id}, "
            f"generation={self.generation_id})"
        )


class QType(Enum):
    """q/K 数据类型。"""
    BOOLEAN = "boolean"  # `boolean$
    BYTE = "byte"  # `byte$
    SHORT = "short"  # `short$
    INT = "int"  # `int$
    LONG = "long"  # `long$
    REAL = "real"  # `real$
    FLOAT = "float"  # `float$
    CHAR = "char"  # `char$
    SYMBOL = "symbol"  # `symbol$
    TIMESTAMP = "timestamp"  # `timestamp$
    DATE = "date"  # `date$
    TIMESPAN = "timespan"  # `timespan$
    TIME = "time"  # `time$


@dataclass(frozen=True)
class QNullSemantics:
    """q null 值语义定义。

    Q2-P0-023: NULL semantics must be distinct from default values.
    - Boolean null is conceptually 3-state but q represents as 0b (indistinguishable from False)
    - Symbol null is ` (empty symbol), distinct from "" (char vector) and " " (char)
    - Integer nulls have distinct sentinels per width
    """
    # q null values by type
    null_boolean: int = 0  # 0b (WARNING: indistinguishable from False in q)
    null_byte: int = 0x00
    null_short: int = -32768  # 0Nh
    null_int: int = -2147483648  # 0Ni
    null_long: int = -9223372036854775808  # 0Nj (note: j suffix)
    null_real: float = float("nan")  # 0Ne
    null_float: float = float("nan")  # 0n
    null_char: str = " "
    null_symbol: str = ""  # ` (backtick; empty symbol, NOT empty string)
    # Infinity
    pos_inf_real: float = float("inf")  # 0we
    neg_inf_real: float = float("-inf")  # -0we
    pos_inf_float: float = float("inf")  # 0w
    neg_inf_float: float = float("-inf")  # -0w


# Adapter 版本（文档 §P2-008: 版本化）
Q_ADAPTER_VERSION = "v1.0.0-phase1"


class QZeroCopyUnavailable(RuntimeError):
    """Raised only when a q boundary explicitly cannot do zero-copy."""


class QTypeAdapter:
    """q/K 类型适配器.

    ``QType`` and ``QNullSemantics`` are the sole type/null authorities.  A
    caller may provide a column-to-``QType`` schema when pandas' dtype is
    ambiguous (notably ``object``); guessing a symbol in that case is unsafe.
    PyKX zero-copy is an optimization and never a semantic fallback.
    """

    def __init__(self):
        self.semantics = QNullSemantics()
        self.version = Q_ADAPTER_VERSION

    def pandas_to_q_type(self, dtype: Any) -> QType:
        """Pandas dtype → q type 映射。

        参数:
            dtype: Pandas/NumPy dtype

        返回:
            对应的 q 类型

        抛出:
            ValueError: 不支持的类型
        """
        if dtype == np.bool_:
            return QType.BOOLEAN
        elif dtype == np.int8:
            return QType.BYTE
        elif dtype == np.int16:
            return QType.SHORT
        elif dtype == np.int32:
            return QType.INT
        elif dtype == np.int64:
            return QType.LONG
        elif dtype == np.float32:
            return QType.REAL
        elif dtype == np.float64:
            return QType.FLOAT
        elif dtype == np.object_:
            raise TypeError(
                f"Object dtype for column {dtype!r} requires an explicit QType schema"
            )
        elif np.issubdtype(dtype, np.datetime64):
            return QType.TIMESTAMP
        else:
            raise ValueError(f"Unsupported pandas dtype for q: {dtype}")

    def q_type_to_pandas(self, q_type: QType) -> np.dtype:
        """q type → Pandas dtype 映射。

        参数:
            q_type: q 类型

        返回:
            对应的 Pandas dtype
        """
        mapping = {
            QType.BOOLEAN: np.bool_,
            QType.BYTE: np.int8,
            QType.SHORT: np.int16,
            QType.INT: np.int32,
            QType.LONG: np.int64,
            QType.REAL: np.float32,
            QType.FLOAT: np.float64,
            QType.CHAR: np.object_,
            QType.SYMBOL: np.object_,
            QType.TIMESTAMP: "datetime64[ns]",
            QType.DATE: "datetime64[ns]",
            QType.TIMESPAN: "timedelta64[ns]",
            QType.TIME: "timedelta64[ns]",
        }
        return np.dtype(mapping[q_type])

    def pandas_to_q(
        self,
        df: pd.DataFrame,
        *,
        preserve_index: bool = True,
        zero_copy: bool = False,
        schema: Mapping[str, QType] | None = None,
        q_module: Any | None = None,
    ) -> Any:
        """Pandas DataFrame → q table。

        参数:
            df: Pandas DataFrame
            preserve_index: 是否保留索引
            zero_copy: 是否尝试零拷贝（仅优化）

        返回:
            q table 对象
        """
        try:
            kx = q_module
            if kx is None:
                import pykx as kx
        except ImportError:
            raise RuntimeError("PyKX not available for pandas_to_q conversion")

        if schema is not None:
            unknown = set(schema) - set(df.columns)
            if unknown:
                raise ValueError(f"Q schema names missing from DataFrame: {sorted(unknown)}")

        # 处理索引
        if preserve_index and not isinstance(df.index, pd.RangeIndex):
            df = df.reset_index()

        converted_df = self._prepare_pandas_for_q(df, schema=schema)

        if zero_copy:
            try:
                q_table = kx.toq(converted_df, zero_copy=True)
            except QZeroCopyUnavailable:
                logger.debug("Zero-copy unavailable, using semantic copy conversion")
                q_table = kx.toq(converted_df, zero_copy=False)
        else:
            q_table = kx.toq(converted_df, zero_copy=False)

        return q_table

    def q_to_pandas(
        self,
        q_obj: Any,
        *,
        handle_nulls: bool = True,
        schema: Mapping[str, QType] | None = None,
    ) -> pd.DataFrame:
        """q table → Pandas DataFrame。

        参数:
            q_obj: q 对象
            handle_nulls: 是否处理 q null → pandas NaN

        返回:
            Pandas DataFrame
        """
        try:
            import pykx as kx
        except ImportError:
            raise RuntimeError("PyKX not available for q_to_pandas conversion")

        # 转换到 Pandas
        df = q_obj.pd()

        # 处理 null 语义
        if handle_nulls:
            df = self._handle_q_nulls(df, schema=schema)

        return df

    def _prepare_pandas_for_q(
        self,
        df: pd.DataFrame,
        *,
        schema: Mapping[str, QType] | None = None,
    ) -> pd.DataFrame:
        """准备 Pandas DataFrame 以兼容 q。

        参数:
            df: 输入 DataFrame

        返回:
            转换后的 DataFrame
        """
        result = df.copy()

        for col in result.columns:
            dtype = result[col].dtype
            q_type = schema.get(col) if schema is not None else None
            if q_type is None:
                q_type = self._infer_nullable_q_type(result[col])

            if q_type in (QType.SYMBOL, QType.CHAR):
                # Keep missing values as pd.NA.  In particular, never stringify
                # the column: None/NA, empty string, and an empty q symbol differ.
                if not (pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)):
                    raise TypeError(f"Column {col!r} is not text-compatible with {q_type.value}")
                result[col] = result[col].astype("string")
            elif q_type == QType.BOOLEAN:
                if not (pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_object_dtype(dtype)):
                    raise TypeError(f"Column {col!r} is not boolean-compatible")
                result[col] = result[col].astype("boolean")
            elif q_type in (QType.BYTE, QType.SHORT, QType.INT, QType.LONG):
                target = {
                    QType.BYTE: "Int8", QType.SHORT: "Int16",
                    QType.INT: "Int32", QType.LONG: "Int64",
                }[q_type]
                if not (pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_object_dtype(dtype)):
                    raise TypeError(f"Column {col!r} is not integer-compatible with {q_type.value}")
                result[col] = result[col].astype(target)
            elif q_type in (QType.REAL, QType.FLOAT):
                if not (pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_object_dtype(dtype)):
                    raise TypeError(f"Column {col!r} is not numeric-compatible with {q_type.value}")
                result[col] = result[col].astype("float32" if q_type == QType.REAL else "float64")
            elif q_type in (QType.TIMESTAMP, QType.DATE, QType.TIMESPAN, QType.TIME):
                if not (pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype)):
                    raise TypeError(f"Column {col!r} is not temporal-compatible with {q_type.value}")
            else:
                raise TypeError(f"Unsupported Q schema type for column {col!r}: {q_type!r}")

        return result

    def _infer_nullable_q_type(self, series: pd.Series) -> QType:
        """Infer only unambiguous extension/numpy dtypes; object requires schema."""
        dtype = series.dtype
        if pd.api.types.is_object_dtype(dtype):
            values = series.dropna()
            if values.empty or all(isinstance(value, str) for value in values):
                raise TypeError(
                    f"Object column {series.name!r} requires an explicit QType schema"
                )
            raise TypeError(f"Object column {series.name!r} has ambiguous values")
        if pd.api.types.is_string_dtype(dtype):
            raise TypeError(
                f"String column {series.name!r} requires an explicit QType schema"
            )
        if pd.api.types.is_bool_dtype(dtype):
            return QType.BOOLEAN
        if pd.api.types.is_integer_dtype(dtype):
            bits = dtype.numpy_dtype.itemsize * 8
            return {8: QType.BYTE, 16: QType.SHORT, 32: QType.INT, 64: QType.LONG}[bits]
        if pd.api.types.is_float_dtype(dtype):
            return QType.REAL if dtype.itemsize == 4 else QType.FLOAT
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return QType.TIMESTAMP
        if pd.api.types.is_timedelta64_dtype(dtype):
            return QType.TIMESPAN
        raise TypeError(f"Unsupported pandas dtype for Q conversion: {dtype}")

    def _handle_q_nulls(
        self,
        df: pd.DataFrame,
        *,
        schema: Mapping[str, QType] | None = None,
    ) -> pd.DataFrame:
        """处理 q null → Pandas NaN/NaT。

        参数:
            df: 从 q 转换的 DataFrame

        返回:
            处理后的 DataFrame
        """
        result = df.copy()

        for col in result.columns:
            dtype = result[col].dtype
            q_type = schema.get(col) if schema is not None else None

            if pd.api.types.is_integer_dtype(dtype):
                # Apply the per-width null sentinel for the declared q type, not
                # one global int null.  Byte is skipped: q byte null (0x00)
                # equals the real value 0, so rewriting it would corrupt data.
                sentinel = None
                nullable_dtype = None
                if q_type in (QType.SHORT, QType.INT, QType.LONG):
                    sentinel = {
                        QType.SHORT: self.semantics.null_short,
                        QType.INT: self.semantics.null_int,
                        QType.LONG: self.semantics.null_long,
                    }[q_type]
                    nullable_dtype = {
                        QType.SHORT: "Int16", QType.INT: "Int32", QType.LONG: "Int64",
                    }[q_type]
                elif q_type is None:
                    try:
                        bits = dtype.numpy_dtype.itemsize * 8
                    except AttributeError:
                        bits = dtype.itemsize * 8
                    if bits == 16:
                        sentinel, nullable_dtype = self.semantics.null_short, "Int16"
                    elif bits == 32:
                        sentinel, nullable_dtype = self.semantics.null_int, "Int32"
                    elif bits == 64:
                        sentinel, nullable_dtype = self.semantics.null_long, "Int64"
                if sentinel is not None:
                    mask = result[col].eq(sentinel)
                    if mask.any():
                        result[col] = result[col].astype(nullable_dtype)
                        result.loc[mask, col] = pd.NA

            # 浮点 null 已经是 NaN，无需处理
            elif pd.api.types.is_floating_dtype(dtype):
                pass

            # 符号 null（仅当 schema 声明为 symbol；否则保持原值，避免把
            # 真实空字符串 "" 误当 null 合并）
            elif q_type == QType.SYMBOL and pd.api.types.is_object_dtype(dtype):
                result[col] = result[col].replace(self.semantics.null_symbol, pd.NA)

        return result

    def infer_pit_metadata(
        self,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """推断 PIT 元数据（确保 PIT 语义保持）。

        文档 §26: q as-of join 再快，也必须由 DataAccess 给定:
        - available_at
        - revision vintage
        - source snapshot
        - DecisionClock

        参数:
            df: DataFrame

        返回:
            PIT 元数据字典
        """
        metadata = {
            "adapter_version": self.version,
            "has_datetime_index": isinstance(df.index, pd.DatetimeIndex),
        }

        # 检测时间列
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.datetime64):
                metadata["time_columns"] = metadata.get("time_columns", []) + [col]

        # 警告：不要让 q 决定 PIT
        logger.debug(
            "PIT metadata inferred from structure only. "
            "Semantic PIT (available_at, vintage, snapshot) must come from DataAccess."
        )

        return metadata


# Global singleton
_TYPE_ADAPTER: QTypeAdapter | None = None


def get_q_type_adapter() -> QTypeAdapter:
    """获取全局 q 类型适配器。"""
    global _TYPE_ADAPTER
    if _TYPE_ADAPTER is None:
        _TYPE_ADAPTER = QTypeAdapter()
    return _TYPE_ADAPTER
