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
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
    """q null 值语义定义。"""
    # q null values by type
    null_boolean: int = 0  # 0b
    null_byte: int = 0x00
    null_short: int = -32768  # 0Nh
    null_int: int = -2147483648  # 0Ni
    null_long: int = -9223372036854775808  # 0N
    null_real: float = float("nan")  # 0Ne
    null_float: float = float("nan")  # 0n
    null_char: str = " "
    null_symbol: str = ""  # `
    # Infinity
    pos_inf_real: float = float("inf")  # 0we
    neg_inf_real: float = float("-inf")  # -0we
    pos_inf_float: float = float("inf")  # 0w
    neg_inf_float: float = float("-inf")  # -0w


# Adapter 版本（文档 §P2-008: 版本化）
Q_ADAPTER_VERSION = "v1.0.0-phase1"


class QTypeAdapter:
    """q/K 类型适配器。

    遵循文档 §P2-007: PyKX zero-copy 只是优化，不是 correctness 前提。
    """

    def __init__(self):
        self.semantics = QNullSemantics()
        self.version = Q_ADAPTER_VERSION

    def pandas_to_q_type(self, dtype: np.dtype) -> QType:
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
            return QType.SYMBOL  # Default for object
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
            import pykx as kx
        except ImportError:
            raise RuntimeError("PyKX not available for pandas_to_q conversion")

        # 处理索引
        if preserve_index and not isinstance(df.index, pd.RangeIndex):
            df = df.reset_index()

        # 类型转换：确保兼容 q
        converted_df = self._prepare_pandas_for_q(df)

        # 转换到 q（zero_copy 只是优化提示）
        try:
            if zero_copy:
                # 尝试零拷贝，但不强制
                q_table = kx.toq(converted_df, zero_copy=True)
            else:
                q_table = kx.toq(converted_df)
        except Exception as e:
            logger.warning(f"Zero-copy conversion failed, falling back: {e}")
            q_table = kx.toq(converted_df)

        return q_table

    def q_to_pandas(
        self,
        q_obj: Any,
        *,
        handle_nulls: bool = True,
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
            df = self._handle_q_nulls(df)

        return df

    def _prepare_pandas_for_q(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备 Pandas DataFrame 以兼容 q。

        参数:
            df: 输入 DataFrame

        返回:
            转换后的 DataFrame
        """
        result = df.copy()

        for col in result.columns:
            dtype = result[col].dtype

            # NaN/inf 处理
            if np.issubdtype(dtype, np.floating):
                # Pandas NaN → q 0n (float null)
                # Pandas inf → q 0w (infinity)
                # 这些在 q 中有标准表示，无需特殊处理
                pass

            # 对象类型 → symbol
            elif dtype == np.object_:
                # 确保字符串可以转为 symbol
                result[col] = result[col].fillna("").astype(str)

            # datetime → timestamp
            elif np.issubdtype(dtype, np.datetime64):
                # Pandas datetime64[ns] → q timestamp
                pass

        return result

    def _handle_q_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理 q null → Pandas NaN/NaT。

        参数:
            df: 从 q 转换的 DataFrame

        返回:
            处理后的 DataFrame
        """
        result = df.copy()

        for col in result.columns:
            dtype = result[col].dtype

            # 整数 null → NaN (升级为 float)
            if np.issubdtype(dtype, np.integer):
                # q null int (-2147483648) → pandas NaN
                null_mask = result[col] == self.semantics.null_int
                if null_mask.any():
                    result[col] = result[col].astype(float)
                    result[col][null_mask] = np.nan

            # 浮点 null 已经是 NaN，无需处理
            elif np.issubdtype(dtype, np.floating):
                pass

            # 符号 null
            elif dtype == np.object_:
                result[col] = result[col].replace(self.semantics.null_symbol, None)

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
