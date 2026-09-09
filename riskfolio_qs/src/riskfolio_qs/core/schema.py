"""
Schema 校验工具函数：提供 DataFrame 格式和内容的基础校验能力。

这些工具函数被 SchemaValidator 调用，也可在数据适配层直接使用。
所有校验函数在失败时抛出异常，成功时无返回值（返回 None）。

主要校验维度：
- 索引类型（必须为 DatetimeIndex）
- 索引单调性（必须单调递增）
- 数值合法性（不允许全 NaN / 全 inf）
- 列对齐（两个 DataFrame 的列名必须一致）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _is_datetime_index(frame: pd.DataFrame) -> bool:
    """判断 DataFrame 的 index 是否为 DatetimeIndex。"""
    return isinstance(frame.index, pd.DatetimeIndex)


def require_datetime_index(frame: pd.DataFrame, name: str) -> None:
    """要求 frame 必须是 DataFrame 且 index 为单调递增的 DatetimeIndex。

    Args:
        frame: 待校验的 DataFrame
        name: 数据名称（用于错误消息）

    Raises:
        TypeError: frame 不是 DataFrame
        ValueError: index 不是 DatetimeIndex 或不是单调递增
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if not _is_datetime_index(frame):
        raise ValueError(f"{name} index must be a DatetimeIndex")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be monotonic increasing")
    if frame.index.has_duplicates:
        raise ValueError(f"{name} index must not contain duplicates")
    if frame.columns.has_duplicates:
        raise ValueError(f"{name} columns must not contain duplicates")
    if frame.empty or len(frame.columns) == 0:
        raise ValueError(f"{name} must not be empty")


def require_no_inf(frame: pd.DataFrame, name: str) -> None:
    """要求 frame 不能是全 NaN 且不能只包含 inf/NaN。

    Args:
        frame: 待校验的 DataFrame
        name: 数据名称

    Raises:
        ValueError: 数据全 NaN 或只含 inf/NaN
    """
    values = frame.to_numpy()
    if pd.isna(values).all():
        raise ValueError(f"{name} cannot be all NaN")
    numeric = pd.to_numeric(pd.Series(values.ravel()), errors="coerce").to_numpy()
    if np.isinf(numeric).any():
        raise ValueError(f"{name} must not contain inf")
    if not np.isfinite(numeric).any():
        raise ValueError(f"{name} cannot contain only NaN")


def require_matching_columns(lhs: pd.DataFrame, rhs: pd.DataFrame, lhs_name: str, rhs_name: str) -> None:
    """要求两个 DataFrame 的列名完全一致。

    Args:
        lhs: 左侧 DataFrame
        rhs: 右侧 DataFrame
        lhs_name: 左侧数据名称
        rhs_name: 右侧数据名称

    Raises:
        ValueError: 列名不一致，并列出不匹配的前 5 个列名
    """
    lhs_cols = list(lhs.columns)
    rhs_cols = list(rhs.columns)
    if lhs_cols != rhs_cols:
        missing_left = [c for c in rhs_cols if c not in lhs_cols]
        missing_right = [c for c in lhs_cols if c not in rhs_cols]
        raise ValueError(
            f"{lhs_name} and {rhs_name} columns mismatch: missing_left={missing_left[:5]}, missing_right={missing_right[:5]}"
        )


def validate_optional_frame(frame: Optional[pd.DataFrame], name: str) -> None:
    """校验可选 DataFrame：None 直接通过，非 None 则要求 DatetimeIndex。

    Args:
        frame: 可选的 DataFrame
        name: 数据名称
    """
    if frame is None:
        return
    require_datetime_index(frame, name)


@dataclass(slots=True)
class SchemaSpec:
    """Schema 字段名规范：定义输入数据中标准字段的命名约定。

    用于 InputAdapter 子类在加载数据时统一字段命名，
    避免硬编码字符串分散在各处。
    """
    alpha_field: str = "alpha"
    market_close_field: str = "close"
    market_return_field: str = "ret_1d"
    benchmark_weight_field: str = "benchmark_weight"
    prev_weight_field: str = "prev_weight"
