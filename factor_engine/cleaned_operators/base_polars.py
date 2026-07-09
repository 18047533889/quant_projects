from __future__ import annotations
# -*- coding: utf-8 -*-
"""
Polars 算子基类（实验/备用 backend）。

与 ``base.py`` 结构相同，但 ``register_operator`` 会把 backend 标为 ``polars``。
当前生产路径 ``PandasBackend`` 使用 ``pandas_numpy`` 实现；Polars 类供后续加速或对照测试。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore
import numpy as np
try:
    from numba import jit, prange
except ImportError:
    def jit(*_a, **_k):
        def _wrap(fn):
            return fn
        return _wrap
    prange = range


@dataclass
class OperatorMetadata:
    """操作符元数据"""
    name: str
    category: str
    description: str = ""
    examples: List[str] = field(default_factory=list)
    param_names: List[str] = field(default_factory=list)
    param_types: Dict[str, type] = field(default_factory=dict)
    return_type: str = "series"
    enabled: bool = True
    tags: List[str] = field(default_factory=list)


class Operator(ABC):
    """所有操作符的基类"""

    metadata: OperatorMetadata

    @abstractmethod
    def calculate(self, *args, **kwargs) -> pl.DataFrame:
        """执行计算，子类必须实现"""
        pass

    def validate_params(self, *args, **kwargs) -> bool:
        """验证参数是否合法，默认直接返回True"""
        return True

    def __repr__(self):
        return f"<Operator: {self.metadata.name}>"

    def __str__(self):
        return f"{self.metadata.name}: {self.metadata.description}"


class SeriesOperator(Operator):
    """序列操作符基类（输入输出都是DataFrame）"""

    def calculate(self, *args, **kwargs) -> pl.DataFrame:
        processed_args = []
        for a in args:
            if isinstance(a, float) and a == int(a):
                processed_args.append(int(a))
            elif isinstance(a, pl.DataFrame):
                processed_args.append(a)
            else:
                processed_args.append(a)
        return self._calculate_series(*processed_args, **kwargs)

    @abstractmethod
    def _calculate_series(self, *args, **kwargs) -> pl.DataFrame:
        """子类实现序列计算逻辑"""
        pass


class ScalarOperator(Operator):
    """标量操作符基类（输出是单个值）"""

    def calculate(self, *args, **kwargs) -> Any:
        return self._calculate_scalar(*args, **kwargs)

    @abstractmethod
    def _calculate_scalar(self, *args, **kwargs) -> Any:
        """子类实现标量计算逻辑"""
        pass


class TransformOperator(Operator):
    """变换操作符基类（输入一个序列，输出一个序列）"""

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """子类实现"""
        raise NotImplementedError

    def calculate(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return self._calculate_series(x, **kwargs)


class TwoVarOperator(Operator):
    """双变量操作符基类"""

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """子类实现"""
        raise NotImplementedError

    def calculate(self, x: pl.DataFrame, y: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return self._calculate_series(x, y, **kwargs)


def register_operator(
    name: str = None,
    category: str = "general",
    business_category: str = "",
    canonical: str = "",
    source: str = "",
    backend: str = "polars",
    status: str = "implemented",
):
    """操作符装饰器"""
    def decorator(cls):
        instance = cls()
        if name:
            instance.metadata.name = name
        if category:
            instance.metadata.category = category
        if business_category:
            instance.metadata.business_category = business_category
        canon = canonical or (name if name else cls.__name__)
        from cleaned_operators.registry import OperatorRegistry
        name_aliases = [name] if name and name != canon else None
        OperatorRegistry.register(
            instance,
            canonical=canon,
            backend=backend,
            source=source or "factor_dsl_np",
            aliases=name_aliases,
            status=status,
        )
        return cls
    return decorator


# Numba 加速的辅助函数
@jit(nopython=True, parallel=True, cache=True)
def _numba_rolling_mean_2d(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba 加速的2D滚动均值计算"""
    rows, cols = arr.shape
    result = np.full_like(arr, np.nan)

    for col in prange(cols):
        for i in range(window - 1, rows):
            sum_val = 0.0
            count = 0
            for j in range(i - window + 1, i + 1):
                if not np.isnan(arr[j, col]):
                    sum_val += arr[j, col]
                    count += 1
            if count > 0:
                result[i, col] = sum_val / count

    return result


@jit(nopython=True, parallel=True, cache=True)
def _numba_rolling_std_2d(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba 加速的2D滚动标准差计算"""
    rows, cols = arr.shape
    result = np.full_like(arr, np.nan)

    for col in prange(cols):
        for i in range(window - 1, rows):
            sum_val = 0.0
            sum_sq = 0.0
            count = 0
            for j in range(i - window + 1, i + 1):
                if not np.isnan(arr[j, col]):
                    sum_val += arr[j, col]
                    sum_sq += arr[j, col] * arr[j, col]
                    count += 1
            if count > 1:
                mean = sum_val / count
                variance = (sum_sq / count) - (mean * mean)
                result[i, col] = np.sqrt(max(variance, 0.0))

    return result


@jit(nopython=True, parallel=True, cache=True)
def _numba_rolling_sum_2d(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba 加速的2D滚动求和计算"""
    rows, cols = arr.shape
    result = np.full_like(arr, np.nan)

    for col in prange(cols):
        for i in range(window - 1, rows):
            sum_val = 0.0
            for j in range(i - window + 1, i + 1):
                if not np.isnan(arr[j, col]):
                    sum_val += arr[j, col]
            result[i, col] = sum_val

    return result


@jit(nopython=True, parallel=True, cache=True)
def _numba_rolling_max_2d(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba 加速的2D滚动最大值计算"""
    rows, cols = arr.shape
    result = np.full_like(arr, np.nan)

    for col in prange(cols):
        for i in range(window - 1, rows):
            max_val = -np.inf
            for j in range(i - window + 1, i + 1):
                if not np.isnan(arr[j, col]):
                    max_val = max(max_val, arr[j, col])
            if max_val != -np.inf:
                result[i, col] = max_val

    return result


@jit(nopython=True, parallel=True, cache=True)
def _numba_rolling_min_2d(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba 加速的2D滚动最小值计算"""
    rows, cols = arr.shape
    result = np.full_like(arr, np.nan)

    for col in prange(cols):
        for i in range(window - 1, rows):
            min_val = np.inf
            for j in range(i - window + 1, i + 1):
                if not np.isnan(arr[j, col]):
                    min_val = min(min_val, arr[j, col])
            if min_val != np.inf:
                result[i, col] = min_val

    return result


@jit(nopython=True, parallel=True, cache=True)
def _numba_zscore_2d(arr: np.ndarray) -> np.ndarray:
    """Numba 加速的2D Z-Score计算（按行计算）"""
    rows, cols = arr.shape
    result = np.full_like(arr, np.nan)

    for row in prange(rows):
        # 计算均值
        sum_val = 0.0
        count = 0
        for col in range(cols):
            if not np.isnan(arr[row, col]):
                sum_val += arr[row, col]
                count += 1

        if count == 0:
            continue

        mean = sum_val / count

        # 计算标准差
        sum_sq = 0.0
        for col in range(cols):
            if not np.isnan(arr[row, col]):
                diff = arr[row, col] - mean
                sum_sq += diff * diff

        std = np.sqrt(sum_sq / count) if count > 0 else 0.0
        if std == 0:
            std = 1.0

        # 计算 Z-Score
        for col in range(cols):
            if not np.isnan(arr[row, col]):
                result[row, col] = (arr[row, col] - mean) / std

    return result


@jit(nopython=True, parallel=True, cache=True)
def _numba_rank_2d(arr: np.ndarray) -> np.ndarray:
    """Numba 加速的2D排名计算（按行计算，百分比排名 0-1）"""
    rows, cols = arr.shape
    result = np.full_like(arr, np.nan)

    for row in prange(rows):
        # 获取有效值索引
        valid_indices = []
        valid_values = []
        for col in range(cols):
            if not np.isnan(arr[row, col]):
                valid_indices.append(col)
                valid_values.append(arr[row, col])

        m = len(valid_values)
        if m == 0:
            continue

        # 计算排名（使用 argsort 的 argsort）
        valid_values = np.array(valid_values)
        order = np.argsort(valid_values)
        ranks = np.argsort(order) + 1  # 1-based rank

        # 归一化到 0-1
        for idx, col in enumerate(valid_indices):
            result[row, col] = ranks[idx] / m

    return result


def apply_numba_rolling(df: pl.DataFrame, window: int, func_name: str = 'mean') -> pl.DataFrame:
    """
    应用 Numba 加速的滚动计算

    Args:
        df: Polars DataFrame
        window: 窗口大小
        func_name: 函数名称 ('mean', 'std', 'sum', 'max', 'min')

    Returns:
        计算后的 Polars DataFrame
    """
    # 获取数值列
    numeric_cols = [c for c in df.columns if c not in ['date', 'stock_code']]
    if not numeric_cols:
        return df

    # 转换为 numpy 数组
    arr = df.select(numeric_cols).to_numpy()

    # 应用 Numba 函数
    func_map = {
        'mean': _numba_rolling_mean_2d,
        'std': _numba_rolling_std_2d,
        'sum': _numba_rolling_sum_2d,
        'max': _numba_rolling_max_2d,
        'min': _numba_rolling_min_2d,
    }

    if func_name not in func_map:
        raise ValueError(f"Unknown function: {func_name}")

    result_arr = func_map[func_name](arr, window)

    # 转换回 Polars DataFrame
    result_df = pl.DataFrame(result_arr, schema=numeric_cols)

    # 添加日期列
    if 'date' in df.columns:
        result_df = result_df.with_columns([df['date']])

    return result_df


def apply_numba_zscore(df: pl.DataFrame, axis: int = 1) -> pl.DataFrame:
    """
    应用 Numba 加速的 Z-Score 计算

    Args:
        df: Polars DataFrame
        axis: 计算轴（1=按行，0=按列）

    Returns:
        计算后的 Polars DataFrame
    """
    numeric_cols = [c for c in df.columns if c not in ['date', 'stock_code']]
    if not numeric_cols:
        return df

    arr = df.select(numeric_cols).to_numpy()

    if axis == 1:
        result_arr = _numba_zscore_2d(arr)
    else:
        # 转置后计算
        result_arr = _numba_zscore_2d(arr.T).T

    result_df = pl.DataFrame(result_arr, schema=numeric_cols)

    if 'date' in df.columns:
        result_df = result_df.with_columns([df['date']])

    return result_df


def apply_numba_rank(df: pl.DataFrame, axis: int = 1) -> pl.DataFrame:
    """
    应用 Numba 加速的排名计算

    Args:
        df: Polars DataFrame
        axis: 计算轴（1=按行，0=按列）

    Returns:
        计算后的 Polars DataFrame
    """
    numeric_cols = [c for c in df.columns if c not in ['date', 'stock_code']]
    if not numeric_cols:
        return df

    arr = df.select(numeric_cols).to_numpy()

    if axis == 1:
        result_arr = _numba_rank_2d(arr)
    else:
        # 转置后计算
        result_arr = _numba_rank_2d(arr.T).T

    result_df = pl.DataFrame(result_arr, schema=numeric_cols)

    if 'date' in df.columns:
        result_df = result_df.with_columns([df['date']])

    return result_df


_SKIP_PANEL = frozenset({"date", "stock_code"})


def panel_pandas_bridge(x: "pl.DataFrame", fn, *args, **kwargs) -> "pl.DataFrame":
    """Polars 宽表 ↔ pandas 桥接（parity 与 pandas_numpy 对齐）。"""
    cols = [c for c in x.columns if c not in _SKIP_PANEL]
    if not cols:
        return x
    pdf = x.select(cols).to_pandas()
    out = fn(pdf, *args, **kwargs)
    return x.with_columns([
        pl.Series(name=c, values=np.asarray(out[c], dtype=np.float64)) for c in cols
    ])
