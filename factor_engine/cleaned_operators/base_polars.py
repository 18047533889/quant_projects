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
    """Polars 算子 catalog 元数据字段。

    与 ``base.OperatorMetadata`` 字段一致，供 polars backend 注册使用。
    """
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
    """所有 Polars runtime 算子的抽象基类。

    子类须实现 ``calculate``，在 Polars 宽表 panel 上执行计算。
    """

    metadata: OperatorMetadata

    @abstractmethod
    def calculate(self, *args, **kwargs) -> pl.DataFrame:
        """在 Polars panel 上执行计算。

        参数:
            *args: 位置参数（宽表及算子特定参数）。
            **kwargs: 关键字参数。

        返回:
            与输入同形的 ``pl.DataFrame`` 结果。
        """
        pass

    def validate_params(self, *args, **kwargs) -> bool:
        """参数校验钩子，子类可覆盖。

        参数:
            *args: 待校验的位置参数。
            **kwargs: 待校验的关键字参数。

        返回:
            参数合法返回 ``True``；默认放行。
        """
        return True

    def __repr__(self):
        return f"<Operator: {self.metadata.name}>"

    def __str__(self):
        return f"{self.metadata.name}: {self.metadata.description}"


class SeriesOperator(Operator):
    """Polars 序列算子基类：规范化窗口参数后调用 ``_calculate_series``。"""

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
        """子类实现具体 Polars 序列计算逻辑。

        参数:
            *args: 输入 panel 及算子参数。
            **kwargs: 额外关键字参数。

        返回:
            计算结果 ``pl.DataFrame``。
        """
        pass


class ScalarOperator(Operator):
    """Polars 标量输出算子基类（panel 路径较少使用）。"""

    def calculate(self, *args, **kwargs) -> Any:
        return self._calculate_scalar(*args, **kwargs)

    @abstractmethod
    def _calculate_scalar(self, *args, **kwargs) -> Any:
        """子类实现标量计算逻辑。

        参数:
            *args: 输入参数。
            **kwargs: 额外关键字参数。

        返回:
            标量结果。
        """
        pass


class TransformOperator(Operator):
    """单序列进、单序列出 Polars 变换算子基类。"""

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """子类实现单输入变换逻辑。

        参数:
            x: 输入 Polars 宽表 panel。
            **kwargs: 额外关键字参数。

        返回:
            变换后的 ``pl.DataFrame``。
        """
        raise NotImplementedError

    def calculate(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return self._calculate_series(x, **kwargs)


class TwoVarOperator(Operator):
    """双序列 Polars 算子基类（如 ``ts_corr``）。"""

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """子类实现双输入序列逻辑。

        参数:
            x: 第一个输入 panel。
            y: 第二个输入 panel。
            **kwargs: 额外关键字参数。

        返回:
            计算结果 ``pl.DataFrame``。
        """
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
    """Polars 算子类装饰器：实例化并注册到 ``OperatorRegistry``。

    参数:
        name: DSL 注册名。
        category: 算子分类。
        business_category: 业务分类标签。
        canonical: registry 主键。
        source: 溯源标记。
        backend: 固定为 ``polars``。
        status: 生命周期状态。

    返回:
        装饰器函数。
    """
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
    """Numba 加速的 2D 滚动均值（按列并行）。

    参数:
        arr: 形状 ``(rows, cols)`` 的二维数组。
        window: 滚动窗口长度。

    返回:
        同形状的滚动均值数组。
    """
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
    """Numba 加速的 2D 滚动标准差（按列并行）。

    参数:
        arr: 二维输入数组。
        window: 滚动窗口长度。

    返回:
        同形状的滚动标准差数组。
    """
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
    """Numba 加速的 2D 滚动求和（按列并行）。

    参数:
        arr: 二维输入数组。
        window: 滚动窗口长度。

    返回:
        同形状的滚动求和数组。
    """
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
    """Numba 加速的 2D 滚动最大值（按列并行）。

    参数:
        arr: 二维输入数组。
        window: 滚动窗口长度。

    返回:
        同形状的滚动最大值数组。
    """
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
    """Numba 加速的 2D 滚动最小值（按列并行）。

    参数:
        arr: 二维输入数组。
        window: 滚动窗口长度。

    返回:
        同形状的滚动最小值数组。
    """
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
    """Numba 加速的 2D 截面 Z-Score（按行计算）。

    参数:
        arr: 二维 panel 数组（行=时间，列=标的）。

    返回:
        同形状的 Z-Score 数组。
    """
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
    """Numba 加速的 2D 截面百分位排名（按行，映射到 0-1）。

    参数:
        arr: 二维 panel 数组。

    返回:
        同形状的百分位排名数组。
    """
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
    """对 Polars 宽表应用 Numba 加速的滚动计算。

    参数:
        df: Polars 宽表 panel（跳过 ``date``/``stock_code`` 列）。
        window: 滚动窗口长度。
        func_name: 聚合函数，``mean``/``std``/``sum``/``max``/``min``。

    返回:
        数值列替换为滚动结果后的 ``pl.DataFrame``。

    异常:
        ValueError: 未知的 ``func_name``。
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
    """对 Polars 宽表应用 Numba 加速的 Z-Score 标准化。

    参数:
        df: Polars 宽表 panel。
        axis: 计算轴，``1`` 按行（截面），``0`` 按列（时序）。

    返回:
        Z-Score 标准化后的 ``pl.DataFrame``。
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
    """对 Polars 宽表应用 Numba 加速的百分位排名。

    参数:
        df: Polars 宽表 panel。
        axis: 计算轴，``1`` 按行（截面），``0`` 按列（时序）。

    返回:
        百分位排名（0-1）后的 ``pl.DataFrame``。
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
    """Polars 宽表 ↔ pandas 桥接，保证与 pandas_numpy backend 数值一致。

    参数:
        x: 输入 Polars 宽表 panel。
        fn: 接收 pandas 宽表并返回同形结果的函数。
        *args: 传给 ``fn`` 的额外位置参数。
        **kwargs: 传给 ``fn`` 的额外关键字参数。

    返回:
        数值列替换为 ``fn`` 结果后的 Polars DataFrame。
    """
    cols = [c for c in x.columns if c not in _SKIP_PANEL]
    if not cols:
        return x
    pdf = x.select(cols).to_pandas()
    out = fn(pdf, *args, **kwargs)
    return x.with_columns([
        pl.Series(name=c, values=np.asarray(out[c], dtype=np.float64)) for c in cols
    ])
