# -*- coding: utf-8 -*-
"""
算子基类与 ``@register_operator`` 装饰器。

约定
----
- **输入/输出**：``calculate`` 接收宽表 ``pd.DataFrame``（index=时间, columns=标的），
  与 ``backend/cleaned_bridge`` 的 panel 格式一致；返回同形 DataFrame 或可对齐的 Series。
- **元数据**：``OperatorMetadata.description`` 会写入 catalog，供文档与 LLM 提示词引用。
- **注册**：模块 import 时装饰器把实例挂到 ``OperatorRegistry``；勿在 api 层重复实现。

子类选型
--------
- ``SeriesOperator``：多参数序列算子（滚动、双序列相关等）；
- ``TransformOperator``：单输入单输出变换（rank、abs）；
- ``TwoVarOperator``：固定两列输入；
- ``ScalarOperator``：输出标量（较少用于 panel 路径）。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import pandas as pd


@dataclass
class OperatorMetadata:
    """算子 catalog 元数据字段。

    供 registry catalog、文档生成与 LLM 提示词引用。
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
    """所有 pandas_numpy runtime 算子的抽象基类。

    子类须实现 ``calculate``，在宽表 panel 上执行计算。
    """


    metadata: OperatorMetadata

    @abstractmethod
    def calculate(self, *args, **kwargs) -> pd.DataFrame:
        """在 panel 上执行计算。

        参数:
            *args: 位置参数（通常为宽表 DataFrame 及窗口/标量参数）。
            **kwargs: 关键字参数（如 ``window``、``periods`` 等）。

        返回:
            与输入同形的 ``pd.DataFrame`` 结果 panel。
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
    """通用序列算子基类：规范化 float 窗口参数后调用 ``_calculate_series``。"""

    def calculate(self, *args, **kwargs) -> pd.DataFrame:
        """规范化参数后委托 ``_calculate_series`` 执行。

        参数:
            *args: 输入序列及标量参数；整型窗口的 ``float`` 会转为 ``int``。
            **kwargs: 额外关键字参数。

        返回:
            计算结果 ``pd.DataFrame``。
        """
        processed_args = []
        for a in args:
            if isinstance(a, float) and a == int(a):
                processed_args.append(int(a))
            elif isinstance(a, pd.DataFrame):
                processed_args.append(a)
            else:
                processed_args.append(a)
        return self._calculate_series(*processed_args, **kwargs)

    @abstractmethod
    def _calculate_series(self, *args, **kwargs) -> pd.DataFrame:
        """子类实现具体 rolling / 时序 / 截面逻辑。

        参数:
            *args: 输入 panel 及算子特定参数。
            **kwargs: 额外关键字参数。

        返回:
            计算结果 ``pd.DataFrame``。
        """
        pass


class ScalarOperator(Operator):
    """标量输出算子基类（panel 路径较少使用）。"""

    def calculate(self, *args, **kwargs) -> Any:
        """委托 ``_calculate_scalar`` 计算标量结果。

        参数:
            *args: 输入参数。
            **kwargs: 额外关键字参数。

        返回:
            标量或标量序列。
        """
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


class TransformOperator(Operator):
    """单序列进、单序列出变换算子（如 ``abs``、``log``）。"""

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """子类实现单输入变换逻辑。

        参数:
            x: 输入宽表 panel。
            **kwargs: 额外关键字参数。

        返回:
            变换后的 ``pd.DataFrame``。
        """
        raise NotImplementedError

    def calculate(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """对单输入 panel 执行变换。

        参数:
            x: 输入宽表 panel。
            **kwargs: 额外关键字参数。

        返回:
            变换后的 ``pd.DataFrame``。
        """
        return self._calculate_series(x, **kwargs)


class TwoVarOperator(Operator):
    """双序列算子基类（如 ``ts_corr(x, y, d)``）。"""

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """子类实现双输入序列逻辑。

        参数:
            x: 第一个输入宽表 panel。
            y: 第二个输入宽表 panel。
            **kwargs: 额外关键字参数。

        返回:
            计算结果 ``pd.DataFrame``。
        """
        raise NotImplementedError

    def calculate(self, x: pd.DataFrame, y: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """对双输入 panel 执行计算。

        参数:
            x: 第一个输入宽表 panel。
            y: 第二个输入宽表 panel。
            **kwargs: 额外关键字参数。

        返回:
            计算结果 ``pd.DataFrame``。
        """
        return self._calculate_series(x, y, **kwargs)


def register_operator(
    name: str = None,
    category: str = "general",
    business_category: str = "",
    canonical: str = "",
    source: str = "",
    backend: str | None = None,
    status: str = "implemented",
):
    """类装饰器：实例化算子并注册到 ``OperatorRegistry``。

    参数:
        name: DSL 注册名（可与 ``canonical`` 不同）。
        category: 算子分类（如 ``time_series``）。
        business_category: 业务分类标签。
        canonical: registry 主键，默认取 ``name`` 或类名。
        source: 溯源标记，写入 catalog。
        backend: 显式 backend；未指定时按类名推断（含 ``Polars`` → ``polars``）。
        status: 生命周期状态，默认 ``implemented``。

    返回:
        装饰器函数，包装目标算子类并完成注册。
    """
    def decorator(cls):
        instance = cls()
        if name:
            instance.metadata.name = name
        if category:
            instance.metadata.category = category
        if business_category:
            instance.metadata.business_category = business_category
        if backend is not None:
            effective_backend = backend
            backend_explicit = True
        elif "Polars" in cls.__name__:
            effective_backend = "polars"
            backend_explicit = False
        else:
            effective_backend = "pandas_numpy"
            backend_explicit = False
        canon = canonical or (name if name else cls.__name__)
        from cleaned_operators.registry import OperatorRegistry
        name_aliases = [name] if name and name != canon else None
        OperatorRegistry.register(
            instance,
            canonical=canon,
            backend=effective_backend,
            source=source or "factor_dsl_np",
            aliases=name_aliases,
            status=status,
            backend_explicit=backend_explicit,
        )
        return cls
    return decorator
