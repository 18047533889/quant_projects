"""
输入适配器抽象基类：定义从外部数据源加载并组装 InputBundle 的接口。

所有生产环境的输入适配器都应继承 InputAdapter 并实现以下方法：
- load_alpha / load_market：必填
- load_industry / load_style / load_benchmark / load_prev_positions / load_tradable_flags：可选
- build_bundle：组装所有的 load_* 结果为 InputBundle

子类示例：MockInputAdapter（测试用）、RealInputAdapter（生产用，待实现）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

import pandas as pd

from ..core.contracts import InputBundle, OptimizationContext


class InputAdapter(ABC):
    """输入适配器抽象基类。

    每个 load_* 方法负责从特定数据源加载一种数据矩阵，
    build_bundle 负责将所有矩阵组装成 InputBundle。
    """

    @abstractmethod
    def load_alpha(self, *args, **kwargs) -> pd.DataFrame:
        """加载 alpha 信号矩阵。

        Returns:
            DataFrame，行=日期(DatetimeIndex)，列=资产代码，值为 alpha 值。
        """
        raise NotImplementedError

    @abstractmethod
    def load_market(self, *args, **kwargs) -> pd.DataFrame:
        """加载行情价格矩阵。

        Returns:
            DataFrame，行=日期(DatetimeIndex)，列=资产代码，值为收盘价。
        """
        raise NotImplementedError

    @abstractmethod
    def load_industry(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        """加载行业分类标签。

        Returns:
            DataFrame 或 None，行=日期，列=资产代码，值为行业标签字符串。
        """
        raise NotImplementedError

    @abstractmethod
    def load_style(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        """加载风格因子暴露矩阵。

        Returns:
            DataFrame 或 None，MultiIndex columns（风格名, 资产代码）。
        """
        raise NotImplementedError

    @abstractmethod
    def load_benchmark(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        """加载基准权重矩阵。

        Returns:
            DataFrame 或 None，行=日期，列=资产代码，值为基准权重。
        """
        raise NotImplementedError

    @abstractmethod
    def load_prev_positions(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        """加载上期持仓权重矩阵。

        Returns:
            DataFrame 或 None，行=日期，列=资产代码，值为上期持仓权重。
        """
        raise NotImplementedError

    @abstractmethod
    def load_tradable_flags(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        """加载可交易标志矩阵。

        Returns:
            DataFrame 或 None，行=日期，列=资产代码，True 表示可交易。
        """
        raise NotImplementedError

    def build_bundle(self, *args, **kwargs) -> InputBundle:
        """组装所有 loaded 数据为 InputBundle。

        子类必须重写此方法，在其中调用各个 load_* 方法并填充 InputBundle。
        """
        raise NotImplementedError
