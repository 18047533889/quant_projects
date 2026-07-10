"""数据源抽象：按列名返回与引擎约定一致的 MultiIndex Series（timestamp × instrument）。"""

from abc import ABC, abstractmethod
from typing import Any


class DataSource(ABC):
    """数据源抽象基类，按列名返回 MultiIndex Series。
    
    参数:
        无
    """

    @abstractmethod
    def load_column(self, name: str) -> Any:
        """加载名为 ``name`` 的一列时间序列面板数据。
        
        参数:
            name: 逻辑列名
        
        返回:
            Any
        """
        raise NotImplementedError
