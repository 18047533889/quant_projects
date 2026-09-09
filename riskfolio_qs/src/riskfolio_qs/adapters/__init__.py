"""适配器层：数据输入/输出适配。

``RealInputAdapter`` 依赖交易所日历。它采用惰性导入，避免只使用
Barra 预计算路径的用户被无关的可选运行时依赖阻断。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .input_adapter import InputAdapter
from .output_adapter import OutputAdapter
from .mock_adapter import MockInputAdapter
from .barra_precomputed_adapter import BarraPrecomputedAdapter
from .data_access_inputs import (
    DataAccessHistoricalMarket,
    DataAccessPortfolioInputs,
    load_historical_market_returns,
    load_portfolio_inputs,
)

if TYPE_CHECKING:
    from .real_adapter import RealInputAdapter

__all__ = [
    "InputAdapter",
    "OutputAdapter",
    "MockInputAdapter",
    "RealInputAdapter",
    "BarraPrecomputedAdapter",
    "DataAccessHistoricalMarket",
    "DataAccessPortfolioInputs",
    "load_historical_market_returns",
    "load_portfolio_inputs",
]


def __getattr__(name: str) -> Any:
    if name == "RealInputAdapter":
        from .real_adapter import RealInputAdapter

        return RealInputAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
