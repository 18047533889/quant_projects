"""将 AlphaGen ``Expression.evaluate`` 路由到 factor_engine。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from alphaprobe.fe_bridge.dsl_convert import expression_to_dsl

if TYPE_CHECKING:
    from shared.alphagen.data.expression import Expression
    from alphaprobe.fe_bridge.stock_data import FactorEngineStockData

_PATCHED = False


def _slice_tensor(full: torch.Tensor, data: FactorEngineStockData, period: slice) -> torch.Tensor:
    assert period.step == 1 or period.step is None
    start = period.start + data.max_backtrack_days
    stop = period.stop + data.max_backtrack_days + data.n_days - 1
    return full[start:stop]


def evaluate_via_factor_engine(
    expr: Expression,
    data: FactorEngineStockData,
    period: slice = slice(0, 1),
) -> torch.Tensor:
    dsl = expression_to_dsl(expr)
    full = data.evaluate_dsl(dsl)
    # 与 qlib StockData 张量长度对齐（factor_engine 可能少 backtrack 天）
    if full.shape[0] < data.data.shape[0]:
        pad = data.data.shape[0] - full.shape[0]
        full = torch.nn.functional.pad(full, (0, 0, 0, pad))
    elif full.shape[0] > data.data.shape[0]:
        full = full[-data.data.shape[0]:]
    return _slice_tensor(full, data, period)


def enable_factor_engine_evaluation() -> None:
    global _PATCHED
    if _PATCHED:
        return

    from alphaprobe.fe_bridge.knowledge_graph import patch_knowledge_graph_for_factor_engine_dsl

    patch_knowledge_graph_for_factor_engine_dsl()

    from shared.alphagen.data.expression import Expression, OutOfDataRangeError
    from alphaprobe.fe_bridge.stock_data import FactorEngineStockData

    original_evaluate = Expression.evaluate

    def patched_evaluate(self, data, period=slice(0, 1)):
        if isinstance(data, FactorEngineStockData):
            if (
                period.start < -data.max_backtrack_days
                or period.stop - 1 > data.max_future_days
            ):
                raise OutOfDataRangeError()
            return evaluate_via_factor_engine(self, data, period)
        return original_evaluate(self, data, period)

    Expression.evaluate = patched_evaluate
    _PATCHED = True
