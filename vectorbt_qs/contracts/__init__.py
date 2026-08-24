"""vectorbt_qs contracts package — the standard backtest contract layer.

This package defines the content-addressed artifacts a backtest simulation
produces (signal, orders, trades, positions, cash ledger) plus the request/result
boundaries, execution policy, and the Accounting Invariant Gate.

Design rules:
- pure stdlib + dataclass (no new runtime dependencies);
- every artifact is frozen and content-hash-addressed;
- metric authority deliberately lives in quant_evaluator (QE); this layer only
  simulates.
"""

from .signal import (
    SignalArtifact,
    SignalSeriesArtifact,
    content_hash_of,
)
from .execution import (
    CapacityPolicy,
    CostParams,
    ExecutionPolicy,
)
from .orders import OrderArtifact, OrdersArtifact
from .trades import TradeArtifact, TradesArtifact
from .positions import PositionArtifact, PositionsArtifact
from .ledger import CashLedgerArtifact, CashLedgerRow
from .backtest import BacktestArtifact, BacktestRequest

__all__ = [
    "SignalArtifact",
    "SignalSeriesArtifact",
    "content_hash_of",
    "CapacityPolicy",
    "CostParams",
    "ExecutionPolicy",
    "OrderArtifact",
    "OrdersArtifact",
    "TradeArtifact",
    "TradesArtifact",
    "PositionArtifact",
    "PositionsArtifact",
    "CashLedgerArtifact",
    "CashLedgerRow",
    "BacktestArtifact",
    "BacktestRequest",
]
