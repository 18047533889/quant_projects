"""Backtest request / artifact contracts for the vectorbt_qs layer.

``BacktestRequest`` is the immutable specification of a simulation; it carries
all policies, references, and the requested window.  ``BacktestArtifact`` is
the immutable, content-addressed result: it references the input signal and
holds every derived ledger (positions, orders, trades, cash/NAV, returns,
turnover, cost drag, unfilled, execution shortfall).

This module is pure stdlib + dataclass; it introduces no runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .execution import CapacityPolicy, ExecutionPolicy
from .ledger import CashLedgerArtifact
from .orders import OrderArtifact, OrdersArtifact
from .positions import PositionArtifact, PositionsArtifact
from .signal import SignalArtifact, content_hash_of
from .trades import TradeArtifact, TradesArtifact


@dataclass(frozen=True)
class BacktestRequest:
    """Frozen description of one backtest to run.

    Attributes
    ----------
    strategy_ref:
        Reference to the strategy definition.
    snapshot_ref:
        Reference to the data snapshot (market data).
    universe_ref:
        Reference to the tradable universe.
    decision_time_policy:
        Policy for signal generation time.
    signal_available_policy:
        Policy for when a signal becomes tradable.
    execution_policy:
        How orders fill (open/vwap/close, lot, fees, slippage).
    cost_policy:
        Fee schedule for the simulation.
    capacity_policy:
        Optional per-order participation-rate cap.
    start / end:
        ISO bounds of the simulation window.
    init_cash:
        Starting cash.
    """

    strategy_ref: str
    snapshot_ref: str
    universe_ref: str
    decision_time_policy: str = "close"
    signal_available_policy: str = "next_open"
    execution_policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    cost_policy: str = "ashare_accurate"
    capacity_policy: Optional[CapacityPolicy] = None
    start: str = ""
    end: str = ""
    init_cash: float = 1_000_000.0
    content_hash: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.strategy_ref or not self.snapshot_ref or not self.universe_ref:
            raise ValueError("strategy_ref / snapshot_ref / universe_ref required")
        if self.init_cash <= 0:
            raise ValueError("init_cash must be positive")

    def raw_payload(self) -> Dict[str, Any]:
        return {
            "strategy_ref": self.strategy_ref,
            "snapshot_ref": self.snapshot_ref,
            "universe_ref": self.universe_ref,
            "decision_time_policy": self.decision_time_policy,
            "signal_available_policy": self.signal_available_policy,
            "execution_policy": self.execution_policy.to_dict(),
            "cost_policy": self.cost_policy,
            "capacity_policy": None
            if self.capacity_policy is None
            else {"max_participation_rate": self.capacity_policy.max_participation_rate},
            "start": self.start,
            "end": self.end,
            "init_cash": self.init_cash,
        }

    def recompute_hash(self) -> str:
        return content_hash_of(self.raw_payload())

    def to_dict(self) -> Dict[str, Any]:
        return {**self.raw_payload(), "content_hash": self.content_hash or self.recompute_hash()}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BacktestRequest":
        cap = payload.get("capacity_policy")
        obj = cls(
            strategy_ref=payload["strategy_ref"],
            snapshot_ref=payload["snapshot_ref"],
            universe_ref=payload["universe_ref"],
            decision_time_policy=payload.get("decision_time_policy", "close"),
            signal_available_policy=payload.get("signal_available_policy", "next_open"),
            execution_policy=ExecutionPolicy.from_dict(payload.get("execution_policy", {})),
            cost_policy=payload.get("cost_policy", "ashare_accurate"),
            capacity_policy=None if not cap else CapacityPolicy(cap.get("max_participation_rate", 1.0)),
            start=payload.get("start", ""),
            end=payload.get("end", ""),
            init_cash=float(payload.get("init_cash", 1_000_000.0)),
            content_hash=payload.get("content_hash"),
        )
        expected = obj.recompute_hash()
        supplied = payload.get("content_hash")
        if supplied and supplied != expected:
            raise ValueError("content_hash mismatch for BacktestRequest")
        return obj


@dataclass(frozen=True)
class BacktestArtifact:
    """The standard, content-addressed output of one backtest simulation.

    It records the simulator identity/version and every derived ledger, plus a
    content hash over the whole result.  Metric authority deliberately lives in
    the platform's quant_evaluator (QE); this artifact carries simulation-only
    ledgers and lets QE be the single authority for any computed metric.
    """

    backtest_id: str
    signal_ref: str
    strategy_ref: str
    snapshot_ref: str
    universe_ref: str
    simulator_version: str = "0.1.0"
    positions: Optional[PositionsArtifact] = None
    orders: Optional[OrdersArtifact] = None
    trades: Optional[TradesArtifact] = None
    cash_ledger: Optional[CashLedgerArtifact] = None
    nav: Optional[tuple] = None
    gross_return: Optional[tuple] = None
    net_return: Optional[tuple] = None
    turnover: Optional[tuple] = None
    cost_drag: Optional[tuple] = None
    unfilled: Optional[tuple] = None
    execution_shortfall: Optional[tuple] = None
    content_hash: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.backtest_id:
            raise ValueError("backtest_id required")
        if not self.signal_ref or not self.snapshot_ref or not self.universe_ref:
            raise ValueError("signal_ref / snapshot_ref / universe_ref required")

    def representation_payload(self) -> Dict[str, Any]:
        """Return a dict suitable for recomputing the content hash (minus the hash itself)."""
        return {
            "backtest_id": self.backtest_id,
            "signal_ref": self.signal_ref,
            "strategy_ref": self.strategy_ref,
            "snapshot_ref": self.snapshot_ref,
            "universe_ref": self.universe_ref,
            "simulator_version": self.simulator_version,
            "positions": None if self.positions is None else self.positions.to_dict(),
            "orders": None if self.orders is None else self.orders.to_dict(),
            "trades": None if self.trades is None else self.trades.to_dict(),
            "cash_ledger": None if self.cash_ledger is None else self.cash_ledger.to_dict(),
            "nav": None if self.nav is None else list(self.nav),
            "gross_return": None if self.gross_return is None else list(self.gross_return),
            "net_return": None if self.net_return is None else list(self.net_return),
            "turnover": None if self.turnover is None else list(self.turnover),
            "cost_drag": None if self.cost_drag is None else list(self.cost_drag),
            "unfilled": None if self.unfilled is None else list(self.unfilled),
            "execution_shortfall": (
                None if self.execution_shortfall is None else list(self.execution_shortfall)
            ),
        }

    def recompute_hash(self) -> str:
        return content_hash_of(self.representation_payload())

    def to_dict(self) -> Dict[str, Any]:
        return {**self.representation_payload(), "content_hash": self.content_hash or self.recompute_hash()}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BacktestArtifact":
        def _to_list(x):
            return None if x is None else tuple(x)

        def _orders(x):
            if x is None:
                return None
            return OrdersArtifact(tuple(OrderArtifact.from_dict(o) for o in x["orders"]))

        def _trades(x):
            if x is None:
                return None
            return TradesArtifact(tuple(TradeArtifact.from_dict(t) for t in x["trades"]))

        def _positions(x):
            if x is None:
                return None
            return PositionsArtifact(tuple(PositionArtifact.from_dict(p) for p in x["positions"]))

        def _ledger(x):
            if x is None:
                return None
            return CashLedgerArtifact.from_dict(x)

        obj = cls(
            backtest_id=payload["backtest_id"],
            signal_ref=payload["signal_ref"],
            strategy_ref=payload["strategy_ref"],
            snapshot_ref=payload["snapshot_ref"],
            universe_ref=payload["universe_ref"],
            simulator_version=payload.get("simulator_version", "0.1.0"),
            positions=_positions(payload.get("positions")),
            orders=_orders(payload.get("orders")),
            trades=_trades(payload.get("trades")),
            cash_ledger=_ledger(payload.get("cash_ledger")),
            nav=_to_list(payload.get("nav")),
            gross_return=_to_list(payload.get("gross_return")),
            net_return=_to_list(payload.get("net_return")),
            turnover=_to_list(payload.get("turnover")),
            cost_drag=_to_list(payload.get("cost_drag")),
            unfilled=_to_list(payload.get("unfilled")),
            execution_shortfall=_to_list(payload.get("execution_shortfall")),
            content_hash=payload.get("content_hash"),
        )
        expected = obj.recompute_hash()
        supplied = payload.get("content_hash")
        if supplied and supplied != expected:
            raise ValueError(
                f"content_hash mismatch for BacktestArtifact {obj.backtest_id}"
            )
        return obj


__all__ = ["BacktestRequest", "BacktestArtifact"]
