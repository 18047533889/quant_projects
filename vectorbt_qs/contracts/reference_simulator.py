"""ReferenceLedgerSimulator — a deliberately simple, slow-but-correct A-share
ledger simulator.

This is the THIRD reference engine in the vectorbt_qs stack, alongside the
Python and Numba mvp planners.  It is intentionally NOT optimized — it favors
clarity and auditability so it can serve as an oracle in differential testing
against the fast engines.

Scope (hard limits):
- at most 20 assets and 100 days per simulation;
- A-share long-only, 100-share board lots;
- signals map to the NEXT trading day's open price (execution_lag = 1);
- orders execute at next-day open with a simplified fee schedule:
  buy commission, sell commission + stamp tax, optional slippage;
- no corporate actions, no suspensions, no limit-up/down, no capacity cap —
  those are mvp-feature territory, not the reference's job.

The simulator walks the ledger day-by-day and returns a cash / shares / NAV
series plus an order list, all as plain stdlib structures.  It accepts any
signal-like object exposing ``timestamps`` / ``position_targets`` /
``universe_ids`` and any price-like object exposing ``open`` and ``close``
(DatetimeIndex-keyed) OR plain aligned lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class ReferenceCostParams:
    commission: float = 0.00025
    stamp_tax: float = 0.0005
    minimum_commission: float = 5.0


@dataclass
class ReferenceLedgerResult:
    """Output of the reference simulator.

    Field layout mirrors the Accounting Invariant Gate's duck-typed contract:
    ``timestamps`` / ``asset_ids`` / ``cash`` / ``shares`` / ``mark`` / ``nav``
    / ``orders``.
    """

    timestamps: List[str]
    asset_ids: List[str]
    cash: List[float]
    shares: List[List[float]]
    mark: List[List[float]]
    nav: List[float]
    orders: List[Dict[str, Any]]

    def to_artifact_dict(self) -> Dict[str, Any]:
        return {
            "timestamps": self.timestamps,
            "asset_ids": self.asset_ids,
            "cash": self.cash,
            "shares": self.shares,
            "mark": self.mark,
            "nav": self.nav,
            "orders": self.orders,
        }


class ReferenceLedgerSimulator:
    """A minimal, auditable A-share ledger simulator (reference engine)."""

    def __init__(
        self,
        *,
        init_cash: float = 1_000_000.0,
        lot_size: int = 100,
        costs: Optional[ReferenceCostParams] = None,
    ) -> None:
        if init_cash <= 0:
            raise ValueError("init_cash must be positive")
        if lot_size <= 0:
            raise ValueError("lot_size must be positive")
        self.init_cash = float(init_cash)
        self.lot_size = int(lot_size)
        self.costs = costs or ReferenceCostParams()

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def simulate(self, signal: Any, prices: Any) -> ReferenceLedgerResult:
        """Run the ledger from ``signal`` against ``prices``.

        ``signal`` must expose:
          - ``position_targets``: ``[n_days, n_assets]`` float weights
          - ``universe_ids``: ordered asset list
          - ``timestamps``: decision day per row (len == n_days)

        ``prices`` must expose ``open`` and ``close`` frames (DatetimeIndex x
        assets) OR be a dict with aligned ``open`` / ``close`` / ``dates``.
        """
        targets = self._coerce(signal, "position_targets")
        asset_ids = list(self._coerce(signal, "universe_ids"))
        signal_dates = list(self._coerce(signal, "timestamps"))
        if len(targets) != len(signal_dates):
            raise ValueError(
                f"target rows {len(targets)} != signal timestamps {len(signal_dates)}"
            )
        n_assets = len(asset_ids)
        for row in targets:
            if len(row) != n_assets:
                raise ValueError("each target row must match universe width")
        if n_assets == 0:
            raise ValueError("universe cannot be empty")
        if n_assets > 20:
            raise ValueError("reference simulator supports at most 20 assets")

        open_price, close_price, dates = self._prices(prices, asset_ids)

        # Build trading-day timeline (dates = open/close index rows).
        if len(dates) > 100:
            raise ValueError("reference simulator supports at most 100 days")

        # Map each signal row to its execution day: next trading day's open.
        exec_idx: List[int] = []  # parallel to target rows
        for sig_date in signal_dates:
            pos = self._next_trading_day(sig_date, dates)
            exec_idx.append(pos)

        cash = float(self.init_cash)
        shares = [0.0] * n_assets
        order_log: List[Dict[str, Any]] = []

        out_timestamps: List[str] = []
        out_cash: List[float] = []
        out_shares: List[List[float]] = []
        out_mark: List[List[float]] = []
        out_nav: List[float] = []

        # Position target changes only matter on the days we act.  Rebalance
        # happens on the exec date's open.  Between execs we just mark to close.
        day_idx = 0
        for d in range(len(dates)):
            date = dates[d]
            # Determine equity for target sizing using the exec-day OPEN.
            if d in exec_idx:
                # Desired shares from target weight on this day's open.
                equity = cash + sum(
                    shares[j] * open_price[d][j] for j in range(n_assets)
                )
                desired = [0.0] * n_assets
                for j in range(n_assets):
                    w = float(targets[exec_idx.index(d)][j])
                    desired[j] = float(
                        (equity * w) // open_price[d][j] // self.lot_size
                    ) * self.lot_size
                # Sell first, then buy.
                for j in range(n_assets):
                    reduction = shares[j] - desired[j]
                    if reduction > 0:
                        sell = reduction  # full exit or reduction
                        notional = sell * open_price[d][j]
                        fee = notional * self.costs.commission + max(
                            0.0,
                            self.costs.minimum_commission
                            - notional * self.costs.commission,
                        )
                        sell_fee = fee + notional * self.costs.stamp_tax
                        cash += notional - sell_fee
                        shares[j] -= sell
                        order_log.append(
                            {
                                "timestamp": str(date),
                                "asset_id": asset_ids[j],
                                "side": "sell",
                                "requested_size": sell,
                                "filled_size": sell,
                                "execution_price": open_price[d][j],
                                "fees": sell_fee,
                                "status": "filled",
                            }
                        )
                for j in range(n_assets):
                    delta = desired[j] - shares[j]
                    if delta > 0:
                        buy = int((delta // self.lot_size) * self.lot_size)
                        notional = buy * open_price[d][j]
                        fee = notional * self.costs.commission + max(
                            0.0,
                            self.costs.minimum_commission
                            - notional * self.costs.commission,
                        )
                        if cash >= notional + fee:
                            cash -= notional + fee
                            shares[j] += buy
                            order_log.append(
                                {
                                    "timestamp": str(date),
                                    "asset_id": asset_ids[j],
                                    "side": "buy",
                                    "requested_size": buy,
                                    "filled_size": buy,
                                    "execution_price": open_price[d][j],
                                    "fees": fee,
                                    "status": "filled",
                                }
                            )

            mark = close_price[d]
            nav = cash + sum(shares[j] * mark[j] for j in range(n_assets))
            out_timestamps.append(str(date))
            out_cash.append(cash)
            out_shares.append(list(shares))
            out_mark.append(list(mark))
            out_nav.append(nav)
            day_idx += 1

        return ReferenceLedgerResult(
            timestamps=out_timestamps,
            asset_ids=asset_ids,
            cash=out_cash,
            shares=out_shares,
            mark=out_mark,
            nav=out_nav,
            orders=order_log,
        )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce(obj: Any, name: str) -> List[Any]:
        if isinstance(obj, dict):
            value = obj.get(name)
        else:
            value = getattr(obj, name, None)
        if value is None:
            raise ValueError(f"signal missing '{name}'")
        if isinstance(value, tuple):
            return list(value)
        return list(value)

    def _prices(
        self, prices: Any, asset_ids: List[str]
    ) -> "tuple[List[List[float]], List[List[float]], List[str]]":
        """Extract open/close matrices + date list.

        Accepts either a pandas-like object with ``open``/``close`` DataFrames
        (index = trading days, columns = assets) or a dict with aligned
        ``open`` / ``close`` / ``timestamps`` lists.
        """
        open_price: List[List[float]] = []
        close_price: List[List[float]] = []
        dates: List[str] = []

        if isinstance(prices, dict):
            opens = prices["open"]
            closes = prices["close"]
            dates = [str(t) for t in prices["timestamps"]]
            for i in range(len(dates)):
                row_o = [float(x) for x in opens[i]]
                row_c = [float(x) for x in closes[i]]
                if len(row_o) != len(asset_ids) or len(row_c) != len(asset_ids):
                    raise ValueError("price row width must match asset_ids")
                open_price.append(row_o)
                close_price.append(row_c)
            return open_price, close_price, dates

        # pandas-like: open/close DataFrames.
        try:
            open_df = prices.open
            close_df = prices.close
        except AttributeError as exc:
            raise ReferenceError(
                "prices must expose open/close (DataFrames) or be a dict "
                "with open/close/timestamps"
            ) from exc
        idx = list(open_df.index)
        dates = [str(d) for d in idx]
        for i, date in enumerate(idx):
            row_o = [float(open_df[col].iloc[i]) for col in asset_ids]
            row_c = [float(close_df[col].iloc[i]) for col in asset_ids]
            open_price.append(row_o)
            close_price.append(row_c)
        return open_price, close_price, dates

    @staticmethod
    def _next_trading_day(sig_date: str, dates: List[str]) -> int:
        """Return index of the first trading day strictly after ``sig_date``."""
        if not dates:
            raise ValueError("no trading days")
        for i, date in enumerate(dates):
            if date > sig_date:
                return i
        raise ValueError(f"no trading day after {sig_date}")


__all__ = ["ReferenceLedgerSimulator", "ReferenceLedgerResult", "ReferenceCostParams"]
