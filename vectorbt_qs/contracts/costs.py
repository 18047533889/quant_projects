"""Effective-dated, research-only A-share transaction cost contracts.

The schedule is an accounting input.  It does not certify that an order was
fillable, or that stock was available to borrow.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Iterable, Mapping, Optional, Sequence


class CostScope(str, Enum):
    GROSS_DIAGNOSTIC = "GROSS_DIAGNOSTIC"
    NET_ASSUMED = "NET_ASSUMED"
    NET_EXECUTABLE = "NET_EXECUTABLE"


@dataclass(frozen=True)
class FixedSlippageProfile:
    profile_id: str = "FIXED_SLIPPAGE_PROFILE_V5"
    bps_per_side: Decimal = Decimal("5")
    includes_half_spread: bool = True
    impact_calibrated: bool = False

    def __post_init__(self) -> None:
        if self.bps_per_side not in {Decimal("2.5"), Decimal("5"), Decimal("10"), Decimal("20")}:
            raise ValueError("V5 fixed-slippage scenario must be 2.5/5/10/20 bps")

    def cost_cny(self, notional_cny: Decimal) -> Decimal:
        return (Decimal(notional_cny) * self.bps_per_side / Decimal("10000")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )


def _date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


@dataclass(frozen=True)
class FeeScheduleEntry:
    market: str
    instrument: str
    account: str
    side: str
    effective_from: date
    effective_to: Optional[date]
    commission_bps: Decimal
    minimum_commission_cny: Decimal
    stamp_tax_bps: Decimal
    transfer_fee_bps: Decimal
    included_components: frozenset[str] = frozenset()
    source_ref: str = ""
    evidence_status: str = "ASSUMED_COST"

    def __post_init__(self) -> None:
        if self.side not in {"buy", "sell"}:
            raise ValueError("side must be buy or sell")
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError("effective_to precedes effective_from")
        for name in ("commission_bps", "minimum_commission_cny", "stamp_tax_bps", "transfer_fee_bps"):
            if Decimal(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class FillForBilling:
    trade_date: date
    market: str
    instrument: str
    account: str
    side: str
    notional_cny: Decimal
    billing_group: Optional[str]


@dataclass(frozen=True)
class FeeBreakdown:
    commission_cny: Decimal
    stamp_tax_cny: Decimal
    transfer_fee_cny: Decimal
    total_cny: Decimal
    status: str
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class FeeSchedule:
    schedule_id: str
    entries: tuple[FeeScheduleEntry, ...]

    def resolve(self, fill: FillForBilling) -> FeeScheduleEntry:
        matches = [e for e in self.entries if (
            e.market == fill.market and e.instrument == fill.instrument
            and e.account == fill.account and e.side == fill.side
            and e.effective_from <= fill.trade_date
            and (e.effective_to is None or fill.trade_date <= e.effective_to)
        )]
        if len(matches) != 1:
            raise ValueError(f"fee schedule must resolve exactly once, got {len(matches)} for {fill}")
        return matches[0]

    def bill(self, fills: Sequence[FillForBilling]) -> FeeBreakdown:
        """Bill fills, applying the commission minimum once per group/side.

        Missing billing groups cannot model minimum commission faithfully and
        therefore retain ``MIN_FEE_NOT_MODELED`` rather than inventing orders.
        """
        penny = Decimal("0.01")
        commission = stamp = transfer = Decimal("0")
        status = "ASSUMED_COST"
        refs: set[str] = set()
        grouped: dict[tuple, list[tuple[FillForBilling, FeeScheduleEntry]]] = {}
        ungrouped: list[tuple[FillForBilling, FeeScheduleEntry]] = []
        for fill in fills:
            if fill.notional_cny < 0:
                raise ValueError("fill notional must be non-negative")
            entry = self.resolve(fill)
            if entry.source_ref:
                refs.add(entry.source_ref)
            pair = (fill, entry)
            if fill.billing_group is None:
                ungrouped.append(pair)
                status = "MIN_FEE_NOT_MODELED"
            else:
                grouped.setdefault((fill.trade_date, fill.account, fill.side, fill.billing_group), []).append(pair)

        for pairs in list(grouped.values()) + [[p] for p in ungrouped]:
            total_notional = sum((p[0].notional_cny for p in pairs), Decimal("0"))
            entry = pairs[0][1]
            if any(p[1] != entry for p in pairs):
                raise ValueError("one billing group resolved to multiple fee rules")
            proportional = total_notional * entry.commission_bps / Decimal("10000")
            group_commission = (Decimal("0") if total_notional == 0 else
                                proportional if pairs[0][0].billing_group is None else
                                max(proportional, entry.minimum_commission_cny))
            commission += group_commission
            if "stamp_tax" not in entry.included_components:
                stamp += total_notional * entry.stamp_tax_bps / Decimal("10000")
            if "transfer_fee" not in entry.included_components:
                transfer += total_notional * entry.transfer_fee_bps / Decimal("10000")
        commission = commission.quantize(penny, rounding=ROUND_HALF_UP)
        stamp = stamp.quantize(penny, rounding=ROUND_HALF_UP)
        transfer = transfer.quantize(penny, rounding=ROUND_HALF_UP)
        return FeeBreakdown(commission, stamp, transfer, commission + stamp + transfer, status, tuple(sorted(refs)))


def ordinary_ashare_research_schedule() -> FeeSchedule:
    """V5 SH/SZ ordinary-A research assumptions, not an account contract."""
    rows = []
    for side in ("buy", "sell"):
        for start, end, stamp, transfer in (
            (date(2015, 8, 1), date(2022, 4, 28), "10" if side == "sell" else "0", "0.2"),
            (date(2022, 4, 29), date(2023, 8, 27), "10" if side == "sell" else "0", "0.1"),
            (date(2023, 8, 28), None, "5" if side == "sell" else "0", "0.1"),
        ):
            rows.append(FeeScheduleEntry(
                market="CN_SH_SZ", instrument="ORDINARY_A", account="RESEARCH",
                side=side, effective_from=start, effective_to=end,
                commission_bps=Decimal("2.5"), minimum_commission_cny=Decimal("5"),
                stamp_tax_bps=Decimal(stamp), transfer_fee_bps=Decimal(transfer),
                included_components=frozenset({"exchange_handling", "regulatory"}),
                source_ref="V5_DELTA_SPEC_20260907#3.2",
            ))
    return FeeSchedule("CN_SH_SZ_A_ORDINARY_RESEARCH_V5", tuple(rows))


__all__ = ["CostScope", "FixedSlippageProfile", "FeeScheduleEntry", "FillForBilling", "FeeBreakdown", "FeeSchedule", "ordinary_ashare_research_schedule"]
