# -*- coding: utf-8 -*-
"""Financial point-in-time row-bundle loading and period selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import pandas as pd

from pit_contract import (
    AvailabilityPrecision,
    PITColumns,
    select_visible_row_bundles,
)

PeriodSelector = Literal["latest_visible_period", "annual_only", "quarterly_only"]


@dataclass(frozen=True)
class FinancialFieldContract:
    table: str
    field: str
    period_selector: PeriodSelector = "latest_visible_period"
    pit_required: bool = True


FUNDAMENTAL_FIELD_CONTRACTS: dict[str, FinancialFieldContract] = {}


def register_fundamental_field(
    name: str,
    *,
    table: str,
    field: str,
    period_selector: PeriodSelector = "latest_visible_period",
    pit_required: bool = True,
) -> FinancialFieldContract:
    if period_selector not in {"latest_visible_period", "annual_only", "quarterly_only"}:
        raise ValueError(f"unknown period_selector {period_selector!r}")
    contract = FinancialFieldContract(table, field, period_selector, pit_required)
    existing = FUNDAMENTAL_FIELD_CONTRACTS.get(str(name))
    if existing is not None and existing != contract:
        raise ValueError(f"fundamental field {name!r} already has a different contract")
    FUNDAMENTAL_FIELD_CONTRACTS[str(name)] = contract
    return contract


# R24-058/059: market-scoped financial field contract registry — keyed by
# ``(market, field_id)``, never by bare field name alone (a bare name lets A
# revenue and US revenue share a contract by accident).
MARKET_FUNDAMENTAL_FIELD_CONTRACTS: dict[tuple[str, str], FinancialFieldContract] = {}


def get_fundamental_contract(market: str, field_name: str) -> FinancialFieldContract | None:
    """R24-058/059: market-scoped contract lookup (``market + field_id`` key)."""
    return MARKET_FUNDAMENTAL_FIELD_CONTRACTS.get((str(market).strip().lower(), str(field_name)))


def _register_catalog_fundamental_fields() -> None:
    """Populate the period-selection contracts from the authoritative catalog.

    Every ``financial_pit`` feature field (StockBalance/StockIncome/StockCashFlow/
    StockIndicator statement columns) must declare how its visible report period
    is selected (audit §2.10).  Balance and YTD-cumulative flow fields read the
    latest visible statement by default; a field that cannot be read that way is
    rejected at registration rather than silently period-ambiguous.

    R24-058/059: BOTH ``ASHARE_FIELD_SPECS`` and ``US_FIELD_SPECS`` are
    registered, keyed by ``(market, field_id)`` — A revenue and US revenue never
    share a contract.  The legacy bare-name dict is kept only for backward
    compatibility and is NOT the authority for market-scoped reads.

    Shared structural columns (``pub_date``/``update_time``/
    ``report_period_end_date`` are deliberately reused by name across the four
    statements) are ``mining_allowed=False`` metadata, not period-selected
    features, and are exempt from period contracts.
    """
    from fields import ASHARE_FIELD_SPECS
    from fields.catalog_us import US_FIELD_SPECS
    from .logical_tables import logical_table_contract

    for market, specs in (("ashare", ASHARE_FIELD_SPECS), ("us", US_FIELD_SPECS)):
        for spec in specs:
            if not spec.mining_allowed:
                continue
            table = str(spec.table)
            try:
                contract = logical_table_contract(table, market=market)
            except KeyError:
                # Table not registered for this market: no period contract.
                continue
            if contract is None or contract.join_policy != "financial_pit":
                continue
            selector = str(spec.metadata.get("period_selector", "latest_visible_period"))
            if selector not in {"latest_visible_period", "annual_only", "quarterly_only"}:
                continue
            fc = FinancialFieldContract(table, spec.source_name, selector, spec.strict_pit_allowed)  # type: ignore[arg-type]
            MARKET_FUNDAMENTAL_FIELD_CONTRACTS[(market, str(spec.name))] = fc
            # Legacy bare-name registry (A-share only) kept for back-compat.
            if market == "ashare":
                register_fundamental_field(
                    spec.name,
                    table=table,
                    field=spec.source_name,
                    period_selector=selector,  # type: ignore[arg-type]
                    pit_required=spec.strict_pit_allowed,
                )


_register_catalog_fundamental_fields()


def load_financial_row_bundle(
    decisions: pd.DataFrame,
    events: pd.DataFrame,
    fields: Iterable[str],
    *,
    selector: PeriodSelector = "latest_visible_period",
    decision_time: str = "decision_timestamp",
    columns: PITColumns = PITColumns(),
    production: bool = False,
    market_calendar: pd.DatetimeIndex | None = None,
    market_timezone: str = "UTC",
    precision: AvailabilityPrecision | str | None = None,
    timeframe_column: str | None = None,
    fiscal_quarter_column: str | None = None,
) -> pd.DataFrame:
    """Select all requested fields from one visible statement row per decision.

    This intentionally selects rows before projecting fields.  Loading fields
    independently can combine values from different periods or revisions.

    R24-052/053: the PRODUCTION path threads the execution context
    (``production`` / ``market_calendar`` / ``market_timezone`` / ``precision``)
    from the top level into the temporal selector — helpers never default a
    production-capable call to research semantics.

    R24-054/055: the output retains the ORIGINAL ``knowledge_at`` /
    ``market_visible_at`` alongside the shifted ``available_at`` so same-day
    proof, age and revision latency stay reconstructible.

    R24-056/057: the requested ``selector`` may only keep or tighten each
    field's registered period contract (explicit lattice), never loosen it.
    """
    field_names = tuple(dict.fromkeys(str(field) for field in fields))
    if not field_names:
        raise ValueError("financial row bundle requires at least one field")
    missing = sorted(set(field_names) - set(events.columns))
    if missing:
        raise ValueError(f"financial events missing bundle fields: {missing}")
    # R24-056/057: explicit selector lattice — the caller may only keep or
    # tighten a field's registered contract (annual_only ⊂ latest_visible_period).
    from pit_contract import selector_compatible

    for name in field_names:
        contract = FUNDAMENTAL_FIELD_CONTRACTS.get(name)
        if contract is None:
            continue
        required = contract.period_selector
        if not selector_compatible(required, selector):
            raise ValueError(
                f"financial field {name!r} requires period_selector "
                f"{required!r}; requested {selector!r} loosens the contract "
                "(R24-056/057 — only keep-or-tighten is allowed)"
            )
    selected = select_visible_row_bundles(
        decisions,
        events,
        selector=selector,
        decision_time=decision_time,
        columns=columns,
        production=production,
        market_calendar=market_calendar,
        market_timezone=market_timezone,
        precision=precision,
        timeframe_column=timeframe_column,
        fiscal_quarter_column=fiscal_quarter_column,
    )
    keep = list(decisions.columns)
    for name in (
        columns.period_end, columns.available_at, columns.revision_id,
        # R24-054/055: keep the original knowledge time and market-visible time.
        "knowledge_at", "market_visible_at", "decision_at", *field_names,
    ):
        if name in selected.columns and name not in keep:
            keep.append(name)
    return selected[keep]
