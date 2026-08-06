# -*- coding: utf-8 -*-
"""Financial point-in-time row-bundle loading and period selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import pandas as pd

from pit_contract import PITColumns, select_visible_row_bundles

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


def _register_catalog_fundamental_fields() -> None:
    """Populate the period-selection contracts from the authoritative catalog.

    Every ``financial_pit`` feature field (StockBalance/StockIncome/StockCashFlow/
    StockIndicator statement columns) must declare how its visible report period
    is selected (audit §2.10).  Balance and YTD-cumulative flow fields read the
    latest visible statement by default; a field that cannot be read that way is
    rejected at registration rather than silently period-ambiguous.

    Shared structural columns (``pub_date``/``update_time``/
    ``report_period_end_date`` are deliberately reused by name across the four
    statements) are ``mining_allowed=False`` metadata, not period-selected
    features, and are exempt from period contracts.
    """
    from fields import ASHARE_FIELD_SPECS
    from .logical_tables import logical_table_contract

    for spec in ASHARE_FIELD_SPECS:
        if not spec.mining_allowed:
            continue
        table = str(spec.table)
        contract = logical_table_contract(table)
        if contract is None or contract.join_policy != "financial_pit":
            continue
        selector = str(spec.metadata.get("period_selector", "latest_visible_period"))
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
) -> pd.DataFrame:
    """Select all requested fields from one visible statement row per decision.

    This intentionally selects rows before projecting fields.  Loading fields
    independently can combine values from different periods or revisions.
    """
    field_names = tuple(dict.fromkeys(str(field) for field in fields))
    if not field_names:
        raise ValueError("financial row bundle requires at least one field")
    missing = sorted(set(field_names) - set(events.columns))
    if missing:
        raise ValueError(f"financial events missing bundle fields: {missing}")
    # Contract enforcement (audit §2.10): a catalog field whose registered period
    # selector is narrower than the caller's must not be read under a looser
    # selection.  latest_visible_period satisfies any contract; annual_only and
    # quarterly_only are incompatible with each other and with the opposite view.
    # Ad-hoc (non-catalog) event columns are left to the caller.
    for name in field_names:
        contract = FUNDAMENTAL_FIELD_CONTRACTS.get(name)
        if contract is None:
            continue
        required = contract.period_selector
        if selector == "latest_visible_period":
            continue
        if required not in {"latest_visible_period", selector}:
            raise ValueError(
                f"financial field {name!r} requires period_selector "
                f"{required!r}, got {selector!r}"
            )
    selected = select_visible_row_bundles(
        decisions,
        events,
        selector=selector,
        decision_time=decision_time,
        columns=columns,
    )
    keep = list(decisions.columns)
    for name in (columns.period_end, columns.available_at, columns.revision_id, *field_names):
        if name in selected.columns and name not in keep:
            keep.append(name)
    return selected[keep]
