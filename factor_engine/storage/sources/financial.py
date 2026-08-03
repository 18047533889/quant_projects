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
    contract = FinancialFieldContract(table, field, period_selector, pit_required)
    existing = FUNDAMENTAL_FIELD_CONTRACTS.get(str(name))
    if existing is not None and existing != contract:
        raise ValueError(f"fundamental field {name!r} already has a different contract")
    FUNDAMENTAL_FIELD_CONTRACTS[str(name)] = contract
    return contract


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
