# -*- coding: utf-8 -*-
"""PIT-safe relation primitives for classifications, memberships and holders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from pit_contract import PITColumns, pit_asof_join


@dataclass(frozen=True)
class IndustrySelection:
    industry_source: str

    def __post_init__(self) -> None:
        if not str(self.industry_source).strip():
            raise ValueError("IndustrySource must be a single non-empty value")


@dataclass(frozen=True)
class IndexSelection:
    index_symbol: str

    def __post_init__(self) -> None:
        if not str(self.index_symbol).strip():
            raise ValueError("IndexSymbol must be a non-empty exact identifier")


def filter_industry(rows: pd.DataFrame, industry_source: str, *, source_column: str = "IndustrySource") -> pd.DataFrame:
    """Filter one and only one industry taxonomy."""
    selection = IndustrySelection(industry_source)
    if source_column not in rows.columns:
        raise ValueError(f"industry rows missing {source_column!r}")
    return rows.loc[rows[source_column].astype(str) == selection.industry_source].copy()


def filter_index_constituents(rows: pd.DataFrame, index_symbol: str, *, index_column: str = "IndexSymbol") -> pd.DataFrame:
    """Filter exact index membership; prefixes and fuzzy identifiers are forbidden."""
    selection = IndexSelection(index_symbol)
    if index_column not in rows.columns:
        raise ValueError(f"index constituent rows missing {index_column!r}")
    return rows.loc[rows[index_column].astype(str) == selection.index_symbol].copy()


def effective_dividends(
    rows: pd.DataFrame,
    *,
    decision_time: str | pd.Timestamp,
    effective_column: str = "effective_at",
) -> pd.DataFrame:
    """Return only dividends effective by the decision time and mark the contract."""
    if effective_column not in rows.columns:
        raise ValueError(f"dividend rows missing effective column {effective_column!r}")
    out = rows.copy()
    effective = pd.to_datetime(out[effective_column], errors="raise", utc=True)
    cutoff = pd.Timestamp(decision_time)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    out = out.loc[effective <= cutoff].copy()
    out["effective_only"] = True
    return out


def aggregate_holder_rows(
    rows: pd.DataFrame,
    *,
    group_columns: Iterable[str] = ("instrument", "period_end", "available_at"),
    amount_column: str = "holding_amount",
    ratio_column: str = "holding_ratio",
    top_n: int = 10,
) -> pd.DataFrame:
    """Aggregate holder rows into stable TopTen primitives per report bundle."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    groups = tuple(group_columns)
    required = set(groups) | {amount_column, ratio_column}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"holder rows missing columns: {missing}")
    ordered = rows.copy()
    ordered[amount_column] = pd.to_numeric(ordered[amount_column], errors="coerce")
    ordered[ratio_column] = pd.to_numeric(ordered[ratio_column], errors="coerce")
    finite_ratio = ordered[ratio_column].replace([np.inf, -np.inf], np.nan)
    # COS holder ratios are percentages.  Canonical relation metrics operate on
    # decimal ratios; tolerate already-normalized input without scaling it twice.
    if finite_ratio.abs().max(skipna=True) > 1.0:
        ordered[ratio_column] = finite_ratio * 0.01
    else:
        ordered[ratio_column] = finite_ratio
    entity_column = next(
        (name for name in ("holder_entity_id", "entity_id", "holder_name") if name in ordered.columns),
        None,
    )
    if entity_column is not None:
        ordered = ordered.dropna(subset=[entity_column])
        ordered = (
            ordered.groupby([*groups, entity_column], sort=False, dropna=False)
            .agg({amount_column: "sum", ratio_column: "sum"})
            .reset_index()
        )
    ordered = ordered.sort_values([*groups, amount_column], ascending=[True] * len(groups) + [False], kind="stable")
    top = ordered.groupby(list(groups), sort=False, dropna=False).head(top_n)
    result = top.groupby(list(groups), sort=False, dropna=False).agg(
        top_ten_holding_amount=(amount_column, "sum"),
        top_ten_holding_ratio=(ratio_column, "sum"),
        top_ten_holder_count=(ratio_column, "count"),
        top_ten_largest_ratio=(ratio_column, "max"),
    ).reset_index()
    squares = top.assign(_ratio_sq=top[ratio_column].pow(2)).groupby(
        list(groups), sort=False, dropna=False
    )["_ratio_sq"].sum().reset_index(name="top_ten_concentration_hhi")
    return result.merge(squares, on=list(groups), how="left", validate="one_to_one")


def relation_snapshot_change(
    rows: pd.DataFrame,
    *,
    value_column: str,
    instrument_column: str = "instrument",
    snapshot_column: str = "snapshot_id",
    available_column: str = "available_at",
    periods: int = 1,
    pct: bool = False,
) -> pd.DataFrame:
    """Change across distinct visible relation snapshots, never trading-day rows."""
    if periods <= 0:
        raise ValueError("periods must be positive")
    required = {instrument_column, snapshot_column, available_column, value_column}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"relation snapshot rows missing columns: {missing}")
    out = rows.copy()
    out[available_column] = pd.to_datetime(out[available_column], errors="raise", utc=True)
    out = out.sort_values(
        [instrument_column, available_column, snapshot_column], kind="stable"
    ).drop_duplicates([instrument_column, snapshot_column], keep="last")
    previous = out.groupby(instrument_column, sort=False)[value_column].shift(periods)
    change = out[value_column] - previous
    if pct:
        change = change / previous.replace(0, np.nan)
    out[f"{value_column}_{'pct_change' if pct else 'change'}"] = change.replace(
        [np.inf, -np.inf], np.nan
    )
    return out


def top_ten_features_asof(
    decisions: pd.DataFrame,
    holder_rows: pd.DataFrame,
    *,
    decision_time: str = "decision_timestamp",
    columns: PITColumns = PITColumns(),
    amount_column: str = "holding_amount",
    ratio_column: str = "holding_ratio",
    top_n: int = 10,
) -> pd.DataFrame:
    aggregated = aggregate_holder_rows(
        holder_rows,
        group_columns=(columns.instrument, columns.period_end, columns.available_at),
        amount_column=amount_column,
        ratio_column=ratio_column,
        top_n=top_n,
    )
    # Top-ten holder filings land after close; conservative next_trading_day
    # visibility (audit §2.9) for the shareholder snapshot as-of join.
    return pit_asof_join(
        decisions,
        aggregated,
        decision_time=decision_time,
        columns=columns,
        max_age_days=None,
        available_policy="next_trading_day",
    )
