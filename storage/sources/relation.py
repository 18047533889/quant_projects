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
    ratio_unit: Literal["percent", "decimal"] | None = None,
    top_n: int = 10,
    rank_column: str | None = None,
    entity_id_column: str | None = None,
    entity_name_column: str | None = None,
    allow_display_name_fallback: bool = False,
) -> pd.DataFrame:
    """Aggregate holder rows into stable TopTen primitives per report bundle.

    R24-019..022: the ratio unit is taken ONLY from the declared source
    contract (``ratio_unit``) — the data-value heuristic
    (``max(abs(ratio) > 1 → /100``) is deleted.  A value outside the declared
    unit's bounds raises a source-contract error instead of scaling the whole
    series.

    R24-027..029: when ``rank_column`` (e.g. the source ``ShareholderRank``)
    is provided, the official TopTen snapshot is selected by rank 1..N, NOT
    re-sorted by ``amount_column``; a conflict between the official rank and
    the amount ordering emits a DQ warning and never silently reorders the
    "official top ten".

    R24-023..026: entity identity — cross-period tracking callers MUST pass
    ``entity_id_column`` (EntityStableId).  ``entity_name_column`` is a
    display name that may fall back as identity ONLY for a static same-snapshot
    aggregation and only with ``allow_display_name_fallback=True``.  Two rows
    sharing a display name but carrying distinct entity ids are never merged.
    """
    import warnings

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if ratio_unit not in ("percent", "decimal"):
        raise ValueError(
            "ratio_unit must be declared from the source contract "
            "('percent' or 'decimal'); data-value inference is forbidden (R24-019/020)"
        )
    groups = tuple(group_columns)
    required = set(groups) | {amount_column, ratio_column}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"holder rows missing columns: {missing}")
    ordered = rows.copy()
    ordered[amount_column] = pd.to_numeric(ordered[amount_column], errors="coerce")
    ordered[ratio_column] = pd.to_numeric(ordered[ratio_column], errors="coerce")
    finite_ratio = ordered[ratio_column].replace([np.inf, -np.inf], np.nan)
    # R24-019: no heuristic.  Declared percent → decimal conversion only.
    if ratio_unit == "percent":
        ordered[ratio_column] = finite_ratio * 0.01
    else:
        ordered[ratio_column] = finite_ratio
    # R24-022: source-contract bounds check — a ratio outside the declared unit
    # is a contract violation, not a rescale trigger.
    rmin, rmax = ordered[ratio_column].min(skipna=True), ordered[ratio_column].max(skipna=True)
    if not pd.isna(rmax) and (rmax > 1.0001 or (not pd.isna(rmin) and rmin < -0.0001)):
        raise ValueError(
            f"holder ratio values outside the declared '{ratio_unit}' contract "
            f"bounds (min={rmin!r}, max={rmax!r}) — source-contract error, "
            "no data-value rescaling (R24-022)"
        )
    # R24-023..026: entity identity.  Prefer the stable id; a display-name
    # fallback is only allowed for static aggregation with explicit opt-in.
    entity_column = entity_id_column
    if entity_column is None and entity_name_column is not None:
        if not allow_display_name_fallback:
            raise ValueError(
                "entity display-name fallback requires "
                "allow_display_name_fallback=True (R24-025); cross-period "
                "tracking must use an EntityStableId column (R24-024)"
            )
        entity_column = entity_name_column
    if entity_column is not None:
        if entity_column not in ordered.columns:
            raise ValueError(f"holder rows missing entity column {entity_column!r}")
        ordered = ordered.dropna(subset=[entity_column])
        ordered = (
            ordered.groupby([*groups, entity_column], sort=False, dropna=False)
            .agg({amount_column: "sum", ratio_column: "sum"})
            .reset_index()
        )
    # R24-027..029: official TopTen rank by source ShareholderRank when present.
    if rank_column is not None:
        if rank_column not in ordered.columns:
            raise ValueError(f"holder rows missing official rank column {rank_column!r}")
        ordered[rank_column] = pd.to_numeric(ordered[rank_column], errors="coerce")
        ordered = ordered.sort_values(
            [*groups, rank_column], ascending=[True] * len(groups) + [True], kind="stable"
        )
        top = ordered.groupby(list(groups), sort=False, dropna=False).head(top_n)
        # DQ: the official rank ordering must not contradict the amount ordering.
        by_amount = ordered.groupby(list(groups), sort=False, dropna=False).apply(
            lambda g: g.sort_values(amount_column, ascending=False)[rank_column].tolist(),
            include_groups=False,
        )
        official = top.groupby(list(groups), sort=False, dropna=False).apply(
            lambda g: g[rank_column].tolist(), include_groups=False,
        )
        for key in by_amount.index:
            amt_ranks = by_amount[key]
            off_ranks = official[key]
            if off_ranks and amt_ranks and off_ranks != sorted(amt_ranks)[: len(off_ranks)]:
                warnings.warn(
                    f"holder TopTen official rank {off_ranks} conflicts with "
                    f"amount ordering ranks {amt_ranks} for {key} (R24-028 DQ warning)",
                    stacklevel=2,
                )
    else:
        ordered = ordered.sort_values(
            [*groups, amount_column], ascending=[True] * len(groups) + [False], kind="stable"
        )
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
    decision_time: str | pd.Timestamp,
    periods: int = 1,
    pct: bool = False,
    revision_column: str | None = None,
    vintage_column: str | None = None,
) -> pd.DataFrame:
    """Change across distinct relation snapshots, bitemporal (R24-030..032).

    The (snapshot_id, available_at, revision) triple is the snapshot identity.
    Selection rule (R24-031): the caller's ``decision_time`` → only vintages
    visible by then → for each specific snapshot the LATEST visible revision.
    A full-sample ``drop_duplicates(keep='last')`` is FORBIDDEN — it would let
    a future revision of an old snapshot overwrite the vintage a historical
    replay would have seen (lookahead).
    """
    if periods <= 0:
        raise ValueError("periods must be positive")
    required = {instrument_column, snapshot_column, available_column, value_column}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"relation snapshot rows missing columns: {missing}")
    out = rows.copy()
    out[available_column] = pd.to_datetime(out[available_column], errors="raise", utc=True)
    cutoff = pd.Timestamp(decision_time)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    # R24-031: only vintages VISIBLE at the decision time.
    out = out.loc[out[available_column] <= cutoff].copy()
    if out.empty:
        return out
    # R24-032: latest VISIBLE revision per snapshot — no full-sample keep-last.
    sort_key = [instrument_column, available_column, snapshot_column]
    if revision_column is not None:
        sort_key.append(revision_column)
    if vintage_column is not None:
        sort_key.append(vintage_column)
    out = out.sort_values(sort_key, kind="stable")
    dedup_key = [instrument_column, snapshot_column]
    if vintage_column is not None:
        dedup_key.append(vintage_column)
    out = out.drop_duplicates(dedup_key, keep="last")
    previous = out.groupby(instrument_column, sort=False)[value_column].shift(periods)
    change = out[value_column] - previous
    if pct:
        change = change / previous.replace(0, np.nan)
    out[f"{value_column}_{'pct_change' if pct else 'change'}"] = change.replace(
        [np.inf, -np.inf], np.nan
    )
    out["snapshot_vintage_visible"] = True
    return out


def top_ten_features_asof(
    decisions: pd.DataFrame,
    holder_rows: pd.DataFrame,
    *,
    decision_time: str = "decision_timestamp",
    columns: PITColumns = PITColumns(),
    amount_column: str = "holding_amount",
    ratio_column: str = "holding_ratio",
    ratio_unit: Literal["percent", "decimal"],
    top_n: int = 10,
    rank_column: str | None = None,
    max_age_days: int | None = None,
    allow_unbounded_staleness: bool = False,
) -> pd.DataFrame:
    """Top-ten snapshot features as-of, with snapshot staleness/age metadata.

    R24-033..035: every snapshot relation output must carry the visible
    ``snapshot_available_at`` / ``snapshot_age`` so downstream mining can gate.
    Production mining must pass ``max_age_days`` (a real bound) or explicitly
    ``allow_unbounded_staleness=True`` — unbounded ffill is never the default.
    """
    if max_age_days is None and not allow_unbounded_staleness:
        raise ValueError(
            "top_ten_features_asof requires max_age_days or explicit "
            "allow_unbounded_staleness=True (R24-034/035 — unbounded snapshot "
            "staleness is not a default)"
        )
    if max_age_days is not None and (not isinstance(max_age_days, int) or max_age_days <= 0):
        raise ValueError("max_age_days must be a positive integer")
    aggregated = aggregate_holder_rows(
        holder_rows,
        group_columns=(columns.instrument, columns.period_end, columns.available_at),
        amount_column=amount_column,
        ratio_column=ratio_column,
        ratio_unit=ratio_unit,
        top_n=top_n,
        rank_column=rank_column,
    )
    # Top-ten holder filings land after close; conservative next_trading_day
    # visibility (audit §2.9) for the shareholder snapshot as-of join.
    out = pit_asof_join(
        decisions,
        aggregated,
        decision_time=decision_time,
        columns=columns,
        max_age_days=max_age_days,
        available_policy="next_trading_day",
    )
    # R24-033: snapshot age metadata for mining gates.
    out["snapshot_available_at"] = pd.to_datetime(out[columns.available_at], errors="coerce", utc=True)
    decision = pd.to_datetime(out[decision_time], errors="coerce", utc=True)
    out["snapshot_age_days"] = (decision - out["snapshot_available_at"]).dt.total_seconds() / 86400.0
    return out
