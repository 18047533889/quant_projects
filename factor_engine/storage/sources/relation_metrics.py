# -*- coding: utf-8
"""PIT-visible relation snapshot metrics.

These functions operate on long relation rows and intentionally stay outside
ordinary numeric SeriesOperator registration until a relation source has
materialized a unique snapshot panel.
"""
from __future__ import annotations

from typing import Any
import math

import numpy as np
import pandas as pd


def relation_jaccard(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    group_columns=("instrument", "snapshot_id"),
    entity_column="entity_id",
) -> pd.DataFrame:
    """Compute one Jaccard score per PIT-visible snapshot group."""
    groups = tuple(group_columns)
    required = set(groups) | {entity_column}
    for name, frame in (("left", left), ("right", right)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} relation rows missing {missing}")
    # Build maps without allowing duplicate relation rows to inflate cardinality.
    # ``groupby`` returns a scalar for one grouping column and a tuple otherwise;
    # normalize both forms so every grouping component, including snapshot_id,
    # remains part of the key.
    def grouped(frame: pd.DataFrame) -> dict[tuple[Any, ...], set[Any]]:
        result: dict[tuple[Any, ...], set[Any]] = {}
        for raw_key, rows in frame.groupby(list(groups), dropna=False):
            key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            result[tuple(key)] = set(rows[entity_column].dropna())
        return result

    lhs = grouped(left)
    rhs = grouped(right)
    all_groups = sorted(set(lhs) | set(rhs), key=str)
    records = []
    for key in all_groups:
        union = lhs.get(key, set()) | rhs.get(key, set())
        intersection = lhs.get(key, set()) & rhs.get(key, set())
        records.append({**dict(zip(groups, key)), "relation_jaccard": float(len(intersection) / len(union)) if union else np.nan})
    return pd.DataFrame(records, columns=[*groups, "relation_jaccard"])


def relation_entropy(
    entity_id: pd.Series,
    weight: pd.Series,
    snapshot_id: pd.Series,
    *,
    instrument: pd.Series | None = None,
    normalized: bool = True,
) -> pd.DataFrame:
    """Compute (optionally normalized) entropy per instrument/snapshot."""
    if not (len(entity_id) == len(weight) == len(snapshot_id) and (instrument is None or len(instrument) == len(entity_id))):
        raise ValueError("relation entropy inputs must have equal lengths")
    data = pd.DataFrame({"entity_id": entity_id, "weight": pd.to_numeric(weight, errors="coerce"), "snapshot_id": snapshot_id})
    data["instrument"] = 0 if instrument is None else instrument
    data = data.dropna(subset=["entity_id", "weight", "snapshot_id"])
    if (data["weight"] < 0).any():
        raise ValueError("relation entropy weights must be non-negative")
    data = data.groupby(["instrument", "snapshot_id", "entity_id"], as_index=False)["weight"].sum()
    records = []
    for (inst, snapshot), group in data.groupby(["instrument", "snapshot_id"], dropna=False):
        total = float(group["weight"].sum())
        if total <= 0:
            value = np.nan
        else:
            probabilities = group["weight"].to_numpy(dtype=float) / total
            probabilities = probabilities[probabilities > 0]
            entropy = float(-(probabilities * np.log(probabilities)).sum())
            value = entropy / math.log(len(probabilities)) if normalized and len(probabilities) > 1 else (0.0 if normalized else entropy)
        records.append({"instrument": inst, "snapshot_id": snapshot, "relation_entropy": value})
    return pd.DataFrame(records, columns=["instrument", "snapshot_id", "relation_entropy"])
