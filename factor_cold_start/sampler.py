"""Deterministic diversity-aware sampling for factor-mining cold starts."""
from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Iterable

from .catalog import filter_catalog
from .model import ColdStartFactor


def _stable_seed(seed: int | str, market: str, surface: str) -> int:
    raw = f"{seed}|{market}|{surface}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def sample_factors(
    *,
    market: str,
    surface: str = "daily",
    size: int = 64,
    seed: int | str = 0,
    available_fields: Iterable[str] | None = None,
    families: Iterable[str] | None = None,
    availability_tiers: Iterable[str] | None = None,
    max_per_family: int | None = None,
) -> tuple[ColdStartFactor, ...]:
    """Sample a reproducible set balanced across family, horizon and complexity."""
    if size < 1:
        raise ValueError("size must be positive")
    rows = list(
        filter_catalog(
            market,
            surface,
            available_fields=available_fields,
            families=families,
            availability_tiers=availability_tiers,
        )
    )
    if not rows:
        raise ValueError("no eligible cold-start factors")
    if size >= len(rows):
        return tuple(rows)

    rng = random.Random(_stable_seed(seed, market, surface))
    buckets: dict[tuple[str, str, str], list[ColdStartFactor]] = defaultdict(list)
    for row in rows:
        horizon_bucket = (
            "na" if row.horizon is None else "short" if row.horizon <= 10 else "medium" if row.horizon <= 60 else "long"
        )
        buckets[(row.family, horizon_bucket, row.complexity)].append(row)
    for values in buckets.values():
        rng.shuffle(values)

    family_counts: dict[str, int] = defaultdict(int)
    keys = list(buckets)
    rng.shuffle(keys)
    selected: list[ColdStartFactor] = []
    while len(selected) < size and keys:
        progressed = False
        for key in list(keys):
            family = key[0]
            if max_per_family is not None and family_counts[family] >= max_per_family:
                continue
            bucket = buckets[key]
            if not bucket:
                continue
            row = bucket.pop()
            selected.append(row)
            family_counts[family] += 1
            progressed = True
            if len(selected) >= size:
                break
        keys = [key for key in keys if buckets[key]]
        rng.shuffle(keys)
        if not progressed:
            break

    if len(selected) < size:
        remaining = [row for row in rows if row.factor_id not in {x.factor_id for x in selected}]
        rng.shuffle(remaining)
        selected.extend(remaining[: size - len(selected)])
    return tuple(selected)
