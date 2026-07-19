"""AutoFactorEvaluation provider for the FactorEngine cold-start catalog."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from evaluation.factor_pack import FactorDefinition, FactorPack
from factor_cold_start.catalog import load_catalog


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load(market: str, surface: str) -> FactorPack:
    rows = load_catalog(market, surface)
    factors = tuple(
        FactorDefinition(
            name=row.factor_id,
            formula=row.formula,
            source_formula=row.formula,
            description=row.rationale,
            metadata={
                **dict(row.metadata),
                "domain": "price_volume",
                "market": row.market,
                "family": row.family,
                "subfamily": row.subfamily,
                "horizon": row.horizon,
                "complexity": row.complexity,
                "availability_tier": row.availability_tier,
                "required_fields": list(row.required_fields),
                "operators": list(row.operators),
                "dsl_surface": "daily" if surface == "daily" else "compat",
                "owner": "factor_cold_start",
            },
        )
        for row in rows
    )
    source_rows = [{"name": row.factor_id, "formula_hash": row.formula_hash} for row in rows]
    source_hash = _sha(_canonical(source_rows))
    name = f"factor_cold_start_{market}_{surface}"
    pack_hash = _sha(_canonical({"name": name, "version": "1", "source_hash": source_hash, "factors": source_rows}))
    return FactorPack(
        name=name,
        version="1",
        source_hash=source_hash,
        pack_hash=pack_hash,
        factors=factors,
        metadata={
            "owner": "factor_cold_start",
            "market": market,
            "surface": surface,
            "dsl_surface": "daily" if surface == "daily" else "compat",
            "factor_count": len(factors),
        },
    )


def load_ashare_daily_pack() -> FactorPack:
    return _load("ashare", "daily")


def load_us_daily_pack() -> FactorPack:
    return _load("us", "daily")


def load_ashare_extended_pack() -> FactorPack:
    return _load("ashare", "extended")


def load_us_extended_pack() -> FactorPack:
    return _load("us", "extended")


def load_pack() -> FactorPack:
    market = os.environ.get("FACTOR_COLD_START_MARKET", "ashare").strip().lower()
    surface = os.environ.get("FACTOR_COLD_START_SURFACE", "daily").strip().lower()
    return _load(market, surface)
