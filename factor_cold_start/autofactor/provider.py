"""AutoFactorEvaluation providers for production and opt-in research catalogs."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from evaluation.factor_pack import FactorDefinition, FactorPack
from factor_cold_start.catalog import load_catalog
from factor_cold_start.production_admission import admit_factor


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load(market: str, surface: str) -> FactorPack:
    rows = load_catalog(market, surface)
    production_pack = surface == "daily"

    factor_rows: list[FactorDefinition] = []
    for row in rows:
        admission = admit_factor(row) if production_pack else None
        factor_rows.append(
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
                    "authoring_surface": row.surface,
                    "dsl_surface": "compat",
                    "output_frequency": "daily",
                    "readiness": "production" if production_pack else "research",
                    "production_admitted": bool(admission and admission.eligible),
                    "certified_backends": (
                        admission.backend_map if admission is not None else {}
                    ),
                    "physical_plan_backend": (
                        admission.physical_plan_backend if admission is not None else ""
                    ),
                    "physical_plan_candidates": (
                        list(admission.physical_plan_candidates)
                        if admission is not None
                        else []
                    ),
                    "routing_basis": admission.routing_basis if admission is not None else "",
                    "owner": "factor_cold_start",
                },
            )
        )

    factors = tuple(factor_rows)
    source_rows = [{"name": row.factor_id, "formula_hash": row.formula_hash} for row in rows]
    source_hash = _sha(_canonical(source_rows))
    name = f"factor_cold_start_{market}_{surface}"
    pack_hash = _sha(
        _canonical(
            {
                "name": name,
                "version": "2",
                "source_hash": source_hash,
                "factors": source_rows,
            }
        )
    )
    return FactorPack(
        name=name,
        version="2",
        source_hash=source_hash,
        pack_hash=pack_hash,
        factors=factors,
        metadata={
            "owner": "factor_cold_start",
            "market": market,
            "surface": surface,
            "authoring_surface": "mixed" if production_pack else "extended",
            "dsl_surface": "compat",
            "output_frequency": "daily",
            "readiness": "production" if production_pack else "research",
            "factor_count": len(factors),
        },
    )


def load_ashare_daily_pack() -> FactorPack:
    """Default A-share production-admitted daily-output pack."""
    return _load("ashare", "daily")


def load_us_daily_pack() -> FactorPack:
    """Default US production-admitted daily-output pack."""
    return _load("us", "daily")


def load_ashare_extended_pack() -> FactorPack:
    """Opt-in A-share research candidate archive."""
    return _load("ashare", "extended")


def load_us_extended_pack() -> FactorPack:
    """Opt-in US research candidate archive."""
    return _load("us", "extended")


def load_pack() -> FactorPack:
    market = os.environ.get("FACTOR_COLD_START_MARKET", "ashare").strip().lower()
    surface = os.environ.get("FACTOR_COLD_START_SURFACE", "daily").strip().lower()
    return _load(market, surface)