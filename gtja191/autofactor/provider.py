"""GTJA191 factor-pack provider for external AutoFactorEvaluation."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from evaluation.factor_pack import FactorDefinition, FactorPack
from lib.catalog import DELIVERABLE_COUNT, deliverable_catalog, deliverable_names


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_pack() -> FactorPack:
    catalog = deliverable_catalog()
    names = deliverable_names(catalog)
    rows: list[dict[str, Any]] = []
    factors: list[FactorDefinition] = []
    for name in names:
        item = catalog[name]
        formula = str(item["dsl_formula"]).strip()
        source_formula = str(item.get("source_formula") or "").strip()
        formula_hash = _sha(formula)
        rows.append(
            {
                "name": name,
                "formula": formula,
                "source_formula": source_formula,
                "formula_hash": formula_hash,
            }
        )
        factors.append(
            FactorDefinition(
                name=name,
                formula=formula,
                source_formula=source_formula,
                description=f"GTJA191 canonical factor {name}",
                formula_hash=formula_hash,
                metadata={
                    "domain": "price_volume",
                    "frequency": "1d",
                    "universe": "GTJA185",
                    "owner": "gtja191",
                },
            )
        )
    if len(factors) != DELIVERABLE_COUNT:
        raise ValueError(
            f"expected {DELIVERABLE_COUNT} deliverable factors, got {len(factors)}"
        )
    source_hash = _sha(_canonical(rows))
    pack_hash = _sha(
        _canonical(
            {
                "name": "gtja185",
                "version": "1",
                "source_hash": source_hash,
                "factors": rows,
            }
        )
    )
    return FactorPack(
        name="gtja185",
        version="1",
        source_hash=source_hash,
        pack_hash=pack_hash,
        factors=tuple(factors),
        metadata={
            "source": "gtja191/lib/catalog.py",
            "deliverable_count": DELIVERABLE_COUNT,
            "owner": "gtja191",
        },
    )
