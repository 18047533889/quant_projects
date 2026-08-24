#!/usr/bin/env python3
"""生成 factor_engine/docs/market_field_manifest.json — 每个 canonical 概念的双市场 provider 清单。

对 ``fields.concepts`` 中的每个概念，汇总 A/US 两侧的 provider binding：
status / dataset / physical / transform / source_unit / coverage / pit。
用于跨市场语义审计与文档展示（spec §79）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]  # factor_engine
OUT_JSON = ROOT / "factor_engine" / "docs" / "market_field_manifest.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_manifest() -> dict[str, Any]:
    from factor_engine.fields.concepts import list_concepts
    from factor_engine.fields.providers import PROVIDER_REGISTRY
    from factor_engine.fields.units_v2 import legacy_unit_string
    from factor_engine.market.capabilities import MarketStatus

    rows: dict[str, Any] = {}
    for concept in list_concepts():
        per_market: dict[str, Any] = {}
        for market in ("ashare", "us"):
            b = PROVIDER_REGISTRY.binding(concept.concept_id, market)
            if b is None:
                per_market[market] = {
                    "status": MarketStatus.UNKNOWN.value,
                    "providers": [],
                }
                continue
            per_market[market] = {
                "status": (
                    MarketStatus.CERTIFIED_NATIVE.value
                    if b.quality.production_usable
                    else b.quality.value
                ),
                "dataset": b.dataset,
                "physical": list(b.physical_fields),
                "transform": b.transform_description,
                "source_unit": legacy_unit_string(b.source_unit),
                "canonical_unit": legacy_unit_string(b.canonical_unit),
                "coverage": b.coverage.value,
                "pit": b.temporal_model,
                "knowledge_time": b.knowledge_time,
                "effective_time": b.effective_time,
                "available_at": b.available_at,
                "required_filters": [f.to_dict() for f in b.required_filters],
                "source_certified": b.source_certified,
                "providers": [b.provider_id],
                "notes": b.notes,
            }
        rows[concept.concept_id] = {
            "concept": concept.concept_id,
            "canonical_unit": legacy_unit_string(concept.canonical_unit),
            "domain": concept.domain,
            "value_kind": concept.value_kind,
            "cross_market_comparable": concept.cross_market_comparable,
            "price_basis": concept.price_basis,
            "ashare": per_market["ashare"],
            "us": per_market["us"],
        }
    return {
        "schema_version": "factor_engine.market_field_manifest.v1",
        "generated_by": "factor_engine/scripts/build_market_field_manifest.py",
        "concept_count": len(rows),
        "concepts": rows,
    }


def main() -> int:
    doc = build_manifest()
    OUT_JSON.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {OUT_JSON} ({doc['concept_count']} concepts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
