#!/usr/bin/env python3
"""Generate AutoFactorEvaluation's immutable GTJA185 bundle."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
GTJA_ROOT = REPO / "gtja191"
OUTPUT = PROJECT / "factor_packs" / "gtja185_pack.json"
SCHEMA = "autofactor.factor_pack.v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> int:
    if not GTJA_ROOT.is_dir():
        raise RuntimeError(f"GTJA root not found: {GTJA_ROOT}")
    sys.path.insert(0, str(GTJA_ROOT))
    from lib.catalog import DELIVERABLE_COUNT, deliverable_catalog  # noqa: WPS433

    catalog = deliverable_catalog()
    if len(catalog) != DELIVERABLE_COUNT or DELIVERABLE_COUNT != 185:
        raise RuntimeError(f"expected 185 deliverable factors, got {len(catalog)}")
    source_catalog_path = GTJA_ROOT / "dsl" / "gtja191_dsl_catalog.json"
    source_catalog_hash = _sha(source_catalog_path.read_text(encoding="utf-8"))
    factors: list[dict[str, Any]] = []
    for name in sorted(catalog):
        item = catalog[name]
        formula = str(item["dsl_formula"]).strip()
        source = str(item.get("source_formula") or "").strip()
        if not formula:
            raise RuntimeError(f"empty formula: {name}")
        if "VWAP" in source.upper() and "col('vwap')" not in formula and 'col("vwap")' not in formula:
            raise RuntimeError(f"VWAP semantic drift: {name}: {formula}")
        factors.append(
            {
                "name": name,
                "formula": formula,
                "source_formula": source,
                "description": f"GTJA191 canonical factor {name}; source: {source}",
                "formula_hash": _sha(formula),
                "metadata": {
                    "conversion": item.get("conversion"),
                    "validation": item.get("validation"),
                    "source_catalog_hash": source_catalog_hash,
                    "pack_version": "1",
                },
            }
        )
    payload = {
        "schema_version": SCHEMA,
        "name": "gtja185",
        "version": "1",
        "source_catalog_hash": source_catalog_hash,
        "factors": factors,
    }
    pack_hash = _sha(_canonical(payload))
    output = {
        **payload,
        "pack_hash": pack_hash,
        "metadata": {
            "source_catalog_hash": source_catalog_hash,
            "generator": "AutoFactorEvaluation-RECONSTRUCT/scripts/generate_gtja185_pack.py",
            "source_catalog": str(source_catalog_path.relative_to(REPO)),
            "factor_count": len(factors),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "factor_count": len(factors), "pack_hash": pack_hash}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
