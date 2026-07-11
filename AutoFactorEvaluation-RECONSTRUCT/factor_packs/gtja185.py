"""Validated GTJA185 factor pack.

The bundled JSON is generated from the canonical repository-root ``gtja191``
catalog.  Runtime code consumes the immutable bundle so AutoFactorEvaluation
never depends on ad-hoc directory scans or candidate naming conventions.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

EXPECTED_FACTOR_COUNT = 185
PACK_SCHEMA_VERSION = "autofactor.factor_pack.v1"
PACK_PATH = Path(__file__).with_name("gtja185_pack.json")


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    formula: str
    source_formula: str
    description: str
    formula_hash: str
    metadata: Mapping[str, Any]

    def to_manifest(
        self,
        *,
        market: str = "ashare",
        universe_id: str = "A_SHARE_ALL_A_EX_ST",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            "schema_version": "disk.v1",
            "candidate_id": self.name,
            "factor_id": self.name,
            "campaign_id": "gtja185_builtin",
            "formula": self.formula,
            "expression_type": "dsl",
            "generator_name": "gtja191_canonical_catalog",
            "generator_version": str(self.metadata.get("source_catalog_hash", "unknown")),
            "mined_by": "quant_projects",
            "market": market,
            "universe_id": universe_id,
            "domain_root": "price_volume",
            "domain": "price_volume",
            "frequency_bucket": "1d",
            "signal_structure": "hybrid",
            "asset_class": "equity",
            "description": self.description,
            "source_formula": self.source_formula,
            "formula_hash": self.formula_hash,
            "factor_pack": "gtja185",
            "factor_pack_version": str(self.metadata.get("pack_version", "1")),
            "metrics": {"train": {}, "valid": {}, "test": {}},
        }
        if start_date is not None or end_date is not None:
            manifest["date_range"] = {"start": start_date, "end": end_date}
        return manifest


@dataclass(frozen=True)
class FactorPack:
    name: str
    version: str
    source_catalog_hash: str
    pack_hash: str
    factors: tuple[FactorDefinition, ...]
    metadata: Mapping[str, Any]

    def names(self) -> list[str]:
        return [factor.name for factor in self.factors]

    def by_name(self) -> dict[str, FactorDefinition]:
        return {factor.name: factor for factor in self.factors}

    def select(self, names: Iterable[str] | None = None, limit: int | None = None) -> tuple[FactorDefinition, ...]:
        selected = list(self.factors)
        if names is not None:
            requested = list(dict.fromkeys(str(name) for name in names))
            mapping = self.by_name()
            missing = [name for name in requested if name not in mapping]
            if missing:
                raise KeyError(f"unknown GTJA185 factors: {missing[:10]}")
            selected = [mapping[name] for name in requested]
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive")
            selected = selected[:limit]
        return tuple(selected)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_factor_entry(raw: Mapping[str, Any], *, index: int) -> FactorDefinition:
    name = str(raw.get("name") or "").strip()
    formula = str(raw.get("formula") or "").strip()
    source_formula = str(raw.get("source_formula") or "").strip()
    description = str(raw.get("description") or "").strip()
    formula_hash = str(raw.get("formula_hash") or "").strip()
    if not name:
        raise ValueError(f"factor entry {index} has no name")
    if not formula:
        raise ValueError(f"factor {name} has no formula")
    expected_hash = _sha256_text(formula)
    if formula_hash != expected_hash:
        raise ValueError(
            f"factor {name} formula hash mismatch: expected {expected_hash}, got {formula_hash}"
        )
    if not name.startswith("gtja191_alpha_"):
        raise ValueError(f"unexpected GTJA factor name: {name}")
    return FactorDefinition(
        name=name,
        formula=formula,
        source_formula=source_formula,
        description=description or f"GTJA191 canonical factor {name}",
        formula_hash=formula_hash,
        metadata=dict(raw.get("metadata") or {}),
    )


def load_gtja185_pack(path: str | Path | None = None) -> FactorPack:
    pack_path = Path(path) if path is not None else PACK_PATH
    if not pack_path.is_file():
        raise FileNotFoundError(
            f"GTJA185 pack not found: {pack_path}. Run scripts/generate_gtja185_pack.py from repository root."
        )
    raw = json.loads(pack_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("GTJA185 pack root must be a mapping")
    if raw.get("schema_version") != PACK_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported factor pack schema: {raw.get('schema_version')!r}"
        )
    factor_rows = raw.get("factors")
    if not isinstance(factor_rows, list):
        raise ValueError("GTJA185 pack factors must be a list")
    factors = tuple(
        _validate_factor_entry(item, index=index)
        for index, item in enumerate(factor_rows, start=1)
        if isinstance(item, Mapping)
    )
    if len(factors) != EXPECTED_FACTOR_COUNT:
        raise ValueError(
            f"GTJA185 pack must contain {EXPECTED_FACTOR_COUNT} factors, got {len(factors)}"
        )
    names = [factor.name for factor in factors]
    if len(set(names)) != len(names):
        raise ValueError("GTJA185 pack contains duplicate factor names")
    formulas = [factor.formula for factor in factors]
    if len(set(formulas)) != len(formulas):
        duplicates = sorted({formula for formula in formulas if formulas.count(formula) > 1})
        raise ValueError(f"GTJA185 pack contains duplicate formulas: {duplicates[:3]}")

    canonical_payload = {
        "schema_version": raw["schema_version"],
        "name": raw.get("name"),
        "version": raw.get("version"),
        "source_catalog_hash": raw.get("source_catalog_hash"),
        "factors": factor_rows,
    }
    expected_pack_hash = _sha256_text(_canonical_json(canonical_payload))
    if str(raw.get("pack_hash") or "") != expected_pack_hash:
        raise ValueError(
            f"GTJA185 pack hash mismatch: expected {expected_pack_hash}, got {raw.get('pack_hash')}"
        )

    return FactorPack(
        name=str(raw.get("name") or "gtja185"),
        version=str(raw.get("version") or "1"),
        source_catalog_hash=str(raw.get("source_catalog_hash") or ""),
        pack_hash=expected_pack_hash,
        factors=factors,
        metadata=dict(raw.get("metadata") or {}),
    )
