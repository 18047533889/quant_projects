# -*- coding: utf-8 -*-
"""Versioned derived-field definitions evaluated by production FactorEngine.

Derived fields are transitive factor dependencies, not an escape hatch around
PIT/backend policy. Every definition is parsed on the production-compatible
Daily+Extended surface, compiled in production mode, input-DQ/PIT checked, and
returns only the computed Series to its SourceRef resolver.
"""
from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_ACTIVE = threading.local()


@dataclass(frozen=True)
class DerivedFieldDefinition:
    name: str
    version: int
    expression: str
    definition_hash: str


def _registry_path() -> Path | None:
    raw = os.environ.get("FACTOR_ENGINE_DERIVED_FIELD_REGISTRY", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    default = Path(__file__).resolve().parent.parent / "config" / "derived_fields.yaml"
    return default if default.is_file() else None


def load_derived_field_definition(
    name: str,
    *,
    path: str | Path | None = None,
) -> DerivedFieldDefinition:
    resolved = Path(path).expanduser().resolve() if path is not None else _registry_path()
    if resolved is None or not resolved.is_file():
        raise RuntimeError(
            f"derived field {name!r} has no configured definition; "
            "set FACTOR_ENGINE_DERIVED_FIELD_REGISTRY"
        )
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    fields = payload.get("derived_fields", payload)
    raw = fields.get(name) if isinstance(fields, dict) else None
    if not isinstance(raw, dict) or not isinstance(raw.get("expression"), str):
        raise RuntimeError(f"derived field {name!r} is not defined in {resolved}")
    version = int(raw.get("version", 1))
    expr = str(raw["expression"]).strip()
    if version <= 0 or not expr:
        raise ValueError(f"invalid derived field definition for {name!r}")
    digest = hashlib.sha256(f"{name}\0{version}\0{expr}".encode()).hexdigest()
    expected = raw.get("definition_hash")
    if expected and str(expected) != digest:
        raise RuntimeError(f"derived field {name!r} definition_hash mismatch")
    return DerivedFieldDefinition(name, version, expr, digest)


def derived_field_lineage(name: str) -> dict[str, Any]:
    definition = load_derived_field_definition(name)
    return {
        "name": definition.name,
        "version": definition.version,
        "definition_hash": definition.definition_hash,
    }


def evaluate_derived_field(name: str, data_source: Any):
    definition = load_derived_field_definition(name)
    active = set(getattr(_ACTIVE, "names", set()))
    if name in active:
        chain = " -> ".join([*sorted(active), name])
        raise RuntimeError(f"cyclic derived-field dependency detected: {chain}")
    _ACTIVE.names = active | {name}
    try:
        from api.dsl_parser import parse_factor
        from backend.pandas_backend import PandasBackend
        from runtime.engine import FactorEngine

        factor = parse_factor(
            definition.expression,
            name=f"derived::{name}@{definition.version}",
            surface="compat",
            dialect="lqtp",
            dialect_version="2026-07-19",
        )
        engine = FactorEngine(PandasBackend(), data_source, cache=None, run_mode="production")
        output = engine.run(
            factor,
            input_dq_check=True,
            input_dq_strict=True,
            auto_warmup=True,
            trim_warmup=False,
            pit_enforce=True,
        )
        result = output.get("result")
        if result is None:
            raise RuntimeError(f"derived field {name!r} produced no result")
        return result
    finally:
        _ACTIVE.names = active
