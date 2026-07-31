# -*- coding: utf-8 -*-
"""Versioned derived-field definitions evaluated by the canonical FactorEngine DSL."""
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
    raw=os.environ.get("FACTOR_ENGINE_DERIVED_FIELD_REGISTRY","").strip()
    if raw:return Path(raw).expanduser().resolve()
    default=Path(__file__).resolve().parent.parent/"config"/"derived_fields.yaml"
    return default if default.is_file() else None


def load_derived_field_definition(name: str, *, path: str|Path|None=None) -> DerivedFieldDefinition:
    resolved=Path(path).expanduser().resolve() if path is not None else _registry_path()
    if resolved is None or not resolved.is_file():
        raise RuntimeError(
            f"derived field {name!r} has no configured definition; set FACTOR_ENGINE_DERIVED_FIELD_REGISTRY"
        )
    payload=yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    fields=payload.get("derived_fields",payload)
    raw=fields.get(name) if isinstance(fields,dict) else None
    if not isinstance(raw,dict) or not isinstance(raw.get("expression"),str):
        raise RuntimeError(f"derived field {name!r} is not defined in {resolved}")
    version=int(raw.get("version",1)); expr=str(raw["expression"]).strip()
    if version<=0 or not expr:raise ValueError(f"invalid derived field definition for {name!r}")
    digest=hashlib.sha256(f"{name}\0{version}\0{expr}".encode()).hexdigest()
    expected=raw.get("definition_hash")
    if expected and str(expected)!=digest:
        raise RuntimeError(f"derived field {name!r} definition_hash mismatch")
    return DerivedFieldDefinition(name=name,version=version,expression=expr,definition_hash=digest)


def evaluate_derived_field(name: str, data_source: Any):
    """Evaluate a registered derived expression on the same source/PIT context."""
    definition=load_derived_field_definition(name)
    active=set(getattr(_ACTIVE,"names",set()))
    if name in active:
        raise RuntimeError(f"cyclic derived-field dependency detected for {name!r}")
    _ACTIVE.names=active|{name}
    try:
        from api.dsl_parser import parse_factor
        from backend.pandas_backend import PandasBackend
        from runtime.engine import FactorEngine
        factor=parse_factor(
            definition.expression,
            name=f"derived::{name}@{definition.version}",
            surface="compat_research",
            dialect="lqtp",
            dialect_version="2026-07-19",
        )
        engine=FactorEngine(data_source,backend=PandasBackend(),enable_cache=True)
        return engine.run(factor, auto_warmup=True, trim_warmup=False, pit_enforce=True)
    finally:
        _ACTIVE.names=active
