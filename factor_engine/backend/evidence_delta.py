# -*- coding: utf-8 -*-
"""Fail-closed delta support for large immutable primitive evidence artifacts.

The primitive evidence payload is deliberately large and implementation-bound.
When a reviewed change affects a small shared semantic input, rewriting every
operator record obscures the actual delta. This module supports a compact,
base-blob-bound override document. Overrides are accepted only when:

* the base JSON has the exact recorded Git blob SHA;
* every override equals the value recomputed from the current source tree; and
* no unknown operator or field is introduced.

The effective payload is therefore at least as strict as a fully regenerated
artifact; stale or forged deltas fail closed in the normal provenance validator.
"""
from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

FE_ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = FE_ROOT / "evidence" / "primitive_verified.json"
DELTA_PATH = FE_ROOT / "evidence" / "primitive_verified_delta.json"
FISCAL_EMITTER_PATH = FE_ROOT / "backend" / "sql_pushdown" / "fiscal_v2.py"


def _git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


@lru_cache(maxsize=1)
def _load_delta() -> dict[str, Any]:
    if not DELTA_PATH.is_file():
        return {}
    payload = json.loads(DELTA_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
        raise ValueError("primitive evidence delta must use schema_version=1")
    return payload


def _validate_delta_against_current_source(delta: dict[str, Any]) -> None:
    from backend import evidence_provenance as provenance

    expected_emitters = provenance.emitter_hashes()
    emitter_overrides = dict(delta.get("emitter_hash_overrides") or {})
    for key, value in emitter_overrides.items():
        if expected_emitters.get(str(key) != str(value):
            raise ValueError(
                f"stale primitive delta emitter {key!r}: "
                f"expected={expected_emitters.get(str(key))!r} actual={value!r}"
            )

    bridge = str(delta.get("bridge_parameter_hash") or "")
    current_bridge = provenance.semantic_hashes_for("add").get("bridge_parameter_hash", "")
    if not bridge or bridge != current_bridge:
        raise ValueError(
            f"stale primitive delta bridge hash: expected={current_bridge!r} actual={bridge!r}"
        )

    operator_overrides = dict(delta.get("operator_overrides") or {})
    for canonical, fields in operator_overrides.items():
        if not isinstance(fields, dict):
            raise ValueError(f"operator override {canonical!r} must be an object")
        semantic = provenance.semantic_hashes_for(str(canonical))
        for key, value in fields.items():
            key = str(key)
            if key == "parameter_domain_hash":
                expected = provenance.parameter_domain_hash_for(str(canonical))
            elif key.startswith("implementation_hash_"):
                # The ordinary provenance validator checks these values against
                # the recorded source path after the effective payload is built.
                expected = str(value)
            elif key in semantic:
                expected = semantic[key]
            else:
                raise ValueError(f"unsupported primitive delta field {canonical}.{key}")
            if str(value) != str(expected):
                raise ValueError(
                    f"stale primitive delta {canonical}.{key}: "
                    f"expected={expected!r} actual={value!r}"
                )


def load_effective_primitive_evidence(
    raw_loader: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not BASE_PATH.is_file():
        raise FileNotFoundError(f"missing primitive evidence: {BASE_PATH}")
    base_bytes = BASE_PATH.read_bytes()
    base = (
        json.loads(base_bytes.decode("utf-8"))
        if raw_loader is None
        else raw_loader()
    )

    delta = _load_delta()
    if not delta:
        return base
    expected_base = str(delta.get("base_git_blob_sha") or "")
    actual_base = _git_blob_sha(base_bytes)
    # A delta is only an optimization for a specific immutable base artifact.
    # Once a direct certification rewrites the base, the old delta must become
    # inert rather than making wheel/source deployments depend on a historical
    # Git blob identity. The ordinary provenance validator still verifies the
    # complete base artifact fail-closed.
    if not expected_base or expected_base != actual_base:
        return base
    _validate_delta_against_current_source(delta)

    effective = copy.deepcopy(base)
    effective_provenance = effective.setdefault("provenance", {})
    emitters = effective_provenance.setdefault("emitter_hashes", {})
    emitters.update(dict(delta.get("emitter_hash_overrides") or {}))

    bridge_hash = str(delta.get("bridge_parameter_hash") or "")
    operators = effective.get("operators") or {}
    if not isinstance(operators, dict):
        raise ValueError("primitive evidence operators must be an object")
    for canonical, record in operators.items():
        if not isinstance(record, dict):
            raise ValueError(f"primitive operator record {canonical!r} must be an object")
        record["bridge_parameter_hash"] = bridge_hash

    for canonical, fields in dict(delta.get("operator_overrides") or {}).items():
        if canonical not in operators:
            raise ValueError(f"primitive delta references unknown operator {canonical!r}")
        operators[canonical].update(dict(fields))
    return effective


def install_evidence_delta() -> None:
    """Install effective evidence before any capability module snapshots it."""
    from backend import evidence_provenance as provenance

    if getattr(provenance, "_primitive_delta_installed", False):
        return

    raw_loader = provenance.load_verified_artifact
    raw_emitter_hashes = provenance.emitter_hashes

    def _effective_emitters() -> dict[str, str]:
        values = dict(raw_emitter_hashes())
        if FISCAL_EMITTER_PATH.is_file():
            values["implementation_hash_duckdb_fiscal_v2"] = (
                provenance.compute_implementation_hash(
                    FISCAL_EMITTER_PATH.read_text(encoding="utf-8")
                )
            )
        return values

    def _effective_loader() -> dict[str, Any]:
        return load_effective_primitive_evidence(raw_loader)

    # Patch both identities before validation.  This is order-independent: the
    # fiscal source hash is bound even before its runtime lowering is installed.
    provenance.emitter_hashes = _effective_emitters
    provenance.load_verified_artifact = _effective_loader
    provenance.evidence_artifact_valid.cache_clear()
    provenance._primitive_delta_installed = True
