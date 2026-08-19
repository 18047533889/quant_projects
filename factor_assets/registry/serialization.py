"""Versioned deterministic JSON codecs for durable registry records."""
from __future__ import annotations
import json
from dataclasses import asdict
from typing import Any
from factor_assets.contracts.asset import AssetMetadata, FactorAsset
from factor_assets.contracts.evidence_ref import EvidenceBundleRef
from factor_assets.contracts.lineage import LineageRef, ParentRef
from factor_assets.contracts.lifecycle import (
    HealthState,
    LifecycleState,
    StateEvent,
    ValidationStatus,
)
from factor_assets.errors import SchemaVersionError

# v1: initial codec (metadata/lineage/evidence/asset/state_event).
# v2: asset payload gains validation_status and health_state (FA-P0-12).
# v2 readers still accept v1 payloads (the two new fields default
# fail-closed to UNVALIDATED/ACTIVE), but never write them.
CODEC_VERSION = 2

# Readers accept any version in this set; writers always emit
# CODEC_VERSION.  A version outside the set is rejected rather than
# guessed at (fail-closed on unknown future formats).
_READABLE_CODEC_VERSIONS = (1, 2)

def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def _pack(kind: str, value: Any) -> str:
    return _dump({"codec": kind, "version": CODEC_VERSION, "value": value})

def _unpack(text: str, kind: str) -> Any:
    obj = json.loads(text)
    if not isinstance(obj, dict) or obj.get("codec") != kind:
        raise ValueError(f"unsupported or invalid {kind} codec")
    version = obj.get("version")
    # bool is an int subclass (True == 1) and 1.0 == 1, so a type-weak
    # membership test would accept malformed versions as v1.
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version not in _READABLE_CODEC_VERSIONS
    ):
        raise SchemaVersionError(
            f"unsupported {kind} codec version {version!r}; "
            f"readable versions are {list(_READABLE_CODEC_VERSIONS)}"
        )
    return obj["value"]

def metadata_to_json(value: AssetMetadata) -> str:
    return _pack("asset_metadata", asdict(value))
def metadata_from_json(text: str) -> AssetMetadata:
    obj = _unpack(text, "asset_metadata")
    obj["domains"] = tuple(obj.get("domains", ()))
    obj["required_fields"] = tuple(obj.get("required_fields", ()))
    return AssetMetadata(**obj)

def lineage_to_json(value: LineageRef) -> str:
    return _pack("lineage", asdict(value))
def lineage_from_json(text: str) -> LineageRef:
    obj = _unpack(text, "lineage")
    obj["parents"] = tuple(ParentRef(**p) for p in obj.get("parents", ()))
    return LineageRef(**obj)

def evidence_to_json(value: EvidenceBundleRef | None) -> str | None:
    return None if value is None else _pack("evidence_bundle", asdict(value))
def evidence_from_json(text: str | None) -> EvidenceBundleRef | None:
    if text is None:
        return None
    obj = _unpack(text, "evidence_bundle")
    obj["factor_ids"] = tuple(obj.get("factor_ids", ()))
    obj["warnings"] = tuple(obj.get("warnings", ()))
    return EvidenceBundleRef(**obj)

def asset_to_json(value: FactorAsset) -> str:
    obj = asdict(value)
    obj["metadata"] = asdict(value.metadata)
    obj["lineage"] = asdict(value.lineage)
    obj["lifecycle_state"] = value.lifecycle_state.value
    obj["validation_status"] = value.validation_status.value
    obj["health_state"] = value.health_state.value
    return _pack("asset", obj)
def asset_from_json(text: str) -> FactorAsset:
    obj = _unpack(text, "asset")
    metadata = obj["metadata"]
    metadata["domains"] = tuple(metadata.get("domains", ()))
    metadata["required_fields"] = tuple(metadata.get("required_fields", ()))
    obj["metadata"] = AssetMetadata(**metadata)
    obj["lineage"] = lineage_from_json(_pack("lineage", obj["lineage"]))
    obj["tags"] = tuple(obj.get("tags", ()))
    obj["lifecycle_state"] = LifecycleState(obj["lifecycle_state"])
    obj["validation_status"] = ValidationStatus(
        obj.get("validation_status", ValidationStatus.UNVALIDATED.value)
    )
    obj["health_state"] = HealthState(obj.get("health_state", HealthState.ACTIVE.value))
    obj["latest_evidence_ref"] = evidence_from_json(_pack("evidence_bundle", obj["latest_evidence_ref"]) if obj.get("latest_evidence_ref") else None)
    return FactorAsset(**obj)

def event_to_json(value: StateEvent) -> str:
    obj = asdict(value); obj["from_state"] = value.from_state.value; obj["to_state"] = value.to_state.value
    return _pack("state_event", obj)
def event_from_json(text: str) -> StateEvent:
    obj = _unpack(text, "state_event"); obj["from_state"] = LifecycleState(obj["from_state"]); obj["to_state"] = LifecycleState(obj["to_state"])
    obj["evidence_refs"] = tuple(obj.get("evidence_refs", ()))
    return StateEvent(**obj)
