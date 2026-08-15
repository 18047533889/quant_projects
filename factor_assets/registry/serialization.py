"""Versioned deterministic JSON codecs for durable registry records."""
from __future__ import annotations
import json
from dataclasses import asdict
from typing import Any
from factor_assets.contracts.asset import AssetMetadata, FactorAsset
from factor_assets.contracts.evidence_ref import EvidenceBundleRef
from factor_assets.contracts.lineage import LineageRef, ParentRef
from factor_assets.contracts.lifecycle import LifecycleState, StateEvent

CODEC_VERSION = 1

def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def _pack(kind: str, value: Any) -> str:
    return _dump({"codec": kind, "version": CODEC_VERSION, "value": value})

def _unpack(text: str, kind: str) -> Any:
    obj = json.loads(text)
    if not isinstance(obj, dict) or obj.get("codec") != kind or obj.get("version") != CODEC_VERSION:
        raise ValueError(f"unsupported or invalid {kind} codec")
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
    obj["latest_evidence_ref"] = evidence_from_json(_pack("evidence_bundle", obj["latest_evidence_ref"]) if obj.get("latest_evidence_ref") else None)
    return FactorAsset(**obj)

def event_to_json(value: StateEvent) -> str:
    obj = asdict(value); obj["from_state"] = value.from_state.value; obj["to_state"] = value.to_state.value
    return _pack("state_event", obj)
def event_from_json(text: str) -> StateEvent:
    obj = _unpack(text, "state_event"); obj["from_state"] = LifecycleState(obj["from_state"]); obj["to_state"] = LifecycleState(obj["to_state"])
    obj["evidence_refs"] = tuple(obj.get("evidence_refs", ()))
    return StateEvent(**obj)
