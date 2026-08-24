# -*- coding: utf-8
"""Evidence schema v2：参数域认证 + semantic hash 失效。"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from factor_engine.planner.logical_plan import PlanNode

_EVIDENCE_JSON = Path(__file__).resolve().parents[2] / "evidence" / "primitive_verified.json"


@lru_cache(maxsize=1)
def load_evidence_v2() -> dict[str, Any]:
    if not _EVIDENCE_JSON.is_file():
        raise FileNotFoundError(f"missing evidence: {_EVIDENCE_JSON}")
    return json.loads(_EVIDENCE_JSON.read_text(encoding="utf-8"))


def evidence_schema_version() -> int:
    return int(load_evidence_v2().get("schema_version", 1))


def operator_evidence_record(canon: str) -> dict[str, Any] | None:
    records = load_evidence_v2().get("operators") or {}
    if isinstance(records, dict):
        rec = records.get(canon)
        return rec if isinstance(rec, dict) else None
    return None


def _literal_param(node: PlanNode, key: str, input_index: int | None = None) -> Any:
    if key in (node.attrs or {}):
        return node.attrs[key]
    if input_index is not None and input_index < len(node.inputs):
        child = node.inputs[input_index]
        if child.op == "literal":
            return child.attrs.get("value")
    return None


def supported_calls_match(node: PlanNode, supported: Mapping[str, Any]) -> bool:
    """检查 plan 调用是否落在 ``supported_calls`` 参数域内。"""
    if not supported:
        return True
    for key, allowed in supported.items():
        if key == "mode":
            from factor_engine.backend.plan_params import int_mode_from_plan_node

            mode = int_mode_from_plan_node(node, input_index=2, default=None)
            if mode is None:
                mode = _literal_param(node, "mode", input_index=2)
            if mode is None:
                continue
            if isinstance(allowed, list) and mode not in allowed:
                return False
        elif key == "to":
            val = _literal_param(node, "to", input_index=1)
            if val is None:
                val = 1.0
            try:
                val_f = float(val)
            except (TypeError, ValueError):
                return False
            if isinstance(allowed, list):
                allowed_f = {float(x) for x in allowed}
                if val_f not in allowed_f:
                    return False
        elif key in (node.attrs or {}):
            val = node.attrs[key]
            if isinstance(allowed, list) and val not in allowed:
                return False
    return True


def parameter_domain_verified(canon: str, node: PlanNode | None = None, *, production: bool = False) -> bool:
    """算子是否在 evidence 中声明了参数域，且当前调用匹配（production 默认 fail-closed）。"""
    rec = operator_evidence_record(canon)
    if rec is None:
        if production:
            return False
        return True
    supported = rec.get("supported_calls")
    if not supported:
        if production:
            from factor_engine.backend.production_signature import signature_for

            sig = signature_for(canon)
            if sig is not None and sig.default_status != "production":
                return False
            return node is not None
        return True
    if node is None:
        return False
    return supported_calls_match(node, supported)


def semantic_hash_stale(canon: str, *, polars_hash: str, duckdb_hash: str) -> bool:
    rec = operator_evidence_record(canon)
    if rec is None:
        return False
    expected_p = rec.get("implementation_hash_polars")
    expected_d = rec.get("implementation_hash_duckdb")
    if expected_p and expected_p != polars_hash:
        return True
    if expected_d and expected_d != duckdb_hash:
        return True
    return False


def compute_implementation_hash(source: str) -> str:
    """Return the legacy 16-hex implementation hash used by evidence records."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
