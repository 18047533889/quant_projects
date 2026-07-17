# -*- coding: utf-8 -*-
"""Attach checkpoint contracts to the final runtime catalog."""
from __future__ import annotations

from dataclasses import replace

from cleaned_operators.registry import OperatorRegistry
from stateful_contract import StatefulCheckpointRegistry


def attach_final_stateful_metadata() -> None:
    # A final backend replacement may raise the operator semantic version. The
    # checkpoint registry must follow it so pre-upgrade state cannot be reused.
    for canonical, spec in list(StatefulCheckpointRegistry._specs.items()):
        if canonical not in OperatorRegistry._operators:
            raise RuntimeError(
                f"stateful checkpoint contract references unavailable operator: {canonical}"
            )
        catalog = OperatorRegistry._catalog.setdefault(canonical, {})
        pandas_meta = dict((catalog.get("backend_meta") or {}).get("pandas_numpy") or {})
        runtime_version = str(pandas_meta.get("semantic_version") or spec.semantic_version)
        if runtime_version != spec.semantic_version:
            spec = replace(spec, semantic_version=runtime_version)
            StatefulCheckpointRegistry._specs[canonical] = spec

        payload = {
            "canonical": spec.canonical,
            "semantic_version": spec.semantic_version,
            "state_schema_version": spec.state_schema_version,
            "missing_policy": spec.missing_policy,
            "checkpoint_required_for_segmented": spec.checkpoint_required_for_segmented,
            "required_state_fields": list(spec.required_state_fields),
        }
        catalog["stateful"] = True
        catalog["segmented_execution_requires_checkpoint"] = bool(
            spec.checkpoint_required_for_segmented
        )
        catalog["checkpoint_contract"] = payload


__all__ = ["attach_final_stateful_metadata"]
