# -*- coding: utf-8 -*-
"""Attach checkpoint contracts to the final runtime catalog."""
from __future__ import annotations

from dataclasses import asdict

from cleaned_operators.registry import OperatorRegistry
from stateful_contract import StatefulCheckpointRegistry


def attach_final_stateful_metadata() -> None:
    for canonical, spec_payload in StatefulCheckpointRegistry.catalog().items():
        if canonical not in OperatorRegistry._operators:
            raise RuntimeError(
                f"stateful checkpoint contract references unavailable operator: {canonical}"
            )
        catalog = OperatorRegistry._catalog.setdefault(canonical, {})
        catalog["stateful"] = True
        catalog["segmented_execution_requires_checkpoint"] = bool(
            spec_payload["checkpoint_required_for_segmented"]
        )
        catalog["checkpoint_contract"] = dict(spec_payload)


__all__ = ["attach_final_stateful_metadata"]
