#!/usr/bin/env python3
"""Generate a fail-closed promotion matrix for research operators."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
FE = ROOT / "factor_engine"
if str(FE) not in sys.path:
    sys.path.insert(0, str(FE))

from cleaned_operators import load_all
from cleaned_operators.operator_policy import infer_operator_policy
from cleaned_operators.operator_surface import RESEARCH_ONLY_CANONICALS, classify_canonical
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.operator_spec import build_operator_spec
from stateful_contract import StatefulCheckpointRegistry

OUT = FE / "docs" / "research_operator_promotion_matrix.json"


def main() -> None:
    load_all()
    rows = []
    for canonical in sorted(RESEARCH_ONLY_CANONICALS):
        op = OperatorRegistry.get(canonical)
        policy = infer_operator_policy(op, canonical=canonical) if op else None
        spec = build_operator_spec(canonical)
        state = StatefulCheckpointRegistry.get(canonical)
        backends = OperatorRegistry.backends_for(canonical)
        has_all_backends = {"pandas_numpy", "polars", "sql"}.issubset(backends)
        reasons = []
        if not op:
            reasons.append("missing_pandas_runtime")
        if not has_all_backends:
            reasons.append("missing_one_or_more_backends")
        if not policy or not policy.pit_safe:
            reasons.append("missing_pit_contract")
        if spec is None or not spec.allow_in_production:
            reasons.append("production_policy_not_allowed")
        if state is not None and state.checkpoint_required_for_segmented and not state.checkpoint_fields:
            reasons.append("checkpoint_schema_missing")
        rows.append({
            "canonical": canonical,
            "surface": classify_canonical(canonical),
            "backends": backends,
            "pit_safe": bool(policy and policy.pit_safe),
            "semantic_version": getattr(state, "semantic_version", None),
            "checkpoint_contract": (
                {
                    "state_schema_version": state.state_schema_version,
                    "fields": list(state.checkpoint_fields),
                    "segmented_execution_supported": state.segmented_execution_supported,
                }
                if state else None
            ),
            "production_ready": not reasons,
            "blocking_reasons": reasons,
        })
    OUT.write_text(json.dumps({
        "schema_version": "research_operator_promotion_matrix.v1",
        "operator_count": len(rows),
        "operators": rows,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
