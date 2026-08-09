#!/usr/bin/env python3
"""Fast diagnostic: compute audit fixture gaps + policy failures WITHOUT
executing operator kernels.  Iteration aid for the unusable-operators sweep."""
from __future__ import annotations

import sys
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]
for p in (str(FE_ROOT.parent), str(FE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cleaned_operators import load_all
from cleaned_operators.production_hardening import factor_production_targets
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.operator_policy import infer_operator_policy
from cleaned_operators.semantic_certification import should_fail_closed
from scripts.audit_all_factor_production import (
    _build_call,
    _minute_source,
    _panels,
)


def main() -> int:
    load_all()
    targets = sorted(factor_production_targets())
    panels = _panels()
    fixture_gaps: list[str] = []
    policy_gaps: list[str] = []
    shape_gaps: list[str] = []
    ok: list[str] = []
    for canonical in targets:
        if should_fail_closed(canonical):
            continue
        operator = OperatorRegistry.get(canonical, "pandas_numpy")
        if operator is None:
            policy_gaps.append(f"{canonical}: no pandas reference")
            continue
        policy = infer_operator_policy(operator, canonical=canonical)
        if not policy.pit_safe or policy.lag < 0:
            policy_gaps.append(f"{canonical}: PIT policy not causal (pit_safe={policy.pit_safe}, lag={policy.lag})")
        if not policy.shape_preserving and not _minute_source(canonical):
            shape_gaps.append(f"{canonical}: shape_preserving=False")
        try:
            _build_call(canonical, operator, panels)
        except KeyError as exc:
            fixture_gaps.append(f"{canonical}: {exc}")
        except Exception as exc:
            fixture_gaps.append(f"{canonical}: {type(exc).__name__}: {exc}")
    print("=== FIXTURE GAPS (%d) ===" % len(fixture_gaps))
    for line in fixture_gaps:
        print("  " + line)
    print("=== POLICY GAPS (%d) ===" % len(policy_gaps))
    for line in policy_gaps:
        print("  " + line)
    print("=== SHAPE GAPS (%d) ===" % len(shape_gaps))
    for line in shape_gaps:
        print("  " + line)
    print("=== OK (%d) ===" % (len(targets) - len(fixture_gaps) - len(policy_gaps) - len(shape_gaps)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
