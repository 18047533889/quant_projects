---
name: verify
description: Verify FactorEngine planner changes through the public planner package boundary.
---

# FactorEngine Runtime Verification

Run from repo `factor_engine/` with `PYTHONPATH` set to that directory (do not hardcode a home path).

- For planner/compiler changes, execute a representative plan through public `planner` exports and `Optimizer.compile()`.
- Capture optimized output, stage, pass names, equivalence declarations, invariant results, and legacy API compatibility.
- Probe fail-closed behavior with one invariant violation and one production numeric-policy violation.
- Keep verification processes serial; do not run pytest in this skill.
