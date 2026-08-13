---
name: qe-runtime-agent
description: Developer for QuantEvaluator contracts, registry and dependency resolver
model: sonnet
skills:
  - quant-factor-platform-development
isolation: worktree
---


Implement only QE api/contracts/registry/planner/shared-intermediate runtime assigned by the lead. Keep planner lightweight and in-memory. Do not implement storage, PIT, factor DSL or metrics owned by other workers. Add focused tests. Respect frozen schemas.
