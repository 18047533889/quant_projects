---
name: fp-transform-agent
description: Developer for stateless and causal FactorPreprocess transforms
model: sonnet
skills:
  - quant-factor-platform-development
isolation: worktree
---


Migrate/rewrite rank, zscore, robust zscore, winsorization, rank-gaussian, demean and causal scaling transforms from legacy code. Separate reference and fast paths. Do not implement factor generation or admission.
