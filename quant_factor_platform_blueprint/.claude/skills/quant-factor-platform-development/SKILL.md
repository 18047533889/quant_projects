---
name: quant-factor-platform-development
description: Use for any development, migration, audit, optimization, or review work on the pre-model quant factor platform: DataAccess/FactorEngine integration, QuantEvaluator, FactorOptimizer, FactorAssets, FactorPreprocess, research ledger, legacy migration, tests, performance, and package isolation.
---

# Quant Factor Platform Development

Before editing code:

1. Treat the server working tree as the source of truth.
2. Read `CLAUDE.md` and `AI_GUIDE/00_START_HERE.md`.
3. Inspect DataAccess and FactorEngine before implementing any shared/data/factor-computation capability.
4. Search legacy code and classify reuse before writing replacements.
5. Do not rebuild PIT/storage/DSL/materialization/cache platforms already owned by DA/FE.
6. Preserve package independence and public-contract-only integration.
7. Use Reference -> Golden Test -> Fast Kernel for numerical code.
8. Use worktree-isolated agents for parallel edits where possible.
9. Third-party GPL source is reference/corpus by default, not code-copy material.
10. Do not mark work complete until independent audit and extraction tests pass.

Read supporting references when relevant:

- `references/architecture.md`
- `references/migration.md`
- `references/testing.md`
- `references/performance.md`
- `references/license.md`

For full implementation detail, read the matching `AI_GUIDE/*.md` spec before changing a package.
