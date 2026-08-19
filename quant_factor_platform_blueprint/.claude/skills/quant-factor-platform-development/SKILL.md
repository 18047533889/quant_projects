---
name: quant-factor-platform-development
description: Use for development/migration/audit on pre-model packages (QE/FO/FA/FP) and DA/FE integration. Keep prompts small.
---

# Quant Factor Platform Development

Before editing:

1. Working tree is truth.
2. Read repo `CLAUDE.md` + this skill; **do not** paste entire `AI_GUIDE/` into context.
3. Open only the `AI_GUIDE/*.md` file for the package you change.
4. Search DA/FE/legacy before rebuilding shared capabilities.
5. No second PIT/storage/DSL/materialization/cache/DAG platform.
6. Public-contract-only cross-package imports; independent installable packages.
7. Reference → golden test → fast kernel for numerical work.
8. GPL third-party = reference/corpus by default.
9. Done only after independent audit + extraction tests where required.

Optional references (open on demand, not wholesale):

- `references/architecture.md`
- `references/migration.md`
- `references/testing.md`
- `references/performance.md`
- `references/license.md`
