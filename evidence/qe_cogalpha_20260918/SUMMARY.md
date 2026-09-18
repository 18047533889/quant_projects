# QuantEvaluator / CogAlpha review — 2026-09-18

Scope: server-c `/home/sunhaiwei/quant_projects/quant_evaluator`, directly on main; no branch, worktree, production deployment or fleet run.

Reference: user-supplied 13-page Evaluator_CogAlpha_指标口径审查_20260917(1).pdf. Its server-B mining.py/CogAlpha code is not present in this server-c tree. Document suggestions were assessed, not executed as instructions.

Implemented: shared default 20-pair daily IC policy for means/IR and CPU/GPU; strict unknown-weight turnover and aligned rank proxies; signed quantile rank monotonicity as a separate ID; version-aware evidence invalidation; full 164-ID formula reference and policy decisions for F01–F11. Membership turnover semantics/version remain unchanged.

Validation:
- before.log: three deliberately failing regression cases reproduced the IC/turnover defects before repairs.
- regression.log: intermediate run, NOT final acceptance; failures were repaired.
- final.log: 1266 passed, 2 skipped, 13 warnings, 131.13 seconds; includes available GPU tests.
- final.json marks SOURCE_CHANGED_DURING_RUN because another task changed DataAccess/FactorEngine files. Its explicit changed-file list contains no QuantEvaluator files. This is not claimed as a stable whole-repository receipt.
- After the full run, only a turnover docstring and the unchanged membership metric version were corrected; regenerated reference. Focused conventions/document/scoped-impact tests: 7 passed in 0.65 seconds.
- The two skipped tests require absent zscore/standardize and dropna/row-drop kernels; skips are not passes.
- Generated reference --check and git diff --check are required before commit.

This validates the named evaluator test scope, not external CogAlpha integration, production profitability, every backend, or an absence of all bugs. Historical artifacts and external gate thresholds were not rewritten.
