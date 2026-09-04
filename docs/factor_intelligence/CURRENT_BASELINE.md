# R61 Factor Intelligence & Auto Repair — CURRENT BASELINE

Audit date: 2026-09-04. Plan anchor: `FACTOR_INTELLIGENCE_AUTO_REPAIR_MASTER_IMPLEMENTATION_PLAN_20260904.md`.
Repository authority: LOCAL working tree at `/home/sunhaiwei/quant_projects` (git HEAD `2f3abe229` = R60, matches plan anchor exactly — no downgrade needed).

## Repository state

- Working tree is the implementation authority (R60). 7 package repos are **gitlinks in one root**? No — single repo `quant_projects`; all six packages are folders committed in-tree (only `riskfolio_qs` is a gitlink submodule). Local HEAD == R60 anchor.
- Unrelated dirty files in `alphaprobe/...` (V3 P0 workstream) exist in working tree; **do not touch**.
- `docs/factor_intelligence/` did not exist → created for R61.

## Environment

- Shared 32-core / 92GB machine, NVIDIA L20 (46GB VRAM). Other tenants active (zhangjiayin/xiangyurui cogalpha).
- `/home/sunhaiwei/quant_projects/.venv/bin/python` = Python 3.12.3, full deps incl. cupy 14.2.0. **Use this for everything** (system `/usr/bin/python3` has NO cupy).
- OMP_NUM_THREADS=31 for any parallel/GPU work (31-core hard cap).

## Package physical layout (critical for pytest/import)

| Package | Importable from repo root | Physical layout | Notes |
|---|---|---|---|
| data_access | yes (`data_access/__init__.py`) | flat | tests 1929 |
| factor_engine | yes | flat-ish (api/, cleaned_operators/, expr/, ir/, backend/…) | tests 23449 + 2 pre-existing collection errors |
| quant_evaluator | yes | flat (metrics/, kernels/gpu/, runtime/, contracts/, diagnosis/) | tests 502 |
| factor_preprocess | **only from its own dir** | NESTED `factor_preprocess/factor_preprocess/` | site-packages copy (2026-08-27) is STALE — tests must run with cwd=factor_preprocess so source wins |
| factor_optimizer | **only from its own dir** | NESTED `factor_optimizer/factor_optimizer/` | same staleness note; tests 591 |
| factor_assets | yes | flat (identity/, selection/, clustering/, similarity/, seen_index/, assembly/, library/, aggregation/, lifecycle/, novelty/, contracts/, optimizer/, graph/, registry/, campaigns/) | tests 1097 |

**Test command convention (baseline, recorded 2026-09-04):**
```
# data_access / quant_evaluator / factor_assets / factor_engine run from repo root:
/home/sunhaiwei/quant_projects/.venv/bin/python -m pytest -q data_access/tests --timeout=300
/home/sunhaiwei/quant_projects/.venv/bin/python -m pytest -q quant_evaluator/tests --timeout=300
/home/sunhaiwei/quant_projects/.venv/bin/python -m pytest -q factor_assets/tests --timeout=300

# factor_preprocess / factor_optimizer: cd into package dir first (source not installed as editable):
cd factor_preprocess && .venv/bin/python -m pytest -q tests --timeout=300
cd factor_optimizer && .venv/bin/python -m pytest -q tests --timeout=300
```

## Baseline test collection (not full run yet — collected 2026-09-04)

| Package | Collected | Errors | Notes |
|---|---|---|---|
| data_access | 1929 | 0 | |
| factor_engine | 23449 | 2 | pre-existing collection errors — see below |
| quant_evaluator | 502 | 0 | |
| factor_preprocess | 164 | 0 | |
| factor_optimizer | 591 | 0 | |
| factor_assets | 1097 | 0 | |

## Pre-existing factor_engine collection errors (documented, do NOT "fix" by thawing registry)

Both errors share one root cause: module-level `@register_operator` decoration on
`factor_engine/cleaned_operators/polars_native/cs_batch1.py` fires when the
production `OperatorRegistry` is already `frozen` — that module is loaded at
collection via a raw importlib path load in
`factor_engine/tests/operators/test_cs_polars_native_batch1.py:57` and via
`factor_engine/tests/test_weighted_cs_operators.py:35`, but it is **not** in the
P0-14 `_LATE_SURFACE_MODULES` staging whitelist in `factor_engine/tests/conftest.py`,
so it was never pre-registered during the building window before `load_all()` froze
the registry.

Erroring files:
1. `factor_engine/tests/operators/test_cs_polars_native_batch1.py` → `RuntimeError: operator registry is not writable: frozen`
2. `factor_engine/tests/test_weighted_cs_operators.py` → same root cause

These are R59/R60-era evidence-boundary artifacts, unrelated to R61 scope. They are
the documented clean-baseline for FE. If R61 touches `cleaned_operators/polars_native/`
or those two test files, re-verify the failure mode is unchanged; otherwise leave alone.
Registry lifecycle is production-governed (P0-14); tests must never thaw the frozen singleton.

## R61 capability gap decision summary (per-package — detail in CAPABILITY_GAP_MATRIX.md)

- **data_access**: ADD_NEW field-domain taxonomy metadata + FieldTaxonomyProvider (no existing typed semantic read API found under that name in first pass; `data_access/r30/` holds semantic specs/contracts to reuse).
- **factor_engine**: EXTEND existing api/factor_explain.py + identity/canonical.py + cleaned_operators registry metadata; ADD static-analysis manifest contracts (FieldUsage/OperatorUsage/FactorStaticAnalysisArtifact) only if no equivalent.
- **quant_evaluator**: EXTEND metrics registry + GPU infra; ADD EvidenceProfile named registry + missing shape/generalization/drawdown/exposure/statistical evidence only where absent.
- **factor_preprocess**: EXTEND grammar/presets + registry metadata (implementation_origin/fe_operator_id); resolve production_full ordering; ADD lineage-duplicate guard if missing.
- **factor_optimizer**: EXTEND treatment_decision.py / repair.py / treatment_result.py / adapters; ADD FactorFitnessSpec + typed missing-evidence semantics + conditional search + provider port.
- **factor_assets**: EXTEND selection/ + clustering/ + similarity/ + seen_index/ + library/; ADD profiling/ (taxonomy/health/grades/diagnosis) + admission profiles + variant-family selection + TTL/GC lifecycle only where absent.

_Note: this file is authoritative for BASELINE. Per-capability resolution lives in CAPABILITY_GAP_MATRIX.md which is regenerated as audits land._
