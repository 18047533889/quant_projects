# R2 Local Loop Engineering Resume State

**Updated:** 2026-08-16
**Mode:** ACTIVE, local-only
**Taskbook:** `FactorEngine_DataAccess_LoopEngineering_8H_Enterprise_Master_Taskbook_20260814_R2.md`
**Current HEAD:** `040a2f541d4637fddacf83846d758ed73d615fb5`

## Non-Negotiable Constraints

- Do not use GitHub, `gh`, remote fetch, remote push, or remote CI state.
- Work only under `/home/shw/quant_projects`.
- Total memory ceiling is 15 GiB across this session and the other Claude session.
- Default concurrency: one Writer plus one read-only Reviewer. Heavy tests are serial.
- Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` for tests.
- Never use `git checkout`, `git restore`, `git stash`, or `git clean`.
- Never bulk AST/regex rewrite `cleaned_operators/`.
- Do not edit `operator_catalog.py` or `operator_policy.py` concurrently.
- Never claim PASS or production-ready for an unrun gate.

## Current Wave: Identity Integration and Planner Authority

### Completed and Committed

1. `R2-P0-032/033` query-cache correctness and security identities use strict full SHA-256 digests.
2. Query parameters are canonicalized through the DataAccess read contract; unsupported values fail closed.
3. Serial verification passed: 25 focused/legacy tests plus DataAccess/FactorEngine import smoke.
4. Independent read-only review found no defect in the narrow identity delta; broad gates remain `NOT_RUN`.
5. Machine-readable evidence is at `evidence/r2/R2-P0-032-033-query-cache-identity.yaml`; implementation and oracle tests are committed at `040a2f541d4637fddacf83846d758ed73d615fb5`.

### Reproduced P0

1. `OPT2-P0-001`: production batch execution never calls `optimize_batch_global()` and never consumes `PhysicalRegionPlan`.
2. `runtime/batch_service.py` records per-root heuristic routes, then executes logical scheduler roots.
3. `HybridBackend.execute()` performs runtime rerouting and cannot be used as a physical-plan executor.
4. The first honest executable scope is one admitted region, zero transfer edges, and one fixed concrete backend; unsupported multi-region plans must fail closed.

### Active Work

- Query-cache identity manifest is bound to local commit `040a2f541d4637fddacf83846d758ed73d615fb5`; evidence/status records remain working-tree deltas.
- Next writer scope: add production admission and fixed-backend single-region physical execution oracles without claiming multi-region support.
- Independent reviewer: required after the runtime authority implementation passes focused serial tests.

## Next Priority Queue

1. Wire `BatchGlobalOptimizer` into production batch admission and execute only admitted single-region/no-transfer plans on a fixed backend.
2. Add fail-closed runtime oracles for non-ready plans, transfers, unsupported backends, and any attempt to reroute.
3. Independently review and locally integrate the Engine authority delta.
4. Continue DataAccess remote snapshot and PIT evidence.
5. Fix and certify remaining selectable Polars mathematics.
6. Continue Q PlanNode ABI, lowering, null semantics, fan-in, workspace, and resident-handle evidence.
7. Repair `ridge` dual-input path, positional alpha, and exception-to-all-NaN behavior.
8. Repair FactorAssets DataAccess adapter against the local public API.
9. Recalculate actual operator production surface from Registry + MiningRole + Evidence; do not reuse the unproven 678 count.
10. Wire local root gates; no remote/GitHub work.

## Scheduling

- No scheduling claim is made from stale task IDs in this file.
- Recurring work, if present in the active session scheduler, expires after 7 days by scheduler policy.

## Truth Status

- Q backend: `NOT_PRODUCTION_CERTIFIED`.
- Polars: `PARTIALLY_VERIFIED`, known formula/parity work remains.
- DuckDB: `PARTIALLY_VERIFIED`.
- FactorAssets: `NOT_PRODUCTION_CERTIFIED`.
- Modeling: `NOT_PRODUCTION_CERTIFIED` until one authority and runtime enforcement are independently proven.
- QuantEvaluator: `NOT_PRODUCTION_CERTIFIED` until shape and orientation regressions are independently verified.
- Root CI/current HEAD: `NOT_RUN` locally unless explicitly executed.
