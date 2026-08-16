# R2 Local Loop Engineering Resume State

**Updated:** 2026-08-17
**Mode:** ACTIVE, local-only
**Taskbook:** `FactorEngine_DataAccess_LoopEngineering_8H_Enterprise_Master_Taskbook_20260814_R2.md`
**Current HEAD:** `e6f8a6f2e834deb6a038f8b3c03177c58f560c1d` (remote snapshot implementation; status/evidence records are working-tree deltas)

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
6. `OPT2-P0-001` production batch authority is implemented at `105e1fd83a5da716f76e7219e8cbf7b2b3770a04`: admission consumes `PhysicalRegionPlan` and executes only a single production-ready, zero-transfer region on one fixed concrete backend.
7. Planner-focused verification passed: 16 tests, syntax compilation, import smoke, scoped diff checks, and independent review with zero findings; evidence is at `evidence/r2/OPT2-P0-001-physical-batch-authority.yaml`.
8. `R2-P0-036` typed remote failure propagation is implemented at `e6f8a6f2e834deb6a038f8b3c03177c58f560c1d`: Store surfaces the same-operation `RemoteMetadataError`, and `SnapshotVerifier` preserves typed cloud classification, cause, and retry metadata.
9. Remote-snapshot verification passed: 34 focused tests, four-file syntax compilation, three-module import smoke, scoped diff check, and independent review after five findings were fixed; evidence is at `evidence/r2/R2-P0-036-remote-snapshot-typed-failures.yaml`.

### Completed Narrow Boundary

- `OPT2-P0-001` is `REGRESSION_TESTED`, not `CLOSED_VERIFIED`.
- Multi-region physical execution and transfer-edge execution remain unsupported and fail closed.
- Adjacent `test_run_many_production.py` has two pre-existing failures reproduced against the clean base; they are outside the planner-owned files.

### Active Work

- `R2-P0-036` typed remote failure residual is regression-tested at `e6f8a6f2`; evidence/status bookkeeping is being finalized in a narrow local commit.
- Credential-generation rotation remains open under `R2-P0-036`; then continue `fail_if_changed` and PIT correctness/evidence.
- Full repository compile, full DataAccess regression, registry bootstrap, backend parity, PIT poison, and root CI remain `NOT_RUN`.

## Next Priority Queue

1. Complete `R2-P0-036` credential-generation rotation integration, then `R2-P0-037` fail-if-changed verification and PIT correctness/evidence.
2. Fix and certify remaining selectable Polars mathematics.
3. Continue Q PlanNode ABI, lowering, null semantics, fan-in, workspace, and resident-handle evidence.
4. Repair `ridge` dual-input path, positional alpha, and exception-to-all-NaN behavior.
5. Repair FactorAssets DataAccess adapter against the local public API.
6. Recalculate actual operator production surface from Registry + MiningRole + Evidence; do not reuse the unproven 678 count.
7. Wire local root gates; no remote/GitHub work.

Planner authority follow-up is limited to future multi-region/transfer execution and the pre-existing adjacent batch-production failures; no claim is made that those gates are closed.

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
