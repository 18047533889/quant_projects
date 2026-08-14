# R2 Local Loop Engineering Resume State

**Updated:** 2026-08-14
**Mode:** ACTIVE, local-only
**Taskbook:** `FactorEngine_DataAccess_LoopEngineering_8H_Enterprise_Master_Taskbook_20260814_R2.md`
**Base HEAD:** `d78ed761b7d098d27e3acf916f3961d75096b3b3`

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

## Current Wave: Repository Recovery

### Reproduced P0

1. `factor_engine/backend/q_backend/q_executor.py` has 51 lines at HEAD; its local preimage at `HEAD^` has 372 lines.
2. `factor_engine/backend/q_backend/q_backend.py` imports `QExecutionFallbackPolicy` and `get_q_executor`, which do not exist in the current executor.
3. Importing `backend.q_backend.q_backend` raises `ImportError` because `QExecutor` is missing.
4. The current executor contains hard-coded `PASS` constants but no executor implementation; these constants are not evidence.
5. Read-only audit confirmed additional source patterns: q backend assumes a nonexistent PlanNode API, and registered Polars rolling statistics include formulas based on time-varying historical means.

### Active Work

- Writer: recover Q executor/API integrity in isolated worktree.
- Reviewer slots are intentionally idle while Writer runs to stay inside memory budget; resume one reviewer after Writer finishes.

## Next Priority Queue

1. Independently review and integrate Q executor recovery.
2. Make Q production-safe/readiness use one evidence definition; no fake readiness messages.
3. Fix Q PlanNode ABI and reachable lowering registry.
4. Fix Q fan-in/workspace/resident handle lifecycle.
5. Fix registered Polars `ts_corr`, `ts_cov`, rolling regression, `ts_moment`, and `ts_kurt` formulas.
6. Repair `ridge` dual-input path, positional alpha, and exception-to-all-NaN behavior.
7. Repair FactorAssets DataAccess adapter against the local public API.
8. Fix QuantEvaluator batch/Numba bare `.squeeze()` for `T=1` and `N=1`.
9. Recalculate actual operator production surface from Registry + MiningRole + Evidence; do not reuse the unproven 678 count.
10. Remove DataAccess/FactorEngine correctness identity `repr`, `default=str`, and 64-bit truncation paths.
11. Replace BatchGlobalOptimizer scaffold constants and empty transfer edges with honest unsupported/fail-closed behavior or real planning.
12. Wire local root gates; no remote/GitHub work.

## Scheduling

- Durable local continuation job: `c0d3c644`, every 17 minutes.
- Durable one-shot final report job: `1f71defb`.
- Recurring tasks auto-expire after 7 days; the one-shot finalizer should stop new task assignment at the configured end.

## Truth Status

- Q backend: `NOT_PRODUCTION_CERTIFIED`.
- Polars: `PARTIALLY_VERIFIED`, known formula/parity work remains.
- DuckDB: `PARTIALLY_VERIFIED`.
- FactorAssets: `NOT_PRODUCTION_CERTIFIED`.
- Modeling: `NOT_PRODUCTION_CERTIFIED` until one authority and runtime enforcement are independently proven.
- QuantEvaluator: `NOT_PRODUCTION_CERTIFIED` until shape and orientation regressions are independently verified.
- Root CI/current HEAD: `NOT_RUN` locally unless explicitly executed.
