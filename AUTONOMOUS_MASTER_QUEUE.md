# R2 Local Autonomous Master Queue

**Updated:** 2026-08-14
**Authority:** Local source at HEAD `d78ed761b7d098d27e3acf916f3961d75096b3b3`
**Taskbook:** `FactorEngine_DataAccess_LoopEngineering_8H_Enterprise_Master_Taskbook_20260814_R2.md`
**Remote/GitHub:** out of scope and forbidden

## Workflow

`DISCOVERED → REPRODUCED → FIXING → LOCAL_TESTED → INDEPENDENT_REVIEW → REGRESSION_TESTED → CLOSED_VERIFIED`

No task closes from a report, hard-coded constant, or documentation claim.

## Active

| ID | Priority | Status | Owner | Scope | Reproduction |
|---|---:|---|---|---|---|
| R2-REC-001 | P0 | FIXING | Q executor writer | `q_executor.py`, `q_backend.py`, q tests | importing `backend.q_backend.q_backend` raises missing `QExecutor`; executor 51 lines vs 372 preimage |
| QE2-P0-003 | P0 | FIXING | QE shape writer | quantile NumPy/Numba paths | `(T,N,F)=(1,1,1)` collapses to scalar through `.squeeze()` |

## Independent Review Required

| ID | Priority | Status | Evidence Needed |
|---|---:|---|---|
| QE2-P0-001 | P0 | LOCAL_TESTED | verify orientation plus singleton shape and all callers |
| MODEL2-P0-006 | P0 | PARTIALLY_FIXED | runtime enforcement tests pass; authority routing still contradicted by imports |
| ABI2-P0-001 | P0 | REJECTED | one test has mathematically wrong 15.5% expectation; fix test oracle, not production formula |
| Q2-P0-001..005 | P0 | REJECTED | Q gates contain hard-coded/unconditional PASS and disconnected empty registry |

## Reproduced Queue

1. `Q2-P0-001..005`: one honest Q capability authority; compile/runtime/parity evidence mandatory.
2. `Q2-P0-006..014`: reachable lowering registry, no backend-owned defaults, q semantic parity.
3. `Q2-P0-020..022`: fan-in, workspace, connection, resident handle lifecycle.
4. `POL2-P0-002`: registered Polars ts_corr formula remains wrong in another implementation.
5. `POL2-P0-003`: ts_cov and rolling regression use time-varying historical means incorrectly.
6. `POL2-P0-004`: ts_moment and ts_kurt have the same rolling-mean-of-rolling-mean defect.
7. `RIDGE2-P0-001`: dual-input path returns NaN; positional alpha wrong; broad exception makes all-NaN output.
8. `FA2-P0-001..006`: FactorAssets imports wrong DataAccess package/API, uses `date.today()`, handle-only availability, fabricated catalog.
9. `OP2-P0-001`: recalculate real production surface from Registry + MiningRole + Evidence; discard unproven 678 claim.
10. `DA2-P0-001`: build identity has divergent versions and runtime git dependence.
11. `DA2-P0-002`: correctness identity still has repr/str and 64-bit truncation paths.
12. `OPT2-P0-001`: BatchGlobalOptimizer is scaffolded with empty transfers and fake estimates.
13. `CI2-P0-001`: root local gates not enforced.

## Closed Verified From Current Evidence

None for the R2 delta yet. Earlier focused fixes remain historical context but require current-HEAD regression evidence before R2 closure.

## Truth Matrix

- q/K: `NOT_PRODUCTION_CERTIFIED`
- Polars: `PARTIALLY_VERIFIED`
- DuckDB: `PARTIALLY_VERIFIED`
- FactorAssets: `NOT_PRODUCTION_CERTIFIED`
- Modeling: `NOT_PRODUCTION_CERTIFIED`
- QuantEvaluator: `NOT_PRODUCTION_CERTIFIED`
- Current-head root CI: `NOT_RUN`
