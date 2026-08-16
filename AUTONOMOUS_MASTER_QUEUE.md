# R2 Local Autonomous Master Queue

**Updated:** 2026-08-17
**Authority:** Local source at HEAD `7795c41b670998447e26631a83198dabfcb3320d` plus explicitly identified working-tree evidence/status deltas
**Taskbook:** `FactorEngine_DataAccess_LoopEngineering_8H_Enterprise_Master_Taskbook_20260814_R2.md`
**Remote/GitHub:** out of scope and forbidden

## Workflow

`DISCOVERED → REPRODUCED → FIXING → LOCAL_TESTED → INDEPENDENT_REVIEW → REGRESSION_TESTED → CLOSED_VERIFIED`

No task closes from a report, hard-coded constant, or documentation claim.

## Active

| ID | Priority | Status | Owner | Scope | Reproduction |
|---|---:|---|---|---|---|
| R2-P0-032/033 | P0 | REGRESSION_TESTED | local identity writer + independent reviewer | `dataaccess/read/query_cache.py`, focused identity tests | strict full SHA-256 correctness/security identities committed at `040a2f54`; 25 serial tests + import smoke passed post-commit; independent review found no delta defect; manifest at `evidence/r2/R2-P0-032-033-query-cache-identity.yaml` |
| OPT2-P0-001 | P0 | REGRESSION_TESTED | local planner authority writer + independent reviewer | `factor_engine/runtime/engine.py`, `factor_engine/runtime/batch_service.py`, planner/runtime tests | production batch admission now calls `optimize_batch_global()` and consumes only production-ready single-region, zero-transfer plans with all roots on one fixed concrete backend; commit `105e1fd8`; 16 focused tests passed; independent review found no delta defect; manifest at `evidence/r2/OPT2-P0-001-physical-batch-authority.yaml` |
| R2-P0-036-residual | P0 | REGRESSION_TESTED | local DataAccess writer + independent reviewer | typed remote failures plus credential-generation cache identity | request-local `RemoteMetadataError` propagation is committed at `e6f8a6f2`; structured credential material and provider generation jointly namespace remote metadata at `7795c41b`; 34 snapshot tests and 4 credential tests passed; both independent reviews passed; manifests at `evidence/r2/R2-P0-036-remote-snapshot-typed-failures.yaml` and `evidence/r2/R2-P0-036-credential-generation-material-identity.yaml` |

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
12. `R2-P0-036`: typed Store-to-verifier propagation and credential-generation material identity are regression-tested at `e6f8a6f2` and `7795c41b`; broad DataAccess gates remain `NOT_RUN`.
13. `OPT2-P0-001`: narrow production planner authority is implemented and regression-tested: one production-ready physical region, zero transfer edges, all roots covered, and one fixed concrete backend; multi-region execution remains unsupported. Adjacent `test_run_many_production.py` failures were independently reproduced on the clean base and remain pre-existing.
14. `CI2-P0-001`: root local gates not enforced.

## Closed Verified From Current Evidence

None for the current working-tree R2 delta. `R2-P0-032/033` is committed and `REGRESSION_TESTED`, but full DataAccess regression and root gates are `NOT_RUN`.

## Truth Matrix

- q/K: `NOT_PRODUCTION_CERTIFIED`
- Polars: `PARTIALLY_VERIFIED`
- DuckDB: `PARTIALLY_VERIFIED`
- FactorAssets: `NOT_PRODUCTION_CERTIFIED`
- Modeling: `NOT_PRODUCTION_CERTIFIED`
- QuantEvaluator: `NOT_PRODUCTION_CERTIFIED`
- Current-head root CI: `NOT_RUN`
