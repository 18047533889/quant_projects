# R2 Findings Ledger

**Local HEAD reviewed:** `105e1fd83a5da716f76e7219e8cbf7b2b3770a04` (planner implementation commit) plus identified working-tree evidence/status deltas
**Updated:** 2026-08-17
**Remote/GitHub evidence:** not used

## Current R2 Delta

### R2-P0-032/033 — query-cache identity hardening

- Status: `REGRESSION_TESTED`, committed at `040a2f541d4637fddacf83846d758ed73d615fb5`.
- Correctness and security identities now use strict full SHA-256 digests.
- Query parameters use the authoritative DataAccess read-contract canonicalizer; unsupported values and opaque `extra` values fail closed.
- Verification: 25 serial focused/legacy tests passed; import smoke passed; independent review ran 8 focused tests and found no delta defect.
- Manifest: `evidence/r2/R2-P0-032-033-query-cache-identity.yaml`.
- Not run: full repository compile, full DataAccess regression, registry bootstrap, backend parity, PIT poison, root CI.

### OPT2-P0-001 — planner is not production runtime authority

- Status: `REGRESSION_TESTED` at commit `105e1fd83a5da716f76e7219e8cbf7b2b3770a04`.
- Production batch admission now calls `optimize_batch_global()` and consumes its `PhysicalRegionPlan`.
- Admission is intentionally narrow and fail-closed: production-ready, exactly one region, zero transfer edges, valid topology, every logical batch root covered, and one fixed concrete backend.
- Shared CSE tasks and roots execute through that same fixed backend; production does not invoke `plan_batch_route()`, `record_batch_route()`, or `choose_plan_route()` after physical admission.
- Physical provenance records plan identity, planned/actual backend, region, switch count, materialization count, resident reuse, and Python-to-q bytes.
- Verification: 16 focused physical-region tests passed; syntax compilation, import smoke, and scoped diff checks passed; independent review found no delta defect.
- Adjacent `test_run_many_production.py`: 1 passed, 2 failed; failures were independently reproduced against the clean base, are pre-existing, and are outside the owned files.
- Manifest: `evidence/r2/OPT2-P0-001-physical-batch-authority.yaml`.
- Not run: full repository compile, full DataAccess regression, registry bootstrap, backend parity, PIT poison, root CI, multi-region physical execution, and transfer-edge execution.
- `CLOSED_VERIFIED` is not claimed; multi-region and transfer execution remain unsupported.

## Confirmed P0

### R2-REC-001 — Q executor deleted while consumers remain

- Status: `FIXING`
- Evidence: `q_executor.py` is 51 lines at HEAD vs 372 at `HEAD^`.
- Reproduction: `PYTHONPATH=factor_engine python3 -c 'import backend.q_backend.q_backend'` raises missing `QExecutor`.
- Consumers still import `QExecutor`, `QExecutionFallbackPolicy`, and `get_q_executor`.
- Hard-coded PASS strings at lines 48-51 are not executable evidence.

### Q2-P0-001..005 — Q capability authority and gates are fake/disconnected

- Status: `REPRODUCED`
- `compute_q_capability_evidence()` sets compile/runtime/parity false/TODO.
- `QPhysicalImplementationRegistry` starts empty and is not populated.
- Empty registry makes missing-evidence gate vacuous.
- `gate_q_capability_single_authority` contains a literal `pass` and returns success.
- Compiler admission still uses `_operator_map`, not evidence authority.

### MODEL2-P0-006 — Modeling split-brain not closed

- Status: `PARTIALLY_FIXED`
- Approved narrowly: `CrossSectionalScaler.transform(..., apply_start_time=None)` rejects; 21 focused tests passed.
- Rejected closure: both packages retain executable contracts and preprocessing logic.
- Gap enforcement misses train→test when no validation split, accepts negative `gap_days`, and does not enforce validation→test gap.

### ABI2-P0-001 — return_decomp hard-gate test has wrong oracle

- Status: `REJECTED`
- Focused suite: 13 passed, 1 failed.
- For preclose=100, open=105, close=110, overnight=5%, intraday=4.7619%, total=10% multiplicatively.
- Test incorrectly expects 15.5%; production formula must not be changed to satisfy it.

### QE2-P0-003 — singleton shape collapse

- Status: `FIXING`
- `assign_quantiles_batch(np.ones((1,1,1))).shape == ()` due to unrestricted `.squeeze()`.
- Expected documented shape retains T/N axes.

## Narrowly Approved

### QE2-P0-001 — top-bottom orientation

- Status: `LOCAL_TESTED`
- Focused orientation suite: 7 passed.
- Highest quantile minus lowest quantile is now default.
- Broader QE certification remains open because singleton shape ABI is broken.

## Additional Confirmed Source Patterns

- Q backend assumes legacy/nonexistent PlanNode attributes.
- Registered Polars rolling statistics include formulas based on time-varying historical means.
- FactorAssets adapter imports `dataaccess` while local public package/API is different; it uses unsupported `DataRequest(source/start_date/end_date)` and `date.today()` defaults.

## Truth Rule

No item is `CLOSED_VERIFIED` until targeted regression, import smoke, independent review, and current-local-HEAD integration proof pass.
