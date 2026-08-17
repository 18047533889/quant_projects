# R2 Findings Ledger

**Local implementation reviewed:** `2520532b755c6c80096877d137e45f62c5702030` (R2-P0-037 verified snapshot policy) plus identified bookkeeping deltas
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

### R2-P0-036 residual — typed remote snapshot failures

- Status: `REGRESSION_TESTED` at commit `e6f8a6f2e834deb6a038f8b3c03177c58f560c1d`.
- Production Store remote HEAD now propagates the `RemoteMetadataError` raised by the same operation; it no longer associates the request with a process-global last-error slot.
- `SnapshotVerifier` attaches a machine-readable `CloudErrorCode` to `SourceSnapshotUnavailable`, preserves the typed exception as `__cause__`, and retains retryability and `Retry-After` metadata.
- Strict absent/error metadata remains fail-closed; non-strict mode remains suppressive.
- Verification: 34 focused snapshot tests passed; four files syntax-compiled; three imports and scoped diff checks passed; independent review passed after five findings were fixed.
- Adjacent suite: 17 passed, 1 `CalendarUnavailableError` failure outside the owned files; it was not reproduced against the base, so no pre-existing attribution is claimed.
- Manifest: `evidence/r2/R2-P0-036-remote-snapshot-typed-failures.yaml`.
- Not run: full repository compile, full DataAccess regression, registry bootstrap, backend parity, PIT poison, and root CI.
- `CLOSED_VERIFIED` is not claimed because the broad gates remain unrun.

### R2-P0-036 residual — credential-generation material identity

- Status: `REGRESSION_TESTED` at commit `7795c41b670998447e26631a83198dabfcb3320d`.
- Remote metadata cache identity now binds explicit provider generation and strictly encoded resolved credential material; rotation cannot reuse stale entries even when provider generation is stale.
- Structured identity encoding prevents delimiter aliases while keeping credential material out of plaintext cache keys.
- Verification: 4 focused credential tests passed; two files syntax-compiled; import smoke and scoped diff checks passed; independent review found zero residual defects.
- Adjacent snapshot verification: 34 passed. The broader adjacent closure suite retains one unrelated `CalendarUnavailableError`; no pre-existing attribution is claimed.
- Manifest: `evidence/r2/R2-P0-036-credential-generation-material-identity.yaml`.
- Not run: full repository compile, full DataAccess regression, registry bootstrap, backend parity, PIT poison, and root CI.
- `CLOSED_VERIFIED` is not claimed because the broad gates remain unrun.

### R2-P0-037 — verified fail-if-changed snapshot authority

- Status: `REGRESSION_TESTED` at commit `2520532b755c6c80096877d137e45f62c5702030`.
- Explicit `verified_fail_if_changed` requires publisher `SourceManifest` authority, preserves publisher and local identity dimensions independently, and freezes the exact object set through terminal execution.
- Explicit verified policy forces strict pre/post verification even with a non-strict ambient verifier; missing or failed remote HEAD providers and incomplete identity metadata fail closed.
- PyArrow pin enforcement and authoritative-empty declared schemas, including `timestamp[ms, tz=UTC]`, are covered by real terminal-path oracles.
- Verification: 148 serial closure/adjacent snapshot tests passed; independent reviewer reran all 58 closure tests and returned CLEAN with zero residual findings; seven files syntax-compiled; DataAccess import smoke and scoped diff check passed.
- Runtime regression subset: 69 passed, 2 failed from unrelated dirty-tree/build-metadata conditions; neither failure is counted as passing evidence.
- Manifest: `evidence/r2/R2-P0-037-verified-fail-if-changed.yaml`.
- Not run: full repository compile, full DataAccess/FactorEngine regression, registry/duplicate-authority gates, backend parity, live COS/S3 integration, concurrent terminal stress, PIT poison, root/remote CI, and production certification.
- `CLOSED_VERIFIED` is not claimed because broad gates and live remote integration remain unrun; `R2-P0-039` remains `NOT_RUN`.

## Confirmed P0

### R2-REC-001 — Q executor recovery verified at current HEAD

- Status: `REGRESSION_TESTED`.
- The prior 51-line/truncation finding is stale: current `q_executor.py` exports `QExecutor`, `QExecutionFallbackPolicy`, and `get_q_executor` and imports successfully.
- Focused executor recovery tests pass; no executor source change was required in this wave.
- Remaining executor work is separate: telemetry still lacks `backend_switch_count` and `materialization_count`.

### Q2-P0-001..005 — Q capability evidence admission hardened

- Status: `INDEPENDENT_REVIEW`.
- Compiler lowerings populate the registry, but production admission now remains closed unless typed compile/runtime/parity artifacts validate against exact bytes, exact current Git SHA, timezone-aware generation time, implementation and parameter-domain hash, and applicable q/PyKX versions.
- Default compiler bootstrap has no validation context or evidence artifacts, so it certifies zero production operators.
- Current authority truth remains FAIL: 110 declared targets, 80 executable lowerings, 30 declaration/lowering disagreements, zero production-safe operators.
- Focused evidence, compatibility, and executor recovery verification: 72 passed. Legacy `test_q_backend.py`: 18 passed, 12 failed from stale or unsafe expectations and is not counted as passing evidence.
- Broad repository compile, q runtime parity/null semantics, PIT poison, and root CI remain `NOT_RUN`.

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
