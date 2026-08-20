# R21 Production Runbook

R21-252..255. For each production incident class: symptom → error code → first
checks → safe remediation → prohibited actions → rollback.

---

## 1. PIT violation

- **Symptom**: job fails with `PIT_VIOLATION` / `PIT_VIOLATED_ROWS`.
- **Error code**: `PIT` family.
- **First checks**: source PIT mode; `pit.enforce`; whether a field's
  knowledge time was violated by the requested window.
- **Safe remediation**: widen the as-of rule; add a PIT-certified provider;
  rerun with `pit_forbid_forward_fill` off only for research.
- **Prohibited**: silently filling forward under production; disabling PIT on a
  production endpoint (R21-005 rejects it).
- **Rollback**: restore the previous certified field/source generation.

## 2. DQ failure

- **Symptom**: `OUTPUT_DOMAIN_VIOLATION` / `FACTOR_DEGENERATE` /
  `EXPECTED_SPARSE`.
- **First checks**: is the factor naturally sparse (event) or genuinely
  degenerate? Check the coverage profile by date (R21-102..105).
- **Safe remediation**: apply the correct role-aware DQ contract; adjust
  expected coverage per FieldSpec; do NOT mask.
- **Prohibited**: auto fill/clip/drop to pass DQ (R21-285).

## 3. Source stale / schema drift

- **Symptom**: `SOURCE_STALE` / `SOURCE_SCHEMA_DRIFT`.
- **First checks**: latest source timestamp vs expected trading session; schema
  version / partition schema; sidecar null-ratio semantics (R21-111).
- **Safe remediation**: republish the source; quarantine the drifted dataset;
  raise freshness SLA alert.

## 4. OOM

- **Symptom**: `RESOURCE_BUDGET_EXCEEDED`, process RSS near budget.
- **First checks**: cost estimate in the job manifest; peak RSS metric;
  materialization bytes; whether the SQL batch materializes the whole wide
  result before chunking (R21-186..188).
- **Safe remediation**: lower `FACTOR_ENGINE_SERVICE_MAX_CELLS` /
  `MAX_EXPECTED_MEMORY_MB`; stream/partition the materialization; reject at
  admission instead of OOM (R21-194).

## 5. Cache corruption

- **Symptom**: `CACHE` family errors, wrong cached subtree.
- **First checks**: cache scope; evidence/registry generation; cache cold/warm
  diff.
- **Safe remediation**: invalidate the per-scope cache; rebuild evidence.

## 6. SQL outage

- **Symptom**: `BACKEND_EXECUTION_FAILED` for DuckDB/ClickHouse.
- **First checks**: ClickHouse settings (max_execution_time / max_result_* /
  query_id); DuckDB store; egress allowlist (R21-021).
- **Safe remediation**: bounded retry + jitter for transient network failures
  (R21-168); do not retry validation/DQ/PIT (R21-167).

## 7. Materialization partial failure

- **Symptom**: `MATERIALIZE` family, partial partitions.
- **First checks**: staging vs publish; watermark; resume_materialize flag.
- **Safe remediation**: retry is idempotent (no double publish/watermark,
  R21-170); reconcile factor partitions vs catalog (R21-231).
- **Prohibited**: re-running a publish without an explicit approval boundary.

## 8. Job stuck

- **Symptom**: a `running` job with a stale heartbeat.
- **First checks**: heartbeat age; queue depth; worker count; `cancel` endpoint.
- **Safe remediation**: cancel; on restart the store marks non-terminal jobs
  `INTERRUPTED` (R21-073) — no phantom running.

## 9. Artifact / generation mismatch

- **Symptom**: `S29` readiness gate; evidence version differs from operator
  registry generation.
- **First checks**: evidence artifact generation; lock digest; build hash.
- **Safe remediation**: rebuild evidence/artifacts from the same commit;
  re-run the closure scorecard in a fresh process (R21-300).

---

## Ownership (R21-254..255)

| Module | Owner |
|---|---|
| data contract / fields / providers | fields / data_access |
| operator semantics / kernels | cleaned_operators |
| compiler / backend / cache | planner / backend / cache |
| runtime / storage / materialize | runtime / storage |
| HTTP service / security / jobs | service |
| deployment / CI / release | build / CI |

Every R21 audit artifact records its owning module so issues have an owner.
