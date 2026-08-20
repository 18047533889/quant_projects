# Hard Gates Reference

**Version:** 1.0  
**Last Updated:** 2026-08-13  
**Status:** Production

## Overview

Hard gates are **binary pass/fail checks** that must all pass before code can be deployed to production. They verify correctness, safety, performance, and completeness across the Factor Engine. This document provides a comprehensive reference for all 85+ hard gates.

## Table of Contents

1. [Understanding Hard Gates](#understanding-hard-gates)
2. [Gate Categories](#gate-categories)
3. [Complete Gate Reference](#complete-gate-reference)
4. [Viewing Gate Status](#viewing-gate-status)
5. [Common Failures & Fixes](#common-failures--fixes)
6. [Adding New Gates](#adding-new-gates)

---

## Understanding Hard Gates

### What Are Hard Gates?

Hard gates are **automated quality checks** that:

- **Must all pass** before production deployment (no exceptions)
- **Are deterministic**: Same code always produces same result
- **Are executable**: Run as Python code, not manual checklists
- **Are fast**: Complete suite runs in <5 minutes
- **Are documented**: Each gate has clear pass criteria

### Gate Lifecycle

```
┌──────────────┐
│ Code Change  │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Run Gate Audit   │  pytest tests/ or python scripts/audit_rXX_hard_gates.py
└──────┬───────────┘
       │
       ▼
┌──────────────────┐    YES
│ All Gates Pass?  ├─────────► Deploy to Production
└──────┬───────────┘
       │ NO
       ▼
┌──────────────────┐
│  Fix & Re-test   │
└──────────────────┘
```

### Hard Gates vs Soft Checks

| Aspect | Hard Gate | Soft Check |
|--------|-----------|------------|
| Deployment blocker | YES ✅ | No |
| Pass/fail binary | YES ✅ | May have thresholds |
| Automated | YES ✅ | May be manual |
| Runtime | <5min total | Variable |
| Override allowed | NEVER | Sometimes |

---

## Gate Categories

### 1. Correctness Gates (25 gates)

Verify mathematical and semantic correctness:

- Operator output matches ground truth
- Backend parity (pandas = polars = duckdb)
- Numerical stability (no drift, overflow, precision loss)
- Semantic hash stability (cache invalidation control)

### 2. Safety Gates (30 gates)

Ensure data integrity and security:

- No path traversal vulnerabilities
- No credential leakage in logs
- Concurrency safety (no data races)
- Fail-closed behavior (errors don't produce silent wrong results)
- Boundary validation (all inputs checked)

### 3. Performance Gates (15 gates)

Maintain computational efficiency:

- Throughput SLAs (batch processing time)
- Memory bounds (no unbounded growth)
- Latency budgets (p99 response time)
- Resource governance (CPU/memory limits enforced)

### 4. Completeness Gates (15 gates)

Verify system completeness:

- All operators have tests
- All backends certified
- Documentation coverage
- Evidence artifacts present

---

## Complete Gate Reference

### R32: System Architecture Gates (50 gates)

**Audit Script:** `scripts/audit_r32_hard_gates.py`

#### Time System Gates

##### R32_CALENDAR_OUT_OF_COVERAGE_FAILS

**Purpose:** Calendar operations must fail gracefully when dates are out of range

**Pass Criteria:**
```python
cal = TradingCalendar(["2024-01-01", "2024-01-02", "2024-01-03"])
cal.offset("2024-01-01", -5)  # Goes before start
# Must raise CalendarCoverageError
```

**Why It Matters:** Prevents silent wrong results when computing factors near data boundaries

**Common Failure:** Calendar falls back to assumptions instead of failing

**Fix:**
```python
# Bad: Silent fallback
def offset(self, date, days):
    try:
        return self._dates[self._dates.index(date) + days]
    except IndexError:
        return date  # WRONG: Silent failure

# Good: Explicit error
def offset(self, date, days):
    try:
        return self._dates[self._dates.index(date) + days]
    except IndexError:
        raise CalendarCoverageError(f"Date {date} + {days} out of range")
```

---

##### R32_CALENDAR_PRODUCTION_FALLBACK_ZERO

**Purpose:** Production mode must never use default/fallback calendar

**Pass Criteria:**
```python
os.environ["QUANT_PRODUCTION_MODE"] = "1"
trading_day_offset("2024-01-01", 2)  # No calendar specified
# Must raise CalendarUnavailableError
```

**Why It Matters:** Production factors must use verified trading calendars, not assumptions

**Common Failure:** Code provides "reasonable default" calendar in production

**Fix:**
```python
# Bad: Fallback calendar
def get_calendar():
    return user_calendar or DEFAULT_CALENDAR  # WRONG

# Good: Fail if not provided
def get_calendar():
    if user_calendar is None:
        if is_production_mode():
            raise CalendarUnavailableError("Calendar required in production")
    return user_calendar or DEFAULT_CALENDAR
```

---

##### R32_INTRADAY_WINDOW_IS_SESSION_CLOCKED

**Purpose:** Intraday lookback windows must respect session boundaries

**Pass Criteria:**
```python
w = resolve_incremental_window_for_bar_freq(
    watermark_end="2024-01-05",
    lookback_bars=125,
    bar_freq="1min",
    market="ashare"
)
assert w["window_mode"] == "intraday_session_clock"
```

**Why It Matters:** Prevents data leakage across trading sessions (overnight gap)

---

##### R32_TIMEZONE_NAIVE_STRIP_ZERO

**Purpose:** Converting timezone-aware to naive must go through UTC

**Pass Criteria:**
```python
# Code must use: dt.tz_convert("UTC").tz_localize(None)
# Not: dt.tz_localize(None)
```

**Why It Matters:** Prevents accidental timezone shifts when stripping timezone info

---

#### CSE & DAG Gates

##### R32_CSE_ORPHAN_SHARED_ZERO

**Purpose:** Common Subexpression Elimination must not create orphaned shared nodes

**Pass Criteria:**
```python
roots, shared = apply_cse([plan1, plan2, plan3])
violations = verify_cse_dag(roots, shared)
assert not any("orphan" in v for v in violations)
```

**Why It Matters:** Orphaned nodes cause memory leaks and incorrect recomputation

**Common Failure:** Shared node removed from roots but still in shared dict

**Fix:** Ensure CSE traversal marks all reachable shared nodes

---

##### R32_CSE_DANGLING_REF_ZERO

**Purpose:** No plan_ref nodes pointing to non-existent shared nodes

**Pass Criteria:**
```python
violations = verify_cse_dag(roots, shared)
assert not any("dangling" in v for v in violations)
```

**Why It Matters:** Dangling refs cause runtime KeyError during execution

---

##### R32_CSE_TYPED_SEMANTIC_HASH_ONLY

**Purpose:** CSE hashing must use typed semantic hash, not repr fallback

**Pass Criteria:**
```python
# rolling_cse.py must contain "_typed_semantic_digest"
# Must NOT contain "default=repr"
```

**Why It Matters:** Repr-based hashing is fragile and breaks across Python versions

---

##### R32_MAX_PLAN_DEPTH_ENFORCED

**Purpose:** Plan depth must be bounded to prevent stack overflow

**Pass Criteria:**
```python
deep_plan = create_plan_with_depth(600)
assert_plan_depth_bounded(deep_plan, max_depth=512)
# Must raise PlanDepthLimitError
```

**Why It Matters:** Deeply nested plans cause stack overflow in recursive traversal

**Fix:** Use iterative traversal or reject plans exceeding depth limit

---

##### R32_GRAPH_TRAVERSAL_STACK_SAFE

**Purpose:** Graph traversal must use iterative algorithms (no recursion)

**Pass Criteria:**
```python
# cse._postorder must not contain "def visit" (recursive helper)
# Must use iterative stack-based traversal
```

**Why It Matters:** Large DAGs (1000+ nodes) cause RecursionError with recursive traversal

---

##### R32_CRITICAL_PATH_LINEAR_COMPLEXITY

**Purpose:** Critical path computation must be O(n), not exponential

**Pass Criteria:**
```python
dag = build_diamond_dag(1000)  # 1000-node diamond
time_start = time.time()
critical_path = dag.critical_path_remaining_ms("root", costs)
elapsed = time.time() - time_start
assert elapsed < 0.1  # Must be near-instant for 1000 nodes
```

**Why It Matters:** Exponential algorithms timeout on real DAGs (10K+ nodes)

**Fix:** Use dynamic programming with memoization

---

#### Catalog Gates

##### R32_SQLITE_FOREIGN_KEYS_ON

**Purpose:** SQLite must enforce foreign key constraints

**Pass Criteria:**
```python
cat = FactorCatalog(db_path)
fk_enabled = cat._conn.execute("PRAGMA foreign_keys;").fetchone()[0]
assert fk_enabled == 1
```

**Why It Matters:** Without FK enforcement, orphaned records corrupt the catalog

---

##### R32_CATALOG_TRANSACTION_INTERLEAVE_ZERO

**Purpose:** Catalog transactions must be serializable (BEGIN IMMEDIATE)

**Pass Criteria:**
```python
# transaction() method must use "BEGIN IMMEDIATE" and "COMMIT"
```

**Why It Matters:** BEGIN DEFERRED allows race conditions between readers/writers

---

##### R32_CATALOG_SCHEMA_VERSIONED

**Purpose:** Catalog schema must have version tracking

**Pass Criteria:**
```python
version = cat._conn.execute(
    "SELECT MAX(version) FROM catalog_schema_version"
).fetchone()[0]
assert version >= 1
```

**Why It Matters:** Schema migrations require version tracking to avoid corruption

---

##### R32_CATALOG_MIGRATION_RACE_PASS

**Purpose:** Schema migrations must be atomic (exclusive lock)

**Pass Criteria:**
```python
# _run_versioned_migration must use "BEGIN IMMEDIATE"
```

**Why It Matters:** Concurrent migrations corrupt the database

---

##### R32_PARTITION_KEY_NULL_ZERO

**Purpose:** partition_key column must be NOT NULL

**Pass Criteria:**
```python
schema = cat._conn.execute("PRAGMA table_info(factor_materialize_checkpoint)")
assert schema["partition_key"]["notnull"] == 1
```

**Why It Matters:** NULL partition keys break query optimization and indexing

---

##### R32_PRODUCTION_MODE_FAIL_OPEN_ZERO

**Purpose:** If production mode detection fails, must error (not assume research)

**Pass Criteria:**
```python
# Monkeypatch is_production_mode to raise RuntimeError
cat.register("factor_x", ...)
# Must raise ProductionModeResolutionError, not silently proceed
```

**Why It Matters:** False research mode in production bypasses safety checks

---

#### Job Queue Gates

##### R32_JOB_IDEMPOTENCY_CROSS_USER_COLLISION_ZERO

**Purpose:** Idempotency keys must be scoped to user (alice's job ≠ bob's job)

**Pass Criteria:**
```python
job1 = JobRecord(run_id="r1", owner="alice", idempotency_key="k1")
job2 = JobRecord(run_id="r2", owner="bob", idempotency_key="k1")
store.create(job1)
store.create(job2)
assert "r2" in store._jobs  # Both jobs exist
```

**Why It Matters:** Shared keys allow users to interfere with each other's jobs

---

##### R32_JOB_IDEMPOTENCY_CROSS_TYPE_COLLISION_ZERO

**Purpose:** Idempotency keys must be scoped to job type

**Pass Criteria:**
```python
job1 = JobRecord(owner="alice", job_type="compute", idempotency_key="k1")
job2 = JobRecord(owner="alice", job_type="materialize", idempotency_key="k1")
# Both must coexist
```

**Why It Matters:** Different job types have different retry semantics

---

##### R32_IDEMPOTENCY_REPLACE_ZERO

**Purpose:** Idempotent job creation must not replace existing jobs

**Pass Criteria:**
```python
# SQLite INSERT must use "ON CONFLICT ... DO NOTHING"
# Not "INSERT OR REPLACE"
```

**Why It Matters:** Replacing jobs destroys audit trail and interrupts running jobs

---

##### R32_RECONCILIATION_DURABLE

**Purpose:** Job reconciliation must persist to disk (write_manifest=True)

**Pass Criteria:**
```python
# _reconcile_stale_running must use write_manifest=True
```

**Why It Matters:** Non-durable reconciliation loses state on crash

---

##### R32_TERMINAL_STATE_OVERWRITE_ZERO

**Purpose:** Terminal job states (SUCCESS, FAILED) cannot be overwritten

**Pass Criteria:**
```python
job.status = JobStatus.SUCCEEDED
store.update(job)
job.status = JobStatus.RUNNING  # Try to overwrite
assert store.update(job) is False  # Must reject
```

**Why It Matters:** Overwriting terminal states corrupts job history

---

##### R32_QUEUE_DRAIN_PASS

**Purpose:** Queue drain must wait for all jobs to complete

**Pass Criteria:**
```python
queue.submit(job1)
queue.submit(job2)
queue.submit(job3)
queue.drain(timeout=5)
assert all_jobs_completed()
```

**Why It Matters:** Premature shutdown loses in-flight work

---

##### R32_UNEXPECTED_JOB_EXCEPTION_WORKER_SURVIVES

**Purpose:** Worker threads must survive job exceptions (no crash)

**Pass Criteria:**
```python
def boom_job():
    raise RuntimeError("Job failed")

queue.submit(boom_job)
time.sleep(0.5)
assert all(worker.is_alive() for worker in queue._workers)
```

**Why It Matters:** Worker crash prevents queue from processing remaining jobs

---

##### R32_QUEUE_ADMISSION_NONBLOCKING

**Purpose:** Job submission must not block (use put_nowait)

**Pass Criteria:**
```python
# submit() must use queue.put_nowait(), not queue.put()
```

**Why It Matters:** Blocking admission causes deadlock when queue is full

---

#### Materializer Gates

##### R32_MATERIALIZER_FLOAT64_HISTORY_DOWNCAST_ZERO

**Purpose:** Must not downcast float64 history to float32

**Pass Criteria:**
```python
# _upsert_partition must use np.promote_types, not astype("float32")
```

**Why It Matters:** Downcasting loses precision in historical data

---

##### R32_OUTPUT_GRAIN_SILENT_DROP_ZERO

**Purpose:** Must error if output grain exceeds max levels (not silently drop)

**Pass Criteria:**
```python
# _normalize_to_long_table must check OUTPUT_GRAIN_MAX_NLEVELS
```

**Why It Matters:** Silent drops corrupt materialized factors

---

##### R32_WIDE_LONG_WRITE_MODE_PARITY

**Purpose:** Wide and long write modes must support same write_mode options

**Pass Criteria:**
```python
# _upsert_partition_wide must handle write_mode parameter
```

**Why It Matters:** Inconsistent modes break round-trip guarantees

---

##### R32_FACTOR_LAKE_SCHEMA_SINGLE_TRUTH

**Purpose:** Factor lake metadata columns must be centrally defined

**Pass Criteria:**
```python
assert "resolved_snapshot_id" in factor_schema.FACTOR_METADATA_COLUMNS
assert "storage_precision_policy" in factor_schema.FACTOR_METADATA_COLUMNS
```

**Why It Matters:** Scattered schema definitions cause version skew

---

##### R32_CHECKPOINT_SUCCESS_REQUIRES_IDENTITY_DURABLE

**Purpose:** Checkpoint success requires identity sidecar persisted first

**Pass Criteria:**
```python
# _write_checkpoint_fingerprint_file must check production mode
# and raise if sidecar write fails
```

**Why It Matters:** Checkpoint without identity is unverifiable

---

##### R32_PRECISION_NAN_MASK_MISMATCH_ZERO

**Purpose:** Materialized factors must preserve exact NaN mask

**Pass Criteria:**
```python
live_result = compute_factor(...)
materialized = materialize_and_reload(live_result)
assert (live_result.isna() == materialized.isna()).all()
```

**Why It Matters:** NaN placement affects downstream computations

---

##### R32_PRECISION_INF_MASK_MISMATCH_ZERO

**Purpose:** Materialized factors must preserve exact Inf mask

**Pass Criteria:**
```python
assert (np.isinf(live) == np.isinf(materialized)).all()
```

**Why It Matters:** Inf/finite distinction has semantic meaning

---

##### R32_PRECISION_RANK_IS_CROSS_SECTIONAL

**Purpose:** Precision rank verification must be cross-sectional (per date)

**Pass Criteria:**
```python
# compare_live_vs_materialized must use .rank(axis=1)
```

**Why It Matters:** Time-series rank is meaningless for factor comparison

---

#### Identity & Lineage Gates

##### R32_FACTOR_ID_TRUNCATION_ZERO

**Purpose:** Factor IDs must not be silently truncated

**Pass Criteria:**
```python
# models.py must call validate_factor_id, not str(v)[:128]
```

**Why It Matters:** Truncation causes ID collisions

---

##### R32_FACTOR_ID_PATH_TRAVERSAL_ZERO

**Purpose:** Factor IDs must reject path traversal (../)

**Pass Criteria:**
```python
validate_factor_id("../evil")  # Must raise FactorIdError
```

**Why It Matters:** Path traversal allows reading/writing arbitrary files

**Fix:**
```python
def validate_factor_id(factor_id: str):
    if ".." in factor_id or "/" in factor_id:
        raise FactorIdError(f"Invalid factor ID: {factor_id}")
```

---

##### R32_DELETE_PATH_ESCAPE_ZERO

**Purpose:** Factor deletion must be confined to factor root

**Pass Criteria:**
```python
confine_path(root, os.path.join(root, "..", "evil"))
# Must raise FactorIdError
```

**Why It Matters:** Path escape allows deleting arbitrary files

---

##### R32_SOURCE_DSN_IDENTITY_COLLISION_ZERO

**Purpose:** Data source identity must ignore credentials, not connection params

**Pass Criteria:**
```python
hash1 = hash_data_source_config({"url": "postgres://u:pw1@h:5432/db"})
hash2 = hash_data_source_config({"url": "postgres://u:pw2@h:5432/db"})
hash3 = hash_data_source_config({"url": "postgres://u:pw1@other:5432/db"})
assert hash1 == hash2  # Same source, different credentials
assert hash1 != hash3  # Different host
```

**Why It Matters:** Credential changes shouldn't invalidate cache

---

##### R32_SOURCE_URL_SECRET_LEAK_ZERO

**Purpose:** Data source hashes must not contain credentials

**Pass Criteria:**
```python
h = hash_data_source_config({"url": "postgres://u:TOPSECRET@h:5432/db"})
assert "TOPSECRET" not in h
```

**Why It Matters:** Prevents credential leakage in logs and metadata

---

##### R32_FACTOR_IDENTITY_UNRELATED_OPERATOR_INVALIDATION_ZERO

**Purpose:** Factor identity must only depend on operators actually used

**Pass Criteria:**
```python
plan_a = make_plan("ts_mean", "close", 20)
plan_b = make_plan("ts_mean", "close", 20)  # Identical
plan_c = make_plan("ts_std", "close", 20)   # Different

hash_a = scoped_operator_contract_hash(plan_a)
hash_b = scoped_operator_contract_hash(plan_b)
hash_c = scoped_operator_contract_hash(plan_c)

assert hash_a == hash_b  # Identical plans
assert hash_a != hash_c  # Different operator
```

**Why It Matters:** Prevents unnecessary cache invalidation

---

##### R32_PRODUCTION_LINEAGE_FIELD_HASH_EMPTY_ZERO

**Purpose:** Production lineage must never have empty field hash

**Pass Criteria:**
```python
# lineage.py must not contain 'field_catalog_hash = ""'
```

**Why It Matters:** Empty hash prevents dependency tracking

---

##### R32_LINEAGE_ACTUAL_ENGINE_VERSION_PRESENT

**Purpose:** Lineage must capture actual engine version (git SHA, Python, numpy)

**Pass Criteria:**
```python
ev = build_engine_version()
assert "git_sha" in ev
assert "python" in ev
assert "numpy" in ev
```

**Why It Matters:** Version tracking required for reproducibility

---

#### Cache Gates

##### R32_PERSISTENT_CACHE_LOCK_TABLE_BOUNDED

**Purpose:** Cache lock table must have bounded size (prevent memory leak)

**Pass Criteria:**
```python
# cache.py must contain _SAVE_LOCK_MAX
```

**Why It Matters:** Unbounded lock table causes OOM

---

##### R32_PRODUCTION_UNKNOWN_CACHE_NAMESPACE_ZERO

**Purpose:** Production mode must fail if cache namespace is UNKNOWN

**Pass Criteria:**
```python
os.environ["QUANT_PRODUCTION_MODE"] = "1"
cache._operator_namespace = lambda: UNKNOWN_CACHE_NAMESPACE
cache._namespace_root()  # Must raise RuntimeError
```

**Why It Matters:** Unknown namespace causes cache pollution

---

#### Release & DR Gates

##### R32_FACTOR_LAKE_SCHEMA_VERSIONED

**Purpose:** Factor lake schema must have version number

**Pass Criteria:**
```python
assert hasattr(factor_schema, "FACTOR_LAKE_SCHEMA_VERSION")
assert int(factor_schema.FACTOR_LAKE_SCHEMA_VERSION) >= 1
```

**Why It Matters:** Schema evolution requires version tracking

---

##### R32_GENERATION_ROLLBACK_PASS

**Purpose:** Must support generation rollback (undo publish)

**Pass Criteria:**
```python
assert hasattr(lake_version, "rollback_factor_publish")
# or hasattr(lake_version, "generation_pointer")
```

**Why It Matters:** Rollback required for disaster recovery

---

##### R32_RELEASE_ENVIRONMENT_LOCKED

**Purpose:** Release must have locked dependencies (uv.lock or requirements.txt)

**Pass Criteria:**
```python
assert (Path("uv.lock").exists() or Path("requirements.txt").exists())
```

**Why It Matters:** Unlocked deps cause non-reproducible builds

---

##### R32_COLD_START_REMOVED_OPERATOR_REF_ZERO

**Purpose:** Cold-start code must not reference tombstoned operators

**Pass Criteria:**
```python
# Scan mining/*.py and factor_recipes/*.py
# Must not contain references to ALL_TOMBSTONED_NAMES
```

**Why It Matters:** Removed operators cause import errors

---

##### R32_DR_RESTORE_PASS

**Purpose:** Catalog must be restorable from backup

**Pass Criteria:**
```python
cat.register("f1", ...)
cat.close()
shutil.copy2(db_path, backup_path)
os.remove(db_path)
restored = FactorCatalog(backup_path)
ic = restored.catalog_integrity_check()
assert ic["quick_check"] == "ok"
assert restored.get_factor_info("f1") is not None
```

**Why It Matters:** DR requires verified backup/restore

---

#### Determinism Gates

##### R32_THREAD_COUNT_DETERMINISM_PASS

**Purpose:** Numerical operators must be deterministic regardless of thread count

**Pass Criteria:**
```python
result1 = compute_with_threads(1)
result2 = compute_with_threads(8)
assert np.array_equal(result1, result2, equal_nan=True)
```

**Why It Matters:** Thread-dependent results break reproducibility

---

### R35: Model Operator Gates (14 gates)

**Audit Script:** `scripts/generate_model_hard_gates.py`

These gates verify predictive model operators (GARCH, HAR, Kalman, etc.):

- Timing contracts (fit on t-1, predict for t)
- Lane assignments (MODEL_FIT vs MODEL_PREDICT)
- State management (no leakage across dates)
- Numba kernel parity (JIT matches reference)

See memory: `factor-engine-r35-model-final-numba-fe-da.md` for details.

---

### R36: Resource Governance Gates (46 gates)

**Audit Script:** `scripts/audit_r36_hard_gates.py`

These gates verify resource management:

- Memory calibration (PSI-based adaptive control)
- Host resource coordination (single authority)
- Buffer store governance (fail-closed on limits)
- Process family PSS accounting
- Per-shape P99 persistence

All 46 gates pass (0 failures). See memory: `factor-engine-r36-resource-autopilot-2026-08.md`.

---

### R38: Execution & Scheduling Gates (35 gates)

**Audit Script:** `scripts/audit_r38_hard_gates.py`

These gates verify execution reliability:

- AutoShard intersection semantics
- OOM replan (no same-shape retry)
- Calibration real observations only
- SpillStore checksum verification
- Dynamic sink + QoS lanes

Status: 35/1/0 (35 pass, 1 partial, 0 fail). See memory: `factor-engine-r38-r35-r36-remediation-2026-08.md`.

---

### R39: Performance & Concurrency Gates (12 gates)

**Audit Script:** `scripts/r39_hard_gates_audit.py`

These gates verify performance:

- TTDC (Time To Data Center) < 100s for 300×252×100 workload
- Batch transaction count ≤ 2 (reduced from 100)
- Streaming result sink timeout correctness
- Configuration default leak prevention

Status: 78 CLOSED, 1 PARTIAL, 5 NOT_CLOSED (deep architecture). See memory: `factor-engine-r39-closure-2026-08-11.md`.

---

## Viewing Gate Status

### Run All Gates for a Round

```bash
# R32 system architecture gates
python scripts/audit_r32_hard_gates.py

# Output (JSON):
{
  "R32_CALENDAR_OUT_OF_COVERAGE_FAILS": true,
  "R32_CALENDAR_PRODUCTION_FALLBACK_ZERO": true,
  ...
  "R32_HARD_BLOCKERS_ZERO": true  ← Overall result
}
```

### Run Gates via Pytest

```bash
# Run all hard gate tests
pytest tests/ -k "gate" -v

# Run specific round's gates
pytest tests/r32/ -v

# Run backend production gates
pytest tests/backend/test_production_gate.py
```

### Check Gate Status Programmatically

```python
from scripts.audit_r32_hard_gates import run as run_r32_gates

gates = run_r32_gates()
if not gates["R32_HARD_BLOCKERS_ZERO"]:
    failed = [k for k, v in gates.items() if not v]
    print(f"BLOCKING: {len(failed)} gates failed: {failed}")
else:
    print("✅ All R32 gates pass")
```

---

## Common Failures & Fixes

### Calendar Gates

**Symptom:** `R32_CALENDAR_OUT_OF_COVERAGE_FAILS = False`

**Cause:** Calendar returns fallback value instead of raising exception

**Fix:**
```python
# In storage/trading_calendar.py
def offset(self, date, n):
    try:
        idx = self._dates.index(date)
        new_idx = idx + n
        if new_idx < 0 or new_idx >= len(self._dates):
            raise CalendarCoverageError(
                f"Date {date} offset {n} out of range"
            )
        return self._dates[new_idx]
    except ValueError:
        raise CalendarCoverageError(f"Date {date} not in calendar")
```

---

### CSE Gates

**Symptom:** `R32_CSE_ORPHAN_SHARED_ZERO = False`

**Cause:** CSE creates shared nodes not reachable from any root

**Diagnosis:**
```python
from planner.cse import apply_cse, verify_cse_dag

roots, shared = apply_cse(plans)
violations = verify_cse_dag(roots, shared)
print(violations)  # Shows which nodes are orphaned
```

**Fix:** Ensure CSE marks all nodes reachable from roots:
```python
def apply_cse(plans):
    shared = {}
    roots = []
    
    # Step 1: Identify shared subexpressions
    counts = count_references(plans)
    
    # Step 2: Extract shared nodes
    for node_id, count in counts.items():
        if count > 1:
            shared[node_id] = node_id
    
    # Step 3: Rewrite roots to use plan_ref
    for plan in plans:
        roots.append(rewrite_with_refs(plan, shared))
    
    # Step 4: Remove unreachable shared nodes
    reachable = find_reachable_from_roots(roots)
    shared = {k: v for k, v in shared.items() if k in reachable}
    
    return roots, shared
```

---

### Identity Gates

**Symptom:** `R32_FACTOR_ID_PATH_TRAVERSAL_ZERO = False`

**Cause:** Factor ID validation doesn't check for `../`

**Fix:**
```python
# In security/factor_id.py
def validate_factor_id(factor_id: str):
    if not factor_id:
        raise FactorIdError("Factor ID cannot be empty")
    
    # Check for path traversal
    if ".." in factor_id:
        raise FactorIdError(f"Factor ID contains '..': {factor_id}")
    
    # Check for absolute paths
    if factor_id.startswith("/"):
        raise FactorIdError(f"Factor ID cannot be absolute path: {factor_id}")
    
    # Check for invalid characters
    if any(c in factor_id for c in ["\\", "\0"]):
        raise FactorIdError(f"Factor ID contains invalid characters: {factor_id}")
    
    return factor_id
```

---

### Job Queue Gates

**Symptom:** `R32_TERMINAL_STATE_OVERWRITE_ZERO = False`

**Cause:** update() method allows overwriting terminal states

**Fix:**
```python
# In service/jobstore.py
class JobStore:
    def update(self, job: JobRecord) -> bool:
        """Update job record. Returns False if update rejected."""
        existing = self.get(job.run_id)
        if existing is None:
            return False
        
        # Reject updates to terminal states
        if existing.status in (JobStatus.SUCCEEDED, JobStatus.FAILED, 
                               JobStatus.INTERRUPTED):
            return False  # Cannot overwrite terminal state
        
        self._jobs[job.run_id] = job
        self._persist(job)
        return True
```

---

### Materializer Gates

**Symptom:** `R32_PRECISION_NAN_MASK_MISMATCH_ZERO = False`

**Cause:** Parquet round-trip changes NaN representation

**Diagnosis:**
```python
live = compute_factor("my_factor")
materialized = materialize_and_reload(live)

nan_mismatch = (live.isna() != materialized.isna()).sum()
print(f"NaN mismatches: {nan_mismatch}")
```

**Fix:** Use PyArrow for NaN-preserving Parquet I/O:
```python
import pyarrow as pa
import pyarrow.parquet as pq

# Write with explicit schema
schema = pa.schema([
    ("date", pa.timestamp("ns")),
    ("instrument", pa.string()),
    ("value", pa.float64())  # Explicit float64, preserves NaN
])

table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
pq.write_table(table, path, compression="snappy")

# Read back
table = pq.read_table(path)
df = table.to_pandas()
```

---

### Cache Gates

**Symptom:** `R32_PRODUCTION_UNKNOWN_CACHE_NAMESPACE_ZERO = False`

**Cause:** Production mode allows UNKNOWN_CACHE_NAMESPACE

**Fix:**
```python
# In storage/cache.py
def _namespace_root(self):
    ns = _operator_namespace()
    
    if ns == UNKNOWN_CACHE_NAMESPACE:
        if is_production_mode():
            raise RuntimeError(
                "Cache namespace is UNKNOWN in production mode. "
                "This indicates operator registry is not properly initialized."
            )
    
    return os.path.join(self.root, ns)
```

---

## Adding New Gates

### When to Add a Gate

Add a hard gate when:

1. **Critical correctness requirement** that must never regress
2. **Safety boundary** that protects production data
3. **Performance SLA** that must be maintained
4. **Completeness check** that verifies system readiness

Do NOT add a gate for:
- Nice-to-have improvements
- Aspirational goals
- Metrics that vary by environment

### Gate Implementation Pattern

```python
# In scripts/audit_rXX_hard_gates.py

def _my_new_gate_group() -> None:
    """Check [category] correctness."""
    
    # GATE_NAME: Brief description
    try:
        # Setup test condition
        result = perform_check()
        
        # Verify expected behavior
        gates["R32_MY_NEW_GATE_PASS"] = bool(result.is_correct())
    except Exception as exc:
        # Gate fails on exception
        gates["R32_MY_NEW_GATE_PASS"] = False
        print(f"[FAIL] MY_NEW_GATE: {exc}")

# Add to run() function
def run() -> dict[str, bool]:
    for fn in (
        _calendar_gates,
        _cse_gates,
        _my_new_gate_group,  # ← Add here
        ...
    ):
        try:
            fn()
        except Exception as exc:
            gates[f"_GATE_GROUP_{fn.__name__}"] = False
            print(f"[EXC] {fn.__name__}: {exc!r}")
    
    all_true = all(gates.values())
    gates["R32_HARD_BLOCKERS_ZERO"] = bool(all_true)
    return gates
```

### Gate Naming Convention

```
R{ROUND}_{CATEGORY}_{SPECIFIC_CHECK}_{PASS|ZERO|TRUE|FALSE}
```

Examples:
- `R32_CALENDAR_PRODUCTION_FALLBACK_ZERO` - No fallback (zero occurrences)
- `R32_CSE_ORPHAN_SHARED_ZERO` - No orphaned nodes (zero count)
- `R32_DR_RESTORE_PASS` - DR restore works (pass)
- `R32_SQLITE_FOREIGN_KEYS_ON` - Foreign keys enabled (true)

### Gate Documentation Template

```markdown
##### R32_MY_NEW_GATE_NAME

**Purpose:** Brief one-sentence description

**Pass Criteria:**
```python
# Executable code showing what must be true
assert expected_behavior()
```

**Why It Matters:** Explain the consequences of failure

**Common Failure:** Typical ways this gate fails

**Fix:**
```python
# Code example showing correct implementation
```
```

---

## References

- [TESTING_STRATEGY.md](TESTING_STRATEGY.md) - Overall testing approach
- [BACKEND_SELECTION_GUIDE.md](BACKEND_SELECTION_GUIDE.md) - Backend-specific gates
- [COST_MODEL_EXPLAINED.md](COST_MODEL_EXPLAINED.md) - Performance gates
- `scripts/audit_*_hard_gates.py` - Gate audit scripts
- `tests/r*/test_*_hard_gates_*.py` - Gate test suites
