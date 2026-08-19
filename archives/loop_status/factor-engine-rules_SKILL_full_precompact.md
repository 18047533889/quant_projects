---
name: factor-engine-rules
description: Mandatory governance for all agents working in FactorEngine
auto_trigger: true
skill_type: always-on
---

# FactorEngine Governance Rules

All agents working in this codebase MUST follow these rules:

## 1. Memory Discipline
- Never load full datasets in repair/audit agents
- Use syntax-only validation (`compile(text, path, 'exec')`) not runtime imports
- Real data testing only when user explicitly requests it
- Check `git status --porcelain | wc -l` before bulk operations — abort if >300 dirty files

## 2. Syntax Repair Protocol
- NEVER guess at missing denominators — that creates silent wrong answers
- When repairing division-guard corruption:
  1. Check for `.pre_lazy_opt` or `.backup` sibling first
  2. If clean pre-image exists, restore from it
  3. If no pre-image: mark site as MANUAL_REVIEW, don't auto-fix
  4. Verify fix compiles AND passes semantic spot-check
- Always write before/after diff evidence to `/tmp/<cluster>_repair_report.md`

## 3. Model Operators (HIGHEST PRIORITY)
- All `cleaned_operators/models/`, `fundamental/`, `research_models/` algorithms
- Check for:
  - Look-ahead bias (using future data in features)
  - Label leakage (target in training features)
  - Temporal leakage (shuffling panel data across time)
  - Overfitting (no train/val split, grid-search on full set)
  - Missing walk-forward validation
  - Static parameter evidence when dynamic calibration needed
- Fix rolling window training, proper OOS evaluation, PIT-safe feature extraction

## 4. Backend Truth
- Pandas is reference: `register_polars_bridge(canon)` is the production path
- DuckDB/SQL is opt-in optimization, fails closed when unavailable
- Q language backend not yet wired — do NOT implement until user confirms integration points

## 5. Concurrency
- Check dirty file count before starting — if another agent is mid-edit, wait
- Never overwrite files another agent just wrote
- Report via `/tmp/` evidence files, don't spam stdout

## 6. Testing
- Existing test suite in `tests/` must pass after changes
- For model operators: add PIT validation test showing no future data used
- For numerical fixes: add regression fixture with known-correct output

## 7. Performance
- Check for O(n²) loops in panel operators (nested instrument × time)
- Numba-eligible kernels should use `@jit(nopython=True, cache=True)`
- Never call `.values` in hot path when native array access available

## 8. Backend Architecture (NEW: 2026-08-13)

### Single Authority
- Only ONE `BackendCapabilityRegistry` — merge duplicates immediately
- Only ONE `RegionPlanner` — no parallel routers
- Only ONE semantic schema — all backends reference same canonical contract

### Backend Capability Classification
Never guess `ExecutionKind` from source inspection. Require explicit declaration:
- `PANDAS_REFERENCE` — authoritative NumPy/pandas implementation
- `POLARS_NATIVE_EXPR` — true Polars lazy expression (no Python loop)
- `POLARS_NUMPY_KERNEL` — Polars wrapping NumPy kernel
- `POLARS_PANDAS_DELEGATE` — Polars calling pandas via `.to_pandas()`
- `DUCKDB_NATIVE_SQL` — pure SQL execution
- `Q_NATIVE` — native q/kdb+ implementation

Do NOT call `POLARS_NATIVE` when code contains:
- `.to_pandas()`
- NumPy loop over Polars columns
- Python UDF applied row-wise

### Q Backend Integration
- Q canonicals MUST prove semantic parity before `PRODUCTION_SAFE`
- Q `mavg`/`mdev`/`cor` have different null/warmup semantics than pandas — verify explicitly
- Q Regions must maintain **backend residency**: no `Python→q→Python→q` pingpong
- Use `BackendResidentHandle` / `QResidentTableHandle` for cross-region values
- Hard gate: `Q_OPERATOR_PINGPONG_ZERO = PASS`

### Polars Threading
- Do NOT use `pl.Config.set_thread_count()` — API does not support runtime changes
- `POLARS_MAX_THREADS` must be set before `import polars`
- `ResourceBroker` controls **count of concurrent Polars jobs**, not thread pool size
- For different thread counts: use separate worker processes with `spawn`

### Region Execution
- `PhysicalRegionPlan` is the ONLY execution contract
- Executor must not re-route backends at runtime
- Cross-region transfers only at explicit `TransferEdge`
- Telemetry MUST track:
  - `python_to_q_bytes`
  - `backend_switch_count`
  - `materialization_count`
  - `resident_reuse_count`

## 9. Polars Native Module (NEW)

### Do NOT force-load `polars_native/` until:
1. All files parse
2. All classes import without `AttributeError`
3. ABI matches `__init__.py` (class names vs import names)
4. `OperatorMetadata` field contract matches `base_polars.OperatorMetadata`
5. No placeholder/TODO canonicals registered as production-ready
6. Semantic parity certified

### Stub/Placeholder Policy
If canonical is registered but implementation is:
```python
# TODO: Implement proper X
# Placeholder: using Y instead
```
Then:
- Either implement真实 X + certify parity
- Or mark `RESEARCH_ONLY` / `NOT_IMPLEMENTED`
- NEVER leave proxy under production canonical name

Examples that must be fixed or quarantined:
- `ts_markov_committor` → currently `rolling_mean`
- `ts_local_lyapunov_exponent` → currently `volatility_change`
- `ts_km_equilibrium_distance` → currently `distance_from_rolling_mean`

## 10. Evidence & Gates (NEW)

### No Vacuous Tests
Every test MUST:
- Instantiate the object under test
- Execute the code path being tested
- Assert expected behavior OR catch expected exception
- Use `monkeypatch.setenv`/`delenv`, not manual `os.environ` pollution

Forbidden patterns:
```python
def test_x():
    # Would raise if...
    pass

def test_y():
    # Can't easily test without real source
    pass
```

### Evidence Binding
All evidence files MUST record:
- `git_sha` of HEAD when generated
- `generation_timestamp`
- Evidence becomes STALE if `git_sha != current HEAD`

Before claiming `PRODUCTION_CERTIFIED`:
1. Compile entire repo
2. Import all production modules
3. Registry bootstrap with no duplicates
4. Backend parity (Pandas ↔ Polars ↔ SQL ↔ q where supported)
5. PIT poison tests pass
6. CI passes on actual merged HEAD

### Hard Gates for Final HEAD
```python
FULL_REPO_COMPILE = PASS
REGISTRY_BOOTSTRAP = PASS
REGISTRY_DUPLICATE_AUTHORITY = PASS  # Only ONE of each authority
OPERATOR_ABI = PASS
NO_PRODUCTION_PLACEHOLDER = PASS

PANDAS_REFERENCE = PASS
POLARS_PARITY = PASS
POLARS_NO_FAKE_NATIVE = PASS

Q_CANONICAL_IR_ONLY = PASS  # Q has zero semantic authority
Q_OPERATOR_PINGPONG_ZERO = PASS
Q_NULL_TIME_PARITY = PASS

BACKEND_CAPABILITY_SINGLE_AUTHORITY = PASS
REGION_PLANNER_SINGLE_AUTHORITY = PASS

MODEL_EVIDENCE_CURRENT_HEAD = PASS
CI_FINAL_HEAD = PASS
```

If gate not executed: status = `NOT_RUN`, NEVER `PASS`

## 11. Multi-Agent Coordination (NEW)

### Allowed Parallelism
- Multiple agents CAN work simultaneously on non-overlapping file sets
- Each agent MUST declare `owned_files` before starting
- Orchestrator MUST reject if `modified_files ⊄ allowed_files`

### Merge Waves
Prefer staged integration:
1. **Wave 1**: Leaf implementation fixes (non-overlapping operator families)
2. **Wave 2**: Shared contract consolidation (authority merges)
3. **Wave 3**: Region planner + residency + resource fixes
4. **Wave 4**: Parity + PIT + E2E + stress tests
5. **Wave 5**: Evidence + docs

Each wave merges, then runs integration gates before next wave.

### Agent Deliverable
Every agent commit MUST include machine-readable manifest:
```yaml
task_id: syntax-fix-batch-3
base_sha: 5242d347
commit_sha: <actual>
owned_files: [cleaned_operators/common/polars_ts_stats.py, ...]
modified_files: [...]
canonicals_touched: [ts_recovery_factor, ts_drawdown_duration, ...]
backends_touched: [polars]
tests_run: 47
tests_passed: 47
tests_not_run: []
known_limitations: ["Polars thread control not yet production-safe"]
evidence_generated: [/tmp/batch3_parity.json]
```

### Absolute Prohibitions for Concurrent Agents
1. `git checkout` / `git restore` / `git stash` / `git clean` — these destroy other agents' work
2. Editing `operator_catalog.py` / `operator_policy.py` — single-writer only
3. Regenerating evidence while others are modifying canonicals
4. Bulk text-replace on formulas (unless you are solo and have explicit permission)

All agents: read this skill on spawn, report rule violations immediately, stop if uncertain.
