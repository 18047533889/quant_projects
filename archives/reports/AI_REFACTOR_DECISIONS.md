# AI Refactor Decisions
**Created**: 2026-08-14  
**Session**: 6-Hour Autonomous Platform Refactor  
**Working Directory**: `/home/shw/quant_projects`

## Purpose

This document records architectural and implementation decisions made during the refactor campaign, with rationale and alternatives considered.

---

## Session Organization Decisions

### DECISION-001: Use Task Ledger + Findings + Decisions Structure
**Date**: 2026-08-14  
**Context**: Need to manage 112+ known issues plus discovered issues over 6-hour session  
**Decision**: Create three separate documents:
- Task Ledger: track all work items with status
- Findings: document problems with evidence
- Decisions: record choices and rationale

**Rationale**:
- Separation of concerns: what vs why vs status
- Easier for multiple agents to update different aspects
- Clear audit trail
- Prevents mixing discovery with tracking

**Alternatives Considered**:
- Single monolithic document: rejected, too hard to navigate
- Issues in code TODOs: rejected, not queryable
- External tracking system: rejected, keep local to repo

**Status**: Implemented

---

### DECISION-002: Prioritize Smoke Tests First
**Date**: 2026-08-14  
**Context**: Original assignment had smoke tests as P0; comprehensive plan adds 112+ tasks  
**Decision**: Execute 4 smoke tests before expanding task ledger to full 112+ items

**Rationale**:
- Smoke tests validate current packaging state
- Fast to execute (minutes not hours)
- Failures inform other architectural decisions
- Original requester priority signal

**Alternatives Considered**:
- Expand all tasks first: rejected, delays actionable work
- Skip smoke tests: rejected, violates P0 assignment

**Status**: In progress

---

### DECISION-003: Break FP-001 Immediately After Smoke Tests
**Date**: 2026-08-14  
**Context**: FP imports QE cache - severe boundary violation  
**Decision**: Make FP-001 the first implementation task after smoke tests

**Rationale**:
- Blocking issue for FP independence
- Affects package extraction validation
- Clear violation of stated architecture boundaries
- Relatively isolated change scope

**Alternatives Considered**:
- Wait for full task list: rejected, this blocks extraction
- Skip as "working currently": rejected, violates boundaries

**Status**: Queued

---

## Architectural Boundaries

### DECISION-004: Strict Package Independence
**Date**: 2026-08-14  
**Context**: Comprehensive plan defines clear boundaries: DA/FE/QE/FO/FA/FP  
**Decision**: Enforce strict dependency rules:
- QE: May depend on DA/FE, not on FO/FA/FP
- FO: May depend on DA/FE/QE (adapters), not on FA/FP
- FA: May depend on DA/FE/QE (adapters), not on FO/FP
- FP: May depend on DA/FE, not on QE/FO/FA

**Rationale**:
- Enables independent extraction
- Prevents circular dependencies
- Clear responsibilities
- Testable in isolation

**Alternatives Considered**:
- Allow cross-dependencies: rejected, creates coupling
- Require zero dependencies: rejected, would duplicate common code

**Status**: Target state, not yet enforced

---

### DECISION-005: DA/FE Capabilities Not Reimplemented
**Date**: 2026-08-14  
**Context**: Plan explicitly forbids duplicating DA/FE in new packages  
**Decision**: Continuous audit to prevent reimplementation of:
- Data lake / storage backend
- Parquet / DuckDB / COS access
- PIT / Calendar / Universe / Snapshot
- Factor DSL / Parser / AST / IR / Operator engine
- Factor materialization
- Universal cache platform
- Universal DAG scheduler

**Rationale**:
- Avoid code duplication
- Single source of truth
- Shared bug fixes and improvements
- Consistent behavior

**Alternatives Considered**:
- Allow "lightweight" reimplementation: rejected, scope creeps
- Vendor new capabilities into DA/FE: depends on use case

**Status**: Audit mechanism needed (CORE-001)

---

## Cache Strategy Decisions

### DECISION-006: Simplify QE Cache
**Date**: 2026-08-14  
**Context**: QE-013 finding - potentially over-engineered cache  
**Decision**: Default to request/runtime scoped cache; move L1/L2/Redis/compression to optional unless benchmarked

**Rationale**:
- YAGNI - no evidence advanced features needed
- Simpler = fewer bugs
- Can add back if benchmarks prove value
- Core use case: avoid recomputing within single evaluation

**Alternatives Considered**:
- Keep all features: rejected, no justification
- Remove all caching: rejected, valid within-request use case

**Status**: Task QE-013 created, not yet implemented

---

### DECISION-007: Remove FP Cache Dependency on QE
**Date**: 2026-08-14  
**Context**: FP-001 - direct import of QE cache  
**Decision**: Three options:
1. Implement minimal cache in FP if needed
2. Accept cache via dependency injection
3. Remove caching from FP core entirely

Will choose based on actual usage patterns found.

**Rationale**:
- Must break hard dependency
- May not need cache at all
- If needed, can be injected
- FP operations are often cheap (rank, zscore)

**Alternatives Considered**:
- Keep dependency: rejected, violates boundaries
- Create new shared cache package: rejected, over-engineering

**Status**: Task FP-001 created, approach TBD based on code inspection

---

## Testing Strategy Decisions

### DECISION-008: Test After Every Change
**Date**: 2026-08-14  
**Context**: Memory constraints (15 GiB), serial execution required  
**Decision**: Run relevant test subset after each change, full suite at integration waves

**Rationale**:
- Catch regressions immediately
- Cheaper to fix when fresh
- Know which change caused failure
- Memory limit prevents massive parallel test runs

**Alternatives Considered**:
- Only test at integration: rejected, too slow feedback
- Test everything always: rejected, exceeds memory

**Status**: Active practice

---

### DECISION-009: No xfail Without Justification
**Date**: 2026-08-14  
**Context**: Comprehensive plan section 28 - no "fake completion"  
**Decision**: Failing tests must be fixed or marked BLOCKED with external reason, not xfail'd

**Rationale**:
- xfail hides real problems
- Tests exist for a reason
- If test is wrong, fix or delete test
- If implementation is wrong, fix implementation

**Alternatives Considered**:
- Liberal xfail usage: rejected, defeats purpose of testing

**Status**: Active practice

---

## Split/Leakage Prevention Decisions

### DECISION-010: Test Data Sealed During Search
**Date**: 2026-08-14  
**Context**: FO-003, FO-004 - split contamination risk  
**Decision**: SearchRunner and related objects must not have access to test data, metrics, or shapes until after candidate freeze

**Rationale**:
- Fundamental research validity requirement
- Any test access during search = overfitting
- Type system should enforce if possible

**Alternatives Considered**:
- Trust developers not to peek: rejected, too easy to accidentally access
- Manual code review: rejected, insufficient guarantee

**Status**: Task FO-004 created, not yet implemented

---

### DECISION-011: fit_transform Pattern Forbidden for Fitted Transforms
**Date**: 2026-08-14  
**Context**: FP-006 - risk of fit-on-all-data  
**Decision**: Fitted transforms must use explicit two-phase API:
```python
transform.fit(train_data)
train_transformed = transform.transform(train_data)
val_transformed = transform.transform(val_data)
test_transformed = transform.transform(test_data)
```

**Rationale**:
- Makes train/test separation explicit
- Prevents accidental leakage
- Standard sklearn pattern
- Easy to audit

**Alternatives Considered**:
- Allow fit_transform with documentation: rejected, too easy to misuse
- Runtime checks: considered supplementary, not primary defense

**Status**: To be enforced in FP implementation

---

## Clustering Strategy Decisions

### DECISION-012: Leiden for Production, Hierarchical for Small Scale
**Date**: 2026-08-14  
**Context**: FA-013 - clustering algorithm choice  
**Decision**: Scale-aware clustering strategy:
- <20 factors: No formal clustering
- 20-100: Hierarchical optional
- 100-1k: Hierarchical or sparse graph
- 1k+: Fingerprint → ANN → exact similarity → sparse graph → Leiden

Use mature Leiden library (igraph/leidenalg), not approximation.

**Rationale**:
- Leiden is state-of-art for large-scale
- Small scale doesn't need complexity
- ANN is for neighbor finding, not clustering
- Reproducibility requires standard implementations

**Alternatives Considered**:
- Leiden everywhere: rejected, overkill for small scale
- Custom approximation: rejected, not reproducible
- HDBSCAN as primary: rejected, different use case

**Status**: Task FA-013 created, not yet implemented

---

### DECISION-013: Correlation Not Hard Deletion
**Date**: 2026-08-14  
**Context**: FA-020 - factor screening by correlation  
**Decision**: Correlation is cheap screening gate, not final decision. Consider multi-view:
- Signal rank correlation
- PnL correlation
- Top/bottom K Jaccard
- Residual IC
- Exposure cosine
- Horizon profile
- Regime profile

**Rationale**:
- High correlation ≠ redundancy if residual IC high
- Moderate correlation can still mean redundancy if PnL/tail/exposure identical
- Need multiple views for robust decision
- All thresholds configurable

**Alternatives Considered**:
- Single correlation threshold: rejected, too simplistic
- No correlation screening: rejected, need cheap filter

**Status**: To be implemented in FA

---

## Performance Strategy Decisions

### DECISION-014: Reference Then Optimize
**Date**: 2026-08-14  
**Context**: Comprehensive plan section 20 - performance work  
**Decision**: Workflow for any optimization:
1. Reference implementation (correct, readable)
2. Golden test suite
3. Profile to find bottlenecks
4. Fast kernel implementation
5. Parity tests (fast matches reference)
6. Benchmark (measure actual improvement)

**Rationale**:
- Premature optimization wastes time
- Need correctness baseline
- Parity prevents bugs in fast path
- Benchmark proves value

**Alternatives Considered**:
- Optimize first: rejected, no correctness anchor
- Never optimize: rejected, performance matters at scale

**Status**: Active methodology

---

## Package Extraction Strategy

### DECISION-015: Fresh Venv Validation Required
**Date**: 2026-08-14  
**Context**: Comprehensive plan section 21 - extraction  
**Decision**: Each package must validate in fresh venv without monorepo PYTHONPATH:
1. Copy to temp
2. Fresh venv
3. pip install wheel
4. Import core
5. Run tests
6. Validate extras separately

**Rationale**:
- Only way to prove true independence
- Catches hidden dependencies
- Validates wheel packaging
- Simulates end-user environment

**Alternatives Considered**:
- Trust import analysis: rejected, insufficient
- Test in monorepo: rejected, can't detect hidden deps

**Status**: Smoke tests implement this for 4 packages

---

## Memory Management Decisions

### DECISION-016: Max 2 Concurrent Subagents
**Date**: 2026-08-14  
**Context**: 15 GiB memory limit, BLAS/OpenMP constraints  
**Decision**: Maximum 2 subagents at once, all with:
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

**Rationale**:
- Platform memory ceiling
- BLAS libraries spawn many threads
- pytest can consume significant memory
- Better to complete 2 tasks than OOM on 6

**Alternatives Considered**:
- More parallelism: rejected, OOM risk
- Pure serial: rejected, underutilizes capacity

**Status**: Active constraint

---

## Audit Strategy Decisions

### DECISION-017: Periodic Audit Sweeps
**Date**: 2026-08-14  
**Context**: Comprehensive plan sections 24-25  
**Decision**: 
- Every 30-45 min: Audit sweep (tests, imports, TODO, duplicates, dead code, leakage, docs drift)
- Every 60-90 min: Integration wave (all packages, contracts, extraction sample, benchmarks)

**Rationale**:
- Continuous discovery of new issues
- Prevents regression accumulation
- Integration validates cross-package changes
- Regular cadence prevents long silent periods

**Alternatives Considered**:
- Only audit at end: rejected, accumulates too much debt
- Audit after every change: rejected, too expensive

**Status**: To be implemented

---

## Convergence Criteria

### DECISION-018: Quality-Based Completion, Not Time-Based
**Date**: 2026-08-14  
**Context**: Comprehensive plan section 27 - convergence  
**Decision**: Session succeeds when:
- All known P0-P3 DONE or legitimately BLOCKED
- Core tests green
- Integration tests green
- Extraction tests green
- No DA/FE duplication
- No test leakage
- No invalid chunk merge
- No cache false-hit
- Benchmark no catastrophic regression
- **Two consecutive audit sweeps with no new high/medium priority issues**

NOT when 6 hours elapse.

**Rationale**:
- Time is arbitrary
- Quality is measurable
- Two clean sweeps = genuine convergence
- Work should continue if issues remain

**Alternatives Considered**:
- Fixed time: rejected, ignores actual state
- Zero issues: rejected, unrealistic

**Status**: Target criteria

---

## Next Decision Points

- **FP-001 Approach**: Which of 3 options for removing QE cache dependency
- **QE Cache Simplification**: What features to keep vs remove
- **Test Baseline**: Pass/fail counts for current state
- **Agent Pool**: Which specialist agents to spawn first
- **Integration Test Scope**: What cross-package scenarios to validate

---

## Decision Change Log

- 2026-08-14: Initial decisions document created
- All decisions: Status = proposed, awaiting implementation validation
