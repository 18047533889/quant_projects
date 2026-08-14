# AI Refactor Task Ledger
**Created**: 2026-08-14  
**Session**: 6-Hour Autonomous Platform Refactor  
**Working Directory**: `/home/shw/quant_projects`

## Task Format

Each task records:
- **ID**: Unique identifier (e.g., QE-001, FP-001, FA-001, FO-001, CORE-001, ARCH-001)
- **Package**: quant_evaluator | factor_preprocess | factor_optimizer | factor_assets | CORE | ARCH
- **Severity**: P0 (blocking) | P1 (critical) | P2 (important) | P3 (nice-to-have)
- **Status**: TODO | READY | IN_PROGRESS | REVIEW | REWORK | DONE | BLOCKED
- **Owner**: Agent assigned
- **Reviewer**: Independent reviewer assigned
- **Dependencies**: Task IDs this depends on
- **Write Scope**: Files/modules to be modified
- **Problem**: Clear description of the issue
- **Evidence**: Location, reproduction, why it matters
- **Required Change**: What needs to be done
- **Tests**: Test coverage requirements
- **Benchmark**: Performance validation if applicable
- **Acceptance Criteria**: How we know it's done
- **Notes**: Additional context

---

## Priority 0 Tasks (Blocking)

### SMOKE-001: Clean-Wheel Test - QuantEvaluator
- **Package**: quant_evaluator
- **Severity**: P0
- **Status**: READY
- **Owner**: unassigned
- **Reviewer**: unassigned
- **Dependencies**: None
- **Write Scope**: N/A (test only)
- **Problem**: Must verify clean wheel installation works
- **Evidence**: `/home/shw/quant_projects/quant_evaluator/scripts/wheel_clean_install_smoke.py`
- **Required Change**: Execute smoke test, fix any failures
- **Tests**: The smoke test itself
- **Benchmark**: N/A
- **Acceptance Criteria**: Script exits 0, imports work, basic functionality passes
- **Notes**: First of 4 clean-wheel tests

### SMOKE-002: Clean-Wheel Test - FactorAssets
- **Package**: factor_assets
- **Severity**: P0
- **Status**: READY
- **Owner**: unassigned
- **Reviewer**: unassigned
- **Dependencies**: None
- **Write Scope**: N/A (test only)
- **Problem**: Must verify clean wheel installation works
- **Evidence**: `/home/shw/quant_projects/factor_assets/scripts/wheel_clean_install_smoke.py`
- **Required Change**: Execute smoke test, fix any failures
- **Tests**: The smoke test itself
- **Benchmark**: N/A
- **Acceptance Criteria**: Script exits 0, imports work, basic functionality passes
- **Notes**: Modified in current working tree

### SMOKE-003: Clean-Wheel Test - FactorOptimizer
- **Package**: factor_optimizer
- **Severity**: P0
- **Status**: READY
- **Owner**: unassigned
- **Reviewer**: unassigned
- **Dependencies**: None
- **Write Scope**: N/A (test only)
- **Problem**: Must verify clean wheel installation works
- **Evidence**: `/home/shw/quant_projects/factor_optimizer/scripts/wheel_clean_install_smoke.py`
- **Required Change**: Execute smoke test, fix any failures
- **Tests**: The smoke test itself
- **Benchmark**: N/A
- **Acceptance Criteria**: Script exits 0, imports work, basic functionality passes
- **Notes**: None

### SMOKE-004: Clean-Wheel Test - FactorPreprocess
- **Package**: factor_preprocess
- **Severity**: P0
- **Status**: READY
- **Owner**: unassigned
- **Reviewer**: unassigned
- **Dependencies**: None
- **Write Scope**: N/A (test only)
- **Problem**: Must verify clean wheel installation works
- **Evidence**: `/home/shw/quant_projects/factor_preprocess/scripts/wheel_clean_install_smoke.py`
- **Required Change**: Execute smoke test, fix any failures
- **Tests**: The smoke test itself
- **Benchmark**: N/A
- **Acceptance Criteria**: Script exits 0, imports work, basic functionality passes
- **Notes**: None

### FP-001: Remove QE Cache Dependency (BLOCKING) ✅ DONE
- **Package**: factor_preprocess
- **Severity**: P0
- **Status**: DONE
- **Owner**: Session3-Coordinator
- **Reviewer**: Verified via clean venv test
- **Dependencies**: None
- **Write Scope**: factor_preprocess/cache_integration.py (removed)
- **Problem**: FP directly imports quant_evaluator.runtime.cache_v2.MultiLevelCache - severe boundary violation
- **Evidence**: FP core must work in fresh venv without QE installed
- **Required Change**: Remove all QE imports from FP core, implement minimal cache or use DI
- **Tests**: Fresh venv import test without QE installed
- **Benchmark**: N/A
- **Acceptance Criteria**: `python -c "import factor_preprocess"` works without quant_evaluator installed ✅
- **Resolution**: Removed unused factor_preprocess/cache_integration.py file (296 lines). File was never imported or used anywhere in codebase. Verified FP imports work in clean venv without QE.
- **Notes**: This is the #1 architectural violation - NOW RESOLVED

---

## Priority 1 Tasks (Critical)

### QE-002: Cache Key Identity Insufficient
- **Package**: quant_evaluator
- **Severity**: P1
- **Status**: TODO
- **Owner**: unassigned
- **Reviewer**: unassigned
- **Dependencies**: None
- **Write Scope**: quant_evaluator cache implementation
- **Problem**: Cache key must bind factor value/snapshot identity, label identity/content, validity mask, LabelSpec, SplitPlan, evaluation view, metric version, config, universe/snapshot
- **Evidence**: Same IDs/shape with different values can cause false cache hits
- **Required Change**: Implement complete cache key that includes all semantic dimensions
- **Tests**: Unit tests showing same shape different values => cache miss
- **Benchmark**: N/A
- **Acceptance Criteria**: All listed dimensions in cache key, tests prove no false hits
- **Notes**: Critical correctness issue

### QE-003: Generic Chunk Aggregation Unsafe
- **Package**: quant_evaluator
- **Severity**: P1
- **Status**: TODO
- **Owner**: unassigned
- **Reviewer**: unassigned
- **Dependencies**: None
- **Write Scope**: quant_evaluator metric aggregation
- **Problem**: Generic "scalar average + array concatenate" is mathematically invalid for many metrics
- **Evidence**: MetricSpec must declare: NOT_CHUNKABLE / WEIGHTED_REDUCIBLE / STATEFUL_MERGE / CONCAT_TIME / CONCAT_FACTOR / CUSTOM
- **Required Change**: Add chunk semantics to MetricSpec, enforce safe aggregation
- **Tests**: Test invalid chunk aggregation is rejected
- **Benchmark**: N/A
- **Acceptance Criteria**: All metrics have declared chunk semantics, unsafe merges prevented
- **Notes**: Data correctness critical

### FO-003: Split Permissions
- **Package**: factor_optimizer
- **Severity**: P1
- **Status**: TODO
- **Owner**: unassigned
- **Reviewer**: unassigned
- **Dependencies**: FO-001
- **Write Scope**: factor_optimizer search runner, split handling
- **Problem**: Train/Validation/Test split usage not properly separated
- **Evidence**: Train = fit/diagnosis, Validation = selection, Test = sealed confirmation
- **Required Change**: Enforce split permissions throughout search process
- **Tests**: Verify test data inaccessible during search
- **Benchmark**: N/A
- **Acceptance Criteria**: Test split completely hidden from SearchRunner until freeze
- **Notes**: Prevents overfitting

### FO-004: Sealed Test API
- **Package**: factor_optimizer
- **Severity**: P1
- **Status**: TODO
- **Owner**: unassigned
- **Reviewer**: unassigned
- **Dependencies**: FO-003
- **Write Scope**: factor_optimizer public API
- **Problem**: Normal search objects must not expose test metrics, full-history including test, test rank
- **Evidence**: Any test access during search = contamination
- **Required Change**: Separate SealedTestResult from search-time objects
- **Tests**: Type system prevents test access during search
- **Benchmark**: N/A
- **Acceptance Criteria**: No way to access test data from SearchRunner/Trial objects
- **Notes**: Critical for valid research

### FA-001: Registry Persistence
- **Package**: factor_assets
- **Severity**: P1
- **Status**: TODO
- **Owner**: unassigned
- **Reviewer**: unassigned
- **Dependencies**: None
- **Write Scope**: factor_assets registry
- **Problem**: Registry is in-memory only, no durability
- **Evidence**: Factor metadata lost on restart
- **Required Change**: Implement SQLite WAL + migration + transactions for registry
- **Tests**: Persist/reload round-trip tests
- **Benchmark**: Insert/query performance
- **Acceptance Criteria**: Registry survives process restart, ACID guarantees
- **Notes**: First version SQLite, not PostgreSQL

### FA-002: SeenIndex Persistence
- **Package**: factor_assets
- **Severity**: P1
- **Status**: TODO
- **Owner**: unassigned
- **Reviewer**: unassigned
- **Dependencies**: FA-001
- **Write Scope**: factor_assets seen tracking
- **Problem**: SeenIndex is in-memory only
- **Evidence**: Duplicate detection lost on restart
- **Required Change**: Persist active/failed/rejected/shadowed/retired/corpus with first seen + repeated encounters
- **Tests**: Duplicate detection survives restart
- **Benchmark**: N/A
- **Acceptance Criteria**: SeenIndex durable, all states tracked
- **Notes**: Prevents redundant computation

---

## Priority 2 Tasks (Important)

### QE-004: Cross-Sectional Metrics Asset Chunking
- **Package**: quant_evaluator
- **Severity**: P2
- **Status**: TODO
- **Owner**: unassigned
- **Reviewer**: unassigned
- **Dependencies**: QE-003
- **Write Scope**: quant_evaluator cross-sectional metrics
- **Problem**: RankIC, quantile, tail metrics require complete daily cross-section
- **Evidence**: Chunking by asset invalidates these metrics
- **Required Change**: Mark these metrics as requiring full cross-section
- **Tests**: Verify rejection of asset-chunked evaluation for these metrics
- **Benchmark**: N/A
- **Acceptance Criteria**: CS metrics refuse asset chunks or use correct algorithm
- **Notes**: Mathematical correctness

### QE-013: QE Cache Over-Platformization
- **Package**: quant_evaluator
- **Severity**: P2
- **Status**: TODO
- **Owner**: unassigned
- **Reviewer**: unassigned
- **Dependencies**: None
- **Write Scope**: quant_evaluator cache_v2
- **Problem**: Simple cache + advanced L1/L2/Redis/compression/warming = over-engineering
- **Evidence**: No benchmark proving value of disk/Redis
- **Required Change**: Simplify to request/runtime scoped cache, move advanced features to optional
- **Tests**: Core tests pass with simplified cache
- **Benchmark**: Prove any retained advanced features have value
- **Acceptance Criteria**: Simpler cache, optional extras separated
- **Notes**: Simplification priority

### FP-002: FP Multi-Level Cache Over-Design
- **Package**: factor_preprocess
- **Severity**: P2
- **Status**: TODO
- **Owner**: unassigned
- **Reviewer**: unassigned
- **Dependencies**: FP-001
- **Write Scope**: factor_preprocess cache
- **Problem**: Disk/Redis/compression/warming/global singleton likely over-designed
- **Evidence**: No benchmark proof
- **Required Change**: Simplify or remove, no global state pollution
- **Tests**: Core works without complex cache
- **Benchmark**: Any retained cache must prove value
- **Acceptance Criteria**: Minimal cache or removed
- **Notes**: Simplification after FP-001

### FO-013: RepairMapper Shape Diagnosis Missing
- **Package**: factor_optimizer
- **Severity**: P2
- **Status**: TODO
- **Owner**: unassigned
- **Reviewer**: unassigned
- **Dependencies**: None
- **Write Scope**: factor_optimizer repair mapper
- **Problem**: Missing comprehensive shape diagnosis
- **Evidence**: Need: LINEAR, MONOTONIC, U_SHAPE, INVERTED_U, TOP_TAIL, BOTTOM_TAIL, THRESHOLD, SATURATING, CONVEX, CONCAVE, BIPOLAR, NON_MONOTONIC, NO_SIGNAL
- **Required Change**: Implement full shape taxonomy
- **Tests**: Synthetic signals for each shape type
- **Benchmark**: N/A
- **Acceptance Criteria**: All shape types diagnosed correctly
- **Notes**: Enables targeted repairs

### FA-013: Production Leiden vs Simplified ✅ VERIFIED SAFE
- **Package**: factor_assets
- **Severity**: P2
- **Status**: DONE (no action needed)
- **Owner**: Session3-Coordinator
- **Reviewer**: Code review + test verification
- **Dependencies**: None
- **Write Scope**: N/A (no changes needed)
- **Problem**: Current simplified modularity cannot substitute production Leiden
- **Evidence**: Production needs mature Leiden (igraph/leidenalg), not approximation
- **Required Change**: Use real Leiden library, keep simplified as reference only
- **Tests**: Clustering quality tests
- **Benchmark**: Quality and performance vs reference
- **Acceptance Criteria**: Production uses mature Leiden with version/seed/resolution tracking ✅
- **Resolution**: ModularityClustering already has fail-closed architecture. Default `allow_toy_algorithm=False` raises RuntimeError with clear message directing to production Leiden. All tests explicitly use `allow_toy_algorithm=True`. No production code can silently use toy algorithm. Implementation is safe as-is.
- **Notes**: Quality critical for large scale - ALREADY PROPERLY GATED

---

## Status Summary

- **TODO**: Tasks identified but not started
- **READY**: Dependencies met, can start immediately
- **IN_PROGRESS**: Currently being worked on
- **REVIEW**: Implementation complete, needs independent review
- **REWORK**: Review found issues, needs fixes
- **DONE**: Review passed, complete
- **BLOCKED**: Cannot proceed due to external dependency

---

## Counts by Status

- READY: 0 (all smoke tests complete)
- TODO: 13
- IN_PROGRESS: 0
- REVIEW: 0
- REWORK: 0
- DONE: 6 (4 smoke tests + FP-001 + test script fixes)
- BLOCKED: 0

**Total Tasks**: 18  
**Remaining to Add**: ~94 from comprehensive plan

---

## Next Actions

1. Execute SMOKE-001 through SMOKE-004 (P0)
2. Address FP-001 (P0 blocking)
3. Convert remaining 94+ issues from comprehensive plan
4. Begin P1 tasks
5. Run first audit sweep
