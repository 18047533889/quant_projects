# AI Refactor Findings
**Created**: 2026-08-14  
**Session**: 6-Hour Autonomous Platform Refactor  
**Working Directory**: `/home/shw/quant_projects`

## Purpose

This document records all problems discovered during the refactor campaign, with evidence and impact analysis. Each finding is assigned a task ID in the Task Ledger.

---

## Critical Architectural Violations

### FP-001: FactorPreprocess Imports QuantEvaluator Cache
**Severity**: P0 (Blocking)  
**Package**: factor_preprocess  
**Discovery**: Comprehensive plan section 7  
**Evidence**: `factor_preprocess` directly imports `quant_evaluator.runtime.cache_v2.MultiLevelCache`  
**Impact**: FP cannot be used independently, violates package boundaries, breaks extraction  
**Root Cause**: Cache functionality added without considering dependency architecture  
**Task**: FP-001

### Boundary: No DA/FE Duplication Enforcement
**Severity**: P1  
**Package**: CORE  
**Discovery**: Comprehensive plan sections 1, 10  
**Evidence**: No automated checks prevent reimplementing DA/FE capabilities in QE/FO/FA/FP  
**Impact**: Risk of duplicated PIT logic, data lake access, factor DSL, materialization  
**Root Cause**: No continuous boundary auditing  
**Task**: CORE-001

---

## QuantEvaluator Findings

### Cache Identity Insufficient
**Severity**: P1  
**Package**: quant_evaluator  
**Discovery**: Comprehensive plan QE-002  
**Evidence**: Cache keys may not bind all semantic dimensions (factor value identity, label identity, mask, split, view, metric version, config, universe)  
**Impact**: False cache hits possible → wrong evaluation results  
**Example**: Same factor IDs/shape but different values could cache-hit  
**Task**: QE-002

### Generic Chunk Aggregation
**Severity**: P1  
**Package**: quant_evaluator  
**Discovery**: Comprehensive plan QE-003  
**Evidence**: May use "scalar average + array concat" without metric-specific semantics  
**Impact**: Mathematically invalid aggregation for many metrics  
**Example**: RankIC cannot be averaged across chunks without sufficient statistics  
**Task**: QE-003

### Cross-Sectional Metric Chunking
**Severity**: P2  
**Package**: quant_evaluator  
**Discovery**: Comprehensive plan QE-004  
**Evidence**: RankIC, quantile, tail metrics may accept asset-axis chunks  
**Impact**: Wrong results if incomplete cross-section used  
**Example**: Percentile ranks computed on subset of assets  
**Task**: QE-004

### Streaming Coordinate Ambiguity
**Severity**: P2  
**Package**: quant_evaluator  
**Discovery**: Comprehensive plan QE-005  
**Evidence**: Chunks may rely on shape heuristic instead of explicit time/asset/factor coordinates  
**Impact**: Misinterpretation of chunk semantics  
**Task**: To be added

### Cache Over-Platformization
**Severity**: P2  
**Package**: quant_evaluator  
**Discovery**: Comprehensive plan QE-013  
**Evidence**: Potentially complex cache with L1/L2/Redis/compression/warming without benchmarks  
**Impact**: Unnecessary complexity, maintenance burden  
**Task**: QE-013

---

## FactorOptimizer Findings

### External Split Not Primary
**Severity**: P1  
**Package**: factor_optimizer  
**Discovery**: Comprehensive plan FO-001  
**Evidence**: May recreate splits instead of accepting external split plan  
**Impact**: Wastes data, inconsistent with upstream research  
**Task**: To be added

### Split Permission Violations
**Severity**: P1  
**Package**: factor_optimizer  
**Discovery**: Comprehensive plan FO-003  
**Evidence**: Train/Validation/Test usage not strictly separated  
**Impact**: Test data leakage, invalid research  
**Task**: FO-003

### Unsealed Test API
**Severity**: P1  
**Package**: factor_optimizer  
**Discovery**: Comprehensive plan FO-004  
**Evidence**: Search objects may expose test metrics/data  
**Impact**: Contamination risk during hyperparameter search  
**Task**: FO-004

### RepairMapper Shape Diagnosis Missing
**Severity**: P2  
**Package**: factor_optimizer  
**Discovery**: Comprehensive plan FO-013  
**Evidence**: Incomplete shape taxonomy (U-shape, tail, threshold, etc.)  
**Impact**: Cannot target repairs effectively  
**Task**: FO-013

### U-Shape Repair Absent
**Severity**: P2  
**Package**: factor_optimizer  
**Discovery**: Comprehensive plan FO-014  
**Evidence**: No distance-from-center, centered square, symmetric tail rank transforms  
**Impact**: U-shaped signals rejected instead of repaired  
**Task**: To be added

---

## FactorAssets Findings

### Registry Not Persistent
**Severity**: P1  
**Package**: factor_assets  
**Discovery**: Comprehensive plan FA-001  
**Evidence**: In-memory registry only  
**Impact**: Factor metadata lost on restart, no ACID guarantees  
**Task**: FA-001

### SeenIndex Not Persistent
**Severity**: P1  
**Package**: factor_assets  
**Discovery**: Comprehensive plan FA-002  
**Evidence**: In-memory seen tracking only  
**Impact**: Duplicate detection lost on restart  
**Task**: FA-002

### Simplified Leiden Not Production-Grade
**Severity**: P2  
**Package**: factor_assets  
**Discovery**: Comprehensive plan FA-013  
**Evidence**: Using simplified modularity approximation instead of mature Leiden library  
**Impact**: Lower quality clustering, not reproducible with standard implementations  
**Task**: FA-013

### ANN Used as Clustering
**Severity**: P2  
**Package**: factor_assets  
**Discovery**: Comprehensive plan section 13  
**Evidence**: HNSW may be treated as clustering instead of neighbor recall  
**Impact**: Conceptual confusion, incorrect usage  
**Task**: To be added

### Correlation as Hard Deletion
**Severity**: P2  
**Package**: factor_assets  
**Discovery**: Comprehensive plan FA-020  
**Evidence**: May use single correlation threshold for hard rejection  
**Impact**: Factors with high correlation but real novelty incorrectly removed  
**Task**: To be added

---

## FactorPreprocess Findings

### QE Cache Dependency (Critical)
**Severity**: P0  
**Package**: factor_preprocess  
**Discovery**: Comprehensive plan FP-001  
**Evidence**: Direct import of `quant_evaluator.runtime.cache_v2.MultiLevelCache`  
**Impact**: Cannot use FP independently, breaks package extraction  
**Task**: FP-001

### Cache Over-Design
**Severity**: P2  
**Package**: factor_preprocess  
**Discovery**: Comprehensive plan FP-002  
**Evidence**: May have Disk/Redis/compression/warming without benchmarks  
**Impact**: Unnecessary complexity  
**Task**: FP-002

### Stateless/Fitted Not Separated
**Severity**: P2  
**Package**: factor_preprocess  
**Discovery**: Comprehensive plan FP-005  
**Evidence**: Rank/zscore vs PCA/ICA/fitted transforms not type-level separated  
**Impact**: Risk of fit-on-all-data instead of fit-on-train  
**Task**: To be added

### fit_transform Anti-Pattern
**Severity**: P1  
**Package**: factor_preprocess  
**Discovery**: Comprehensive plan FP-006  
**Evidence**: May have `fit_transform(full_data)` instead of `fit(train)` then `transform(val/test)`  
**Impact**: Data leakage, invalid results  
**Task**: To be added

### Raw Not Preserved
**Severity**: P2  
**Package**: factor_preprocess  
**Discovery**: Comprehensive plan FP-007  
**Evidence**: Transforms may overwrite source factors in-place  
**Impact**: Cannot recover original values  
**Task**: To be added

---

## Cross-Cutting Concerns

### Wheel Packaging Not Validated
**Severity**: P0  
**Package**: ALL  
**Discovery**: Original task assignment  
**Evidence**: 4 smoke tests need execution  
**Impact**: Cannot confirm packages extract and install correctly  
**Tasks**: SMOKE-001, SMOKE-002, SMOKE-003, SMOKE-004

### Import Graph Violations
**Severity**: P1  
**Package**: CORE  
**Discovery**: Comprehensive plan section 10  
**Evidence**: No continuous auditing of cross-package imports  
**Impact**: Boundary violations like FP-001 go undetected  
**Task**: CORE-002

### Test Baseline Unknown
**Severity**: P1  
**Package**: ALL  
**Discovery**: Need current state  
**Evidence**: Don't know if current tests pass  
**Impact**: Cannot detect regressions  
**Task**: ARCH-001 (to be added)

---

## Discovery Methods

- **Comprehensive Plan**: Pre-identified 112+ issues from systematic review
- **Code Inspection**: Direct examination of implementations
- **Audit Sweeps**: Periodic automated checks (not yet run)
- **Integration Testing**: Cross-package validation (not yet run)
- **Red Team**: Adversarial testing (not yet started)

---

## Findings by Severity

- **P0 (Blocking)**: 5 findings (4 smoke tests + FP-001)
- **P1 (Critical)**: 8 findings
- **P2 (Important)**: 10 findings
- **P3 (Nice-to-have)**: 0 findings so far

**Total**: 23 findings documented  
**Remaining**: ~89 from comprehensive plan to document

---

## Next Discovery Actions

1. Execute smoke tests to validate/refute packaging assumptions
2. Check FP imports to confirm QE cache dependency
3. Run test baseline across all 4 packages
4. First audit sweep for imports, TODO, dead code
5. Red team leakage scenarios
