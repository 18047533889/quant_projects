# Code Quality Analysis Report
**Generated:** 2026-08-14
**Project:** quant_projects
**Packages Analyzed:** dataaccess, factor_layer, toolkit, quant_evaluator

---

## Executive Summary

### Overall Metrics

| Package | Pylint Score | Flake8 Issues | Files Needing Black Format | Import Sort Issues |
|---------|--------------|---------------|----------------------------|-------------------|
| dataaccess | Not completed | 850 | 300+ | High |
| factor_layer | 4.35/10 | 2,694 | 100+ | High |
| toolkit | 9.38/10 | 13 | Few | Low |
| quant_evaluator | Not completed | 74 | Moderate | Moderate |
| **Total** | - | **3,631** | **358 files** | **~10,000 lines** |

### Critical Findings

1. **Code Complexity Crisis**: 29 functions with F-grade complexity (>40), 173 functions exceed complexity threshold
2. **Documentation Gap**: 1,617 missing function docstrings, 125 missing class docstrings
3. **Import Organization**: ~1,580 imports outside toplevel, 10,366 lines needing import reordering
4. **Code Formatting**: 358 files fail black formatting check
5. **Maintainability**: 23 files with B/C grade maintainability index

---

## 1. Pylint Analysis

### Package Scores

- **toolkit**: 9.38/10 ⭐ (Excellent)
- **factor_layer**: 4.35/10 ⚠️ (Needs significant improvement)
- **dataaccess**: Analysis incomplete (large codebase)
- **quant_evaluator**: Analysis incomplete

### Top 20 Issue Categories

| Category | Count | Priority | Auto-Fix |
|----------|-------|----------|----------|
| missing-function-docstring | 1,617 | Medium | No |
| import-outside-toplevel | 1,578 | Low | Manual |
| line-too-long | 790 | Low | Yes (Black) |
| trailing-whitespace | 528 | Low | Yes (Black) |
| use-dict-literal | 487 | Low | Yes |
| bad-indentation | 476 | Medium | Yes (Black) |
| invalid-name | 462 | Low | Manual |
| protected-access | 458 | Low | Review |
| broad-exception-caught | 419 | Medium | Manual |
| redefined-outer-name | 303 | Low | Manual |
| unused-import | 292 | Medium | Yes (autoflake) |
| unused-argument | 290 | Low | Manual |
| too-many-locals | 223 | Low | Refactor |
| unused-variable | 184 | Medium | Yes |
| too-many-arguments | 177 | Medium | Refactor |
| duplicate-code | 170 | Medium | Refactor |
| missing-class-docstring | 125 | Medium | No |
| too-few-public-methods | 114 | Low | Review |
| too-many-branches | 97 | Medium | Refactor |
| reimported | 93 | Low | Yes |

---

## 2. Flake8 Analysis (PEP 8 Compliance)

### Issues by Package

| Package | Total Issues | Critical (E) | Warnings (W) | F-errors |
|---------|--------------|--------------|--------------|----------|
| dataaccess | 850 | 134 (E501) | 115 (W293) | 340 (F401) |
| factor_layer | 2,694 | High | High | High |
| toolkit | 13 | Minimal | Minimal | Minimal |
| quant_evaluator | 74 | Low | Low | Moderate |

### Top Flake8 Issues (dataaccess)

| Code | Issue | Count | Auto-Fix |
|------|-------|-------|----------|
| F401 | Unused import | 340 | Yes |
| E501 | Line too long (>120) | 134 | Yes |
| W293 | Blank line contains whitespace | 100 | Yes |
| E402 | Module import not at top | 48 | Manual |
| F841 | Unused variable | 44 | Yes |
| F821 | Undefined name | 41 | Manual |
| F811 | Redefinition | 28 | Manual |
| F541 | f-string missing placeholders | 17 | Yes |
| W605 | Invalid escape sequence | 12 | Yes |
| E741 | Ambiguous variable name 'l' | 9 | Manual |

---

## 3. Code Complexity Analysis (Radon)

### Complexity Distribution

- **F-grade (>40 complexity)**: 29 functions
- **E-grade (31-40)**: 26 functions  
- **D-grade (21-30)**: 118 functions
- **Total high complexity (>20)**: 173 functions

### Top 10 Most Complex Functions

| Complexity | Grade | Function | Location |
|------------|-------|----------|----------|
| 89 | F | DataAccessStore._read_joined_sql | dataaccess/store.py:4975 |
| 73 | F | parse_source_manifest | dataaccess/snapshot/resolver.py:151 |
| 68 | F | _execute_composed | dataaccess/read/physical_plan.py:220 |
| 64 | F | build_contract_ir | dataaccess/read/contract_ir.py:236 |
| 61 | F | SourceSnapshotResolver.resolve | dataaccess/snapshot/resolver.py:426 |
| 57 | F | estimate_scan_cost | dataaccess/read/scan_cost.py:162 |
| 53 | F | main | dataaccess/scripts/benchmark_workloads.py:88 |
| 48 | F | compute_coverage | dataaccess/read/coverage.py:151 |
| 47 | F | run_quality_checks | dataaccess/quality/contracts.py:141 |
| 45 | F | _build_pit_index_locked | dataaccess/read/pit_event_index.py:602 |

### Average Complexity by Package

- **dataaccess**: C (18.8) - Moderate complexity
- **factor_layer**: High complexity in evaluation modules
- **toolkit**: Low complexity
- **quant_evaluator**: Moderate complexity

---

## 4. Maintainability Index

### Files with Low Maintainability (<65 or Grade B/C)

**Critical (Grade C, MI < 10):**
- `dataaccess/store.py` - MI: 0.00 (C) ⚠️ CRITICAL
- `factor_layer/factor_evaluation_alphapurify/FactorAnalyzer.py` - MI: 0.00 (C) ⚠️
- `factor_layer/factor_evaluation_alphapurify/Exposures.py` - MI: 3.52 (C)
- `dataaccess/read/manifest.py` - MI: 4.86 (C)
- `dataaccess/read/session_calendar.py` - MI: 5.17 (C)

**Warning (Grade B, MI 10-20):**
- `dataaccess/cos/remote.py` - MI: 9.45 (B)
- `dataaccess/cos/mirror.py` - MI: 10.73 (B)
- `dataaccess/read/pit_event_index.py` - MI: 15.90 (B)
- `factor_layer/factor_evaluation_alphapurify/pipeline.py` - MI: 15.96 (B)
- `dataaccess/read/sql_escape.py` - MI: 16.10 (B)

**Total files with MI < 65:** 468 files

---

## 5. Code Formatting (Black & isort)

### Black Formatting

- **Files needing reformatting:** 358
- **Most affected packages:** dataaccess (300+), factor_layer, quant_evaluator

### Import Sorting (isort)

- **Total lines needing reordering:** 10,366 lines
- **Status:** Major import organization needed across all packages

---

## 6. Top 20 Priority Improvements

### P0 - Critical (Immediate Action Required)

1. **Refactor `DataAccessStore._read_joined_sql`** (Complexity: 89)
   - Location: `dataaccess/store.py:4975`
   - Action: Split into smaller functions, extract SQL building logic
   
2. **Fix `store.py` maintainability** (MI: 0.00)
   - Action: Break into smaller modules by responsibility

3. **Refactor `parse_source_manifest`** (Complexity: 73)
   - Location: `dataaccess/snapshot/resolver.py:151`
   - Action: Extract validation and parsing logic

4. **Refactor `_execute_composed`** (Complexity: 68)
   - Location: `dataaccess/read/physical_plan.py:220`
   - Action: Extract execution strategies

5. **Refactor `FactorAnalyzer.calc_stats_for_period`** (Complexity: 43)
   - Location: `factor_layer/factor_evaluation_alphapurify/FactorAnalyzer.py:406`
   - Action: Split calculation steps

### P1 - High Priority (This Sprint)

6. **Remove 340 unused imports** (F401 errors)
   - Action: Run autoflake or manual cleanup
   - Command: `autoflake --remove-all-unused-imports --in-place --recursive .`

7. **Fix 134 line-too-long violations** (E501)
   - Action: Run black formatter
   - Command: `black dataaccess factor_layer toolkit quant_evaluator`

8. **Remove 100+ blank lines with whitespace** (W293)
   - Action: Run black formatter

9. **Fix 48 module-level import ordering** (E402)
   - Action: Manual review and reorganization

10. **Address 41 undefined names** (F821)
    - Location: Multiple files referencing 'ManagedBatchReader'
    - Action: Add missing imports or fix references

### P2 - Medium Priority (Next 2 Sprints)

11. **Add docstrings to 1,617 functions**
    - Strategy: Start with public APIs, then internal functions
    - Use docstring generator tools as starting point

12. **Fix 1,578 import-outside-toplevel warnings**
    - Review each case: some are intentional for circular imports
    - Move imports to module level where safe

13. **Reduce function complexity in 23 additional F/E-grade functions**
    - Target complexity < 20 for all production code

14. **Fix 419 broad-exception-caught warnings**
    - Replace `except Exception:` with specific exception types

15. **Resolve 290 unused-argument warnings**
    - Prefix with `_` if intentionally unused, remove if not needed

16. **Fix 184 unused-variable warnings**
    - Remove unused variables or use them appropriately

17. **Address 177 too-many-arguments warnings**
    - Introduce parameter objects or builder patterns

18. **Improve 23 files with B/C maintainability grade**
    - Focus on the 10 files with MI < 20

19. **Run isort on all packages**
    - Command: `isort dataaccess factor_layer toolkit quant_evaluator`
    - Will fix 10,366 lines of import ordering

20. **Fix 170 duplicate-code violations**
    - Extract common logic into shared utilities

---

## 7. Automated Fix Commands

### Quick Wins (Can run immediately)

```bash
# 1. Format all code with black
black dataaccess factor_layer toolkit quant_evaluator

# 2. Sort imports
isort dataaccess factor_layer toolkit quant_evaluator

# 3. Remove unused imports (run with caution, review diff)
autoflake --remove-all-unused-imports --remove-unused-variables --in-place --recursive dataaccess factor_layer toolkit quant_evaluator

# 4. Fix trailing whitespace and blank lines (included in black)
# Already covered by black above
```

### Expected Impact

| Action | Files Fixed | Issues Resolved | Time Required |
|--------|-------------|-----------------|---------------|
| Black formatting | 358 | ~1,000 | 5 minutes |
| isort | All packages | 10,366 lines | 5 minutes |
| autoflake | ~200 files | ~500 imports | 10 minutes |
| **Total Quick Wins** | **500+** | **~2,000** | **20 minutes** |

---

## 8. Manual Review Checklist

### Code Architecture

- [ ] Review `dataaccess/store.py` for decomposition opportunities
- [ ] Extract query building logic from large functions
- [ ] Identify and extract common patterns across modules
- [ ] Review all F-grade complexity functions (29 functions)

### Error Handling

- [ ] Replace broad exception handlers with specific types
- [ ] Add proper error messages to all exception handlers
- [ ] Implement proper exception hierarchies

### Documentation

- [ ] Document all public APIs (classes, functions)
- [ ] Add module-level docstrings
- [ ] Create architecture decision records for complex modules

### Testing

- [ ] Ensure all refactored functions have tests
- [ ] Add integration tests for high-complexity modules
- [ ] Verify test coverage after cleanup

---

## 9. Long-term Recommendations

### Architecture

1. **Decompose `store.py`**: Split 9,000+ line file into:
   - `store_core.py` - Core store logic
   - `store_read.py` - Read operations
   - `store_write.py` - Write operations
   - `store_sql.py` - SQL query building
   - `store_join.py` - Join operations

2. **Extract complexity from snapshot/resolver**:
   - Create dedicated parser classes
   - Implement strategy pattern for different manifest types

3. **Refactor evaluation pipeline**:
   - `FactorAnalyzer` is too large (0.00 MI)
   - Split into calculation, statistics, and reporting modules

### Process Improvements

1. **Pre-commit hooks**: Add black, isort, flake8 to prevent regression
2. **CI/CD gates**: Fail builds on:
   - Flake8 critical errors (E, F series)
   - Functions with complexity > 30
   - Files with MI < 20

3. **Code review standards**:
   - Require docstrings for all new functions
   - Maximum function complexity: 15
   - Maximum function length: 100 lines

---

## 10. Estimated Effort

| Phase | Tasks | Effort | Dependencies |
|-------|-------|--------|--------------|
| **Phase 1: Automated Fixes** | Black, isort, autoflake | 2 hours | None |
| **Phase 2: Import Cleanup** | Manual import organization | 4 hours | Phase 1 |
| **Phase 3: Complexity P0** | Refactor 5 critical functions | 3 days | Phase 2 |
| **Phase 4: Documentation** | Add docstrings to public APIs | 2 days | Phase 1 |
| **Phase 5: Complexity P1** | Refactor remaining F/E grade | 1 week | Phase 3 |
| **Phase 6: Architecture** | Split large modules | 2 weeks | Phase 5 |

**Total Estimated Effort:** 4-5 weeks (1 developer full-time)

---

## Conclusion

The codebase has **significant technical debt** in areas of:
1. Code complexity (29 F-grade functions)
2. Documentation (1,617 missing docstrings)
3. Code formatting (358 files need formatting)
4. Maintainability (23 files with poor MI scores)

**Immediate actions:**
1. Run automated formatters (Black + isort) - 20 minutes
2. Tackle P0 complexity issues in `store.py` and `resolver.py` - 3 days
3. Set up pre-commit hooks to prevent regression - 1 hour

**Success metrics after Phase 1-3:**
- Pylint scores > 8.0 for all packages
- Zero functions with complexity > 30
- All public APIs documented
- Zero critical flake8 violations
