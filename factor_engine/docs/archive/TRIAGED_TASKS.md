# Loop Engineering Task Queue - TRIAGE REPORT
**Date:** 2026-08-14  
**Triaged by:** TriageSpecialist  
**Source:** TASK_QUEUE.md (9 discovered issues)

---

## EXECUTIVE SUMMARY

Completed systematic triage of 9 discovered issues. **Critical finding:** P0-01 is a REAL BLOCKER preventing all registry imports.

**Key Verdicts:**
- **1 P0 BLOCKER** (registry error) - requires immediate fix
- **2 P0 gaps** (technical indicators + base.py) - confirmed real issues
- **3 P1 legitimate** (statistical ops, cs_zscore, fake native)
- **2 P1 reduced severity** (Q backend, ewm divergences)
- **2 P2 acceptable** (Q TODO, base.py abstract)

**Top 3 for immediate implementation:**
1. **P0-01** - Fix MeanAbsoluteDeviation/MedianAbsoluteDeviation missing registration
2. **P0-02** - Add technical indicator parity tests (EMA, SMA, RSI, MACD, Bollinger)
3. **P1-01** - Add statistical operator parity tests (correlation, covariance, skew, kurt, quantile)

---

## TRIAGED TASKS

### P0-01: Registry initialization error ✅ CONFIRMED BLOCKER

**Status:** REAL BUG - BLOCKS ALL TESTING  
**Severity:** P0 (BLOCKER)  
**Complexity:** SIMPLE (30 minutes)

**Root Cause Analysis:**
- File: `/home/shw/quant_projects/factor_engine/cleaned_operators/common/statistics.py:1696`
- Two operator classes defined WITHOUT `@register_operator` decorator:
  - `MeanAbsoluteDeviation` (line 347) - metadata.name = "ts_mean_abs_deviation"
  - `MedianAbsoluteDeviation` (line 374) - metadata.name = "ts_median_abs_deviation"
- Line 1696 attempts: `OperatorRegistry.register_alias("ts_mean_absolute_deviation", "ts_mean_abs_deviation")`
- Alias fails because target "ts_mean_abs_deviation" was never registered
- No `__init_subclass__` auto-registration exists in base.py - classes MUST use `@register_operator` decorator

**Evidence:**
```python
# Line 347-370: Class without decorator
class MeanAbsoluteDeviation(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_mean_abs_deviation",  # Target name for alias
        ...
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs):
        ...

# Line 1696: Alias registration fails
OperatorRegistry.register_alias("ts_mean_absolute_deviation", "ts_mean_abs_deviation")
# KeyError: "alias target is not registered: 'ts_mean_abs_deviation'"
```

**Fix Strategy:**
1. Add `@register_operator` decorator to both classes:
   ```python
   @register_operator(
       name="ts_mean_abs_deviation",
       canonical="ts_mean_abs_deviation",
       backend="pandas_numpy"
   )
   class MeanAbsoluteDeviation(SeriesOperator):
       ...
   
   @register_operator(
       name="ts_median_abs_deviation", 
       canonical="ts_median_abs_deviation",
       backend="pandas_numpy"
   )
   class MedianAbsoluteDeviation(SeriesOperator):
       ...
   ```
2. Verify import: `python3 -c "from cleaned_operators.registry import OperatorRegistry; print('OK')"`

**Dependencies:** None - standalone fix  
**Blocked tasks:** ALL other tasks depend on this (can't import cleaned_operators)  
**Ready for implementation:** YES

---

### P0-02: Technical indicators untested ✅ CONFIRMED GAP

**Status:** REAL GAP - NO BACKEND PARITY TESTS  
**Severity:** P0 (production risk - widely used operators)  
**Complexity:** MEDIUM (4-6 hours for all 8 operators)

**Root Cause Analysis:**
- 8 operators exist and are registered: EMA, SMA, RSI, MACD_line, MACD_signal, bollinger_upper, bollinger_mid, bollinger_lower
- Found in `composite_reference_helpers.py` COMPOSITE_REFERENCE_CASES (lines 43-45)
- Test file `test_composite_reference_semantics.py` exists but ONLY tests "original pandas operator vs lowered-plan pandas"
- **Missing:** Three-way backend parity tests (pandas == polars == duckdb)
- BACKEND_PARITY_FINDINGS.md confirms 0/8 have parity tests

**Evidence:**
```python
# Operators exist in composite reference cases:
CompositeReferenceCase("MACD_line", ("close",), window=3, ...),
CompositeReferenceCase("MACD_signal", ("close",), window=3, ...),
# But test_composite_reference_semantics.py has NO three-way parity tests
# Search results: 0 matches for "def test.*EMA|SMA|RSI|MACD|bollinger" in parity tests
```

**Fix Strategy:**
1. Create `test_technical_indicators_triple_parity.py` in tests/backend_parity/
2. Follow pattern from `test_three_backend_parity.py` (line 716, 20 tests)
3. Add parametrized tests for all 8 indicators:
   ```python
   @pytest.mark.parametrize("op_name,kwargs", [
       ("EMA", {"window": 20}),
       ("SMA", {"window": 20}),
       ("RSI", {"window": 14}),
       ("MACD_line", {"fast": 12, "slow": 26}),
       ("MACD_signal", {"fast": 12, "slow": 26, "signal": 9}),
       ("bollinger_upper", {"window": 20, "std_dev": 2.0}),
       ("bollinger_mid", {"window": 20}),
       ("bollinger_lower", {"window": 20, "std_dev": 2.0}),
   ])
   def test_indicator_triple_parity(panel, duckdb_source, op_name, kwargs):
       # Three-way assertion: pandas == polars == duckdb
   ```
4. Run: `pytest tests/backend_parity/test_technical_indicators_triple_parity.py -v`

**Dependencies:** Blocked by P0-01 (can't import registry)  
**Blocked tasks:** None  
**Ready for implementation:** YES (after P0-01)

---

### P1-01: Statistical operators untested ✅ CONFIRMED GAP

**Status:** REAL GAP - 5 OPERATORS LACK PARITY TESTS  
**Severity:** P1 (core statistical operations)  
**Complexity:** MEDIUM (3-4 hours)

**Root Cause Analysis:**
- BACKEND_PARITY_FINDINGS.md §1.1 line 58-69: Only 2/7 statistical operators have parity tests
- Tested: `rank`, `zscore`
- **Missing:** `correlation`, `covariance`, `skew`, `kurt`, `quantile`
- Operators found in test files (grep results) but NOT in systematic three-way parity tests

**Fix Strategy:**
1. Add to existing `test_elementwise_family_systematic_parity.py` OR create new file
2. Pattern:
   ```python
   @pytest.mark.parametrize("op_name,kwargs", [
       ("correlation", {"window": 20}),
       ("covariance", {"window": 20}),
       ("skew", {"window": 20}),
       ("kurt", {"window": 20}),
       ("quantile", {"quantile": 0.5, "window": 20}),
   ])
   def test_statistical_ops_triple_parity(panel, duckdb_source, op_name, kwargs):
       # pandas == polars == duckdb
   ```

**Dependencies:** Blocked by P0-01  
**Blocked tasks:** None  
**Ready for implementation:** YES (after P0-01)

---

### P1-02: cs_zscore missing parity test ✅ CONFIRMED GAP

**Status:** REAL GAP - 1 OPERATOR  
**Severity:** P1 (cross-sectional family incomplete)  
**Complexity:** SIMPLE (30 minutes)

**Root Cause Analysis:**
- BACKEND_PARITY_FINDINGS.md line 45: "cs_zscore ❌ NOT TESTED"
- Only 3/4 cross-sectional operators have parity tests: cs_mean, cs_std, cs_demean
- Found one reference in `test_triple_probe_2026_08.py` but not systematic parity test

**Fix Strategy:**
1. Add to `test_cs_family_systematic_parity.py` (257 lines, 6 tests)
2. Follow pattern of existing cs_mean/cs_std/cs_demean tests
3. Single parametrized test case

**Dependencies:** Blocked by P0-01  
**Blocked tasks:** None  
**Ready for implementation:** YES (after P0-01)

---

### P1-03: Q backend parity unverified ⚠️ REDUCED SEVERITY

**Status:** INFRASTRUCTURE GAP (not a bug)  
**Severity:** P1 → **P2** (Q backend exists but not systematically verified)  
**Complexity:** COMPLEX (8-16 hours)

**Root Cause Analysis:**
- Q backend fully implemented: 8 modules (2,656 lines) in backend/q_backend/
- 4 test files exist: test_q_backend.py, test_q_backend_basic.py, test_q_benchmark.py, test_q_capability_evidence.py
- **Missing:** Four-way parity tests (pandas/polars/duckdb/q)
- Current tests verify Q backend functionality but NOT cross-backend equivalence
- 348 grep matches for "Q backend|q backend|kdb" in parity test files indicate infrastructure exists

**Downgrade Justification:**
- Q backend has dedicated test suite (4 files)
- q_capability.py declares 50+ native operators with capability tracking
- Issue is test methodology (four-way vs three-way), not missing coverage
- Lower production risk than untested core operators (P0-02, P1-01)

**Fix Strategy:**
1. Extend existing three-way parity tests to four-way
2. Add Q backend execution path to test helpers
3. Conditional execution: `if is_q_available():`
4. Start with subset: 10-15 high-priority operators

**Dependencies:** None  
**Blocked tasks:** None  
**Ready for implementation:** NEEDS_MORE_INFO (Q backend availability in test environment)

---

### P1-04: Known ewm divergences ⚠️ DOCUMENTED, NOT A BUG

**Status:** DOCUMENTED BEHAVIOR  
**Severity:** P1 → **P2** (known limitation, not regression)  
**Complexity:** COMPLEX (requires algorithm change, not bug fix)

**Root Cause Analysis:**
- Memory file evidence: "Polars NaN hole behavior differs from pandas recursive calculation"
- factor-engine-backend-consistency-2026-08-09.md: EWM/Wilder family (RSI/ATR/DMI/DX/ADX/MACD/DEMA/TEMA) moved to _SQL_FALLBACK_CANONICALS
- Reason: "pandas ewm(adjust=False) ignore_na=False 语义无法在 SQL 精确复刻"
- These operators ALREADY use polars fallback to preserve pandas semantics

**Downgrade Justification:**
- Not a bug - documented backend capability limitation
- Workaround already implemented (polars fallback)
- No production risk - system knows to use pandas-compatible backend
- Fix requires new EWM algorithm, not bug correction

**Fix Strategy (if prioritized):**
1. Implement NaN-aware EWM algorithm in Polars/SQL
2. Match pandas recursive calculation with NaN holes
3. Add comprehensive NaN edge case tests

**Dependencies:** None  
**Blocked tasks:** None  
**Ready for implementation:** NO (requires research + algorithm design)

---

### P1-05: Polars .to_pandas() fake native ✅ CONFIRMED ISSUE

**Status:** REAL PERFORMANCE REGRESSION  
**Severity:** P1 (performance, not correctness)  
**Complexity:** MEDIUM-COMPLEX (6-10 hours, depends on context)

**Root Cause Analysis:**
- BACKEND_SCAN_FINDINGS.txt: 7 instances across 5 files
- **Files affected:**
  1. backend/polars_panel.py: `return panel.to_pandas()`
  2. backend/panel_polars.py: `pdf = result.select([...]).to_pandas()`
  3. backend/long_frame.py: 2 instances
  4. backend/sql_pushdown/duckdb_optimizer_integration.py: `pdf = sid_df.to_pandas().set_index([...])`
  5. backend/sql_pushdown/duckdb_performance.py: 2 instances

**Impact:**
- Performance regression: Round-trip through pandas negates native backend benefits
- Not correctness issue: Results are correct, just slower
- Affects batch operations and large datasets

**Fix Strategy:**
1. **polars_panel.py / panel_polars.py:** Keep operations in Polars until final result needed
2. **long_frame.py:** Check if caller actually needs pandas or can accept Polars DataFrame
3. **duckdb_*.py:** Use DuckDB native operations instead of Polars intermediate
4. Each fix requires careful analysis of call sites and return type contracts

**Dependencies:** None  
**Blocked tasks:** None  
**Ready for implementation:** YES (requires careful refactoring per file)

---

### P2-01: Q backend test infrastructure TODO ✅ CONFIRMED TODO

**Status:** DOCUMENTED TODO (not blocking)  
**Severity:** P2 (documentation/tooling gap)  
**Complexity:** MEDIUM (4-6 hours)

**Root Cause Analysis:**
- File: `/home/shw/quant_projects/factor_engine/backend/q_backend/q_capability_evidence.py`
- Two TODO comments:
  1. "TODO: compile/runtime/parity passes need test infrastructure"
  2. "TODO: Add compile/runtime/parity verification."
- This is related to P1-03 but more about test tooling than actual testing

**Fix Strategy:**
1. Create test infrastructure for Q backend capability verification
2. Add compile-time checks (operator → q code generation)
3. Add runtime checks (q execution correctness)
4. Add parity checks (q results == pandas results)
5. Similar to existing capability evidence framework

**Dependencies:** None  
**Blocked tasks:** P1-03 (four-way parity)  
**Ready for implementation:** YES (but lower priority than P0/P1)

---

### P2-02: Base backend abstract method ⚠️ FALSE POSITIVE

**Status:** FALSE POSITIVE (expected behavior)  
**Severity:** P2 → **REJECTED**  
**Complexity:** N/A

**Root Cause Analysis:**
- File: `/home/shw/quant_projects/factor_engine/backend/base.py:28`
- Abstract method `execute()` with `raise NotImplementedError`
- This is **correct design** for abstract base class

**Evidence:**
```python
class Backend(ABC):
    @abstractmethod
    def execute(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        """求值整棵计划树。
        ...
        """
        raise NotImplementedError
```

**Verdict:** This is standard Python ABC pattern. Abstract methods SHOULD raise NotImplementedError if called on base class. Concrete implementations (PandasBackend, PolarsBackend, DuckDBBackend) override this method.

**Fix Strategy:** None needed - this is correct code.

**Dependencies:** N/A  
**Blocked tasks:** N/A  
**Ready for implementation:** NO (rejected as false positive)

---

## PRIORITY MATRIX

### Immediate (Today)
1. **P0-01** - Fix registry initialization (30 min) ⭐ BLOCKER
2. **P0-02** - Add technical indicator parity tests (4-6 hrs) ⭐ HIGH VALUE

### High Priority (This Week)
3. **P1-01** - Add statistical operator parity tests (3-4 hrs)
4. **P1-02** - Add cs_zscore parity test (30 min)
5. **P1-05** - Fix fake native .to_pandas() calls (6-10 hrs)

### Medium Priority (Next Week)
6. **P1-03** - Q backend four-way parity integration (8-16 hrs)
7. **P2-01** - Q backend test infrastructure (4-6 hrs)

### Deferred/Rejected
8. **P1-04** - ewm divergences (documented limitation, no action)
9. **P2-02** - Base backend abstract method (FALSE POSITIVE, rejected)

---

## DEPENDENCY GRAPH

```
P0-01 (registry fix)
  └─── BLOCKS ALL OTHER TASKS
       ├─── P0-02 (technical indicators)
       ├─── P1-01 (statistical ops)
       └─── P1-02 (cs_zscore)

P2-01 (Q test infrastructure)
  └─── ENABLES P1-03 (Q backend parity)

P1-05 (fake native) - INDEPENDENT
P1-04 (ewm) - DOCUMENTED, NO ACTION
P2-02 (base.py) - REJECTED
```

---

## IMPLEMENTATION RECOMMENDATIONS

### Session 1: Critical Path (5-7 hours)
1. **P0-01** - Fix registry (30 min)
   - Add @register_operator decorators
   - Verify import works
   - Run quick smoke test
2. **P0-02** - Technical indicators (4-6 hrs)
   - Create test file
   - Add 8 parametrized tests
   - Run full suite
   - Document any discovered divergences

### Session 2: Core Coverage (4-5 hours)
3. **P1-01** - Statistical operators (3-4 hrs)
   - Add 5 operators to parity tests
   - Verify three-way parity
4. **P1-02** - cs_zscore (30 min)
   - Quick add to existing test file

### Session 3: Performance (6-10 hours)
5. **P1-05** - Eliminate fake native conversions
   - Tackle one file at a time
   - Verify performance improvement with benchmarks
   - polars_panel.py → panel_polars.py → long_frame.py → duckdb files

### Session 4: Q Backend (Optional, 12-22 hours)
6. **P2-01** + **P1-03** - Q backend infrastructure + parity
   - Only if Q backend is production-critical
   - Can defer if not actively used

---

## RISK ASSESSMENT

### High Risk (Blocking Production)
- **P0-01**: Blocks ALL testing and development ❌ CRITICAL

### Medium Risk (Production Operators Unverified)
- **P0-02**: Technical indicators used in production without parity verification
- **P1-01**: Core statistical operations may have silent divergences

### Low Risk (Performance/Documentation)
- **P1-05**: Performance issue, not correctness
- **P1-02**: Single operator gap
- **P1-03**: Q backend has dedicated tests, just not four-way parity
- **P2-01**: Documentation/tooling gap

### No Risk
- **P1-04**: Already handled with fallback strategy
- **P2-02**: False positive

---

## TESTING STRATEGY

All tasks (except P0-01) follow this pattern:
1. **Create test file** (or extend existing)
2. **Add parametrized test cases** following existing parity test patterns
3. **Run tests** with: `pytest tests/backend_parity/test_*.py -v`
4. **Document divergences** if any found
5. **Add to BACKEND_PARITY_FINDINGS.md** coverage report

Expected outcomes:
- **Best case:** All new tests pass → coverage complete
- **Expected case:** 2-3 divergences discovered → document and fix or accept
- **Worst case:** Systematic divergences → need algorithm review

---

## NEXT ACTIONS

**Immediate (blocking all other work):**
```bash
# 1. Fix P0-01 registry error
vim cleaned_operators/common/statistics.py
# Add @register_operator decorators to lines 347 and 374

# 2. Verify fix
python3 -c "from cleaned_operators.registry import OperatorRegistry; print('SUCCESS')"
```

**After P0-01 resolved:**
```bash
# 3. Create technical indicator parity tests
vim tests/backend_parity/test_technical_indicators_triple_parity.py
pytest tests/backend_parity/test_technical_indicators_triple_parity.py -v
```

**Estimated total effort:** 20-30 hours for all P0+P1 tasks

---

## SUMMARY STATISTICS

- **Total issues triaged:** 9
- **Real bugs/gaps:** 6 (P0-01, P0-02, P1-01, P1-02, P1-05, P2-01)
- **Documented limitations:** 1 (P1-04)
- **False positives:** 1 (P2-02)
- **Severity adjusted:** 2 (P1-03→P2, P1-04→P2, P2-02→REJECTED)
- **Blocking issues:** 1 (P0-01)
- **Ready for implementation:** 6 (P0-01, P0-02, P1-01, P1-02, P1-05, P2-01)
- **Needs more info:** 1 (P1-03 - Q availability)
- **No action needed:** 2 (P1-04, P2-02)

**Recommendation:** Start with P0-01 immediately (30 minutes), then P0-02 (4-6 hours). This provides maximum risk reduction for minimum effort.
