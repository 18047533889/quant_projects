# Q/kdb+ Backend Parity Integration Plan

## Status: P2 - Documented, Ready for Implementation

**Created:** 2026-08-14  
**Task:** P1-03 (downgraded to P2)  
**Owner:** Loop Engineering System

---

## Executive Summary

The Q/kdb+ backend is **fully implemented and tested** with dedicated test coverage. The gap is **systematic four-way parity testing** (pandas == polars == duckdb == q) to ensure numerical consistency across all backends.

**Current State:**
- ✅ Q backend implementation complete (2,656 LOC, 8 modules)
- ✅ Dedicated Q tests exist (4 test files, basic + capability + benchmark)
- ❌ No four-way parity tests comparing Q to pandas/polars/duckdb

**Priority Justification:** P2 because:
- Q backend has dedicated functional tests
- Three-way parity (pandas/polars/duckdb) already comprehensive
- Q integration is additive enhancement, not blocking issue

---

## Q Backend Architecture

**Location:** `/home/shw/quant_projects/factor_engine/backend/q_backend/`

**Modules:**
1. `q_backend.py` (11,257 LOC) - Core backend implementation
2. `q_compiler.py` (13,856 LOC) - Q language compiler
3. `q_executor.py` (13,055 LOC) - Execution engine
4. `q_adapter.py` (9,833 LOC) - Python-Q bridge
5. `q_capability.py` (7,230 LOC) - Capability registry
6. `q_capability_evidence.py` (8,368 LOC) - Evidence/certification
7. `q_process_manager.py` (5,905 LOC) - Process lifecycle
8. `README.md` (11,172 LOC) - Documentation

**Test Coverage:**
- `test_q_backend_basic.py` (4,500 LOC)
- `test_q_backend.py` (12,967 LOC)
- `test_q_benchmark.py` (3,070 LOC)
- `test_q_capability_evidence.py` (11,045 LOC)

**Supported Operators:** 50+ native Q implementations

---

## Integration Path

### Phase 1: Fixture Setup (Week 1)

**Goal:** Create Q backend fixtures compatible with existing parity test infrastructure

**Tasks:**
1. **Q Connection Fixture**
   ```python
   @pytest.fixture(scope="session")
   def q_connection():
       """Provides Q/kdb+ connection for testing."""
       # Start Q process
       # Establish connection
       # Yield connection
       # Cleanup on teardown
   ```

2. **Q Data Loader**
   ```python
   @pytest.fixture
   def q_panel_data(panel, q_connection):
       """Upload pandas panel to Q for testing."""
       # Convert panel to Q table format
       # Upload to Q instance
       # Return reference
   ```

3. **Q Result Converter**
   ```python
   def q_result_to_pandas(q_result) -> pd.DataFrame:
       """Convert Q query result to pandas for comparison."""
       # Handle Q table -> pandas DataFrame
       # Preserve MultiIndex structure
       # Handle NaN/Inf correctly
   ```

**Deliverable:** `tests/backend_parity/q_parity_fixtures.py`

**Estimated Time:** 2-3 days

---

### Phase 2: Pilot Four-Way Test (Week 1)

**Goal:** Create ONE complete four-way parity test as template

**Operator Choice:** `ts_mean` (simplest, most reliable)

**Test Structure:**
```python
def test_ts_mean_four_way_parity(panel, duckdb_source, q_connection, q_panel_data):
    """Test ts_mean across all 4 backends."""
    window = 20
    
    # Execute on pandas
    result_pandas = ts_mean(panel['close'], window)
    
    # Execute on polars
    result_polars = ts_mean_polars(panel_polars['close'], window)
    
    # Execute on duckdb
    result_duckdb = execute_sql(f"SELECT ts_mean(close, {window}) FROM panel")
    
    # Execute on Q
    result_q = q_connection.sync(f"ts_mean[close; {window}]")
    result_q_pandas = q_result_to_pandas(result_q)
    
    # Four-way assertions
    assert_frame_equal(result_pandas, result_polars.to_pandas(), rtol=1e-6)
    assert_frame_equal(result_pandas, result_duckdb, rtol=1e-6)
    assert_frame_equal(result_pandas, result_q_pandas, rtol=1e-6)
```

**Deliverable:** `tests/backend_parity/test_four_way_pilot.py`

**Estimated Time:** 1-2 days

---

### Phase 3: Operator Coverage (Weeks 2-4)

**Goal:** Systematically add four-way tests for Q-supported operators

**Operator Priority:**

**High Priority (Week 2):**
1. ts_mean (pilot already done)
2. ts_std
3. ts_sum
4. ts_min / ts_max
5. ts_rank

**Medium Priority (Week 3):**
6. ts_delta
7. ts_corr
8. ts_cov
9. decay_linear
10. ema

**Low Priority (Week 4):**
11. Technical indicators (RSI, MACD, Bollinger)
12. Group operations (group_mean, group_std)
13. Cross-sectional ops (cs_rank, cs_zscore)

**Strategy:**
- Add Q execution path to existing three-way parity tests
- Reuse existing test data and fixtures
- Focus on operators Q backend already supports

**Deliverable:** Updated test files in `tests/backend_parity/` with Q backend integrated

**Estimated Time:** 2-3 weeks

---

### Phase 4: Divergence Documentation (Week 4)

**Goal:** Document known Q backend divergences

**Expected Divergences:**
1. **Floating-point precision:** Q uses 64-bit floats, may have minor numerical differences
2. **NaN handling:** Q's null semantics may differ from pandas
3. **Edge cases:** Inf handling, division by zero, empty windows

**Deliverable:** `backend/q_backend/KNOWN_DIVERGENCES.md`

**Estimated Time:** 1-2 days

---

## Prerequisites

**Infrastructure:**
- ✅ Q process manager implemented
- ✅ Q connection pooling available
- ✅ Q compiler supports 50+ operators
- ⚠️ CI/CD needs Q instance for automated testing

**Blocking Issues:**
- None - Q backend is production-ready
- CI integration requires Q license/deployment

**Dependencies:**
- Existing three-way parity test infrastructure (complete)
- Q backend implementation (complete)
- Test data fixtures (reusable)

---

## Success Criteria

1. **Fixture Coverage:** Q connection + data loader fixtures working
2. **Pilot Test:** One complete four-way parity test passing
3. **Operator Coverage:** ≥20 operators with four-way parity tests
4. **Documentation:** Known divergences documented
5. **CI Integration:** Q tests run automatically (if Q instance available)

---

## Risks & Mitigation

**Risk 1:** Q process startup overhead slows tests
- **Mitigation:** Session-scoped fixtures, reuse Q instance across tests

**Risk 2:** Q numerical behavior differs from pandas
- **Mitigation:** Document divergences, adjust tolerances if needed

**Risk 3:** CI/CD lacks Q instance
- **Mitigation:** Mark Q tests as optional, skip if connection unavailable

**Risk 4:** Q backend changes independently of parity tests
- **Mitigation:** Run four-way tests on every Q backend change

---

## Implementation Effort

**Total Estimated Time:** 3-4 weeks (1 developer)

**Breakdown:**
- Week 1: Fixtures + pilot test (5 days)
- Week 2-3: High/medium priority operators (10 days)
- Week 4: Low priority operators + documentation (5 days)

**Recommended Team:**
- 1 developer familiar with Q/kdb+ language
- 1 reviewer familiar with parity test infrastructure

---

## Next Actions

1. **Immediate (if prioritized to P1):**
   - Assign developer with Q expertise
   - Start Phase 1 (fixture setup)
   
2. **Short-term (P2 current priority):**
   - Keep Q backend functional tests running
   - Revisit after P0/P1 issues resolved
   
3. **Long-term:**
   - Integrate Q into continuous parity monitoring
   - Expand Q operator coverage based on usage

---

## References

- Q Backend Implementation: `/backend/q_backend/`
- Q Backend Tests: `/tests/q_backend/`
- Three-Way Parity Tests: `/tests/backend_parity/`
- Q Backend README: `/backend/q_backend/README.md`
