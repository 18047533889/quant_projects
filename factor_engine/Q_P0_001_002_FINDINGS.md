# Q-P0-001/002: Q Backend Capability Authority Audit

**Date:** 2026-08-14  
**Worker:** Worker-Q  
**Campaign:** 6-hour autonomous audit  

## Executive Summary

The Q backend currently uses a **manually maintained `_PHASE1_NATIVE_OPS` list as capability authority**, creating a critical gap between declared capabilities and actual implementation. This audit identified **30 operators declared as native without lowering implementations**, violating production safety requirements.

**Status:** ✗ FAIL - Production not ready  
**Critical Finding:** Q_NATIVE_WITHOUT_LOWERING = 30 (MUST be 0)

---

## Findings

### Q-P0-001: Manual Authority Problem

**Problem:** The `_PHASE1_NATIVE_OPS` set in `backend/q_backend/q_capability.py` serves as the capability authority but is manually maintained and can diverge from implementation reality.

**Current State:**
- Manual list: 110 operators declared as "native"
- Actual lowering implementations: 80 operators
- Gap: 30 operators

**Code Location:**
```
backend/q_backend/q_capability.py:37-103
_PHASE1_NATIVE_OPS = {
    "add", "subtract", ..., "vwap", "ema", "wma", "sma"
}
```

**Risk:**
1. Manual list can be updated without implementing lowering
2. Operators can be declared native but fall back to Pandas silently
3. No automatic verification of capability claims
4. Production admission based on false authority

---

### Q-P0-002: Native Without Lowering

**Problem:** 30 operators are declared as `NATIVE` in `_PHASE1_NATIVE_OPS` but have no corresponding lowering implementation in `QCompiler._operator_map`.

**Critical Gap:**
```
Q_NATIVE_WITHOUT_LOWERING = DeclaredNative - LoweringExists
                          = 110 - 80
                          = 30 operators
```

**The 30 Missing Lowering Implementations:**

1. `bfill` - Backward fill
2. `cs_clip` - Cross-section clip
3. `cs_percentile_rank` - CS percentile rank
4. `cs_quantile` - CS quantile
5. `cs_winsorize` - CS winsorization
6. `group_count` - Group count
7. `group_max` - Group maximum
8. `group_mean` - Group mean
9. `group_median` - Group median
10. `group_min` - Group minimum
11. `group_std` - Group standard deviation
12. `group_sum` - Group sum
13. `replace` - Value replacement
14. `resample` - Time resampling
15. `time_bucket` - Time bucketing
16. `true_range` - True range indicator
17. `ts_argmax_age` - Time since argmax
18. `ts_argmin_age` - Time since argmin
19. `ts_decay_exp` - Exponential decay
20. `ts_decay_linear` - Linear decay
21. `ts_distance_to_high` - Distance to high
22. `ts_distance_to_low` - Distance to low
23. `ts_max_drawdown` - Maximum drawdown
24. `ts_moment` - Statistical moment
25. `ts_new_high` - New high detection
26. `ts_new_low` - New low detection
27. `ts_percentile` - Time series percentile
28. `ts_quantile` - Time series quantile
29. `ts_sum_decay` - Decayed sum
30. `vwap` - Volume-weighted average price

**Impact:**
- These operators are claimed as "native" but will silently fall back to Pandas
- No compile-time detection of the gap
- Performance expectations violated (user expects Q speedup, gets Pandas)
- Cost model inaccuracy

---

## Evidence Framework Architecture

### Current State: Manual Authority
```
_PHASE1_NATIVE_OPS (manual list)
         ↓
   QBackendCapability
         ↓
   supports_native(op) → True/False
```

### Required State: Evidence-Based Authority
```
DeclaredNative (intent)
         ↓
LoweringExists (implementation) ──→ QCompiler._operator_map
         ↓
CompilePass (correctness) ──→ Q code compiles
         ↓
RuntimePass (execution) ──→ Q code runs
         ↓
ParityPass (semantics) ──→ matches Pandas oracle
         ↓
Production Capability = ∩ all passes
```

---

## Evidence Implementation

Created `backend/q_backend/q_capability_evidence.py` implementing:

### 1. Evidence Pass Types
```python
class QEvidencePass(Enum):
    DECLARED_NATIVE = "declared_native"    # Intent declaration
    LOWERING_EXISTS = "lowering_exists"    # Implementation exists
    COMPILE_PASS = "compile_pass"          # Compiles to Q
    RUNTIME_PASS = "runtime_pass"          # Executes successfully
    PARITY_PASS = "parity_pass"            # Matches Pandas
```

### 2. Evidence Record
```python
@dataclass(frozen=True)
class QCapabilityEvidence:
    canonical: str
    declared_native: bool
    lowering_exists: bool
    compile_pass: bool
    runtime_pass: bool
    parity_pass: bool
    
    @property
    def production_safe(self) -> bool:
        return all 5 passes
    
    @property
    def native_without_lowering(self) -> bool:
        return declared_native and not lowering_exists
```

### 3. Auto-Derived Capability
```python
def get_declared_native_ops() -> frozenset[str]:
    """Authoritative source for native intent."""
    return frozenset({...})  # Explicit set

def get_lowering_exists_ops() -> frozenset[str]:
    """Query actual compiler implementations."""
    from backend.q_backend.q_compiler import get_q_compiler
    compiler = get_q_compiler()
    return frozenset(compiler._operator_map.keys())

def get_q_native_without_lowering() -> frozenset[str]:
    """Q-P0-002: Must be empty for production."""
    return get_declared_native_ops() - get_lowering_exists_ops()
```

### 4. Hard Gates
```python
class QCapabilityGate:
    @staticmethod
    def gate_q_native_without_lowering() -> tuple[bool, str]:
        """MUST pass for production readiness."""
        missing = get_q_native_without_lowering()
        if not missing:
            return True, "PASS"
        return False, f"FAIL: {len(missing)} ops without lowering"
```

---

## Test Results

### Evidence Framework Tests
```
✓ Test 1: Generate capability report
  - Declared native: 110
  - Lowering exists: 80
  - Native without lowering: 30

✓ Test 2: Run hard gates
  ✗ Q_NATIVE_WITHOUT_LOWERING: False (30 gaps)
  ✗ Q_MANUAL_AUTHORITY_REMOVED: False (still exists)

✓ Test 3: Compute evidence for all operators
  - Evidence computed for 110 operators

✓ Test 4: Check known operators
  ✓ add: declared=True, lowering=True
  ✓ subtract: declared=True, lowering=True
  ✓ ts_mean: declared=True, lowering=True
  ✓ cs_rank: declared=True, lowering=True

✓ Test 5: Gap detection
  - Detected 30 operators with gaps
```

---

## Comparison with Other Backends

### Polars/DuckDB Evidence Model
The Polars and DuckDB backends use a **6-stage evidence certification pipeline**:

1. `polars_reference_parity` - Reference implementation matches Pandas
2. `duckdb_reference_parity` - DuckDB reference matches Pandas
3. `duckdb_real_sql_verified` - Real SQL pushdown works
4. `polars_edge_verified` - Edge cases pass
5. `duckdb_edge_verified` - DuckDB edge cases pass
6. `no_fallback_verified` - No Pandas fallback path

**Evidence Storage:**
- `evidence/primitive_verified.json` - 84 operators with full evidence
- `evidence/composite_verified.json` - Composite operator evidence

**Certification Flow:**
```python
# backend/operator_certification.py
def run_certification(canonical: str) -> CertificationReport:
    for stage in [POLARS_REF, DUCKDB_REF, REAL_SQL, ...]:
        passed, detail = _pytest_for_stage(stage, canonical)
        if passed and write_evidence:
            _write_primitive_evidence(canonical, stage)
```

**Q Backend Gap:** No equivalent evidence pipeline exists.

---

## Recommendations

### Phase 1: Immediate (This Audit)
✅ **COMPLETED:**
1. Created `backend/q_backend/q_capability_evidence.py`
2. Implemented evidence framework
3. Created `tests/q_backend/test_q_capability_evidence.py`
4. Generated Q_NATIVE_WITHOUT_LOWERING report (30 ops)
5. Documented findings

### Phase 2: Fix Authority (Next Steps)

**Step 1: Update q_capability.py to use evidence**
```python
# backend/q_backend/q_capability.py

from backend.q_backend.q_capability_evidence import (
    get_q_production_safe_ops,
    get_q_native_without_lowering,
)

class QBackendCapability:
    def supports_native(self, op_name: str) -> bool:
        """Auto-derived from evidence, not manual list."""
        from backend.q_backend.q_capability_evidence import (
            get_q_production_safe_ops
        )
        return op_name in get_q_production_safe_ops()
```

**Step 2: Remove manual _PHASE1_NATIVE_OPS list**
- Delete lines 37-103 in `q_capability.py`
- Migrate to `get_declared_native_ops()` in evidence module

**Step 3: Add startup assertion**
```python
# backend/q_backend/__init__.py
def _assert_q_capability_integrity():
    from backend.q_backend.q_capability_evidence import (
        get_q_native_without_lowering
    )
    missing = get_q_native_without_lowering()
    if missing:
        raise RuntimeError(
            f"Q backend integrity violation: "
            f"{len(missing)} ops declared native without lowering. "
            f"Missing: {sorted(missing)[:10]}"
        )
```

### Phase 3: Implement Missing Lowerings

**Option A: Implement all 30 operators**
- Add Q lowering for each of the 30 missing operators
- Verify compilation
- Run against test data

**Option B: Reduce declared scope**
- Remove operators from declared native set if not needed for Phase 1
- Keep only operators with proven implementation

**Recommendation:** Option B first (reduce scope), then Option A incrementally.

### Phase 4: Add Parity Testing

**Create Q backend parity test suite:**
```python
# tests/backend_parity/test_q_backend_parity.py

def test_q_native_ops_parity(op_name):
    """Verify Q backend matches Pandas for native ops."""
    # Generate test data
    # Execute in Pandas
    # Execute in Q
    # Assert results match
```

**Evidence storage:**
```json
// evidence/q_verified.json
{
  "schema_version": 1,
  "q_reference_parity": ["add", "subtract", ...],
  "q_edge_verified": ["add", "subtract", ...],
  "q_production_safe": ["add", "subtract", ...]
}
```

### Phase 5: Integration

**Update capability registry:**
```python
# backend/operator_capability.py

def query_q_capability(op_name: str) -> BackendCapability:
    from backend.q_backend.q_capability_evidence import (
        compute_q_capability_evidence
    )
    evidence = compute_q_capability_evidence()
    if op_name not in evidence:
        return BackendCapability(
            canonical=op_name,
            backend="q_kdb",
            status="unsupported"
        )
    ev = evidence[op_name]
    return BackendCapability(
        canonical=op_name,
        backend="q_kdb",
        status="production_safe" if ev.production_safe else "implemented",
        execution_kind="native_expr" if ev.lowering_exists else "unsupported"
    )
```

---

## Hard Gates

### Q_NATIVE_WITHOUT_LOWERING
**Status:** ✗ FAIL  
**Requirement:** MUST equal 0 for production  
**Current:** 30 operators  
**Action:** Remove from declared native OR implement lowering

### Q_MANUAL_AUTHORITY_REMOVED
**Status:** ✗ FAIL  
**Requirement:** No manual capability list  
**Current:** `_PHASE1_NATIVE_OPS` still exists  
**Action:** Migrate to evidence-based authority

---

## Production Readiness

### Current State
```
Q Backend Production Ready: ✗ NO

Blocking Issues:
1. 30 operators declared without implementation
2. Manual authority can diverge from reality
3. No parity testing infrastructure
4. No evidence storage/verification
```

### Required for Production
```
✓ Evidence framework implemented
✗ Q_NATIVE_WITHOUT_LOWERING == 0
✗ Manual authority removed
✗ Parity tests created
✗ Evidence JSON storage
✗ Integration with capability registry
```

**Estimated Work:**
- Phase 2 (Fix authority): 2 hours
- Phase 3 (Implement/reduce): 4-8 hours
- Phase 4 (Parity tests): 8-12 hours
- Phase 5 (Integration): 2-4 hours

**Total:** 16-26 hours of engineering work

---

## Files Created

1. **`backend/q_backend/q_capability_evidence.py`** (321 lines)
   - Evidence framework implementation
   - Auto-derived capability
   - Hard gates

2. **`tests/q_backend/test_q_capability_evidence.py`** (286 lines)
   - Comprehensive test suite
   - Evidence validation
   - Gate verification

3. **`Q_P0_001_002_FINDINGS.md`** (this document)
   - Audit findings
   - Architecture recommendations
   - Remediation plan

---

## Conclusion

The Q backend capability system currently violates the principle that **production capability must be derived from actual evidence, not manual declarations**. The 30-operator gap between declared native support and actual lowering implementations creates a critical production safety issue.

The evidence framework created in this audit provides:
1. Automatic detection of capability gaps
2. Hard gates for production admission
3. Clear path to evidence-based authority
4. Alignment with Polars/DuckDB evidence model

**Next Step:** Implement Phase 2 recommendations to replace manual authority with evidence-based capability, then incrementally close the 30-operator gap.

---

**Audit Complete:** Q-P0-001 and Q-P0-002  
**Evidence Generated:** YES  
**Production Ready:** NO  
**Blocking Gates:** 2 FAIL
