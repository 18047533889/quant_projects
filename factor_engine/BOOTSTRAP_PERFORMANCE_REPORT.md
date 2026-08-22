## Bootstrap Performance Investigation Report

**Date**: 2026-08-13  
**Total Bootstrap Time**: 196.4 seconds (3 minutes 16 seconds)

---

## Executive Summary

The `cleaned_operators.load_all()` bootstrap takes **95-196 seconds** due to two critical bottlenecks that together account for **97%** of total time:

1. **`apply_evidence_certification_overlay`**: 107.7s (54.9%)
2. **`apply_production_hardening`**: 82.6s (42.1%)

The root cause is **recursive bootstrap triggering** within the evidence certification process.

---

## Detailed Breakdown

### Time Distribution

| Operation | Time (ms) | Time (s) | Percentage |
|-----------|-----------|----------|------------|
| apply_evidence_certification_overlay | 107,740.7 | 107.7 | 54.9% |
| apply_production_hardening | 82,614.5 | 82.6 | 42.1% |
| ALL_MODULE_IMPORTS_TOTAL | 1,801.7 | 1.8 | 0.9% |
| finalize_layer_governance | 537.5 | 0.5 | 0.3% |
| register_polars_gap_coverage | 469.0 | 0.5 | 0.2% |
| apply_final_contract_hardening | 216.2 | 0.2 | 0.1% |
| Other operations | < 200 each | < 1.0 | < 3% |

### Key Findings

1. **Module loading is NOT the problem**: All 219 module imports take only 1.8s (0.9%)
   - Slowest single module: `microstructure.intraday_agg` at 619ms
   - Top 5 modules total: < 1 second

2. **Evidence certification triggers recursive bootstrap**:
   - `apply_evidence_certification_overlay()` iterates through ~1000+ operators
   - For each non-daily operator, calls `pandas_reference_production_safe()`
   - This function calls `_production_sets()` 
   - `_production_sets()` checks if registry is loaded and calls `load_all()` AGAIN
   - **This creates O(N²) behavior or deep recursion**

3. **Production hardening iterates massive operator sets**:
   - Calls `factor_production_targets()` which filters thousands of operators
   - Iterates through all targets calling multiple certification functions
   - Each iteration does registry lookups, policy patches, and contract syncing

---

## Root Cause Analysis

### Problem 1: Recursive Bootstrap in Evidence Certification

**File**: `backend/factor_operator_evidence.py`  
**Function**: `_production_sets()` (line 184)

```python
def _production_sets() -> tuple[set[str], set[str]]:
    import cleaned_operators
    from cleaned_operators.operator_surface import DAILY_CANONICALS
    from cleaned_operators.production_hardening import factor_production_targets
    from cleaned_operators.registry import OperatorRegistry

    if (
        OperatorRegistry.lifecycle() == "building"
        and not getattr(cleaned_operators, "_LOADED", False)
    ):
        cleaned_operators.load_all()  # ⚠️ RECURSIVE CALL DURING BOOTSTRAP
    # ...
```

**Call chain**:
```
load_all()
  → _load_all_impl()
    → apply_evidence_certification_overlay()
      → pandas_reference_production_safe(canonical)  [for each of ~1000 operators]
        → _production_sets()
          → load_all()  ⚠️ RECURSION!
```

**Impact**: Even though `RegistryBootstrap` prevents actual re-initialization, the check and potential blocking on the condition variable happens for EVERY operator being certified.

### Problem 2: Expensive Per-Operator Processing

**File**: `cleaned_operators/production_certification_overlay.py`  
**Function**: `apply_evidence_certification_overlay()` (line 85)

For each of ~1000+ operators:
- Loads evidence JSON file: `load_factor_operator_evidence()`
- Calls `_per_gate_verified()` which loads evidence again
- Calls `pandas_reference_production_safe()` which may trigger `_production_sets()`
- Calls `_record_alias_evidence_origin()`
- Calls `reconcile_operator_certification()`

**File**: `cleaned_operators/production_hardening.py`  
**Function**: `apply_production_hardening()` (line 493)

For each of ~1000+ operators:
- Calls `factor_production_targets()` (heavy filtering operation)
- Calls `_sync_final_runtime_contract()` (attribute introspection)
- Calls `attach_four_certificates()` 
- Calls `_policy_patch()` (scope inference, tag parsing)
- Multiple dictionary operations and registry updates

---

## Optimization Recommendations

### Priority 1: Fix Recursive Bootstrap (HIGH IMPACT)

**File**: `backend/factor_operator_evidence.py`

**Option A - Cache production sets** (Recommended):
```python
_PRODUCTION_SETS_CACHE: tuple[set[str], set[str]] | None = None

def _production_sets() -> tuple[set[str], set[str]]:
    global _PRODUCTION_SETS_CACHE
    if _PRODUCTION_SETS_CACHE is not None:
        return _PRODUCTION_SETS_CACHE
        
    # ... rest of the logic ...
    _PRODUCTION_SETS_CACHE = (all_targets, nonprimitive)
    return _PRODUCTION_SETS_CACHE
```

**Option B - Pass registry state as parameter**:
Don't call `load_all()` from within `_production_sets()`. Instead, have the caller ensure the registry is loaded before calling.

**Expected speedup**: 50-80 seconds (eliminate recursive checks)

### Priority 2: Cache Evidence Loading (HIGH IMPACT)

**File**: `cleaned_operators/production_certification_overlay.py`

Load evidence JSON **once** instead of for every operator:

```python
def apply_evidence_certification_overlay() -> None:
    # ... setup code ...
    
    # Load evidence ONCE at the start
    try:
        from backend.factor_operator_evidence import load_factor_operator_evidence
        evidence_payload = load_factor_operator_evidence()
        operators_evidence = evidence_payload.get("operators", {})
    except Exception:
        evidence_payload = {}
        operators_evidence = {}
    
    for canonical in sorted(factor_production_targets()):
        # Use cached evidence
        per_gate = _per_gate_verified_from_payload(canonical, operators_evidence)
        # ...
```

**Expected speedup**: 20-40 seconds

### Priority 3: Optimize `factor_production_targets()` (MEDIUM IMPACT)

**File**: `cleaned_operators/production_hardening.py`

Cache the result since it's called multiple times with the same inputs:

```python
_FACTOR_PRODUCTION_TARGETS_CACHE: frozenset[str] | None = None

def factor_production_targets() -> frozenset[str]:
    global _FACTOR_PRODUCTION_TARGETS_CACHE
    if _FACTOR_PRODUCTION_TARGETS_CACHE is not None:
        return _FACTOR_PRODUCTION_TARGETS_CACHE
    
    # ... existing computation ...
    _FACTOR_PRODUCTION_TARGETS_CACHE = result
    return result
```

**Expected speedup**: 10-15 seconds

### Priority 4: Batch Registry Operations (MEDIUM IMPACT)

Instead of calling `OperatorRegistry._catalog.setdefault()` and multiple dictionary updates for each operator, batch the updates:

```python
# Prepare all updates first
updates = {}
for canonical in sorted(targets):
    updates[canonical] = compute_catalog_updates(canonical)

# Apply all updates at once
for canonical, update_dict in updates.items():
    catalog = OperatorRegistry._catalog.setdefault(canonical, {})
    catalog.update(update_dict)
```

**Expected speedup**: 5-10 seconds

### Priority 5: Add Research Mode Fast Path (LOW-MEDIUM IMPACT)

For `include_research=False` or research/development environments, skip expensive production certification:

```python
def apply_evidence_certification_overlay(*, skip_for_research: bool = False) -> None:
    if skip_for_research:
        # Minimal certification for development
        return
    # ... full certification ...
```

**Expected speedup**: 50-100 seconds in research mode

---

## Summary of Expected Improvements

| Optimization | Expected Speedup | Difficulty | Priority |
|--------------|------------------|------------|----------|
| Fix recursive bootstrap | 50-80s | Low | P1 |
| Cache evidence loading | 20-40s | Low | P1 |
| Cache production targets | 10-15s | Low | P2 |
| Batch registry operations | 5-10s | Medium | P3 |
| Research mode fast path | 50-100s (dev only) | Low | P2 |

**Total potential speedup**: 85-145 seconds  
**Target bootstrap time**: 10-50 seconds (from current 196s)

---

## Implementation Priority

1. **Immediate** (P1): Fix recursive bootstrap + cache evidence loading → ~70-120s speedup
2. **Short-term** (P2): Cache production targets + research fast path → ~60-115s additional
3. **Nice-to-have** (P3): Batch operations → ~5-10s additional

With P1+P2 implemented, bootstrap time should drop from **196s to 15-40s** (87-92% improvement).
