# hash_cache_key Caller Inventory

**Date:** 2026-08-14  
**Scope:** DataAccess package worktree agent-aa34bfee869d5e2e9  
**Coordinator Review:** Address hash_cache_key correctness vs ephemeral distinction

## Executive Summary

**FINDING:** Zero production callers of `hash_cache_key()` found in DataAccess codebase.

- **Production callers:** 0
- **Test callers:** 2 (test_da_p0_010_014_identity.py, test_algorithm_golden.py)
- **Risk:** NONE - No migration needed

## API Changes Made

### 1. New Explicit API: `hash_ephemeral_cache_key()`

```python
def hash_ephemeral_cache_key(value: Any, *, bits: int = 64) -> str:
    """
    Hash ephemeral performance cache key - SHORT-LIVED IN-MEMORY ONLY.
    
    Use case: In-memory cache for performance optimization (LRU cache, memoization)
    Lifetime: Process lifetime or shorter
    Persistence: NEVER persisted to disk or database
    Non-strict mode: May use repr() fallback for convenience
    """
```

### 2. Deprecated Alias: `hash_cache_key()`

```python
def hash_cache_key(value: Any, *, bits: int = 64) -> str:
    """
    DEPRECATED: Use hash_ephemeral_cache_key or hash_correctness_identity.
    
    Kept for backward compatibility only. Will be removed in future version.
    """
```

### 3. Algorithm Change: SHA-256 Instead of MD5

**Before (DA-P1-028 violation):**
- 64-bit: MD5 truncated
- 128-bit: Full MD5

**After (DA-P1-028 compliant):**
- 64-bit: SHA-256 truncated to 16 hex chars
- 128-bit: SHA-256 truncated to 32 hex chars
- 256-bit: Full SHA-256
- 384-bit: Full SHA-384
- 512-bit: Full SHA-512

## Caller Inventory Results

### Search Commands Executed

```bash
# Search entire worktree for hash_cache_key usage
find . -name "*.py" -type f ! -path "*/__pycache__/*" ! -path "*/.*" \
  -exec grep -l "hash_cache_key\|hash_ephemeral_cache_key" {} \;
```

### Files Found

1. `./dataaccess/core/identity_encoder.py` - Definition site (not a caller)
2. `./dataaccess/tests/test_da_p0_010_014_identity.py` - Test usage
3. `./dataaccess/tests/test_algorithm_golden.py` - Test usage

### Production Caller Analysis

**Result:** NONE FOUND

- No cache managers call hash_cache_key
- No query planning code calls hash_cache_key
- No data loading code calls hash_cache_key
- No factor computation code calls hash_cache_key

## Risk Assessment

### Migration Risk: NONE

Since there are zero production callers:
- No code needs to migrate to hash_ephemeral_cache_key
- No correctness caches need to migrate to hash_correctness_identity
- No risk of incorrect cache reuse from algorithm change

### Future Usage Guidance

For any NEW code in DataAccess or FactorEngine:

**Use `hash_ephemeral_cache_key()` when:**
- Building in-memory LRU cache
- Memoizing expensive function calls (transient)
- Cache lifetime ≤ process lifetime
- Cache loss acceptable (no correctness impact)

**Use `hash_correctness_identity()` when:**
- Caching factor computation results
- Caching data transformations feeding into factors
- Cache persisted to disk/database
- Incorrect cache hit would produce wrong trading signals
- Cache affects quantitative result correctness

## Test Coverage

### Existing Tests: 23 tests in test_da_p0_010_014_identity.py
- DA-P0-010: Correctness identity minimum 128-bit
- DA-P0-011: No repr fallback in strict mode
- DA-P0-012: Typed set canonicalization
- DA-P0-013: Typed dict key canonicalization
- DA-P0-014: Special float tokens (NaN/Inf/-0.0)

### New Golden Tests: 15 tests in test_algorithm_golden.py
- SHA-256 64-bit/128-bit/256-bit verification
- MD5 prohibition verification (DA-P1-028)
- Typed canonicalization golden values
- Deterministic ordering verification
- Ephemeral API equivalence tests
- Strict mode no-str-fallback verification

**Total Coverage:** 38 tests, 100% pass rate

## Recommendations

### Immediate Actions: COMPLETE
✅ Replace MD5 with SHA-256 for all bit widths  
✅ Remove str() fallback in strict mode for sets/dicts  
✅ Create explicit hash_ephemeral_cache_key() API  
✅ Deprecate ambiguous hash_cache_key()  
✅ Add algorithm golden tests  
✅ Inventory callers (zero found)

### Future Actions (when callers emerge)
- Monitor new usage via code review
- Enforce correctness vs ephemeral distinction in code review checklist
- Consider removing deprecated hash_cache_key() in v1.0.0

## Verification

All changes verified through:
1. Unit tests: 23/23 passed (DA-P0-010 through DA-P0-014)
2. Golden tests: 15/15 passed (algorithm verification)
3. No production callers to migrate
4. API backward compatible (deprecated alias preserved)

## Sign-off

**Status:** COMPLETE  
**Risk:** ZERO (no production callers)  
**Tests:** 38/38 PASSED  
**Compliance:** DA-P1-028 (no MD5), DA-P0-010..014 (all satisfied)
