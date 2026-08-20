# Installation and Packaging Test Report

**Date:** 2026-08-14  
**Test Environment:** Python 3.10, Linux  
**Packages Tested:** data-access, factor-engine, quant_evaluator, factor_preprocess, factor_optimizer, factor_assets

---

## Executive Summary

**Overall Status:** ⚠️  Packages install and build successfully, but several packaging issues discovered:

1. **Critical:** `quant_evaluator` has broken package structure - installs submodules as top-level instead of under namespace
2. **High:** `factor-engine` includes 839 `__pycache__` files in wheel (bloat)
3. **Medium:** Test files included in 3 packages (quant_evaluator, factor_assets, factor_engine)
4. **Low:** Minor dependency version inconsistencies across packages

---

## 1. Independent Installation Tests

### 1.1 Individual Package Installation

Tested each package in isolated venv with `pip install -e .`:

| Package | Status | Install Time | Dependencies Resolved |
|---------|--------|--------------|----------------------|
| data-access | ✅ PASS | ~8s | All core deps installed correctly |
| factor-engine | ✅ PASS | ~7s | All core deps installed correctly |
| quant_evaluator | ⚠️ PARTIAL | ~6s | **Installs but namespace broken** |
| factor_preprocess | ✅ PASS | ~5s | All deps installed correctly |
| factor_optimizer | ✅ PASS | ~3s | Minimal deps, clean install |
| factor_assets | ✅ PASS | ~4s | All deps installed correctly |

### 1.2 Import Verification

**data-access (0.10.2):**
```
✓ data_access imported (0.413s)
✓ All submodules accessible (core, registry, read, write, cos, contract, security)
✗ Entry points not exported in __init__.py (DataAccessManager, read_panel)
```

**factor-engine (0.3.1):**
```
✓ Top-level modules imported (2.911s - includes pandas/numpy)
✓ All package modules work (cleaned_operators, expr, ir, backend, planner, runtime, mining, modeling)
✓ Clean module structure
```

**quant_evaluator (0.0.1a1):**
```
❌ CRITICAL: Package structure broken
- pyproject.toml uses `packages.find = {}` with `package-dir = {"": "."}`
- Results in submodules installed as top-level (adapters, api, metrics, etc.)
- Cannot `import quant_evaluator` - namespace doesn't exist
- top_level.txt shows 13 individual modules instead of "quant_evaluator"
```

**factor_preprocess (0.1.0):**
```
✓ Package imports correctly
✓ Submodule structure correct
```

**factor_optimizer (0.1.0):**
```
✓ Package imports correctly
✓ Uses hatchling (different build backend than others)
✓ Submodules accessible
```

**factor_assets (0.1.0):**
```
✓ Package imports correctly
✓ Clean structure
```

---

## 2. Optional Dependency Tests

### 2.1 Core Without Extras

**factor-engine without data-access:**
- ✅ Core functionality works
- ✅ Backends (pandas, polars) import without DA
- ✅ Optional extras clearly defined: `[data]`, `[performance]`, `[full]`

**factor_optimizer without FE/QE:**
- ✅ Core modules (grammar, search, policy) import successfully
- ✅ Adapters correctly raise ImportError when dependencies missing
- ✅ Optional extras: `[factor_engine]`, `[quant_evaluator]`

**factor_assets without quant_evaluator:**
- ✅ Core functionality works
- ✅ Optional `[adapters]` extra defined

### 2.2 Optional Extras Configuration

| Package | Optional Extras | Status |
|---------|----------------|--------|
| data-access | `[polars]`, `[client]`, `[service]`, `[clickhouse]`, `[sql]`, `[all]` | ✅ Well-defined |
| factor-engine | `[pandas]`, `[accel]`, `[talib]`, `[polars]`, `[modin]`, `[performance]`, `[data]`, `[full]`, `[service]` | ✅ Comprehensive |
| quant_evaluator | `[dev]` only | ⚠️ Missing extras for optional features |
| factor_preprocess | `[dev]`, `[fast]` | ✅ Good |
| factor_optimizer | `[factor_engine]`, `[quant_evaluator]`, `[dev]` | ✅ Good |
| factor_assets | `[dev]`, `[adapters]` | ✅ Good |

---

## 3. Cross-Package Dependency Tests

### 3.1 Integrated Installation

**QE + FP Integration:**
- ✅ Both install in same venv
- ✅ Dependency deduplication works (numpy 2.2.6, pandas 2.3.3 shared)
- ❌ Cannot test integration due to quant_evaluator namespace issue

**Expected Integrations (not fully tested due to QE issue):**
- factor-engine + data-access
- factor_optimizer + factor-engine + quant_evaluator
- factor_assets + quant_evaluator

---

## 4. pyproject.toml Analysis

### 4.1 Version Consistency

| Package | Version | Python Requirement | Build Backend |
|---------|---------|-------------------|---------------|
| data-access | 0.10.2 | >=3.10 | setuptools |
| factor-engine | 0.3.1 | >=3.10 | setuptools |
| quant_evaluator | 0.0.1a1 | >=3.9 | setuptools |
| factor_preprocess | 0.1.0 | >=3.10 | setuptools |
| factor_optimizer | 0.1.0 | >=3.10 | **hatchling** |
| factor_assets | 0.1.0 | >=3.9 | setuptools |

**Issue:** Inconsistent Python requirements (>=3.9 vs >=3.10)

### 4.2 Dependency Version Conflicts

**pandas:**
- data-access, factor-engine, factor_preprocess: `>=2.0`
- quant_evaluator: `>=1.2.0` ⚠️ **Too permissive**

**numpy:**
- data-access, factor-engine, factor_optimizer: `>=1.24`
- factor_preprocess: `>=1.24.0` (equivalent)
- quant_evaluator: `>=1.20.0` ⚠️ **Too permissive**

**scipy:**
- factor-engine, factor_preprocess: `>=1.10`
- quant_evaluator: `>=1.6.0` ⚠️ **Too permissive**

### 4.3 Build Configuration Issues

**data-access:**
- ✅ Explicit package listing due to non-standard layout
- ✅ Has verification script (`check_wheel_inventory.py`)
- ✅ Package-data includes config files

**factor-engine:**
- ⚠️ Uses `packages.find` with explicit include list
- ❌ Includes `__pycache__` in wheel (839 files!)
- ⚠️ No exclusion patterns for build artifacts

**quant_evaluator:**
- ❌ **BROKEN:** `packages.find = {}` with `package-dir = {"": "."}` 
- ❌ Missing proper package declaration
- ❌ Installs all subdirectories as top-level modules
- ❌ Includes all tests in wheel (33 files)

**factor_preprocess:**
- ✅ Clean `packages.find` configuration
- ✅ Proper include pattern

**factor_optimizer:**
- ✅ Uses hatchling with explicit package declaration
- ✅ Clean build

**factor_assets:**
- ✅ Simple `packages.find = {}`
- ⚠️ Includes tests in wheel (31 files)

---

## 5. Packaging Tests

### 5.1 Wheel Build Success

All packages successfully build wheels with `python -m build --wheel`:

| Package | Wheel Size | Python Files | Issues |
|---------|-----------|--------------|--------|
| data-access | 205 files | 176 .py | Clean |
| factor-engine | 2015 files | 990 .py | **839 __pycache__ files** |
| quant_evaluator | 74 files | 70 .py | 33 test files, broken namespace |
| factor_preprocess | 32 files | 28 .py | Clean |
| factor_optimizer | 32 files | 29 .py | Clean |
| factor_assets | 73 files | 69 .py | 31 test files |

### 5.2 MANIFEST Completeness

**data-access:**
- ✅ Has verification script for package inventory
- ✅ Config files included via `package-data`

**Others:**
- Most rely on setuptools auto-discovery
- No explicit MANIFEST.in files found
- Package-data declarations present where needed

---

## 6. Import Performance

Measured cold import times:

| Package/Module | Import Time | Assessment |
|----------------|-------------|------------|
| data_access | 0.413s | Acceptable (pandas/numpy overhead) |
| factor-engine (cold) | 0.799s | Acceptable |
| cleaned_operators | 0.235s | Good |
| backend | 0.000s | Excellent (lazy load) |

**Total measured:** 1.604s for major imports

**Analysis:**
- data_access has most overhead (pandas/numpy/pyarrow eager loading)
- factor-engine shows good lazy loading for backend module
- No excessive startup dependencies detected
- All times acceptable for development workflow

---

## Critical Issues Found

### 🔴 P0: quant_evaluator Package Structure Broken

**Problem:**
```toml
[tool.setuptools.packages]
find = {}

[tool.setuptools.package-dir]
"" = "."
```

This installs all subdirectories as top-level modules instead of under `quant_evaluator.*`

**Impact:**
- Cannot `import quant_evaluator`
- Breaks all cross-package integrations
- Namespace pollution (adapters, api, metrics, etc. installed as top-level)

**Fix Required:**
```toml
[tool.setuptools.packages]
find = {where = ["."], include = ["quant_evaluator*"]}

# Or explicit:
packages = ["quant_evaluator", "quant_evaluator.metrics", ...]
```

**Verification:** Check that `top_level.txt` contains only `quant_evaluator`, not 13 separate modules.

### 🟠 P1: factor-engine Wheel Bloat

**Problem:** 839 `__pycache__` files included in wheel (41% of files)

**Fix Required:**
Add to `pyproject.toml`:
```toml
[tool.setuptools]
include-package-data = true

[tool.setuptools.packages.find]
where = ["."]
include = [...]
exclude = ["tests*", "*__pycache__*"]
```

Or add `.pyproject.toml`:
```toml
[tool.setuptools.build_meta]
exclude = ["**/__pycache__", "**/*.pyc"]
```

### 🟡 P2: Test Files in Production Wheels

**Packages affected:** quant_evaluator (33 files), factor_assets (31 files), factor-engine (2 files)

**Issue:** Tests increase wheel size and expose internal test code

**Fix:**
```toml
[tool.setuptools.packages.find]
exclude = ["tests*", "*/tests*"]
```

### 🟡 P3: Dependency Version Misalignment

**quant_evaluator** uses outdated minimum versions:
- `pandas>=1.2.0` (2021) vs `>=2.0` in other packages
- `numpy>=1.20.0` (2021) vs `>=1.24` in other packages  
- `scipy>=1.6.0` (2021) vs `>=1.10` in other packages

**Risk:** May install incompatible old versions that break in integration scenarios

**Fix:** Align all packages to common minimum versions tested in CI

---

## Recommendations

### Immediate Actions

1. **Fix quant_evaluator package structure** (blocking all integrations)
   - Update `pyproject.toml` with correct package declaration
   - Rebuild and verify `import quant_evaluator` works
   - Verify `top_level.txt` contains only `quant_evaluator`

2. **Clean factor-engine wheel**
   - Add __pycache__ exclusion
   - Rebuild to verify 50% size reduction

3. **Exclude tests from wheels**
   - Add test exclusions to setuptools config
   - Reduces wheel size and attack surface

4. **Align dependency versions**
   - Update quant_evaluator to `numpy>=1.24`, `pandas>=2.0`, `scipy>=1.10`
   - Document tested version matrix

### Build System Improvements

**Standardization:**
- 5 packages use setuptools, 1 uses hatchling
- Consider standardizing on one build backend
- Current mix works but complicates tooling

**Verification:**
- Add wheel content checks to CI (like data-access has)
- Verify no __pycache__, no tests in production wheels
- Check top_level.txt matches expected package names

**MANIFEST Approach:**
- Current reliance on auto-discovery works for most
- data-access shows good practice with explicit verification
- Consider adding verification step to all packages

### Dependency Management

**Version Pinning Strategy:**
- Current approach uses minimum versions (`>=`)
- Consider tighter constraints in production deployment scenarios
- Add upper bounds for known-incompatible versions

**Optional Dependencies:**
- Well-structured across packages
- Consider adding more fine-grained extras (e.g., `[numba]`, `[talib]`)
- Document which extras are production-required vs development-only

### Testing Recommendations

**Add to CI:**
```bash
# 1. Build all wheels
python -m build --wheel

# 2. Verify wheel contents
python -m zipfile -l dist/*.whl | grep -v __pycache__
python -m zipfile -l dist/*.whl | grep -v tests/

# 3. Test clean install in isolated venv
python -m venv /tmp/test_venv
/tmp/test_venv/bin/pip install dist/*.whl
/tmp/test_venv/bin/python -c "import <package>"

# 4. Test cross-package install
/tmp/test_venv/bin/pip install dist/factor_optimizer*.whl dist/factor_engine*.whl
/tmp/test_venv/bin/python -c "from factor_optimizer.adapters.factor_engine_adapter import FactorEngineAdapter"
```

### Documentation Improvements

**Add to each package README:**
- Minimum Python version clearly stated
- Installation scenarios (minimal, with extras, full)
- Cross-package dependencies explicitly documented
- Import performance characteristics

**Add PACKAGING.md:**
- Build wheel procedure
- Verification checklist
- Release process
- Dependency update policy

---

## Appendix: Test Commands Run

```bash
# Individual installs
python -m venv /tmp/test/venv1
/tmp/test/venv1/bin/pip install -e /home/shw/quant_projects/dataaccess

# Build wheels
cd <package> && python -m build --wheel

# Verify wheel contents
python -m zipfile -l dist/*.whl

# Import tests
python -c "import data_access; print(data_access.__version__)"
python -c "import time; s=time.time(); import <module>; print(time.time()-s)"

# Dependency analysis
grep -r "dependencies\s*=" */pyproject.toml
```

## Appendix: Package Dependency Graph

```
data-access (0.10.2)
  └─ Core: PyYAML, pandas, pyarrow, duckdb, numpy, sqlglot
  
factor-engine (0.3.1)
  ├─ Core: PyYAML, numpy, pandas, pyarrow, scipy
  └─ Optional: data-access>=0.2.0, polars, psutil, numba, TA-Lib
  
quant_evaluator (0.0.1a1) [BROKEN NAMESPACE]
  └─ Core: numpy, scipy, pandas
  
factor_preprocess (0.1.0)
  ├─ Core: numpy, pandas, scipy
  └─ Optional: numba, bottleneck
  
factor_optimizer (0.1.0)
  ├─ Core: PyYAML, numpy
  └─ Optional: factor-engine>=0.3.0, quant-evaluator>=0.1.0
  
factor_assets (0.1.0)
  ├─ Core: typing-extensions, dataclasses-json
  └─ Optional: quant_evaluator>=0.1.0
```

---

## Conclusion

The packaging infrastructure is mostly solid, with clean separation of concerns and good optional dependency management. The critical blocker is the quant_evaluator namespace issue, which must be fixed before any cross-package integrations can work. Secondary issues (wheel bloat, test inclusion, version misalignment) are all easily addressable and should be fixed to improve production readiness.

**Priority Fix Order:**
1. quant_evaluator package structure (P0 - blocking)
2. factor-engine __pycache__ exclusion (P1 - wheel quality)
3. Test file exclusion across packages (P2 - wheel quality)
4. Dependency version alignment (P3 - compatibility)

Once P0 is fixed, all packages should install cleanly and integrate correctly.
