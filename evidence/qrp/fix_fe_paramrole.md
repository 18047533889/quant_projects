# QRP-P0-B1 — FactorEngine ParamRole contamination fix

Status: DONE (ParamRole root cause fixed and verified)
Date: 2026-08-27
Author: FE-FIX agent
Scope: root-cause fix only, no test-time monkeypatching / conftest aliases /
try-except wrappers.

## 1. Root cause

### 1a. The `ParamRole.NUMERICAL` collection failure (fixed)

- File: `factor_engine/tests/test_ts_batch1_rank_if_direct.py`
- Mechanism: this direct-import parity test builds a **standalone fake
  `factor_engine.cleaned_operators.base` module** via
  `types.ModuleType(...)` and registers it into `sys.modules` BEFORE executing
  the production module `cleaned_operators/polars_native/ts_batch1.py` under a
  synthetic spec (`spec_from_file_location` + `module_from_spec`).  Its fake
  `ParamRole` shim was a `types.SimpleNamespace` with only
  `HORIZON/SUPPORT_POLICY/SCALAR`:

  ```python
  base.ParamRole = types.SimpleNamespace(
      HORIZON="horizon", SUPPORT_POLICY="support_policy", SCALAR="scalar"
  )
  ```

  `ts_batch1.py`'s class bodies build `ParamSpec(..., param_role=ParamRole.NUMERICAL)`
  at 35 call sites (lines 232, 262, 794, 825, …, 4263).  When pytest imports the
  production module through this shim, `ParamRole.NUMERICAL` raises
  `AttributeError: 'types.SimpleNamespace' object has no attribute 'NUMERICAL'`,
  which aborted **collection of the entire FE test tree** (pytest stops at the
  first collection error it hits in this module).

  This is exactly the "intentional lightweight shim" case the task anticipated:
  the replacement must be a real enum matching the canonical `ParamRole`
  (`factor_engine/cleaned_operators/base.py:46 class ParamRole(str, enum.Enum)`),
  which canonically declares `NUMERICAL = "numerical"` at base.py:80.

### 1b. The remaining 25 collection errors (pre-existing, NOT ParamRole)

After the ParamRole fix, the full FE tree still has **25 collection errors**, all
the same pre-existing pattern: `RuntimeError: operator registry is not writable:
frozen` (plus 2 `NameError: name 'cleaned_operators' is not defined` in
`test_r11_round2_closure_audit.py` / `test_r11_round3_closure_audit_ext.py`, and
one `FileNotFoundError` / `ValueError` each).

Mechanism (traced via instrumented `REGISTRY_BOOTSTRAP.ensure_ready` + full stack
capture during `--collect-only`):

1. Test collection imports modules in filesystem/alphabetical order.
2. `factor_engine/tests/backend_sql/test_alpha_language_sql_parity.py` (the
   alphabetically-first module whose body runs `ensure_cleaned_loaded()`) invokes
   the registry bootstrap `load_all()`, which runs the full import-time
   initialization sequence and ends with `OperatorRegistry.finalize()` +
   `OperatorRegistry.freeze()` — the registry becomes **frozen**.
3. During that bootstrap, `apply_production_hardening` →
   `semantic_certification` → `production_eligible_backends` → `_sql_status` →
   `_sql_emitter_ok` → `compile_plan_to_sql` → `_structural_use_counts` →
   `planner.plan_hash.structural_key` → `OperatorSemanticContractDigest.for_canonical`
   → `production_signature.signature_for` → `typed_signature_generator._generated_signatures`
   → `ensure_cleaned_loaded()` (re-entrant, owner-thread no-op — safe).  The
   bootstrap completes and freezes.
4. Every **later** test module that imports an operator module which registers
   operators **at module level** (`cross_section/panel_batch1.py` has
   `@register_operator` at class-body time; `time_semantic.py:610 register()`;
   `same_clock_lag.py`; direct `spec_from_file_location` re-exec of
   `polars_native/panel_group_misc.py` in `test_polars_panel_group_misc_pit.py`
   re-registers `ALMA/polars` → `ValueError: duplicate operator registration
   rejected: ALMA/polars`) then hits the frozen registry.

This is a **test-ordering / registry-lifecycle interaction** and is orthogonal to
the ParamRole contamination.  It is not fixed by this change (fixing it would
require restructuring the test lifecycle — out of scope for this task's
root-cause mandate), and is recorded as BLOCKED below.

## 2. The fix

- File: `factor_engine/tests/test_ts_batch1_rank_if_direct.py`
- Diff summary:
  - Added `import enum` (stdlib).
  - Replaced the `types.SimpleNamespace` shim with a real `str`-based enum
    mirroring the canonical members referenced by the module under test:

    ```python
    class _ParamRole(str, enum.Enum):
        HORIZON = "horizon"
        SUPPORT_POLICY = "support_policy"
        SCALAR = "scalar"
        NUMERICAL = "numerical"

    base.ParamRole = _ParamRole
    ```

  - The shim now matches canonical `ParamRole` semantics (a `str` enum, not a
    bare namespace), so `ParamRole.NUMERICAL` and any `isinstance(x, str)` /
    `str(x) == "numerical"` comparisons behave like the canonical enum.
- No production module was modified.  No conftest / test-time monkeypatch was
  added.  No `try/except` wrapper was introduced.

## 3. Verification

### 3a. Previously-broken module alone (proves the ParamRole fix)

Command:
```
cd /home/sunhaiwei/quant_projects && .venv/bin/python -m pytest factor_engine/tests/test_ts_batch1_rank_if_direct.py -q --no-header -p no:cacheprovider
```
Tail output:
```
....                                                                     [100%]
4 passed in 0.05s
```
(The environment has a local `platform/` package that shadows the stdlib
`platform` module when cwd is repo root, breaking `import uuid` for pytest's own
`_pytest._py.path`; the canonical test command fails at pytest *self-import*
before any test runs.  The verified runs above preload the stdlib `platform`
module first — a pure environment fix, not a code change.)

### 3b. Full FE tree collection (after fix, with stdlib platform preload)

Command:
```
cd /home/sunhaiwei/quant_projects && .venv/bin/python -c "import sys, importlib.util; spec=importlib.util.spec_from_file_location('platform','/usr/lib/python3.12/platform.py'); p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p); sys.modules['platform']=p; import pytest; raise SystemExit(pytest.main(['factor_engine/tests','-q','--no-header','-p','no:cacheprovider','--collect-only']))"
```
Tail output:
```
ERROR factor_engine/tests/test_same_clock_lag.py - RuntimeError: operator reg...
ERROR factor_engine/tests/test_time_semantic_ops.py - RuntimeError: operator ...
!!!!!!!!!!!!!!!!!!! Interrupted: 25 errors during collection !!!!!!!!!!!!!!!!!!!
22131 tests collected, 25 errors in 62.41s (0:01:02)
```
- `factor_engine/tests/test_ts_batch1_rank_if_direct.py` now **collects cleanly**
  (its 4 tests appear in the collection list; the `NUMERICAL` AttributeError is
  gone).
- The 25 remaining errors are the pre-existing frozen-registry test-ordering
  issue described in §1b (all present before this fix, unrelated to ParamRole).

### 3c. Collection error count
- Before fix: **26** (25 frozen-registry/NameError/FileNotFound/ValueError +
  1 `ParamRole.NUMERICAL` AttributeError on `test_ts_batch1_rank_if_direct.py`).
- After fix: **25** (the ParamRole error is eliminated; the rest are the
  pre-existing lifecycle-ordering errors).

## 4. BLOCKED / NOT_RUN

- **BLOCKED (pre-existing, unrelated to this task):** the 25 frozen-registry
  collection errors described in §1b.  Root cause: `backend_sql/test_alpha_language_sql_parity.py`
  triggers `load_all()` (freeze) during collection, then later modules with
  module-level `@register_operator`/`register()` fail.  Fixing that requires a
  lifecycle/ordering redesign of the FE test suite and is outside this task's
  ParamRole scope.
- **BLOCKED (environment):** `pytest factor_engine/tests` run verbatim from the
  repo root fails at pytest *self-import* because the untracked local
  `platform/` package shadows stdlib `platform` (breaks `import uuid` used by
  `_pytest._py.path`).  All verified runs preload the stdlib `platform` module.
- **NOT_RUN:** full test *execution* of the FE tree (collection is blocked by
  the 25 pre-existing errors; executing the 22131 collected tests without those
  modules is out of scope).  `tests/r44` root set NOT run (would hit the same
  stdlib-platform shadowing and is not part of this fix's verification).
