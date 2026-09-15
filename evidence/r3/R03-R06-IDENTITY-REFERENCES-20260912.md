# R03 / R06 focused verification — 2026-09-12

Scope: `cleaned_operators/registry.py`, `cleaned_operators/math_certificate.py`, and focused regression tests. `backend/cleaned_bridge.py` was inspected read-only and was not changed by this work.

## R03 implementation dependency identity

- Schema discriminator: `execution-dependency-v3`.
- Implementation identity composes the class-defined execution wrapper and wrapped/default kernel; `_fn` no longer causes an early return that hides wrapper changes.
- Python functions bind bytecode, defaults, kwdefaults, named closure cells, and globals actually named by bytecode.
- Referenced scalar/config globals are frozen; modules are represented by name and version without walking `module.__dict__`; external-library callables are bounded by module, version, qualname, and code identity.
- Bound methods bind auditable `self` state. Frozen dataclasses and explicit `semantic_identity()` objects are accepted. Arbitrary Python callable instances fail closed instead of receiving a type-name-only identity. NumPy/native callables use their native callable plus library version identity.
- Existing bytecode operand, closure-name/order, integer, ndarray/axis, and bounded 64-depth / 1 MiB resource rules remain in place.

Focused counterexamples covered: wrapper `+1` versus `+2` over the same `_fn`; referenced global `GAIN=2` versus `GAIN=3`; unreferenced 100,000-item global does not enter payload; undeclared callable instance rejection; frozen-dataclass bound `self.gain=2` versus `3`.

## R06 reference contracts

- `_rank_pct_rowwise_np` now follows pandas `rank(pct=True)` missing-value semantics: only NaN is excluded, while `+/-Inf` participates in ranking.
- `_rolling_cov_ref` now enforces equal-length 1-D inputs, positive `window`, `1 <= min_periods <= window`, non-negative `ddof`, and both `count >= min_periods` and `count > ddof` before division.

## Commands and results

```text
.venv/bin/python_p3_12 -c 'from factor_engine.cleaned_operators import load_all; load_all(include_research=True); print("LOAD_ALL_RESEARCH_OK")'
LOAD_ALL_RESEARCH_OK

.venv/bin/python_p3_12 -m pytest -q \
  factor_engine/tests/operators/test_r03_r06_identity_references.py \
  factor_engine/tests/operators/test_r10_registry_identity.py \
  factor_engine/tests/operators/test_r9_registry_hash_2026_08.py
40 passed, 2 warnings in 55.16s
```

The two warnings are existing diagnostics for test-only operators lacking an explicit Polars `PhysicalImplementationSpec`; they are not failures and no certification labels were changed.

## Follow-up boundary review

The initial verified-library shortcut was retested against arbitrary `user_ops` module names. Three negative controls originally collided: a user helper closure with `gain=2/3`, a user callable instance with state `gain=2/3`, and same-named unversioned modules with different contents.

The boundary is now limited to an explicit set of verified numeric dependency roots and only their Python functions. User functions always take the full identity path regardless of module name; callable instances still require a dataclass or `semantic_identity()` and cannot enter the library shortcut. Versionless modules use a bounded source digest; a sourceless dynamic module raises `UncertifiableImplementationIdentity` rather than receiving a name-only identity. Registry bookkeeping catches that specific condition and records a `None` implementation hash so research candidates remain discoverable, while direct `_impl_source_hash` consumers—including execution/certification identity—fail closed.

```text
.venv/bin/python_p3_12 -c 'from factor_engine.cleaned_operators import load_all; load_all(include_research=True); print("LOAD_ALL_RESEARCH_OK")'
LOAD_ALL_RESEARCH_OK

.venv/bin/python_p3_12 -m pytest -q \
  factor_engine/tests/operators/test_r03_r06_identity_references.py \
  factor_engine/tests/operators/test_r10_registry_identity.py \
  factor_engine/tests/operators/test_r9_registry_hash_2026_08.py
44 passed, 2 warnings in 59.12s
```
