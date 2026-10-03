# Layered decay execution identity (2026-10-04)

## Purpose

Layered decay proposal deduplication needs to distinguish plans whose
parameters and training context match but whose selected execution source
differs. The binding supplies source and runtime-version evidence to the
execution signature. It does not alter decay calculations or certify their
numerical correctness.

## Bound execution scope

`build_layered_decay_execution_identity()` records hashes of inspectable
source for the selected execution elements:

- Sparse layered-decay validated runner.
- Selected `LayeredDecayState` class and its `__init__`, `step`, `step_sparse`,
  `_step_sparse_trusted`, and `_apply_inputs` methods.
- The selected adapter frame validator and `LayeredDecayPlan.execute`.
- QE `assign_quantiles_batch` and the tie-policy, quantile-count, and
  search-bin helpers it uses.
- Source for the containing plan-adapter, sparse-runner, state, QE quantile,
  and frame-validation modules.
- Python version and implementation, plus NumPy and Pandas versions.

The route remains
`factor_optimizer.adapters.layered_decay_long._execute_sparse_layered_decay_validated`
with mapping version `layered-decay.v1`. These values join plan parameters,
orientation, and training context in the existing execution signature.
If a selected implementation is absent, not callable, or has unavailable or
empty source, identity construction raises and the proposals remain separate
rather than being merged without a binding.

## Limits

This is scoped source/version evidence, not a transitive runtime closure.
It does not hash every imported helper, dependency, native library, build
artifact, or runtime state. Hashing Python source cannot distinguish two
executions of unchanged function source that behave differently because a
referenced global, closure value, monkeypatch target, or external state
changed. Only explicitly selected sources and the listed containing module
sources are covered; this should not be read as a claim that all globals or
closures have been captured.

The binding establishes neither numerical equivalence nor performance
superiority. In particular, it does not establish that this implementation
is universally fastest across data shapes, machines, or environments.

## Verification record

The historical RED probe in
[LAYERED_DECAY_IDENTITY_AUDIT_20261004.md](LAYERED_DECAY_IDENTITY_AUDIT_20261004.md)
replaced the selected runner and observed an unchanged signature before the
repair. The probe did not execute the runner or change a source file.

After the repair, the replacement-based signature check passed (1 passed,
7 deselected). The focused execution-identity suite passed (8 passed). The
fresh identity-plus-prepared test selection passed (15 passed, 1.42
seconds). Earlier, the broader relevant selection passed 50 tests with 1
pre-existing Pandas warning in 12.97 seconds. These are source-identity
and regression checks; they do not expand the binding's stated scope.

The source change is present in the formal working tree on server-c and is
unpublished. This document records regression evidence; it is not a release
or production-factor certification.
