# Layered decay execution identity audit (2026-10-04)

Status: confirmed identity coverage defect; repair pending.

Formal tree: server-c `/home/sunhaiwei/quant_projects`.
`search/execution_dedup.py::_execution_binding` binds LayeredDecayPlan only
to its route string and `layered-decay.v1`, not its selected implementations.
A bounded in-process probe constructed a plan with twenty half-lives of 2,
recorded `_execution_signature(plan, 1, None)`, temporarily replaced
`adapters.layered_decay_long._execute_sparse_layered_decay_validated`,
and recorded the signature again. Both signatures were equal. The runner
was never executed, the original callable was restored in a finally block,
no source file changed, and the probe exited zero.

Required follow-up: scoped source identity for the selected sparse runner,
LayeredDecayState and QE assign_quantiles_batch, the adapter selection,
and relevant runtime versions. Fail closed when source is unavailable.
Test signature invalidation when each selected implementation changes.
This audit does not show numerical decay output is wrong, and a scoped
identity is not certification of every transitive runtime dependency.
