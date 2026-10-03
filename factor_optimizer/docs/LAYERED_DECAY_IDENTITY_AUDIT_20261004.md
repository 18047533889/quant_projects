# Layered decay execution identity audit (2026-10-04)

Status: scoped source-bound identity repair is present in the formal working tree; unpublished.

Formal tree: server-c `/home/sunhaiwei/quant_projects`.
Before the repair, `search/execution_dedup.py::_execution_binding` bound LayeredDecayPlan only
to its route string and `layered-decay.v1`, not its selected implementations.
A bounded in-process probe constructed a plan with twenty half-lives of 2,
recorded `_execution_signature(plan, 1, None)`, temporarily replaced
`adapters.layered_decay_long._execute_sparse_layered_decay_validated`,
and recorded the signature again. Both signatures were equal. The runner
was never executed, the original callable was restored in a finally block,
no source file changed, and the probe exited zero.

The repair adds a scoped identity binding for the selected sparse runner,
LayeredDecayState methods, selected QE quantile functions, plan adapter,
frame validator, relevant containing module sources, and Python/NumPy/Pandas
runtime versions. Source identity fails closed when selected source or
required implementations are unavailable. It changes deduplication
signatures for the tested selected implementation replacements.

This audit does not show numerical decay output is wrong, and a scoped
identity is not certification of every transitive runtime dependency.
Same-source globals and closure values are not captured by source hashing.
The identity therefore documents selected source/version evidence only; it
does not establish a complete runtime dependency closure or prove this path
is universally fastest. The repair is present in the working tree and has
not been published as a production factor.

See [LAYERED_DECAY_EXECUTION_IDENTITY_20261004.md](LAYERED_DECAY_EXECUTION_IDENTITY_20261004.md)
for the scope, limitations, and verification record.
