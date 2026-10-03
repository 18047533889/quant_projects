# MMR constraint counters and similarity sample counts (2026-10-03)

## Selection logic

The assembly selector retains the existing greedy objective:

$$
J(f\mid S)=w_q\,Q(f)-w_r\max_{s\in S}\operatorname{sim}(f,s).
$$

It checks family, microcluster and macrocluster limits before health and
turnover requirements. It uses the existing factor-ID ordering for ties.
Provider errors still propagate, and infeasible candidates do not trigger
extra similarity calls.

The selector now maintains integer counts for selected families and clusters.
For each group g, it stores the same quantity formerly computed by scans:

$$
C_g(S)=\#\{s\in S:g(s)=g\}.
$$

The family counter is used only when a family limit is configured and values
are `None` or exact built-in `str` objects. Cluster counters are used only for
exact built-in `str` cluster IDs. Counts update only after committing a
selection. This removes repeated selected-list scans for ordinary metadata
without coercing or hashing unsupported values.

If a selected member has a nonstandard family or cluster ID, that dimension is
marked counter-incompatible and subsequent feasibility checks use the original
selected-list comparisons for that dimension. This preserves legacy equality
behavior for values such as lists and non-reflexive NaN families. When no
family limit is configured, selection does not read or count family values.
The existing unknown-cluster grouping policy is unchanged.

This fallback preserves selector-level comparison semantics; it does not
promise that arbitrary cluster metadata is accepted by the public assembly
path. Later validation/aggregation may still require hashable cluster IDs.
The selector regression uses hashable list subclasses to reach this fallback.
This change does not alter similarities, quality scores, rejection priority,
the greedy objective or factor provenance.

## Turnover budget boundaries

The selector retains Python's built-in `sum` for used turnover and reported
remaining budget. Python 3.12 uses compensated float summation; replacing
that operation with a running `+=` accumulator can change admission at a
floating-point budget boundary.

The first attempted accumulator optimization changed a decision at budget
3.4. The final code discards that accumulator optimization and records this
constraint in a code comment. Regressions compare the original selected-list
scan at 3.4 and the next representable float above it, including selected IDs,
rejections, provider call order, quality, redundancy, objective and headroom.
Separate tests cover a three-term sum boundary and the adjacent lower float.

## Similarity sample-count contract

`SimilarityResult.sample_size` must be a non-boolean `numbers.Integral`
greater than or equal to zero. The constructor rejects Python/NumPy booleans,
fractional floats and other non-integer values. It normalizes accepted NumPy
integers to Python `int`. Zero remains supported; negative integers retain
their existing rejection. This validation does not prove a caller's sample
provenance or certify an ANN query.

## Verification and timing scope

The independent constrained-selection fixture uses 220 candidates, target 34,
and family, cluster, health and turnover limits. Both boundary runs selected
28 and rejected 192 candidates, with 4,686 provider calls per path.
The reference scan and public assembly produced matching structured outputs.

The two reference-scan timing windows measured 0.068072 / 0.075448 seconds;
the public assembly calls measured 0.031857 / 0.032996 seconds.
These windows compare a standalone oracle with a public wrapper call.
They do not constitute a matched selector-to-selector A/B benchmark or
establish a general throughput ratio.

The final full-suite run passed 1,696 tests with 97 warnings in 78.46 seconds.
The four compatibility regressions also passed independently. Intermediate
runs found mistakes in the new test fixtures, which were corrected before
freezing them. A subsequent run encountered 19 FE integration failures while
another FE module had a transient indentation error; a fresh syntax check and
nine targeted integration tests passed after that module recovered. The final
1,696-test run above was performed against the recovered current worktree.
