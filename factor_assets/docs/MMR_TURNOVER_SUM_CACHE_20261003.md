# Constrained MMR turnover-sum reuse (2026-10-03)

When an `AssemblyPolicy` enables a turnover budget, the selector now computes
the selected members' turnover with Python's built-in `sum` once after each
commit. It reuses that result for each candidate feasibility check while the
selected set stays fixed, and for the selection trace's remaining-budget field.
This keeps Python 3.12 compensated summation, candidate order, rejection
priority, objective values, and budget-boundary decisions. It does not use a
floating-point `+=` accumulator.

The evidence is stable for the duration of an assembly call: the assembler
copies the caller's evidence mapping into an owned dictionary, and
`AssemblyEvidence` is frozen with scores normalized to finite floats. A
similarity callback mutating the caller's original mapping cannot change the
selection snapshot or make the cached total stale.

For N=1,000 candidates and K=50 commits with no early pruning, the old two
feasibility scans traversed about `2 * sum((N-i)*i)` selected turnover values
for i=1..K-1: 2,369,150 value visits. The cached path traverses the selected
prefix once per commit: `sum(i for i in 1..K)` = 1,275 values. A separate
independent old-scan oracle checks candidate and provider-call order, selected
IDs, rejection codes, greedy step scores, capacity and turnover headroom, and
trace equality. Its fixtures include ties, missing families, family/micro/macro
caps, finite negative turnover, and a compensated-sum budget with its adjacent
representable float.

The benchmark reports a constrained-sum kernel comparison separately from
end-to-end public assembly. Results are workload-specific; the end-to-end path
also includes candidate iteration, similarity calls, identity checks, and
artifact construction. No general speedup claim follows from the kernel ratio.

## Controlled timing snapshot

The measured source snapshot had SHA-256 `731e0e7e63c86d0c5ee85e632003b9025ba8b2ec6d3de51fefa4a6d4d3dc075b2`, matching the current `factor_assets/assembly/engine.py`. The public A/B baseline was commit `cf21d7da6` (the original engine implementation). At N=1,000 and K=50, each arm had one warmup and six measured samples over three ABBA rounds. Values were captured from rounded stdout at six decimal places. The public assembly medians were 0.374426 s old and 0.139840 s cached (about 2.68x); selected-50 public artifact equality passed. The similarity callback returned zero, so this result measures assembly overhead with a cheap callback, not production expensive-similarity cost.

The isolated old-scan kernel medians were 0.096133 s and 0.002466 s cached (about 39x). This is a microbenchmark of the turnover-sum work only and must not be read as an end-to-end public speedup. Full rounded samples, benchmark scope, and source identity are recorded in [the JSON evidence](benchmarks/mmr_turnover_sum_cache_20261003.json).
