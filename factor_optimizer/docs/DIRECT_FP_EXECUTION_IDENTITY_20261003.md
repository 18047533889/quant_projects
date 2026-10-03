# Direct FP repair execution identity

## Defect and change

Direct `ValueRepairPlan` routes previously used route names and mapping versions
as their execution binding. Replacing the actually invoked `capped_zscore`
function could change output without changing the proposal execution signature.
This makes implementation-distinct proposals appear equivalent during dedup.

The dedicated `adapters/repair_execution_identity.py` now binds the selected
callable source, its containing module source, the `ValueRepairPlan.execute`
dispatcher, the identity selector/resolver, and Python/NumPy/pandas versions.
Missing source or a noncallable selected target fails closed. The search hook
resolves a fresh binding rather than retaining a process-global stale identity.

Covered selections: `rank_shape`, `fp_cs_rank_min`, `ts_rank_history`,
`ts_zscore_history`, `capped_zscore`, `tail_hinge`, `robust_scale`, and `cs_rank`
with a non-average method. Average ranking remains an FE binding; FE sign,
winsorization and SMA paths are unchanged by this patch.

These routes execute FP functions directly and bypass its registry. Registry
metadata therefore must not be presented as authority for these executions.
The identity is scoped: imported dependencies outside the containing kernel
module, globals/closure state and a complete transitive runtime closure are not
certified. This fixes the reproduced source-implementation collision, not every
possible runtime equivalence question or migration to FE-native execution.

## Verification

The focused identity suite passed 12 tests, including all eight dynamically
replaced selected functions, actual dispatch matching, unsupported routes and
unavailable selector source. The independent joint run with FactorAssets passed
36 tests in 0.75 seconds. Related dedup/mapping/sparse/holdout regression passed
35 tests with 54 warnings in 37.76 seconds before the final selector-hash change;
do not treat that earlier run as a full final optimizer regression.
