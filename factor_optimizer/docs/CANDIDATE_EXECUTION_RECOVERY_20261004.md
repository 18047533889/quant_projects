# TRAIN candidate failure recovery

Status: applied in formal server-c working tree; not yet published.

## Failure reproduced

Audited read-only plans share an invocation-local TRAIN dataframe. A fault
injected into the first cross-sectional rank executor changed date, asset and
value columns before raising. Later rank candidates became ineligible, whereas
the nonmutating exception control evaluated them normally. RAW input remained
unchanged. This demonstrates a failure-isolation gap, not an ordinary numerical
defect in the unchanged read-only rank implementation.

## Recovery scope

Only an exception during execution or result conversion of a shared-frame plan
triggers rebuilding. `candidate_recovery.rebuild_train_frame` uses original
batch axes and independent baseline values, restoring row order and detached
storage. The optimizer clears last-plan output reuse, prepared rank references,
and cached rank data. It preserves the original ineligible-candidate reason.
If rebuilding fails, the candidate ledger also retains `recovery_error`;
the primary execution error remains the factor's reason.
A failure during recovery aborts the current factor to `error_raw_retained`,
rather than continuing on corrupted input; other factors can still run.
Detached candidate frames and ordinary scoring-floor failures do not rebuild.
The successful path adds no full dataframe copy; the initial copy is unchanged.
This does not isolate external side effects or a mutating executor that returns
success. It is not a numerical or universal-performance certification.

## Verification

Formal pytest `test_train_frame_ownership_oct04.py -k executor_mutation_then_exception`
failed before the fix because later candidates were not train-evaluated, then
passed with the fix (1 passed, 4 deselected). Combined ownership, prepared rank
and prepared label tests passed: 23 tests, 3 existing FE classification warnings,
32.66 seconds. Additional edge tests were run after implementation, not claimed
as separate red-green cycles. Real-data performance A/B remains pending.

The recovery-error diagnostic had its own failing expectation before its fix,
then passed. A populated-cache regression forces reuse admission on a small
fixture and verifies an actual held cache/prepared rank before failure, cache
eviction without resetting counters, no stale prepared route, and exact later
candidate/output parity. This additional test passed after implementation;
it is not claimed as a separate red-green cycle or real-speed evidence.
