# Real COS factor: prespecified method execution audit

## First actual run

On server-c the manifest-bound DataAccess reader loaded
`weekly_4cbd7ca6dccc61dc` from the existing authorized candidate pool. Panel:
500 trading dates, 256 assets, one factor. The automatic split reserved 267
scoring TRAIN dates, 97 VALIDATION dates and 100 TEST dates; warmup/purge rows
account for the remaining dates. TEST was not scored and no production factor
was published. Labels use the existing adjusted VWAP t+1 to t+2 return loader.

The fixed exploratory catalogue executed 77 of 85 cases without a failure.
Eight cases explicitly require external exposure data, DSL recompilation or
control/row-selection support. This is not a fitted optimizer selection run,
not acceptance of all methods, and not evidence that every factor improves.
Of the 77 executed cases, 65 had available costed economic metrics and 12 did
not (3 missing-observation cases and 9 undefined-ratio cases). Unavailable
metrics must not be filled with fabricated zeros or called successful gains.

| Family | Cases executed | Economic metrics available | All TRAIN floors passed |
|---|---:|---:|---:|
| DECAY_REFINEMENT | 5 | 5 | 0 |
| CAUSAL_SMOOTHING | 27 | 27 | 0 |
| U_SHAPE_REPAIR | 7 | 2 | 0 |
| INVERTED_U_REPAIR | 7 | 7 | 7 |
| REPRESENTATION_RANK | 8 | 5 | 4 |
| REPRESENTATION_ZSCORE | 4 | 2 | 1 |

Passing exploratory TRAIN floors is not holdout acceptance. No method should
be forced onto an input merely because it executes or scores well on TRAIN.
Prefix/permutation invariants cover the pre-validation feature prefix,
including warmup/purged rows; all return metrics use exact TRAIN indices.

Evidence: `real_training_method_audit_oct03.json` (98,300 bytes). Input data
and provenance hashes before/after matched
`74c0289589c93f35d77f283a26805b899002a30112f5cca5dbdec199b48cba37`.
This hash is not a complete source/runtime closure. Review after the run found
that the entry-point guard verified shape but not exact date/asset alignment;
the existing loader creates matching axes, but the guard must reject shifted
or permuted alternate inputs. The first report predates that guard hardening
and must not be represented as validation of the later guard implementation.

## Subsequent guard verification

Exact factor/label dates and assets are now checked before method scoring.
Label axes, availability/execution/return-window metadata and source/calendar
references are included in the input fingerprint. Independent scoped guard
tests passed 12 cases in 0.93 seconds. A later independent holdout run passed
the TEST-label/window mutation test and frozen-TRAIN-winner-only validation
test (2 cases, 3.37 seconds). These tests prove their stated invariants, not
automatic profitability or complete runtime/backend correctness.
