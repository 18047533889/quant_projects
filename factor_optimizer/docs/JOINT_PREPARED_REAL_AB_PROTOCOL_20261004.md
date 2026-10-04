# Real optimization A/B protocol: prepared TRAIN labels

Status: bounded runner and offline checks are implemented; real workload not
executed. No performance result. Historical measurements below are not results
of the new prepared-label comparison.

Use existing strict F16 manifest-bound cohort loading and admission from
`scripts/audit_real_scaling_cohort_oct03.py`, then its nested first-four-factor
prefix. Historical receipt: 500 dates x 256 assets x 4 factors, 151 TRAIN
candidate evaluations, 480 ledger rows, 47.75 seconds; all four retained RAW.
These are historical workload-selection evidence, not results for this change.
The separate 1260 x 256 x 8 receipt proves loading only, not optimization.

Compare current production prepared-split scoring against a benchmark-only
wrapper that removes only `prepared_split` from paired_series. All other
scoring, split, config, source and runtime state must be identical. Keep
default 128 candidate budget, joint objective, cost .001, bootstrap, coverage,
hard floors and validation gates. Never weaken them to force a winner.

Load once; perform two unmeasured warmups, then at least three randomized
paired blocks (six complete optimizer runs), using a preregistered seed.
Every call creates fresh invocation-local optimizer caches. Validate exact
optimized values/validity, plans, statuses, gains, validation lower bounds and
coverage, full candidate ledgers, diagnostics, split and TEST-not-evaluated.
Record actual attempted/evaluated counts; reject comparisons on any mismatch,
error, changed admission or source/runtime drift. Score correctness first.

Retain existing resource checks before loading and every optimizer run;
historical admission was at least 16 GiB available RAM and 2 GiB disk.
Panel bytes do not bound process RSS. Capture measured process RSS separately
from per-run timing; never call cumulative ru_maxrss a per-run peak.
Do not overlap QE timing or other new heavy workloads. Record shared background
load explicitly. No dataset/repository/virtualenv copies; report under 1 MiB.
Manifest/ETag consistency is not PIT or investability certification. RAW
retention is a legitimate gate outcome, not evidence of broken methods.

## Runner usage

From the formal project root, run
`.venv/bin/python factor_optimizer/scripts/benchmark_prepared_joint_real_oct04.py`
to display the dry-run protocol; it does not load COS or invoke optimization.
After coordinated timing-slot admission, explicitly add
`--execute --report NEW_NONEXISTENT_JSON_PATH` to run the workload. The report
parent must already exist and be writable. Current execution requires at least
32 GiB available RAM (the 16 GiB above is historical) and existing disk gates.
The runner checks exact output/mask, complete ledgers and diagnostics via hashes,
restores the benchmark-only scoring wrapper even on errors, and measures the
optimizer API call separately from validation/hashing. Twelve offline tests and
the default CLI passed; neither proves a real-data speed improvement.

Snapshot tests exercise the actual output/mask/axis and complete ledger digest
wiring, not just hand-written mismatch dictionaries. Object-axis digests use
typed scalar values rather than allocation addresses; unsupported object
payloads fail closed. That allocation-independence check failed before the
semantic hash fix, then passed. Numeric arrays retain exact dtype/shape/byte
digests. Full real-data paired execution remains pending.

## Lineage drift boundary (2026-10-04)

The loader supplies treatment signatures separately from array/provenance
metadata. The runner now fingerprints the complete ExistingTreatmentSignature
for each actual prefix factor, including nested parameters and completeness
status. Every source check compares this digest; the final report includes
leaf_lineage_fingerprint. The representation is typed, mapping-order stable,
and does not stringify unsupported objects. Nonfinite parameter floats and
NumPy parameter objects are rejected rather than silently aliased.

A bounded integration test changed only one live leaf signature between pair
calls while values, labels and provenance remained unchanged. It failed before
the fix (drift accepted, 1.55 seconds), then the identical test passed after the
fix (1.10 seconds). Protocol tests including type separation and nested/status
changes passed: 15 tests; combined with 25 FA research-receipt tests, 40 passed
in 0.72 seconds. Additional boundary checks are regression tests after the
implementation, not all independent RED/GREEN cycles.

The runtime source fingerprint remains scoped to its explicitly listed files
and installed versions, not a claim of transitive runtime closure. No real COS
paired run was executed for this change. FE cold-start initialization currently
fails at the intra_same_slot_zscore owner-identity guard in an independently
reproduced broader integration test; do not bypass that guard or start real
optimizer timing until its owner restores a verified cold-start path.
