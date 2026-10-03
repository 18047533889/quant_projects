# Real scaling audit: one F16 cohort, nested F1/F4/F16

Status: an authoritative COS-backed run is recorded in
`factor_optimizer/docs/real_scaling_cohort_oct04.json`. It completed F1, F4,
and F16 sequentially over one loaded strict cohort. These are audit runtimes
and process memory observations, not a controlled speed benchmark; they do not
support a claim that any prefix is fastest or more profitable.

## Design and scope

The script loads one strict cohort of 16 manifest-bound factors and retains the
same date axis, asset axis, labels, and per-factor lineage for three sequential
optimizer runs. F1 and F4 are nested prefixes of the loaded F16 cohort, not
separate loader requests. This keeps each factor's values and labels paired
across cohort sizes.

Exactly 16 factor IDs must be retained and the loader must report no quarantined
factors. The script validates all 16 one-factor source records before any
optimizer run, including their source sizes, manifest binding, and alignment.
The asset-selection split recorded by the loader must have the same identity as
the optimizer's automatic TRAIN/VALIDATION/TEST split. Asset selection uses
purged TRAIN coverage only. TEST is reserved and its labels are not scored.

Each prefix's report includes selected-result headline fields, candidate counts,
candidate status/family counts, and a SHA-256 digest over its canonical full
candidate ledger. The report is strict JSON and refuses nonfinite values.

The report compares each factor across every prefix in which it appears, using
exact canonical-JSON equality over the entire selection summary. Top-level
`all_matched`, `compared_factor_count`, and `comparison_count` summarize the
comparison; per-factor entries list compared prefixes, `matched`, differing
fields, and `differences` as field-to-prefix-to-value mappings. A factor present
in only one prefix has `matched: null` and is not counted as a comparison. If
there are no comparable factors, `all_matched` is false. Differences are kept
for diagnosis and do not fail the audit. The comparison excludes elapsed time,
cumulative RSS, and cohort/source fingerprints; canonical JSON distinguishes
negative from positive zero.

## Recorded run

The report records a strict cohort of 16 factors, 500 dates × 256 assets,
with no quarantined factors. All three prefix runs completed and their
before/after source fingerprints matched. F1, F4, and F16 took 45.28 s,
47.75 s, and 187.39 s respectively. These wall times include each prefix's
audit work and are not comparative benchmark results.

Candidate ledgers contain 118, 480, and 1,912 rows in F1, F4, and F16.
Summing `train_evaluated` candidate statuses gives 20, 151, and 989
evaluations respectively; 989 is the F16 count. Every factor selected
`NO_OP_RAW` (`raw_retained`) after TRAIN selection was not confirmed on
VALIDATION. The summary contains four factors shared by more than one prefix;
all four matched across five reference comparisons across nested prefixes by exact canonical JSON
of the complete selection summaries, with no differing fields. The other 12
factors occur only in F16 and therefore have no cross-prefix comparison.

For `ewma_smoothness_volume`, the reported validation lower bound is
0.005117099978440225, which is positive but below the configured minimum
improvement of 0.01 (`BatchOptimizationConfig.minimum_improvement`). The
acceptance condition in `research_batch.py` requires both a strictly positive
lower bound and a lower bound at least that minimum; thus the candidate was
not accepted and RAW was retained. In F16, nine of 16 validation lower bounds
are null. The report does not include the underlying diagnostics for those
null bounds, so their reason cannot be reconstructed from this artifact.

The reported `process_cumulative_peak_rss_bytes` is 4,234,035,200 after each
prefix. It is the process high-water RSS since process start, sampled after
each run, not an independent peak attributable to each prefix.

The automatic split has 267 TRAIN, 97 VALIDATION, and 100 reserved TEST days.
`test_evaluated` is false; TEST is unscored. This run supports completion and
prefix-consistency statements only, not factor quality, trading performance,
production suitability, or a global runtime ranking. Source validation binds
the 16 loaded objects to the declared manifest and records their digests,
ETags, and sizes, but lineage is source-declared: upstream point-in-time
correctness and investability are not certified. A complete runtime and
source-closure snapshot was not captured, so service/runtime provenance and
repeat-run behavior are not closed by this report.

## Environment and invocation

Run from `/home/sunhaiwei/quant_projects` using the configured server-c
DataAccess environment. The harness requires these exact settings:

- `ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data`
- `DATA_ACCESS_COS_CLI=admin-cos`
- `DATA_ACCESS_COS_CACHE_ROOT=/home/sunhaiwei/.cache/quant-dataaccess/research`

Use a new report path; existing files and symlinks are rejected before loading:

```sh
.venv/bin/python factor_optimizer/scripts/audit_real_scaling_cohort_oct03.py \
  --report /home/sunhaiwei/quant_projects/factor_optimizer/docs/real_scaling_cohort_oct03.json
```

The scaling entry point requires at least 16 GiB available RAM and 2 GiB free
on the configured COS cache filesystem, both before source loading and again
before each prefix run. Its explicit large-source budgets are 128 MiB per
factor object and 2 GiB total downloaded factor objects; panel memory remains
bounded to 512 MiB for the declared array, with 500 dates × 256 assets, and a
1 MiB final JSON report. The default one-factor training-method audit remains
at 8 MiB per object and 128 MiB aggregate; these larger limits are opt-in to
this scaling harness.

DataAccess still enforces its existing 128 MiB decoded-result cap per object,
in addition to the declared-object read cap. A compressed source whose decoded
result exceeds that cap can still fail to load; the larger scaling budgets do
not bypass or relax the DataAccess cap. The recorded run confirms these
particular sources loaded successfully; it does not change the cap or guarantee
that other sources will load.

`elapsed_seconds` is the elapsed wall time for a prefix's audit work.
`process_cumulative_peak_rss_bytes` is the process high-water RSS since process
start, sampled after that prefix. It is cumulative—not an independent peak for
that prefix—and includes memory peaks from earlier process work. The harness is
not a speed benchmark and makes no claim that larger or smaller cohorts run
faster.
