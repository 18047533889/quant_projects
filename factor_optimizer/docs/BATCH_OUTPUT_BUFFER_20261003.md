# Batch output allocation

The research optimizer now writes each completed factor into one factor-major
float64 buffer. Its final transpose is a view, rather than stacking a retained
list of all two-dimensional outputs into another full batch array.

FactorBatch still performs its immutable ownership copy. Output IDs, ordering,
validity, dtype and non-aliasing remain unchanged. A failed frozen-plan TEST
materialization retains its validated prefix and marks its later values invalid;
it does not substitute RAW or rerun selection using TEST labels.

This removes the old list-retention plus stack allocation path. It does not
eliminate the final ownership copy or candidate-evaluation work. Allocation is
reserved earlier, so memory exhaustion can surface earlier. Output-assembly
measurements are recorded below; no full-optimizer speedup is established.

Characterization covers mixed finite/invalid RAW outputs and a middle-factor
materialization failure with intact neighboring RAW outputs. The regression and
paired measurements below qualify bounded assembly characterization only;
full-optimizer paired A/B remains required for an end-to-end speedup claim.

## Regression evidence

Session 45448 completed with exit 0: 1755 passed, 16 warnings, 157.27 seconds.
The command was `.venv/bin/python -m pytest -q factor_optimizer/tests
--ignore=factor_optimizer/tests/test_real_scaling_cohort_oct03.py`, with
OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS set to 1. The excluded
new audit test was still being authored and must pass separately. The tested
research_batch.py SHA256 was
`ca07831e8c717f555096575a8074eeea1908b7f893e4e8f89fc475ee688a8add`.
This is correctness regression evidence, not an A/B performance measurement.

Current regression session 51221 completed with exit 0 on 2026-10-04:
1773 passed, 16 warnings, 175.01 seconds, using the same single-thread command
and exclusion above. The output-buffer source hash remained the same. This run
includes the output-assembly benchmark and expanded source-budget tests, but
excludes the cohort diagnostics test file being edited concurrently; its final
revision requires separate verification. FE classification warnings remain
research/production qualification limitations, not proof of production eligibility.

The new cohort/report tests separately passed: session 53449, 15 passed in
1.71 seconds (13 mocked cohort cases plus 2 documentation cases). Real loader
preflight session 6132 rejected F16 before optimization: only 2 source records
fit the 8 MiB object cap. A read-only DataAccess manifest inspection found 71
records, 68 eligible-status records and 3 quality-blocked records; the smallest
16 eligible objects sum to 449717716 bytes. No real F16 optimizer result or
performance measurement was produced by these checks.

## Output-assembly A/B (not full optimization)

Session 33891 completed with exit 0 on F16 x T500 x N5000 (40M cells),
three alternating fresh-process pairs, with OMP/BLAS/MKL threads set to 1.
All six frozen value and validity digests matched; source and parent runtime
snapshots were unchanged. Median elapsed time: old 0.927562s, factor-major 0.869279s
(about 6.3% less). Median process peak RSS: old 1210454016 bytes,
factor-major 930041856 bytes (about 23.2% less). Timing includes deterministic
fixture generation, assembly and immutable FactorBatch construction. This does
not measure candidate optimization, real COS scoring, or full QE throughput.
Evidence: `benchmarks/output_buffer_oct04_f16_t500_n5000.json`.
