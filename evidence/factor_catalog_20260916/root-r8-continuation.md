# Continuation evidence — 2026-09-16

Formal server-c worktree only; no branch, commit, push, deployment or production publication. Original factor CSV unchanged. Disk check: 755 GiB available.

## User-authorized semantic change

The user explicitly authorized rewriting the 99 legacy intraday limit formulas to current formal definitions with original formulas and change records retained. This is SEMANTIC_REDESIGN, not equivalence. Full r9 static audit: 103627 PARSED_FIELDS_BOUND, 8814 FIELDS_UNRESOLVED, 1452 PARSE_FAILED, 239 NON_FACTOR. All 99 recorded the redesign; 71 bound, 28 unresolved. Static success is not execution success. The first rewritten real smoke failed; its multi-source/minute data cause is still being investigated. Native subexpression tests must not certify whole factors.

## Newly closed real-data sweep

`evidence/factor_catalog_20260915/resume-daily12503-external-r15.ledger.jsonl`: 500 selected daily factors; 459 EXECUTED, 25 COMPILE_FAILED, 16 EXECUTED_ALL_NONFINITE. Source cursor 12503 -> 13776 (not 500 consecutive source rows because of daily selection). Ten independently closed chunks. Watchdog exit 0; sampled peak process-family RSS 825614336 bytes; elapsed 126.73s. This is sampled monitoring, not a hard memory cap or OOM-proof guarantee.

New failures: 6 autocorr half-life calls pass an integer to use_abs; 6 extreme-cluster calls pass q=0; 5 quantile-skew calls pass a fractional min_periods; 8 support/resistance calls lack points. Do not guess missing semantic parameters. The 16 all-nonfinite factors call mass_concentration(ret, window=20); canonical implementation requires nonnegative weights and rejects windows with negative returns. A separate optional user question asks whether to redesign those as absolute-return concentration. Authorization for the 99 limit formulas does not cover this semantic choice.

Next sweep launched at offset13776 for1000 selected daily factors under900s/2048MiB sampled watchdog; do not promote unfinished chunks. Log: `resume-daily13776-external-r16.log` in Sep16 evidence.

## Independent regression verification

Persistent run state, metadata pipeline terminals, pipeline retry: 30 passed, 11 warnings. `root-terminal-state-r8.log`; watchdog exit0, peak950439936 bytes,55.58s. This verifies existing components, not run_many per-factor error isolation.

Prior `root-runmany-stream-r7.log`: 1 failed/28 passed, read-wave broker admission denied; tail agent investigating. Large auto+sink pressure results are not certified until this regression is resolved.

Liquidity-beta real row6170 correction passed earlier. New default-window mismatch in Polars implementation is being corrected by intraday agent; implementation closure changes invalidate prior primitive certification until a fresh post-freeze rerun.

## Still open

Most of the catalog has not been executed. Streaming currently lacks complete per-factor error isolation; common intermediates are reused per wave, not globally across all waves. Default return-panel mode cannot be described as bounded100k persistence. No all-operator, all-backend, fastest-path or zero-bug claim is warranted.
