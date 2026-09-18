# R29 repair checkpoint — 2026-09-18

Worktree: server-c /home/sunhaiwei/quant_projects, main. Changes are made directly
in the shared formal tree; no new branch/worktree/repository copy. User authorized
commit and push. No deployment or production-factor publication.

## Implemented and pushed

- 783d3071: ts_vol_of_vol rejects non-integer minimum observations and minimum
  observations exceeding either window, both through metadata validation and
  direct kernels. Legitimate degenerate NaN outputs remain legitimate.
- 2957334a: ts_topk_sum omitted k now follows its canonical k=None contract,
  rather than unexpectedly computing top five. The five sibling operators keep
  their declared k=5 default.
- c878f6c5: ts_product first-registration metadata, later implementations,
  lookback, support/missing-data parameters and default window agree. The public
  default is 20; old explicit windows remain explicit.
- 903883b1 and 3f0212d7: another 50 canonical pairs have reviewable small-sample
  execution evidence. R28 retains initial oracle/recipe failures and adds actual
  runtime parameter protection rather than relabeling invalid outputs.

## Test evidence (overlapping scopes; not additive)

| Scope | Outcome | Server evidence |
| --- | --- | --- |
| Top-K direct default/explicit/oracle on two backends | 2 passed, 4 deselected | r29-topk-default-fixed.log |
| Actual top-K run_many Pandas/Polars-long/auto plus volatility contract | 15 passed | r29-topk-runmany-initial.log |
| Product contract and 22 rolling default comparisons/oracles | 4 tests passed | r29-product-contract-reviewed.log |
| R28 repaired and triaged four canonical pairs | 8 finite/prefix/oracle executions, 4 fingerprint-bound parity checks | r28_operator_campaign_extra/ledger_attempt04.jsonl |

All the above watchdog runs exited zero without a guard firing. Guards sample
process-family RSS, not kernel-enforced hard memory limits.

The original r29-default-kernels-initial.log records six failed checks. It
exposed top-K default divergence, required implementation arguments despite
declared defaults, and Polars digital_count stopping at d rather than processing
all timestamps. The top-K defect is fixed; digital_count and ts_max_buildup
repairs are still under review at this checkpoint.

## Coverage and limits

Historical reviewed paired-kernel campaign coverage is **352 / 1756**, with
**1404 not yet verified by this protocol**. See R28_REVIEWED_EXECUTION_SUMMARY.json.
A count is neither certification of all parameter domains nor proof of every
backend path. Each fingerprint binds its own tested implementation. The updated
ts_product evidence is in r27_ts_product_r24_refresh_final; earlier records are
preserved as historical evidence.

The factor CSV is unchanged in this checkpoint. No assertion is made that all
113893 catalog factors execute, that every GPU path is verified, or that a
production-scale run is fastest or immune to memory failure.

## Still being repaired

- Polars-long group fallback policy, string/numeric label identity and
  non-finite membership; canonical error policy needs independent validation.
- Missing/default windows in conditional rolling kernels.
- Digital-count full-history truncation and max-buildup defaults/axis preservation.
- ATR long-path gap and positional/keyword window fixes await the combined
  emitter checkpoint after group repairs.
- Remaining operators, end-to-end factor execution and scale/memory benchmarks.
